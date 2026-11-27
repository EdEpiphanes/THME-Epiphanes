// SPDX-License-Identifier: AGPL-3.0-only
//! Key/value cache for incremental decoding in the reference forward pass (THME-3070).
//!
//! Each layer keeps the rotary-embedded keys and the values of every position seen so far,
//! so a new token costs one row of work per layer instead of a pass over the whole sequence.
//! Supports `F32` (matching reference weights) as well as symmetric per-group-32 `Int8` and
//! `Int4` storage (THME-3082).

use anyhow::{bail, Result};
use std::borrow::Cow;

const GROUP_SIZE: usize = 32;

/// Element storage precision for CPU [`KvCache`] tensors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum KvPrecision {
    #[default]
    F32,
    Int8,
    Int4,
}

impl KvPrecision {
    /// Lowercase CLI/JSON name for this KV cache precision.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::Int8 => "int8",
            Self::Int4 => "int4",
        }
    }
}

fn group_scale(group: &[f32], q_max: f32) -> f32 {
    let max_abs = group.iter().fold(0.0_f32, |acc, &v| acc.max(v.abs()));
    if max_abs == 0.0 || !max_abs.is_finite() {
        0.0
    } else {
        max_abs / q_max
    }
}

fn quantize_symmetric(value: f32, scale: f32, q_max: f32) -> i8 {
    if scale == 0.0 {
        return 0;
    }
    (value / scale).round().clamp(-q_max, q_max) as i8
}

enum KvStorage {
    F32(Vec<f32>),
    Int8 {
        data: Vec<i8>,
        scales: Vec<f32>,
        positions: usize,
    },
    Int4 {
        data: Vec<u8>,
        scales: Vec<f32>,
        positions: usize,
    },
}

impl KvStorage {
    fn new(precision: KvPrecision) -> Self {
        match precision {
            KvPrecision::F32 => Self::F32(Vec::new()),
            KvPrecision::Int8 => Self::Int8 {
                data: Vec::new(),
                scales: Vec::new(),
                positions: 0,
            },
            KvPrecision::Int4 => Self::Int4 {
                data: Vec::new(),
                scales: Vec::new(),
                positions: 0,
            },
        }
    }

    fn len(&self, kv_dim: usize) -> usize {
        match self {
            Self::F32(buf) => buf.len().checked_div(kv_dim).unwrap_or(0),
            Self::Int8 { positions, .. } | Self::Int4 { positions, .. } => *positions,
        }
    }

    fn bytes(&self) -> usize {
        match self {
            Self::F32(buf) => buf.len() * std::mem::size_of::<f32>(),
            Self::Int8 { data, scales, .. } => {
                data.len() * std::mem::size_of::<i8>() + scales.len() * std::mem::size_of::<f32>()
            }
            Self::Int4 { data, scales, .. } => {
                data.len() * std::mem::size_of::<u8>() + scales.len() * std::mem::size_of::<f32>()
            }
        }
    }

    fn append_rows(&mut self, kv_dim: usize, src: &[f32]) {
        if kv_dim == 0 {
            return;
        }
        match self {
            Self::F32(buf) => buf.extend_from_slice(src),
            Self::Int8 {
                data,
                scales,
                positions,
            } => {
                for row in src.chunks(kv_dim) {
                    *positions += 1;
                    for group in row.chunks(GROUP_SIZE) {
                        let scale = group_scale(group, 127.0);
                        scales.push(scale);
                        for &v in group {
                            data.push(quantize_symmetric(v, scale, 127.0));
                        }
                    }
                }
            }
            Self::Int4 {
                data,
                scales,
                positions,
            } => {
                for row in src.chunks(kv_dim) {
                    *positions += 1;
                    for group in row.chunks(GROUP_SIZE) {
                        let scale = group_scale(group, 7.0);
                        scales.push(scale);
                        for pair in group.chunks(2) {
                            let q0 = quantize_symmetric(pair[0], scale, 7.0);
                            let q1 = pair
                                .get(1)
                                .copied()
                                .map_or(0, |v| quantize_symmetric(v, scale, 7.0));
                            let packed = ((q0 as u8) & 0x0F) | (((q1 as u8) & 0x0F) << 4);
                            data.push(packed);
                        }
                    }
                }
            }
        }
    }

    fn to_f32_cow(&self, kv_dim: usize) -> Cow<'_, [f32]> {
        match self {
            Self::F32(buf) => Cow::Borrowed(buf),
            Self::Int8 {
                data,
                scales,
                positions,
            } => {
                let mut out = Vec::with_capacity(positions * kv_dim);
                let mut scale_iter = scales.iter().copied();
                let mut remaining = data.as_slice();
                for _ in 0..*positions {
                    let mut left = kv_dim;
                    while left > 0 {
                        let group_len = left.min(GROUP_SIZE);
                        let scale = scale_iter.next().unwrap_or(0.0);
                        let (group_data, rest) = remaining.split_at(group_len.min(remaining.len()));
                        remaining = rest;
                        if scale == 0.0 {
                            out.resize(out.len() + group_len, 0.0);
                        } else {
                            for &q in group_data {
                                out.push(f32::from(q) * scale);
                            }
                        }
                        left -= group_len;
                    }
                }
                Cow::Owned(out)
            }
            Self::Int4 {
                data,
                scales,
                positions,
            } => {
                let mut out = Vec::with_capacity(positions * kv_dim);
                let mut scale_iter = scales.iter().copied();
                let mut remaining = data.as_slice();
                for _ in 0..*positions {
                    let mut left = kv_dim;
                    while left > 0 {
                        let group_len = left.min(GROUP_SIZE);
                        let byte_len = group_len.div_ceil(2);
                        let scale = scale_iter.next().unwrap_or(0.0);
                        let (group_bytes, rest) = remaining.split_at(byte_len.min(remaining.len()));
                        remaining = rest;
                        if scale == 0.0 {
                            out.resize(out.len() + group_len, 0.0);
                        } else {
                            let mut emitted = 0;
                            for &packed in group_bytes {
                                let q0 = ((packed << 4) as i8) >> 4;
                                out.push(f32::from(q0) * scale);
                                emitted += 1;
                                if emitted < group_len {
                                    let q1 = (packed as i8) >> 4;
                                    out.push(f32::from(q1) * scale);
                                    emitted += 1;
                                }
                            }
                        }
                        left -= group_len;
                    }
                }
                Cow::Owned(out)
            }
        }
    }
}

/// Keys and values of one transformer layer, one `kv_dim` row per position.
pub(crate) struct KvLayer {
    kv_dim: usize,
    keys: KvStorage,
    values: KvStorage,
}

impl KvLayer {
    fn new(kv_dim: usize, keys: KvPrecision, values: KvPrecision) -> Self {
        Self {
            kv_dim,
            keys: KvStorage::new(keys),
            values: KvStorage::new(values),
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.keys.len(self.kv_dim)
    }

    fn bytes(&self) -> usize {
        self.keys.bytes() + self.values.bytes()
    }

    /// Appends the rows of `keys` and `values` (both `kv_dim` wide per position).
    pub(crate) fn append(&mut self, keys: &[f32], values: &[f32]) {
        self.keys.append_rows(self.kv_dim, keys);
        self.values.append_rows(self.kv_dim, values);
    }

    pub(crate) fn keys_f32(&self) -> Cow<'_, [f32]> {
        self.keys.to_f32_cow(self.kv_dim)
    }

    pub(crate) fn values_f32(&self) -> Cow<'_, [f32]> {
        self.values.to_f32_cow(self.kv_dim)
    }
}

/// Opaque device-resident KV cache state (e.g., CUDA `CudaSlice<u16>` buffers).
pub(crate) struct DeviceKvHandle {
    pub(crate) len: usize,
    pub(crate) bytes: usize,
    pub(crate) state: Box<dyn std::any::Any + Send + Sync>,
}

/// The cached keys and values of every layer for one sequence.
pub struct KvCache {
    kv_dim: usize,
    key_precision: KvPrecision,
    value_precision: KvPrecision,
    layers: Vec<KvLayer>,
    device: Option<DeviceKvHandle>,
}

impl KvCache {
    /// An empty F32 cache for `num_layers` layers whose keys and values are `kv_dim` wide.
    pub fn new(num_layers: usize, kv_dim: usize) -> Self {
        Self::with_precision(num_layers, kv_dim, KvPrecision::F32)
    }

    /// An empty cache for `num_layers` layers with the specified storage precision.
    pub fn with_precision(num_layers: usize, kv_dim: usize, precision: KvPrecision) -> Self {
        Self::with_precisions(num_layers, kv_dim, precision, precision)
    }

    /// An empty cache for `num_layers` layers with separate storage precisions for keys and values.
    pub fn with_precisions(
        num_layers: usize,
        kv_dim: usize,
        keys: KvPrecision,
        values: KvPrecision,
    ) -> Self {
        Self {
            kv_dim,
            key_precision: keys,
            value_precision: values,
            layers: (0..num_layers)
                .map(|_| KvLayer::new(kv_dim, keys, values))
                .collect(),
            device: None,
        }
    }

    /// Number of positions (tokens) cached so far.
    pub fn len(&self) -> usize {
        if let Some(dev) = &self.device {
            return dev.len;
        }
        self.layers.first().map_or(0, KvLayer::len)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Width of one key (or value) row: KV heads times head dimension.
    pub fn kv_dim(&self) -> usize {
        self.kv_dim
    }

    /// Storage precision configured for cached keys.
    pub fn key_precision(&self) -> KvPrecision {
        self.key_precision
    }

    /// Storage precision configured for cached values.
    pub fn value_precision(&self) -> KvPrecision {
        self.value_precision
    }

    /// Storage precision configured for this cache; returns [`KvPrecision::F32`] only when
    /// both keys and values are `F32`, otherwise prefers the key precision.
    pub fn precision(&self) -> KvPrecision {
        if self.key_precision == self.value_precision {
            self.key_precision
        } else if self.key_precision == KvPrecision::F32 {
            self.value_precision
        } else {
            self.key_precision
        }
    }

    /// Bytes of key and value storage held right now, for measured KV-cache reports.
    pub fn bytes(&self) -> usize {
        if let Some(dev) = &self.device {
            return dev.bytes;
        }
        self.layers.iter().map(KvLayer::bytes).sum()
    }

    pub(crate) fn has_device_state(&self) -> bool {
        self.device.is_some()
    }

    pub(crate) fn device_state_mut<T: 'static>(
        &mut self,
    ) -> Option<(&mut T, &mut usize, &mut usize)> {
        let dev = self.device.as_mut()?;
        let state = dev.state.downcast_mut::<T>()?;
        Some((state, &mut dev.len, &mut dev.bytes))
    }

    pub(crate) fn set_device_state<T: Send + Sync + 'static>(
        &mut self,
        state: T,
        len: usize,
        bytes: usize,
    ) {
        self.device = Some(DeviceKvHandle {
            len,
            bytes,
            state: Box::new(state),
        });
    }

    pub(crate) fn layer_mut(&mut self, index: usize) -> Result<&mut KvLayer> {
        match self.layers.get_mut(index) {
            Some(layer) => Ok(layer),
            None => bail!("KV cache has no layer {index}"),
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::inference::CausalLm;
    use crate::profiling::kv_cache::KvCacheEstimator;

    /// Largest absolute difference allowed between incremental and full-sequence logits.
    /// Both paths sum the same products, but the blocked matmul groups them differently.
    pub(crate) const INCREMENTAL_TOLERANCE: f32 = 1e-4;

    /// Feeds `tokens` into a fresh cache in chunks (first `prefill` tokens, then one token at a
    /// time) and returns the largest gap to `forward_logits` over the full prefix, at every
    /// position from the second token on.
    pub(crate) fn max_incremental_gap(
        model: &dyn CausalLm,
        tokens: &[u32],
        prefill: usize,
    ) -> Result<f32> {
        let Some(mut cache) = model.new_cache() else {
            bail!("model has no KV-cache path");
        };
        let mut worst = 0.0f32;
        let mut end = 0;
        while end < tokens.len() {
            let step = if end == 0 { prefill } else { 1 };
            let next = (end + step).min(tokens.len());
            let logits = model.forward_extend(&mut cache, &tokens[end..next])?;
            end = next;
            if end >= 2 {
                let full = model.forward_logits(&tokens[..end])?;
                assert_eq!(logits.len(), full.len());
                for (a, b) in logits.iter().zip(&full) {
                    assert!(a.is_finite() && b.is_finite(), "non-finite logit");
                    worst = worst.max((a - b).abs());
                }
            }
        }
        assert_eq!(cache.len(), tokens.len());
        Ok(worst)
    }

    /// Feeds `tokens` step by step into a cache of `precision` and an F32 cache, returning the
    /// largest absolute logit gap across all positions (or `NaN` if any logit is non-finite).
    pub(crate) fn kv_precision_logit_gap(
        model: &dyn CausalLm,
        tokens: &[u32],
        precision: KvPrecision,
    ) -> f32 {
        assert!(!tokens.is_empty());
        let mut q_cache = model
            .new_cache_with_precision(precision)
            .expect("model supports requested KV precision");
        let mut f_cache = model
            .new_cache_with_precision(KvPrecision::F32)
            .expect("model supports F32 KV cache");
        let mut worst = 0.0f32;
        for &tok in tokens {
            let q_logits = model.forward_extend(&mut q_cache, &[tok]).unwrap();
            let f_logits = model.forward_extend(&mut f_cache, &[tok]).unwrap();
            assert_eq!(q_logits.len(), f_logits.len());
            for (a, b) in q_logits.iter().zip(&f_logits) {
                if !a.is_finite() || !b.is_finite() {
                    return f32::NAN;
                }
                worst = worst.max((a - b).abs());
            }
        }
        assert_eq!(q_cache.len(), tokens.len());
        assert_eq!(f_cache.len(), tokens.len());
        worst
    }

    #[test]
    fn cache_bytes_match_the_profiling_estimate() {
        // 2 KV heads of width 4 give kv_dim 8; five positions in two layers.
        let mut cache = KvCache::new(2, 8);
        assert!(cache.is_empty());
        for index in 0..2 {
            for _ in 0..5 {
                let row = [0.5f32; 8];
                cache.layer_mut(index).unwrap().append(&row, &row);
            }
        }
        assert_eq!(cache.len(), 5);
        assert_eq!(cache.num_layers(), 2);
        let estimate = KvCacheEstimator::calculate_bytes(2, 1, 5, 2, 4, 4.0);
        assert_eq!(cache.bytes(), estimate);
        assert_eq!(cache.bytes(), 640);
        assert!(cache.layer_mut(2).is_err());
    }

    #[test]
    fn int8_round_trip_stays_within_per_group_error_bound() {
        #[rustfmt::skip]
        let row_k: [f32; 40] = [
            // Group 0 (32 values): max|x| = 15.875, bound = 15.875 / 254 = 0.0625
            15.875, -14.2, 13.1, -12.4, 11.7, -10.3, 9.6, -8.5,
            7.2, -6.8, 5.1, -4.7, 3.3, -2.9, 1.8, -1.1,
            0.9, -0.6, 0.4, -0.2, 0.1, -0.05, 0.0, 2.4,
            -3.7, 6.1, -9.2, 10.8, -11.9, 13.7, -15.1, 4.4,
            // Group 1 (8 values): max|x| = 7.9375, bound = 7.9375 / 254 = 0.03125
            -7.9375, 6.4, -4.1, 2.3, -1.2, 0.5, -0.15, 0.05,
        ];
        let row_v: [f32; 40] = row_k.map(|v| -v);

        let mut cache = KvCache::with_precision(1, 40, KvPrecision::Int8);
        cache.layer_mut(0).unwrap().append(&row_k, &row_v);
        let layer = cache.layer_mut(0).unwrap();
        let keys = layer.keys_f32();
        let values = layer.values_f32();
        assert!(matches!(keys, Cow::Owned(_)));
        assert!(matches!(values, Cow::Owned(_)));
        assert_eq!(keys.len(), 40);
        assert_eq!(values.len(), 40);

        assert_eq!(keys[0], 15.875);
        assert_eq!(keys[22], 0.0);
        assert_eq!(keys[32], -7.9375);
        assert_eq!(keys[39], 0.0625);

        for (i, (&k_dec, &k_ref)) in keys.iter().zip(&row_k).enumerate() {
            let bound = if i < 32 { 0.0625 } else { 0.03125 };
            assert!((k_dec - k_ref).abs() <= bound, "int8 key[{i}]");
        }
        for (i, (&v_dec, &v_ref)) in values.iter().zip(&row_v).enumerate() {
            let bound = if i < 32 { 0.0625 } else { 0.03125 };
            assert!((v_dec - v_ref).abs() <= bound, "int8 val[{i}]");
        }
    }

    #[test]
    fn int4_round_trip_stays_within_per_group_error_bound() {
        #[rustfmt::skip]
        let row_k: [f32; 40] = [
            // Group 0 (32 values): max|x| = 7.0, bound = 7.0 / 14 = 0.5
            7.0, -6.3, 5.4, -4.2, 3.8, -2.6, 1.4, -0.7,
            0.3, -0.1, 0.0, 6.8, -5.9, 4.7, -3.3, 2.2,
            -1.6, 0.9, -0.4, 5.1, -6.6, 3.1, -2.4, 1.8,
            -1.2, 0.6, -3.9, 4.4, -5.2, 6.1, -6.9, 2.5,
            // Group 1 (8 values): max|x| = 1.75, bound = 1.75 / 14 = 0.125
            -1.75, 1.4, -1.1, 0.8, -0.6, 0.35, -0.2, 0.05,
        ];
        let row_v: [f32; 40] = row_k.map(|v| -v);

        let mut cache = KvCache::with_precision(1, 40, KvPrecision::Int4);
        cache.layer_mut(0).unwrap().append(&row_k, &row_v);
        let layer = cache.layer_mut(0).unwrap();
        let keys = layer.keys_f32();
        let values = layer.values_f32();
        assert!(matches!(keys, Cow::Owned(_)));
        assert!(matches!(values, Cow::Owned(_)));
        assert_eq!(keys.len(), 40);
        assert_eq!(values.len(), 40);

        assert_eq!(keys[0], 7.0);
        assert_eq!(keys[1], -6.0);
        assert_eq!(keys[32], -1.75);
        assert_eq!(keys[33], 1.5);
        assert_eq!(keys[37], 0.25);

        for (i, (&k_dec, &k_ref)) in keys.iter().zip(&row_k).enumerate() {
            let bound = if i < 32 { 0.5 } else { 0.125 };
            assert!((k_dec - k_ref).abs() <= bound, "int4 key[{i}]");
        }
        for (i, (&v_dec, &v_ref)) in values.iter().zip(&row_v).enumerate() {
            let bound = if i < 32 { 0.5 } else { 0.125 };
            assert!((v_dec - v_ref).abs() <= bound, "int4 val[{i}]");
        }
    }

    #[test]
    fn all_zero_group_decodes_to_exact_zeros() {
        let zeros = [0.0f32; 40];
        let mut mixed = [0.0f32; 40];
        mixed[32..40].fill(1.0);

        for precision in [KvPrecision::Int8, KvPrecision::Int4] {
            let mut cache = KvCache::with_precision(1, 40, precision);
            let layer = cache.layer_mut(0).unwrap();
            layer.append(&zeros, &zeros);
            layer.append(&mixed, &zeros);

            let keys = layer.keys_f32();
            let values = layer.values_f32();
            assert_eq!(keys.len(), 80);
            assert_eq!(values.len(), 80);

            for &v in &keys[..72] {
                assert_eq!(v.to_bits(), 0, "{precision:?} zero key element");
            }
            for &v in &keys[72..80] {
                assert!((v - 1.0).abs() <= 1e-6, "{precision:?} non-zero tail");
            }
            for &v in values.as_ref() {
                assert_eq!(v.to_bits(), 0, "{precision:?} zero value element");
            }
        }
    }

    #[test]
    fn quantized_cache_bytes_match_packed_layout() {
        let row = [0.25f32; 40];

        let mut int8 = KvCache::with_precision(1, 40, KvPrecision::Int8);
        for _ in 0..3 {
            int8.layer_mut(0).unwrap().append(&row, &row);
        }
        assert_eq!(int8.len(), 3);
        assert_eq!(int8.bytes(), 288);

        let mut int4 = KvCache::with_precision(1, 40, KvPrecision::Int4);
        for _ in 0..3 {
            int4.layer_mut(0).unwrap().append(&row, &row);
        }
        assert_eq!(int4.len(), 3);
        assert_eq!(int4.bytes(), 168);
    }

    #[test]
    fn mixed_precision_cache_bytes_match_packed_layout() {
        let row = [0.25f32; 40];

        let mut f32_int8 = KvCache::with_precisions(1, 40, KvPrecision::F32, KvPrecision::Int8);
        assert_eq!(f32_int8.key_precision(), KvPrecision::F32);
        assert_eq!(f32_int8.value_precision(), KvPrecision::Int8);
        assert_eq!(f32_int8.precision(), KvPrecision::Int8);
        for _ in 0..3 {
            f32_int8.layer_mut(0).unwrap().append(&row, &row);
        }
        assert_eq!(f32_int8.len(), 3);
        // Keys: 40 * 4 = 160 B/row; values: 40 + 2 * 4 = 48 B/row; 3 * (160 + 48) = 624.
        assert_eq!(f32_int8.bytes(), 624);

        let mut int8_f32 = KvCache::with_precisions(1, 40, KvPrecision::Int8, KvPrecision::F32);
        assert_eq!(int8_f32.key_precision(), KvPrecision::Int8);
        assert_eq!(int8_f32.value_precision(), KvPrecision::F32);
        assert_eq!(int8_f32.precision(), KvPrecision::Int8);
        for _ in 0..3 {
            int8_f32.layer_mut(0).unwrap().append(&row, &row);
        }
        assert_eq!(int8_f32.len(), 3);
        assert_eq!(int8_f32.bytes(), 624);
    }

    #[test]
    fn f32_precision_returns_exact_appended_values() {
        assert_eq!(KvPrecision::default(), KvPrecision::F32);
        let mut cache = KvCache::with_precision(1, 40, KvPrecision::F32);
        let keys: [f32; 40] = std::array::from_fn(|i| (i as f32) * 0.125 - 2.5);
        let values: [f32; 40] = std::array::from_fn(|i| 3.0 - (i as f32) * 0.25);
        cache.layer_mut(0).unwrap().append(&keys, &values);

        let layer = cache.layer_mut(0).unwrap();
        let k_out = layer.keys_f32();
        let v_out = layer.values_f32();
        assert!(matches!(k_out, Cow::Borrowed(_)));
        assert!(matches!(v_out, Cow::Borrowed(_)));
        assert_eq!(k_out.as_ref(), &keys[..]);
        assert_eq!(v_out.as_ref(), &values[..]);
    }
}
