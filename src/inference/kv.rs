// SPDX-License-Identifier: AGPL-3.0-only
//! Key/value cache for incremental decoding in the reference forward pass (THME-3070).
//!
//! Each layer keeps the rotary-embedded keys and the values of every position seen so far,
//! so a new token costs one row of work per layer instead of a pass over the whole sequence.
//! The cache stores `f32`, the same precision as the reference weights.

use anyhow::{bail, Result};

/// Keys and values of one transformer layer, one `kv_dim` row per position.
pub(crate) struct KvLayer {
    kv_dim: usize,
    keys: Vec<f32>,
    values: Vec<f32>,
}

impl KvLayer {
    pub(crate) fn len(&self) -> usize {
        self.keys.len().checked_div(self.kv_dim).unwrap_or(0)
    }

    /// Appends the rows of `keys` and `values` (both `kv_dim` wide per position).
    pub(crate) fn append(&mut self, keys: &[f32], values: &[f32]) {
        self.keys.extend_from_slice(keys);
        self.values.extend_from_slice(values);
    }

    pub(crate) fn keys(&self) -> &[f32] {
        &self.keys
    }

    pub(crate) fn values(&self) -> &[f32] {
        &self.values
    }
}

/// The cached keys and values of every layer for one sequence.
pub struct KvCache {
    kv_dim: usize,
    layers: Vec<KvLayer>,
}

impl KvCache {
    /// An empty cache for `num_layers` layers whose keys and values are `kv_dim` wide.
    pub fn new(num_layers: usize, kv_dim: usize) -> Self {
        Self {
            kv_dim,
            layers: (0..num_layers)
                .map(|_| KvLayer {
                    kv_dim,
                    keys: Vec::new(),
                    values: Vec::new(),
                })
                .collect(),
        }
    }

    /// Number of positions (tokens) cached so far.
    pub fn len(&self) -> usize {
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

    /// Bytes of key and value storage held right now, for measured KV-cache reports.
    pub fn bytes(&self) -> usize {
        2 * self.num_layers() * self.len() * self.kv_dim * std::mem::size_of::<f32>()
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
}
