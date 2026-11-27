// SPDX-License-Identifier: AGPL-3.0-only
//! Device-resident weight cache and slice fingerprinting for the WGPU backend.

use std::collections::HashMap;
use std::sync::Arc;

use super::WeightStorage;

/// Default weight bytes kept resident for slice-backed linears before LRU eviction;
/// raise it with [`GpuContext::set_weight_budget`] when the card has the memory.
pub(super) const WEIGHT_CACHE_BUDGET: usize = 10 << 30;

pub(super) struct CachedWeight {
    pub(super) fingerprint: u64,
    pub(super) buffer: Arc<wgpu::Buffer>,
    pub(super) bytes: usize,
    pub(super) last_used: u64,
}

#[derive(Default)]
pub(super) struct WeightCache {
    pub(super) entries: HashMap<(usize, usize, u8), CachedWeight>,
    pub(super) bytes: usize,
    pub(super) tick: u64,
}

/// Cheap identity check for a weight slice: length plus 64 samples spread over it.
/// Guards against a freed model's address being reused by different weights.
pub(super) fn fingerprint(w: &[f32]) -> u64 {
    let step = (w.len() / 64).max(1);
    let mut h: u64 = 0xcbf2_9ce4_8422_2325 ^ w.len() as u64;
    for v in w.iter().step_by(step) {
        h = (h ^ u64::from(v.to_bits())).wrapping_mul(0x0100_0000_01b3);
    }
    h
}

/// A weight matrix on the device with the storage format its kernel must decode.
#[derive(Clone)]
pub(in super::super) struct DeviceWeight {
    pub(in super::super) buffer: Arc<wgpu::Buffer>,
    pub(in super::super) storage: WeightStorage,
}
