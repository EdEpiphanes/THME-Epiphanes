// SPDX-License-Identifier: AGPL-3.0-only
//! Device-resident weight cache with LRU eviction for the CUDA backend.

use std::collections::HashMap;
use std::sync::Arc;

use cudarc::driver::CudaSlice;

/// Default weight bytes kept resident on CUDA before LRU eviction (10 GiB).
const WEIGHT_CACHE_BUDGET: usize = 10 << 30;

#[derive(Clone)]
pub(super) enum CachedBuffer {
    F16(Arc<CudaSlice<u16>>),
    F32(Arc<CudaSlice<f32>>),
    Q4_0 {
        scales: Arc<CudaSlice<u16>>,
        qs: Arc<CudaSlice<u32>>,
    },
}

pub(super) struct CachedEntry {
    pub(super) fingerprint: u64,
    pub(super) buffer: CachedBuffer,
    pub(super) bytes: usize,
    pub(super) last_used: u64,
}

#[derive(Default)]
pub(super) struct WeightCache {
    pub(super) entries: HashMap<(usize, usize, u8), CachedEntry>,
    pub(super) bytes: usize,
    pub(super) tick: u64,
}

impl WeightCache {
    pub(super) fn evict_if_needed(&mut self, incoming: usize) {
        while self.bytes + incoming > WEIGHT_CACHE_BUDGET && !self.entries.is_empty() {
            let Some(oldest_key) = self
                .entries
                .iter()
                .min_by_key(|(_, entry)| entry.last_used)
                .map(|(k, _)| *k)
            else {
                break;
            };
            if let Some(removed) = self.entries.remove(&oldest_key) {
                self.bytes = self.bytes.saturating_sub(removed.bytes);
            }
        }
    }
}
