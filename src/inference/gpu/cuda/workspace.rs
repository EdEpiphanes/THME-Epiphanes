// SPDX-License-Identifier: AGPL-3.0-only
//! Reusable CUDA workspace buffers and captured graph handle for decoder passes.

use cudarc::driver::{CudaGraph, CudaSlice};

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) struct CudaWorkspaceKey {
    pub(super) t: usize,
    pub(super) d: usize,
    pub(super) kv_dim: usize,
    pub(super) inter: usize,
    pub(super) vocab: usize,
    pub(super) nll_mode: bool,
    pub(super) head_dim: usize,
}

pub(super) struct SendCudaGraph(pub(super) CudaGraph);
// SAFETY: `CudaGraph` is only accessed while holding `CudaContext::workspace` and launched on `self.stream`.
unsafe impl Send for SendCudaGraph {}
unsafe impl Sync for SendCudaGraph {}

pub(super) struct CudaWorkspace {
    pub(super) x_a: CudaSlice<f32>,
    pub(super) x_b: CudaSlice<f32>,
    pub(super) x16: CudaSlice<u16>,
    pub(super) q16: CudaSlice<u16>,
    pub(super) k16: CudaSlice<u16>,
    pub(super) v16: CudaSlice<u16>,
    pub(super) attn_scores: CudaSlice<f32>,
    pub(super) attn_probs16: CudaSlice<u16>,
    pub(super) q: CudaSlice<f32>,
    pub(super) k: CudaSlice<f32>,
    pub(super) v: CudaSlice<f32>,
    pub(super) mixed: CudaSlice<f32>,
    pub(super) tmp: CudaSlice<f32>,
    pub(super) tmp16: CudaSlice<u16>,
    pub(super) gate: CudaSlice<f32>,
    pub(super) gate16: CudaSlice<u16>,
    pub(super) up: CudaSlice<f32>,
    pub(super) up16: CudaSlice<u16>,
    pub(super) last: CudaSlice<f32>,
    pub(super) logits: CudaSlice<f32>,
    pub(super) logits16: CudaSlice<u16>,
    pub(super) scores: CudaSlice<f32>,
    pub(super) targets: CudaSlice<u32>,
    pub(super) cos: CudaSlice<f32>,
    pub(super) sin: CudaSlice<f32>,
    pub(super) cos_hash: u64,
    pub(super) sin_hash: u64,
    pub(super) graph: Option<(u64, SendCudaGraph)>,
}
