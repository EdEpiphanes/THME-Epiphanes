// SPDX-License-Identifier: AGPL-3.0-only
//! Pure analytic memory model for a single inference run (THME-3056).
//!
//! Computes weight, KV cache, and single-token activation scratch bytes from model dimensions
//! without reading model files. All outputs are analytic estimates.

use serde::{Deserialize, Serialize};

const KV_GROUP_SIZE: usize = 32;

/// Architectural dimensions needed to estimate run memory without reading weights.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelShape {
    pub layers: usize,
    pub hidden: usize,
    pub intermediate: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub vocab: usize,
    pub gated_mlp: bool,
    pub tied_embeddings: bool,
}

/// Key/value cache element storage precision for memory estimation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum KvType {
    F32,
    F16,
    Int8,
    Int4,
}

impl KvType {
    /// Bytes required to store one KV row of `kv_dim` values at this precision.
    ///
    /// `Int8` uses 1 byte per value plus a 4-byte scale per 32-value group, and `Int4` uses
    /// 0.5 byte per value (rounded up to whole bytes) plus a 4-byte scale per 32-value group,
    /// accounted per row with a shorter final group, matching [`crate::inference::kv::KvPrecision`].
    pub fn row_bytes(self, kv_dim: usize) -> u64 {
        let dim = kv_dim as u64;
        let groups = kv_dim.div_ceil(KV_GROUP_SIZE) as u64;
        match self {
            Self::F32 => dim.saturating_mul(4),
            Self::F16 => dim.saturating_mul(2),
            Self::Int8 => dim.saturating_add(groups.saturating_mul(4)),
            Self::Int4 => (kv_dim.div_ceil(2) as u64).saturating_add(groups.saturating_mul(4)),
        }
    }
}

/// Analytic memory breakdown in bytes for weights, KV cache, and single-token scratch space.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct RunMemory {
    /// Estimated weight storage in bytes.
    pub weights: u64,
    /// KV cache capacity in bytes across all layers and both K and V tensors.
    pub kv: u64,
    /// Estimate of single-token activation scratch space in bytes.
    pub scratch: u64,
}

impl RunMemory {
    /// Total estimated memory in bytes across weights, KV cache, and scratch space.
    pub const fn total(&self) -> u64 {
        self.weights
            .saturating_add(self.kv)
            .saturating_add(self.scratch)
    }
}

/// Estimates weight, KV cache, and single-token activation scratch bytes for a run.
///
/// The `scratch` component is an analytic estimate for one token step:
/// `(hidden * 4 + intermediate * 2 + context * heads)` `f32` values times 4 bytes.
pub fn run_memory(
    shape: &ModelShape,
    weight_bytes_per_param: f64,
    context: usize,
    kv: KvType,
) -> RunMemory {
    let layers = shape.layers as u64;
    let hidden = shape.hidden as u64;
    let intermediate = shape.intermediate as u64;
    let heads = shape.heads as u64;
    let vocab = shape.vocab as u64;
    let ctx = context as u64;

    let head_dim = shape.hidden.checked_div(shape.heads).unwrap_or(0);
    let q_dim = (shape.heads.saturating_mul(head_dim)) as u64;
    let kv_dim = shape.kv_heads.saturating_mul(head_dim);

    let embed_params = vocab.saturating_mul(hidden);
    let head_params = if shape.tied_embeddings {
        0
    } else {
        embed_params
    };

    let attn_params = hidden
        .saturating_mul(q_dim)
        .saturating_mul(2)
        .saturating_add(hidden.saturating_mul(kv_dim as u64).saturating_mul(2));
    let mlp_matrices = if shape.gated_mlp { 3_u64 } else { 2_u64 };
    let mlp_params = mlp_matrices
        .saturating_mul(hidden)
        .saturating_mul(intermediate);
    let layer_norm_params = hidden.saturating_mul(2);
    let per_layer_params = attn_params
        .saturating_add(mlp_params)
        .saturating_add(layer_norm_params);

    let total_params = embed_params
        .saturating_add(head_params)
        .saturating_add(layers.saturating_mul(per_layer_params))
        .saturating_add(hidden);

    let weights = if !weight_bytes_per_param.is_finite() || weight_bytes_per_param <= 0.0 {
        0
    } else {
        ((total_params as f64) * weight_bytes_per_param).ceil() as u64
    };

    let kv_bytes = 2_u64
        .saturating_mul(layers)
        .saturating_mul(ctx)
        .saturating_mul(kv.row_bytes(kv_dim));

    let scratch_elems = hidden
        .saturating_mul(4)
        .saturating_add(intermediate.saturating_mul(2))
        .saturating_add(ctx.saturating_mul(heads));
    let scratch = scratch_elems.saturating_mul(4);

    RunMemory {
        weights,
        kv: kv_bytes,
        scratch,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_llama() -> ModelShape {
        ModelShape {
            layers: 2,
            hidden: 8,
            intermediate: 16,
            heads: 2,
            kv_heads: 1,
            vocab: 10,
            gated_mlp: true,
            tied_embeddings: false,
        }
    }

    fn tiny_neox() -> ModelShape {
        ModelShape {
            layers: 2,
            hidden: 8,
            intermediate: 32,
            heads: 2,
            kv_heads: 2,
            vocab: 10,
            gated_mlp: false,
            tied_embeddings: true,
        }
    }

    #[test]
    fn row_bytes_matches_packed_group_layout() {
        // 0 values -> 0 bytes for all precisions.
        assert_eq!(KvType::F32.row_bytes(0), 0);
        assert_eq!(KvType::F16.row_bytes(0), 0);
        assert_eq!(KvType::Int8.row_bytes(0), 0);
        assert_eq!(KvType::Int4.row_bytes(0), 0);

        // 4 values (1 partial group of 4):
        // F32 = 4 * 4 = 16; F16 = 4 * 2 = 8;
        // Int8 = 4 + 4 = 8; Int4 = 2 + 4 = 6.
        assert_eq!(KvType::F32.row_bytes(4), 16);
        assert_eq!(KvType::F16.row_bytes(4), 8);
        assert_eq!(KvType::Int8.row_bytes(4), 8);
        assert_eq!(KvType::Int4.row_bytes(4), 6);

        // 40 values (group 0 of 32 + group 1 of 8, 2 scales = 8 bytes):
        // Int8 = 40 + 8 = 48 (6 rows = 288 bytes, matching src/inference/kv.rs);
        // Int4 = 20 + 8 = 28 (6 rows = 168 bytes, matching src/inference/kv.rs).
        assert_eq!(KvType::Int8.row_bytes(40), 48);
        assert_eq!(KvType::Int4.row_bytes(40), 28);

        // 33 values (group 0 of 32 + group 1 of 1, 2 scales = 8 bytes):
        // Int8 = 33 + 8 = 41; Int4 = 16 + 1 + 8 = 25.
        assert_eq!(KvType::Int8.row_bytes(33), 41);
        assert_eq!(KvType::Int4.row_bytes(33), 25);
    }

    #[test]
    fn tiny_llama_f32_and_int8_match_hand_computed_literals() {
        // Tiny LLaMA: layers=2, hidden=8, intermediate=16, heads=2, kv_heads=1, vocab=10, gated, untied.
        // head_dim = 8 / 2 = 4, kv_dim = 1 * 4 = 4.
        // Embeddings (10*8 = 80) + untied output head (10*8 = 80) = 160 params.
        // Per layer:
        //   attn q(8*8=64) + k(8*4=32) + v(8*4=32) + o(8*8=64) = 192 params
        //   gated MLP gate(8*16=128) + up(8*16=128) + down(16*8=128) = 384 params
        //   two norms (2*8 = 16) params
        //   layer total = 192 + 384 + 16 = 592 params.
        // 2 layers = 2 * 592 = 1184 params; final norm = 8 params.
        // Total params = 160 + 1184 + 8 = 1352.
        // At 4.0 bytes/param: weights = 1352 * 4 = 5408 bytes.
        //
        // Scratch (context=5): (8*4 + 16*2 + 5*2) * 4 = (32 + 32 + 10) * 4 = 74 * 4 = 296 bytes.
        let shape = tiny_llama();

        // F32 KV at context=5:
        //   1 row of kv_dim=4 at F32 = 4 * 4 = 16 bytes.
        //   kv = 2 * 2 * 5 * 16 = 320 bytes.
        //   total = 5408 + 320 + 296 = 6024 bytes.
        let mem_f32 = run_memory(&shape, 4.0, 5, KvType::F32);
        assert_eq!(mem_f32.weights, 5408);
        assert_eq!(mem_f32.kv, 320);
        assert_eq!(mem_f32.scratch, 296);
        assert_eq!(mem_f32.total(), 6024);

        // Int8 KV at context=5:
        //   1 row of kv_dim=4 at Int8 (1 partial group of 4) = 4 data + 4 scale = 8 bytes.
        //   kv = 2 * 2 * 5 * 8 = 160 bytes.
        //   total = 5408 + 160 + 296 = 5864 bytes.
        let mem_int8 = run_memory(&shape, 4.0, 5, KvType::Int8);
        assert_eq!(mem_int8.weights, 5408);
        assert_eq!(mem_int8.kv, 160);
        assert_eq!(mem_int8.scratch, 296);
        assert_eq!(mem_int8.total(), 5864);
    }

    #[test]
    fn tiny_neox_f32_and_int8_match_hand_computed_literals() {
        // Tiny GPT-NeoX: layers=2, hidden=8, intermediate=32, heads=2, kv_heads=2, vocab=10, not gated, tied.
        // head_dim = 8 / 2 = 4, kv_dim = 2 * 4 = 8.
        // Tied embeddings: 10 * 8 = 80 params (no separate output head).
        // Per layer:
        //   attn q(8*8=64) + k(8*8=64) + v(8*8=64) + o(8*8=64) = 256 params
        //   ungated MLP up(8*32=256) + down(32*8=256) = 512 params
        //   two norms (2*8 = 16) params
        //   layer total = 256 + 512 + 16 = 784 params.
        // 2 layers = 2 * 784 = 1568 params; final norm = 8 params.
        // Total params = 80 + 1568 + 8 = 1656.
        // At 4.0 bytes/param: weights = 1656 * 4 = 6624 bytes.
        //
        // Scratch (context=5): (8*4 + 32*2 + 5*2) * 4 = (32 + 64 + 10) * 4 = 106 * 4 = 424 bytes.
        let shape = tiny_neox();

        // F32 KV at context=5:
        //   1 row of kv_dim=8 at F32 = 8 * 4 = 32 bytes.
        //   kv = 2 * 2 * 5 * 32 = 640 bytes.
        //   total = 6624 + 640 + 424 = 7688 bytes.
        let mem_f32 = run_memory(&shape, 4.0, 5, KvType::F32);
        assert_eq!(mem_f32.weights, 6624);
        assert_eq!(mem_f32.kv, 640);
        assert_eq!(mem_f32.scratch, 424);
        assert_eq!(mem_f32.total(), 7688);

        // Int8 KV at context=5:
        //   1 row of kv_dim=8 at Int8 (1 partial group of 8) = 8 data + 4 scale = 12 bytes.
        //   kv = 2 * 2 * 5 * 12 = 240 bytes.
        //   total = 6624 + 240 + 424 = 7288 bytes.
        let mem_int8 = run_memory(&shape, 4.0, 5, KvType::Int8);
        assert_eq!(mem_int8.weights, 6624);
        assert_eq!(mem_int8.kv, 240);
        assert_eq!(mem_int8.scratch, 424);
        assert_eq!(mem_int8.total(), 7288);
    }

    #[test]
    fn degenerate_inputs_do_not_panic() {
        let zero_heads = ModelShape {
            layers: 1,
            hidden: 8,
            intermediate: 16,
            heads: 0,
            kv_heads: 0,
            vocab: 10,
            gated_mlp: true,
            tied_embeddings: true,
        };
        let mem = run_memory(&zero_heads, f64::NAN, 0, KvType::Int4);
        assert_eq!(mem.weights, 0);
        assert_eq!(mem.kv, 0);
        // (8*4 + 16*2 + 0*0) * 4 = 64 * 4 = 256.
        assert_eq!(mem.scratch, 256);
        assert_eq!(mem.total(), 256);
    }
}
