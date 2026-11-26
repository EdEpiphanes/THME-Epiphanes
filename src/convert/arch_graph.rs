// SPDX-License-Identifier: AGPL-3.0-only
//! §7.10 Cross-Architecture & LLM Model-Type Conversion Engine (Beyond Quantization).

use anyhow::{bail, Result};

/// Supported architectural topologies for structural model-type conversion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelArchKind {
    DenseMha,
    DenseGqa,
    SparseMoe,
    DoryLooped,
    HybridSsm,
}

/// RoPE pair layout dialect.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RopeDialect {
    Interleaved,
    SplitHalf,
}

/// Symmetry-preserving structural conversion operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConversionOp {
    MhaToGqa,
    DenseToMoe,
    DeepToDoryLoop,
    StreamingAttnToSsm,
    DialectPermutation,
}

/// Execution plan enforcing §7.10 two-stage `quality_gate` discipline and mandatory
/// `BitDistill` recovery for looped `Dory` models.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConversionPlan {
    pub source_arch: ModelArchKind,
    pub target_arch: ModelArchKind,
    pub op: ConversionOp,
    pub pre_quant_gate_at_f16_q8: bool,
    pub requires_bitdistill_recovery: bool,
    pub router_pinned_q8_0: bool,
}

impl ConversionPlan {
    /// Builds a validated structural conversion plan with §7.10 safety invariants.
    #[must_use]
    pub fn build(source_arch: ModelArchKind, target_arch: ModelArchKind, op: ConversionOp) -> Self {
        let requires_bitdistill_recovery = matches!(op, ConversionOp::DeepToDoryLoop);
        let router_pinned_q8_0 = matches!(op, ConversionOp::DenseToMoe);
        Self {
            source_arch,
            target_arch,
            op,
            pre_quant_gate_at_f16_q8: true,
            requires_bitdistill_recovery,
            router_pinned_q8_0,
        }
    }
}

/// Merges `group_size` adjacent KV heads via activation-whitened pooling:
/// `W_KV^(g) = argmin sum_{h in g} ||(W_h - W_KV^(g)) L_X||_F^2`.
pub fn mha_to_gqa_whitened_pool(
    heads: &[&[f32]],
    cholesky_lx_diag: &[f32],
    group_size: usize,
) -> Result<Vec<Vec<f32>>> {
    if heads.is_empty() || group_size == 0 || !heads.len().is_multiple_of(group_size) {
        bail!("head count must be a positive multiple of group_size");
    }
    let dim = heads[0].len();
    if dim == 0 || cholesky_lx_diag.len() != dim || heads.iter().any(|h| h.len() != dim) {
        bail!("all heads and cholesky_lx_diag must share a non-zero dimension");
    }
    let mut pooled = Vec::with_capacity(heads.len() / group_size);
    for group in heads.chunks_exact(group_size) {
        let mut merged = vec![0.0f32; dim];
        let mut weight_sum = vec![0.0f64; dim];
        for head in group {
            for (j, (&w, &lx)) in head.iter().zip(cholesky_lx_diag.iter()).enumerate() {
                let act_w = (lx as f64).powi(2).max(1e-12);
                merged[j] += (w as f64 * act_w) as f32;
                weight_sum[j] += act_w;
            }
        }
        for (j, w_sum) in weight_sum.iter().enumerate() {
            merged[j] = (merged[j] as f64 / *w_sum) as f32;
        }
        pooled.push(merged);
    }
    Ok(pooled)
}

/// Initializes a Sinkhorn-balanced router weight matrix `(num_experts, hidden_dim)`
/// from expert centroids so no expert suffers immediate routing collapse.
pub fn sinkhorn_balanced_router_init(
    expert_centroids: &[Vec<f32>],
    iters: usize,
) -> Result<Vec<Vec<f32>>> {
    if expert_centroids.is_empty() || expert_centroids[0].is_empty() {
        bail!("expert_centroids must be non-empty");
    }
    let hidden_dim = expert_centroids[0].len();
    if expert_centroids.iter().any(|c| c.len() != hidden_dim) {
        bail!("all expert centroids must have identical hidden_dim");
    }

    let mut router: Vec<Vec<f32>> = expert_centroids
        .iter()
        .map(|row| row.iter().map(|&v| v.abs() + 1e-6).collect())
        .collect();

    for _ in 0..iters.max(1) {
        // Row normalize (expert balance).
        for row in &mut router {
            let sum: f32 = row.iter().sum::<f32>().max(1e-9);
            for v in row.iter_mut() {
                *v /= sum;
            }
        }
        // Column normalize (feature balance).
        for col in 0..hidden_dim {
            let col_sum: f32 = router.iter().map(|r| r[col]).sum::<f32>().max(1e-9);
            for row in &mut router {
                row[col] /= col_sum;
            }
        }
    }
    Ok(router)
}

/// Exact RoPE permutation `P_rope` between `Interleaved` `(x0, y0, x1, y1, ...)`
/// and `SplitHalf` `(x0, x1, ..., y0, y1, ...)`.
pub fn permute_rope_vector(vec: &mut [f32], from: RopeDialect, to: RopeDialect) -> Result<()> {
    if !vec.len().is_multiple_of(2) {
        bail!("RoPE vector length {} must be even", vec.len());
    }
    if from == to || vec.is_empty() {
        return Ok(());
    }
    let half = vec.len() / 2;
    let src = vec.to_vec();
    match (from, to) {
        (RopeDialect::Interleaved, RopeDialect::SplitHalf) => {
            for i in 0..half {
                vec[i] = src[2 * i];
                vec[half + i] = src[2 * i + 1];
            }
        }
        (RopeDialect::SplitHalf, RopeDialect::Interleaved) => {
            for i in 0..half {
                vec[2 * i] = src[i];
                vec[2 * i + 1] = src[half + i];
            }
        }
        _ => {}
    }
    Ok(())
}

/// Verifies the `FP32` 1-token Oracle check (`<= 1e-5` max abs diff for exact dialect transmutation).
pub fn verify_fp32_oracle_match(
    baseline_logits: &[f32],
    converted_logits: &[f32],
    tol: f32,
) -> Result<f32> {
    if baseline_logits.is_empty() || baseline_logits.len() != converted_logits.len() {
        bail!("logit slices must be non-empty and equal length");
    }
    let mut max_diff = 0.0f32;
    for (&a, &b) in baseline_logits.iter().zip(converted_logits.iter()) {
        let diff = (a - b).abs();
        if diff.is_nan() || diff > max_diff {
            max_diff = diff;
        }
    }
    if max_diff.is_nan() || max_diff > tol {
        bail!("FP32 1-token oracle mismatch: max_diff={max_diff:.6e} > tol={tol:.6e}");
    }
    Ok(max_diff)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rope_permutation_round_trip_passes_1e_5_oracle() -> Result<()> {
        let orig = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut work = orig;
        permute_rope_vector(&mut work, RopeDialect::Interleaved, RopeDialect::SplitHalf)?;
        assert_eq!(work, [1.0, 3.0, 5.0, 2.0, 4.0, 6.0]);
        permute_rope_vector(&mut work, RopeDialect::SplitHalf, RopeDialect::Interleaved)?;
        let diff = verify_fp32_oracle_match(&orig, &work, 1e-5)?;
        assert!(diff <= 1e-5);
        Ok(())
    }

    #[test]
    fn mha_to_gqa_pooling_reduces_kv_heads_and_preserves_exact_group_match() -> Result<()> {
        let h0 = [1.0f32, 2.0];
        let h1 = [3.0f32, 4.0];
        let h2 = [5.0f32, 6.0];
        let h3 = [5.0f32, 6.0];
        let lx = [1.0f32, 2.0];
        let pooled = mha_to_gqa_whitened_pool(&[&h0, &h1, &h2, &h3], &lx, 2)?;
        assert_eq!(pooled.len(), 2);
        assert_eq!(pooled[0], vec![2.0, 3.0]);
        assert_eq!(pooled[1], vec![5.0, 6.0]);
        Ok(())
    }

    #[test]
    fn dory_conversion_mandates_bitdistill_recovery_and_two_stage_gate() -> Result<()> {
        let plan = ConversionPlan::build(
            ModelArchKind::DenseGqa,
            ModelArchKind::DoryLooped,
            ConversionOp::DeepToDoryLoop,
        );
        assert!(plan.pre_quant_gate_at_f16_q8);
        assert!(plan.requires_bitdistill_recovery);

        let router = sinkhorn_balanced_router_init(&[vec![1.0, 2.0], vec![3.0, 1.0]], 5)?;
        assert_eq!(router.len(), 2);
        Ok(())
    }
}
