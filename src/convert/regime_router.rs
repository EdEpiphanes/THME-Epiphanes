// SPDX-License-Identifier: AGPL-3.0-only
//! §7.6 Automated Source-to-Heterogeneous Conversion & Structural Adjustment Engine.

use anyhow::{bail, Result};
use sha2::{Digest, Sha256};

/// Target intra-layer quantization regime codec.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RegimeCodec {
    Binary,
    Ternary,
    Phase,
    E8P,
    Q8_0,
    Q4_0,
}

impl RegimeCodec {
    /// Returns true if the codec satisfies the fragile-boundary floor (`>= E8P` / `Q8_0`).
    #[must_use]
    pub const fn satisfies_boundary_floor(self) -> bool {
        matches!(self, Self::E8P | Self::Q8_0)
    }
}

/// Structural role of a tensor within the model, used for boundary pinning and PRNG role tagging.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerRole {
    Embedding,
    BoundaryFirst,
    BoundaryLast,
    LmHead,
    HiddenMlpGateUp,
    HiddenMlpDown,
    HiddenAttn,
}

impl LayerRole {
    #[must_use]
    pub const fn role_tag(self) -> &'static str {
        match self {
            Self::Embedding => "embedding",
            Self::BoundaryFirst => "layer_0",
            Self::BoundaryLast => "layer_last",
            Self::LmHead => "lm_head",
            Self::HiddenMlpGateUp => "mlp_gate_up",
            Self::HiddenMlpDown => "mlp_down",
            Self::HiddenAttn => "hidden_attn",
        }
    }

    #[must_use]
    pub const fn is_boundary(self) -> bool {
        matches!(
            self,
            Self::Embedding | Self::BoundaryFirst | Self::BoundaryLast | Self::LmHead
        )
    }
}

/// Boundary floor policy for fragile layers (`Embedding`, L0, L_{L-1}, `lm_head`, or high J_eff).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BoundaryFloor {
    pub tau_boundary: f32,
    pub min_codec: RegimeCodec,
}

impl Default for BoundaryFloor {
    fn default() -> Self {
        Self {
            tau_boundary: 1.5,
            min_codec: RegimeCodec::Q8_0,
        }
    }
}

impl BoundaryFloor {
    /// Returns the pinned codec if `role` is a boundary layer or `j_eff > tau_boundary`.
    #[must_use]
    pub fn enforce(self, role: LayerRole, j_eff: f32) -> Option<RegimeCodec> {
        if role.is_boundary() || j_eff > self.tau_boundary {
            Some(self.min_codec)
        } else {
            None
        }
    }
}

/// Deterministic PRNG seeded via `ChaCha20(SHA256(model_hash || tensor_name || role_tag))`
/// so shared `Dory` layers receive independent sketches.
#[derive(Debug, Clone)]
pub struct DeterministicSketchPrng {
    key: [u32; 8],
    counter: u32,
    block: [u32; 16],
    word_idx: usize,
}

impl DeterministicSketchPrng {
    /// Creates a new deterministic ChaCha20 sketch PRNG from model hash, tensor name, and role tag.
    #[must_use]
    pub fn from_seed_parts(model_hash: &[u8], tensor_name: &str, role: LayerRole) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(model_hash);
        hasher.update(b"|");
        hasher.update(tensor_name.as_bytes());
        hasher.update(b"|");
        hasher.update(role.role_tag().as_bytes());
        let digest = hasher.finalize();

        let mut key = [0u32; 8];
        for (idx, chunk) in digest.as_chunks::<4>().0.iter().enumerate() {
            key[idx] = u32::from_le_bytes(*chunk);
        }
        Self {
            key,
            counter: 0,
            block: [0u32; 16],
            word_idx: 16,
        }
    }

    #[inline]
    fn quarter_round(state: &mut [u32; 16], a: usize, b: usize, c: usize, d: usize) {
        state[a] = state[a].wrapping_add(state[b]);
        state[d] ^= state[a];
        state[d] = state[d].rotate_left(16);
        state[c] = state[c].wrapping_add(state[d]);
        state[b] ^= state[c];
        state[b] = state[b].rotate_left(12);
        state[a] = state[a].wrapping_add(state[b]);
        state[d] ^= state[a];
        state[d] = state[d].rotate_left(8);
        state[c] = state[c].wrapping_add(state[d]);
        state[b] ^= state[c];
        state[b] = state[b].rotate_left(7);
    }

    fn refill(&mut self) {
        let mut s = [
            0x6170_7865,
            0x3320_646e,
            0x7962_2d32,
            0x6b20_6574,
            self.key[0],
            self.key[1],
            self.key[2],
            self.key[3],
            self.key[4],
            self.key[5],
            self.key[6],
            self.key[7],
            self.counter,
            0,
            0,
            0,
        ];
        let init = s;
        for _ in 0..10 {
            Self::quarter_round(&mut s, 0, 4, 8, 12);
            Self::quarter_round(&mut s, 1, 5, 9, 13);
            Self::quarter_round(&mut s, 2, 6, 10, 14);
            Self::quarter_round(&mut s, 3, 7, 11, 15);
            Self::quarter_round(&mut s, 0, 5, 10, 15);
            Self::quarter_round(&mut s, 1, 6, 11, 12);
            Self::quarter_round(&mut s, 2, 7, 8, 13);
            Self::quarter_round(&mut s, 3, 4, 9, 14);
        }
        for i in 0..16 {
            self.block[i] = s[i].wrapping_add(init[i]);
        }
        self.counter = self.counter.wrapping_add(1);
        self.word_idx = 0;
    }

    /// Returns the next deterministic `u32`.
    pub fn next_u32(&mut self) -> u32 {
        if self.word_idx >= 16 {
            self.refill();
        }
        let val = self.block[self.word_idx];
        self.word_idx += 1;
        val
    }

    /// Returns a deterministic Rademacher sign (`-1.0` or `+1.0`).
    pub fn rademacher_sign(&mut self) -> f32 {
        if (self.next_u32() & 1) == 0 {
            1.0
        } else {
            -1.0
        }
    }
}

/// Output-row slice aligned to 1,024-weight `BITCOS` super-blocks.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RowSlicePlan {
    pub start_row: usize,
    pub end_row: usize,
    pub weight_count: usize,
    pub fused_gate_up: bool,
}

/// Partitions a matrix `(out_rows, in_cols)` along output rows into slices whose
/// weight counts are multiples of 1,024 (when possible), preserving `BITCOS` alignment.
pub fn partition_row_blocks_1024(
    out_rows: usize,
    in_cols: usize,
    fused_gate_up: bool,
) -> Result<Vec<RowSlicePlan>> {
    if out_rows == 0 || in_cols == 0 {
        bail!("matrix dimensions must be non-zero, got ({out_rows}, {in_cols})");
    }
    let rows_per_block = if in_cols >= 1024 {
        1
    } else {
        1024_usize.div_ceil(in_cols)
    };
    let mut plans = Vec::new();
    let mut start = 0;
    while start < out_rows {
        let end = (start + rows_per_block).min(out_rows);
        plans.push(RowSlicePlan {
            start_row: start,
            end_row: end,
            weight_count: (end - start) * in_cols,
            fused_gate_up,
        });
        start = end;
    }
    Ok(plans)
}

/// 3-axis structural fingerprint for a non-boundary row slice.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StructuralFingerprint {
    pub tau_dead_star: f32,
    pub ternary_affinity: f32,
    pub radial_flatness: f32,
    pub circular_dispersion: f32,
    pub binary_residual_affinity: f32,
}

/// Solves `tau_dead*` via weighted-MSE alternating projection (never raw mean)
/// and returns `(tau_dead_star, ternary_dead_zone_affinity)`.
pub fn solve_weighted_mse_tau_dead(weights: &[f32], hessian_diag: &[f32]) -> Result<(f32, f32)> {
    if weights.is_empty() || weights.len() != hessian_diag.len() {
        bail!("weights and hessian_diag must be non-empty and equal length");
    }
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for (&w, &h) in weights.iter().zip(hessian_diag.iter()) {
        let hw = h.max(1e-12) as f64;
        num += hw * (w.abs() as f64);
        den += hw;
    }
    let mut scale = (num / den.max(1e-12)).max(1e-9);
    let mut tau = 0.5 * scale;

    for _ in 0..8 {
        let mut s_num = 0.0f64;
        let mut s_den = 0.0f64;
        for (&w, &h) in weights.iter().zip(hessian_diag.iter()) {
            let aw = w.abs() as f64;
            if aw > tau {
                let hw = h.max(1e-12) as f64;
                s_num += hw * aw;
                s_den += hw;
            }
        }
        if s_den > 1e-12 {
            scale = s_num / s_den;
            tau = 0.5 * scale;
        }
    }

    let dead_count = weights.iter().filter(|&&w| (w.abs() as f64) <= tau).count();
    let affinity = dead_count as f32 / weights.len() as f32;
    Ok((tau as f32, affinity))
}

impl StructuralFingerprint {
    /// Computes the 3-axis fingerprint over a weight slice and diagonal Hessian proxy.
    pub fn compute(weights: &[f32], hessian_diag: &[f32], eff_rank_ratio: f32) -> Result<Self> {
        let (tau_dead_star, ternary_affinity) = solve_weighted_mse_tau_dead(weights, hessian_diag)?;

        let eps_rad = 1e-6f64;
        let mut log_sum = 0.0f64;
        let mut arith_sum = 0.0f64;
        let mut cos_sum = 0.0f64;
        let mut sin_sum = 0.0f64;
        let mut pair_count = 0usize;

        for pair in weights.as_chunks::<2>().0 {
            let x = pair[0] as f64;
            let y = pair[1] as f64;
            let r = (x * x + y * y).sqrt();
            if r > eps_rad {
                log_sum += r.ln();
                arith_sum += r;
                cos_sum += x / r;
                sin_sum += y / r;
                pair_count += 1;
            }
        }

        let (radial_flatness, circular_dispersion) = if pair_count > 0 {
            let m = pair_count as f64;
            let gm = (log_sum / m).exp();
            let am = (arith_sum / m).max(eps_rad);
            let r_bar = ((cos_sum / m).powi(2) + (sin_sum / m).powi(2)).sqrt();
            (
                (gm / am).clamp(0.0, 1.0) as f32,
                (1.0 - r_bar).clamp(0.0, 1.0) as f32,
            )
        } else {
            (0.0, 1.0)
        };

        Ok(Self {
            tau_dead_star,
            ternary_affinity,
            radial_flatness,
            circular_dispersion,
            binary_residual_affinity: (1.0 - eff_rank_ratio.clamp(0.0, 1.0)).clamp(0.0, 1.0),
        })
    }
}

/// Dynamic per-layer regime mixture selected by minimizing distortion + `tau_split * (k_regimes - 1)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RegimeMixture {
    pub alpha_bin: f32,
    pub alpha_tern: f32,
    pub alpha_phase: f32,
    pub k_regimes: usize,
    pub objective: f32,
}

/// Dynamically decides whether a layer uses a homogeneous codec (`k_regimes = 1`)
/// or an intra-layer `(bin, tern, phase)` mixture via split-hysteresis `tau_split`.
pub fn solve_dynamic_regime_mix(
    slice_distortions: &[[f32; 3]],
    tau_split: f32,
) -> Result<RegimeMixture> {
    if slice_distortions.is_empty() {
        bail!("slice_distortions must not be empty");
    }
    let n = slice_distortions.len() as f32;

    // Evaluate k_regimes = 1 homogeneous baselines.
    let mut best_k1_idx = 0usize;
    let mut best_k1_obj = f32::INFINITY;
    for codec_idx in 0..3 {
        let sum_d: f32 = slice_distortions.iter().map(|d| d[codec_idx]).sum();
        if sum_d < best_k1_obj {
            best_k1_obj = sum_d;
            best_k1_idx = codec_idx;
        }
    }

    // Evaluate per-slice argmin (potentially multi-regime).
    let mut counts = [0usize; 3];
    let mut sum_min = 0.0f32;
    for d in slice_distortions {
        let mut best_i = 0usize;
        let mut best_v = d[0];
        for (i, &val) in d.iter().enumerate().skip(1) {
            if val < best_v {
                best_v = val;
                best_i = i;
            }
        }
        counts[best_i] += 1;
        sum_min += best_v;
    }
    let k_active = counts.iter().filter(|&&c| c > 0).count().max(1);
    let mix_obj = sum_min + tau_split.max(0.0) * (k_active - 1) as f32;

    if best_k1_obj <= mix_obj {
        let mut alphas = [0.0f32; 3];
        alphas[best_k1_idx] = 1.0;
        Ok(RegimeMixture {
            alpha_bin: alphas[0],
            alpha_tern: alphas[1],
            alpha_phase: alphas[2],
            k_regimes: 1,
            objective: best_k1_obj,
        })
    } else {
        Ok(RegimeMixture {
            alpha_bin: counts[0] as f32 / n,
            alpha_tern: counts[1] as f32 / n,
            alpha_phase: counts[2] as f32 / n,
            k_regimes: k_active,
            objective: mix_obj,
        })
    }
}

/// Computes `W_o`/`W_down` residual-stream variance trace alignment scale `sqrt(Tr(Cov_src)/Tr(Cov_q))`.
pub fn residual_variance_scale(source_cov_trace: f32, quant_cov_trace: f32) -> Result<f32> {
    if source_cov_trace <= 0.0 || quant_cov_trace <= 0.0 {
        bail!("covariance traces must be positive");
    }
    Ok((source_cov_trace / quant_cov_trace).sqrt().clamp(0.5, 2.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_chacha20_seed_differs_across_tensor_and_role() {
        let model_hash = b"thme-dory-test-hash";
        let mut p1 = DeterministicSketchPrng::from_seed_parts(
            model_hash,
            "blk.4.attn_q",
            LayerRole::HiddenAttn,
        );
        let mut p2 = DeterministicSketchPrng::from_seed_parts(
            model_hash,
            "blk.4.attn_q",
            LayerRole::HiddenAttn,
        );
        let mut p3 = DeterministicSketchPrng::from_seed_parts(
            model_hash,
            "blk.5.attn_q",
            LayerRole::HiddenAttn,
        );
        let mut p4 =
            DeterministicSketchPrng::from_seed_parts(model_hash, "blk.4.attn_q", LayerRole::LmHead);

        let seq1: Vec<u32> = (0..4).map(|_| p1.next_u32()).collect();
        let seq2: Vec<u32> = (0..4).map(|_| p2.next_u32()).collect();
        let seq3: Vec<u32> = (0..4).map(|_| p3.next_u32()).collect();
        let seq4: Vec<u32> = (0..4).map(|_| p4.next_u32()).collect();

        assert_eq!(seq1, seq2);
        assert_ne!(seq1, seq3);
        assert_ne!(seq1, seq4);
    }

    #[test]
    fn boundary_layers_pinned_and_row_blocks_aligned_to_1024() -> Result<()> {
        let floor = BoundaryFloor::default();
        assert_eq!(
            floor.enforce(LayerRole::LmHead, 0.2),
            Some(RegimeCodec::Q8_0)
        );
        assert_eq!(floor.enforce(LayerRole::HiddenAttn, 0.5), None);
        assert_eq!(
            floor.enforce(LayerRole::HiddenAttn, 2.0),
            Some(RegimeCodec::Q8_0)
        );

        let blocks = partition_row_blocks_1024(4, 2048, true)?;
        assert_eq!(blocks.len(), 4);
        assert!(blocks
            .iter()
            .all(|b| b.weight_count == 2048 && b.fused_gate_up));
        Ok(())
    }

    #[test]
    fn weighted_mse_tau_dead_and_hysteresis_prefers_homogeneous_when_gain_small() -> Result<()> {
        let weights = [0.02f32, -0.01, 0.9, -0.95, 0.0, 0.85];
        let hessian = [1.0f32; 6];
        let (tau, aff) = solve_weighted_mse_tau_dead(&weights, &hessian)?;
        assert!(tau > 0.3 && tau < 0.6);
        assert!((aff - 0.5).abs() < 1e-5);

        let slices = [[1.0f32, 0.95, 1.2], [0.95, 1.0, 1.2]];
        let hom = solve_dynamic_regime_mix(&slices, 0.5)?;
        assert_eq!(hom.k_regimes, 1);

        let mix = solve_dynamic_regime_mix(&slices, 0.01)?;
        assert_eq!(mix.k_regimes, 2);
        assert!((mix.alpha_bin - 0.5).abs() < 1e-5);
        assert!((mix.alpha_tern - 0.5).abs() < 1e-5);
        Ok(())
    }
}
