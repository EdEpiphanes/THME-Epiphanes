// SPDX-License-Identifier: AGPL-3.0-only
//! GPTQ-style second-order rounding (roadmap v0.2.2.5, ticket THME-2205).
//!
//! Round-to-nearest quantizes each weight independently, so every rounding error propagates
//! through the remaining columns of the layer output. GPTQ compensates column by column:
//! given the calibration input second moments `H = X X^T`, quantizing column `j` with error
//! `e_j` lets the not-yet-quantized columns absorb the mistake through the upper Cholesky
//! factor `U` of `H^-1`: `W[:, j+1:] -= (e_j / U[j][j]) * U[j, j+1:]`. For block formats the
//! scale of each row's 32-value block is fitted at the group boundary from the already
//! compensated weights and then held fixed while the group's columns are rounded one by one
//! (reference: IST-DASLab/gptq `fasterquant`, Apache-2.0; algorithm re-implemented here).
//!
//! The grid arithmetic mirrors the exporter's `Q4_0` and `Q8_0` encoders exactly (checked by a
//! test that re-encodes the result). The rule is registered as
//! `thme.gptq_<format>` (e.g. `thme.gptq_q4_0`) and needs `thme evaluate --calibration`.

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};

use super::awq::ChannelStats;

/// Tensor metadata key holding a tensor's Hessian proxy (JSON `HessianProxy`), attached by
/// `thme evaluate --calibration` alongside the calibration statistics.
pub const HESSIAN_METADATA_KEY: &str = "hessian_proxy";

/// Input second moments for one tensor: `hessian[i][j]` is the mean of `x_i * x_j` over the
/// calibration tokens, scaled to the calibration context length. Only the lower triangle is
/// serialized; the full symmetric matrix is reconstructed on load.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HessianProxy {
    pub in_dim: usize,
    pub tokens: usize,
    /// Lower-triangular rows: row `i` holds `i + 1` entries.
    pub lower: Vec<Vec<f64>>,
}

impl HessianProxy {
    /// From the per-token products of an input matrix `[positions, in_dim]`, merged over
    /// windows. `tokens` is the number of token positions summed.
    pub fn measure(x: &[f32], in_dim: usize) -> Self {
        let in_dim = in_dim.max(1);
        let mut lower = vec![vec![0.0f64; 0]; in_dim];
        let mut tokens = 0usize;
        for row in x.chunks_exact(in_dim) {
            tokens += 1;
            for i in 0..in_dim {
                let xi = f64::from(row[i]);
                let cell = &mut lower[i];
                if cell.len() != i + 1 {
                    *cell = vec![0.0; i + 1];
                }
                for j in 0..=i {
                    cell[j] += xi * f64::from(row[j]);
                }
            }
        }
        Self {
            in_dim,
            tokens,
            lower,
        }
    }

    /// Merges another window's moments for the same tensor.
    pub fn merge(&mut self, other: &Self) {
        if self.in_dim != other.in_dim {
            return;
        }
        self.tokens += other.tokens;
        for (cell, row) in self.lower.iter_mut().zip(&other.lower) {
            for (a, b) in cell.iter_mut().zip(row) {
                *a += b;
            }
        }
    }

    /// Mean over calibration tokens; `i >= j`.
    pub fn at(&self, i: usize, j: usize) -> f64 {
        let (i, j) = if i >= j { (i, j) } else { (j, i) };
        let n = self.tokens.max(1) as f64;
        self.lower
            .get(i)
            .and_then(|row| row.get(j))
            .map(|v| v / n)
            .unwrap_or(0.0)
    }

    /// Full symmetric mean matrix `H = E[x x^T]`.
    pub fn full(&self) -> Vec<Vec<f64>> {
        let n = self.in_dim;
        let mut h = vec![vec![0.0f64; n]; n];
        for i in 0..n {
            for j in 0..=i {
                let v = self.at(i, j);
                h[i][j] = v;
                h[j][i] = v;
            }
        }
        h
    }
}

/// Values per block of the supported formats (`Q4_0` and `Q8_0` both use 32).
const GROUP: usize = 32;

/// Relative diagonal damping added to the Hessian before factoring (GPTQ's `percdamp`).
const PERCDAMP: f64 = 0.01;

/// Block formats whose scale is fitted from a group of 32 values and then fixed while
/// individual values are rounded onto the grid. K-quants (nested super-block scales) and
/// the other formats are not supported by this rounder yet.
#[derive(Clone, Copy)]
enum Grid {
    Q4_0,
    Q8_0,
}

impl Grid {
    fn parse(name: &str) -> Result<Self> {
        match name.to_ascii_uppercase().as_str() {
            "Q4_0" => Ok(Self::Q4_0),
            "Q8_0" => Ok(Self::Q8_0),
            other => bail!("GPTQ rounding supports Q4_0 and Q8_0, not {other}"),
        }
    }

    /// Fits the block scale from `block` exactly as the exporter's encoder does. Returns
    /// the unrounded scale (used for the codes) and the FP16 value that gets stored.
    fn fit(self, block: &[f32]) -> Result<(f32, f32)> {
        let scale = match self {
            Self::Q4_0 => {
                let extreme = block
                    .iter()
                    .fold(0.0f32, |m, &v| if v.abs() > m.abs() { v } else { m });
                extreme / -8.0
            }
            Self::Q8_0 => block.iter().fold(0.0f32, |m, v| m.max(v.abs())) / 127.0,
        };
        let half = crate::exporter::conversion::f32_to_f16(scale);
        let stored = crate::profiling::f16_to_f32(half);
        if !stored.is_finite() {
            bail!("block scale exceeds finite FP16 range");
        }
        if scale != 0.0 && half & 0x7fff == 0 {
            bail!("block scale underflows to zero in FP16");
        }
        Ok((scale, stored))
    }

    /// Rounds one value onto the grid of a fitted block (same codes as the encoder).
    fn round(self, x: f32, (scale, stored): (f32, f32)) -> f32 {
        if scale == 0.0 {
            return 0.0;
        }
        match self {
            Self::Q4_0 => {
                let code = ((x * (1.0 / scale) + 8.5) as i32).clamp(0, 15);
                (code - 8) as f32 * stored
            }
            Self::Q8_0 => (x / scale).round().clamp(-127.0, 127.0) * stored,
        }
    }
}

/// Lower Cholesky factor of a symmetric positive-definite matrix, or `None` when a pivot
/// is not positive.
fn cholesky(a: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    let n = a.len();
    let mut l = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        for j in 0..=i {
            let dot: f64 = (0..j).map(|k| l[i][k] * l[j][k]).sum();
            if i == j {
                let d = a[i][i] - dot;
                if d <= 0.0 || !d.is_finite() {
                    return None;
                }
                l[i][i] = d.sqrt();
            } else {
                l[i][j] = (a[i][j] - dot) / l[j][j];
            }
        }
    }
    Some(l)
}

/// Upper Cholesky factor `U` of `H^-1` (so `H^-1 = U^T U`), the matrix GPTQ propagates with.
/// Retries with ten times the damping when the Hessian is ill-conditioned.
/// shortcut: dense O(n^3) f64, fine for tests and small layers; use a blocked f32 version
/// before running it on 4096+ input widths.
fn inverse_upper_factor(h: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
    let n = h.len();
    let mean_diag = (0..n).map(|i| h[i][i]).sum::<f64>() / n.max(1) as f64;
    let mut damp = PERCDAMP;
    for _ in 0..6 {
        let mut a = h.to_vec();
        for (i, row) in a.iter_mut().enumerate() {
            row[i] += damp * mean_diag.max(1e-12);
        }
        if let Some(l) = cholesky(&a) {
            // Linv by forward substitution, then H^-1 = Linv^T Linv.
            let mut linv = vec![vec![0.0f64; n]; n];
            for i in 0..n {
                linv[i][i] = 1.0 / l[i][i];
                for j in 0..i {
                    let s: f64 = (j..i).map(|k| l[i][k] * linv[k][j]).sum();
                    linv[i][j] = -s / l[i][i];
                }
            }
            let mut inv = vec![vec![0.0f64; n]; n];
            for i in 0..n {
                for j in 0..=i {
                    let s: f64 = (i..n).map(|k| linv[k][i] * linv[k][j]).sum();
                    inv[i][j] = s;
                    inv[j][i] = s;
                }
            }
            if let Some(l2) = cholesky(&inv) {
                return Ok((0..n).map(|i| (0..n).map(|j| l2[j][i]).collect()).collect());
            }
        }
        damp *= 10.0;
    }
    bail!("Hessian proxy is not positive definite even with heavy damping")
}

/// Output error of a rounding: `sum over rows of d H d^T` with `d = original - rounded`,
/// the quantity GPTQ minimises (`||(W - Q) X||^2` up to the token count).
fn output_error(original: &[f32], rounded: &[f32], in_dim: usize, h: &[Vec<f64>]) -> f64 {
    original
        .chunks_exact(in_dim)
        .zip(rounded.chunks_exact(in_dim))
        .map(|(o, r)| {
            let d: Vec<f64> = o
                .iter()
                .zip(r)
                .map(|(a, b)| f64::from(*a) - f64::from(*b))
                .collect();
            (0..in_dim)
                .map(|i| d[i] * (0..in_dim).map(|j| h[i][j] * d[j]).sum::<f64>())
                .sum::<f64>()
        })
        .sum()
}

/// GPTQ rounding of `values` (row-major `[out][in_dim]`) onto the `format` grid, with the
/// reference algorithm (IST-DASLab `fasterquant`): columns are rounded left to right; at
/// every 32-column group boundary each row's block scale is fitted from the already
/// compensated weights; each rounding error, divided by the inverse-Hessian Cholesky
/// diagonal, is propagated into the remaining columns. `compensation` in `[0, 1]` scales
/// the propagation (1 is full GPTQ, 0 is plain per-block rounding). Returns the output error
/// `sum d H d^T` of the result.
pub fn second_order_round(
    values: &mut [f32],
    in_dim: usize,
    stats: &ChannelStats,
    hessian: &HessianProxy,
    format: &str,
    compensation: f32,
) -> Result<f64> {
    let grid = Grid::parse(format)?;
    if in_dim == 0 || !in_dim.is_multiple_of(GROUP) || !values.len().is_multiple_of(in_dim) {
        bail!(
            "Weights of length {} are not whole rows of {in_dim} inputs (a multiple of {GROUP})",
            values.len()
        );
    }
    if stats.mean_abs.len() != in_dim || stats.rms.len() != in_dim {
        bail!(
            "Calibration statistics cover {} channels, the tensor has {in_dim} inputs",
            stats.mean_abs.len()
        );
    }
    if hessian.in_dim != in_dim {
        bail!(
            "Hessian proxy covers {} channels, the tensor has {in_dim} inputs",
            hessian.in_dim
        );
    }
    let strength = f64::from(compensation.clamp(0.0, 1.0));
    let mut h = hessian.full();
    let original = values.to_vec();
    let mut w: Vec<Vec<f64>> = original
        .chunks_exact(in_dim)
        .map(|row| row.iter().map(|&v| f64::from(v)).collect())
        .collect();
    // Dead input channels (never active in calibration) carry no information: GPTQ zeroes
    // their weights and gives them a unit diagonal so the factorisation stays defined.
    for i in 0..in_dim {
        if h[i][i] == 0.0 {
            h[i][i] = 1.0;
            for row in w.iter_mut() {
                row[i] = 0.0;
            }
        }
    }
    let u = inverse_upper_factor(&h).context("factoring the Hessian proxy")?;
    let mut fitted: Vec<(f32, f32)> = vec![(0.0, 0.0); w.len()];
    for i in 0..in_dim {
        if i % GROUP == 0 {
            for (slot, row) in fitted.iter_mut().zip(&w) {
                let block: Vec<f32> = row[i..i + GROUP].iter().map(|&v| v as f32).collect();
                *slot = grid.fit(&block).context("fitting a block scale")?;
            }
        }
        let d = u[i][i].max(1e-12);
        for (row, params) in w.iter_mut().zip(&fitted) {
            let q = f64::from(grid.round(row[i] as f32, *params));
            let err = (row[i] - q) / d * strength;
            row[i] = q;
            for j in i + 1..in_dim {
                row[j] -= err * u[i][j];
            }
        }
    }
    let rounded: Vec<f32> = w.iter().flatten().map(|&v| v as f32).collect();
    let error = output_error(&original, &rounded, in_dim, &hessian.full());
    values.copy_from_slice(&rounded);
    Ok(error)
}

/// Fast O(numel) diagonal-Hessian fallback (`H_ii = rms_i^2`) when full `X X^T` second
/// moments are unavailable. Fits each 32-weight block's scale to minimize activation-weighted
/// squared error `sum rms_j^2 (w_j - Q(w_j, s))^2`.
pub fn second_order_round_diagonal(
    values: &mut [f32],
    in_dim: usize,
    stats: &ChannelStats,
    format: &str,
) -> Result<f64> {
    let grid = Grid::parse(format)?;
    if in_dim == 0 || !in_dim.is_multiple_of(GROUP) || !values.len().is_multiple_of(in_dim) {
        bail!(
            "Weights of length {} are not whole rows of {in_dim} inputs (a multiple of {GROUP})",
            values.len()
        );
    }
    if stats.rms.len() != in_dim {
        bail!(
            "Calibration statistics cover {} channels, the tensor has {in_dim} inputs",
            stats.rms.len()
        );
    }
    const ALPHAS: [f32; 5] = [0.85, 0.90, 0.94, 0.97, 1.00];
    let mut total_err = 0.0f64;
    for row in values.chunks_exact_mut(in_dim) {
        for (g_idx, block) in row.chunks_exact_mut(GROUP).enumerate() {
            let base_col = g_idx * GROUP;
            let (base_scale, base_stored) = grid.fit(block)?;
            if base_scale == 0.0 {
                continue;
            }
            let mut best_params = (base_scale, base_stored);
            let mut best_werr = f64::INFINITY;
            for &alpha in &ALPHAS {
                let s = base_scale * alpha;
                let half = crate::exporter::conversion::f32_to_f16(s);
                let stored = crate::profiling::f16_to_f32(half);
                if !stored.is_finite() || (s != 0.0 && half & 0x7fff == 0) {
                    continue;
                }
                let params = (s, stored);
                let mut werr = 0.0f64;
                for (k, &v) in block.iter().enumerate() {
                    let q = grid.round(v, params);
                    let d = f64::from(v) - f64::from(q);
                    let r = f64::from(stats.rms[base_col + k]);
                    werr += r * r * d * d;
                }
                if werr < best_werr {
                    best_werr = werr;
                    best_params = params;
                }
            }
            for v in block.iter_mut() {
                *v = grid.round(*v, best_params);
            }
            total_err += best_werr;
        }
    }
    Ok(total_err)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::fake_quantize;
    use crate::TensorFormat;

    /// 4 rows x 64 inputs with one salient channel, as in the AWQ tests.
    fn salient() -> (Vec<f32>, ChannelStats, HessianProxy) {
        let in_dim = 64;
        let mut state = 9u64;
        let mut inputs: Vec<f32> = Vec::new();
        let weights: Vec<f32> = (0..4 * in_dim)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((state >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.2
            })
            .collect();
        // Fake input matrix with channel 3 50x larger, then the Hessian from it.
        let positions = 128usize;
        for _p in 0..positions {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            for c in 0..in_dim {
                let base = ((state >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.1;
                inputs.push(if c == 3 { base * 50.0 } else { base });
            }
        }
        let mut mean_abs = vec![0.02f32; in_dim];
        mean_abs[3] = 1.0;
        let rms = mean_abs.iter().map(|m| m * 1.25).collect();
        let stats = ChannelStats { mean_abs, rms };
        let hessian = HessianProxy::measure(&inputs, in_dim);
        (weights, stats, hessian)
    }

    #[test]
    fn result_is_exactly_representable_in_the_format() {
        let (weights, stats, hessian) = salient();
        for (name, format) in [("Q4_0", TensorFormat::Q4_0), ("Q8_0", TensorFormat::Q8_0)] {
            let mut gptq = weights.clone();
            second_order_round(&mut gptq, 64, &stats, &hessian, name, 1.0).unwrap();
            // Re-encoding with the real exporter codec must change nothing.
            let mut again = gptq.clone();
            fake_quantize(&mut again, &format).unwrap();
            assert_eq!(gptq, again, "{name} output is not on the format grid");
        }
    }

    #[test]
    fn second_order_rounding_beats_plain_rounding_on_output_error() {
        let (weights, stats, hessian) = salient();
        let mut plain = weights.clone();
        fake_quantize(&mut plain, &TensorFormat::Q4_0).unwrap();
        let e_plain = output_error(&weights, &plain, 64, &hessian.full());
        let mut gptq = weights.clone();
        let error = second_order_round(&mut gptq, 64, &stats, &hessian, "Q4_0", 1.0).unwrap();
        assert!(
            error < e_plain,
            "GPTQ output error {error} is not below plain rounding {e_plain}"
        );
    }

    #[test]
    fn zero_compensation_is_plain_block_rounding() {
        let (weights, stats, hessian) = salient();
        let mut plain = weights.clone();
        fake_quantize(&mut plain, &TensorFormat::Q4_0).unwrap();
        let mut none = weights.clone();
        second_order_round(&mut none, 64, &stats, &hessian, "Q4_0", 0.0).unwrap();
        assert_eq!(none, plain);
    }

    #[test]
    fn hessian_proxy_measures_merges_and_reconstructs() {
        // Rows (1, -2) and (3, 4): sum x x^T = [[10, 10], [10, 20]], mean over 2 tokens.
        let x = [1.0f32, -2.0, 3.0, 4.0];
        let h = HessianProxy::measure(&x, 2);
        assert_eq!(h.tokens, 2);
        assert!((h.at(0, 0) - 5.0).abs() < 1e-9);
        assert!((h.at(1, 0) - 5.0).abs() < 1e-9);
        assert!((h.at(0, 1) - 5.0).abs() < 1e-9, "symmetric access");
        assert!((h.at(1, 1) - 10.0).abs() < 1e-9);
        // Rows (-1, 2) and (0, 1) add [[1, -2], [-2, 5]]; 4 tokens in all.
        let mut m = h.clone();
        m.merge(&HessianProxy::measure(&[-1.0, 2.0, 0.0, 1.0], 2));
        assert_eq!(m.tokens, 4);
        assert!((m.at(0, 0) - 2.75).abs() < 1e-9);
        assert!((m.at(1, 0) - 2.0).abs() < 1e-9);
        assert!((m.at(1, 1) - 6.25).abs() < 1e-9);
        assert_eq!(m.full().len(), 2);
        assert!((m.full()[1][0] - m.full()[0][1]).abs() < 1e-12);
    }

    #[test]
    fn bad_inputs_are_errors() {
        let (mut weights, stats, hessian) = salient();
        // Not whole rows of 64.
        assert!(
            second_order_round(&mut weights[..250], 64, &stats, &hessian, "Q4_0", 1.0).is_err()
        );
        // Row width not a multiple of the 32-value group.
        assert!(second_order_round(&mut weights, 60, &stats, &hessian, "Q4_0", 1.0).is_err());
        // Unsupported format.
        assert!(second_order_round(&mut weights, 64, &stats, &hessian, "Q4_K", 1.0).is_err());
        // Statistics for the wrong width.
        let mut short = stats.clone();
        short.mean_abs.truncate(63);
        assert!(second_order_round(&mut weights, 64, &short, &hessian, "Q4_0", 1.0).is_err());
    }
}
