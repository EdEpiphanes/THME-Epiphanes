// SPDX-License-Identifier: AGPL-3.0-only
//! AWQ-style activation-aware quantization (roadmap v0.2.2.4).
//!
//! Input channels that carry large activations matter more to a layer's output than their
//! weights' size suggests. Scaling column j of `W` by `s_j = mean|x_j|^alpha` before
//! rounding gives those channels finer effective steps; dividing by `s_j` afterwards keeps
//! the stored layer equivalent. For each alpha in a grid the rounded weights are compared
//! with the originals under the activation-weighted error
//! `sum_o sum_j rms_j^2 (W_oj - W'_oj)^2` (the expected output error for inputs with
//! per-channel rms `rms_j`), and the best alpha is kept. alpha = 0 is plain rounding, so the
//! result is never worse than plain rounding under that error.
//!
//! The block format's rounding is passed in by the caller, so this module does not depend
//! on any codec; the inference path supplies THME's own encoders and decoders.

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};

use crate::ir::TensorFormat;

/// Tensor metadata key holding a tensor's calibration statistics (JSON `ChannelStats`),
/// attached by `thme evaluate --calibration`.
pub const CALIBRATION_METADATA_KEY: &str = "calibration_stats";

/// Per-input-channel activation statistics for one tensor.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChannelStats {
    pub mean_abs: Vec<f32>,
    pub rms: Vec<f32>,
}

/// Which per-channel scales an `ActivationAware` rule applies. `PermutedScales` is control A
/// (THME-2204): the layer's real scale vector keeps its values but is assigned to the input
/// channels by a seeded permutation. The activation-weighted error still uses the true
/// statistics.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum ScaleSource {
    #[default]
    Calibrated,
    PermutedScales {
        seed: u64,
    },
}

/// Copy of `stats` with `mean_abs` (the scale source) permuted across input channels. The
/// shuffle is seeded by `seed` mixed with the layer's own statistics, so every layer draws its
/// own permutation. `rms` is unchanged.
/// shortcut: the key is the seed and the statistics, not the tensor name, so two layers with
/// identical statistics draw the same permutation; pass the tensor name through `ApplyContext`
/// if that ever matters.
pub fn permute_scales(stats: &ChannelStats, seed: u64) -> ChannelStats {
    let mut state = stats
        .mean_abs
        .iter()
        .fold(seed, |h, m| h.rotate_left(13) ^ u64::from(m.to_bits()));
    let mut mean_abs = stats.mean_abs.clone();
    // Fisher-Yates over SplitMix64 outputs.
    for i in (1..mean_abs.len()).rev() {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        let j = (z % (i as u64 + 1)) as usize;
        mean_abs.swap(i, j);
    }
    ChannelStats {
        mean_abs,
        rms: stats.rms.clone(),
    }
}

/// The block format named by an `ActivationAware` rule.
pub fn parse_format(name: &str) -> Result<TensorFormat> {
    Ok(match name.to_ascii_uppercase().as_str() {
        "Q8_0" => TensorFormat::Q8_0,
        "Q4_0" => TensorFormat::Q4_0,
        "Q4_K" => TensorFormat::Q4_K,
        "Q5_K" => TensorFormat::Q5_K,
        "Q6_K" => TensorFormat::Q6_K,
        "TERNARY" | "TQ1_0" => TensorFormat::Ternary,
        other => bail!("Unsupported activation-aware format '{other}'"),
    })
}

/// Activation-weighted squared error between two `[out][in_dim]` weight matrices.
pub fn weighted_error(original: &[f32], rounded: &[f32], rms: &[f32]) -> f64 {
    let in_dim = rms.len().max(1);
    original
        .chunks_exact(in_dim)
        .zip(rounded.chunks_exact(in_dim))
        .map(|(a, b)| {
            a.iter()
                .zip(b)
                .zip(rms)
                .map(|((x, y), r)| f64::from(r * r) * f64::from(x - y).powi(2))
                .sum::<f64>()
        })
        .sum()
}

/// Replaces `values` (row-major `[out][in_dim]`) by their activation-aware rounding, where
/// `round` replaces a slice by what it becomes after storage in the target format; returns
/// the chosen alpha.
pub fn activation_aware_quantize(
    values: &mut [f32],
    in_dim: usize,
    stats: &ChannelStats,
    round: &dyn Fn(&mut [f32]) -> Result<()>,
    alpha_grid: &[f32],
) -> Result<f32> {
    if in_dim == 0 || !values.len().is_multiple_of(in_dim) {
        bail!(
            "Weights of length {} are not rows of {in_dim}",
            values.len()
        );
    }
    if stats.mean_abs.len() != in_dim || stats.rms.len() != in_dim {
        bail!(
            "Calibration statistics cover {} channels, the tensor has {in_dim} inputs",
            stats.mean_abs.len()
        );
    }
    if alpha_grid.is_empty() {
        bail!("Activation-aware quantization needs at least one alpha");
    }
    let original = values.to_vec();
    let mut best: Option<(f64, f32, Vec<f32>)> = None;
    for &alpha in alpha_grid {
        let mut scales: Vec<f32> = stats
            .mean_abs
            .iter()
            .map(|&m| m.max(1e-8).powf(alpha))
            .collect();
        // Normalize around 1 so the rounding scale of each block stays well conditioned.
        let (lo, hi) = scales.iter().fold((f32::INFINITY, 0.0f32), |(lo, hi), &s| {
            (lo.min(s), hi.max(s))
        });
        let mid = (lo * hi).sqrt();
        if mid.is_finite() && mid > 0.0 {
            scales.iter_mut().for_each(|s| *s /= mid);
        }
        let mut scaled: Vec<f32> = original
            .chunks_exact(in_dim)
            .flat_map(|row| row.iter().zip(&scales).map(|(w, s)| w * s))
            .collect();
        round(&mut scaled).context("rounding in the target format failed")?;
        for row in scaled.chunks_exact_mut(in_dim) {
            for (w, s) in row.iter_mut().zip(&scales) {
                *w /= s;
            }
        }
        let error = weighted_error(&original, &scaled, &stats.rms);
        if best.as_ref().is_none_or(|(e, _, _)| error < *e) {
            best = Some((error, alpha, scaled));
        }
    }
    let (_, alpha, rounded) = best.context("no alpha evaluated")?;
    values.copy_from_slice(&rounded);
    Ok(alpha)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::fake_quantize;

    fn rounder(format: TensorFormat) -> impl Fn(&mut [f32]) -> Result<()> {
        move |v: &mut [f32]| fake_quantize(v, &format)
    }

    /// 4 rows x 64 inputs; channel 3 carries activations 50x larger than the rest.
    fn salient() -> (Vec<f32>, ChannelStats) {
        let in_dim = 64;
        let mut state = 9u64;
        let weights: Vec<f32> = (0..4 * in_dim)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((state >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.2
            })
            .collect();
        let mut mean_abs = vec![0.02f32; in_dim];
        mean_abs[3] = 1.0;
        let rms = mean_abs.iter().map(|m| m * 1.25).collect();
        (weights, ChannelStats { mean_abs, rms })
    }

    #[test]
    fn awq_beats_plain_rounding_on_a_salient_channel_and_never_loses() {
        let (weights, stats) = salient();
        let grid: Vec<f32> = (0..=10).map(|k| k as f32 / 10.0).collect();
        for format in [TensorFormat::Q4_0, TensorFormat::Q8_0] {
            let mut plain = weights.clone();
            fake_quantize(&mut plain, &format).unwrap();
            let mut aware = weights.clone();
            let alpha =
                activation_aware_quantize(&mut aware, 64, &stats, &rounder(format.clone()), &grid)
                    .unwrap();
            let e_plain = weighted_error(&weights, &plain, &stats.rms);
            let e_aware = weighted_error(&weights, &aware, &stats.rms);
            assert!(e_aware <= e_plain, "{format:?}: {e_aware} > {e_plain}");
            if format == TensorFormat::Q4_0 {
                assert!(alpha > 0.0, "a salient channel should pick alpha > 0");
                assert!(e_aware < 0.8 * e_plain, "{e_aware} vs {e_plain}");
            }
        }
        // alpha = 0 alone is exactly plain rounding.
        let mut only_zero = weights.clone();
        activation_aware_quantize(
            &mut only_zero,
            64,
            &stats,
            &rounder(TensorFormat::Q4_0),
            &[0.0],
        )
        .unwrap();
        let mut plain = weights.clone();
        fake_quantize(&mut plain, &TensorFormat::Q4_0).unwrap();
        let close = only_zero
            .iter()
            .zip(&plain)
            .all(|(a, b)| (a - b).abs() < 1e-6);
        assert!(close);
    }

    #[test]
    fn permuted_scales_keep_the_multiset_and_move_channels() {
        let (_, stats) = salient();
        let mut moved = false;
        for seed in 1..=5u64 {
            let permuted = permute_scales(&stats, seed);
            assert_eq!(
                permuted,
                permute_scales(&stats, seed),
                "same seed, same shuffle"
            );
            let mut a = stats.mean_abs.clone();
            let mut b = permuted.mean_abs.clone();
            a.sort_by(f32::total_cmp);
            b.sort_by(f32::total_cmp);
            assert_eq!(a, b, "a permutation keeps the scale multiset");
            assert_eq!(
                permuted.rms, stats.rms,
                "errors still use the true activations"
            );
            moved |= permuted.mean_abs[3] != stats.mean_abs[3];
        }
        assert!(
            moved,
            "the salient channel should leave position 3 for some seed"
        );
        // The default source is the real statistics, so existing recipes keep their meaning.
        assert_eq!(ScaleSource::default(), ScaleSource::Calibrated);
    }

    #[test]
    fn bad_shapes_and_formats_are_errors() {
        let (mut weights, stats) = salient();
        assert!(activation_aware_quantize(
            &mut weights,
            60,
            &stats,
            &rounder(TensorFormat::Q4_0),
            &[0.5]
        )
        .is_err());
        assert!(activation_aware_quantize(
            &mut weights,
            64,
            &stats,
            &rounder(TensorFormat::Q4_0),
            &[]
        )
        .is_err());
        assert!(parse_format("q4_k").is_ok());
        assert!(parse_format("Q3_K").is_err());
    }
}
