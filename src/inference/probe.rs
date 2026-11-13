// SPDX-License-Identifier: AGPL-3.0-only
//! Forward-pass probe: what every layer and tensor actually did on real tokens.
//!
//! A trace is collected during a normal forward pass. For each linear tensor it records
//! the mean output magnitude per group of output channels, so a tensor of any size is
//! summarised by `ACTIVITY_BINS` numbers that can be drawn live. Layer-level figures show
//! how strongly each layer rewrites the residual stream, and anomalies (non-finite values,
//! exploding norms, dead MLP units) are flagged for the GUI.

use serde::Serialize;

/// Output-channel groups recorded per tensor.
pub const ACTIVITY_BINS: usize = 32;

/// Per-input-channel statistics of the activations entering one linear tensor, summed over
/// token positions so windows can be merged. Activation-aware quantization reads them:
/// mean |x| (AWQ-style scaling), max |x| (SmoothQuant-style migration), rms.
#[derive(Debug, Clone, Default, Serialize)]
pub struct InputChannelStats {
    /// Short role name of the tensor the activations feed: `q`, `k`, `v`, `o`, `gate`, `up`,
    /// `down`, `qkv`.
    pub name: &'static str,
    pub in_dim: usize,
    /// Token positions summed.
    pub tokens: usize,
    pub sum_abs: Vec<f64>,
    pub sum_sq: Vec<f64>,
    pub max_abs: Vec<f32>,
}

impl InputChannelStats {
    /// Statistics of a row-major `[positions, in_dim]` input.
    pub fn measure(name: &'static str, x: &[f32], in_dim: usize) -> Self {
        let in_dim = in_dim.max(1);
        let mut stats = Self {
            name,
            in_dim,
            tokens: 0,
            sum_abs: vec![0.0; in_dim],
            sum_sq: vec![0.0; in_dim],
            max_abs: vec![0.0; in_dim],
        };
        for row in x.chunks_exact(in_dim) {
            stats.tokens += 1;
            for (c, &v) in row.iter().enumerate() {
                let a = v.abs();
                stats.sum_abs[c] += f64::from(a);
                stats.sum_sq[c] += f64::from(v) * f64::from(v);
                if a > stats.max_abs[c] || a.is_nan() {
                    stats.max_abs[c] = a;
                }
            }
        }
        stats
    }

    /// Adds another window's statistics for the same tensor.
    pub fn merge(&mut self, other: &Self) {
        if self.in_dim != other.in_dim {
            return;
        }
        self.tokens += other.tokens;
        for c in 0..self.in_dim {
            self.sum_abs[c] += other.sum_abs[c];
            self.sum_sq[c] += other.sum_sq[c];
            self.max_abs[c] = self.max_abs[c].max(other.max_abs[c]);
        }
    }

    pub fn mean_abs(&self) -> Vec<f32> {
        let n = self.tokens.max(1) as f64;
        self.sum_abs.iter().map(|s| (s / n) as f32).collect()
    }

    pub fn rms(&self) -> Vec<f32> {
        let n = self.tokens.max(1) as f64;
        self.sum_sq.iter().map(|s| (s / n).sqrt() as f32).collect()
    }
}

/// Activity of one linear tensor during a forward pass.
#[derive(Debug, Clone, Default, Serialize)]
pub struct TensorActivity {
    /// Short role name: `q`, `k`, `v`, `o`, `gate`, `up`, `down`, `qkv`.
    pub name: &'static str,
    pub out_dim: usize,
    /// Mean |output| per contiguous group of output channels.
    pub bins: Vec<f32>,
    pub mean_abs: f32,
    pub max_abs: f32,
    /// Count of NaN or infinite outputs.
    pub nonfinite: usize,
}

impl TensorActivity {
    /// Summarises a row-major `[positions, out_dim]` output.
    pub fn measure(name: &'static str, y: &[f32], out_dim: usize) -> Self {
        let bins_n = ACTIVITY_BINS.min(out_dim.max(1));
        let mut sums = vec![0.0f64; bins_n];
        let mut counts = vec![0usize; bins_n];
        let (mut total, mut max_abs, mut nonfinite) = (0.0f64, 0.0f32, 0usize);
        for row in y.chunks_exact(out_dim.max(1)) {
            for (c, &v) in row.iter().enumerate() {
                if !v.is_finite() {
                    nonfinite += 1;
                    continue;
                }
                let a = v.abs();
                let bin = c * bins_n / out_dim;
                sums[bin] += f64::from(a);
                counts[bin] += 1;
                total += f64::from(a);
                max_abs = max_abs.max(a);
            }
        }
        let finite = y.len().saturating_sub(nonfinite).max(1);
        Self {
            name,
            out_dim,
            bins: sums
                .iter()
                .zip(&counts)
                .map(|(s, &n)| (s / n.max(1) as f64) as f32)
                .collect(),
            mean_abs: (total / finite as f64) as f32,
            max_abs,
            nonfinite,
        }
    }
}

/// What one applied layer did.
#[derive(Debug, Clone, Default, Serialize)]
pub struct LayerTrace {
    /// Index of the stored layer that was applied (repeats appear when a schedule loops).
    pub layer: usize,
    /// Root-mean-square of the residual stream entering the layer.
    pub resid_rms_in: f32,
    /// Root-mean-square of the residual stream leaving the layer.
    pub resid_rms_out: f32,
    /// rms(output − input) / rms(input): how strongly the layer rewrote the stream.
    pub update_ratio: f32,
    /// Share of MLP hidden units whose activation is near zero.
    pub mlp_dead_fraction: f32,
    pub tensors: Vec<TensorActivity>,
    /// Input-channel statistics per linear tensor, in the same order as `tensors`.
    pub inputs: Vec<InputChannelStats>,
}

/// Trace of one forward pass over a window of tokens.
#[derive(Debug, Clone, Default, Serialize)]
pub struct ForwardTrace {
    pub tokens: usize,
    pub layers: Vec<LayerTrace>,
}

impl ForwardTrace {
    /// Human-readable anomalies worth looking at.
    pub fn anomalies(&self) -> Vec<String> {
        let mut found = Vec::new();
        for (step, layer) in self.layers.iter().enumerate() {
            for tensor in &layer.tensors {
                if tensor.nonfinite > 0 {
                    found.push(format!(
                        "Step {step} (layer {}): {} produced {} non-finite values",
                        layer.layer, tensor.name, tensor.nonfinite
                    ));
                }
            }
            if !layer.resid_rms_out.is_finite() || layer.resid_rms_out > 1.0e4 {
                found.push(format!(
                    "Step {step} (layer {}): residual stream exploding (rms {:.3e})",
                    layer.layer, layer.resid_rms_out
                ));
            }
            if layer.mlp_dead_fraction > 0.95 {
                found.push(format!(
                    "Step {step} (layer {}): {:.0}% of MLP units inactive",
                    layer.layer,
                    100.0 * layer.mlp_dead_fraction
                ));
            }
        }
        found
    }
}

pub fn rms(x: &[f32]) -> f32 {
    if x.is_empty() {
        return 0.0;
    }
    (x.iter().map(|v| f64::from(*v) * f64::from(*v)).sum::<f64>() / x.len() as f64).sqrt() as f32
}

/// rms(after − before) / rms(before).
pub fn update_ratio(before: &[f32], after: &[f32]) -> f32 {
    let delta: f64 = before
        .iter()
        .zip(after)
        .map(|(a, b)| {
            let d = f64::from(b - a);
            d * d
        })
        .sum();
    let base = f64::from(rms(before)).max(1e-12);
    ((delta / before.len().max(1) as f64).sqrt() / base) as f32
}

/// Share of values below 1% of the tensor's own rms.
pub fn dead_fraction(hidden: &[f32]) -> f32 {
    let threshold = 0.01 * rms(hidden);
    let dead = hidden.iter().filter(|v| v.abs() <= threshold).count();
    dead as f32 / hidden.len().max(1) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn activity_bins_cover_channels_and_count_nonfinite() {
        // Two positions, 64 channels: channel c has magnitude c.
        let mut y: Vec<f32> = (0..64)
            .map(|c| c as f32)
            .chain((0..64).map(|c| -(c as f32)))
            .collect();
        y[5] = f32::NAN;
        let a = TensorActivity::measure("q", &y, 64);
        assert_eq!(a.bins.len(), ACTIVITY_BINS);
        assert_eq!(a.nonfinite, 1);
        assert!(a.bins[ACTIVITY_BINS - 1] > a.bins[0]);
        assert_eq!(a.max_abs, 63.0);
    }

    #[test]
    fn update_ratio_and_anomalies() {
        assert_eq!(update_ratio(&[1.0, 1.0], &[1.0, 1.0]), 0.0);
        assert!((update_ratio(&[1.0, 1.0], &[2.0, 2.0]) - 1.0).abs() < 1e-6);
        let trace = ForwardTrace {
            tokens: 4,
            layers: vec![LayerTrace {
                layer: 3,
                resid_rms_out: f32::INFINITY,
                mlp_dead_fraction: 0.99,
                ..Default::default()
            }],
        };
        assert_eq!(trace.anomalies().len(), 2);
    }
}
