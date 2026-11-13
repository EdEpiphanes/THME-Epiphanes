// SPDX-License-Identifier: AGPL-3.0-only
//! SmoothQuant-style scale migration (roadmap v0.2.2.7).
//!
//! Activation outliers make the inputs of a linear layer hard to quantize. Migration moves
//! part of that range into the weights: for each input channel j a scale
//!
//!   s_j = max|x_j|^alpha / max|W_:,j|^(1 - alpha)
//!
//! divides the preceding normalization (weight, and bias for LayerNorm) and multiplies
//! column j of every linear that reads the normalized stream. Because
//! W diag(s) diag(1/s) (gamma * x_hat + beta) = W (gamma * x_hat + beta), the model computes
//! the same function before quantization; afterwards both activations and weights have a
//! flatter per-channel range. Weights are row-major `[out][in]`.

use anyhow::{bail, Result};

use super::calibration::CalibrationFile;
use super::llama::{LlamaLayer, LlamaModel};
use super::neox::{NeoxLayer, NeoxModel};

/// Scales are kept within this range so a dead channel cannot blow a weight up.
const SCALE_LIMIT: f32 = 1e4;

/// Per-input-channel migration scales for the linears in `consumers` (all `[out][in_dim]`).
/// Channels with no activation or no weight keep scale 1.
pub fn smoothing_scales(
    act_max: &[f32],
    consumers: &[&[f32]],
    in_dim: usize,
    alpha: f32,
) -> Result<Vec<f32>> {
    if !(0.0..=1.0).contains(&alpha) {
        bail!("Migration strength alpha must be in [0, 1], got {alpha}");
    }
    if act_max.len() != in_dim {
        bail!(
            "Activation statistics cover {} channels, the layer has {in_dim}",
            act_max.len()
        );
    }
    let mut w_max = vec![0.0f32; in_dim];
    for weights in consumers {
        if !weights.len().is_multiple_of(in_dim) {
            bail!(
                "Consumer weight length {} is not a multiple of {in_dim}",
                weights.len()
            );
        }
        for row in weights.chunks_exact(in_dim) {
            for (m, &w) in w_max.iter_mut().zip(row) {
                *m = m.max(w.abs());
            }
        }
    }
    Ok(act_max
        .iter()
        .zip(&w_max)
        .map(|(&a, &w)| {
            if a > 0.0 && w > 0.0 && a.is_finite() && w.is_finite() {
                (a.powf(alpha) / w.powf(1.0 - alpha)).clamp(1.0 / SCALE_LIMIT, SCALE_LIMIT)
            } else {
                1.0
            }
        })
        .collect())
}

/// Folds `scales` into a normalization (weight and optional bias divided by s) and the
/// linears that read it (column j multiplied by s_j).
pub fn migrate(
    norm_weight: &mut [f32],
    norm_bias: Option<&mut [f32]>,
    consumers: &mut [&mut Vec<f32>],
    scales: &[f32],
) {
    let in_dim = scales.len();
    for (g, s) in norm_weight.iter_mut().zip(scales) {
        *g /= s;
    }
    if let Some(bias) = norm_bias {
        for (b, s) in bias.iter_mut().zip(scales) {
            *b /= s;
        }
    }
    for weights in consumers.iter_mut() {
        for row in weights.chunks_exact_mut(in_dim) {
            for (w, s) in row.iter_mut().zip(scales) {
                *w *= s;
            }
        }
    }
}

/// Activation maxima of the input channels of `role` in `layer`, from a calibration file.
pub fn calibration_maxima(
    calib: &CalibrationFile,
) -> impl Fn(usize, &str) -> Option<Vec<f32>> + '_ {
    move |layer, role| {
        calib
            .tensors
            .iter()
            .find(|t| t.layer == layer && t.role == role)
            .map(|t| t.max_abs.clone())
    }
}

impl LlamaModel {
    /// Migrates activation range into the attention (q, k, v) and MLP (gate, up) inputs of
    /// every layer that `act_max` has statistics for. Returns the number of groups migrated.
    pub fn smooth(
        &mut self,
        act_max: impl Fn(usize, &str) -> Option<Vec<f32>>,
        alpha: f32,
    ) -> Result<usize> {
        let d = self.config.hidden_size;
        let mut groups = 0;
        for (index, layer) in self.layers.iter_mut().enumerate() {
            let LlamaLayer {
                attn_norm,
                q,
                k,
                v,
                mlp_norm,
                gate,
                up,
                ..
            } = layer;
            if let Some(maxima) = act_max(index, "q") {
                let scales = smoothing_scales(&maxima, &[q, k, v], d, alpha)?;
                migrate(attn_norm, None, &mut [q, k, v], &scales);
                groups += 1;
            }
            if let Some(maxima) = act_max(index, "gate") {
                let scales = smoothing_scales(&maxima, &[gate, up], d, alpha)?;
                migrate(mlp_norm, None, &mut [gate, up], &scales);
                groups += 1;
            }
        }
        Ok(groups)
    }
}

impl NeoxModel {
    /// Migrates activation range into the fused QKV input (after LayerNorm 1) and the MLP up
    /// input (after LayerNorm 2), folding the scales into both LayerNorm weight and bias.
    pub fn smooth(
        &mut self,
        act_max: impl Fn(usize, &str) -> Option<Vec<f32>>,
        alpha: f32,
    ) -> Result<usize> {
        let d = self.config.hidden_size;
        let mut groups = 0;
        for (index, layer) in self.layers.iter_mut().enumerate() {
            let NeoxLayer {
                ln1_w,
                ln1_b,
                ln2_w,
                ln2_b,
                qkv_w,
                up_w,
                ..
            } = layer;
            if let Some(maxima) = act_max(index, "qkv") {
                let scales = smoothing_scales(&maxima, &[qkv_w], d, alpha)?;
                migrate(ln1_w, Some(ln1_b), &mut [qkv_w], &scales);
                groups += 1;
            }
            if let Some(maxima) = act_max(index, "up") {
                let scales = smoothing_scales(&maxima, &[up_w], d, alpha)?;
                migrate(ln2_w, Some(ln2_b), &mut [up_w], &scales);
                groups += 1;
            }
        }
        Ok(groups)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::{
        collect_calibration, read_token_file, ArchConfig, CausalLm, WeightPrecision,
    };
    use crate::parser::ModelImporter;

    fn close(a: &[f32], b: &[f32], tol: f32) -> bool {
        a.len() == b.len()
            && a.iter()
                .zip(b)
                .all(|(x, y)| (x - y).abs() <= tol * (1.0 + y.abs()))
    }

    #[test]
    fn scales_follow_the_formula_and_guard_dead_channels() {
        let w = [2.0f32, 1.0, 0.0, 4.0, 0.5, 0.0]; // [2 out][3 in]
        let s = smoothing_scales(&[9.0, 0.0, 4.0], &[&w], 3, 0.5).unwrap();
        assert!((s[0] - 9.0f32.sqrt() / 4.0f32.sqrt()).abs() < 1e-6);
        assert_eq!(s[1], 1.0); // no activation
        assert_eq!(s[2], 1.0); // no weight
        assert!(smoothing_scales(&[1.0, 1.0, 1.0], &[&w], 3, 1.5).is_err());
        assert!(smoothing_scales(&[1.0], &[&w], 3, 0.5).is_err());
    }

    /// THME-2207 done-means: migration is folded into the preceding norm and the model
    /// computes the same function before quantization (LLaMA, from a calibration file).
    #[test]
    fn llama_migration_preserves_outputs() {
        let dir = tempfile::tempdir().unwrap();
        let (model_dir, tokens_path) =
            crate::cli::evaluate::tests::synthetic_checkpoint(dir.path());
        let ir = ModelImporter::import(&model_dir).unwrap();
        let arch = ArchConfig::detect(&ir, &model_dir).unwrap();
        let ArchConfig::Llama(config) = &arch else {
            panic!("synthetic checkpoint is LLaMA-family")
        };
        let tokens = read_token_file(&tokens_path).unwrap();
        let original = LlamaModel::load(&ir, config.clone(), WeightPrecision::AsStored).unwrap();
        let calib = collect_calibration(&original, &ir, &tokens, 32, Some(96)).unwrap();
        let before = original.forward_nll(&tokens[..48]).unwrap();

        let mut smoothed =
            LlamaModel::load(&ir, config.clone(), WeightPrecision::AsStored).unwrap();
        let groups = smoothed.smooth(calibration_maxima(&calib), 0.5).unwrap();
        assert_eq!(groups, 4, "attention and MLP input groups in 2 layers");
        assert_ne!(
            smoothed.layers[0].q, original.layers[0].q,
            "weights were rescaled"
        );
        assert_ne!(smoothed.layers[0].attn_norm, original.layers[0].attn_norm);
        let after = smoothed.forward_nll(&tokens[..48]).unwrap();
        assert!(close(&after, &before, 1e-4), "{after:?} vs {before:?}");
    }

    /// Same check for GPT-NeoX, where the scales also fold into the LayerNorm bias.
    #[test]
    fn neox_migration_preserves_outputs() {
        let mut model = crate::inference::neox::tests::tiny(0.4);
        for layer in &mut model.layers {
            for (i, b) in layer.ln1_b.iter_mut().enumerate() {
                *b = 0.05 * i as f32;
            }
        }
        let tokens: Vec<u32> = (0..20).map(|i| (i * 5 % 24) as u32).collect();
        let schedule: Vec<usize> = (0..model.num_layers()).collect();
        let (_, trace) = model.forward_traced(&tokens, &schedule).unwrap();
        let maxima = |layer: usize, role: &str| {
            trace
                .layers
                .iter()
                .find(|l| l.layer == layer)
                .and_then(|l| l.inputs.iter().find(|s| s.name == role))
                .map(|s| s.max_abs.clone())
        };
        let before = model.forward_nll(&tokens).unwrap();
        assert_eq!(model.smooth(maxima, 0.6).unwrap(), 4);
        let after = model.forward_nll(&tokens).unwrap();
        assert!(close(&after, &before, 1e-4), "{after:?} vs {before:?}");
    }
}
