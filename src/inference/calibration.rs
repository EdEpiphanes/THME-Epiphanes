// SPDX-License-Identifier: AGPL-3.0-only
//! Calibration activations: per-input-channel statistics of every linear tensor, collected
//! with the forward-pass probe over calibration tokens (roadmap v0.2.2.3).
//!
//! Activation-aware quantization reads these files: mean |x| per input channel drives
//! AWQ-style scaling, max |x| drives SmoothQuant-style migration. Statistics are keyed by
//! the full tensor name, so a file can be applied to the model it was collected on.

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::Path;

use super::arch::CausalLm;
use super::probe::InputChannelStats;
use crate::ir::{ModelIR, TensorRole};

/// Statistics of the activations entering one linear tensor.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TensorCalibration {
    /// Full tensor name in the model (e.g. `model.layers.3.mlp.down_proj.weight`).
    pub name: String,
    pub layer: usize,
    /// Short role name (`q`, `k`, `v`, `o`, `gate`, `up`, `down`, `qkv`).
    pub role: String,
    pub in_dim: usize,
    pub tokens: usize,
    pub mean_abs: Vec<f32>,
    pub rms: Vec<f32>,
    pub max_abs: Vec<f32>,
}

/// A saved calibration run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CalibrationFile {
    pub thme_version: String,
    pub model_name: String,
    pub context_len: usize,
    /// Token positions the statistics cover (windows x context).
    pub tokens: usize,
    pub tensors: Vec<TensorCalibration>,
}

impl CalibrationFile {
    pub fn save(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("Failed to create {parent:?}"))?;
        }
        std::fs::write(path, serde_json::to_string_pretty(self)? + "\n")
            .with_context(|| format!("Failed to write {path:?}"))
    }

    pub fn load(path: &Path) -> Result<Self> {
        let text =
            std::fs::read_to_string(path).with_context(|| format!("Failed to read {path:?}"))?;
        serde_json::from_str(&text).with_context(|| format!("Failed to parse {path:?}"))
    }

    /// Statistics for a tensor by full name.
    pub fn get(&self, name: &str) -> Option<&TensorCalibration> {
        self.tensors.iter().find(|t| t.name == name)
    }
}

/// Tensor role for a probe short name.
fn role_of(short: &str) -> Option<TensorRole> {
    Some(match short {
        "q" => TensorRole::AttentionQ,
        "k" => TensorRole::AttentionK,
        "v" => TensorRole::AttentionV,
        "o" => TensorRole::AttentionOut,
        "qkv" => TensorRole::AttentionQkv,
        "gate" => TensorRole::MlpGate,
        "up" => TensorRole::MlpUp,
        "down" => TensorRole::MlpDown,
        _ => return None,
    })
}

/// Runs `model` over non-overlapping windows of `context_len` tokens (at most `max_tokens`)
/// and returns the merged input statistics of every linear tensor, named after `ir`.
pub fn collect_calibration(
    model: &dyn CausalLm,
    ir: &ModelIR,
    tokens: &[u32],
    context_len: usize,
    max_tokens: Option<usize>,
) -> Result<CalibrationFile> {
    if context_len < 2 {
        bail!("Calibration context must be at least 2 tokens");
    }
    let usable = max_tokens.map_or(tokens.len(), |m| m.min(tokens.len()));
    if usable < context_len {
        bail!("Calibration needs at least {context_len} tokens, found {usable}");
    }
    let schedule: Vec<usize> = (0..model.num_layers()).collect();
    let mut merged: BTreeMap<(usize, &'static str), InputChannelStats> = BTreeMap::new();
    let mut covered = 0usize;
    for window in tokens[..usable].chunks_exact(context_len) {
        let (_, trace) = model.forward_traced(window, &schedule)?;
        for layer in &trace.layers {
            for stats in &layer.inputs {
                merged
                    .entry((layer.layer, stats.name))
                    .and_modify(|m| m.merge(stats))
                    .or_insert_with(|| stats.clone());
            }
        }
        covered += window.len();
    }
    let mut tensors = Vec::with_capacity(merged.len());
    for ((layer, short), stats) in merged {
        let role = role_of(short).with_context(|| format!("Unknown probe tensor '{short}'"))?;
        let tensor = ir
            .tensors
            .iter()
            .find(|t| t.layer_index == Some(layer) && t.role == role)
            .with_context(|| format!("No tensor for layer {layer} role {short} in the model"))?;
        tensors.push(TensorCalibration {
            name: tensor.name.clone(),
            layer,
            role: short.to_string(),
            in_dim: stats.in_dim,
            tokens: stats.tokens,
            mean_abs: stats.mean_abs(),
            rms: stats.rms(),
            max_abs: stats.max_abs.clone(),
        });
    }
    Ok(CalibrationFile {
        thme_version: env!("CARGO_PKG_VERSION").to_string(),
        model_name: ir.name.clone(),
        context_len,
        tokens: covered,
        tensors,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::{read_token_file, ArchConfig, WeightPrecision};
    use crate::parser::ModelImporter;

    #[test]
    fn channel_statistics_match_hand_computation_and_merge() {
        let x = [1.0f32, -2.0, 3.0, 4.0];
        let a = InputChannelStats::measure("q", &x, 2);
        assert_eq!(a.tokens, 2);
        assert_eq!(a.mean_abs(), vec![2.0, 3.0]);
        assert_eq!(a.max_abs, vec![3.0, 4.0]);
        let rms = a.rms();
        assert!((rms[0] - 5.0f32.sqrt()).abs() < 1e-6 && (rms[1] - 10.0f32.sqrt()).abs() < 1e-6);
        let mut m = a.clone();
        m.merge(&InputChannelStats::measure("q", &[-7.0, 0.0], 2));
        assert_eq!(m.tokens, 3);
        assert_eq!(m.max_abs, vec![7.0, 4.0]);
        assert_eq!(m.mean_abs(), vec![11.0 / 3.0, 2.0]);
    }

    /// THME-2203 done-means: per-linear-tensor input statistics on calibration tokens, saved
    /// and reloaded, on the synthetic model.
    #[test]
    fn calibration_covers_every_linear_tensor_and_round_trips() {
        let dir = tempfile::tempdir().unwrap();
        let (model_dir, tokens_path) =
            crate::cli::evaluate::tests::synthetic_checkpoint(dir.path());
        let ir = ModelImporter::import(&model_dir).unwrap();
        let lm = ArchConfig::detect(&ir, &model_dir)
            .unwrap()
            .load(&ir, WeightPrecision::AsStored)
            .unwrap();
        let tokens = read_token_file(&tokens_path).unwrap();
        let file = collect_calibration(&*lm, &ir, &tokens, 32, Some(160)).unwrap();

        assert_eq!(file.tokens, 160);
        assert_eq!(file.tensors.len(), 2 * 7, "q k v o gate up down per layer");
        for t in &file.tensors {
            assert_eq!(t.tokens, 160, "{}", t.name);
            let expected_in = if t.role == "down" { 64 } else { 32 };
            assert_eq!(t.in_dim, expected_in, "{}", t.name);
            assert_eq!(t.mean_abs.len(), t.in_dim);
            assert!(t.mean_abs.iter().all(|v| v.is_finite() && *v >= 0.0));
            assert!(t
                .mean_abs
                .iter()
                .zip(&t.rms)
                .zip(&t.max_abs)
                .all(|((m, r), x)| m <= &(r + 1e-6) && r <= &(x + 1e-6)));
        }
        // q, k and v read the same normalized input.
        let q = file.get("model.layers.1.self_attn.q_proj.weight").unwrap();
        let v = file.get("model.layers.1.self_attn.v_proj.weight").unwrap();
        assert_eq!(q.mean_abs, v.mean_abs);

        let path = dir.path().join("calib").join("stats.json");
        file.save(&path).unwrap();
        assert_eq!(CalibrationFile::load(&path).unwrap(), file);
        assert!(collect_calibration(&*lm, &ir, &tokens, 32, Some(10)).is_err());
    }
}
