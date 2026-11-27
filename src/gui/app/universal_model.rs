// SPDX-License-Identifier: AGPL-3.0-only
//! Studio data model: layers, tensors, the universal model, audit rows and history, with the
//! synthetic-model constructor and loaders.

use serde::{Deserialize, Serialize};
use std::path::Path;

// Layer node and model representations
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LayerNode {
    pub layer_index: usize,
    pub layer_type: String, // "gated_deltanet", "phase_recurrence", "sliding_window", "mamba2", "gqa_attention"
    pub alpha_binary: f32,  // 0.0 to 1.0 (1-bit signed, XNOR)
    pub beta_ternary: f32,  // 0.0 to 1.0 (BitNet b1.58, 2-bit packed)
    pub gamma_phase: f32,   // 0.0 to 1.0 (Complex diagonal recurrence)
    pub reach_probability: f32,
    pub is_exit_checkpoint: bool,
    pub channel_width: usize,
    pub is_optimized: bool,
    pub benchmarked_latency_ms: f32,
    pub estimated_distortion: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TensorInfo {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: String,
    pub num_elements: usize,
    pub min_val: f32,
    pub max_val: f32,
    pub mean_val: f32,
    pub std_val: f32,
    pub sparsity: f32,
    pub outlier_ratio: f32,
    pub histogram: [f32; 32],
    pub sample_weights: Vec<f32>,
    pub layer_index: Option<usize>,
    pub role: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct UniversalModel {
    pub name: String,
    pub format: String,
    pub architecture: String,
    pub num_layers: usize,
    pub hidden_size: usize,
    pub total_parameters: u64,
    pub layers: Vec<LayerNode>,
    pub tensors: Vec<TensorInfo>,
    /// Original file-backed tensors, retained for exact source exports.
    #[serde(skip)]
    pub source_model: Option<crate::ir::ModelIR>,
    /// The model file or folder this was loaded from; the Chat tab needs it for the tokenizer.
    #[serde(skip)]
    pub source_path: Option<std::path::PathBuf>,
}

#[derive(Clone, Debug)]
pub struct BaselineMetrics {
    pub prompt: String,
    pub generated_text: String,
    pub tokens_generated: usize,
    pub ttft_ms: f32,
    pub time_per_token_ms: f32,
    pub tokens_per_sec: f32,
    pub kv_cache_mb: f32,
    pub energy_mj_per_token: f32,
    pub layer_latencies: Vec<f32>,
}

#[derive(Clone, Debug)]
pub struct Scorecard {
    pub baseline_tok_sec: f32,
    pub tuned_tok_sec: f32,
    pub speedup_factor: f32,
    pub baseline_storage_mb: f32,
    pub tuned_storage_mb: f32,
    pub compression_ratio: f32,
    pub baseline_kv_cache_mb: f32,
    pub tuned_kv_cache_mb: f32,
    pub cache_savings_percent: f32,
    pub baseline_energy_mj: f32,
    pub tuned_energy_mj: f32,
    pub energy_savings_percent: f32,
}

#[derive(Clone, Debug)]
pub struct AuditRow {
    pub layer: usize,
    pub is_checkpoint: bool,
    pub status: String,
    pub alpha: u32,
    pub beta: u32,
    pub gamma: u32,
    pub reach: f32,
    pub mixer: String,
    pub storage_mb: f32,
    pub traffic_mb: f32,
    pub latency_ms: f32,
    pub distortion: f32,
}

/// History Snapshot Checkpoint for restoring earlier configurations
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HistoryCheckpoint {
    pub id: usize,
    pub timestamp: String,
    pub label: String,
    pub model_name: String,
    pub storage_mb: f32,
    pub throughput_tok_s: f32,
    pub compression_ratio: f32,
    pub avg_distortion: f32,
    pub details: String,
    pub snapshot_model: UniversalModel,
}

/// Structured Audit Log Event
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AuditEvent {
    pub id: usize,
    pub timestamp: String,
    pub level: String, // "INFO", "OPTIMIZE", "SURGERY", "DISTILL", "BENCHMARK", "WARN"
    pub category: String,
    pub message: String,
    pub target: String,
}

#[derive(Clone, Debug)]
pub(super) struct LayerCandidate {
    pub(super) name: &'static str,
    pub(super) alpha: f32,
    pub(super) beta: f32,
    pub(super) gamma: f32,
    pub(super) mixer: &'static str,
}

impl UniversalModel {
    pub fn create_synthetic(num_layers: usize, hidden_size: usize) -> Self {
        let mut layers = Vec::new();
        let cp_idx = (num_layers as f32 * 0.30).round() as usize;

        for i in 0..num_layers {
            let is_cp = i == cp_idx.max(1);
            let layer_type = if i == 0 {
                "gqa_attention".to_string()
            } else if i % 2 == 1 {
                "gated_deltanet".to_string()
            } else {
                "phase_recurrence".to_string()
            };

            let (alpha, beta, gamma, reach) = if i < cp_idx {
                (0.10, 0.80, 0.10, 1.0)
            } else {
                (0.25, 0.45, 0.30, 0.20)
            };

            layers.push(LayerNode {
                layer_index: i,
                layer_type,
                alpha_binary: alpha,
                beta_ternary: beta,
                gamma_phase: gamma,
                reach_probability: reach,
                is_exit_checkpoint: is_cp,
                channel_width: hidden_size,
                is_optimized: false,
                benchmarked_latency_ms: 2.8,
                estimated_distortion: 0.005,
            });
        }

        let mut tensors = Vec::new();
        let default_wp = crate::ir::WeightProfile::default();
        let sample_vals = vec![
            0.024, -0.142, 0.001, 0.089, -0.053, 0.112, -0.007, 0.045, -0.088, 0.019, 0.134,
            -0.021, 0.000, 0.067, -0.115, 0.038, 0.071, -0.049, 0.012, -0.095, 0.028, 0.084,
            -0.003, 0.126, -0.108, 0.033, 0.055, -0.076, 0.000, 0.041, -0.032, 0.099, 0.018,
            -0.062, 0.145, -0.011, 0.083, -0.034, 0.091, -0.077, -0.025, 0.064, -0.019, 0.103,
            -0.041, 0.072, -0.085, 0.039, 0.054, -0.091, 0.022, 0.067, -0.043, 0.115, -0.008,
            0.031, -0.076, 0.048, 0.089, -0.031, 0.012, 0.075, -0.064, 0.027,
        ];

        for l in 0..num_layers {
            tensors.push(TensorInfo {
                name: f!("model.layers.{}.self_attn.q_proj.weight", l),
                shape: vec![hidden_size, hidden_size],
                dtype: "F16".to_string(),
                num_elements: hidden_size * hidden_size,
                min_val: -0.142,
                max_val: 0.138,
                mean_val: 0.0002,
                std_val: 0.024,
                sparsity: 0.03,
                outlier_ratio: 0.0025,
                histogram: default_wp.histogram,
                sample_weights: sample_vals.clone(),
                layer_index: Some(l),
                role: "AttentionQ".to_string(),
            });
            tensors.push(TensorInfo {
                name: f!("model.layers.{}.self_attn.k_proj.weight", l),
                shape: vec![hidden_size / 4, hidden_size],
                dtype: "F16".to_string(),
                num_elements: (hidden_size / 4) * hidden_size,
                min_val: -0.135,
                max_val: 0.141,
                mean_val: 0.0001,
                std_val: 0.022,
                sparsity: 0.04,
                outlier_ratio: 0.0021,
                histogram: default_wp.histogram,
                sample_weights: sample_vals.clone(),
                layer_index: Some(l),
                role: "AttentionK".to_string(),
            });
            tensors.push(TensorInfo {
                name: f!("model.layers.{}.self_attn.v_proj.weight", l),
                shape: vec![hidden_size / 4, hidden_size],
                dtype: "F16".to_string(),
                num_elements: (hidden_size / 4) * hidden_size,
                min_val: -0.128,
                max_val: 0.132,
                mean_val: -0.0002,
                std_val: 0.023,
                sparsity: 0.035,
                outlier_ratio: 0.0019,
                histogram: default_wp.histogram,
                sample_weights: sample_vals.clone(),
                layer_index: Some(l),
                role: "AttentionV".to_string(),
            });
            tensors.push(TensorInfo {
                name: f!("model.layers.{}.self_attn.o_proj.weight", l),
                shape: vec![hidden_size, hidden_size],
                dtype: "F16".to_string(),
                num_elements: hidden_size * hidden_size,
                min_val: -0.150,
                max_val: 0.148,
                mean_val: 0.0003,
                std_val: 0.026,
                sparsity: 0.03,
                outlier_ratio: 0.0028,
                histogram: default_wp.histogram,
                sample_weights: sample_vals.clone(),
                layer_index: Some(l),
                role: "AttentionO".to_string(),
            });
            tensors.push(TensorInfo {
                name: f!("model.layers.{}.mlp.gate_proj.weight", l),
                shape: vec![hidden_size * 4, hidden_size],
                dtype: "F16".to_string(),
                num_elements: hidden_size * hidden_size * 4,
                min_val: -0.198,
                max_val: 0.205,
                mean_val: 0.0001,
                std_val: 0.029,
                sparsity: 0.06,
                outlier_ratio: 0.0038,
                histogram: default_wp.histogram,
                sample_weights: sample_vals.clone(),
                layer_index: Some(l),
                role: "MlpGate".to_string(),
            });
            tensors.push(TensorInfo {
                name: f!("model.layers.{}.mlp.up_proj.weight", l),
                shape: vec![hidden_size * 4, hidden_size],
                dtype: "F16".to_string(),
                num_elements: hidden_size * hidden_size * 4,
                min_val: -0.205,
                max_val: 0.199,
                mean_val: -0.0001,
                std_val: 0.028,
                sparsity: 0.055,
                outlier_ratio: 0.0036,
                histogram: default_wp.histogram,
                sample_weights: sample_vals.clone(),
                layer_index: Some(l),
                role: "MlpUp".to_string(),
            });
            tensors.push(TensorInfo {
                name: f!("model.layers.{}.mlp.down_proj.weight", l),
                shape: vec![hidden_size, hidden_size * 4],
                dtype: "F16".to_string(),
                num_elements: hidden_size * hidden_size * 4,
                min_val: -0.215,
                max_val: 0.208,
                mean_val: -0.0001,
                std_val: 0.031,
                sparsity: 0.05,
                outlier_ratio: 0.0041,
                histogram: default_wp.histogram,
                sample_weights: sample_vals.clone(),
                layer_index: Some(l),
                role: "MlpDown".to_string(),
            });
            tensors.push(TensorInfo {
                name: f!("model.layers.{}.input_layernorm.weight", l),
                shape: vec![hidden_size],
                dtype: "F16".to_string(),
                num_elements: hidden_size,
                min_val: 0.85,
                max_val: 1.15,
                mean_val: 1.002,
                std_val: 0.045,
                sparsity: 0.0,
                outlier_ratio: 0.0005,
                histogram: default_wp.histogram,
                sample_weights: sample_vals[0..8].to_vec(),
                layer_index: Some(l),
                role: "LayerNorm".to_string(),
            });
        }

        let total_parameters =
            (num_layers as u64) * (hidden_size as u64) * (hidden_size as u64) * 12;

        Self {
            name: "thme_12l_model".to_string(),
            format: "SafeTensors".to_string(),
            architecture: "THME-Hybrid".to_string(),
            num_layers,
            hidden_size,
            total_parameters,
            layers,
            tensors,
            source_model: None,
            source_path: None,
        }
    }

    pub fn load_universal(path: &Path) -> Result<Self, String> {
        let model_ir = crate::parser::ModelImporter::import(path)
            .map_err(|e| format!("Failed to import model: {}", e))?;

        let fmt_name = if path
            .extension()
            .is_some_and(|ext| ext.eq_ignore_ascii_case("gguf"))
        {
            "GGUF".to_string()
        } else if path
            .extension()
            .is_some_and(|ext| ext.eq_ignore_ascii_case("epiph"))
        {
            "Epiphanes".to_string()
        } else {
            "SafeTensors".to_string()
        };

        let mut model = Self::from_model_ir(model_ir, &fmt_name);
        model.source_path = Some(path.to_path_buf());
        Ok(model)
    }

    pub fn from_model_ir(model_ir: crate::ir::ModelIR, format_name: &str) -> Self {
        let source_model = (!model_ir.tensors.is_empty()
            && model_ir
                .tensors
                .iter()
                .all(|tensor| tensor.source.is_some()))
        .then(|| model_ir.clone());
        let num_layers = model_ir.num_layers.max(1);
        let cp_idx = (num_layers as f32 * 0.30).round() as usize;

        let mut layers = Vec::new();
        for i in 0..num_layers {
            let is_cp = i == cp_idx.max(1);
            let layer_type = if i == 0 {
                "gqa_attention".to_string()
            } else if i % 2 == 1 {
                "gated_deltanet".to_string()
            } else {
                "phase_recurrence".to_string()
            };

            layers.push(LayerNode {
                layer_index: i,
                layer_type,
                alpha_binary: 0.0,
                beta_ternary: 1.0,
                gamma_phase: 0.0,
                reach_probability: if i < cp_idx { 1.0 } else { 0.20 },
                is_exit_checkpoint: is_cp,
                channel_width: model_ir.hidden_size,
                is_optimized: false,
                benchmarked_latency_ms: 3.2,
                estimated_distortion: 0.002,
            });
        }

        let tensors = model_ir
            .tensors
            .into_iter()
            .map(|t| {
                let wp = t.weight_profile.unwrap_or_default();
                TensorInfo {
                    name: t.name,
                    shape: t.shape,
                    dtype: t.original_dtype,
                    num_elements: t.numel,
                    min_val: wp.min_val,
                    max_val: wp.max_val,
                    mean_val: wp.mean_val,
                    std_val: wp.std_val,
                    sparsity: wp.sparsity,
                    outlier_ratio: wp.outlier_ratio,
                    histogram: wp.histogram,
                    sample_weights: wp.sample_weights,
                    layer_index: t.layer_index,
                    role: format!("{:?}", t.role),
                }
            })
            .collect();

        Self {
            name: model_ir.name,
            format: format_name.to_string(),
            architecture: model_ir.architecture.display_name().to_string(),
            num_layers,
            hidden_size: model_ir.hidden_size,
            total_parameters: (model_ir.num_parameters as u64).max(10_000_000),
            layers,
            tensors,
            source_model,
            source_path: None,
        }
    }

    pub fn to_model_ir(&self) -> crate::ir::ModelIR {
        let tensors = self
            .tensors
            .iter()
            .map(|t| {
                let mut tir = crate::ir::TensorIR::new(
                    t.name.clone(),
                    t.shape.clone(),
                    t.dtype.clone(),
                    t.num_elements * 2,
                );
                tir.weight_profile = Some(crate::ir::WeightProfile {
                    min_val: t.min_val,
                    max_val: t.max_val,
                    mean_val: t.mean_val,
                    std_val: t.std_val,
                    sparsity: t.sparsity,
                    outlier_ratio: t.outlier_ratio,
                    histogram: t.histogram,
                    sample_weights: t.sample_weights.clone(),
                });
                tir
            })
            .collect();

        crate::ir::ModelIR::new(
            self.name.clone(),
            crate::ir::ModelArchitecture::from_str_name(&self.architecture),
            self.num_layers,
            self.hidden_size,
            self.hidden_size.checked_div(128).unwrap_or(16).max(1),
            self.hidden_size.checked_div(128 * 4).unwrap_or(4).max(1),
            32000,
            4096,
            tensors,
        )
    }

    pub fn compute_costs(&self) -> (f32, f32, f32, f32, f32) {
        let mut total_stored_mb = 0.0;
        let mut total_traffic_mb = 0.0;
        let mut depth_sum = 0.0;

        for layer in &self.layers {
            let bytes_pw = (layer.alpha_binary * 0.125)
                + (layer.beta_ternary * 0.25)
                + (layer.gamma_phase * 1.0);
            let l_mb = (layer.channel_width as f32 * layer.channel_width as f32 * bytes_pw * 3.0)
                / (1024.0 * 1024.0);
            let tr_mb = layer.reach_probability * l_mb;

            total_stored_mb += l_mb;
            total_traffic_mb += tr_mb;
            depth_sum += layer.reach_probability;
        }

        let n = self.layers.len().max(1) as f32;
        let f_bar = depth_sum / n;
        let joules = (total_traffic_mb / 1024.0) * 8.0 * 15.0;
        let throughput = 1000.0 / (f_bar * 38.0 + 10.0);

        (total_stored_mb, total_traffic_mb, f_bar, joules, throughput)
    }

    pub fn update_exit_routing(&mut self, exit_ratio_pct: f32, cp_layer_idx: usize) {
        let p_exit = (exit_ratio_pct / 100.0).clamp(0.05, 0.95);
        let tail_reach = 1.0 - p_exit;

        for (i, layer) in self.layers.iter_mut().enumerate() {
            if i < cp_layer_idx {
                layer.reach_probability = 1.0;
                layer.is_exit_checkpoint = false;
            } else if i == cp_layer_idx {
                layer.reach_probability = 1.0;
                layer.is_exit_checkpoint = true;
            } else {
                layer.reach_probability = tail_reach;
                layer.is_exit_checkpoint = false;
            }
        }
    }

    pub fn deconstruct_layer(&self, layer_idx: usize) -> crate::ir::LayerDeconstruction {
        let (mixer_type, reach, is_cp, a, b, g) = if let Some(l) = self.layers.get(layer_idx) {
            (
                l.layer_type.as_str(),
                l.reach_probability,
                l.is_exit_checkpoint,
                l.alpha_binary,
                l.beta_ternary,
                l.gamma_phase,
            )
        } else {
            ("gqa_attention", 1.0, false, 0.0, 1.0, 0.0)
        };

        let decon_tensors: Vec<crate::ir::DeconstructedTensor> = self
            .tensors
            .iter()
            .filter(|t| t.layer_index == Some(layer_idx))
            .map(|t| {
                let role = crate::ir::TensorRole::detect(&t.name);
                let format_name = if b >= 0.70 {
                    "BitNet b1.58"
                } else if a >= 0.35 {
                    "Binary XNOR"
                } else {
                    "FP16"
                };

                let destination = match role {
                    crate::ir::TensorRole::AttentionQ | crate::ir::TensorRole::AttentionK => {
                        "RoPE -> Scaled Dot-Product -> KV Cache"
                    }
                    crate::ir::TensorRole::AttentionV => "KV Cache Append -> Attention Context",
                    crate::ir::TensorRole::AttentionOut => "Attention Dense Out -> Residual Add 1",
                    crate::ir::TensorRole::MlpGate => "SwiGLU Gate Branch -> Multiply W_up",
                    crate::ir::TensorRole::MlpUp => "SwiGLU Linear Up -> Multiply W_gate",
                    crate::ir::TensorRole::MlpDown => "Down Projection -> Residual Add 2",
                    crate::ir::TensorRole::LayerNorm => "Variance Normalize -> Sub-Block Input",
                    _ => "Internal Forward Propagation",
                };

                crate::ir::DeconstructedTensor {
                    name: t.name.clone(),
                    role,
                    shape: t.shape.clone(),
                    dtype: t.dtype.clone(),
                    num_elements: t.num_elements,
                    byte_size: t.num_elements * 2,
                    is_active: reach > 0.05,
                    selected_format: format_name.to_string(),
                    effective_bits: if b >= 0.70 {
                        2.0
                    } else if a >= 0.35 {
                        1.0
                    } else {
                        16.0
                    },
                    routing_destination: destination.to_string(),
                    min_val: t.min_val,
                    max_val: t.max_val,
                    mean_val: t.mean_val,
                    std_val: t.std_val,
                    sparsity: t.sparsity,
                    outlier_ratio: t.outlier_ratio,
                }
            })
            .collect();

        crate::ir::LayerDeconstruction::analyze(
            layer_idx,
            mixer_type,
            reach,
            is_cp,
            a,
            b,
            g,
            &decon_tensors,
        )
    }
}
