use crate::ir::{ModelIR, TensorFormat, TensorIR};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerSensitivityReport {
    pub layer_index: usize,
    pub average_sensitivity: f64,
    pub total_parameters: usize,
    pub total_bytes: usize,
    pub recommended_format: TensorFormat,
}

pub struct SensitivityProfiler;

impl SensitivityProfiler {
    /// Calculate architectural and topological sensitivity scores for all tensors in ModelIR.
    /// Incorporates:
    /// 1. Tensor role weight (Embedding/Head/Norm > Attn QK > Attn V/Out > MLP)
    /// 2. Layer depth curve (U-shaped: early and late layers are significantly more sensitive than middle layers)
    pub fn profile_model(model: &mut ModelIR) {
        let total_layers = model.num_layers.max(1);

        for tensor in &mut model.tensors {
            let role_weight = tensor.role.base_sensitivity_weight();

            // Depth modifier: U-shaped curve centered around middle layers
            let depth_factor = if let Some(l) = tensor.layer_index {
                let norm_pos = l as f64 / total_layers as f64;
                // Parabola: peaks at 0.0 and 1.0 (value 1.4), dips at 0.5 (value 0.8)
                0.8 + 2.4 * (norm_pos - 0.5).powi(2)
            } else {
                // Non-layer global tensors (embeddings, output heads, final norms)
                1.5
            };

            let raw_sensitivity = role_weight * depth_factor;
            // Normalize to a [0.1, 1.0] scale approximately
            tensor.sensitivity_score = (raw_sensitivity / 4.5).clamp(0.05, 1.0);
        }
    }

    /// Aggregate sensitivity by layer
    pub fn summarize_layers(model: &ModelIR) -> Vec<LayerSensitivityReport> {
        let mut layer_map: HashMap<usize, Vec<&TensorIR>> = HashMap::new();
        for t in &model.tensors {
            if let Some(l) = t.layer_index {
                layer_map.entry(l).or_default().push(t);
            }
        }

        let mut reports = Vec::new();
        for (l, tensors) in layer_map {
            let total_params: usize = tensors.iter().map(|t| t.numel).sum();
            let total_bytes: usize = tensors.iter().map(|t| t.byte_size).sum();
            let avg_sens = if !tensors.is_empty() {
                tensors.iter().map(|t| t.sensitivity_score).sum::<f64>() / tensors.len() as f64
            } else {
                0.5
            };

            // Recommendation rule based on sensitivity score:
            // High sensitivity (> 0.6) -> FP16 / INT8
            // Moderate sensitivity (0.35..0.6) -> Q4_0
            // Low sensitivity (< 0.35) -> Ternary / Bit1_5
            let recommended = if avg_sens > 0.6 {
                TensorFormat::INT8
            } else if avg_sens > 0.35 {
                TensorFormat::Q4_0
            } else {
                TensorFormat::Ternary
            };

            reports.push(LayerSensitivityReport {
                layer_index: l,
                average_sensitivity: avg_sens,
                total_parameters: total_params,
                total_bytes,
                recommended_format: recommended,
            });
        }

        reports.sort_by_key(|r| r.layer_index);
        reports
    }
}
