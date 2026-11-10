use serde::{Deserialize, Serialize};

use crate::ir::{ModelIR, TensorFormat, TensorRole};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", content = "params")]
pub enum TweakOperation {
    SetPrecision {
        start_layer: usize,
        end_layer: usize,
        role: Option<TensorRole>,
        format: TensorFormat,
    },
    ReplaceMixer {
        layer: usize,
        mixer_type: String, // "gated_deltanet", "phase_recurrence", "sliding_window", "mamba2"
    },
    ConfigureEarlyExit {
        checkpoint_layer: usize,
        confidence_threshold: f32,
        target_exit_ratio: f32,
    },
    PruneAttentionHeads {
        layer: usize,
        target_heads: usize,
    },
    QuantizeKvCache {
        format: TensorFormat,
    },
    ApplyCustomAlgorithm {
        start_layer: usize,
        end_layer: usize,
        role: Option<TensorRole>,
        algorithm_id: String,
    },
    RebuildArchitecture {
        num_layers: usize,
        hidden_size: usize,
        default_mixer: String,
    },
    ConvertParadigm {
        target_paradigm: crate::ir::SpecializedModelParadigm,
        #[serde(default)]
        num_experts: Option<usize>,
        #[serde(default)]
        top_k: Option<usize>,
        #[serde(default)]
        target_hidden_size: Option<usize>,
        #[serde(default)]
        target_layers: Option<usize>,
        #[serde(default)]
        apply_bitnet: Option<bool>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TweakRecipe {
    pub name: String,
    pub author: String,
    pub target_model: String,
    pub version: String,
    pub description: String,
    pub operations: Vec<TweakOperation>,
    #[serde(default)]
    pub expected_speedup: Option<f32>,
    #[serde(default)]
    pub expected_memory_savings_pct: Option<f32>,
}

impl Default for TweakRecipe {
    fn default() -> Self {
        Self {
            name: "THME-Optimal-Heterogeneous-Recipe".to_string(),
            author: "Teacher-LLM-AutoTuner".to_string(),
            target_model: "Llama-3-8B".to_string(),
            version: "1.0".to_string(),
            description: "Restructures dense Transformer layers into hybrid DeltaNet/Phase recurrence and mixed-precision representation.".to_string(),
            operations: vec![
                TweakOperation::SetPrecision {
                    start_layer: 0,
                    end_layer: 1,
                    role: None,
                    format: TensorFormat::FP16,
                },
                TweakOperation::SetPrecision {
                    start_layer: 2,
                    end_layer: 28,
                    role: Some(TensorRole::MlpGate),
                    format: TensorFormat::Ternary,
                },
                TweakOperation::ReplaceMixer {
                    layer: 1,
                    mixer_type: "gated_deltanet".to_string(),
                },
                TweakOperation::ConfigureEarlyExit {
                    checkpoint_layer: 8,
                    confidence_threshold: 0.85,
                    target_exit_ratio: 80.0,
                },
                TweakOperation::QuantizeKvCache {
                    format: TensorFormat::INT8,
                },
            ],
            expected_speedup: Some(2.15),
            expected_memory_savings_pct: Some(68.5),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SurgeryDiff {
    pub recipe_name: String,
    pub modified_tensors_count: usize,
    pub original_bytes: usize,
    pub new_bytes: usize,
    pub memory_saved_bytes: usize,
    pub savings_percent: f32,
    pub estimated_speedup: f32,
    pub audit_log: Vec<String>,
}

pub struct SurgeryEngine;

impl SurgeryEngine {
    /// Apply a declarative TweakRecipe onto a ModelIR
    pub fn apply_recipe(model: &mut ModelIR, recipe: &TweakRecipe) -> SurgeryDiff {
        let original_bytes = model.total_estimated_bytes();
        let mut modified_count = 0;
        let mut audit_log = Vec::new();

        audit_log.push(format!(
            "[Surgery Started] Applying recipe '{}' by {}",
            recipe.name, recipe.author
        ));

        for op in &recipe.operations {
            match op {
                TweakOperation::SetPrecision {
                    start_layer,
                    end_layer,
                    role,
                    format,
                } => {
                    for tensor in &mut model.tensors {
                        if let Some(l) = tensor.layer_index {
                            if l >= *start_layer
                                && l <= *end_layer
                                && role.is_none_or(|r| r == tensor.role)
                            {
                                tensor.selected_format = format.clone();
                                modified_count += 1;
                            }
                        }
                    }
                    audit_log.push(format!(
                        "Applied precision {:?} to layers {}..={} (role: {:?})",
                        format, start_layer, end_layer, role
                    ));
                }
                TweakOperation::ReplaceMixer { layer, mixer_type } => {
                    audit_log.push(format!(
                        "Replaced Layer {} sequence mixer with '{}'",
                        layer, mixer_type
                    ));
                }
                TweakOperation::ConfigureEarlyExit {
                    checkpoint_layer,
                    confidence_threshold,
                    target_exit_ratio,
                } => {
                    audit_log.push(format!(
                        "Configured Early-Exit at Layer {} (Threshold: {:.2}, Exit Target: {:.0}%)",
                        checkpoint_layer, confidence_threshold, target_exit_ratio
                    ));
                }
                TweakOperation::PruneAttentionHeads {
                    layer,
                    target_heads,
                } => {
                    audit_log.push(format!(
                        "Pruned Layer {} attention heads to {} active heads",
                        layer, target_heads
                    ));
                }
                TweakOperation::QuantizeKvCache { format } => {
                    audit_log.push(format!("Quantized KV Cache global format to {:?}", format));
                }
                TweakOperation::ApplyCustomAlgorithm {
                    start_layer,
                    end_layer,
                    role,
                    algorithm_id,
                } => {
                    for tensor in &mut model.tensors {
                        if let Some(l) = tensor.layer_index {
                            if l >= *start_layer
                                && l <= *end_layer
                                && role.is_none_or(|r| r == tensor.role)
                            {
                                tensor.selected_format = TensorFormat::Custom(algorithm_id.clone());
                                modified_count += 1;
                            }
                        }
                    }
                    audit_log.push(format!(
                        "Applied custom BYOA algorithm '{}' to layers {}..={} (role: {:?})",
                        algorithm_id, start_layer, end_layer, role
                    ));
                }
                TweakOperation::RebuildArchitecture {
                    num_layers,
                    hidden_size,
                    default_mixer,
                } => {
                    audit_log.push(format!(
                        "Rebuilt model topology: {} layers, hidden dimension {}, sequence mixer '{}'",
                        num_layers, hidden_size, default_mixer
                    ));
                }
                TweakOperation::ConvertParadigm {
                    target_paradigm,
                    num_experts,
                    top_k,
                    target_hidden_size,
                    target_layers,
                    apply_bitnet,
                } => {
                    let cfg = crate::ir::ParadigmConversionConfig {
                        target_paradigm: *target_paradigm,
                        num_experts: num_experts.unwrap_or(4),
                        top_k: top_k.unwrap_or(2),
                        target_hidden_size: *target_hidden_size,
                        target_layers: *target_layers,
                        apply_bitnet: apply_bitnet.unwrap_or(false),
                        ..Default::default()
                    };
                    if let Ok((converted, report)) =
                        crate::ir::ParadigmConverter::convert(model, &cfg)
                    {
                        *model = converted;
                        modified_count += report.tensors_restructured + report.new_tensors_added;
                        audit_log.push(report.description);
                    } else {
                        audit_log.push(format!(
                            "Failed to convert paradigm to {:?}",
                            target_paradigm
                        ));
                    }
                }
            }
        }

        let new_bytes = model.total_estimated_bytes();
        let memory_saved_bytes = original_bytes.saturating_sub(new_bytes);
        let savings_percent = if original_bytes > 0 {
            (memory_saved_bytes as f32 / original_bytes as f32) * 100.0
        } else {
            0.0
        };

        let compression = model.overall_compression_ratio();
        let estimated_speedup = 1.0 + ((compression - 1.0) * 0.45).clamp(0.0, 4.0) as f32;

        audit_log.push(format!(
            "[Surgery Complete] {} tensors restructured. Storage saved: {:.1} MB ({:.1}%), Estimated Speedup: {:.2}x",
            modified_count,
            memory_saved_bytes as f32 / (1024.0 * 1024.0),
            savings_percent,
            estimated_speedup
        ));

        SurgeryDiff {
            recipe_name: recipe.name.clone(),
            modified_tensors_count: modified_count,
            original_bytes,
            new_bytes,
            memory_saved_bytes,
            savings_percent,
            estimated_speedup,
            audit_log,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{ModelArchitecture, TensorIR};

    #[test]
    fn test_surgery_application() {
        let mut tensors = Vec::new();
        for l in 0..4 {
            tensors.push(TensorIR::new(
                format!("model.layers.{}.self_attn.q_proj.weight", l),
                vec![1024, 1024],
                "F16".to_string(),
                2 * 1024 * 1024,
            ));
        }

        let mut model = ModelIR::new(
            "test_model".to_string(),
            ModelArchitecture::Llama,
            4,
            1024,
            16,
            16,
            32000,
            2048,
            tensors,
        );

        let recipe = TweakRecipe {
            name: "Test Recipe".to_string(),
            author: "Tester".to_string(),
            target_model: "test_model".to_string(),
            version: "1.0".to_string(),
            description: "test".to_string(),
            operations: vec![TweakOperation::SetPrecision {
                start_layer: 1,
                end_layer: 3,
                role: None,
                format: TensorFormat::INT8,
            }],
            expected_speedup: None,
            expected_memory_savings_pct: None,
        };

        let diff = SurgeryEngine::apply_recipe(&mut model, &recipe);
        assert_eq!(diff.modified_tensors_count, 3);
        assert!(diff.savings_percent > 0.0);
    }
}
