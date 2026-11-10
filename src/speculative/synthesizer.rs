use serde::{Deserialize, Serialize};

use crate::ir::{ModelIR, TensorFormat, TensorRole, WeightProfile};

/// Strategy used to extract a lightweight draft model from a target model
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DraftExtractionStrategy {
    /// Extracts the first K layers (prefix sub-network)
    PrefixLayers,
    /// Samples layers at uniform intervals (e.g. every Nth layer)
    StridedLayers,
    /// Prefix extraction combined with low-rank dimension projection
    ProjectedManifold,
    /// Prefix extraction with BitNet b1.58 ternary discretization for edge/CPU inference
    TernaryBitNet,
}

impl DraftExtractionStrategy {
    pub fn display_name(&self) -> &'static str {
        match self {
            Self::PrefixLayers => "Prefix Layer Extraction",
            Self::StridedLayers => "Strided Layer Subsampling",
            Self::ProjectedManifold => "Projected Low-Rank Manifold",
            Self::TernaryBitNet => "Ternary BitNet b1.58 Prefix",
        }
    }
}

/// Configuration governing draft model synthesis from a parent target model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DraftExtractionConfig {
    /// Strategy to employ
    pub strategy: DraftExtractionStrategy,
    /// Number of layers in the draft model (e.g. 4 layers for a 16 or 32 layer model)
    pub num_layers: usize,
    /// Target hidden dimension if downscaling manifold (e.g. 1024 or 2048)
    pub target_hidden_size: Option<usize>,
    /// Whether to apply BitNet b1.58 ternary discretization to draft feed-forward layers
    pub apply_bitnet: bool,
    /// Layer step interval if using StridedLayers strategy
    pub stride_step: usize,
}

impl Default for DraftExtractionConfig {
    fn default() -> Self {
        Self {
            strategy: DraftExtractionStrategy::PrefixLayers,
            num_layers: 4,
            target_hidden_size: None,
            apply_bitnet: false,
            stride_step: 4,
        }
    }
}

/// Diagnostic report detailing the synthesized draft model metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DraftSynthesisReport {
    pub target_model_name: String,
    pub draft_model_name: String,
    pub target_layers: usize,
    pub draft_layers: usize,
    pub target_hidden_size: usize,
    pub draft_hidden_size: usize,
    pub target_params: usize,
    pub draft_params: usize,
    pub parameter_reduction_pct: f64,
    pub target_byte_size: usize,
    pub draft_byte_size: usize,
    pub memory_savings_pct: f64,
    /// Estimated relative execution time of draft compared to target: T_draft / T_target
    pub latency_cost_ratio: f64,
    pub description: String,
}

impl DraftSynthesisReport {
    pub fn format_summary(&self) -> String {
        format!(
            "Draft Model Synthesis Report: '{}' -> '{}'\n\
            - Layers: {} -> {} ({:.1}% reduction)\n\
            - Hidden Dim: {} -> {}\n\
            - Parameters: {} -> {} ({:.1}% reduction)\n\
            - Memory Footprint: {:.2} MB -> {:.2} MB ({:.1}% savings)\n\
            - Latency Cost Ratio (r = T_draft/T_target): {:.3}\n\
            - Strategy: {}",
            self.target_model_name,
            self.draft_model_name,
            self.target_layers,
            self.draft_layers,
            (1.0 - self.draft_layers as f64 / self.target_layers.max(1) as f64) * 100.0,
            self.target_hidden_size,
            self.draft_hidden_size,
            self.target_params,
            self.draft_params,
            self.parameter_reduction_pct,
            self.target_byte_size as f64 / (1024.0 * 1024.0),
            self.draft_byte_size as f64 / (1024.0 * 1024.0),
            self.memory_savings_pct,
            self.latency_cost_ratio,
            self.description
        )
    }
}

/// Pure Rust synthesizer extracting lightweight draft models from parent checkpoints
pub struct DraftSynthesizer;

impl DraftSynthesizer {
    /// Extracts a draft sub-model from an existing target model IR
    pub fn extract_draft_model(
        target: &ModelIR,
        config: &DraftExtractionConfig,
    ) -> Result<(ModelIR, DraftSynthesisReport), String> {
        if target.tensors.is_empty() {
            return Err("Target model contains no tensors to extract draft model from".to_string());
        }

        let draft_layers_count = config.num_layers.clamp(1, target.num_layers.max(1));
        let draft_hidden = config
            .target_hidden_size
            .unwrap_or(target.hidden_size)
            .min(target.hidden_size);

        // Determine which layers to include
        let selected_layer_indices: Vec<usize> = match config.strategy {
            DraftExtractionStrategy::PrefixLayers | DraftExtractionStrategy::TernaryBitNet => {
                (0..draft_layers_count).collect()
            }
            DraftExtractionStrategy::StridedLayers => {
                let step = config.stride_step.max(1);
                (0..target.num_layers)
                    .step_by(step)
                    .take(draft_layers_count)
                    .collect()
            }
            DraftExtractionStrategy::ProjectedManifold => (0..draft_layers_count).collect(),
        };

        let mut draft_tensors = Vec::new();

        // 1. Copy embeddings
        for t in &target.tensors {
            if matches!(t.role, TensorRole::Embedding) || t.name.contains("embed_tokens") {
                let mut emb = t.clone();
                if draft_hidden < target.hidden_size && emb.shape.len() >= 2 {
                    emb.shape[1] = draft_hidden;
                    emb.numel = emb.shape[0] * draft_hidden;
                    emb.byte_size = emb.numel * 2;
                }
                draft_tensors.push(emb);
            }
        }

        // 2. Extract selected layers and re-map layer indices to 0..draft_layers_count
        for (new_idx, &src_idx) in selected_layer_indices.iter().enumerate() {
            let layer_prefix = format!("layers.{}.", src_idx);
            let new_prefix = format!("layers.{}.", new_idx);

            for t in &target.tensors {
                if t.name.contains(&layer_prefix) || t.layer_index == Some(src_idx) {
                    let mut draft_t = t.clone();
                    draft_t.name = draft_t.name.replace(&layer_prefix, &new_prefix);
                    draft_t.layer_index = Some(new_idx);

                    // Downscale hidden size if projected
                    if draft_hidden < target.hidden_size && draft_t.shape.len() >= 2 {
                        let target_intermediate = (target.hidden_size * 8) / 3;
                        let mut new_shape = draft_t.shape.clone();
                        for dim in new_shape.iter_mut() {
                            if *dim == target.hidden_size {
                                *dim = draft_hidden;
                            } else if *dim == target_intermediate {
                                *dim = (draft_hidden * 8) / 3;
                            }
                        }
                        let numel: usize = new_shape.iter().product();
                        draft_t.shape = new_shape;
                        draft_t.numel = numel;
                        draft_t.byte_size = numel * 2;
                    }

                    // Apply BitNet b1.58 ternary discretization if requested
                    if (config.apply_bitnet
                        || config.strategy == DraftExtractionStrategy::TernaryBitNet)
                        && matches!(
                            draft_t.role,
                            TensorRole::MlpGate | TensorRole::MlpUp | TensorRole::MlpDown
                        )
                    {
                        draft_t.selected_format = TensorFormat::Ternary;
                        draft_t.byte_size = (draft_t.numel * 2) / 8 + 64; // ~2 bits per weight + scale overhead
                        draft_t.weight_profile = Some(WeightProfile {
                            sparsity: 0.50,
                            ..Default::default()
                        });
                    }

                    draft_tensors.push(draft_t);
                }
            }
        }

        // 3. Copy final norm and LM head
        for t in &target.tensors {
            if matches!(t.role, TensorRole::OutputHead | TensorRole::FinalNorm)
                || t.name.contains("lm_head")
                || t.name.contains("norm.weight")
            {
                let mut head = t.clone();
                if draft_hidden < target.hidden_size && head.shape.len() >= 2 {
                    if head.shape[1] == target.hidden_size {
                        head.shape[1] = draft_hidden;
                    } else if head.shape[0] == target.hidden_size {
                        head.shape[0] = draft_hidden;
                    }
                    head.numel = head.shape.iter().product();
                    head.byte_size = head.numel * 2;
                }
                draft_tensors.push(head);
            }
        }

        let draft_name = format!(
            "{}-draft-{}l-{}d",
            target.name, draft_layers_count, draft_hidden
        );

        let draft_model = ModelIR::new(
            draft_name.clone(),
            target.architecture.clone(),
            draft_layers_count,
            draft_hidden,
            target.num_attention_heads.clamp(1, 32),
            target.num_key_value_heads.clamp(1, 8),
            target.vocab_size,
            target.max_position_embeddings,
            draft_tensors,
        );

        let target_params = target.num_parameters.max(1);
        let draft_params = draft_model.num_parameters;
        let param_reduction = (1.0 - (draft_params as f64 / target_params as f64)) * 100.0;

        let target_bytes = target.original_byte_size.max(1);
        let draft_bytes = draft_model.original_byte_size;
        let mem_savings = (1.0 - (draft_bytes as f64 / target_bytes as f64)) * 100.0;

        // Empirical latency cost ratio: layers ratio scaled by hidden dimension quadratic factor
        let layer_ratio = draft_layers_count as f64 / target.num_layers.max(1) as f64;
        let dim_ratio = draft_hidden as f64 / target.hidden_size.max(1) as f64;
        let bitnet_speedup =
            if config.apply_bitnet || config.strategy == DraftExtractionStrategy::TernaryBitNet {
                0.65 // BitNet integer addition speedup
            } else {
                1.0
            };
        let latency_ratio =
            (layer_ratio * (dim_ratio * dim_ratio) * bitnet_speedup).clamp(0.04, 0.50);

        let report = DraftSynthesisReport {
            target_model_name: target.name.clone(),
            draft_model_name: draft_name,
            target_layers: target.num_layers,
            draft_layers: draft_layers_count,
            target_hidden_size: target.hidden_size,
            draft_hidden_size: draft_hidden,
            target_params,
            draft_params,
            parameter_reduction_pct: param_reduction,
            target_byte_size: target_bytes,
            draft_byte_size: draft_bytes,
            memory_savings_pct: mem_savings,
            latency_cost_ratio: latency_ratio,
            description: format!(
                "Synthesized draft sub-manifold using {} across {} layers (hidden={}).",
                config.strategy.display_name(),
                draft_layers_count,
                draft_hidden
            ),
        };

        Ok((draft_model, report))
    }
}
