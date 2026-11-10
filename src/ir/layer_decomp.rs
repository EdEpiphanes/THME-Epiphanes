//! Layer Deconstruction & Topological Component Disassembly Engine
//!
//! Deconstructs any Transformer/LLM layer into its discrete mathematical anatomy:
//! 1. Input Normalization (RMSNorm / LayerNorm)
//! 2. Sequence Mixer sub-block (GQA Attention Q/K/V/O, Gated DeltaNet, Phase Recurrence, Mamba-2 SSM)
//! 3. Residual Highways (Skip connections bypassing Mixer and FFN)
//! 4. Post-Mixer Normalization
//! 5. Feed-Forward MLP sub-block (Gate / Up / Down projections, SwiGLU / GeLU)
//! 6. Routing & Early-Exit Decision Block ($q_l$ reach probability, early-exit checkpoint branch)

use serde::{Deserialize, Serialize};

use super::tensor::TensorRole;

/// High-level topology classification of a layer
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum LayerTopologyStyle {
    StandardMonolithic,
    HybridDualMixer,    // e.g. GQA Attention + Gated DeltaNet
    TribridTriMixer,    // e.g. Attention + Gated DeltaNet + Phase Recurrence
    MambaSsmHybrid,     // Attention + Mamba-2 SSM
    CustomByoaTopology, // User-defined BYOA sequence mixer
}

impl LayerTopologyStyle {
    pub fn display_name(&self) -> &'static str {
        match self {
            Self::StandardMonolithic => "Standard Monolithic Transformer",
            Self::HybridDualMixer => "Hybrid Dual-Mixer (Attention + DeltaNet)",
            Self::TribridTriMixer => "Tri-brid Tri-Mixer (Attn + DeltaNet + Phase)",
            Self::MambaSsmHybrid => "Mamba-2 SSM Hybrid",
            Self::CustomByoaTopology => "Custom BYOA Topology",
        }
    }
}

/// Category of an intra-layer sub-block
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SubBlockCategory {
    InputNorm,
    SequenceMixer,
    ResidualHighway,
    PostMixerNorm,
    FeedForwardMlp,
    RoutingExitBranch,
}

impl SubBlockCategory {
    pub fn display_name(&self) -> &'static str {
        match self {
            Self::InputNorm => "Input Normalization",
            Self::SequenceMixer => "Sequence Mixer",
            Self::ResidualHighway => "Residual Highway",
            Self::PostMixerNorm => "Post-Mixer Normalization",
            Self::FeedForwardMlp => "Feed-Forward Network (FFN)",
            Self::RoutingExitBranch => "Routing & Exit Branch",
        }
    }
}

/// Individual tensor representation within a deconstructed sub-block
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DeconstructedTensor {
    pub name: String,
    pub role: TensorRole,
    pub shape: Vec<usize>,
    pub dtype: String,
    pub num_elements: usize,
    pub byte_size: usize,
    pub is_active: bool,
    pub selected_format: String,
    pub effective_bits: f32,
    pub routing_destination: String,
    pub min_val: f32,
    pub max_val: f32,
    pub mean_val: f32,
    pub std_val: f32,
    pub sparsity: f32,
    pub outlier_ratio: f32,
}

/// A discrete sub-block component within a layer
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LayerSubBlock {
    pub category: SubBlockCategory,
    pub title: String,
    pub description: String,
    pub tensors: Vec<DeconstructedTensor>,
    pub total_parameters: usize,
    pub total_bytes: usize,
    pub is_active: bool,
    pub routing_source: String,
    pub routing_target: String,
}

/// Complete structural deconstruction and dataflow analysis of a single layer
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LayerDeconstruction {
    pub layer_index: usize,
    pub topology_style: LayerTopologyStyle,
    pub mixer_type: String,
    pub input_norm: Option<LayerSubBlock>,
    pub mixer_block: LayerSubBlock,
    pub post_mixer_norm: Option<LayerSubBlock>,
    pub ffn_block: Option<LayerSubBlock>,
    pub residual_highway: LayerSubBlock,
    pub routing_block: LayerSubBlock,
    pub total_parameters: usize,
    pub total_bytes: usize,
    pub active_bytes_per_token: usize,
    pub reach_probability: f32,
    pub is_exit_checkpoint: bool,
    pub alpha_binary: f32,
    pub beta_ternary: f32,
    pub gamma_phase: f32,
}

impl LayerDeconstruction {
    /// Deconstructs a layer from an arbitrary list of tensors and layer metadata
    #[allow(clippy::too_many_arguments)]
    pub fn analyze(
        layer_index: usize,
        mixer_type: &str,
        reach_probability: f32,
        is_exit_checkpoint: bool,
        alpha_binary: f32,
        beta_ternary: f32,
        gamma_phase: f32,
        tensors: &[DeconstructedTensor],
    ) -> Self {
        let is_active = reach_probability > 0.05;

        // 1. Separate tensors by role and name
        let mut input_norm_tensors = Vec::new();
        let mut mixer_tensors = Vec::new();
        let mut post_norm_tensors = Vec::new();
        let mut ffn_tensors = Vec::new();

        for t in tensors {
            let lower = t.name.to_lowercase();
            match t.role {
                TensorRole::LayerNorm => {
                    if lower.contains("post_attention") || lower.contains("ffn_norm") {
                        post_norm_tensors.push(t.clone());
                    } else {
                        input_norm_tensors.push(t.clone());
                    }
                }
                TensorRole::AttentionQ
                | TensorRole::AttentionK
                | TensorRole::AttentionV
                | TensorRole::AttentionOut => {
                    mixer_tensors.push(t.clone());
                }
                TensorRole::MlpGate | TensorRole::MlpUp | TensorRole::MlpDown => {
                    ffn_tensors.push(t.clone());
                }
                _ => {
                    if lower.contains("attn") || lower.contains("mixer") {
                        mixer_tensors.push(t.clone());
                    } else if lower.contains("mlp") || lower.contains("ffn") {
                        ffn_tensors.push(t.clone());
                    }
                }
            }
        }

        // Determine topology style
        let topology_style = if mixer_type.contains("tribrid")
            || (alpha_binary > 0.05 && beta_ternary > 0.05 && gamma_phase > 0.05)
        {
            LayerTopologyStyle::TribridTriMixer
        } else if mixer_type.contains("hybrid")
            || (mixer_type == "gated_deltanet" && alpha_binary > 0.1)
        {
            LayerTopologyStyle::HybridDualMixer
        } else if mixer_type == "mamba2" {
            LayerTopologyStyle::MambaSsmHybrid
        } else if mixer_type == "phase_recurrence" {
            LayerTopologyStyle::CustomByoaTopology
        } else {
            LayerTopologyStyle::StandardMonolithic
        };

        // Build Input Norm Sub-Block
        let input_norm = if !input_norm_tensors.is_empty() {
            let params: usize = input_norm_tensors.iter().map(|t| t.num_elements).sum();
            let bytes: usize = input_norm_tensors.iter().map(|t| t.byte_size).sum();
            Some(LayerSubBlock {
                category: SubBlockCategory::InputNorm,
                title: "Input Normalization (RMSNorm)".to_string(),
                description: "Normalizes incoming residual stream variance prior to sequence mixer projection.".to_string(),
                tensors: input_norm_tensors,
                total_parameters: params,
                total_bytes: bytes,
                is_active,
                routing_source: format!("Layer {} Input Residual Highway", layer_index),
                routing_target: format!("Mixer Sub-Block ({})", mixer_type),
            })
        } else {
            None
        };

        // Build Sequence Mixer Sub-Block
        let mixer_params: usize = mixer_tensors.iter().map(|t| t.num_elements).sum();
        let mixer_bytes: usize = mixer_tensors.iter().map(|t| t.byte_size).sum();
        let mixer_block = LayerSubBlock {
            category: SubBlockCategory::SequenceMixer,
            title: format!("Sequence Mixer [{}]", mixer_type),
            description: match mixer_type {
                "gqa_attention" => "Grouped-Query Attention (GQA): Computes dot-product multi-head sequence attention with shared KV heads.",
                "gated_deltanet" => "Gated DeltaNet: Linear state-space recurrent associative memory with input data-dependent gating.",
                "phase_recurrence" => "Phase Recurrence: Unitary complex diagonal state evolution preserving long-range gradient norm.",
                "sliding_window" => "Sliding Window Attention: Banded local context mixer reducing quadratic memory to linear window.",
                "mamba2" => "Mamba-2 State Space Duality (SSD): Hardware-aligned matrix multi-head state space recurrence.",
                _ => "Heterogeneous sequence mixer decomposing token dependencies across channel manifolds.",
            }.to_string(),
            tensors: mixer_tensors,
            total_parameters: mixer_params,
            total_bytes: mixer_bytes,
            is_active,
            routing_source: "Input Normalization".to_string(),
            routing_target: "Residual Highway 1 (⊕ Add)".to_string(),
        };

        // Build Post-Mixer Norm Sub-Block
        let post_mixer_norm = if !post_norm_tensors.is_empty() {
            let params: usize = post_norm_tensors.iter().map(|t| t.num_elements).sum();
            let bytes: usize = post_norm_tensors.iter().map(|t| t.byte_size).sum();
            Some(LayerSubBlock {
                category: SubBlockCategory::PostMixerNorm,
                title: "Post-Mixer Normalization (RMSNorm)".to_string(),
                description: "Re-centers and scales intermediate residual activations prior to non-linear MLP expansion.".to_string(),
                tensors: post_norm_tensors,
                total_parameters: params,
                total_bytes: bytes,
                is_active,
                routing_source: "Residual Highway 1 (⊕ Add)".to_string(),
                routing_target: "Feed-Forward MLP (SwiGLU)".to_string(),
            })
        } else {
            None
        };

        // Build FFN MLP Sub-Block
        let ffn_params: usize = ffn_tensors.iter().map(|t| t.num_elements).sum();
        let ffn_bytes: usize = ffn_tensors.iter().map(|t| t.byte_size).sum();
        let ffn_block = if !ffn_tensors.is_empty() {
            Some(LayerSubBlock {
                category: SubBlockCategory::FeedForwardMlp,
                title: "Feed-Forward Network (SwiGLU MLP)".to_string(),
                description: "Non-linear channel expansion: W_gate(x) * Swish(W_up(x)) projected back by W_down.".to_string(),
                tensors: ffn_tensors,
                total_parameters: ffn_params,
                total_bytes: ffn_bytes,
                is_active,
                routing_source: "Post-Mixer Normalization".to_string(),
                routing_target: "Residual Highway 2 (⊕ Add)".to_string(),
            })
        } else {
            None
        };

        // Build Residual Highway Sub-Block
        let residual_highway = LayerSubBlock {
            category: SubBlockCategory::ResidualHighway,
            title: "Dual Residual Highway (Identity Shortcuts)".to_string(),
            description: "Zero-latency identity pass-through connections preserving deep gradient flow without transformation loss.".to_string(),
            tensors: Vec::new(),
            total_parameters: 0,
            total_bytes: 0,
            is_active,
            routing_source: format!("Layer {} Input Signal", layer_index),
            routing_target: format!("Layer {} Output Collector", layer_index),
        };

        // Build Routing Sub-Block
        let next_layer_str = format!("Layer {}", layer_index + 1);
        let (routing_desc, routing_target) = if is_exit_checkpoint {
            (
                format!(
                    "Early-Exit Checkpoint: {:.0}% exit directly to Early-Exit Head; {:.0}% continue to {}",
                    (1.0 - reach_probability) * 100.0,
                    reach_probability * 100.0,
                    next_layer_str
                ),
                format!("80% -> Early-Exit Classifier Head | 20% -> {}", next_layer_str),
            )
        } else {
            (
                format!(
                    "Sequential Pass-Through: 100% of tokens propagate to {}",
                    next_layer_str
                ),
                next_layer_str,
            )
        };

        let routing_block = LayerSubBlock {
            category: SubBlockCategory::RoutingExitBranch,
            title: if is_exit_checkpoint {
                "Adaptive Early-Exit Routing Node".to_string()
            } else {
                "Sequential Routing Gateway".to_string()
            },
            description: routing_desc,
            tensors: Vec::new(),
            total_parameters: 0,
            total_bytes: 0,
            is_active,
            routing_source: "Residual Highway 2 (⊕ Add)".to_string(),
            routing_target,
        };

        let total_parameters = mixer_params
            + ffn_params
            + input_norm.as_ref().map(|b| b.total_parameters).unwrap_or(0)
            + post_mixer_norm
                .as_ref()
                .map(|b| b.total_parameters)
                .unwrap_or(0);

        let total_bytes = mixer_bytes
            + ffn_bytes
            + input_norm.as_ref().map(|b| b.total_bytes).unwrap_or(0)
            + post_mixer_norm.as_ref().map(|b| b.total_bytes).unwrap_or(0);

        let active_bytes_per_token = (total_bytes as f32 * reach_probability) as usize;

        Self {
            layer_index,
            topology_style,
            mixer_type: mixer_type.to_string(),
            input_norm,
            mixer_block,
            post_mixer_norm,
            ffn_block,
            residual_highway,
            routing_block,
            total_parameters,
            total_bytes,
            active_bytes_per_token,
            reach_probability,
            is_exit_checkpoint,
            alpha_binary,
            beta_ternary,
            gamma_phase,
        }
    }
}
