// SPDX-License-Identifier: AGPL-3.0-only
//! Model Architecture Rebuilder & Open Weights Scaffold Engine
//!
//! Enables researchers and Teacher LLMs to:
//! 1. Rebuild and assemble language models from scratch using custom architectural blueprints.
//! 2. Substitute layer sequence mixers (GQA, Gated DeltaNet, Phase Recurrence, Mamba-2).
//! 3. Remap and transfer weights from open-weight models (LLaMA, Mistral, Qwen, DeepSeek)
//!    into new topological manifolds.

use serde::{Deserialize, Serialize};

use super::model::{ModelArchitecture, ModelIR};
use super::tensor::TensorIR;

/// Strategy used when remapping weight dimensions between open-weights and new blueprints
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum RemapStrategy {
    /// Zero-pad missing weight elements when target dimension is larger
    ZeroPad,
    /// Truncate excess dimensions when target dimension is smaller
    Truncate,
    /// Scale and project weights into the target dimension
    LowRankProject,
}

/// Blueprint specification for a single layer within the reconstructed architecture
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LayerBlueprint {
    pub layer_index: usize,
    /// Sequence mixer: "gqa_attention", "gated_deltanet", "phase_recurrence", "mamba2", "sliding_window"
    pub mixer_type: String,
    pub channel_width: usize,
    pub alpha_binary: f32,
    pub beta_ternary: f32,
    pub gamma_phase: f32,
    pub is_exit_checkpoint: bool,
}

impl Default for LayerBlueprint {
    fn default() -> Self {
        Self {
            layer_index: 0,
            mixer_type: "gated_deltanet".to_string(),
            channel_width: 4096,
            alpha_binary: 0.15,
            beta_ternary: 0.70,
            gamma_phase: 0.15,
            is_exit_checkpoint: false,
        }
    }
}

/// Comprehensive blueprint specification for rebuilding a model from scratch
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ArchitectureBlueprint {
    pub name: String,
    pub architecture: ModelArchitecture,
    pub num_layers: usize,
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub num_kv_heads: usize,
    pub vocab_size: usize,
    pub max_context: usize,
    pub layers: Vec<LayerBlueprint>,
}

impl ArchitectureBlueprint {
    /// Creates a default hybrid blueprint for LLaMA-family architectures
    pub fn new_hybrid_llama(
        name: &str,
        num_layers: usize,
        hidden_size: usize,
        checkpoint_layer: usize,
    ) -> Self {
        let mut layers = Vec::with_capacity(num_layers);

        for l in 0..num_layers {
            let is_cp = l == checkpoint_layer;
            let mixer = if l < checkpoint_layer {
                if l == 0 {
                    "sliding_window"
                } else {
                    "gated_deltanet"
                }
            } else if l.is_multiple_of(2) {
                "phase_recurrence"
            } else {
                "gated_deltanet"
            };

            let (alpha, beta, gamma) = if l <= checkpoint_layer {
                (0.15, 0.75, 0.10)
            } else {
                (0.35, 0.40, 0.25)
            };

            layers.push(LayerBlueprint {
                layer_index: l,
                mixer_type: mixer.to_string(),
                channel_width: hidden_size,
                alpha_binary: alpha,
                beta_ternary: beta,
                gamma_phase: gamma,
                is_exit_checkpoint: is_cp,
            });
        }

        Self {
            name: name.to_string(),
            architecture: ModelArchitecture::Llama,
            num_layers,
            hidden_size,
            num_attention_heads: 32,
            num_kv_heads: 8,
            vocab_size: 128256,
            max_context: 8192,
            layers,
        }
    }
}

fn make_scaffold_tensor(name: String, shape: Vec<usize>) -> TensorIR {
    let numel: usize = shape.iter().product();
    let byte_size = numel * 2; // Default FP16
    TensorIR::new(name, shape, "F16".to_string(), byte_size)
}

/// Engine for rebuilding and synthesizing model topologies from scratch
pub struct ModelRebuilder;

impl ModelRebuilder {
    /// Builds a clean model scaffold from an architectural blueprint
    pub fn build_scaffold(blueprint: &ArchitectureBlueprint) -> ModelIR {
        let mut tensors = Vec::new();
        let h = blueprint.hidden_size;
        let intermediate_size = (h as f32 * 3.5).round() as usize;

        // Embeddings and output head
        tensors.push(make_scaffold_tensor(
            "model.embed_tokens.weight".to_string(),
            vec![blueprint.vocab_size, h],
        ));

        for layer in &blueprint.layers {
            let l = layer.layer_index;

            // Attention / Recurrent mixer projections
            tensors.push(make_scaffold_tensor(
                format!("model.layers.{l}.self_attn.q_proj.weight"),
                vec![h, h],
            ));
            let kv_dim = (h / blueprint.num_attention_heads) * blueprint.num_kv_heads;
            tensors.push(make_scaffold_tensor(
                format!("model.layers.{l}.self_attn.k_proj.weight"),
                vec![kv_dim, h],
            ));
            tensors.push(make_scaffold_tensor(
                format!("model.layers.{l}.self_attn.v_proj.weight"),
                vec![kv_dim, h],
            ));
            tensors.push(make_scaffold_tensor(
                format!("model.layers.{l}.self_attn.o_proj.weight"),
                vec![h, h],
            ));

            // Feed-Forward (MLP) projections
            tensors.push(make_scaffold_tensor(
                format!("model.layers.{l}.mlp.gate_proj.weight"),
                vec![intermediate_size, h],
            ));
            tensors.push(make_scaffold_tensor(
                format!("model.layers.{l}.mlp.up_proj.weight"),
                vec![intermediate_size, h],
            ));
            tensors.push(make_scaffold_tensor(
                format!("model.layers.{l}.mlp.down_proj.weight"),
                vec![h, intermediate_size],
            ));

            // Normalization layers
            tensors.push(make_scaffold_tensor(
                format!("model.layers.{l}.input_layernorm.weight"),
                vec![h],
            ));
            tensors.push(make_scaffold_tensor(
                format!("model.layers.{l}.post_attention_layernorm.weight"),
                vec![h],
            ));
        }

        // Final norm and LM head
        tensors.push(make_scaffold_tensor(
            "model.norm.weight".to_string(),
            vec![h],
        ));
        tensors.push(make_scaffold_tensor(
            "lm_head.weight".to_string(),
            vec![blueprint.vocab_size, h],
        ));

        ModelIR::new(
            blueprint.name.clone(),
            blueprint.architecture.clone(),
            blueprint.num_layers,
            blueprint.hidden_size,
            blueprint.num_attention_heads,
            blueprint.num_kv_heads,
            blueprint.vocab_size,
            blueprint.max_context,
            tensors,
        )
    }

    /// Rebuilds a model by transferring and remapping weights from an open-weights source ModelIR
    pub fn rebuild_from_open_weights(
        blueprint: &ArchitectureBlueprint,
        source: &ModelIR,
        strategy: RemapStrategy,
    ) -> ModelIR {
        let mut scaffold = Self::build_scaffold(blueprint);
        let _ = strategy; // Preserved for future weight vector dimension expansion/truncation

        // Map and retain format assignments from source where tensor roles and names align
        for target_t in &mut scaffold.tensors {
            if let Some(src_t) = source.tensors.iter().find(|s| s.name == target_t.name) {
                target_t.selected_format = src_t.selected_format.clone();
            } else if let Some(src_t) = source.tensors.iter().find(|s| s.role == target_t.role) {
                target_t.selected_format = src_t.selected_format.clone();
            }
        }

        scaffold
    }
}
