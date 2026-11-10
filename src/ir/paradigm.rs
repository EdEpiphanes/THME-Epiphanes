//! 8 Specialized AI Model Paradigms & Universal Conversion Manifold
//!
//! Categorization and conversion engine covering the complete 8-model taxonomy:
//! 1. LLM (Large Language Model) - Autoregressive Causal Transformer
//! 2. LCM (Large / Latent Concept Model) - Concept space embedding & diffusion
//! 3. LAM (Large Action Model) - Intent perception, neuro-symbolic memory & action planner
//! 4. MoE (Mixture of Experts) - Router gating network & sparse expert sub-networks
//! 5. VLM (Vision-Language Model) - Dual vision/text encoders & multimodal projection
//! 6. SLM (Small Language Model) - Extreme parameter efficiency & edge optimization
//! 7. MLM (Masked Language Model) - Bidirectional attention & token masking
//! 8. SAM (Segment Anything Model) - Spatial attention, prompt encoder & mask decoders

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::model::ModelIR;
use super::tensor::{TensorFormat, TensorIR, TensorRole, WeightProfile};

/// The 8 primary specialized AI model paradigms
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SpecializedModelParadigm {
    /// Large Language Model: Autoregressive causal generation (LLaMA, Mistral, Qwen, Gemma)
    LLM,
    /// Large Concept Model: Sentence segmentation, SONAR concept space, latent diffusion
    LCM,
    /// Large Action Model: Intent recognition, neuro-symbolic reasoning, action trajectory planning
    LAM,
    /// Mixture of Experts: Gating router mechanism and sparse expert sub-networks (Mixtral, DeepSeek)
    MoE,
    /// Vision-Language Model: Multimodal vision encoder, projection interface, causal backbone
    VLM,
    /// Small Language Model: Compact tokenization, edge deployment, BitNet b1.58, early exit
    SLM,
    /// Masked Language Model: Bidirectional attention, token masking prediction (BERT, ModernBERT)
    MLM,
    /// Segment Anything Model: Spatial visual attention, prompt encoder, mask decoder
    SAM,
}

impl SpecializedModelParadigm {
    pub fn all() -> [Self; 8] {
        [
            Self::LLM,
            Self::LCM,
            Self::LAM,
            Self::MoE,
            Self::VLM,
            Self::SLM,
            Self::MLM,
            Self::SAM,
        ]
    }

    pub fn code(&self) -> &'static str {
        match self {
            Self::LLM => "LLM",
            Self::LCM => "LCM",
            Self::LAM => "LAM",
            Self::MoE => "MoE",
            Self::VLM => "VLM",
            Self::SLM => "SLM",
            Self::MLM => "MLM",
            Self::SAM => "SAM",
        }
    }

    pub fn display_name(&self) -> &'static str {
        match self {
            Self::LLM => "Large Language Model",
            Self::LCM => "Large Concept Model",
            Self::LAM => "Large Action Model",
            Self::MoE => "Mixture of Experts",
            Self::VLM => "Vision-Language Model",
            Self::SLM => "Small Language Model",
            Self::MLM => "Masked Language Model",
            Self::SAM => "Segment Anything Model",
        }
    }

    pub fn description(&self) -> &'static str {
        match self {
            Self::LLM => "Autoregressive causal generation: Tokenize -> Embed -> Transformer -> Next-Token Head",
            Self::LCM => "Concept-level generation: Segmentation -> SONAR Embed -> Latent Diffusion -> Concept Decoder",
            Self::LAM => "Agentic execution: Perception -> Intent Recognition -> Neuro-Symbolic Memory -> Action Planner",
            Self::MoE => "Sparse dynamic routing: Router Mechanism -> Top-K Expert Selection -> Weighted Combination",
            Self::VLM => "Multimodal reasoning: Vision Encoder + Text Encoder -> Projection Interface -> Language Model",
            Self::SLM => "Edge efficiency: Compact Tokenizer -> Grouped Query Attn -> BitNet b1.58 -> Low-Latency",
            Self::MLM => "Bidirectional representations: Token Masking -> Dual Left/Right Context -> Masked Token Head",
            Self::SAM => "Spatial segmentation: Image Encoder + Prompt Encoder -> Embedding -> Promptable Mask Decoder",
        }
    }

    pub fn primary_components(&self) -> Vec<&'static str> {
        match self {
            Self::LLM => vec![
                "Token Embedder",
                "Causal Attention",
                "SwiGLU MLP",
                "LM Head",
            ],
            Self::LCM => vec![
                "Sentence Segmenter",
                "SONAR Embedder",
                "Latent Diffusion",
                "Concept Quantizer",
            ],
            Self::LAM => vec![
                "Perception System",
                "Intent Router",
                "Neuro-Symbolic Memory",
                "Action Planner",
            ],
            Self::MoE => vec![
                "Router Gate",
                "Expert MLP Bank",
                "Top-K Dispatcher",
                "Weighted Combiner",
            ],
            Self::VLM => vec![
                "Vision Encoder",
                "Text Encoder",
                "Multimodal Projector",
                "Fusion LM",
            ],
            Self::SLM => vec![
                "Compact Tokenizer",
                "Efficient GQA",
                "BitNet b1.58 MLP",
                "Edge Cache",
            ],
            Self::MLM => vec![
                "Token Masking",
                "Bidirectional Attn",
                "Dual Context Highway",
                "Mask Prediction Head",
            ],
            Self::SAM => vec![
                "Image Encoder",
                "Prompt Encoder",
                "Dense Feature Correlator",
                "Mask Decoder",
            ],
        }
    }

    pub fn from_str_code(s: &str) -> Option<Self> {
        match s.to_uppercase().trim() {
            "LLM" => Some(Self::LLM),
            "LCM" => Some(Self::LCM),
            "LAM" => Some(Self::LAM),
            "MOE" => Some(Self::MoE),
            "VLM" => Some(Self::VLM),
            "SLM" => Some(Self::SLM),
            "MLM" => Some(Self::MLM),
            "SAM" => Some(Self::SAM),
            _ => None,
        }
    }
}

/// Configuration parameters governing cross-paradigm model transformations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParadigmConversionConfig {
    /// Target paradigm to convert into
    pub target_paradigm: SpecializedModelParadigm,
    /// Number of experts when converting to MoE
    pub num_experts: usize,
    /// Top-K active experts per token when converting to MoE
    pub top_k: usize,
    /// Target hidden dimension when downscaling to SLM or upscaling
    pub target_hidden_size: Option<usize>,
    /// Target layer count when pruning or expanding
    pub target_layers: Option<usize>,
    /// Whether to apply BitNet b1.58 ternary discretization (ideal for SLM/edge)
    pub apply_bitnet: bool,
    /// Vision dimension for VLM/SAM vision tower
    pub vision_hidden_dim: usize,
    /// Number of action trajectories for LAM action planner
    pub num_action_outputs: usize,
    /// Concept embedding dimension for LCM
    pub concept_dim: usize,
}

impl ParadigmConversionConfig {
    pub fn new(target_paradigm: SpecializedModelParadigm) -> Self {
        Self {
            target_paradigm,
            ..Default::default()
        }
    }
}

impl Default for ParadigmConversionConfig {
    fn default() -> Self {
        Self {
            target_paradigm: SpecializedModelParadigm::MoE,
            num_experts: 4,
            top_k: 2,
            target_hidden_size: None,
            target_layers: None,
            apply_bitnet: false,
            vision_hidden_dim: 1024,
            num_action_outputs: 32,
            concept_dim: 1024,
        }
    }
}

/// Analysis summary of an architectural paradigm conversion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParadigmConversionReport {
    pub source_paradigm: SpecializedModelParadigm,
    pub target_paradigm: SpecializedModelParadigm,
    pub original_parameters: usize,
    pub new_parameters: usize,
    pub parameter_delta_pct: f64,
    pub original_byte_size: usize,
    pub new_byte_size: usize,
    pub storage_savings_pct: f64,
    pub new_tensors_added: usize,
    pub tensors_restructured: usize,
    pub description: String,
}

impl ParadigmConversionReport {
    pub fn format_summary(&self) -> String {
        format!(
            "Conversion Summary: {} -> {}\n- Parameters: {} -> {} ({:+.1}%)\n- Footprint: {:.2} MB -> {:.2} MB ({:+.1}% savings)\n- Tensors Added: {} | Restructured: {}\n- Details: {}",
            self.source_paradigm.code(),
            self.target_paradigm.code(),
            self.original_parameters,
            self.new_parameters,
            self.parameter_delta_pct,
            self.original_byte_size as f64 / (1024.0 * 1024.0),
            self.new_byte_size as f64 / (1024.0 * 1024.0),
            self.storage_savings_pct,
            self.new_tensors_added,
            self.tensors_restructured,
            self.description
        )
    }
}

/// Universal Model Paradigm Converter & Architectural Manifold Engine
pub struct ParadigmConverter;

impl ParadigmConverter {
    /// Automatically detects the prevailing paradigm of an imported ModelIR
    pub fn detect_paradigm(model_ir: &ModelIR) -> SpecializedModelParadigm {
        let has_moe = model_ir.tensors.iter().any(|t| {
            t.name.contains("feed_forward.router")
                || t.name.contains("moe_router")
                || t.name.contains("experts.")
                || t.name.contains("gate_network")
        });
        if has_moe {
            return SpecializedModelParadigm::MoE;
        }

        let has_vision = model_ir.tensors.iter().any(|t| {
            t.name.contains("vision")
                || t.name.contains("mm_projector")
                || t.name.contains("image_encoder")
        });
        let has_mask_decoder = model_ir
            .tensors
            .iter()
            .any(|t| t.name.contains("mask_decoder") || t.name.contains("prompt_encoder"));
        if has_vision && has_mask_decoder {
            return SpecializedModelParadigm::SAM;
        }
        if has_vision {
            return SpecializedModelParadigm::VLM;
        }

        let has_action = model_ir
            .tensors
            .iter()
            .any(|t| t.name.contains("action_head") || t.name.contains("actor_trajectory"));
        if has_action {
            return SpecializedModelParadigm::LAM;
        }

        let has_concept = model_ir
            .tensors
            .iter()
            .any(|t| t.name.contains("concept_") || t.name.contains("sonar"));
        if has_concept {
            return SpecializedModelParadigm::LCM;
        }

        let has_mlm = model_ir.tensors.iter().any(|t| {
            t.name.contains("cls.predictions") || t.name.contains("token_type_embeddings")
        });
        if has_mlm {
            return SpecializedModelParadigm::MLM;
        }

        let is_slm = (model_ir.name.to_lowercase().contains("-slm-")
            && !model_ir.name.to_lowercase().contains("standardllm"))
            || (model_ir.num_parameters < 1_500_000_000
                && model_ir
                    .tensors
                    .iter()
                    .any(|t| matches!(t.selected_format, TensorFormat::Ternary)));
        if is_slm {
            return SpecializedModelParadigm::SLM;
        }

        SpecializedModelParadigm::LLM
    }

    /// Creates a synthetic 12-layer baseline LLM for benchmarking and dry-runs
    pub fn create_synthetic_baseline() -> ModelIR {
        let mut tensors = Vec::new();
        let hidden = 2048;
        let intermediate = 5632;

        tensors.push(TensorIR::new(
            "model.embed_tokens.weight".to_string(),
            vec![32000, hidden],
            "F16".to_string(),
            32000 * hidden * 2,
        ));

        for i in 0..12 {
            tensors.push(TensorIR::new(
                format!("model.layers.{}.self_attn.q_proj.weight", i),
                vec![hidden, hidden],
                "F16".to_string(),
                hidden * hidden * 2,
            ));
            tensors.push(TensorIR::new(
                format!("model.layers.{}.self_attn.k_proj.weight", i),
                vec![hidden, hidden],
                "F16".to_string(),
                hidden * hidden * 2,
            ));
            tensors.push(TensorIR::new(
                format!("model.layers.{}.self_attn.v_proj.weight", i),
                vec![hidden, hidden],
                "F16".to_string(),
                hidden * hidden * 2,
            ));
            tensors.push(TensorIR::new(
                format!("model.layers.{}.self_attn.o_proj.weight", i),
                vec![hidden, hidden],
                "F16".to_string(),
                hidden * hidden * 2,
            ));
            tensors.push(TensorIR::new(
                format!("model.layers.{}.mlp.gate_proj.weight", i),
                vec![intermediate, hidden],
                "F16".to_string(),
                intermediate * hidden * 2,
            ));
            tensors.push(TensorIR::new(
                format!("model.layers.{}.mlp.up_proj.weight", i),
                vec![intermediate, hidden],
                "F16".to_string(),
                intermediate * hidden * 2,
            ));
            tensors.push(TensorIR::new(
                format!("model.layers.{}.mlp.down_proj.weight", i),
                vec![hidden, intermediate],
                "F16".to_string(),
                hidden * intermediate * 2,
            ));
        }

        tensors.push(TensorIR::new(
            "lm_head.weight".to_string(),
            vec![32000, hidden],
            "F16".to_string(),
            32000 * hidden * 2,
        ));

        ModelIR::new(
            "synthetic-llama-baseline".to_string(),
            crate::ir::ModelArchitecture::Llama,
            12,
            hidden,
            32,
            32,
            32000,
            intermediate,
            tensors,
        )
    }

    /// Converts an imported ModelIR from any paradigm to any target paradigm
    pub fn convert(
        source_model: &ModelIR,
        config: &ParadigmConversionConfig,
    ) -> Result<(ModelIR, ParadigmConversionReport), String> {
        let source_paradigm = Self::detect_paradigm(source_model);
        let target_paradigm = config.target_paradigm;

        let mut converted_model = source_model.clone();
        let original_params = source_model.num_parameters;
        let original_bytes = source_model.original_byte_size;

        let (new_tensors_added, tensors_restructured) = match target_paradigm {
            SpecializedModelParadigm::MoE => {
                let (new_tensors, added, modded) =
                    Self::upcycle_to_moe(&converted_model, config.num_experts, config.top_k)?;
                converted_model.tensors = new_tensors;
                converted_model.name = format!(
                    "{}-MoE-{}x{}",
                    converted_model.name, config.num_experts, config.top_k
                );
                (added, modded)
            }
            SpecializedModelParadigm::SLM => {
                let target_h = config
                    .target_hidden_size
                    .unwrap_or_else(|| (converted_model.hidden_size / 2).max(1024));
                let target_l = config
                    .target_layers
                    .unwrap_or_else(|| (converted_model.num_layers / 2).max(8));
                let (new_tensors, added, modded) = Self::compress_to_slm(
                    &converted_model,
                    target_h,
                    target_l,
                    config.apply_bitnet,
                )?;
                converted_model.tensors = new_tensors;
                converted_model.hidden_size = target_h;
                converted_model.num_layers = target_l;
                converted_model.name =
                    format!("{}-SLM-{}L-{}D", converted_model.name, target_l, target_h);
                (added, modded)
            }
            SpecializedModelParadigm::VLM => {
                let (new_tensors, added, modded) =
                    Self::augment_to_vlm(&converted_model, config.vision_hidden_dim)?;
                converted_model.tensors = new_tensors;
                converted_model.name = format!("{}-VLM-Multimodal", converted_model.name);
                (added, modded)
            }
            SpecializedModelParadigm::LAM => {
                let (new_tensors, added, modded) =
                    Self::augment_to_lam(&converted_model, config.num_action_outputs)?;
                converted_model.tensors = new_tensors;
                converted_model.name = format!("{}-LAM-ActionAgent", converted_model.name);
                (added, modded)
            }
            SpecializedModelParadigm::LCM => {
                let (new_tensors, added, modded) =
                    Self::augment_to_lcm(&converted_model, config.concept_dim)?;
                converted_model.tensors = new_tensors;
                converted_model.name = format!("{}-LCM-ConceptDiffusion", converted_model.name);
                (added, modded)
            }
            SpecializedModelParadigm::MLM => {
                let (new_tensors, added, modded) = Self::convert_to_mlm(&converted_model)?;
                converted_model.tensors = new_tensors;
                converted_model.name = format!("{}-MLM-Bidirectional", converted_model.name);
                (added, modded)
            }
            SpecializedModelParadigm::SAM => {
                let (new_tensors, added, modded) =
                    Self::augment_to_sam(&converted_model, config.vision_hidden_dim)?;
                converted_model.tensors = new_tensors;
                converted_model.name = format!("{}-SAM-SpatialSegmenter", converted_model.name);
                (added, modded)
            }
            SpecializedModelParadigm::LLM => {
                let (new_tensors, added, modded) = Self::consolidate_to_llm(&converted_model)?;
                converted_model.tensors = new_tensors;
                let clean_name = converted_model
                    .name
                    .replace("-SLM-", "-")
                    .replace("-MoE-", "-")
                    .replace("-VLM-", "-")
                    .replace("-LAM-", "-")
                    .replace("-LCM-", "-")
                    .replace("-MLM-", "-")
                    .replace("-SAM-", "-");
                converted_model.name = format!("{}-StandardLLM", clean_name);
                (added, modded)
            }
        };

        // Recompute global metrics
        converted_model.num_parameters = converted_model.tensors.iter().map(|t| t.numel).sum();
        converted_model.original_byte_size =
            converted_model.tensors.iter().map(|t| t.byte_size).sum();

        let new_params = converted_model.num_parameters;
        let new_bytes = converted_model.original_byte_size;
        let param_delta_pct = if original_params > 0 {
            ((new_params as f64 - original_params as f64) / original_params as f64) * 100.0
        } else {
            0.0
        };
        let storage_savings_pct = if original_bytes > 0 {
            ((original_bytes as f64 - new_bytes as f64) / original_bytes as f64) * 100.0
        } else {
            0.0
        };

        let report = ParadigmConversionReport {
            source_paradigm,
            target_paradigm,
            original_parameters: original_params,
            new_parameters: new_params,
            parameter_delta_pct: param_delta_pct,
            original_byte_size: original_bytes,
            new_byte_size: new_bytes,
            storage_savings_pct,
            new_tensors_added,
            tensors_restructured,
            description: format!(
                "Successfully converted model from {} to {}. Parameters: {} -> {} ({:+.1}%), Tensors Restructured: {}.",
                source_paradigm.code(),
                target_paradigm.code(),
                original_params,
                new_params,
                param_delta_pct,
                tensors_restructured + new_tensors_added
            ),
        };

        Ok((converted_model, report))
    }

    /// Sparse Upcycling: Converts monolithic MLP layers into sparse Mixture of Experts (MoE)
    fn upcycle_to_moe(
        model: &ModelIR,
        num_experts: usize,
        _top_k: usize,
    ) -> Result<(Vec<TensorIR>, usize, usize), String> {
        let mut new_tensors = Vec::new();
        let mut added = 0;
        let mut restructured = 0;

        for t in &model.tensors {
            let is_mlp = matches!(
                t.role,
                TensorRole::MlpGate | TensorRole::MlpUp | TensorRole::MlpDown
            );

            if is_mlp {
                restructured += 1;
                // Replicate into N experts with distinct expert identifiers
                for e in 0..num_experts {
                    let mut exp_t = t.clone();
                    let role_tag = match t.role {
                        TensorRole::MlpGate => "gate_proj",
                        TensorRole::MlpUp => "up_proj",
                        TensorRole::MlpDown => "down_proj",
                        _ => "mlp",
                    };
                    exp_t.name = if let Some(l) = t.layer_index {
                        format!(
                            "layers.{}.feed_forward.experts.{}.{}.weight",
                            l, e, role_tag
                        )
                    } else {
                        format!("feed_forward.experts.{}.{}.weight", e, role_tag)
                    };
                    new_tensors.push(exp_t);
                    if e > 0 {
                        added += 1;
                    }
                }
            } else {
                new_tensors.push(t.clone());
            }
        }

        // Synthesize router gating tensors for each layer
        for l in 0..model.num_layers {
            let router_shape = vec![num_experts, model.hidden_size];
            let router_numel = num_experts * model.hidden_size;
            let router_bytes = router_numel * 2; // FP16

            let router_tensor = TensorIR::new(
                format!("layers.{}.feed_forward.router.weight", l),
                router_shape,
                "F16".to_string(),
                router_bytes,
            );
            new_tensors.push(router_tensor);
            added += 1;
        }

        Ok((new_tensors, added, restructured))
    }

    /// Compresses a large LLM into an edge-optimized Small Language Model (SLM)
    fn compress_to_slm(
        model: &ModelIR,
        target_hidden: usize,
        target_layers: usize,
        apply_bitnet: bool,
    ) -> Result<(Vec<TensorIR>, usize, usize), String> {
        let mut new_tensors = Vec::new();
        let mut restructured = 0;

        for t in &model.tensors {
            // Drop layers beyond target layer depth
            if let Some(l) = t.layer_index {
                if l >= target_layers {
                    continue;
                }
            }

            let mut compact_t = t.clone();

            // Downscale projection dimensions if 2D weight matrix
            if compact_t.shape.len() == 2 {
                let out_dim = (compact_t.shape[0] * target_hidden / model.hidden_size).max(1);
                let in_dim = (compact_t.shape[1] * target_hidden / model.hidden_size).max(1);
                compact_t.shape = vec![out_dim, in_dim];
                compact_t.numel = out_dim * in_dim;
                compact_t.byte_size = compact_t.numel * 2;
                restructured += 1;
            }

            // Apply BitNet b1.58 ternary format if requested
            if apply_bitnet
                && matches!(
                    compact_t.role,
                    TensorRole::MlpGate | TensorRole::MlpUp | TensorRole::MlpDown
                )
            {
                compact_t.selected_format = TensorFormat::Ternary;
                compact_t.weight_profile = Some(WeightProfile {
                    sparsity: 0.50,
                    ..Default::default()
                });
                restructured += 1;
            }

            new_tensors.push(compact_t);
        }

        Ok((new_tensors, 0, restructured))
    }

    /// Attaches Vision Encoder & Multimodal Projector to create a Vision-Language Model (VLM)
    fn augment_to_vlm(
        model: &ModelIR,
        vision_hidden_dim: usize,
    ) -> Result<(Vec<TensorIR>, usize, usize), String> {
        let mut new_tensors = model.tensors.clone();
        let mut added = 0;

        // 1. Multimodal Projection Layer: [model.hidden_size, vision_hidden_dim]
        let proj_shape = vec![model.hidden_size, vision_hidden_dim];
        let proj_numel = model.hidden_size * vision_hidden_dim;
        let proj_bytes = proj_numel * 2;

        let projector_tensor = TensorIR::new(
            "model.mm_projector.weight".to_string(),
            proj_shape,
            "F16".to_string(),
            proj_bytes,
        );
        new_tensors.push(projector_tensor);
        added += 1;

        // 2. Vision Tower Patch Embedding & Attention Scaffold (ViT / SigLIP)
        let vit_layers = 12;
        for vl in 0..vit_layers {
            let vit_shape = vec![vision_hidden_dim, vision_hidden_dim];
            let vit_numel = vision_hidden_dim * vision_hidden_dim;
            let vit_bytes = vit_numel * 2;

            new_tensors.push(TensorIR::new(
                format!("model.vision_tower.encoder.layers.{}.attn.qkv.weight", vl),
                vit_shape.clone(),
                "F16".to_string(),
                vit_bytes,
            ));
            new_tensors.push(TensorIR::new(
                format!("model.vision_tower.encoder.layers.{}.mlp.down.weight", vl),
                vit_shape,
                "F16".to_string(),
                vit_bytes,
            ));
            added += 2;
        }

        Ok((new_tensors, added, 0))
    }

    /// Augments the backbone with intent recognition and neuro-symbolic action planning (LAM)
    fn augment_to_lam(
        model: &ModelIR,
        num_action_outputs: usize,
    ) -> Result<(Vec<TensorIR>, usize, usize), String> {
        let mut new_tensors = model.tensors.clone();
        let mut added = 0;

        // 1. Intent Recognition Router Head
        let intent_shape = vec![16, model.hidden_size];
        let intent_numel = 16 * model.hidden_size;
        new_tensors.push(TensorIR::new(
            "model.action_head.intent_router.weight".to_string(),
            intent_shape,
            "F16".to_string(),
            intent_numel * 2,
        ));

        // 2. Neuro-Symbolic Action Trajectory Actor Head
        let actor_shape = vec![num_action_outputs, model.hidden_size];
        let actor_numel = num_action_outputs * model.hidden_size;
        new_tensors.push(TensorIR::new(
            "model.action_head.actor_trajectory.weight".to_string(),
            actor_shape,
            "F16".to_string(),
            actor_numel * 2,
        ));

        // 3. Action Critic & Verification Scorer
        let critic_shape = vec![1, model.hidden_size];
        let critic_numel = model.hidden_size;
        new_tensors.push(TensorIR::new(
            "model.action_head.critic_evaluator.weight".to_string(),
            critic_shape,
            "F16".to_string(),
            critic_numel * 2,
        ));
        added += 3;

        Ok((new_tensors, added, 0))
    }

    /// Attaches SONAR concept segmentation and latent concept diffusion projection (LCM)
    fn augment_to_lcm(
        model: &ModelIR,
        concept_dim: usize,
    ) -> Result<(Vec<TensorIR>, usize, usize), String> {
        let mut new_tensors = model.tensors.clone();
        let mut added = 0;

        // 1. SONAR Concept Projector: [concept_dim, model.hidden_size]
        let proj_shape = vec![concept_dim, model.hidden_size];
        let proj_numel = concept_dim * model.hidden_size;
        new_tensors.push(TensorIR::new(
            "model.concept_projector.sonar.weight".to_string(),
            proj_shape,
            "F16".to_string(),
            proj_numel * 2,
        ));

        // 2. Latent Concept Diffusion Mixer: [concept_dim, concept_dim]
        let diff_shape = vec![concept_dim, concept_dim];
        let diff_numel = concept_dim * concept_dim;
        new_tensors.push(TensorIR::new(
            "model.concept_diffusion.backbone.weight".to_string(),
            diff_shape,
            "F16".to_string(),
            diff_numel * 2,
        ));

        // 3. Non-Autoregressive Concept Decoder
        let dec_shape = vec![model.vocab_size, concept_dim];
        let dec_numel = model.vocab_size * concept_dim;
        new_tensors.push(TensorIR::new(
            "model.concept_decoder.weight".to_string(),
            dec_shape,
            "F16".to_string(),
            dec_numel * 2,
        ));
        added += 3;

        Ok((new_tensors, added, 0))
    }

    /// Converts causal triangular attention to bidirectional Masked Language Model (MLM)
    fn convert_to_mlm(model: &ModelIR) -> Result<(Vec<TensorIR>, usize, usize), String> {
        let mut new_tensors = Vec::new();
        let mut restructured = 0;

        for t in &model.tensors {
            let mut mlm_t = t.clone();
            // Re-label causal LM head into masked prediction transformation head
            if matches!(mlm_t.role, TensorRole::OutputHead) {
                mlm_t.name = "cls.predictions.decoder.weight".to_string();
                restructured += 1;
            }
            new_tensors.push(mlm_t);
        }

        // Add Bidirectional Context Transformation Dense Layer
        let cls_shape = vec![model.hidden_size, model.hidden_size];
        let cls_numel = model.hidden_size * model.hidden_size;
        new_tensors.push(TensorIR::new(
            "cls.predictions.transform.dense.weight".to_string(),
            cls_shape,
            "F16".to_string(),
            cls_numel * 2,
        ));

        Ok((new_tensors, 1, restructured))
    }

    /// Converts multimodal model into a promptable Spatial Attention & Mask Segmenter (SAM)
    fn augment_to_sam(
        model: &ModelIR,
        vision_hidden_dim: usize,
    ) -> Result<(Vec<TensorIR>, usize, usize), String> {
        let (mut new_tensors, mut added, _) = Self::augment_to_vlm(model, vision_hidden_dim)?;

        // 1. Prompt Encoder (Points, Boxes, Text Prompts)
        let prompt_shape = vec![256, model.hidden_size];
        let prompt_numel = 256 * model.hidden_size;
        new_tensors.push(TensorIR::new(
            "model.prompt_encoder.pe_layer.weight".to_string(),
            prompt_shape,
            "F16".to_string(),
            prompt_numel * 2,
        ));

        // 2. Promptable Mask Decoder & IoU Token
        let mask_shape = vec![4, 256];
        new_tensors.push(TensorIR::new(
            "model.mask_decoder.mask_tokens.weight".to_string(),
            mask_shape,
            "F16".to_string(),
            4 * 256 * 2,
        ));

        let iou_shape = vec![1, 256];
        new_tensors.push(TensorIR::new(
            "model.mask_decoder.iou_token.weight".to_string(),
            iou_shape,
            "F16".to_string(),
            256 * 2,
        ));
        added += 3;

        Ok((new_tensors, added, 0))
    }

    /// Consolidates any specialized architecture back into standard monolithic dense LLM
    fn consolidate_to_llm(model: &ModelIR) -> Result<(Vec<TensorIR>, usize, usize), String> {
        let mut new_tensors = Vec::new();
        let mut restructured = 0;
        let mut seen_experts: HashMap<(usize, TensorRole), bool> = HashMap::new();

        for t in &model.tensors {
            // Drop specialized vision/action/concept/MLM accessories
            if t.name.contains("vision_tower")
                || t.name.contains("mm_projector")
                || t.name.contains("action_head")
                || t.name.contains("concept_")
                || t.name.contains("mask_decoder")
                || t.name.contains("prompt_encoder")
                || t.name.contains("router")
                || t.name.contains("cls.predictions.transform")
            {
                continue;
            }

            // Restore MLM decoder head back to causal lm_head
            if t.name.contains("cls.predictions.decoder") {
                let mut head_t = t.clone();
                head_t.name = "lm_head.weight".to_string();
                head_t.role = TensorRole::OutputHead;
                new_tensors.push(head_t);
                restructured += 1;
                continue;
            }

            // Consolidate expert tensors (keep expert 0 as base representative)
            if t.name.contains("experts.") {
                if let Some(l) = t.layer_index {
                    if seen_experts.contains_key(&(l, t.role)) {
                        continue;
                    }
                    seen_experts.insert((l, t.role), true);
                    let mut dense_t = t.clone();
                    let role_tag = match t.role {
                        TensorRole::MlpGate => "gate_proj",
                        TensorRole::MlpUp => "up_proj",
                        TensorRole::MlpDown => "down_proj",
                        _ => "mlp",
                    };
                    dense_t.name = format!("layers.{}.feed_forward.{}.weight", l, role_tag);
                    new_tensors.push(dense_t);
                    restructured += 1;
                    continue;
                }
            }

            let mut regular_t = t.clone();
            if matches!(regular_t.selected_format, TensorFormat::Ternary) {
                regular_t.selected_format = TensorFormat::FP16;
                restructured += 1;
            }
            new_tensors.push(regular_t);
        }

        Ok((new_tensors, 0, restructured))
    }
}
