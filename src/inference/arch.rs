// SPDX-License-Identifier: AGPL-3.0-only
//! Architecture dispatch for the reference forward pass.

use anyhow::{bail, Context, Result};
use std::path::Path;
use std::sync::Arc;

use super::gemma::{GemmaConfig, GemmaModel};
use super::gpu::GpuBackend;
use super::kv::{KvCache, KvPrecision};
use super::llama::{DepthExit, LlamaConfig, LlamaModel, WeightPrecision};
use super::neox::{NeoxConfig, NeoxModel};
use super::probe::ForwardTrace;
use crate::ir::{ModelArchitecture, ModelIR};

/// A causal language model the reference implementation can run.
pub trait CausalLm: Send + Sync {
    fn vocab_size(&self) -> usize;
    fn num_layers(&self) -> usize;

    /// Runs the layers named by `schedule` and reads out after each count in `exits`.
    fn forward_scheduled(
        &self,
        tokens: &[u32],
        schedule: &[usize],
        exits: &[usize],
    ) -> Result<Vec<DepthExit>>;

    /// Runs schedule once and returns the final readout with a per-layer trace.
    fn forward_traced(
        &self,
        tokens: &[u32],
        schedule: &[usize],
    ) -> Result<(DepthExit, ForwardTrace)>;

    /// Negative log-likelihood of each next token through the full layer stack.
    fn forward_nll(&self, tokens: &[u32]) -> Result<Vec<f32>> {
        let schedule: Vec<usize> = (0..self.num_layers()).collect();
        self.forward_scheduled(tokens, &schedule, &[schedule.len()])?
            .pop()
            .map(|exit| exit.nll)
            .context("Forward pass produced no readout")
    }

    /// Forward pass returning unnormalized logits of shape `[vocab_size]` for the final token position.
    fn forward_logits(&self, tokens: &[u32]) -> Result<Vec<f32>>;

    /// An empty KV cache for incremental decoding, or `None` when this architecture has no cached path.
    fn new_cache(&self) -> Option<KvCache> {
        None
    }

    /// An empty KV cache at `precision` for incremental decoding, or `None` when unsupported.
    fn new_cache_with_precision(&self, precision: KvPrecision) -> Option<KvCache> {
        if precision == KvPrecision::F32 {
            self.new_cache()
        } else {
            None
        }
    }

    /// An empty KV cache with separate key and value precisions, or `None` when unsupported.
    fn new_cache_with_precisions(&self, keys: KvPrecision, values: KvPrecision) -> Option<KvCache> {
        if keys == values {
            self.new_cache_with_precision(keys)
        } else {
            None
        }
    }

    /// Appends `tokens` after the positions in `cache` and returns the logits of the last one.
    /// Equal to `forward_logits` over the whole sequence; only architectures with a cache implement it.
    fn forward_extend(&self, _cache: &mut KvCache, _tokens: &[u32]) -> Result<Vec<f32>> {
        bail!("This architecture has no KV-cache path; use forward_logits")
    }
}

impl CausalLm for LlamaModel {
    fn vocab_size(&self) -> usize {
        LlamaModel::vocab_size(self)
    }
    fn num_layers(&self) -> usize {
        LlamaModel::num_layers(self)
    }
    fn forward_scheduled(
        &self,
        tokens: &[u32],
        schedule: &[usize],
        exits: &[usize],
    ) -> Result<Vec<DepthExit>> {
        LlamaModel::forward_scheduled(self, tokens, schedule, exits)
    }
    fn forward_traced(
        &self,
        tokens: &[u32],
        schedule: &[usize],
    ) -> Result<(DepthExit, ForwardTrace)> {
        LlamaModel::forward_traced(self, tokens, schedule)
    }
    fn forward_nll(&self, tokens: &[u32]) -> Result<Vec<f32>> {
        LlamaModel::forward_nll(self, tokens)
    }
    fn forward_logits(&self, tokens: &[u32]) -> Result<Vec<f32>> {
        LlamaModel::forward_logits(self, tokens)
    }
    fn new_cache(&self) -> Option<KvCache> {
        Some(LlamaModel::new_cache(self))
    }
    fn new_cache_with_precision(&self, precision: KvPrecision) -> Option<KvCache> {
        Some(LlamaModel::new_cache_with_precision(self, precision))
    }
    fn new_cache_with_precisions(&self, keys: KvPrecision, values: KvPrecision) -> Option<KvCache> {
        Some(LlamaModel::new_cache_with_precisions(self, keys, values))
    }
    fn forward_extend(&self, cache: &mut KvCache, tokens: &[u32]) -> Result<Vec<f32>> {
        LlamaModel::forward_extend(self, cache, tokens)
    }
}

impl CausalLm for NeoxModel {
    fn vocab_size(&self) -> usize {
        NeoxModel::vocab_size(self)
    }
    fn num_layers(&self) -> usize {
        NeoxModel::num_layers(self)
    }
    fn forward_scheduled(
        &self,
        tokens: &[u32],
        schedule: &[usize],
        exits: &[usize],
    ) -> Result<Vec<DepthExit>> {
        NeoxModel::forward_scheduled(self, tokens, schedule, exits)
    }
    fn forward_traced(
        &self,
        tokens: &[u32],
        schedule: &[usize],
    ) -> Result<(DepthExit, ForwardTrace)> {
        NeoxModel::forward_traced(self, tokens, schedule)
    }
    fn forward_nll(&self, tokens: &[u32]) -> Result<Vec<f32>> {
        NeoxModel::forward_nll(self, tokens)
    }
    fn forward_logits(&self, tokens: &[u32]) -> Result<Vec<f32>> {
        NeoxModel::forward_logits(self, tokens)
    }
    fn new_cache(&self) -> Option<KvCache> {
        Some(NeoxModel::new_cache(self))
    }
    fn new_cache_with_precision(&self, precision: KvPrecision) -> Option<KvCache> {
        Some(NeoxModel::new_cache_with_precision(self, precision))
    }
    fn new_cache_with_precisions(&self, keys: KvPrecision, values: KvPrecision) -> Option<KvCache> {
        Some(NeoxModel::new_cache_with_precisions(self, keys, values))
    }
    fn forward_extend(&self, cache: &mut KvCache, tokens: &[u32]) -> Result<Vec<f32>> {
        NeoxModel::forward_extend(self, cache, tokens)
    }
}

impl CausalLm for GemmaModel {
    fn vocab_size(&self) -> usize {
        GemmaModel::vocab_size(self)
    }
    fn num_layers(&self) -> usize {
        GemmaModel::num_layers(self)
    }
    fn forward_scheduled(
        &self,
        tokens: &[u32],
        schedule: &[usize],
        exits: &[usize],
    ) -> Result<Vec<DepthExit>> {
        GemmaModel::forward_scheduled(self, tokens, schedule, exits)
    }
    fn forward_traced(
        &self,
        tokens: &[u32],
        schedule: &[usize],
    ) -> Result<(DepthExit, ForwardTrace)> {
        GemmaModel::forward_traced(self, tokens, schedule)
    }
    fn forward_logits(&self, tokens: &[u32]) -> Result<Vec<f32>> {
        GemmaModel::forward_logits(self, tokens)
    }
}

/// Hyperparameters for one of the supported architecture families.
#[derive(Debug, Clone, PartialEq)]
pub enum ArchConfig {
    Llama(LlamaConfig),
    NeoX(NeoxConfig),
    Gemma(GemmaConfig),
}

impl ArchConfig {
    /// Detects the architecture from tensor names and reads `config.json` beside the weights.
    pub fn detect(model: &ModelIR, model_path: &Path) -> Result<Self> {
        let has = |fragment: &str| model.tensors.iter().any(|t| t.name.contains(fragment));
        if model.architecture == crate::ir::ModelArchitecture::Gemma
            || (has("attn_q") && (has("attn_q_norm") || has("post_attention_norm")))
        {
            Ok(Self::Gemma(
                GemmaConfig::from_model_ir(model).with_hf_config(model_path)?,
            ))
        } else if has("query_key_value") && has("dense_h_to_4h") {
            Ok(Self::NeoX(
                NeoxConfig::from_model_ir(model).with_hf_config(model_path)?,
            ))
        } else if (has("q_proj") && has("gate_proj"))
            || (has("attn_q") && has("ffn_gate"))
            || matches!(
                model.architecture,
                crate::ir::ModelArchitecture::Llama
                    | crate::ir::ModelArchitecture::Mistral
                    | crate::ir::ModelArchitecture::Qwen
            )
        {
            Ok(Self::Llama(
                LlamaConfig::from_model_ir(model).with_hf_config(model_path)?,
            ))
        } else {
            bail!(
                "Unsupported architecture for the reference forward pass; supported families are LLaMA (q_proj/gate_proj), GPT-NeoX (query_key_value/dense_h_to_4h), and Gemma (attn_q/attn_q_norm)"
            )
        }
    }

    pub fn family(&self) -> &'static str {
        match self {
            Self::Llama(_) => "LLaMA",
            Self::NeoX(_) => "GPT-NeoX",
            Self::Gemma(_) => "Gemma",
        }
    }

    /// Loads `model`'s weights into a runnable model of this architecture.
    pub fn load(&self, model: &ModelIR, precision: WeightPrecision) -> Result<Box<dyn CausalLm>> {
        self.load_with_gpu(model, precision, None)
    }

    /// Loads `model`'s weights into a runnable model with optional GPU compute acceleration.
    pub fn load_with_gpu(
        &self,
        model: &ModelIR,
        precision: WeightPrecision,
        gpu: Option<Arc<dyn GpuBackend>>,
    ) -> Result<Box<dyn CausalLm>> {
        Ok(match self {
            Self::Llama(config) => {
                let mut m = LlamaModel::load(model, config.clone(), precision)?;
                if let Some(backend) = gpu {
                    m.set_gpu(Some(backend));
                }
                Box::new(m)
            }
            Self::NeoX(config) => {
                let mut m = NeoxModel::load(model, config.clone(), precision)?;
                if let Some(backend) = gpu {
                    m.set_gpu(Some(backend));
                }
                Box::new(m)
            }
            Self::Gemma(config) => {
                let mut m = GemmaModel::load(model, config.clone(), precision)?;
                if let Some(backend) = gpu {
                    m = m.with_gpu(backend);
                }
                Box::new(m)
            }
        })
    }
}

/// Imports a model file or folder and loads its weights as stored, the path `thme forward` takes.
pub fn load_reference_model(
    path: &Path,
    device: super::gpu::DeviceArg,
) -> Result<(ModelIR, ArchConfig, Box<dyn CausalLm>)> {
    let model_ir = crate::parser::ModelImporter::import(path)
        .with_context(|| format!("Failed to import model {path:?}"))?;
    let arch_cfg = ArchConfig::detect(&model_ir, path)
        .with_context(|| format!("Failed to detect architecture for {path:?}"))?;
    let gpu = device.resolve();
    let runnable = arch_cfg
        .load_with_gpu(&model_ir, WeightPrecision::AsStored, gpu)
        .context("Failed to load runnable model weights")?;
    Ok((model_ir, arch_cfg, runnable))
}

/// Architecture family read from tensor names (v0.2.9.4, WP17).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InferredFamily {
    Llama,
    NeoX,
    Gemma,
    Unknown,
}

/// How sure an inference is about its family.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Confidence {
    /// Every tensor the family's forward pass reads in layer 0 is present.
    High,
    /// The family's distinctive tensors are present, but some required ones are missing.
    Medium,
    /// Only the architecture name recorded in the model points at the family.
    Low,
    /// No evidence for any family.
    None,
}

/// A family guess with its confidence and the evidence behind it, one line each.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArchInference {
    pub family: InferredFamily,
    pub confidence: Confidence,
    pub evidence: Vec<String>,
}

/// Infers the family from tensor names alone, without reading any weight or `config.json`.
/// Shapes cannot give the head count (any divisor of the hidden width fits), so the evidence
/// always says so.
pub fn infer(model: &ModelIR) -> ArchInference {
    let has = |fragment: &str| model.tensors.iter().any(|t| t.name.contains(fragment));
    let mut evidence = vec![
        "attention head count is not readable from tensor shapes; take it from config.json".into(),
    ];
    for sig in SIGNATURES {
        let marker = sig.markers.iter().find(|m| m.iter().all(|f| has(f)));
        let Some(marker) = marker else { continue };
        evidence.push(format!("found {}", marker.join(" and ")));
        // Judge completeness against the naming scheme that is most complete.
        let missing = sig
            .schemes
            .iter()
            .map(|scheme| {
                scheme
                    .iter()
                    .copied()
                    .filter(|f| !has(f))
                    .collect::<Vec<_>>()
            })
            .min_by_key(Vec::len)
            .unwrap_or_default();
        let confidence = if missing.is_empty() {
            Confidence::High
        } else {
            evidence.push(format!("missing layer-0 tensors: {}", missing.join(", ")));
            Confidence::Medium
        };
        return ArchInference {
            family: sig.family,
            confidence,
            evidence,
        };
    }
    let from_name = match model.architecture {
        ModelArchitecture::Gemma => Some(InferredFamily::Gemma),
        ModelArchitecture::Llama | ModelArchitecture::Mistral | ModelArchitecture::Qwen => {
            Some(InferredFamily::Llama)
        }
        _ => None,
    };
    match from_name {
        Some(family) => {
            evidence.push(format!(
                "no tensor evidence; the recorded architecture is {}",
                model.architecture.display_name()
            ));
            ArchInference {
                family,
                confidence: Confidence::Low,
                evidence,
            }
        }
        None => ArchInference {
            family: InferredFamily::Unknown,
            confidence: Confidence::None,
            evidence,
        },
    }
}

/// Tensor names that mark a family, and the layer-0 tensors each naming scheme needs.
struct Signature {
    family: InferredFamily,
    /// Any one of these alternatives, all of whose fragments must appear, marks the family.
    markers: &'static [&'static [&'static str]],
    /// Each naming scheme lists the fragments one complete layout must contain.
    schemes: &'static [&'static [&'static str]],
}

/// Checked in the same order as [`ArchConfig::detect`], so Gemma's GGUF names win over LLaMA's.
const SIGNATURES: [Signature; 3] = [
    Signature {
        family: InferredFamily::Gemma,
        markers: &[
            &["attn_q_norm"],
            &["post_attention_norm"],
            &["post_ffw_norm"],
        ],
        schemes: &[&[
            "token_embd",
            "blk.0.attn_norm",
            "blk.0.attn_q.",
            "blk.0.attn_q_norm",
            "blk.0.attn_k.",
            "blk.0.attn_k_norm",
            "blk.0.attn_v.",
            "blk.0.attn_output",
            "blk.0.post_attention_norm",
            "blk.0.ffn_norm",
            "blk.0.ffn_gate",
            "blk.0.ffn_up",
            "blk.0.ffn_down",
            "blk.0.post_ffw_norm",
            "output_norm",
        ]],
    },
    Signature {
        family: InferredFamily::NeoX,
        markers: &[&["query_key_value", "dense_h_to_4h"]],
        schemes: &[&[
            "embed_in",
            "layers.0.input_layernorm",
            "layers.0.post_attention_layernorm",
            "layers.0.attention.query_key_value",
            "layers.0.attention.dense.",
            "layers.0.mlp.dense_h_to_4h",
            "layers.0.mlp.dense_4h_to_h",
            "final_layer_norm",
        ]],
    },
    Signature {
        family: InferredFamily::Llama,
        markers: &[&["q_proj", "gate_proj"], &["attn_q", "ffn_gate"]],
        schemes: &[
            &[
                "model.embed_tokens",
                "layers.0.input_layernorm",
                "layers.0.self_attn.q_proj",
                "layers.0.self_attn.k_proj",
                "layers.0.self_attn.v_proj",
                "layers.0.self_attn.o_proj",
                "layers.0.post_attention_layernorm",
                "layers.0.mlp.gate_proj",
                "layers.0.mlp.up_proj",
                "layers.0.mlp.down_proj",
                "model.norm",
            ],
            &[
                "token_embd",
                "blk.0.attn_norm",
                "blk.0.attn_q.",
                "blk.0.attn_k.",
                "blk.0.attn_v.",
                "blk.0.attn_output",
                "blk.0.ffn_norm",
                "blk.0.ffn_gate",
                "blk.0.ffn_up",
                "blk.0.ffn_down",
                "output_norm",
            ],
        ],
    },
];

/// Imports a model file and infers its family. Refuses unless the model has a permission
/// record (see [`crate::permission::require`]); reverse-engineering features go through here.
pub fn infer_from_file(path: &Path) -> Result<ArchInference> {
    crate::permission::require(path)?;
    let model = crate::parser::ModelImporter::import(path)
        .with_context(|| format!("Failed to import model {path:?}"))?;
    Ok(infer(&model))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{ModelArchitecture, TensorIR};

    struct F32OnlyCacheModel;

    impl CausalLm for F32OnlyCacheModel {
        fn vocab_size(&self) -> usize {
            8
        }
        fn num_layers(&self) -> usize {
            1
        }
        fn forward_scheduled(
            &self,
            _tokens: &[u32],
            _schedule: &[usize],
            _exits: &[usize],
        ) -> Result<Vec<DepthExit>> {
            Ok(Vec::new())
        }
        fn forward_traced(
            &self,
            _tokens: &[u32],
            _schedule: &[usize],
        ) -> Result<(DepthExit, ForwardTrace)> {
            bail!("unused in test")
        }
        fn forward_logits(&self, _tokens: &[u32]) -> Result<Vec<f32>> {
            Ok(vec![0.0; 8])
        }
        fn new_cache(&self) -> Option<KvCache> {
            Some(KvCache::new(1, 4))
        }
    }

    #[test]
    fn default_new_cache_with_precision_allows_only_f32() {
        let model = F32OnlyCacheModel;
        assert!(model.new_cache_with_precision(KvPrecision::F32).is_some());
        assert!(model.new_cache_with_precision(KvPrecision::Int8).is_none());
        assert!(model.new_cache_with_precision(KvPrecision::Int4).is_none());
    }

    #[test]
    fn default_new_cache_with_precisions_returns_none_for_mixed_precisions() {
        let model = F32OnlyCacheModel;
        assert!(model
            .new_cache_with_precisions(KvPrecision::F32, KvPrecision::F32)
            .is_some());
        assert!(model
            .new_cache_with_precisions(KvPrecision::F32, KvPrecision::Int8)
            .is_none());
        assert!(model
            .new_cache_with_precisions(KvPrecision::Int8, KvPrecision::F32)
            .is_none());
        assert!(model
            .new_cache_with_precisions(KvPrecision::Int8, KvPrecision::Int4)
            .is_none());
        assert!(model
            .new_cache_with_precisions(KvPrecision::Int8, KvPrecision::Int8)
            .is_none());
    }

    /// Names only: inference never reads shapes or data, so every tensor gets shape `[1]`.
    fn model_named(names: &[&str], architecture: ModelArchitecture) -> ModelIR {
        let tensors = names
            .iter()
            .map(|n| TensorIR::new(n.to_string(), vec![1], "F32".into(), 4))
            .collect();
        ModelIR::new("m".into(), architecture, 1, 16, 4, 2, 40, 128, tensors)
    }

    /// Hugging Face LLaMA layout for one layer, written out by hand.
    const LLAMA_HF: &[&str] = &[
        "model.embed_tokens.weight",
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.v_proj.weight",
        "model.layers.0.self_attn.o_proj.weight",
        "model.layers.0.post_attention_layernorm.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.mlp.up_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
        "model.norm.weight",
        "lm_head.weight",
    ];

    /// GGUF Gemma layout for one block, as the Gemma loader reads it.
    const GEMMA_GGUF: &[&str] = &[
        "token_embd.weight",
        "blk.0.attn_norm.weight",
        "blk.0.attn_q.weight",
        "blk.0.attn_q_norm.weight",
        "blk.0.attn_k.weight",
        "blk.0.attn_k_norm.weight",
        "blk.0.attn_v.weight",
        "blk.0.attn_output.weight",
        "blk.0.post_attention_norm.weight",
        "blk.0.ffn_norm.weight",
        "blk.0.ffn_gate.weight",
        "blk.0.ffn_up.weight",
        "blk.0.ffn_down.weight",
        "blk.0.post_ffw_norm.weight",
        "output_norm.weight",
    ];

    /// Hugging Face GPT-NeoX layout for one layer.
    const NEOX_HF: &[&str] = &[
        "embed_in.weight",
        "layers.0.input_layernorm.weight",
        "layers.0.post_attention_layernorm.weight",
        "layers.0.attention.query_key_value.weight",
        "layers.0.attention.dense.weight",
        "layers.0.mlp.dense_h_to_4h.weight",
        "layers.0.mlp.dense_4h_to_h.weight",
        "final_layer_norm.weight",
        "embed_out.weight",
    ];

    #[test]
    fn full_llama_layout_is_llama_with_high_confidence() {
        let m = model_named(LLAMA_HF, ModelArchitecture::Llama);
        let r = infer(&m);
        assert_eq!(r.family, InferredFamily::Llama);
        assert_eq!(r.confidence, Confidence::High);
    }

    #[test]
    fn llama_missing_o_proj_is_medium_and_names_the_gap() {
        let names: Vec<&str> = LLAMA_HF
            .iter()
            .copied()
            .filter(|n| !n.contains("o_proj"))
            .collect();
        let r = infer(&model_named(&names, ModelArchitecture::Llama));
        assert_eq!(r.family, InferredFamily::Llama);
        assert_eq!(r.confidence, Confidence::Medium);
        assert!(r.evidence.iter().any(|e| e.contains("self_attn.o_proj")));
    }

    #[test]
    fn full_gguf_gemma_layout_is_gemma_with_high_confidence() {
        let r = infer(&model_named(
            GEMMA_GGUF,
            ModelArchitecture::Unknown("gemma".into()),
        ));
        assert_eq!(r.family, InferredFamily::Gemma);
        assert_eq!(r.confidence, Confidence::High);
    }

    #[test]
    fn full_neox_layout_is_neox_with_high_confidence() {
        let r = infer(&model_named(
            NEOX_HF,
            ModelArchitecture::Unknown("gpt_neox".into()),
        ));
        assert_eq!(r.family, InferredFamily::NeoX);
        assert_eq!(r.confidence, Confidence::High);
    }

    #[test]
    fn gguf_llama_layout_is_llama_not_gemma() {
        // GGUF LLaMA shares attn_q and ffn_gate with Gemma but has no Gemma-only norms.
        let names = [
            "token_embd.weight",
            "blk.0.attn_norm.weight",
            "blk.0.attn_q.weight",
            "blk.0.attn_k.weight",
            "blk.0.attn_v.weight",
            "blk.0.attn_output.weight",
            "blk.0.ffn_norm.weight",
            "blk.0.ffn_gate.weight",
            "blk.0.ffn_up.weight",
            "blk.0.ffn_down.weight",
            "output_norm.weight",
        ];
        let r = infer(&model_named(&names, ModelArchitecture::Llama));
        assert_eq!(r.family, InferredFamily::Llama);
        assert_eq!(r.confidence, Confidence::High);
    }

    #[test]
    fn architecture_name_alone_is_low_confidence() {
        let r = infer(&model_named(&["foo.weight"], ModelArchitecture::Gemma));
        assert_eq!(r.family, InferredFamily::Gemma);
        assert_eq!(r.confidence, Confidence::Low);
    }

    #[test]
    fn no_evidence_is_unknown_with_no_confidence() {
        let r = infer(&model_named(
            &["foo.weight"],
            ModelArchitecture::Unknown("x".into()),
        ));
        assert_eq!(r.family, InferredFamily::Unknown);
        assert_eq!(r.confidence, Confidence::None);
    }

    #[test]
    fn every_inference_says_head_count_is_not_in_the_shapes() {
        let r = infer(&model_named(LLAMA_HF, ModelArchitecture::Llama));
        assert!(r.evidence.iter().any(|e| e.contains("head count")));
    }

    #[test]
    fn inference_agrees_with_the_forward_pass_detector() {
        let dir = tempfile::tempdir().unwrap();
        let cases = [
            (LLAMA_HF, ModelArchitecture::Llama, "LLaMA"),
            (
                NEOX_HF,
                ModelArchitecture::Unknown("gpt_neox".into()),
                "GPT-NeoX",
            ),
            (
                GEMMA_GGUF,
                ModelArchitecture::Unknown("gemma".into()),
                "Gemma",
            ),
        ];
        for (names, arch, family) in cases {
            let m = model_named(names, arch);
            let detected = ArchConfig::detect(&m, dir.path()).unwrap().family();
            assert_eq!(detected, family);
            assert_ne!(infer(&m).family, InferredFamily::Unknown);
        }
    }

    #[test]
    fn inference_from_file_is_refused_without_a_permission_record() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("m.bin");
        std::fs::write(&path, b"abc").unwrap();
        let err = infer_from_file(&path).unwrap_err();
        assert!(
            format!("{err:#}").contains("no permission record"),
            "{err:#}"
        );
    }

    #[test]
    fn inference_from_file_checks_permission_before_importing() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("m.bin");
        std::fs::write(&path, b"abc").unwrap();
        crate::permission::grant(&path, crate::permission::Basis::Owner, "unit test", 1).unwrap();
        // The bytes are not a model, so the import fails, but not on the permission check.
        let err = infer_from_file(&path).unwrap_err();
        assert!(!format!("{err:#}").contains("permission record"), "{err:#}");
    }
}
