// SPDX-License-Identifier: AGPL-3.0-only
//! Architecture dispatch for the reference forward pass.

use anyhow::{bail, Context, Result};
use std::path::Path;
use std::sync::Arc;

use super::gemma::{GemmaConfig, GemmaModel};
use super::gpu::GpuBackend;
use super::llama::{DepthExit, LlamaConfig, LlamaModel, WeightPrecision};
use super::neox::{NeoxConfig, NeoxModel};
use super::probe::ForwardTrace;
use crate::ir::ModelIR;

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
    fn forward_logits(&self, tokens: &[u32]) -> Result<Vec<f32>> {
        LlamaModel::forward_logits(self, tokens)
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
    fn forward_logits(&self, tokens: &[u32]) -> Result<Vec<f32>> {
        NeoxModel::forward_logits(self, tokens)
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
            Self::NeoX(config) => Box::new(NeoxModel::load(model, config.clone(), precision)?),
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
