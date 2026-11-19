// SPDX-License-Identifier: AGPL-3.0-only
//! Gemma-family (Gemma 2 / Gemma 3) causal language model reference forward pass.
//!
//! Implements embedding scaling by `sqrt(hidden_size)`, Grouped Query Attention (GQA),
//! per-head QK-RMSNorm, dual residual layernorms (post-attention and post-FFW),
//! GeGLU feedforward activation with `gelu_approx`, and tied embedding readout.

use anyhow::{bail, Context, Result};
use std::path::Path;
use std::sync::Arc;

use super::gpu::{dispatch_linear_slice, dispatch_score_head_slice, GpuBackend};
use super::llama::{
    add_in_place, attention, load_tensor, rms_norm, AttentionInputs, DepthExit, RopeTable,
    WeightPrecision,
};
use super::probe::{
    dead_fraction, rms, update_ratio, ForwardTrace, InputChannelStats, LayerTrace, TensorActivity,
};
use crate::ir::{ModelIR, TensorIR};

/// Architectural hyperparameters for the Gemma forward pass.
#[derive(Debug, Clone, PartialEq)]
pub struct GemmaConfig {
    pub num_layers: usize,
    pub hidden_size: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub rope_scaling_type: Option<String>,
    pub rope_scaling_factor: Option<f32>,
    pub embed_scale: f32,
}

impl GemmaConfig {
    /// Builds a Gemma configuration from `ModelIR`.
    pub fn from_model_ir(model: &ModelIR) -> Self {
        let d = model.hidden_size;
        let num_heads = model.num_attention_heads.max(1);
        let head_dim = if model.head_dim > 0 {
            model.head_dim
        } else {
            d / num_heads
        };

        // Resolve intermediate size from ffn_gate tensor if present, else default
        let intermediate_size = model
            .tensors
            .iter()
            .find(|t| t.name.contains("ffn_gate") || t.name.contains("gate_proj"))
            .map(|t| {
                if t.shape.len() >= 2 && t.shape[0] > 0 {
                    t.shape[0]
                } else if d > 0 && t.numel > d {
                    t.numel / d
                } else {
                    10240
                }
            })
            .unwrap_or(10240);

        let embed_scale = (d as f32).sqrt();

        Self {
            num_layers: model.num_layers,
            hidden_size: d,
            num_heads,
            num_kv_heads: model.num_key_value_heads.max(1),
            head_dim,
            intermediate_size,
            rms_norm_eps: model.norm_eps.unwrap_or(1e-6),
            rope_theta: model.rope_theta.unwrap_or(1_000_000.0),
            rope_scaling_type: model.rope_scaling_type.clone(),
            rope_scaling_factor: model.rope_scaling_factor,
            embed_scale,
        }
    }

    /// Overrides configuration parameters from a sibling `config.json` when present.
    pub fn with_hf_config(mut self, model_path: &Path) -> Result<Self> {
        if crate::epiph::is_epiph(model_path) {
            return Ok(self);
        }
        let dir = if model_path.is_file() {
            model_path.parent().unwrap_or_else(|| Path::new("."))
        } else {
            model_path
        };
        let path = dir.join("config.json");
        if !path.exists() {
            return Ok(self);
        }
        let text =
            std::fs::read_to_string(&path).with_context(|| format!("Failed to read {path:?}"))?;
        let value: serde_json::Value =
            serde_json::from_str(&text).with_context(|| format!("Failed to parse {path:?}"))?;

        if let Some(eps) = value
            .get("rms_norm_eps")
            .and_then(serde_json::Value::as_f64)
        {
            self.rms_norm_eps = eps as f32;
        }
        if let Some(theta) = value.get("rope_theta").and_then(serde_json::Value::as_f64) {
            self.rope_theta = theta as f32;
        }
        if let Some(hd) = value.get("head_dim").and_then(serde_json::Value::as_u64) {
            self.head_dim = hd as usize;
        }
        if let Some(scaling) = value.get("rope_scaling") {
            if let Some(factor) = scaling.get("factor").and_then(serde_json::Value::as_f64) {
                self.rope_scaling_factor = Some(factor as f32);
            }
            if let Some(t) = scaling
                .get("type")
                .or_else(|| scaling.get("rope_type"))
                .and_then(serde_json::Value::as_str)
            {
                self.rope_scaling_type = Some(t.to_string());
            }
        }
        Ok(self)
    }
}

/// Approximate GELU with tanh: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
#[inline]
pub fn gelu_approx(x: f32) -> f32 {
    let sqrt_2_over_pi = 0.797_884_6_f32;
    let coeff = 0.044_715_f32;
    0.5 * x * (1.0 + (sqrt_2_over_pi * (x + coeff * x * x * x)).tanh())
}

/// Normalizes row chunks of length `weight.len()` in place using RMSNorm weights.
#[inline]
pub fn apply_rms_norm_in_place(x: &mut [f32], weight: &[f32], eps: f32) {
    let d = weight.len();
    if d == 0 {
        return;
    }
    for row in x.chunks_exact_mut(d) {
        let mean_square = row.iter().map(|&v| v * v).sum::<f32>() / d as f32;
        let scale = 1.0 / (mean_square + eps).sqrt();
        for (v, &w) in row.iter_mut().zip(weight) {
            *v = *v * scale * w;
        }
    }
}

#[derive(Clone)]
pub(crate) struct GemmaLayer {
    pub(crate) attn_norm: Vec<f32>,
    pub(crate) q: Vec<f32>,
    pub(crate) q_norm: Vec<f32>,
    pub(crate) k: Vec<f32>,
    pub(crate) k_norm: Vec<f32>,
    pub(crate) v: Vec<f32>,
    pub(crate) o: Vec<f32>,
    pub(crate) post_attention_norm: Vec<f32>,
    pub(crate) ffn_norm: Vec<f32>,
    pub(crate) gate: Vec<f32>,
    pub(crate) up: Vec<f32>,
    pub(crate) down: Vec<f32>,
    pub(crate) post_ffw_norm: Vec<f32>,
}

/// In-memory or streaming Gemma model implementing the `CausalLm` forward pass.
pub struct GemmaModel {
    pub(crate) config: GemmaConfig,
    pub(crate) vocab_size: usize,
    pub(crate) embed: Vec<f32>,
    pub(crate) layers: Vec<GemmaLayer>,
    pub(crate) final_norm: Vec<f32>,
    pub(crate) lm_head: Option<Vec<f32>>,
    pub(crate) model_ir: Option<ModelIR>,
    pub(crate) precision: WeightPrecision,
    pub(crate) gpu_backend: Option<Arc<dyn GpuBackend>>,
}

/// Tensor lookup with bidirectional GGUF and Hugging Face naming resolution.
pub fn find_tensor<'a>(model: &'a ModelIR, name: &str) -> Option<&'a TensorIR> {
    if let Some(t) = model.tensors.iter().find(|t| t.name == name) {
        return Some(t);
    }

    let candidate: Vec<String> =
        if name == "token_embd.weight" || name == "model.embed_tokens.weight" {
            vec![
                "token_embd.weight".to_string(),
                "model.embed_tokens.weight".to_string(),
            ]
        } else if name == "output_norm.weight" || name == "model.norm.weight" {
            vec![
                "output_norm.weight".to_string(),
                "model.norm.weight".to_string(),
            ]
        } else if name.starts_with("blk.") {
            let parts: Vec<&str> = name.split('.').collect();
            if parts.len() >= 3 {
                let i = parts[1];
                let suffix = parts[2..].join(".");
                let hf_suffix = match suffix.as_str() {
                    "attn_norm.weight" => "input_layernorm.weight",
                    "attn_q.weight" => "self_attn.q_proj.weight",
                    "attn_q_norm.weight" => "self_attn.q_norm.weight",
                    "attn_k.weight" => "self_attn.k_proj.weight",
                    "attn_k_norm.weight" => "self_attn.k_norm.weight",
                    "attn_v.weight" => "self_attn.v_proj.weight",
                    "attn_output.weight" => "self_attn.o_proj.weight",
                    "post_attention_norm.weight" => "post_attention_layernorm.weight",
                    "ffn_norm.weight" => "pre_feedforward_layernorm.weight",
                    "ffn_gate.weight" => "mlp.gate_proj.weight",
                    "ffn_up.weight" => "mlp.up_proj.weight",
                    "ffn_down.weight" => "mlp.down_proj.weight",
                    "post_ffw_norm.weight" => "post_feedforward_layernorm.weight",
                    _ => "",
                };
                if !hf_suffix.is_empty() {
                    vec![name.to_string(), format!("model.layers.{i}.{hf_suffix}")]
                } else {
                    vec![name.to_string()]
                }
            } else {
                vec![name.to_string()]
            }
        } else if name.starts_with("model.layers.") {
            let parts: Vec<&str> = name.split('.').collect();
            if parts.len() >= 4 {
                let i = parts[2];
                let suffix = parts[3..].join(".");
                let gguf_suffix = match suffix.as_str() {
                    "input_layernorm.weight" => "attn_norm.weight",
                    "self_attn.q_proj.weight" => "attn_q.weight",
                    "self_attn.q_norm.weight" => "attn_q_norm.weight",
                    "self_attn.k_proj.weight" => "attn_k.weight",
                    "self_attn.k_norm.weight" => "attn_k_norm.weight",
                    "self_attn.v_proj.weight" => "attn_v.weight",
                    "self_attn.o_proj.weight" => "attn_output.weight",
                    "post_attention_layernorm.weight" => "post_attention_norm.weight",
                    "pre_feedforward_layernorm.weight" | "ffn_norm.weight" => "ffn_norm.weight",
                    "mlp.gate_proj.weight" => "ffn_gate.weight",
                    "mlp.up_proj.weight" => "ffn_up.weight",
                    "mlp.down_proj.weight" => "ffn_down.weight",
                    "post_feedforward_layernorm.weight" | "post_ffw_norm.weight" => {
                        "post_ffw_norm.weight"
                    }
                    _ => "",
                };
                if !gguf_suffix.is_empty() {
                    vec![name.to_string(), format!("blk.{i}.{gguf_suffix}")]
                } else {
                    vec![name.to_string()]
                }
            } else {
                vec![name.to_string()]
            }
        } else {
            vec![name.to_string()]
        };

    for cand in &candidate {
        let cand_dotted = format!(".{cand}");
        if let Some(t) = model
            .tensors
            .iter()
            .find(|t| t.name == *cand || t.name.ends_with(&cand_dotted))
        {
            return Some(t);
        }
    }

    None
}

impl GemmaModel {
    /// Loads a Gemma model from the IR.
    ///
    /// # Memory Architecture
    /// - **Persistent Embedding Matrix**: Full vocabulary table (262,208 * 2,560 * 4 ≈ 2.685 GB FP32)
    ///   is kept resident in RAM to serve token lookup and tied logit projections.
    /// - **Streaming Transformer Layers**: Rather than loading all 34 dense layers (~15.6 GB RAM),
    ///   individual layer blocks (~378 MB FP32) are decoded on-the-fly and dropped after execution.
    /// - **Peak Working Heap**: ~3.06 to 3.11 GB RAM during forward pass execution.
    pub fn load(model: &ModelIR, config: GemmaConfig, precision: WeightPrecision) -> Result<Self> {
        let d = config.hidden_size;
        if d == 0 || config.num_heads == 0 || config.num_kv_heads == 0 || config.num_layers == 0 {
            bail!("Gemma configuration has a zero dimension: {config:?}");
        }

        let required = |name: &str| -> Result<&TensorIR> {
            find_tensor(model, name)
                .with_context(|| format!("Gemma model is missing required tensor '{name}'"))
        };

        let embed_tensor = required("token_embd.weight")?;
        if embed_tensor.numel == 0 || !embed_tensor.numel.is_multiple_of(d) {
            bail!(
                "Embedding tensor '{}' has {} values, not a multiple of hidden size {d}",
                embed_tensor.name,
                embed_tensor.numel
            );
        }
        let vocab_size = embed_tensor.numel / d;
        let embed = load_tensor(embed_tensor, vocab_size * d, precision)?;

        let final_norm_tensor = required("output_norm.weight")?;
        let final_norm = load_tensor(final_norm_tensor, d, precision)?;

        let lm_head = match find_tensor(model, "lm_head.weight") {
            Some(t) if t.name != embed_tensor.name => {
                Some(load_tensor(t, vocab_size * d, precision)?)
            }
            _ => None,
        };

        // Validate presence of all layer tensors
        for index in 0..config.num_layers {
            let _ = required(&format!("blk.{index}.attn_norm.weight"))?;
            let _ = required(&format!("blk.{index}.attn_q.weight"))?;
            let _ = required(&format!("blk.{index}.attn_q_norm.weight"))?;
            let _ = required(&format!("blk.{index}.attn_k.weight"))?;
            let _ = required(&format!("blk.{index}.attn_k_norm.weight"))?;
            let _ = required(&format!("blk.{index}.attn_v.weight"))?;
            let _ = required(&format!("blk.{index}.attn_output.weight"))?;
            let _ = required(&format!("blk.{index}.post_attention_norm.weight"))?;
            let _ = required(&format!("blk.{index}.ffn_norm.weight"))?;
            let _ = required(&format!("blk.{index}.ffn_gate.weight"))?;
            let _ = required(&format!("blk.{index}.ffn_up.weight"))?;
            let _ = required(&format!("blk.{index}.ffn_down.weight"))?;
            let _ = required(&format!("blk.{index}.post_ffw_norm.weight"))?;
        }

        Ok(Self {
            config,
            vocab_size,
            embed,
            layers: Vec::new(),
            final_norm,
            lm_head,
            model_ir: Some(model.clone()),
            precision,
            gpu_backend: None,
        })
    }

    pub fn with_gpu(mut self, backend: Arc<dyn GpuBackend>) -> Self {
        self.gpu_backend = Some(backend);
        self
    }

    pub fn set_gpu(&mut self, backend: Option<Arc<dyn GpuBackend>>) {
        self.gpu_backend = backend;
    }

    pub fn gpu_backend(&self) -> Option<&dyn GpuBackend> {
        self.gpu_backend.as_deref()
    }

    /// Loads a specific layer on demand (or from memory).
    pub(crate) fn get_layer(&self, index: usize) -> Result<GemmaLayer> {
        if !self.layers.is_empty() && index < self.layers.len() {
            return Ok(self.layers[index].clone());
        }

        let model = self
            .model_ir
            .as_ref()
            .context("Gemma model has no active ModelIR to load layers from")?;

        let required = |name: &str| -> Result<&TensorIR> {
            find_tensor(model, name)
                .with_context(|| format!("Gemma model is missing required tensor '{name}'"))
        };

        let d = self.config.hidden_size;
        let q_dim = self.config.num_heads * self.config.head_dim;
        let kv_dim = self.config.num_kv_heads * self.config.head_dim;
        let inter = self.config.intermediate_size;
        let hd = self.config.head_dim;
        let precision = self.precision;

        let fetch = |suffix: &str, expected: usize| -> Result<Vec<f32>> {
            let name = format!("blk.{index}.{suffix}");
            load_tensor(required(&name)?, expected, precision)
        };

        Ok(GemmaLayer {
            attn_norm: fetch("attn_norm.weight", d)?,
            q: fetch("attn_q.weight", q_dim * d)?,
            q_norm: fetch("attn_q_norm.weight", hd)?,
            k: fetch("attn_k.weight", kv_dim * d)?,
            k_norm: fetch("attn_k_norm.weight", hd)?,
            v: fetch("attn_v.weight", kv_dim * d)?,
            o: fetch("attn_output.weight", d * q_dim)?,
            post_attention_norm: fetch("post_attention_norm.weight", d)?,
            ffn_norm: fetch("ffn_norm.weight", d)?,
            gate: fetch("ffn_gate.weight", inter * d)?,
            up: fetch("ffn_up.weight", inter * d)?,
            down: fetch("ffn_down.weight", d * inter)?,
            post_ffw_norm: fetch("post_ffw_norm.weight", d)?,
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    pub fn num_layers(&self) -> usize {
        self.config.num_layers
    }

    fn validate(&self, tokens: &[u32], schedule: &[usize], exits: &[usize]) -> Result<()> {
        let t_len = tokens.len();
        if t_len < 2 {
            bail!("A forward pass needs at least 2 tokens, received {t_len}");
        }
        if let Some(bad) = tokens.iter().find(|&&t| t as usize >= self.vocab_size) {
            bail!(
                "Token id {bad} is outside the model vocabulary of {}",
                self.vocab_size
            );
        }
        if let Some(bad) = schedule.iter().find(|&&l| l >= self.num_layers()) {
            bail!(
                "Schedule names layer {bad}, but the model has {} layers",
                self.num_layers()
            );
        }
        if let Some(bad) = exits.iter().find(|&&e| e > schedule.len()) {
            bail!(
                "Exit after {bad} layers is beyond the schedule length {}",
                schedule.len()
            );
        }
        Ok(())
    }

    /// Runs one causal forward pass and returns the negative log-likelihood (nats).
    pub fn forward_nll(&self, tokens: &[u32]) -> Result<Vec<f32>> {
        let schedule: Vec<usize> = (0..self.num_layers()).collect();
        let mut exits = self.forward_scheduled(tokens, &schedule, &[schedule.len()])?;
        match exits.pop() {
            Some(exit) => Ok(exit.nll),
            None => bail!("Forward pass produced no readout"),
        }
    }

    /// Runs scheduled forward pass with intermediate depth readouts.
    pub fn forward_scheduled(
        &self,
        tokens: &[u32],
        schedule: &[usize],
        exits: &[usize],
    ) -> Result<Vec<DepthExit>> {
        self.validate(tokens, schedule, exits)?;
        self.run(tokens, schedule, exits, None)
    }

    /// Runs forward pass recording detailed per-layer trace.
    pub fn forward_traced(
        &self,
        tokens: &[u32],
        schedule: &[usize],
    ) -> Result<(DepthExit, ForwardTrace)> {
        self.validate(tokens, schedule, &[schedule.len()])?;
        let mut trace = ForwardTrace {
            tokens: tokens.len(),
            layers: Vec::with_capacity(schedule.len()),
        };
        let exit = self
            .run(tokens, schedule, &[schedule.len()], Some(&mut trace))?
            .pop()
            .context("Forward pass produced no readout")?;
        Ok((exit, trace))
    }

    /// Runs a full forward pass and returns unnormalized logits for the final token position.
    pub fn forward_logits(&self, tokens: &[u32]) -> Result<Vec<f32>> {
        let t_len = tokens.len();
        if t_len < 2 {
            bail!("Forward pass requires at least 2 tokens, received {t_len}");
        }
        let schedule: Vec<usize> = (0..self.config.num_layers).collect();
        self.validate(tokens, &schedule, &[schedule.len()])?;

        let d = self.config.hidden_size;
        let mut x = vec![0.0f32; t_len * d];
        let scale = self.config.embed_scale;
        for (row, &token) in x.chunks_exact_mut(d).zip(tokens) {
            let start = token as usize * d;
            for (dst, &src) in row.iter_mut().zip(&self.embed[start..start + d]) {
                *dst = src * scale;
            }
        }

        let scaling_factor = match self.config.rope_scaling_type.as_deref() {
            Some("linear") => self.config.rope_scaling_factor.unwrap_or(1.0),
            _ => self.config.rope_scaling_factor.unwrap_or(1.0),
        };
        let rope = RopeTable::new_scaled(
            t_len,
            self.config.head_dim,
            self.config.head_dim,
            self.config.rope_theta,
            scaling_factor,
        );
        let mut normed = vec![0.0f32; t_len * d];

        for &index in &schedule {
            let layer = self.get_layer(index)?;
            self.apply_layer(&layer, &mut x, &mut normed, &rope, None);
        }

        rms_norm(&x, &self.final_norm, self.config.rms_norm_eps, &mut normed);
        let last_normed = &normed[(t_len - 1) * d..t_len * d];
        let head = self.lm_head.as_deref().unwrap_or(&self.embed);
        Ok(dispatch_linear_slice(
            self.gpu_backend.as_deref(),
            last_normed,
            head,
            d,
            self.vocab_size,
        ))
    }

    fn run(
        &self,
        tokens: &[u32],
        schedule: &[usize],
        exits: &[usize],
        mut trace: Option<&mut ForwardTrace>,
    ) -> Result<Vec<DepthExit>> {
        let t_len = tokens.len();
        let d = self.config.hidden_size;
        let mut x = vec![0.0f32; t_len * d];

        // 1. Scaled token embeddings: x = embed[t] * sqrt(d)
        let scale = self.config.embed_scale;
        for (row, &token) in x.chunks_exact_mut(d).zip(tokens) {
            let start = token as usize * d;
            for (dst, &src) in row.iter_mut().zip(&self.embed[start..start + d]) {
                *dst = src * scale;
            }
        }

        let scaling_factor = match self.config.rope_scaling_type.as_deref() {
            Some("linear") => self.config.rope_scaling_factor.unwrap_or(1.0),
            _ => self.config.rope_scaling_factor.unwrap_or(1.0),
        };
        let rope = RopeTable::new_scaled(
            t_len,
            self.config.head_dim,
            self.config.head_dim,
            self.config.rope_theta,
            scaling_factor,
        );
        let mut normed = vec![0.0f32; t_len * d];
        let mut out = Vec::with_capacity(exits.len());

        if exits.contains(&0) {
            out.push(self.read_out(0, &x, &mut normed, tokens));
        }

        for (step, &index) in schedule.iter().enumerate() {
            let layer = self.get_layer(index)?;
            match trace.as_deref_mut() {
                Some(tr) => {
                    let before = x.clone();
                    let mut layer_trace = LayerTrace {
                        layer: index,
                        resid_rms_in: rms(&before),
                        ..Default::default()
                    };
                    self.apply_layer(&layer, &mut x, &mut normed, &rope, Some(&mut layer_trace));
                    layer_trace.resid_rms_out = rms(&x);
                    layer_trace.update_ratio = update_ratio(&before, &x);
                    tr.layers.push(layer_trace);
                }
                None => self.apply_layer(&layer, &mut x, &mut normed, &rope, None),
            }

            if exits.contains(&(step + 1)) {
                out.push(self.read_out(step + 1, &x, &mut normed, tokens));
            }
        }

        Ok(out)
    }

    fn apply_layer(
        &self,
        layer: &GemmaLayer,
        x: &mut [f32],
        normed: &mut [f32],
        rope: &RopeTable,
        mut trace: Option<&mut LayerTrace>,
    ) {
        let cfg = &self.config;
        let d = cfg.hidden_size;
        let head_dim = cfg.head_dim;
        let q_dim = cfg.num_heads * head_dim;
        let kv_dim = cfg.num_kv_heads * head_dim;
        let inter = cfg.intermediate_size;
        let t_len = x.len() / d;

        let mut record = |name: &'static str,
                          in_slice: &[f32],
                          in_dim: usize,
                          out_slice: &[f32],
                          out_dim: usize| {
            if let Some(t) = trace.as_deref_mut() {
                t.tensors
                    .push(TensorActivity::measure(name, out_slice, out_dim));
                t.inputs
                    .push(InputChannelStats::measure(name, in_slice, in_dim));
            }
        };

        // 1. Attention Norm
        rms_norm(x, &layer.attn_norm, cfg.rms_norm_eps, normed);

        let gpu = self.gpu_backend.as_deref();

        // 2. Q, K, V Projections
        let mut q = dispatch_linear_slice(gpu, normed, &layer.q, d, q_dim);
        let mut k = dispatch_linear_slice(gpu, normed, &layer.k, d, kv_dim);
        let v = dispatch_linear_slice(gpu, normed, &layer.v, d, kv_dim);

        record("q", normed, d, &q, q_dim);
        record("k", normed, d, &k, kv_dim);
        record("v", normed, d, &v, kv_dim);

        // 3. Per-head QK-RMSNorm
        apply_rms_norm_in_place(&mut q, &layer.q_norm, cfg.rms_norm_eps);
        apply_rms_norm_in_place(&mut k, &layer.k_norm, cfg.rms_norm_eps);

        // 4. RoPE
        rope.apply(&mut q);
        rope.apply(&mut k);

        // 5. GQA Attention
        let mixed = attention(&AttentionInputs {
            q: &q,
            k: &k,
            v: &v,
            num_heads: cfg.num_heads,
            num_kv_heads: cfg.num_kv_heads,
            head_dim,
        });

        // 6. Attention Out projection
        let attn_out = dispatch_linear_slice(gpu, &mixed, &layer.o, q_dim, d);
        record("o", &mixed, q_dim, &attn_out, d);

        // 7. Post-Attention Norm (Gemma dual residual)
        let mut normed_attn = vec![0.0f32; t_len * d];
        rms_norm(
            &attn_out,
            &layer.post_attention_norm,
            cfg.rms_norm_eps,
            &mut normed_attn,
        );
        add_in_place(x, &normed_attn);

        // 8. Pre-FFN Norm
        rms_norm(x, &layer.ffn_norm, cfg.rms_norm_eps, normed);

        // 9. GeGLU Feedforward: gate * gelu_approx(gate) * up
        let mut gate = dispatch_linear_slice(gpu, normed, &layer.gate, d, inter);
        let up = dispatch_linear_slice(gpu, normed, &layer.up, d, inter);
        record("gate", normed, d, &gate, inter);
        record("up", normed, d, &up, inter);

        for (g, u) in gate.iter_mut().zip(&up) {
            *g = gelu_approx(*g) * u;
        }

        // 10. Down projection
        let ffn_out = dispatch_linear_slice(gpu, &gate, &layer.down, inter, d);
        record("down", &gate, inter, &ffn_out, d);

        if let Some(t) = trace {
            t.mlp_dead_fraction = dead_fraction(&gate);
        }

        // 11. Post-FFN Norm (Gemma dual residual)
        let mut normed_ffn = vec![0.0f32; t_len * d];
        rms_norm(
            &ffn_out,
            &layer.post_ffw_norm,
            cfg.rms_norm_eps,
            &mut normed_ffn,
        );
        add_in_place(x, &normed_ffn);
    }

    fn read_out(
        &self,
        layers_applied: usize,
        x: &[f32],
        normed: &mut [f32],
        tokens: &[u32],
    ) -> DepthExit {
        rms_norm(x, &self.final_norm, self.config.rms_norm_eps, normed);
        let head = self.lm_head.as_deref().unwrap_or(&self.embed);
        dispatch_score_head_slice(
            self.gpu_backend.as_deref(),
            head,
            self.config.hidden_size,
            self.vocab_size,
            normed,
            tokens,
            layers_applied,
        )
    }
}
