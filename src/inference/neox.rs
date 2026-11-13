//! GPT-NeoX (Pythia) forward pass.
//!
//! Differences from the LLaMA path: LayerNorm with bias, a fused QKV projection laid out
//! per head as `[q | k | v]`, rotary embeddings on only the leading `rotary_pct` of each
//! head, biases on every linear layer, exact (erf) GELU, a parallel attention/MLP
//! residual, and an untied output head (`embed_out`).

use anyhow::{bail, Context, Result};
use std::path::Path;

use super::llama::{
    add_in_place, attention, find_tensor, linear, load_tensor, score_head, AttentionInputs,
    DepthExit, RopeTable, WeightPrecision,
};
use super::probe::{dead_fraction, rms, update_ratio, ForwardTrace, LayerTrace, TensorActivity};
use crate::ir::{ModelIR, TensorIR};

/// Architectural hyperparameters needed by the GPT-NeoX forward pass.
#[derive(Debug, Clone, PartialEq)]
pub struct NeoxConfig {
    pub num_layers: usize,
    pub hidden_size: usize,
    pub num_heads: usize,
    /// Fraction of each head's dimensions that receive rotary embeddings.
    pub rotary_pct: f32,
    pub rope_theta: f32,
    pub layer_norm_eps: f32,
    /// `x + attn(ln1(x)) + mlp(ln2(x))` when true; sequential residuals when false.
    pub parallel_residual: bool,
}

impl NeoxConfig {
    /// Builds a config from the IR, using the `GPTNeoXConfig` defaults for the rest.
    pub fn from_model_ir(model: &ModelIR) -> Self {
        Self {
            num_layers: model.num_layers,
            hidden_size: model.hidden_size,
            num_heads: model.num_attention_heads,
            rotary_pct: 0.25,
            rope_theta: model.rope_theta.unwrap_or(10_000.0),
            layer_norm_eps: model.norm_eps.unwrap_or(1e-5),
            parallel_residual: true,
        }
    }

    /// Overrides rotary, normalization and residual settings from a Hugging Face `config.json`.
    pub fn with_hf_config(mut self, model_path: &Path) -> Result<Self> {
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
        let number = |key: &str| value.get(key).and_then(serde_json::Value::as_f64);
        if let Some(v) = number("rotary_pct") {
            self.rotary_pct = v as f32;
        }
        if let Some(v) = number("rotary_emb_base").or_else(|| number("rope_theta")) {
            self.rope_theta = v as f32;
        }
        if let Some(v) = number("layer_norm_eps") {
            self.layer_norm_eps = v as f32;
        }
        if let Some(v) = value
            .get("use_parallel_residual")
            .and_then(serde_json::Value::as_bool)
        {
            self.parallel_residual = v;
        }
        if let Some(act) = value.get("hidden_act").and_then(serde_json::Value::as_str) {
            if act != "gelu" {
                bail!("Unsupported GPT-NeoX activation '{act}'; only exact 'gelu' is implemented");
            }
        }
        Ok(self)
    }

    fn rotary_dims(&self) -> usize {
        let head_dim = self.hidden_size / self.num_heads;
        (head_dim as f32 * self.rotary_pct) as usize
    }
}

pub(crate) struct NeoxLayer {
    pub(crate) ln1_w: Vec<f32>,
    pub(crate) ln1_b: Vec<f32>,
    pub(crate) ln2_w: Vec<f32>,
    pub(crate) ln2_b: Vec<f32>,
    pub(crate) qkv_w: Vec<f32>,
    pub(crate) qkv_b: Vec<f32>,
    pub(crate) o_w: Vec<f32>,
    pub(crate) o_b: Vec<f32>,
    pub(crate) up_w: Vec<f32>,
    pub(crate) up_b: Vec<f32>,
    pub(crate) down_w: Vec<f32>,
    pub(crate) down_b: Vec<f32>,
}

/// Dense in-memory GPT-NeoX model ready for a forward pass.
pub struct NeoxModel {
    pub(crate) config: NeoxConfig,
    pub(crate) vocab_size: usize,
    pub(crate) intermediate_size: usize,
    pub(crate) embed: Vec<f32>,
    pub(crate) layers: Vec<NeoxLayer>,
    pub(crate) final_w: Vec<f32>,
    pub(crate) final_b: Vec<f32>,
    /// `None` when the output head is tied to the input embedding.
    pub(crate) head: Option<Vec<f32>>,
}

impl NeoxModel {
    /// Loads every tensor of a file-backed GPT-NeoX model into dense `f32` storage.
    pub fn load(model: &ModelIR, config: NeoxConfig, precision: WeightPrecision) -> Result<Self> {
        let d = config.hidden_size;
        if d == 0 || config.num_heads == 0 || config.num_layers == 0 {
            bail!("Model configuration has a zero dimension: {config:?}");
        }
        if !d.is_multiple_of(config.num_heads) {
            bail!(
                "Hidden size {d} is not divisible by {} heads",
                config.num_heads
            );
        }
        let required = |name: &str| -> Result<&TensorIR> {
            find_tensor(model, name)
                .with_context(|| format!("Model is missing required tensor '{name}'"))
        };

        let embed_tensor = required("embed_in.weight")?;
        if embed_tensor.numel == 0 || !embed_tensor.numel.is_multiple_of(d) {
            bail!(
                "Embedding tensor '{}' has {} values, not a multiple of hidden size {d}",
                embed_tensor.name,
                embed_tensor.numel
            );
        }
        let vocab_size = embed_tensor.numel / d;
        let embed = load_tensor(embed_tensor, vocab_size * d, precision)?;

        let up0 = required("layers.0.mlp.dense_h_to_4h.weight")?;
        if up0.numel == 0 || !up0.numel.is_multiple_of(d) {
            bail!(
                "MLP tensor '{}' has {} values, not a multiple of hidden size {d}",
                up0.name,
                up0.numel
            );
        }
        let inter = up0.numel / d;

        let mut layers = Vec::with_capacity(config.num_layers);
        for index in 0..config.num_layers {
            let fetch = |suffix: &str, expected: usize| -> Result<Vec<f32>> {
                let name = format!("layers.{index}.{suffix}");
                load_tensor(required(&name)?, expected, precision)
            };
            layers.push(NeoxLayer {
                ln1_w: fetch("input_layernorm.weight", d)?,
                ln1_b: fetch("input_layernorm.bias", d)?,
                ln2_w: fetch("post_attention_layernorm.weight", d)?,
                ln2_b: fetch("post_attention_layernorm.bias", d)?,
                qkv_w: fetch("attention.query_key_value.weight", 3 * d * d)?,
                qkv_b: fetch("attention.query_key_value.bias", 3 * d)?,
                o_w: fetch("attention.dense.weight", d * d)?,
                o_b: fetch("attention.dense.bias", d)?,
                up_w: fetch("mlp.dense_h_to_4h.weight", inter * d)?,
                up_b: fetch("mlp.dense_h_to_4h.bias", inter)?,
                down_w: fetch("mlp.dense_4h_to_h.weight", d * inter)?,
                down_b: fetch("mlp.dense_4h_to_h.bias", d)?,
            });
        }
        let final_w = load_tensor(required("final_layer_norm.weight")?, d, precision)?;
        let final_b = load_tensor(required("final_layer_norm.bias")?, d, precision)?;
        let head = match find_tensor(model, "embed_out.weight") {
            Some(tensor) => Some(load_tensor(tensor, vocab_size * d, precision)?),
            None => None,
        };
        Ok(Self {
            config,
            vocab_size,
            intermediate_size: inter,
            embed,
            layers,
            final_w,
            final_b,
            head,
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Runs the layers named by `schedule` and reads out after each count in `exits`.
    /// See `LlamaModel::forward_scheduled` for the meaning of intermediate readouts.
    pub fn forward_scheduled(
        &self,
        tokens: &[u32],
        schedule: &[usize],
        exits: &[usize],
    ) -> Result<Vec<DepthExit>> {
        self.validate(tokens, schedule, exits)?;
        Ok(self.run(tokens, schedule, exits, None))
    }

    /// Runs `schedule` once, recording a [`ForwardTrace`], and reads out at the end.
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
            .run(tokens, schedule, &[schedule.len()], Some(&mut trace))
            .pop()
            .context("Forward pass produced no readout")?;
        Ok((exit, trace))
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
        if let Some(bad) = schedule.iter().find(|&&l| l >= self.layers.len()) {
            bail!(
                "Schedule names layer {bad}, but the model has {} layers",
                self.layers.len()
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

    fn run(
        &self,
        tokens: &[u32],
        schedule: &[usize],
        exits: &[usize],
        mut trace: Option<&mut ForwardTrace>,
    ) -> Vec<DepthExit> {
        let t_len = tokens.len();
        let d = self.config.hidden_size;
        let head_dim = d / self.config.num_heads;
        let mut x = vec![0.0f32; t_len * d];
        for (row, &token) in x.chunks_exact_mut(d).zip(tokens) {
            let start = token as usize * d;
            row.copy_from_slice(&self.embed[start..start + d]);
        }
        let rope = RopeTable::new(
            t_len,
            head_dim,
            self.config.rotary_dims(),
            self.config.rope_theta,
        );
        let mut out = Vec::with_capacity(exits.len());
        if exits.contains(&0) {
            out.push(self.read_out(0, &x, tokens));
        }
        for (step, &index) in schedule.iter().enumerate() {
            match trace.as_deref_mut() {
                Some(trace) => {
                    let before = x.clone();
                    let mut layer_trace = LayerTrace {
                        layer: index,
                        resid_rms_in: rms(&before),
                        ..Default::default()
                    };
                    self.apply_layer(&self.layers[index], &mut x, &rope, Some(&mut layer_trace));
                    layer_trace.resid_rms_out = rms(&x);
                    layer_trace.update_ratio = update_ratio(&before, &x);
                    trace.layers.push(layer_trace);
                }
                None => self.apply_layer(&self.layers[index], &mut x, &rope, None),
            }
            if exits.contains(&(step + 1)) {
                out.push(self.read_out(step + 1, &x, tokens));
            }
        }
        out
    }

    fn apply_layer(
        &self,
        layer: &NeoxLayer,
        x: &mut [f32],
        rope: &RopeTable,
        mut trace: Option<&mut LayerTrace>,
    ) {
        let cfg = &self.config;
        let d = cfg.hidden_size;
        let heads = cfg.num_heads;
        let hd = d / heads;
        let inter = self.intermediate_size;
        let t_len = x.len() / d;

        let mut normed = vec![0.0f32; x.len()];
        layer_norm(
            x,
            &layer.ln1_w,
            &layer.ln1_b,
            cfg.layer_norm_eps,
            &mut normed,
        );
        let mut qkv = linear(&normed, &layer.qkv_w, d, 3 * d);
        add_bias(&mut qkv, &layer.qkv_b);
        if let Some(t) = trace.as_deref_mut() {
            t.tensors.push(TensorActivity::measure("qkv", &qkv, 3 * d));
        }
        let (mut q, mut k, mut v) = (
            vec![0.0f32; x.len()],
            vec![0.0f32; x.len()],
            vec![0.0f32; x.len()],
        );
        for t in 0..t_len {
            for h in 0..heads {
                let src = t * 3 * d + h * 3 * hd;
                let dst = t * d + h * hd;
                q[dst..dst + hd].copy_from_slice(&qkv[src..src + hd]);
                k[dst..dst + hd].copy_from_slice(&qkv[src + hd..src + 2 * hd]);
                v[dst..dst + hd].copy_from_slice(&qkv[src + 2 * hd..src + 3 * hd]);
            }
        }
        rope.apply(&mut q);
        rope.apply(&mut k);
        let mixed = attention(&AttentionInputs {
            q: &q,
            k: &k,
            v: &v,
            num_heads: heads,
            num_kv_heads: heads,
            head_dim: hd,
        });
        let mut attn_out = linear(&mixed, &layer.o_w, d, d);
        add_bias(&mut attn_out, &layer.o_b);
        if let Some(t) = trace.as_deref_mut() {
            t.tensors.push(TensorActivity::measure("o", &attn_out, d));
        }

        if cfg.parallel_residual {
            layer_norm(
                x,
                &layer.ln2_w,
                &layer.ln2_b,
                cfg.layer_norm_eps,
                &mut normed,
            );
            add_in_place(x, &attn_out);
        } else {
            add_in_place(x, &attn_out);
            layer_norm(
                x,
                &layer.ln2_w,
                &layer.ln2_b,
                cfg.layer_norm_eps,
                &mut normed,
            );
        }
        let mut hidden = linear(&normed, &layer.up_w, d, inter);
        add_bias(&mut hidden, &layer.up_b);
        if let Some(t) = trace.as_deref_mut() {
            t.tensors
                .push(TensorActivity::measure("up", &hidden, inter));
        }
        for h in hidden.iter_mut() {
            *h = gelu(*h);
        }
        let mut mlp_out = linear(&hidden, &layer.down_w, inter, d);
        add_bias(&mut mlp_out, &layer.down_b);
        if let Some(t) = trace {
            t.tensors.push(TensorActivity::measure("down", &mlp_out, d));
            t.mlp_dead_fraction = dead_fraction(&hidden);
        }
        add_in_place(x, &mlp_out);
    }

    fn read_out(&self, layers_applied: usize, x: &[f32], tokens: &[u32]) -> DepthExit {
        let d = self.config.hidden_size;
        let mut normed = vec![0.0f32; x.len()];
        layer_norm(
            x,
            &self.final_w,
            &self.final_b,
            self.config.layer_norm_eps,
            &mut normed,
        );
        let head = self.head.as_deref().unwrap_or(&self.embed);
        score_head(head, d, self.vocab_size, &normed, tokens, layers_applied)
    }
}

fn add_bias(rows: &mut [f32], bias: &[f32]) {
    for row in rows.chunks_exact_mut(bias.len()) {
        for (value, b) in row.iter_mut().zip(bias) {
            *value += b;
        }
    }
}

fn layer_norm(x: &[f32], weight: &[f32], bias: &[f32], eps: f32, out: &mut [f32]) {
    let d = weight.len();
    for (row, out_row) in x.chunks_exact(d).zip(out.chunks_exact_mut(d)) {
        let mean = row.iter().sum::<f32>() / d as f32;
        let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / d as f32;
        let scale = 1.0 / (var + eps).sqrt();
        for (((o, v), w), b) in out_row.iter_mut().zip(row).zip(weight).zip(bias) {
            *o = (v - mean) * scale * w + b;
        }
    }
}

/// Error function, Abramowitz & Stegun 7.1.26 (absolute error below 1.5e-7).
fn erf(x: f32) -> f32 {
    let x = f64::from(x);
    let t = 1.0 / (1.0 + 0.327_591_1 * x.abs());
    let poly = t
        * (0.254_829_592
            + t * (-0.284_496_736
                + t * (1.421_413_741 + t * (-1.453_152_027 + t * 1.061_405_429))));
    let y = 1.0 - poly * (-x * x).exp();
    (if x < 0.0 { -y } else { y }) as f32
}

/// Exact GELU: `x * Φ(x)`.
fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + erf(x * std::f32::consts::FRAC_1_SQRT_2))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny(scale: f32) -> NeoxModel {
        let (d, heads, inter, vocab) = (16usize, 2usize, 32usize, 24usize);
        let mut seed = 11u64;
        let mut rand = |len: usize| -> Vec<f32> {
            (0..len)
                .map(|_| {
                    seed = seed
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    (((seed >> 33) as f32 / 2_147_483_648.0) - 1.0) * scale
                })
                .collect()
        };
        let layers = (0..2)
            .map(|_| NeoxLayer {
                ln1_w: vec![1.0; d],
                ln1_b: vec![0.0; d],
                ln2_w: vec![1.0; d],
                ln2_b: vec![0.0; d],
                qkv_w: rand(3 * d * d),
                qkv_b: rand(3 * d),
                o_w: rand(d * d),
                o_b: rand(d),
                up_w: rand(inter * d),
                up_b: rand(inter),
                down_w: rand(d * inter),
                down_b: rand(d),
            })
            .collect();
        NeoxModel {
            config: NeoxConfig {
                num_layers: 2,
                hidden_size: d,
                num_heads: heads,
                rotary_pct: 0.25,
                rope_theta: 10_000.0,
                layer_norm_eps: 1e-5,
                parallel_residual: true,
            },
            vocab_size: vocab,
            intermediate_size: inter,
            embed: rand(vocab * d),
            layers,
            final_w: vec![1.0; d],
            final_b: vec![0.0; d],
            head: Some(rand(vocab * d)),
        }
    }

    #[test]
    fn erf_and_gelu_match_reference_values() {
        for (x, expected) in [
            (0.0f32, 0.0f32),
            (0.5, 0.520_499_9),
            (1.0, 0.842_700_8),
            (-2.0, -0.995_322_3),
        ] {
            assert!((erf(x) - expected).abs() < 1e-6, "erf({x})");
        }
        assert!((gelu(1.0) - 0.841_344_7).abs() < 1e-6);
        assert!((gelu(-1.0) + 0.158_655_3).abs() < 1e-6);
    }

    #[test]
    fn zero_head_gives_uniform_distribution() {
        let mut model = tiny(0.5);
        model.head = Some(vec![0.0; 24 * 16]);
        let nll = model
            .forward_scheduled(&[1, 2, 3, 4], &[0, 1], &[2])
            .unwrap();
        for value in &nll[0].nll {
            assert!((value - (24.0f32).ln()).abs() < 1e-5);
        }
    }

    #[test]
    fn predictions_are_causal() {
        let model = tiny(0.6);
        let base: Vec<u32> = vec![3, 17, 8, 20, 1, 9, 12, 5];
        let full = |tokens: &[u32]| {
            model
                .forward_scheduled(tokens, &[0, 1], &[2])
                .unwrap()
                .remove(0)
                .nll
        };
        let reference = full(&base);
        let mut future = base.clone();
        future[7] = 23;
        assert_eq!(&reference[..6], &full(&future)[..6]);
        let mut past = base.clone();
        past[1] = 2;
        assert!(reference[3..]
            .iter()
            .zip(&full(&past)[3..])
            .any(|(a, b)| (a - b).abs() > 1e-4));
    }

    #[test]
    fn sequential_and_parallel_residuals_differ() {
        let parallel = tiny(0.6);
        let mut sequential = tiny(0.6);
        sequential.config.parallel_residual = false;
        let tokens = [4u32, 9, 2, 7, 1];
        let a = parallel.forward_scheduled(&tokens, &[0, 1], &[2]).unwrap();
        let b = sequential
            .forward_scheduled(&tokens, &[0, 1], &[2])
            .unwrap();
        assert_ne!(a[0].nll, b[0].nll);
    }
}
