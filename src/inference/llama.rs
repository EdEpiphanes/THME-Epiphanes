//! LLaMA-family decoder forward pass (RMSNorm, RoPE, grouped-query attention, SwiGLU).
//!
//! Tensor naming follows the Hugging Face `LlamaForCausalLM` layout. Weights are
//! held as dense `f32` regardless of how they are stored on disk, so a quantized
//! tensor contributes exactly the values a consumer would decode from it.

use anyhow::{bail, Context, Result};
use std::path::Path;

use super::probe::{
    dead_fraction, rms, update_ratio, ForwardTrace, InputChannelStats, LayerTrace, TensorActivity,
};
use super::weights::{apply_selected_format, fake_quantize, read_tensor_f32};
use crate::ir::{ModelIR, TensorFormat, TensorIR};

/// Positions processed together per weight row, so each row is streamed once per block.
const BLOCK: usize = 8;
/// Accumulator width; one 256-bit vector of `f32`.
const LANES: usize = 8;

/// Architectural hyperparameters needed by the forward pass.
#[derive(Debug, Clone, PartialEq)]
pub struct LlamaConfig {
    pub num_layers: usize,
    pub hidden_size: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
}

impl LlamaConfig {
    /// Builds a config from the IR. Epsilon and RoPE base come from the model's recorded
    /// values (`config.json` or GGUF metadata), falling back to LLaMA defaults.
    pub fn from_model_ir(model: &ModelIR) -> Self {
        Self {
            num_layers: model.num_layers,
            hidden_size: model.hidden_size,
            num_heads: model.num_attention_heads,
            num_kv_heads: model.num_key_value_heads,
            rms_norm_eps: model.norm_eps.unwrap_or(1e-6),
            rope_theta: model.rope_theta.unwrap_or(10_000.0),
        }
    }

    /// Overrides epsilon and RoPE base from a Hugging Face `config.json`, when one is present.
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
        if let Some(eps) = value
            .get("rms_norm_eps")
            .and_then(serde_json::Value::as_f64)
        {
            self.rms_norm_eps = eps as f32;
        }
        if let Some(theta) = value.get("rope_theta").and_then(serde_json::Value::as_f64) {
            self.rope_theta = theta as f32;
        }
        Ok(self)
    }
}

/// Which values the forward pass should see for each tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightPrecision {
    /// Exactly what is stored in the source file.
    AsStored,
    /// Stored values round-tripped through each tensor's `selected_format`.
    SelectedFormat,
}

pub(crate) struct LlamaLayer {
    pub(crate) attn_norm: Vec<f32>,
    pub(crate) q: Vec<f32>,
    pub(crate) k: Vec<f32>,
    pub(crate) v: Vec<f32>,
    pub(crate) o: Vec<f32>,
    pub(crate) mlp_norm: Vec<f32>,
    pub(crate) gate: Vec<f32>,
    pub(crate) up: Vec<f32>,
    pub(crate) down: Vec<f32>,
}

/// Predictions read out after a given number of applied layers.
#[derive(Debug, Clone)]
pub struct DepthExit {
    pub layers_applied: usize,
    /// Negative log-likelihood (nats) of each true next token; entry `i` scores `tokens[i + 1]`.
    pub nll: Vec<f32>,
    /// Most likely next token at each scored position.
    pub top1: Vec<u32>,
}

/// Dense in-memory LLaMA-family model ready for a forward pass.
pub struct LlamaModel {
    pub(crate) config: LlamaConfig,
    pub(crate) vocab_size: usize,
    pub(crate) intermediate_size: usize,
    pub(crate) embed: Vec<f32>,
    pub(crate) layers: Vec<LlamaLayer>,
    pub(crate) final_norm: Vec<f32>,
    /// `None` when the output head is tied to the embedding matrix.
    pub(crate) lm_head: Option<Vec<f32>>,
}

pub(crate) fn find_tensor<'a>(model: &'a ModelIR, name: &str) -> Option<&'a TensorIR> {
    let dotted = format!(".{name}");
    model
        .tensors
        .iter()
        .find(|t| t.name == name || t.name.ends_with(&dotted))
}

pub(crate) fn load_tensor(
    tensor: &TensorIR,
    expected: usize,
    precision: WeightPrecision,
) -> Result<Vec<f32>> {
    let mut values = read_tensor_f32(tensor)?;
    if values.len() != expected {
        bail!(
            "Tensor '{}' holds {} values, expected {expected}",
            tensor.name,
            values.len()
        );
    }
    // Block formats apply to weight matrices; 1-D tensors (norms, biases) stay as stored,
    // as GGML-style exporters keep them.
    if precision == WeightPrecision::SelectedFormat && tensor.shape.len() >= 2 {
        apply_selected_format(tensor, &mut values).with_context(|| {
            format!(
                "Failed to apply {} to tensor '{}'",
                tensor.selected_format.display_name(),
                tensor.name
            )
        })?;
    }
    Ok(values)
}

impl LlamaModel {
    /// Loads every tensor of a file-backed model into dense `f32` storage.
    pub fn load(model: &ModelIR, config: LlamaConfig, precision: WeightPrecision) -> Result<Self> {
        let d = config.hidden_size;
        if d == 0 || config.num_heads == 0 || config.num_kv_heads == 0 || config.num_layers == 0 {
            bail!("Model configuration has a zero dimension: {config:?}");
        }
        if !d.is_multiple_of(config.num_heads) {
            bail!(
                "Hidden size {d} is not divisible by {} heads",
                config.num_heads
            );
        }
        if !config.num_heads.is_multiple_of(config.num_kv_heads) {
            bail!(
                "{} attention heads cannot be grouped over {} key/value heads",
                config.num_heads,
                config.num_kv_heads
            );
        }
        let head_dim = d / config.num_heads;
        if !head_dim.is_multiple_of(2) {
            bail!("Head dimension {head_dim} must be even for rotary embeddings");
        }
        let kv_dim = head_dim * config.num_kv_heads;

        let required = |name: &str| -> Result<&TensorIR> {
            find_tensor(model, name)
                .with_context(|| format!("Model is missing required tensor '{name}'"))
        };

        let embed_tensor = required("model.embed_tokens.weight")?;
        if embed_tensor.numel == 0 || !embed_tensor.numel.is_multiple_of(d) {
            bail!(
                "Embedding tensor '{}' has {} values, not a multiple of hidden size {d}",
                embed_tensor.name,
                embed_tensor.numel
            );
        }
        let vocab_size = embed_tensor.numel / d;
        let embed = load_tensor(embed_tensor, vocab_size * d, precision)?;

        let gate0 = required("model.layers.0.mlp.gate_proj.weight")?;
        if gate0.numel == 0 || !gate0.numel.is_multiple_of(d) {
            bail!(
                "MLP gate tensor '{}' has {} values, not a multiple of hidden size {d}",
                gate0.name,
                gate0.numel
            );
        }
        let intermediate_size = gate0.numel / d;

        let mut layers = Vec::with_capacity(config.num_layers);
        for index in 0..config.num_layers {
            let fetch = |suffix: &str, expected: usize| -> Result<Vec<f32>> {
                let name = format!("model.layers.{index}.{suffix}");
                load_tensor(required(&name)?, expected, precision)
            };
            layers.push(LlamaLayer {
                attn_norm: fetch("input_layernorm.weight", d)?,
                q: fetch("self_attn.q_proj.weight", d * d)?,
                k: fetch("self_attn.k_proj.weight", kv_dim * d)?,
                v: fetch("self_attn.v_proj.weight", kv_dim * d)?,
                o: fetch("self_attn.o_proj.weight", d * d)?,
                mlp_norm: fetch("post_attention_layernorm.weight", d)?,
                gate: fetch("mlp.gate_proj.weight", intermediate_size * d)?,
                up: fetch("mlp.up_proj.weight", intermediate_size * d)?,
                down: fetch("mlp.down_proj.weight", d * intermediate_size)?,
            });
        }

        let final_norm = load_tensor(required("model.norm.weight")?, d, precision)?;
        let lm_head = match find_tensor(model, "lm_head.weight") {
            Some(tensor) => Some(load_tensor(tensor, vocab_size * d, precision)?),
            None => None,
        };

        Ok(Self {
            config,
            vocab_size,
            intermediate_size,
            embed,
            layers,
            final_norm,
            lm_head,
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Runs one causal forward pass and returns the negative log-likelihood (nats)
    /// of each token given the tokens before it. Entry `i` scores `tokens[i + 1]`.
    pub fn forward_nll(&self, tokens: &[u32]) -> Result<Vec<f32>> {
        let schedule: Vec<usize> = (0..self.layers.len()).collect();
        let mut exits = self.forward_scheduled(tokens, &schedule, &[schedule.len()])?;
        match exits.pop() {
            Some(exit) => Ok(exit.nll),
            None => bail!("Forward pass produced no readout"),
        }
    }

    /// Runs the layers named by `schedule` in order (indices may repeat or be skipped)
    /// and reads out predictions after each applied-layer count listed in `exits`.
    ///
    /// A readout applies the model's own final norm and output head to the state at
    /// that depth. The model was never trained to be read at intermediate depths or
    /// with a modified schedule, so these readouts are diagnostics of the stored
    /// weights, not the behaviour of a model trained for early exit or looping.
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
        let rope = RopeTable::new(t_len, head_dim, head_dim, self.config.rope_theta);
        let mut normed = vec![0.0f32; t_len * d];

        let mut out = Vec::with_capacity(exits.len());
        if exits.contains(&0) {
            out.push(self.read_out(0, &x, &mut normed, tokens));
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
                    self.apply_layer(
                        &self.layers[index],
                        &mut x,
                        &mut normed,
                        &rope,
                        Some(&mut layer_trace),
                    );
                    layer_trace.resid_rms_out = rms(&x);
                    layer_trace.update_ratio = update_ratio(&before, &x);
                    trace.layers.push(layer_trace);
                }
                None => self.apply_layer(&self.layers[index], &mut x, &mut normed, &rope, None),
            }
            if exits.contains(&(step + 1)) {
                out.push(self.read_out(step + 1, &x, &mut normed, tokens));
            }
        }
        out
    }

    fn apply_layer(
        &self,
        layer: &LlamaLayer,
        x: &mut [f32],
        normed: &mut [f32],
        rope: &RopeTable,
        mut trace: Option<&mut LayerTrace>,
    ) {
        let cfg = &self.config;
        let d = cfg.hidden_size;
        let head_dim = d / cfg.num_heads;
        let kv_dim = head_dim * cfg.num_kv_heads;
        let inter = self.intermediate_size;
        let mut record =
            |name: &'static str, x: &[f32], in_dim: usize, y: &[f32], out_dim: usize| {
                if let Some(t) = trace.as_deref_mut() {
                    t.tensors.push(TensorActivity::measure(name, y, out_dim));
                    t.inputs.push(InputChannelStats::measure(name, x, in_dim));
                }
            };

        rms_norm(x, &layer.attn_norm, cfg.rms_norm_eps, normed);
        let mut q = linear(normed, &layer.q, d, d);
        let mut k = linear(normed, &layer.k, d, kv_dim);
        let v = linear(normed, &layer.v, d, kv_dim);
        record("q", normed, d, &q, d);
        record("k", normed, d, &k, kv_dim);
        record("v", normed, d, &v, kv_dim);
        rope.apply(&mut q);
        rope.apply(&mut k);
        let mixed = attention(&AttentionInputs {
            q: &q,
            k: &k,
            v: &v,
            num_heads: cfg.num_heads,
            num_kv_heads: cfg.num_kv_heads,
            head_dim,
        });
        let attn_out = linear(&mixed, &layer.o, d, d);
        record("o", &mixed, d, &attn_out, d);
        add_in_place(x, &attn_out);

        rms_norm(x, &layer.mlp_norm, cfg.rms_norm_eps, normed);
        let mut hidden = linear(normed, &layer.gate, d, inter);
        let up = linear(normed, &layer.up, d, inter);
        record("gate", normed, d, &hidden, inter);
        record("up", normed, d, &up, inter);
        for (g, u) in hidden.iter_mut().zip(&up) {
            *g = *g / (1.0 + (-*g).exp()) * u;
        }
        let mlp_out = linear(&hidden, &layer.down, inter, d);
        record("down", &hidden, inter, &mlp_out, d);
        if let Some(t) = trace {
            t.mlp_dead_fraction = dead_fraction(&hidden);
        }
        add_in_place(x, &mlp_out);
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
        score_head(
            head,
            self.config.hidden_size,
            self.vocab_size,
            normed,
            tokens,
            layers_applied,
        )
    }
}

/// Applies an output head of shape `[vocab, d]` to normalized states and scores each
/// position against the next token: negative log-likelihood and most likely token.
pub(crate) fn score_head(
    head: &[f32],
    d: usize,
    vocab: usize,
    normed: &[f32],
    tokens: &[u32],
    layers_applied: usize,
) -> DepthExit {
    let mut scores = vec![0.0f32; 2 * (tokens.len() - 1)];
    par_row_ranges(&mut scores, 2, |first, chunk| {
        let mut logits = vec![0.0f32; BLOCK * vocab];
        for (block_index, out) in chunk.chunks_mut(2 * BLOCK).enumerate() {
            let start = first + block_index * BLOCK;
            let rows = out.len() / 2;
            matmul(
                head,
                &normed[start * d..(start + rows) * d],
                &mut logits[..rows * vocab],
                d,
                vocab,
            );
            for (r, slot) in out.chunks_exact_mut(2).enumerate() {
                let row = &logits[r * vocab..(r + 1) * vocab];
                let target = tokens[start + r + 1] as usize;
                let mut max = f32::NEG_INFINITY;
                let mut best = 0usize;
                for (i, &l) in row.iter().enumerate() {
                    if l > max {
                        max = l;
                        best = i;
                    }
                }
                let sum: f64 = row.iter().map(|&l| f64::from((l - max).exp())).sum();
                slot[0] = (f64::from(max - row[target]) + sum.ln()) as f32;
                // Token ids are far below 2^24, so the f32 round-trip is exact.
                slot[1] = best as f32;
            }
        }
    });
    DepthExit {
        layers_applied,
        nll: scores.chunks_exact(2).map(|s| s[0]).collect(),
        top1: scores.chunks_exact(2).map(|s| s[1] as u32).collect(),
    }
}
/// Splits `out` into contiguous row ranges and runs `f(first_row, rows)` on each in parallel.
fn par_row_ranges<F>(out: &mut [f32], row_len: usize, f: F)
where
    F: Fn(usize, &mut [f32]) + Sync,
{
    let rows = out.len() / row_len.max(1);
    let threads = std::thread::available_parallelism()
        .map(std::num::NonZeroUsize::get)
        .unwrap_or(1)
        .min(rows)
        .max(1);
    if threads == 1 {
        f(0, out);
        return;
    }
    let per_thread = rows.div_ceil(threads);
    std::thread::scope(|scope| {
        for (index, chunk) in out.chunks_mut(per_thread * row_len).enumerate() {
            let f = &f;
            scope.spawn(move || f(index * per_thread, chunk));
        }
    });
}

/// `y[t] = W · x[t]` for a row-major `W` of shape `[out_dim, in_dim]`.
pub(crate) fn linear(x: &[f32], w: &[f32], in_dim: usize, out_dim: usize) -> Vec<f32> {
    let rows = x.len() / in_dim;
    let mut y = vec![0.0f32; rows * out_dim];
    par_row_ranges(&mut y, out_dim, |first, chunk| {
        let count = chunk.len() / out_dim;
        matmul(
            w,
            &x[first * in_dim..(first + count) * in_dim],
            chunk,
            in_dim,
            out_dim,
        );
    });
    y
}

#[cfg(target_arch = "x86_64")]
fn has_avx2_fma() -> bool {
    std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
}

fn matmul(w: &[f32], xs: &[f32], ys: &mut [f32], in_dim: usize, out_dim: usize) {
    #[cfg(target_arch = "x86_64")]
    if has_avx2_fma() {
        // SAFETY: AVX2 and FMA availability was verified at runtime on the line above.
        unsafe { simd::matmul(w, xs, ys, in_dim, out_dim) };
        return;
    }
    matmul_portable(w, xs, ys, in_dim, out_dim);
}

/// AVX2 + FMA kernels. Every function requires the caller to have verified both features.
#[cfg(target_arch = "x86_64")]
mod simd {
    use super::{BLOCK, LANES};
    use std::arch::x86_64::{
        __m256, _mm256_add_ps, _mm256_fmadd_ps, _mm256_loadu_ps, _mm256_set1_ps, _mm256_setzero_ps,
        _mm256_storeu_ps,
    };

    #[inline]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn horizontal_sum(v: __m256) -> f32 {
        let mut lanes = [0.0f32; LANES];
        // SAFETY: `lanes` holds exactly eight f32 values.
        unsafe { _mm256_storeu_ps(lanes.as_mut_ptr(), v) };
        lanes.iter().sum()
    }

    #[inline]
    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn dot(a: &[f32], b: &[f32]) -> f32 {
        let len = a.len().min(b.len());
        let vector_len = len / LANES * LANES;
        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();
        let mut i = 0;
        while i + 2 * LANES <= vector_len {
            // SAFETY: `i + 16 <= vector_len <= len`, so all four loads are in bounds.
            unsafe {
                acc0 = _mm256_fmadd_ps(
                    _mm256_loadu_ps(a.as_ptr().add(i)),
                    _mm256_loadu_ps(b.as_ptr().add(i)),
                    acc0,
                );
                acc1 = _mm256_fmadd_ps(
                    _mm256_loadu_ps(a.as_ptr().add(i + LANES)),
                    _mm256_loadu_ps(b.as_ptr().add(i + LANES)),
                    acc1,
                );
            }
            i += 2 * LANES;
        }
        if i < vector_len {
            // SAFETY: `i + 8 <= vector_len <= len`.
            unsafe {
                acc0 = _mm256_fmadd_ps(
                    _mm256_loadu_ps(a.as_ptr().add(i)),
                    _mm256_loadu_ps(b.as_ptr().add(i)),
                    acc0,
                );
            }
        }
        // SAFETY: the caller verified AVX2 and FMA.
        let mut sum = unsafe { horizontal_sum(_mm256_add_ps(acc0, acc1)) };
        for (x, y) in a[vector_len..len].iter().zip(&b[vector_len..len]) {
            sum += x * y;
        }
        sum
    }

    /// `y += alpha * x`
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn axpy(alpha: f32, x: &[f32], y: &mut [f32]) {
        let len = x.len().min(y.len());
        let vector_len = len / LANES * LANES;
        let scale = _mm256_set1_ps(alpha);
        let mut i = 0;
        while i < vector_len {
            // SAFETY: `i + 8 <= vector_len <= len` for both slices.
            unsafe {
                let updated = _mm256_fmadd_ps(
                    scale,
                    _mm256_loadu_ps(x.as_ptr().add(i)),
                    _mm256_loadu_ps(y.as_ptr().add(i)),
                );
                _mm256_storeu_ps(y.as_mut_ptr().add(i), updated);
            }
            i += LANES;
        }
        for (o, v) in y[vector_len..len].iter_mut().zip(&x[vector_len..len]) {
            *o += alpha * v;
        }
    }

    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn matmul(
        w: &[f32],
        xs: &[f32],
        ys: &mut [f32],
        in_dim: usize,
        out_dim: usize,
    ) {
        let vector_len = in_dim / LANES * LANES;
        let mut x_blocks = xs.chunks_exact(BLOCK * in_dim);
        let mut y_blocks = ys.chunks_exact_mut(BLOCK * out_dim);
        for (xb, yb) in (&mut x_blocks).zip(&mut y_blocks) {
            for (o, w_row) in w.chunks_exact(in_dim).enumerate() {
                let mut acc = [_mm256_setzero_ps(); BLOCK];
                let mut i = 0;
                while i < vector_len {
                    // SAFETY: `w_row` has `in_dim` values and `xb` has `BLOCK * in_dim`;
                    // `i + 8 <= vector_len <= in_dim` keeps every load in bounds.
                    unsafe {
                        let wv = _mm256_loadu_ps(w_row.as_ptr().add(i));
                        for (b, a) in acc.iter_mut().enumerate() {
                            let xv = _mm256_loadu_ps(xb.as_ptr().add(b * in_dim + i));
                            *a = _mm256_fmadd_ps(wv, xv, *a);
                        }
                    }
                    i += LANES;
                }
                for (b, a) in acc.iter().enumerate() {
                    // SAFETY: the caller verified AVX2 and FMA.
                    let mut sum = unsafe { horizontal_sum(*a) };
                    let x_row = &xb[b * in_dim..(b + 1) * in_dim];
                    for (wv, xv) in w_row[vector_len..].iter().zip(&x_row[vector_len..]) {
                        sum += wv * xv;
                    }
                    yb[b * out_dim + o] = sum;
                }
            }
        }
        let x_rest = x_blocks.remainder();
        let y_rest = y_blocks.into_remainder();
        for (x, y) in x_rest
            .chunks_exact(in_dim)
            .zip(y_rest.chunks_exact_mut(out_dim))
        {
            for (w_row, out) in w.chunks_exact(in_dim).zip(y.iter_mut()) {
                // SAFETY: the caller verified AVX2 and FMA.
                *out = unsafe { dot(w_row, x) };
            }
        }
    }
}

fn matmul_portable(w: &[f32], xs: &[f32], ys: &mut [f32], in_dim: usize, out_dim: usize) {
    let mut x_blocks = xs.chunks_exact(BLOCK * in_dim);
    let mut y_blocks = ys.chunks_exact_mut(BLOCK * out_dim);
    for (xb, yb) in (&mut x_blocks).zip(&mut y_blocks) {
        let rows: [&[f32]; BLOCK] = std::array::from_fn(|b| &xb[b * in_dim..(b + 1) * in_dim]);
        for (o, w_row) in w.chunks_exact(in_dim).enumerate() {
            let dots = dot_block(w_row, &rows);
            for (b, value) in dots.iter().enumerate() {
                yb[b * out_dim + o] = *value;
            }
        }
    }
    let x_rest = x_blocks.remainder();
    let y_rest = y_blocks.into_remainder();
    for (x, y) in x_rest
        .chunks_exact(in_dim)
        .zip(y_rest.chunks_exact_mut(out_dim))
    {
        for (w_row, out) in w.chunks_exact(in_dim).zip(y.iter_mut()) {
            *out = dot(w_row, x);
        }
    }
}

/// Dot products of one weight row against `BLOCK` input rows at once.
#[inline(always)]
fn dot_block(w: &[f32], rows: &[&[f32]; BLOCK]) -> [f32; BLOCK] {
    let mut acc = [[0.0f32; LANES]; BLOCK];
    let vector_len = w.len() / LANES * LANES;
    let mut i = 0;
    while i < vector_len {
        let wv = &w[i..i + LANES];
        for (a, row) in acc.iter_mut().zip(rows) {
            let xv = &row[i..i + LANES];
            for ((a, w), x) in a.iter_mut().zip(wv).zip(xv) {
                *a += w * x;
            }
        }
        i += LANES;
    }
    let mut out = [0.0f32; BLOCK];
    for ((slot, a), row) in out.iter_mut().zip(&acc).zip(rows) {
        let mut sum: f32 = a.iter().sum();
        for (w, x) in w[vector_len..].iter().zip(&row[vector_len..]) {
            sum += w * x;
        }
        *slot = sum;
    }
    out
}

#[inline(always)]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0.0f32; LANES];
    let a_chunks = a.chunks_exact(LANES);
    let b_chunks = b.chunks_exact(LANES);
    let mut sum = 0.0f32;
    for (x, y) in a_chunks.remainder().iter().zip(b_chunks.remainder()) {
        sum += x * y;
    }
    for (x, y) in a_chunks.zip(b_chunks) {
        for ((a, x), y) in acc.iter_mut().zip(x).zip(y) {
            *a += x * y;
        }
    }
    sum + acc.iter().sum::<f32>()
}

pub(crate) fn add_in_place(x: &mut [f32], delta: &[f32]) {
    for (a, b) in x.iter_mut().zip(delta) {
        *a += b;
    }
}

fn rms_norm(x: &[f32], weight: &[f32], eps: f32, out: &mut [f32]) {
    let d = weight.len();
    for (row, out_row) in x.chunks_exact(d).zip(out.chunks_exact_mut(d)) {
        let mean_square = row.iter().map(|v| v * v).sum::<f32>() / d as f32;
        let scale = 1.0 / (mean_square + eps).sqrt();
        for ((o, v), w) in out_row.iter_mut().zip(row).zip(weight) {
            *o = v * scale * w;
        }
    }
}

/// Rotary position table in the Hugging Face `rotate_half` convention.
pub(crate) struct RopeTable {
    cos: Vec<f32>,
    sin: Vec<f32>,
    head_dim: usize,
    /// Leading dimensions of each head that are rotated (GPT-NeoX rotates only part of each head).
    rotary_dims: usize,
}

impl RopeTable {
    pub(crate) fn new(t_len: usize, head_dim: usize, rotary_dims: usize, theta: f32) -> Self {
        let rotary_dims = rotary_dims.min(head_dim) / 2 * 2;
        let half = rotary_dims / 2;
        let mut cos = Vec::with_capacity(t_len * half);
        let mut sin = Vec::with_capacity(t_len * half);
        for t in 0..t_len {
            for j in 0..half {
                let inv_freq = f64::from(theta).powf(-((2 * j) as f64) / rotary_dims as f64);
                let angle = t as f64 * inv_freq;
                cos.push(angle.cos() as f32);
                sin.push(angle.sin() as f32);
            }
        }
        Self {
            cos,
            sin,
            head_dim,
            rotary_dims,
        }
    }

    /// Rotates every head of every position in place; `x` rows are whole positions.
    pub(crate) fn apply(&self, x: &mut [f32]) {
        let half = self.rotary_dims / 2;
        if half == 0 {
            return;
        }
        let positions = self.cos.len() / half;
        let row_len = x.len() / positions;
        for (t, row) in x.chunks_exact_mut(row_len).enumerate() {
            let cos = &self.cos[t * half..(t + 1) * half];
            let sin = &self.sin[t * half..(t + 1) * half];
            for head in row.chunks_exact_mut(self.head_dim) {
                let (first, second) = head[..self.rotary_dims].split_at_mut(half);
                for (((a, b), c), s) in first.iter_mut().zip(second.iter_mut()).zip(cos).zip(sin) {
                    let (x1, x2) = (*a, *b);
                    *a = x1 * c - x2 * s;
                    *b = x2 * c + x1 * s;
                }
            }
        }
    }
}

pub(crate) struct AttentionInputs<'a> {
    pub(crate) q: &'a [f32],
    pub(crate) k: &'a [f32],
    pub(crate) v: &'a [f32],
    pub(crate) num_heads: usize,
    pub(crate) num_kv_heads: usize,
    pub(crate) head_dim: usize,
}

/// Causal grouped-query attention; returns the per-position concatenation of head outputs.
pub(crate) fn attention(inputs: &AttentionInputs<'_>) -> Vec<f32> {
    let d = inputs.num_heads * inputs.head_dim;
    let t_len = inputs.q.len() / d;
    let mut out = vec![0.0f32; t_len * d];
    par_row_ranges(&mut out, d, |first, chunk| {
        #[cfg(target_arch = "x86_64")]
        if has_avx2_fma() {
            // SAFETY: AVX2 and FMA availability was verified at runtime on the line above.
            unsafe { attention_rows_simd(inputs, first, chunk) };
            return;
        }
        attention_rows(inputs, first, chunk, dot, |alpha, x, y| {
            for (o, v) in y.iter_mut().zip(x) {
                *o += alpha * v;
            }
        });
    });
    out
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn attention_rows_simd(inputs: &AttentionInputs<'_>, first: usize, out: &mut [f32]) {
    attention_rows(
        inputs,
        first,
        out,
        // SAFETY: this function's caller verified AVX2 and FMA.
        |a, b| unsafe { simd::dot(a, b) },
        // SAFETY: as above.
        |alpha, x, y| unsafe { simd::axpy(alpha, x, y) },
    );
}

#[inline(always)]
fn attention_rows(
    inputs: &AttentionInputs<'_>,
    first: usize,
    out: &mut [f32],
    dot: impl Fn(&[f32], &[f32]) -> f32,
    axpy: impl Fn(f32, &[f32], &mut [f32]),
) {
    let hd = inputs.head_dim;
    let d = inputs.num_heads * hd;
    let kv_dim = inputs.num_kv_heads * hd;
    let group = inputs.num_heads / inputs.num_kv_heads;
    let scale = 1.0 / (hd as f32).sqrt();
    let mut scores: Vec<f32> = Vec::new();

    for (r, out_row) in out.chunks_exact_mut(d).enumerate() {
        let t = first + r;
        for h in 0..inputs.num_heads {
            let kv_offset = (h / group) * hd;
            let qh = &inputs.q[t * d + h * hd..t * d + (h + 1) * hd];
            scores.clear();
            let mut max = f32::NEG_INFINITY;
            for s in 0..=t {
                let start = s * kv_dim + kv_offset;
                let score = dot(qh, &inputs.k[start..start + hd]) * scale;
                max = max.max(score);
                scores.push(score);
            }
            let mut total = 0.0f32;
            for score in scores.iter_mut() {
                *score = (*score - max).exp();
                total += *score;
            }
            let inv_total = 1.0 / total;
            let out_head = &mut out_row[h * hd..(h + 1) * hd];
            for (s, weight) in scores.iter().enumerate() {
                let start = s * kv_dim + kv_offset;
                axpy(weight * inv_total, &inputs.v[start..start + hd], out_head);
            }
        }
    }
}

/// Builds a small LLaMA-shaped model with pseudo-random weights for the Studio's live view.
///
/// Some tensors are round-tripped through low-bit formats so its layers have a mixed
/// composition to display. Its predictions are meaningless: it exists so the visual
/// pipeline can run without a downloaded checkpoint, and the GUI labels it as synthetic.
pub fn synthetic_demo(seed: u64) -> (LlamaModel, Vec<Vec<(&'static str, TensorFormat)>>) {
    let (layers_n, d, heads, kv_heads, inter, vocab) =
        (8usize, 64usize, 4usize, 2usize, 128usize, 256usize);
    let kv_dim = d / heads * kv_heads;
    let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    let mut rand = |len: usize, scale: f32| -> Vec<f32> {
        (0..len)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (((state >> 33) as f32 / 2_147_483_648.0) - 1.0) * scale
            })
            .collect()
    };
    let attn_formats = [TensorFormat::Q8_0, TensorFormat::FP16];
    let gate_formats = [
        TensorFormat::Ternary,
        TensorFormat::Q4_0,
        TensorFormat::FP16,
    ];
    let down_formats = [TensorFormat::FP16, TensorFormat::Q8_0, TensorFormat::Q4_0];
    let mut composition = Vec::with_capacity(layers_n);
    let mut layers = Vec::with_capacity(layers_n);
    for i in 0..layers_n {
        let attn = attn_formats[i % 2].clone();
        let mlp = gate_formats[i % 3].clone();
        let down = down_formats[i % 3].clone();
        let mut make = |len: usize, format: &TensorFormat| -> Vec<f32> {
            let mut w = rand(len, 0.2);
            // Shapes are chosen so every format's block size divides them.
            let _ = fake_quantize(&mut w, format);
            w
        };
        layers.push(LlamaLayer {
            attn_norm: vec![1.0; d],
            q: make(d * d, &attn),
            k: make(kv_dim * d, &attn),
            v: make(kv_dim * d, &attn),
            o: make(d * d, &attn),
            mlp_norm: vec![1.0; d],
            gate: make(inter * d, &mlp),
            up: make(inter * d, &mlp),
            down: make(d * inter, &down),
        });
        composition.push(vec![
            ("q", attn.clone()),
            ("k", attn.clone()),
            ("v", attn.clone()),
            ("o", attn),
            ("gate", mlp.clone()),
            ("up", mlp),
            ("down", down),
        ]);
    }
    let model = LlamaModel {
        config: LlamaConfig {
            num_layers: layers_n,
            hidden_size: d,
            num_heads: heads,
            num_kv_heads: kv_heads,
            rms_norm_eps: 1e-5,
            rope_theta: 10_000.0,
        },
        vocab_size: vocab,
        intermediate_size: inter,
        embed: rand(vocab * d, 1.0),
        layers,
        final_norm: vec![1.0; d],
        lm_head: None,
    };
    (model, composition)
}
#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    fn pseudo_random(len: usize, seed: &mut u64, scale: f32) -> Vec<f32> {
        (0..len)
            .map(|_| {
                *seed = seed
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (((*seed >> 33) as f32 / 2_147_483_648.0) - 1.0) * scale
            })
            .collect()
    }

    /// Small model: 2 layers, hidden 16, 4 heads over 2 key/value heads, vocab 40.
    pub(crate) fn tiny_model(scale: f32) -> LlamaModel {
        let (d, heads, kv_heads, inter, vocab, layers) = (16, 4, 2, 32, 40, 2);
        let kv_dim = d / heads * kv_heads;
        let mut seed = 7u64;
        let mut rand = |len: usize| pseudo_random(len, &mut seed, scale);
        let layers = (0..layers)
            .map(|_| LlamaLayer {
                attn_norm: vec![1.0; d],
                q: rand(d * d),
                k: rand(kv_dim * d),
                v: rand(kv_dim * d),
                o: rand(d * d),
                mlp_norm: vec![1.0; d],
                gate: rand(inter * d),
                up: rand(inter * d),
                down: rand(d * inter),
            })
            .collect();
        LlamaModel {
            config: LlamaConfig {
                num_layers: 2,
                hidden_size: d,
                num_heads: heads,
                num_kv_heads: kv_heads,
                rms_norm_eps: 1e-5,
                rope_theta: 10_000.0,
            },
            vocab_size: vocab,
            intermediate_size: inter,
            embed: rand(vocab * d),
            layers,
            final_norm: vec![1.0; d],
            lm_head: None,
        }
    }

    #[test]
    fn blocked_matmul_matches_naive_reference() {
        let (rows, in_dim, out_dim) = (19, 13, 5);
        let mut seed = 3u64;
        let w = pseudo_random(out_dim * in_dim, &mut seed, 1.0);
        let x = pseudo_random(rows * in_dim, &mut seed, 1.0);
        let y = linear(&x, &w, in_dim, out_dim);
        for t in 0..rows {
            for o in 0..out_dim {
                let expected: f32 = (0..in_dim)
                    .map(|i| w[o * in_dim + i] * x[t * in_dim + i])
                    .sum();
                assert!((y[t * out_dim + o] - expected).abs() < 1e-4);
            }
        }
    }

    #[test]
    fn zero_weights_give_uniform_distribution() {
        let model = tiny_model(0.0);
        let nll = model.forward_nll(&[1, 2, 3, 4, 5]).unwrap();
        assert_eq!(nll.len(), 4);
        for value in nll {
            assert!((value - (40.0f32).ln()).abs() < 1e-5);
        }
    }

    #[test]
    fn predictions_depend_on_the_past_and_never_on_the_future() {
        let model = tiny_model(0.6);
        let base: Vec<u32> = vec![3, 17, 8, 25, 1, 30, 12, 9, 22, 5, 14, 2];
        let reference = model.forward_nll(&base).unwrap();

        // Changing the final token may only change the score of that token.
        let mut future = base.clone();
        future[11] = 33;
        let changed = model.forward_nll(&future).unwrap();
        assert_eq!(&reference[..10], &changed[..10]);

        // Changing an early token must reach later predictions through attention.
        let mut past = base.clone();
        past[1] = 36;
        let changed = model.forward_nll(&past).unwrap();
        assert!(reference[5..]
            .iter()
            .zip(&changed[5..])
            .any(|(a, b)| (a - b).abs() > 1e-4));
    }

    #[test]
    fn scheduled_forward_matches_plain_forward_and_validates_inputs() {
        let model = tiny_model(0.5);
        let tokens: Vec<u32> = vec![4, 9, 30, 2, 17, 21, 8];
        let plain = model.forward_nll(&tokens).unwrap();
        let exits = model
            .forward_scheduled(&tokens, &[0, 1], &[0, 1, 2])
            .unwrap();
        assert_eq!(exits.len(), 3);
        assert_eq!(exits[2].layers_applied, 2);
        assert_eq!(exits[2].nll, plain);
        assert_eq!(exits[0].top1.len(), tokens.len() - 1);
        // Repeating a layer changes the result; skipping all layers still reads out.
        let looped = model.forward_scheduled(&tokens, &[0, 1, 1], &[3]).unwrap();
        assert_ne!(looped[0].nll, plain);
        assert!(model.forward_scheduled(&tokens, &[], &[0]).is_ok());
        assert!(model.forward_scheduled(&tokens, &[2], &[1]).is_err());
        assert!(model.forward_scheduled(&tokens, &[0], &[2]).is_err());
    }

    #[test]
    fn traced_forward_matches_plain_forward_and_records_every_tensor() {
        let (model, composition) = super::synthetic_demo(5);
        let tokens: Vec<u32> = (0..24).map(|i| (i * 37 % 256) as u32).collect();
        let schedule: Vec<usize> = (0..8).collect();
        let plain = model.forward_nll(&tokens).unwrap();
        let (exit, trace) = model.forward_traced(&tokens, &schedule).unwrap();
        assert_eq!(exit.nll, plain);
        assert_eq!(trace.layers.len(), 8);
        assert_eq!(composition.len(), 8);
        for layer in &trace.layers {
            assert_eq!(layer.tensors.len(), 7);
            assert!(layer.update_ratio > 0.0 && layer.resid_rms_out > 0.0);
        }
        assert!(trace.anomalies().is_empty());
    }

    #[test]
    fn rejects_out_of_vocabulary_and_short_inputs() {
        let model = tiny_model(0.1);
        assert!(model.forward_nll(&[1]).is_err());
        assert!(model.forward_nll(&[1, 40]).is_err());
    }
}
