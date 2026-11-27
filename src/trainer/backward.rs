// SPDX-License-Identifier: AGPL-3.0-only
//! Hand-written CPU backward pass for the LLaMA reference forward pass (THME-3036).
//!
//! Structure follows llm.c (MIT, Andrej Karpathy); Rust rewrite. The forward and backward
//! formulas are the published ones (RMSNorm, rotate-half rotary embedding, causal grouped
//! attention, SwiGLU, softmax cross-entropy). Every operation is checked against central finite
//! differences in the tests below. The model math is `f32`; the `Real` parameter lets the same
//! code run in `f64` for those checks.
//!
//! shortcut: one sequence at a time, no batching, dropout or KV cache, dense CPU copies of the
//! weights, and LLaMA blocks only (no Gemma or NeoX). Upgrade when training needs batches or
//! another architecture.

use anyhow::{bail, Result};
use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub, SubAssign};

use super::lora::{lora_backward, lora_mid, lora_out, LoraPair, LoraRole, LoraSet};
use crate::inference::llama::{LlamaConfig, LlamaModel};

/// Scalar type of the training math: `f32` for real use, `f64` for the finite-difference checks.
pub trait Real:
    Copy
    + Default
    + PartialOrd
    + Send
    + Sync
    + std::fmt::Debug
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + AddAssign
    + SubAssign
{
    fn from_f64(value: f64) -> Self;
    fn to_f64(self) -> f64;
    fn exp(self) -> Self;
    fn sqrt(self) -> Self;
}

macro_rules! impl_real {
    ($t:ty) => {
        impl Real for $t {
            fn from_f64(value: f64) -> Self {
                value as $t
            }
            fn to_f64(self) -> f64 {
                self as f64
            }
            fn exp(self) -> Self {
                <$t>::exp(self)
            }
            fn sqrt(self) -> Self {
                <$t>::sqrt(self)
            }
        }
    };
}
impl_real!(f32);
impl_real!(f64);

fn one<T: Real>() -> T {
    T::from_f64(1.0)
}

fn dot<T: Real>(a: &[T], b: &[T]) -> T {
    a.iter()
        .zip(b)
        .fold(T::default(), |acc, (&x, &y)| acc + x * y)
}

/// `y[t, o] = sum_i w[o, i] * x[t, i]` for a row-major `w` of shape `[out_dim, in_dim]`.
pub(crate) fn linear_fwd<T: Real>(w: &[T], x: &[T], in_dim: usize, out_dim: usize) -> Vec<T> {
    let mut y = vec![T::default(); x.len() / in_dim * out_dim];
    for (xr, yr) in x.chunks_exact(in_dim).zip(y.chunks_exact_mut(out_dim)) {
        for (o, yo) in yr.iter_mut().enumerate() {
            *yo = dot(&w[o * in_dim..(o + 1) * in_dim], xr);
        }
    }
    y
}

/// Gradient with respect to the input: `dx[t, i] = sum_o dy[t, o] * w[o, i]`.
pub(crate) fn linear_dx<T: Real>(w: &[T], dy: &[T], in_dim: usize, out_dim: usize) -> Vec<T> {
    let mut dx = vec![T::default(); dy.len() / out_dim * in_dim];
    for (dyr, dxr) in dy.chunks_exact(out_dim).zip(dx.chunks_exact_mut(in_dim)) {
        for (wo, &g) in w.chunks_exact(in_dim).zip(dyr) {
            for (d, &wi) in dxr.iter_mut().zip(wo) {
                *d += g * wi;
            }
        }
    }
    dx
}

/// Accumulates the weight gradient `dw[o, i] += sum_t dy[t, o] * x[t, i]`.
pub(crate) fn linear_dw<T: Real>(x: &[T], dy: &[T], in_dim: usize, out_dim: usize, dw: &mut [T]) {
    for (xr, dyr) in x.chunks_exact(in_dim).zip(dy.chunks_exact(out_dim)) {
        for (dwo, &g) in dw.chunks_exact_mut(in_dim).zip(dyr) {
            for (d, &xi) in dwo.iter_mut().zip(xr) {
                *d += g * xi;
            }
        }
    }
}

/// RMSNorm over rows of width `g.len()`: `y = x / sqrt(mean(x^2) + eps) * g`. Also returns the
/// inverse root per row, which the backward pass reuses.
pub(crate) fn rms_fwd<T: Real>(x: &[T], g: &[T], eps: T) -> (Vec<T>, Vec<T>) {
    let d = g.len();
    let dn = T::from_f64(d as f64);
    let mut y = vec![T::default(); x.len()];
    let mut inv = Vec::with_capacity(x.len() / d);
    for (xr, yr) in x.chunks_exact(d).zip(y.chunks_exact_mut(d)) {
        let r = one::<T>() / (dot(xr, xr) / dn + eps).sqrt();
        inv.push(r);
        for ((yi, &xi), &gi) in yr.iter_mut().zip(xr).zip(g) {
            *yi = xi * r * gi;
        }
    }
    (y, inv)
}

/// RMSNorm backward. Returns `(dx, dg)` for the upstream gradient `dy`, with
/// `dx_j = g_j dy_j r - r^3 x_j / d * sum_i dy_i g_i x_i` and `dg_j = sum_rows dy_j x_j r`.
pub(crate) fn rms_bwd<T: Real>(x: &[T], g: &[T], inv: &[T], dy: &[T]) -> (Vec<T>, Vec<T>) {
    let d = g.len();
    let dn = T::from_f64(d as f64);
    let mut dx = vec![T::default(); x.len()];
    let mut dg = vec![T::default(); d];
    for (((xr, dyr), dxr), &r) in x
        .chunks_exact(d)
        .zip(dy.chunks_exact(d))
        .zip(dx.chunks_exact_mut(d))
        .zip(inv)
    {
        let s = dyr
            .iter()
            .zip(g)
            .zip(xr)
            .fold(T::default(), |acc, ((&dyi, &gi), &xi)| acc + dyi * gi * xi);
        let k = r * r * r * s / dn;
        for ((((dxj, &dyj), &gj), &xj), dgj) in dxr.iter_mut().zip(dyr).zip(g).zip(xr).zip(&mut dg)
        {
            *dxj = gj * dyj * r - k * xj;
            *dgj += dyj * xj * r;
        }
    }
    (dx, dg)
}

/// Rotary table in the Hugging Face `rotate_half` convention: pair `j` of each head rotates
/// dimensions `j` and `j + head_dim / 2` by `position * theta^(-2j / head_dim)`.
pub(crate) struct Rope<T> {
    cos: Vec<T>,
    sin: Vec<T>,
    half: usize,
}

impl<T: Real> Rope<T> {
    pub(crate) fn new(t_len: usize, head_dim: usize, theta: f64) -> Self {
        let half = head_dim / 2;
        let mut cos = Vec::with_capacity(t_len * half);
        let mut sin = Vec::with_capacity(t_len * half);
        for t in 0..t_len {
            for j in 0..half {
                let angle = t as f64 * theta.powf(-((2 * j) as f64) / head_dim as f64);
                cos.push(T::from_f64(angle.cos()));
                sin.push(T::from_f64(angle.sin()));
            }
        }
        Self { cos, sin, half }
    }

    /// Rotates every head of every position in `x`, whose rows are `heads * head_dim` wide.
    /// `inverse` applies the transpose of the rotation, which is the backward pass.
    pub(crate) fn apply(&self, x: &mut [T], heads: usize, head_dim: usize, inverse: bool) {
        let half = self.half;
        for (t, row) in x.chunks_exact_mut(heads * head_dim).enumerate() {
            let cos = &self.cos[t * half..(t + 1) * half];
            let sin = &self.sin[t * half..(t + 1) * half];
            for head in row.chunks_exact_mut(head_dim) {
                let (a, b) = head.split_at_mut(half);
                for (((ai, bi), &c), &s0) in a.iter_mut().zip(b.iter_mut()).zip(cos).zip(sin) {
                    let s = if inverse { -s0 } else { s0 };
                    let (x1, x2) = (*ai, *bi);
                    *ai = x1 * c - x2 * s;
                    *bi = x2 * c + x1 * s;
                }
            }
        }
    }
}

/// Shape of one attention call: `t_len` positions, `heads` query heads sharing `kv_heads`
/// key/value heads (grouped-query attention), each `head_dim` wide.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Geom {
    pub t_len: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
}

impl Geom {
    fn group(self) -> usize {
        self.heads / self.kv_heads
    }
    fn q_dim(self) -> usize {
        self.heads * self.head_dim
    }
    fn kv_dim(self) -> usize {
        self.kv_heads * self.head_dim
    }
}

/// Causal grouped-query attention. Returns the mixed values `[t, heads * head_dim]` and the
/// softmax probabilities `[heads, t, t]` (entries with a key after the query are zero).
pub(crate) fn attn_fwd<T: Real>(q: &[T], k: &[T], v: &[T], g: Geom) -> (Vec<T>, Vec<T>) {
    let (hd, t_len, qd, kd) = (g.head_dim, g.t_len, g.q_dim(), g.kv_dim());
    let scale = T::from_f64(1.0 / (hd as f64).sqrt());
    let mut out = vec![T::default(); t_len * qd];
    let mut probs = vec![T::default(); g.heads * t_len * t_len];
    for h in 0..g.heads {
        let kv = (h / g.group()) * hd;
        for t in 0..t_len {
            let qh = &q[t * qd + h * hd..][..hd];
            let row = &mut probs[(h * t_len + t) * t_len..][..t_len];
            let live = &mut row[..=t];
            for (s, p) in live.iter_mut().enumerate() {
                *p = scale * dot(qh, &k[s * kd + kv..][..hd]);
            }
            let max = live
                .iter()
                .copied()
                .fold(
                    T::from_f64(f64::NEG_INFINITY),
                    |m, x| if x > m { x } else { m },
                );
            let mut sum = T::default();
            for p in live.iter_mut() {
                *p = (*p - max).exp();
                sum += *p;
            }
            for p in live.iter_mut() {
                *p = *p / sum;
            }
            let oh = &mut out[t * qd + h * hd..][..hd];
            for (s, &p) in live.iter().enumerate() {
                for (o, &vi) in oh.iter_mut().zip(&v[s * kd + kv..][..hd]) {
                    *o += p * vi;
                }
            }
        }
    }
    (out, probs)
}

/// Backward of [`attn_fwd`] given its probabilities. Returns `(dq, dk, dv)`. For each query row,
/// `dS = P * (dP - sum(P * dP))` with `dP_s = dout . v_s`, and `dq`, `dk` carry the score scale.
pub(crate) fn attn_bwd<T: Real>(
    q: &[T],
    k: &[T],
    v: &[T],
    probs: &[T],
    dout: &[T],
    g: Geom,
) -> (Vec<T>, Vec<T>, Vec<T>) {
    let (hd, t_len, qd, kd) = (g.head_dim, g.t_len, g.q_dim(), g.kv_dim());
    let scale = T::from_f64(1.0 / (hd as f64).sqrt());
    let mut dq = vec![T::default(); q.len()];
    let mut dk = vec![T::default(); k.len()];
    let mut dv = vec![T::default(); v.len()];
    let mut dp = vec![T::default(); t_len];
    for h in 0..g.heads {
        let kv = (h / g.group()) * hd;
        for t in 0..t_len {
            let qh = &q[t * qd + h * hd..][..hd];
            let doh = &dout[t * qd + h * hd..][..hd];
            let live = t + 1;
            let p = &probs[(h * t_len + t) * t_len..][..live];
            for (s, dps) in dp[..live].iter_mut().enumerate() {
                *dps = dot(doh, &v[s * kd + kv..][..hd]);
            }
            let mean = p
                .iter()
                .zip(&dp[..live])
                .fold(T::default(), |acc, (&ps, &dps)| acc + ps * dps);
            for (s, (&ps, &dps)) in p.iter().zip(&dp[..live]).enumerate() {
                let ds = ps * (dps - mean) * scale;
                let base = s * kd + kv;
                for (d, &gv) in dv[base..][..hd].iter_mut().zip(doh) {
                    *d += ps * gv;
                }
                for (d, &kk) in dq[t * qd + h * hd..][..hd].iter_mut().zip(&k[base..][..hd]) {
                    *d += ds * kk;
                }
                for (d, &qq) in dk[base..][..hd].iter_mut().zip(qh) {
                    *d += ds * qq;
                }
            }
        }
    }
    (dq, dk, dv)
}

/// SwiGLU gate: `h = silu(g) * u` with `silu(g) = g * sigmoid(g)`.
pub(crate) fn swiglu_fwd<T: Real>(g: &[T], u: &[T]) -> Vec<T> {
    g.iter()
        .zip(u)
        .map(|(&gi, &ui)| gi / (one::<T>() + (-gi).exp()) * ui)
        .collect()
}

/// Backward of [`swiglu_fwd`]: returns `(dg, du)` for the upstream gradient `dh`, using
/// `d silu / dg = sig (1 + g (1 - sig))`.
pub(crate) fn swiglu_bwd<T: Real>(g: &[T], u: &[T], dh: &[T]) -> (Vec<T>, Vec<T>) {
    let one = one::<T>();
    let mut dg = Vec::with_capacity(g.len());
    let mut du = Vec::with_capacity(g.len());
    for ((&gi, &ui), &dhi) in g.iter().zip(u).zip(dh) {
        let sig = one / (one + (-gi).exp());
        dg.push(dhi * ui * sig * (one + gi * (one - sig)));
        du.push(dhi * gi * sig);
    }
    (dg, du)
}

/// Mean next-token cross-entropy over the rows of `logits` (`[targets.len(), vocab]`). Returns
/// the loss and its gradient with respect to the logits, `(softmax - onehot) / rows`. The
/// softmax is max-subtracted and its sum is accumulated in f64.
pub(crate) fn cross_entropy<T: Real>(
    logits: &[T],
    vocab: usize,
    targets: &[u32],
) -> Result<(T, Vec<T>)> {
    if targets.is_empty() || logits.len() != targets.len() * vocab {
        bail!(
            "cross-entropy needs {vocab} logits for each of {} rows, got {} values",
            targets.len(),
            logits.len()
        );
    }
    let rows = targets.len() as f64;
    let mut grad = vec![T::default(); logits.len()];
    let mut total = 0.0f64;
    for ((row, out), &target) in logits
        .chunks_exact(vocab)
        .zip(grad.chunks_exact_mut(vocab))
        .zip(targets)
    {
        let target = target as usize;
        if target >= vocab {
            bail!("target {target} is outside the vocabulary of {vocab}");
        }
        let z: Vec<f64> = row.iter().map(|v| v.to_f64()).collect();
        let max = z.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let sum: f64 = z.iter().map(|&v| (v - max).exp()).sum();
        total += sum.ln() - (z[target] - max);
        for (i, (o, &zi)) in out.iter_mut().zip(&z).enumerate() {
            let onehot = if i == target { 1.0 } else { 0.0 };
            *o = T::from_f64(((zi - max).exp() / sum - onehot) / rows);
        }
    }
    Ok((T::from_f64(total / rows), grad))
}

/// Dense weights of one LLaMA block, in the row-major `[out, in]` layout of the reference.
#[derive(Clone, Debug)]
pub struct LayerWeights<T> {
    pub attn_norm: Vec<T>,
    pub q: Vec<T>,
    pub k: Vec<T>,
    pub v: Vec<T>,
    pub o: Vec<T>,
    pub mlp_norm: Vec<T>,
    pub gate: Vec<T>,
    pub up: Vec<T>,
    pub down: Vec<T>,
}

/// Dense weights of a LLaMA-family model, as the training path reads them.
#[derive(Clone, Debug)]
pub struct TrainWeights<T> {
    pub config: LlamaConfig,
    pub vocab: usize,
    pub inter: usize,
    pub embed: Vec<T>,
    pub layers: Vec<LayerWeights<T>>,
    pub final_norm: Vec<T>,
    /// `None` when the output head is tied to the embedding matrix.
    pub lm_head: Option<Vec<T>>,
}

impl TrainWeights<f32> {
    /// Copies the dense weights of a loaded model.
    pub fn from_model(model: &LlamaModel) -> Self {
        Self {
            config: model.config.clone(),
            vocab: model.vocab_size,
            inter: model.intermediate_size,
            embed: model.embed.clone(),
            layers: model
                .layers
                .iter()
                .map(|l| LayerWeights {
                    attn_norm: l.attn_norm.clone(),
                    q: l.q.clone(),
                    k: l.k.clone(),
                    v: l.v.clone(),
                    o: l.o.clone(),
                    mlp_norm: l.mlp_norm.clone(),
                    gate: l.gate.clone(),
                    up: l.up.clone(),
                    down: l.down.clone(),
                })
                .collect(),
            final_norm: model.final_norm.clone(),
            lm_head: model.lm_head.clone(),
        }
    }
}

impl<T: Real> TrainWeights<T> {
    pub fn cast<U: Real>(&self) -> TrainWeights<U> {
        let conv = |v: &[T]| -> Vec<U> { v.iter().map(|x| U::from_f64(x.to_f64())).collect() };
        TrainWeights {
            config: self.config.clone(),
            vocab: self.vocab,
            inter: self.inter,
            embed: conv(&self.embed),
            layers: self
                .layers
                .iter()
                .map(|l| LayerWeights {
                    attn_norm: conv(&l.attn_norm),
                    q: conv(&l.q),
                    k: conv(&l.k),
                    v: conv(&l.v),
                    o: conv(&l.o),
                    mlp_norm: conv(&l.mlp_norm),
                    gate: conv(&l.gate),
                    up: conv(&l.up),
                    down: conv(&l.down),
                })
                .collect(),
            final_norm: conv(&self.final_norm),
            lm_head: self.lm_head.as_deref().map(conv),
        }
    }

    /// Every tensor, in the order of [`TrainWeights::tensor_names`].
    pub fn tensors(&self) -> Vec<&Vec<T>> {
        let mut out = vec![&self.embed];
        for l in &self.layers {
            out.extend([
                &l.attn_norm,
                &l.q,
                &l.k,
                &l.v,
                &l.o,
                &l.mlp_norm,
                &l.gate,
                &l.up,
                &l.down,
            ]);
        }
        out.push(&self.final_norm);
        out.extend(self.lm_head.as_ref());
        out
    }

    pub fn tensors_mut(&mut self) -> Vec<&mut Vec<T>> {
        let mut out = vec![&mut self.embed];
        for l in &mut self.layers {
            out.extend([
                &mut l.attn_norm,
                &mut l.q,
                &mut l.k,
                &mut l.v,
                &mut l.o,
                &mut l.mlp_norm,
                &mut l.gate,
                &mut l.up,
                &mut l.down,
            ]);
        }
        out.push(&mut self.final_norm);
        out.extend(self.lm_head.as_mut());
        out
    }

    /// Names matching [`TrainWeights::tensors`], e.g. `layers.0.q`.
    pub fn tensor_names(&self) -> Vec<String> {
        let mut names = vec!["embed".to_string()];
        for l in 0..self.layers.len() {
            for part in [
                "attn_norm",
                "q",
                "k",
                "v",
                "o",
                "mlp_norm",
                "gate",
                "up",
                "down",
            ] {
                names.push(format!("layers.{l}.{part}"));
            }
        }
        names.push("final_norm".to_string());
        if self.lm_head.is_some() {
            names.push("lm_head".to_string());
        }
        names
    }

    /// Every weight as one vector, in [`TrainWeights::tensors`] order.
    pub fn flatten(&self) -> Vec<T> {
        self.tensors().into_iter().flatten().copied().collect()
    }

    pub fn unflatten(&mut self, flat: &[T]) -> Result<()> {
        let expected: usize = self.tensors().iter().map(|t| t.len()).sum();
        if flat.len() != expected {
            bail!("weights hold {expected} values, got {}", flat.len());
        }
        let mut offset = 0;
        for t in self.tensors_mut() {
            let n = t.len();
            t.copy_from_slice(&flat[offset..offset + n]);
            offset += n;
        }
        Ok(())
    }

    fn zeros_like(&self) -> Self {
        let mut zeros = self.clone();
        for t in zeros.tensors_mut() {
            t.fill(T::default());
        }
        zeros
    }
}

/// Per-block activations the backward pass reads.
struct LayerCache<T> {
    x_in: Vec<T>,
    n1: Vec<T>,
    inv1: Vec<T>,
    q: Vec<T>,
    k: Vec<T>,
    v: Vec<T>,
    probs: Vec<T>,
    mixed: Vec<T>,
    x_mid: Vec<T>,
    n2: Vec<T>,
    inv2: Vec<T>,
    gate: Vec<T>,
    up: Vec<T>,
    hidden: Vec<T>,
}

/// Everything the forward pass computed for one sequence.
pub(crate) struct Cache<T> {
    tokens: Vec<u32>,
    rope: Rope<T>,
    layers: Vec<LayerCache<T>>,
    x_final: Vec<T>,
    normed: Vec<T>,
    inv_final: Vec<T>,
    /// Unnormalized logits, `[tokens.len(), vocab]`.
    pub(crate) logits: Vec<T>,
}

/// One base linear layer `y = W x`, plus `scale * (x A^T) B^T` when it carries an adapter.
pub(crate) struct Linear<'a, T> {
    w: &'a [T],
    in_dim: usize,
    out_dim: usize,
    lora: Option<&'a LoraPair<T>>,
    scale: T,
}

fn lin<'a, T: Real>(
    w: &'a [T],
    (in_dim, out_dim): (usize, usize),
    lora: Option<&'a LoraPair<T>>,
    scale: T,
) -> Linear<'a, T> {
    Linear {
        w,
        in_dim,
        out_dim,
        lora,
        scale,
    }
}

impl<T: Real> Linear<'_, T> {
    fn forward(&self, x: &[T]) -> Vec<T> {
        let mut y = linear_fwd(self.w, x, self.in_dim, self.out_dim);
        if let Some(p) = self.lora {
            let delta = lora_out(p, &lora_mid(p, x), self.scale);
            accumulate(&mut y, &delta);
        }
        y
    }

    /// Gradient with respect to `x`. `dw` receives the base-weight gradient (only when the base
    /// is trained) and `dlora` the adapter gradient.
    fn backward(
        &self,
        x: &[T],
        dy: &[T],
        dw: Option<&mut [T]>,
        dlora: Option<&mut LoraPair<T>>,
    ) -> Vec<T> {
        let mut dx = linear_dx(self.w, dy, self.in_dim, self.out_dim);
        if let Some(dw) = dw {
            linear_dw(x, dy, self.in_dim, self.out_dim, dw);
        }
        if let Some(p) = self.lora {
            let u = lora_mid(p, x);
            let dxa = lora_backward(p, x, &u, dy, self.scale, dlora);
            accumulate(&mut dx, &dxa);
        }
        dx
    }
}

fn accumulate<T: Real>(dst: &mut [T], src: &[T]) {
    for (d, &s) in dst.iter_mut().zip(src) {
        *d += s;
    }
}

/// The buffer to accumulate into when the base is trained, otherwise nothing.
fn sink<T>(train: bool, buf: &mut [T]) -> Option<&mut [T]> {
    train.then_some(buf)
}

fn pair_mut<T: Real>(
    set: &mut Option<LoraSet<T>>,
    layer: usize,
    role: LoraRole,
) -> Option<&mut LoraPair<T>> {
    set.as_mut()?.find_mut(layer, role)
}

fn head_dim(c: &LlamaConfig) -> Result<usize> {
    if c.num_heads == 0
        || c.num_kv_heads == 0
        || !c.hidden_size.is_multiple_of(c.num_heads)
        || !c.num_heads.is_multiple_of(c.num_kv_heads)
    {
        bail!("unsupported attention shape: {c:?}");
    }
    let hd = c.hidden_size / c.num_heads;
    if !hd.is_multiple_of(2) {
        bail!("head dimension {hd} must be even for rotary embeddings");
    }
    Ok(hd)
}

/// Runs the reference LLaMA forward pass for `tokens`, with an optional adapter, and keeps what
/// the backward pass needs.
pub(crate) fn forward<T: Real>(
    w: &TrainWeights<T>,
    lora: Option<&LoraSet<T>>,
    tokens: &[u32],
) -> Result<Cache<T>> {
    let c = &w.config;
    let t_len = tokens.len();
    if t_len < 2 {
        bail!("training needs at least 2 tokens, got {t_len}");
    }
    if let Some(bad) = tokens.iter().find(|&&t| t as usize >= w.vocab) {
        bail!("token {bad} is outside the vocabulary of {}", w.vocab);
    }
    let hd = head_dim(c)?;
    let (d, inter) = (c.hidden_size, w.inter);
    let kv_dim = hd * c.num_kv_heads;
    let geom = Geom {
        t_len,
        heads: c.num_heads,
        kv_heads: c.num_kv_heads,
        head_dim: hd,
    };
    let eps = T::from_f64(f64::from(c.rms_norm_eps));
    let scale = T::from_f64(lora.map_or(0.0, |l| l.scale()));
    let rope = Rope::new(t_len, hd, f64::from(c.rope_theta));

    let mut x = Vec::with_capacity(t_len * d);
    for &tok in tokens {
        x.extend_from_slice(&w.embed[tok as usize * d..][..d]);
    }
    let mut layers = Vec::with_capacity(w.layers.len());
    for (l, lw) in w.layers.iter().enumerate() {
        let adapter = |role| lora.and_then(|s| s.get(l, role));
        let x_in = x.clone();
        let (n1, inv1) = rms_fwd(&x_in, &lw.attn_norm, eps);
        let mut q = lin(&lw.q, (d, d), adapter(LoraRole::Q), scale).forward(&n1);
        let mut k = lin(&lw.k, (d, kv_dim), adapter(LoraRole::K), scale).forward(&n1);
        let v = lin(&lw.v, (d, kv_dim), adapter(LoraRole::V), scale).forward(&n1);
        rope.apply(&mut q, c.num_heads, hd, false);
        rope.apply(&mut k, c.num_kv_heads, hd, false);
        let (mixed, probs) = attn_fwd(&q, &k, &v, geom);
        let attn_out = lin(&lw.o, (d, d), adapter(LoraRole::O), scale).forward(&mixed);
        let mut x_mid = x_in.clone();
        accumulate(&mut x_mid, &attn_out);
        let (n2, inv2) = rms_fwd(&x_mid, &lw.mlp_norm, eps);
        let gate = lin(&lw.gate, (d, inter), adapter(LoraRole::Gate), scale).forward(&n2);
        let up = lin(&lw.up, (d, inter), adapter(LoraRole::Up), scale).forward(&n2);
        let hidden = swiglu_fwd(&gate, &up);
        let mlp = lin(&lw.down, (inter, d), adapter(LoraRole::Down), scale).forward(&hidden);
        x = x_mid.clone();
        accumulate(&mut x, &mlp);
        layers.push(LayerCache {
            x_in,
            n1,
            inv1,
            q,
            k,
            v,
            probs,
            mixed,
            x_mid,
            n2,
            inv2,
            gate,
            up,
            hidden,
        });
    }
    let (normed, inv_final) = rms_fwd(&x, &w.final_norm, eps);
    let head = w.lm_head.as_deref().unwrap_or(&w.embed[..]);
    let logits = linear_fwd(head, &normed, d, w.vocab);
    Ok(Cache {
        tokens: tokens.to_vec(),
        rope,
        layers,
        x_final: x,
        normed,
        inv_final,
        logits,
    })
}

/// Gradients of the loss: one buffer per base tensor (all zero unless the base is trained) and
/// one per adapter value when an adapter is in use.
pub struct Grads<T> {
    pub weights: TrainWeights<T>,
    pub lora: Option<LoraSet<T>>,
}

/// Backward pass from `dlogits` (`[tokens, vocab]`) through the whole model.
pub(crate) fn backward<T: Real>(
    w: &TrainWeights<T>,
    lora: Option<&LoraSet<T>>,
    cache: &Cache<T>,
    dlogits: &[T],
    train_base: bool,
) -> Grads<T> {
    let c = &w.config;
    let hd = c.hidden_size / c.num_heads;
    let (d, inter, vocab) = (c.hidden_size, w.inter, w.vocab);
    let kv_dim = hd * c.num_kv_heads;
    let geom = Geom {
        t_len: cache.tokens.len(),
        heads: c.num_heads,
        kv_heads: c.num_kv_heads,
        head_dim: hd,
    };
    let scale = T::from_f64(lora.map_or(0.0, |l| l.scale()));
    let train = train_base;
    let mut grads = Grads {
        weights: w.zeros_like(),
        lora: lora.map(LoraSet::zeros_like),
    };

    // logits = normed · head^T, so the head contributes a gradient to the normed states and,
    // when the base is trained, to the head itself (the embedding when tied).
    let head = w.lm_head.as_deref().unwrap_or(&w.embed[..]);
    let dnormed = linear_dx(head, dlogits, d, vocab);
    if train {
        let dhead = match grads.weights.lm_head.as_mut() {
            Some(h) => h,
            None => &mut grads.weights.embed,
        };
        linear_dw(&cache.normed, dlogits, d, vocab, dhead);
    }
    let (mut dx, dgf) = rms_bwd(&cache.x_final, &w.final_norm, &cache.inv_final, &dnormed);
    if train {
        accumulate(&mut grads.weights.final_norm, &dgf);
    }

    for (l, (lw, lc)) in w.layers.iter().zip(&cache.layers).enumerate().rev() {
        let adapter = |role| lora.and_then(|s| s.get(l, role));
        let gw = &mut grads.weights.layers[l];
        let gl = &mut grads.lora;

        // MLP: x_out = x_mid + down(silu(gate(n2)) * up(n2)).
        let dh = lin(&lw.down, (inter, d), adapter(LoraRole::Down), scale).backward(
            &lc.hidden,
            &dx,
            sink(train, &mut gw.down),
            pair_mut(gl, l, LoraRole::Down),
        );
        let (dg, du) = swiglu_bwd(&lc.gate, &lc.up, &dh);
        let mut dn2 = lin(&lw.gate, (d, inter), adapter(LoraRole::Gate), scale).backward(
            &lc.n2,
            &dg,
            sink(train, &mut gw.gate),
            pair_mut(gl, l, LoraRole::Gate),
        );
        let dn2_up = lin(&lw.up, (d, inter), adapter(LoraRole::Up), scale).backward(
            &lc.n2,
            &du,
            sink(train, &mut gw.up),
            pair_mut(gl, l, LoraRole::Up),
        );
        accumulate(&mut dn2, &dn2_up);
        let (dx_norm, dgm) = rms_bwd(&lc.x_mid, &lw.mlp_norm, &lc.inv2, &dn2);
        if train {
            accumulate(&mut gw.mlp_norm, &dgm);
        }
        let mut dx_mid = dx.clone();
        accumulate(&mut dx_mid, &dx_norm);

        // Attention: x_mid = x_in + o(attn(rope(q), rope(k), v)).
        let dmixed = lin(&lw.o, (d, d), adapter(LoraRole::O), scale).backward(
            &lc.mixed,
            &dx_mid,
            sink(train, &mut gw.o),
            pair_mut(gl, l, LoraRole::O),
        );
        let (mut dq, mut dk, dv) = attn_bwd(&lc.q, &lc.k, &lc.v, &lc.probs, &dmixed, geom);
        cache.rope.apply(&mut dq, c.num_heads, hd, true);
        cache.rope.apply(&mut dk, c.num_kv_heads, hd, true);
        let mut dn1 = lin(&lw.q, (d, d), adapter(LoraRole::Q), scale).backward(
            &lc.n1,
            &dq,
            sink(train, &mut gw.q),
            pair_mut(gl, l, LoraRole::Q),
        );
        let dn1_k = lin(&lw.k, (d, kv_dim), adapter(LoraRole::K), scale).backward(
            &lc.n1,
            &dk,
            sink(train, &mut gw.k),
            pair_mut(gl, l, LoraRole::K),
        );
        let dn1_v = lin(&lw.v, (d, kv_dim), adapter(LoraRole::V), scale).backward(
            &lc.n1,
            &dv,
            sink(train, &mut gw.v),
            pair_mut(gl, l, LoraRole::V),
        );
        accumulate(&mut dn1, &dn1_k);
        accumulate(&mut dn1, &dn1_v);
        let (dx_in_norm, dga) = rms_bwd(&lc.x_in, &lw.attn_norm, &lc.inv1, &dn1);
        if train {
            accumulate(&mut gw.attn_norm, &dga);
        }
        dx = dx_mid;
        accumulate(&mut dx, &dx_in_norm);
    }

    if train {
        let d_embed = &mut grads.weights.embed;
        for (t, &tok) in cache.tokens.iter().enumerate() {
            accumulate(&mut d_embed[tok as usize * d..][..d], &dx[t * d..][..d]);
        }
    }
    grads
}

/// Mean next-token cross-entropy of `tokens` (entry `i` scores `tokens[i + 1]`), with an
/// optional adapter.
pub fn loss<T: Real>(w: &TrainWeights<T>, lora: Option<&LoraSet<T>>, tokens: &[u32]) -> Result<T> {
    let cache = forward(w, lora, tokens)?;
    let n = tokens.len() - 1;
    let (value, _) = cross_entropy(&cache.logits[..n * w.vocab], w.vocab, &tokens[1..])?;
    Ok(value)
}

/// [`loss`] together with its gradients. With `train_base` false only the adapter gradients are
/// filled in, which is the LoRA case.
pub fn loss_and_grads<T: Real>(
    w: &TrainWeights<T>,
    lora: Option<&LoraSet<T>>,
    tokens: &[u32],
    train_base: bool,
) -> Result<(T, Grads<T>)> {
    let cache = forward(w, lora, tokens)?;
    let (vocab, n) = (w.vocab, tokens.len() - 1);
    let (value, scored) = cross_entropy(&cache.logits[..n * vocab], vocab, &tokens[1..])?;
    let mut dlogits = vec![T::default(); cache.logits.len()];
    dlogits[..n * vocab].copy_from_slice(&scored);
    Ok((value, backward(w, lora, &cache, &dlogits, train_base)))
}

/// Finite-difference helpers shared by the op tests and the trainer tests.
#[cfg(test)]
pub(crate) mod fd {
    /// Step for central differences. In f64 the truncation error is about 1e-12 and the rounding
    /// error about 1e-10 for losses near 1, so the relative error stays far below 1e-4.
    pub(crate) const STEP: f64 = 1e-6;

    /// d(loss)/d(params[i]) for every coordinate, by central differences.
    pub(crate) fn numeric_grad(params: &[f64], loss: &dyn Fn(&[f64]) -> f64) -> Vec<f64> {
        let mut p = params.to_vec();
        (0..params.len())
            .map(|i| {
                let orig = p[i];
                p[i] = orig + STEP;
                let up = loss(&p);
                p[i] = orig - STEP;
                let down = loss(&p);
                p[i] = orig;
                (up - down) / (2.0 * STEP)
            })
            .collect()
    }

    /// Asserts `|a - n| / max(|a|, |n|, 1e-4) < 1e-4` for every coordinate. The 1e-4 floor keeps
    /// near-zero gradients from failing on absolute noise of about 1e-10.
    pub(crate) fn assert_close(analytic: &[f64], numeric: &[f64], what: &str) {
        assert_eq!(analytic.len(), numeric.len(), "{what}: gradient length");
        // A loss that ignores its inputs has zero numeric and zero analytic gradient; refuse that.
        assert!(
            numeric.iter().any(|n| n.abs() > 1e-12),
            "{what}: numeric gradient is all zero, the check would be vacuous"
        );
        for (i, (a, n)) in analytic.iter().zip(numeric).enumerate() {
            let rel = (a - n).abs() / a.abs().max(n.abs()).max(1e-4);
            assert!(
                rel < 1e-4,
                "{what}[{i}]: analytic {a:e}, numeric {n:e}, relative error {rel:e}"
            );
        }
    }

    /// Deterministic values in [-1, 1).
    pub(crate) fn rand_vec(len: usize, seed: u64) -> Vec<f64> {
        let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        (0..len)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
            })
            .collect()
    }

    /// Scalar objective `sum_i y_i * c_i`, so the upstream gradient of `y` is `c`.
    pub(crate) fn weighted_sum(y: &[f64], c: &[f64]) -> f64 {
        y.iter().zip(c).map(|(a, b)| a * b).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::fd::*;
    use super::*;

    #[test]
    fn matmul_gradients_match_finite_differences() {
        let (rows, in_dim, out_dim) = (3, 4, 5);
        let w = rand_vec(out_dim * in_dim, 11);
        let x = rand_vec(rows * in_dim, 12);
        let c = rand_vec(rows * out_dim, 13);
        let nw = w.len();
        let loss = |p: &[f64]| {
            let (w, x) = p.split_at(nw);
            weighted_sum(&linear_fwd(w, x, in_dim, out_dim), &c)
        };
        let p = [w.clone(), x.clone()].concat();
        let numeric = numeric_grad(&p, &loss);
        let mut dw = vec![0.0; nw];
        linear_dw(&x, &c, in_dim, out_dim, &mut dw);
        let dx = linear_dx(&w, &c, in_dim, out_dim);
        assert_close(&[dw, dx].concat(), &numeric, "matmul");
    }

    #[test]
    fn rms_norm_gradients_match_finite_differences() {
        let (rows, d) = (3, 6);
        let eps = 1e-5;
        let x = rand_vec(rows * d, 21);
        let g = rand_vec(d, 22);
        let c = rand_vec(rows * d, 23);
        let nx = x.len();
        let loss = |p: &[f64]| {
            let (x, g) = p.split_at(nx);
            weighted_sum(&rms_fwd(x, g, eps).0, &c)
        };
        let numeric = numeric_grad(&[x.clone(), g.clone()].concat(), &loss);
        let (_, inv) = rms_fwd(&x, &g, eps);
        let (dx, dg) = rms_bwd(&x, &g, &inv, &c);
        assert_close(&[dx, dg].concat(), &numeric, "rms norm");
    }

    #[test]
    fn rotary_gradients_match_and_inverse_undoes_forward() {
        let (t_len, heads, hd) = (3, 2, 4);
        let n = t_len * heads * hd;
        let x = rand_vec(n, 31);
        let c = rand_vec(n, 32);
        let rope = Rope::<f64>::new(t_len, hd, 10_000.0);
        let loss = |p: &[f64]| {
            let mut y = p.to_vec();
            rope.apply(&mut y, heads, hd, false);
            weighted_sum(&y, &c)
        };
        let numeric = numeric_grad(&x, &loss);
        let mut analytic = c.clone();
        rope.apply(&mut analytic, heads, hd, true);
        assert_close(&analytic, &numeric, "rotary");

        let mut y = x.clone();
        rope.apply(&mut y, heads, hd, false);
        let moved = y
            .iter()
            .zip(&x)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        assert!(
            moved > 1e-2,
            "rotation at positions > 0 must move the values"
        );
        rope.apply(&mut y, heads, hd, true);
        for (a, b) in y.iter().zip(&x) {
            assert!(
                (a - b).abs() < 1e-12,
                "inverse rotation drifted: {a} vs {b}"
            );
        }
    }

    #[test]
    fn causal_gqa_attention_gradients_match_finite_differences() {
        // 4 query heads share 2 key/value heads, so heads 0 and 1 read the same keys.
        let g = Geom {
            t_len: 3,
            heads: 4,
            kv_heads: 2,
            head_dim: 2,
        };
        let q = rand_vec(g.t_len * g.q_dim(), 41);
        let k = rand_vec(g.t_len * g.kv_dim(), 42);
        let v = rand_vec(g.t_len * g.kv_dim(), 43);
        let c = rand_vec(g.t_len * g.q_dim(), 44);
        let (nq, nk) = (q.len(), k.len());
        let loss = |p: &[f64]| {
            let (q, rest) = p.split_at(nq);
            let (k, v) = rest.split_at(nk);
            weighted_sum(&attn_fwd(q, k, v, g).0, &c)
        };
        let p = [q.clone(), k.clone(), v.clone()].concat();
        let numeric = numeric_grad(&p, &loss);
        let (_, probs) = attn_fwd(&q, &k, &v, g);
        let (dq, dk, dv) = attn_bwd(&q, &k, &v, &probs, &c, g);
        assert_close(&[dq, dk, dv].concat(), &numeric, "causal GQA attention");

        for h in 0..g.heads {
            for t in 0..g.t_len {
                let row = &probs[(h * g.t_len + t) * g.t_len..][..g.t_len];
                assert!((row.iter().sum::<f64>() - 1.0).abs() < 1e-12);
                assert!(
                    row[t + 1..].iter().all(|&p| p == 0.0),
                    "future key weighted"
                );
            }
        }
    }

    #[test]
    fn swiglu_gradients_match_finite_differences() {
        let n = 12;
        let g = rand_vec(n, 61);
        let u = rand_vec(n, 62);
        let c = rand_vec(n, 63);
        let loss = |p: &[f64]| {
            let (g, u) = p.split_at(n);
            weighted_sum(&swiglu_fwd(g, u), &c)
        };
        let numeric = numeric_grad(&[g.clone(), u.clone()].concat(), &loss);
        let (dg, du) = swiglu_bwd(&g, &u, &c);
        assert_close(&[dg, du].concat(), &numeric, "swiglu");
    }

    #[test]
    fn cross_entropy_gradient_matches_finite_differences() {
        let (rows, vocab) = (3, 5);
        let z = rand_vec(rows * vocab, 51);
        let targets = [2u32, 0, 4];
        let loss = |p: &[f64]| cross_entropy(p, vocab, &targets).expect("rows").0;
        let (_, d) = cross_entropy(&z, vocab, &targets).expect("rows");
        assert_close(&d, &numeric_grad(&z, &loss), "cross-entropy");
    }

    #[test]
    fn cross_entropy_of_zero_logits_is_log_vocab() {
        let (loss, _) = cross_entropy(&[0.0f64; 10], 5, &[1, 3]).expect("rows");
        assert!((loss - 5f64.ln()).abs() < 1e-12);
    }

    fn model_f64(untied: bool) -> TrainWeights<f64> {
        let model = crate::inference::llama::tests::tiny_model(0.25);
        let mut w = TrainWeights::<f32>::from_model(&model).cast::<f64>();
        if untied {
            w.lm_head = Some(rand_vec(w.vocab * w.config.hidden_size, 77));
        }
        w
    }

    const TOKENS: [u32; 7] = [3, 17, 5, 9, 30, 2, 11];

    #[test]
    fn reference_forward_matches_the_inference_path() {
        let model = crate::inference::llama::tests::tiny_model(0.5);
        let w = TrainWeights::<f32>::from_model(&model);
        let nll = model.forward_nll(&TOKENS).expect("nll");
        let mean = nll.iter().sum::<f32>() / nll.len() as f32;
        let ours: f32 = loss(&w, None, &TOKENS).expect("loss");
        assert!(
            (ours - mean).abs() < 1e-4,
            "loss {ours} vs reference {mean}"
        );

        let logits = forward(&w, None, &TOKENS).expect("forward").logits;
        let last = &logits[(TOKENS.len() - 1) * w.vocab..];
        let reference = model.forward_logits(&TOKENS).expect("logits");
        for (a, b) in last.iter().zip(&reference) {
            assert!((a - b).abs() < 1e-4 * b.abs().max(1.0), "logit {a} vs {b}");
        }
    }

    #[test]
    fn whole_model_gradient_matches_finite_differences_on_every_tensor() {
        // Tied head (the tiny model's default) and an untied lm_head, so both paths are checked.
        for untied in [false, true] {
            let w = model_f64(untied);
            let (value, grads) = loss_and_grads(&w, None, &TOKENS, true).expect("loss");
            assert!(value.is_finite() && value > 0.0);
            let analytic = grads.weights.flatten();
            let numeric = numeric_grad(&w.flatten(), &|p: &[f64]| {
                let mut moved = w.clone();
                moved.unflatten(p).expect("same length");
                loss(&moved, None, &TOKENS).expect("loss")
            });
            let names = w.tensor_names();
            let mut offset = 0;
            for (name, tensor) in names.iter().zip(w.tensors()) {
                let n = tensor.len();
                assert_close(
                    &analytic[offset..offset + n],
                    &numeric[offset..offset + n],
                    name,
                );
                offset += n;
            }
            assert_eq!(offset, analytic.len());
        }
    }

    #[test]
    fn cross_entropy_keeps_the_dominant_class_gradient_in_f32() {
        // Margins above 17 (THME-3036 design note). A softmax that rounds p_target to 1.0 in f32
        // returns a zero gradient here; the direct difference p - onehot must not.
        let vocab = 5;
        let logits: Vec<f32> = vec![20.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 25.0];
        let targets = [0u32, 4];
        let (_, d) = cross_entropy(&logits, vocab, &targets).expect("rows");
        assert!(
            d[0] < 0.0 && d[0].abs() > 1e-12,
            "target gradient lost: {}",
            d[0]
        );
        assert!(d[vocab + 4] < 0.0 && d[vocab + 4].abs() > 1e-12);
        for row in d.chunks_exact(vocab) {
            let sum: f32 = row.iter().sum();
            assert!(sum.abs() < 1e-12, "row gradient sums to {sum}");
        }
    }
}
