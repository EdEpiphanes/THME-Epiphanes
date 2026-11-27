// SPDX-License-Identifier: AGPL-3.0-only
//! LoRA adapters and AdamW for the real training path (THME-3036, v0.2.3.3).
//!
//! The LoRA formulas follow Hugging Face PEFT (Apache-2.0): `W' = W + (alpha / r) * B A`, with
//! `A` initialised uniformly in `[-1/sqrt(in), 1/sqrt(in)]` and `B` at zero, so a fresh adapter
//! changes nothing. Only the formulas are used; no PEFT source is copied. AdamW follows
//! Loshchilov and Hutter (2017): decoupled weight decay and bias correction.
//!
//! shortcut: adapters are f32 on dense CPU copies of the base weights; no dropout, no bias terms,
//! no DoRA or QLoRA, and the adapter file is plain JSON. Upgrade with the quantized-base work
//! (v0.2.3.3) and a binary adapter format when adapters get large.

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

use super::backward::{linear_dw, linear_dx, linear_fwd, loss_and_grads, Real, TrainWeights};
use crate::inference::llama::{LlamaConfig, LlamaLayer, LlamaModel};
use crate::interop::PeftLoraAdapter;

/// Linear layers of one LLaMA block that a LoRA adapter can sit on.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LoraRole {
    Q,
    K,
    V,
    O,
    Gate,
    Up,
    Down,
}

impl LoraRole {
    pub const ALL: [LoraRole; 7] = [
        LoraRole::Q,
        LoraRole::K,
        LoraRole::V,
        LoraRole::O,
        LoraRole::Gate,
        LoraRole::Up,
        LoraRole::Down,
    ];

    /// Short name used in configs: `q`, `k`, `v`, `o`, `gate`, `up`, `down`.
    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.iter().copied().find(|role| role.name() == name)
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Q => "q",
            Self::K => "k",
            Self::V => "v",
            Self::O => "o",
            Self::Gate => "gate",
            Self::Up => "up",
            Self::Down => "down",
        }
    }

    /// `(in_dim, out_dim)` of the base layer this role adapts.
    pub fn dims(self, cfg: &LlamaConfig, inter: usize) -> (usize, usize) {
        let d = cfg.hidden_size;
        let kv = d / cfg.num_heads * cfg.num_kv_heads;
        match self {
            Self::Q | Self::O => (d, d),
            Self::K | Self::V => (d, kv),
            Self::Gate | Self::Up => (d, inter),
            Self::Down => (inter, d),
        }
    }
}

/// One adapter: `A` is `[rank, in_dim]` and `B` is `[out_dim, rank]`, both row-major.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LoraPair<T> {
    pub layer: usize,
    pub role: LoraRole,
    pub in_dim: usize,
    pub out_dim: usize,
    pub rank: usize,
    pub a: Vec<T>,
    pub b: Vec<T>,
}

/// A set of adapters sharing one rank and alpha.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LoraSet<T> {
    pub rank: usize,
    pub alpha: f32,
    pub adapters: Vec<LoraPair<T>>,
}

impl<T: Real> LoraSet<T> {
    pub(crate) fn get(&self, layer: usize, role: LoraRole) -> Option<&LoraPair<T>> {
        self.adapters
            .iter()
            .find(|p| p.layer == layer && p.role == role)
    }

    pub(crate) fn find_mut(&mut self, layer: usize, role: LoraRole) -> Option<&mut LoraPair<T>> {
        self.adapters
            .iter_mut()
            .find(|p| p.layer == layer && p.role == role)
    }

    /// `alpha / rank`, the factor on `B A`.
    pub(crate) fn scale(&self) -> f64 {
        f64::from(self.alpha) / self.rank as f64
    }

    /// The same shape with every value zero (a gradient buffer).
    pub fn zeros_like(&self) -> Self {
        let mut zeros = self.clone();
        for p in &mut zeros.adapters {
            p.a.fill(T::default());
            p.b.fill(T::default());
        }
        zeros
    }

    /// Every adapter value, `A` then `B` for each pair in order.
    pub fn flatten(&self) -> Vec<T> {
        self.adapters
            .iter()
            .flat_map(|p| p.a.iter().chain(&p.b))
            .copied()
            .collect()
    }

    /// Writes values back in the order [`LoraSet::flatten`] produces them.
    pub fn unflatten(&mut self, flat: &[T]) -> Result<()> {
        let expected: usize = self.adapters.iter().map(|p| p.a.len() + p.b.len()).sum();
        if flat.len() != expected {
            bail!("LoRA values hold {expected} entries, got {}", flat.len());
        }
        let mut offset = 0;
        for p in &mut self.adapters {
            let n = p.a.len();
            p.a.copy_from_slice(&flat[offset..offset + n]);
            offset += n;
            let n = p.b.len();
            p.b.copy_from_slice(&flat[offset..offset + n]);
            offset += n;
        }
        Ok(())
    }

    pub fn cast<U: Real>(&self) -> LoraSet<U> {
        LoraSet {
            rank: self.rank,
            alpha: self.alpha,
            adapters: self
                .adapters
                .iter()
                .map(|p| LoraPair {
                    layer: p.layer,
                    role: p.role,
                    in_dim: p.in_dim,
                    out_dim: p.out_dim,
                    rank: p.rank,
                    a: p.a.iter().map(|v| U::from_f64(v.to_f64())).collect(),
                    b: p.b.iter().map(|v| U::from_f64(v.to_f64())).collect(),
                })
                .collect(),
        }
    }

    /// Checks that every tensor holds exactly `rank * in_dim` or `out_dim * rank` values.
    pub fn check_shapes(&self) -> Result<()> {
        if self.rank == 0 {
            bail!("LoRA rank cannot be zero");
        }
        for p in &self.adapters {
            if p.rank != self.rank
                || p.a.len() != p.rank * p.in_dim
                || p.b.len() != p.out_dim * p.rank
            {
                bail!(
                    "LoRA adapter on layer {} ({}) has inconsistent shapes",
                    p.layer,
                    p.role.name()
                );
            }
        }
        Ok(())
    }

    /// Checks that every adapter sits on a layer the model has, with the role's shape.
    pub fn check_against(&self, cfg: &LlamaConfig, inter: usize) -> Result<()> {
        self.check_shapes()?;
        for p in &self.adapters {
            let (in_dim, out_dim) = p.role.dims(cfg, inter);
            if p.layer >= cfg.num_layers || (p.in_dim, p.out_dim) != (in_dim, out_dim) {
                bail!(
                    "LoRA adapter on layer {} ({}) does not fit the model",
                    p.layer,
                    p.role.name()
                );
            }
        }
        Ok(())
    }
}

impl LoraSet<f32> {
    /// Adapters on `roles` of every layer. `A` is seeded uniformly, `B` is zero.
    pub fn new(
        cfg: &LlamaConfig,
        inter: usize,
        roles: &[LoraRole],
        rank: usize,
        alpha: f32,
        seed: u64,
    ) -> Result<Self> {
        if rank == 0 {
            bail!("LoRA rank must be at least 1");
        }
        if roles.is_empty() {
            bail!("LoRA needs at least one role");
        }
        if roles
            .iter()
            .enumerate()
            .any(|(i, r)| roles[..i].contains(r))
        {
            bail!("LoRA roles must be distinct");
        }
        let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        let mut adapters = Vec::with_capacity(cfg.num_layers * roles.len());
        for layer in 0..cfg.num_layers {
            for &role in roles {
                let (in_dim, out_dim) = role.dims(cfg, inter);
                let bound = 1.0 / (in_dim as f32).sqrt();
                let a = (0..rank * in_dim)
                    .map(|_| {
                        state = state
                            .wrapping_mul(6364136223846793005)
                            .wrapping_add(1442695040888963407);
                        ((state >> 11) as f32 / (1u64 << 53) as f32 * 2.0 - 1.0) * bound
                    })
                    .collect();
                adapters.push(LoraPair {
                    layer,
                    role,
                    in_dim,
                    out_dim,
                    rank,
                    a,
                    b: vec![0.0; out_dim * rank],
                });
            }
        }
        Ok(Self {
            rank,
            alpha,
            adapters,
        })
    }

    /// Writes the adapter as JSON.
    pub fn save(&self, path: &Path) -> Result<()> {
        let text = serde_json::to_string(self).context("Failed to encode LoRA adapter")?;
        std::fs::write(path, text).with_context(|| format!("Failed to write {path:?}"))
    }

    /// Reads an adapter written by [`LoraSet::save`] and checks its shapes.
    pub fn load(path: &Path) -> Result<Self> {
        let text =
            std::fs::read_to_string(path).with_context(|| format!("Failed to read {path:?}"))?;
        let set: Self =
            serde_json::from_str(&text).with_context(|| format!("Failed to parse {path:?}"))?;
        set.check_shapes()?;
        Ok(set)
    }
}

/// `u = x A^T`: `x` is `[rows, in_dim]`, the result is `[rows, rank]`.
pub(crate) fn lora_mid<T: Real>(p: &LoraPair<T>, x: &[T]) -> Vec<T> {
    linear_fwd(&p.a, x, p.in_dim, p.rank)
}

/// `scale * u B^T` for `u` of shape `[rows, rank]`; the result is `[rows, out_dim]`.
pub(crate) fn lora_out<T: Real>(p: &LoraPair<T>, u: &[T], scale: T) -> Vec<T> {
    let mut y = linear_fwd(&p.b, u, p.rank, p.out_dim);
    for v in &mut y {
        *v = *v * scale;
    }
    y
}

/// Backward of `y = scale * (x A^T) B^T` given `u = x A^T` and `dy`. Returns `dx` and, when
/// `grad` is given, accumulates `dA` and `dB` into it. The base weights get no gradient here.
pub(crate) fn lora_backward<T: Real>(
    p: &LoraPair<T>,
    x: &[T],
    u: &[T],
    dy: &[T],
    scale: T,
    grad: Option<&mut LoraPair<T>>,
) -> Vec<T> {
    let dys: Vec<T> = dy.iter().map(|&v| v * scale).collect();
    let du = linear_dx(&p.b, &dys, p.rank, p.out_dim);
    let dx = linear_dx(&p.a, &du, p.in_dim, p.rank);
    if let Some(g) = grad {
        linear_dw(u, &dys, p.rank, p.out_dim, &mut g.b);
        linear_dw(x, &du, p.in_dim, p.rank, &mut g.a);
    }
    dx
}

/// AdamW with decoupled weight decay and bias correction (Loshchilov and Hutter, 2017):
/// `p <- p (1 - lr wd)`, then `p <- p - lr m_hat / (sqrt(v_hat) + eps)`.
#[derive(Clone, Debug)]
pub struct AdamW {
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    step: i32,
    m: Vec<f32>,
    v: Vec<f32>,
}

impl AdamW {
    /// Optimizer for `len` parameters with the usual betas (0.9, 0.999) and eps 1e-8.
    pub fn new(len: usize, lr: f32, weight_decay: f32) -> Self {
        Self {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay,
            step: 0,
            m: vec![0.0; len],
            v: vec![0.0; len],
        }
    }

    pub fn step(&mut self, params: &mut [f32], grads: &[f32]) -> Result<()> {
        if params.len() != self.m.len() || grads.len() != self.m.len() {
            bail!(
                "AdamW holds state for {} parameters, got {} parameters and {} gradients",
                self.m.len(),
                params.len(),
                grads.len()
            );
        }
        self.step += 1;
        let bias1 = 1.0 - self.beta1.powi(self.step);
        let bias2 = 1.0 - self.beta2.powi(self.step);
        let decay = 1.0 - self.lr * self.weight_decay;
        for (((p, &g), m), v) in params
            .iter_mut()
            .zip(grads)
            .zip(self.m.iter_mut())
            .zip(self.v.iter_mut())
        {
            *m = self.beta1 * *m + (1.0 - self.beta1) * g;
            *v = self.beta2 * *v + (1.0 - self.beta2) * g * g;
            *p = *p * decay - self.lr * (*m / bias1) / ((*v / bias2).sqrt() + self.eps);
        }
        Ok(())
    }
}

/// Settings for [`train_lora`].
#[derive(Clone, Debug)]
pub struct LoraTrainConfig {
    pub rank: usize,
    pub alpha: f32,
    pub roles: Vec<LoraRole>,
    pub seed: u64,
    pub steps: usize,
    pub learning_rate: f32,
    pub weight_decay: f32,
}

impl Default for LoraTrainConfig {
    fn default() -> Self {
        Self {
            rank: 4,
            alpha: 8.0,
            roles: LoraRole::ALL.to_vec(),
            seed: 1,
            steps: 50,
            learning_rate: 1e-2,
            weight_decay: 0.0,
        }
    }
}

/// Result of [`train_lora`]: the trained adapter and the loss before each optimizer step.
#[derive(Clone, Debug)]
pub struct LoraTraining {
    pub adapter: LoraSet<f32>,
    pub losses: Vec<f32>,
}

/// Fine-tunes LoRA adapters on one token sequence with AdamW. The base weights stay frozen, so
/// the backward pass only produces adapter gradients.
///
/// shortcut: one sequence per step, full-batch AdamW, no held-out split and no learning-rate
/// schedule. Upgrade with the recovery data and overfitting guard (v0.2.3.4, v0.2.3.8).
pub fn train_lora(
    model: &LlamaModel,
    tokens: &[u32],
    config: &LoraTrainConfig,
) -> Result<LoraTraining> {
    let w = TrainWeights::<f32>::from_model(model);
    let mut adapter = LoraSet::<f32>::new(
        &w.config,
        w.inter,
        &config.roles,
        config.rank,
        config.alpha,
        config.seed,
    )?;
    let mut opt = AdamW::new(
        adapter.flatten().len(),
        config.learning_rate,
        config.weight_decay,
    );
    let mut losses = Vec::with_capacity(config.steps);
    for _ in 0..config.steps {
        let (loss, grads) = loss_and_grads(&w, Some(&adapter), tokens, false)?;
        losses.push(loss);
        let grad = grads.lora.context("adapter gradients are missing")?;
        let mut params = adapter.flatten();
        opt.step(&mut params, &grad.flatten())?;
        adapter.unflatten(&params)?;
    }
    Ok(LoraTraining { adapter, losses })
}

/// The model with every adapter merged into its base weight, `W + (alpha / r) B A`, through the
/// existing PEFT merger. The result runs on the ordinary inference path.
pub fn merge_into(model: &LlamaModel, lora: &LoraSet<f32>) -> Result<LlamaModel> {
    lora.check_against(&model.config, model.intermediate_size)?;
    let merge = |layer: usize, role: LoraRole, base: &[f32]| -> Result<Vec<f32>> {
        let Some(p) = lora.get(layer, role) else {
            return Ok(base.to_vec());
        };
        PeftLoraAdapter::merge_lora_weights(
            base, &p.a, &p.b, p.out_dim, p.in_dim, p.rank, lora.alpha,
        )
        .map_err(anyhow::Error::msg)
    };
    let mut layers = Vec::with_capacity(model.layers.len());
    for (l, layer) in model.layers.iter().enumerate() {
        layers.push(LlamaLayer {
            attn_norm: layer.attn_norm.clone(),
            q: merge(l, LoraRole::Q, &layer.q)?,
            k: merge(l, LoraRole::K, &layer.k)?,
            v: merge(l, LoraRole::V, &layer.v)?,
            o: merge(l, LoraRole::O, &layer.o)?,
            mlp_norm: layer.mlp_norm.clone(),
            gate: merge(l, LoraRole::Gate, &layer.gate)?,
            up: merge(l, LoraRole::Up, &layer.up)?,
            down: merge(l, LoraRole::Down, &layer.down)?,
        });
    }
    Ok(LlamaModel {
        config: model.config.clone(),
        vocab_size: model.vocab_size,
        intermediate_size: model.intermediate_size,
        embed: model.embed.clone(),
        layers,
        final_norm: model.final_norm.clone(),
        lm_head: model.lm_head.clone(),
        gpu_backend: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::llama::tests::tiny_model;
    use crate::trainer::backward::fd::{assert_close, numeric_grad, rand_vec};
    use crate::trainer::backward::{forward, loss, loss_and_grads, TrainWeights};

    const TOKENS: [u32; 7] = [3, 17, 5, 9, 30, 2, 11];

    #[test]
    fn role_names_and_shapes_follow_the_llama_block() {
        let cfg = tiny_model(0.0).config;
        assert_eq!(LoraRole::from_name("gate"), Some(LoraRole::Gate));
        assert_eq!(LoraRole::from_name("lm_head"), None);
        assert_eq!(LoraRole::K.dims(&cfg, 32), (16, 8));
        assert_eq!(LoraRole::Down.dims(&cfg, 32), (32, 16));
        assert!(LoraSet::<f32>::new(&cfg, 32, &[LoraRole::Q, LoraRole::Q], 2, 4.0, 1).is_err());
        assert!(LoraSet::<f32>::new(&cfg, 32, &LoraRole::ALL, 0, 4.0, 1).is_err());
    }

    #[test]
    fn a_fresh_adapter_changes_nothing() {
        // B starts at zero, so the adapter adds exactly 0.0 to every projection.
        let model = tiny_model(0.25);
        let w = TrainWeights::<f32>::from_model(&model);
        let lora =
            LoraSet::<f32>::new(&w.config, w.inter, &LoraRole::ALL, 2, 4.0, 3).expect("adapter");
        let base: f32 = loss(&w, None, &TOKENS).expect("loss");
        let with: f32 = loss(&w, Some(&lora), &TOKENS).expect("loss");
        assert_eq!(base, with);
    }

    #[test]
    fn adapter_gradients_match_finite_differences_and_the_base_stays_frozen() {
        let model = tiny_model(0.25);
        let w = TrainWeights::<f32>::from_model(&model).cast::<f64>();
        let mut lora = LoraSet::<f32>::new(&w.config, w.inter, &LoraRole::ALL, 2, 4.0, 9)
            .expect("adapter")
            .cast::<f64>();
        // B is zero at initialisation, which makes dA zero; give both matrices values.
        let values: Vec<f64> = rand_vec(lora.flatten().len(), 5)
            .iter()
            .map(|v| v * 0.3)
            .collect();
        lora.unflatten(&values).expect("same length");

        let (value, grads) = loss_and_grads(&w, Some(&lora), &TOKENS, false).expect("loss");
        assert!(value.is_finite() && value > 0.0);
        let analytic = grads.lora.expect("adapter gradients").flatten();
        let numeric = numeric_grad(&lora.flatten(), &|p: &[f64]| {
            let mut moved = lora.clone();
            moved.unflatten(p).expect("same length");
            loss(&w, Some(&moved), &TOKENS).expect("loss")
        });
        assert_close(&analytic, &numeric, "lora A and B");
        assert!(grads.weights.flatten().iter().all(|&g| g == 0.0));
    }

    #[test]
    fn adamw_matches_the_hand_worked_two_steps() {
        // p0 = 1, lr 0.1, weight decay 0.1, gradients 0.5 then -0.25; betas 0.9 and 0.999.
        // Step 1: decay p = 0.99; m = 0.05, v = 0.00025; m_hat = 0.5, v_hat = 0.25;
        //         p = 0.99 - 0.1 * 0.5 / 0.5 = 0.89.
        // Step 2: decay p = 0.8811; m = 0.02, v = 0.00031225; m_hat = 0.02 / 0.19 = 0.105263,
        //         v_hat = 0.00031225 / 0.001999 = 0.156203; p = 0.8811 - 0.1 * 0.105263 / 0.395230.
        let mut opt = AdamW::new(1, 0.1, 0.1);
        let mut p = [1.0f32];
        opt.step(&mut p, &[0.5]).expect("step 1");
        assert!((p[0] - 0.89).abs() < 1e-6, "step 1 gave {}", p[0]);
        opt.step(&mut p, &[-0.25]).expect("step 2");
        assert!((p[0] - 0.854_466).abs() < 1e-5, "step 2 gave {}", p[0]);
        assert!(opt.step(&mut [0.0, 0.0], &[0.0]).is_err());
    }

    #[test]
    fn adapter_round_trips_through_a_file_and_a_short_tensor_is_refused() {
        let model = tiny_model(0.25);
        let lora = LoraSet::<f32>::new(
            &model.config,
            model.intermediate_size,
            &LoraRole::ALL,
            2,
            4.0,
            3,
        )
        .expect("adapter");
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("adapter.json");
        lora.save(&path).expect("save");
        assert_eq!(LoraSet::<f32>::load(&path).expect("load"), lora);

        let mut bad = lora.clone();
        bad.adapters[0].b.pop();
        std::fs::write(&path, serde_json::to_string(&bad).expect("json")).expect("write");
        assert!(LoraSet::<f32>::load(&path).is_err());
    }

    #[test]
    fn merged_model_logits_equal_adapter_logits() {
        let model = tiny_model(0.25);
        let w = TrainWeights::<f32>::from_model(&model);
        let mut lora =
            LoraSet::<f32>::new(&w.config, w.inter, &LoraRole::ALL, 2, 4.0, 11).expect("adapter");
        let values: Vec<f32> = rand_vec(lora.flatten().len(), 12)
            .iter()
            .map(|v| (v * 0.3) as f32)
            .collect();
        lora.unflatten(&values).expect("same length");

        let adapter = forward(&w, Some(&lora), &TOKENS).expect("forward").logits;
        let last = &adapter[(TOKENS.len() - 1) * w.vocab..];
        let merged = merge_into(&model, &lora).expect("merge");
        let merged_logits = merged.forward_logits(&TOKENS).expect("logits");
        for (a, m) in last.iter().zip(&merged_logits) {
            assert!(
                (a - m).abs() < 1e-4 * m.abs().max(1.0),
                "adapter {a} vs merged {m}"
            );
        }
        let base = model.forward_logits(&TOKENS).expect("logits");
        assert!(
            base.iter()
                .zip(&merged_logits)
                .any(|(b, m)| (b - m).abs() > 1e-3),
            "the merge must change the model"
        );
    }

    #[test]
    fn training_lowers_the_loss_by_thirty_percent_in_fifty_steps() {
        // Unit check of the loop on the tiny model and one fixed batch. Not a real-model result.
        // The scale and learning rate were picked by a scan on this same batch (scale 0.25 to 1.0,
        // rate 0.003 to 0.03): at scale 0.25 the best rate only reached about 0.82 of the start.
        let model = tiny_model(1.0);
        let config = LoraTrainConfig {
            learning_rate: 0.03,
            ..LoraTrainConfig::default()
        };
        let run = train_lora(&model, &TOKENS, &config).expect("train");
        assert_eq!(run.losses.len(), 50);
        assert!(run.losses.iter().all(|l| l.is_finite()));
        let (first, last) = (run.losses[0], run.losses[49]);
        assert!(last <= 0.7 * first, "loss went from {first} to {last}");

        // The trained adapter, evaluated on its own, must also be below the threshold.
        let w = TrainWeights::<f32>::from_model(&model);
        let after: f32 = loss(&w, Some(&run.adapter), &TOKENS).expect("loss");
        assert!(
            after <= 0.7 * first,
            "loss after training is {after}, start {first}"
        );
    }

    #[test]
    fn train_lora_refuses_a_single_token() {
        let model = tiny_model(0.25);
        assert!(train_lora(&model, &[3], &LoraTrainConfig::default()).is_err());
    }
}
