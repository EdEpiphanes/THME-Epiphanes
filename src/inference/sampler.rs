// SPDX-License-Identifier: AGPL-3.0-only AND MIT
// Derived from llama.cpp (src/llama-sampler.cpp), Copyright (c) 2023-2026 The ggml authors, MIT License:
// the upstream notice is kept in LICENSES/llama.cpp-MIT.txt. Changes for THME are AGPL-3.0-only.
//! Token sampling chain derived from `llama.cpp` (`src/llama-sampler.cpp` and `common/sampling.cpp`).
//!
//! This module is a derived translation (not clean-room) implementing temperature scaling,
//! top-k, top-p (nucleus), min-p, and Mirostat v1/v2 with `llama.cpp`'s ordering, `min_keep`,
//! tie-breaking, and `mu` update rules.

use std::cmp::Ordering;

use crate::engine::error::{EngineError, EngineResult};

/// Adaptive entropy-targeting sampler mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MirostatMode {
    #[default]
    Disabled,
    V1,
    V2,
}

/// Configuration for [`SamplerChain`].
///
/// Defaults select greedy argmax (`temperature = 0.0`, all truncation filters disabled)
/// so existing greedy decoding behavior is unchanged when used as a drop-in default.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SamplerConfig {
    /// Softmax temperature; `0.0` selects greedy argmax.
    pub temperature: f32,
    /// Keep only the `top_k` highest-logit tokens (`0` disables top-k).
    pub top_k: usize,
    /// Cumulative probability cutoff in `[0.0, 1.0]` (`1.0` disables top-p).
    pub top_p: f32,
    /// Relative floor `min_p * max_prob` in `[0.0, 1.0]` (`0.0` disables min-p).
    pub min_p: f32,
    /// Minimum number of candidates retained by `top_p` and `min_p`.
    pub min_keep: usize,
    /// Mirostat mode (`Disabled`, `V1`, or `V2`).
    pub mirostat: MirostatMode,
    /// Mirostat target surprise `tau > 0.0`.
    pub tau: f32,
    /// Mirostat learning rate `eta > 0.0`.
    pub eta: f32,
    /// Number of top tokens used by Mirostat v1 to estimate Zipf exponent `s_hat`.
    pub mirostat_m: usize,
    /// Deterministic PRNG seed.
    pub seed: u64,
}

impl Default for SamplerConfig {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
            min_keep: 1,
            mirostat: MirostatMode::Disabled,
            tau: 5.0,
            eta: 0.1,
            mirostat_m: 100,
            seed: 0,
        }
    }
}

impl SamplerConfig {
    /// Validates all sampler hyperparameters before constructing a chain.
    pub fn validate(&self) -> EngineResult<()> {
        if !self.temperature.is_finite() || self.temperature < 0.0 {
            return Err(EngineError::InvalidInput(
                "temperature must be finite and >= 0.0".to_string(),
            ));
        }
        if !self.top_p.is_finite() || !(0.0..=1.0).contains(&self.top_p) {
            return Err(EngineError::InvalidInput(
                "top_p must be finite and in [0.0, 1.0]".to_string(),
            ));
        }
        if !self.min_p.is_finite() || !(0.0..=1.0).contains(&self.min_p) {
            return Err(EngineError::InvalidInput(
                "min_p must be finite and in [0.0, 1.0]".to_string(),
            ));
        }
        if !self.tau.is_finite() || self.tau <= 0.0 {
            return Err(EngineError::InvalidInput(
                "tau must be finite and > 0.0".to_string(),
            ));
        }
        if !self.eta.is_finite() || self.eta <= 0.0 {
            return Err(EngineError::InvalidInput(
                "eta must be finite and > 0.0".to_string(),
            ));
        }
        if self.mirostat == MirostatMode::V1 && self.mirostat_m < 2 {
            return Err(EngineError::InvalidInput(
                "mirostat_m must be >= 2 for Mirostat v1".to_string(),
            ));
        }
        Ok(())
    }
}

/// Candidate token entry matching `llama_token_data`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TokenData {
    pub id: u32,
    pub logit: f32,
    pub p: f32,
}

struct TokenDataArray {
    data: Vec<TokenData>,
    selected: Option<usize>,
    sorted: bool,
}

/// Small deterministic 64-bit SplitMix64 PRNG (handles `seed == 0` cleanly).
#[derive(Debug, Clone, Copy)]
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Uniform `f64` in `[0.0, 1.0)` using the top 53 bits.
    fn next_f64(&mut self) -> f64 {
        const SCALE: f64 = 1.0 / ((1_u64 << 53) as f64);
        ((self.next_u64() >> 11) as f64) * SCALE
    }
}

/// Stateful token sampler chain following `llama.cpp`'s ordering and update rules.
#[derive(Debug, Clone)]
pub struct SamplerChain {
    config: SamplerConfig,
    rng: SplitMix64,
    mu: f32,
}

impl SamplerChain {
    /// Builds a validated sampler chain from `config`.
    pub fn new(config: SamplerConfig) -> EngineResult<Self> {
        config.validate()?;
        Ok(Self {
            mu: 2.0 * config.tau,
            rng: SplitMix64::new(config.seed),
            config,
        })
    }

    /// Returns the chain's configuration.
    pub fn config(&self) -> &SamplerConfig {
        &self.config
    }

    /// Returns Mirostat's current surprise threshold `mu`, or `None` when Mirostat is disabled.
    pub fn mirostat_mu(&self) -> Option<f32> {
        match self.config.mirostat {
            MirostatMode::Disabled => None,
            MirostatMode::V1 | MirostatMode::V2 => Some(self.mu),
        }
    }

    /// Samples a single token id from raw `logits`.
    pub fn sample(&mut self, logits: &[f32]) -> EngineResult<u32> {
        if self.config.mirostat == MirostatMode::Disabled && self.config.temperature <= 0.0 {
            let mut cur_p = build_candidates(logits)?;
            let idx = greedy_index(&cur_p.data);
            cur_p.selected = Some(idx);
            return Ok(cur_p.data[idx].id);
        }
        let (token_id, _) = self.sample_with_probs(logits)?;
        Ok(token_id)
    }

    /// Runs the full chain and returns `(selected_token_id, surviving_candidates)` with normalized
    /// probabilities summing to `1.0`.
    pub fn sample_with_probs(&mut self, logits: &[f32]) -> EngineResult<(u32, Vec<TokenData>)> {
        let mut cur_p = build_candidates(logits)?;

        match self.config.mirostat {
            MirostatMode::Disabled => {
                top_k_apply(&mut cur_p, self.config.top_k);
                top_p_apply(&mut cur_p, self.config.top_p, self.config.min_keep);
                min_p_apply(&mut cur_p, self.config.min_p, self.config.min_keep);
                temp_apply(&mut cur_p, self.config.temperature);

                let idx = if self.config.temperature <= 0.0 {
                    softmax_apply(&mut cur_p, false);
                    let best = greedy_index(&cur_p.data);
                    cur_p.selected = Some(best);
                    best
                } else {
                    dist_sample(&mut cur_p, &mut self.rng)
                };
                Ok((cur_p.data[idx].id, cur_p.data))
            }
            MirostatMode::V1 => {
                temp_apply(&mut cur_p, self.config.temperature);
                let idx = self.mirostat_v1_apply(&mut cur_p);
                Ok((cur_p.data[idx].id, cur_p.data))
            }
            MirostatMode::V2 => {
                temp_apply(&mut cur_p, self.config.temperature);
                let idx = self.mirostat_v2_apply(&mut cur_p);
                Ok((cur_p.data[idx].id, cur_p.data))
            }
        }
    }

    fn mirostat_v1_apply(&mut self, cur_p: &mut TokenDataArray) -> usize {
        let n_vocab = cur_p.data.len();
        softmax_apply(cur_p, true);

        let limit = self
            .config
            .mirostat_m
            .saturating_sub(1)
            .min(cur_p.data.len().saturating_sub(1));
        let mut sum_ti_bi = 0.0_f32;
        let mut sum_ti_sq = 0.0_f32;
        let mut zero_tail = false;

        for (i, pair) in cur_p.data.windows(2).take(limit).enumerate() {
            if pair[1].p <= 0.0 {
                zero_tail = i == 0;
                break;
            }
            let t_i = ((i + 2) as f32 / (i + 1) as f32).ln();
            let b_i = (pair[0].p / pair[1].p).ln();
            sum_ti_bi += t_i * b_i;
            sum_ti_sq += t_i * t_i;
        }

        let k = if cur_p.data.len() <= 1 || zero_tail || sum_ti_sq <= 0.0 {
            1
        } else {
            let s_hat = sum_ti_bi / sum_ti_sq;
            if s_hat <= 0.0 || !s_hat.is_finite() {
                cur_p.data.len()
            } else {
                let epsilon_hat = s_hat - 1.0;
                let k_raw = if epsilon_hat.abs() < 1e-6 {
                    2.0_f32.powf(self.mu) / (n_vocab as f32).ln().max(1e-6)
                } else {
                    let denom = 1.0 - (n_vocab as f32).powf(-epsilon_hat);
                    ((epsilon_hat * 2.0_f32.powf(self.mu)) / denom).powf(1.0 / s_hat)
                };
                if k_raw.is_finite() {
                    (k_raw.max(1.0) as usize).min(cur_p.data.len())
                } else {
                    cur_p.data.len()
                }
            }
        };

        top_k_apply(cur_p, k.max(1));
        softmax_apply(cur_p, true);
        let idx = dist_sample(cur_p, &mut self.rng);
        let observed_surprise = -(cur_p.data[idx].p.max(f32::MIN_POSITIVE)).log2();
        let err = observed_surprise - self.config.tau;
        self.mu -= self.config.eta * err;
        idx
    }

    fn mirostat_v2_apply(&mut self, cur_p: &mut TokenDataArray) -> usize {
        softmax_apply(cur_p, true);
        let cutoff = cur_p
            .data
            .iter()
            .position(|candidate| {
                let surprise = if candidate.p > 0.0 {
                    -candidate.p.log2()
                } else {
                    f32::INFINITY
                };
                surprise > self.mu
            })
            .unwrap_or(cur_p.data.len())
            .max(1);
        cur_p.data.truncate(cutoff);
        softmax_apply(cur_p, true);
        let idx = dist_sample(cur_p, &mut self.rng);
        let observed_surprise = -(cur_p.data[idx].p.max(f32::MIN_POSITIVE)).log2();
        let err = observed_surprise - self.config.tau;
        self.mu -= self.config.eta * err;
        idx
    }
}

fn build_candidates(logits: &[f32]) -> EngineResult<TokenDataArray> {
    if logits.is_empty() {
        return Err(EngineError::InvalidInput(
            "logits must not be empty".to_string(),
        ));
    }
    let mut has_finite = false;
    let mut data = Vec::with_capacity(logits.len());
    for (idx, &logit) in logits.iter().enumerate() {
        if logit.is_nan() || (logit.is_infinite() && logit.is_sign_positive()) {
            return Err(EngineError::NonFinite {
                what: "logits".to_string(),
                index: idx,
            });
        }
        if logit.is_finite() {
            has_finite = true;
        }
        let id = u32::try_from(idx)
            .map_err(|_| EngineError::InvalidInput("logits length exceeds u32::MAX".to_string()))?;
        data.push(TokenData { id, logit, p: 0.0 });
    }
    if !has_finite {
        return Err(EngineError::NonFinite {
            what: "logits".to_string(),
            index: 0,
        });
    }
    Ok(TokenDataArray {
        data,
        selected: None,
        sorted: false,
    })
}

fn cmp_token_desc(a: &TokenData, b: &TokenData) -> Ordering {
    b.logit.total_cmp(&a.logit).then_with(|| a.id.cmp(&b.id))
}

fn partial_sort_inplace(cur_p: &mut TokenDataArray, n_partial: usize) {
    if cur_p.sorted || cur_p.data.is_empty() || n_partial == 0 {
        return;
    }
    let k = n_partial.min(cur_p.data.len());
    if k < cur_p.data.len() {
        cur_p.data.select_nth_unstable_by(k - 1, cmp_token_desc);
        cur_p.data.truncate(k);
    }
    cur_p.data.sort_unstable_by(cmp_token_desc);
    cur_p.sorted = true;
}

/// First token achieving the strict maximum logit (lowest token id on ties).
fn greedy_index(data: &[TokenData]) -> usize {
    let mut best_idx = 0;
    let mut best_logit = f32::NEG_INFINITY;
    for (idx, item) in data.iter().enumerate() {
        if idx == 0 || item.logit > best_logit {
            best_idx = idx;
            best_logit = item.logit;
        }
    }
    best_idx
}

fn temp_apply(cur_p: &mut TokenDataArray, temp: f32) {
    if cur_p.data.is_empty() {
        return;
    }
    if temp <= 0.0 {
        let max_i = greedy_index(&cur_p.data);
        for (idx, item) in cur_p.data.iter_mut().enumerate() {
            if idx != max_i {
                item.logit = f32::NEG_INFINITY;
            }
        }
        return;
    }
    for item in &mut cur_p.data {
        item.logit /= temp;
    }
}

fn top_k_apply(cur_p: &mut TokenDataArray, k: usize) {
    if k == 0 || cur_p.data.is_empty() {
        return;
    }
    let k = k.min(cur_p.data.len());
    if !cur_p.sorted {
        partial_sort_inplace(cur_p, k);
    } else {
        cur_p.data.truncate(k);
    }
}

fn softmax_apply(cur_p: &mut TokenDataArray, do_sort: bool) {
    if cur_p.data.is_empty() {
        return;
    }
    if do_sort && !cur_p.sorted {
        partial_sort_inplace(cur_p, cur_p.data.len());
    }
    let max_l = if cur_p.sorted {
        cur_p.data[0].logit
    } else {
        cur_p
            .data
            .iter()
            .map(|d| d.logit)
            .fold(f32::NEG_INFINITY, f32::max)
    };
    let mut sum_cum = 0.0_f64;
    for item in &mut cur_p.data {
        let p = (item.logit - max_l).exp();
        item.p = p;
        sum_cum += f64::from(p);
    }
    if sum_cum > 0.0 {
        for item in &mut cur_p.data {
            item.p = (f64::from(item.p) / sum_cum) as f32;
        }
    }
}

fn top_p_apply(cur_p: &mut TokenDataArray, p: f32, min_keep: usize) {
    if p >= 1.0 || cur_p.data.is_empty() {
        return;
    }
    softmax_apply(cur_p, true);
    let mut cum_sum = 0.0_f32;
    let mut last_idx = cur_p.data.len();
    for (idx, item) in cur_p.data.iter().enumerate() {
        cum_sum += item.p;
        if cum_sum >= p && idx + 1 >= min_keep {
            last_idx = idx + 1;
            break;
        }
    }
    cur_p.data.truncate(last_idx);
}

fn min_p_apply(cur_p: &mut TokenDataArray, p: f32, min_keep: usize) {
    if p <= 0.0 || cur_p.data.is_empty() {
        return;
    }
    if !cur_p.sorted {
        let max_logit = cur_p
            .data
            .iter()
            .map(|d| d.logit)
            .fold(f32::NEG_INFINITY, f32::max);
        let min_logit = max_logit + p.ln();
        let filtered: Vec<TokenData> = cur_p
            .data
            .iter()
            .copied()
            .filter(|d| d.logit >= min_logit)
            .collect();
        if !filtered.is_empty() && filtered.len() >= min_keep {
            cur_p.data = filtered;
            return;
        }
        partial_sort_inplace(cur_p, cur_p.data.len());
    }
    let min_logit = cur_p.data[0].logit + p.ln();
    let keep = cur_p
        .data
        .iter()
        .enumerate()
        .skip(1)
        .find(|&(idx, item)| item.logit < min_logit && idx >= min_keep)
        .map_or(cur_p.data.len(), |(idx, _)| idx);
    cur_p.data.truncate(keep);
}

fn dist_sample(cur_p: &mut TokenDataArray, rng: &mut SplitMix64) -> usize {
    if cur_p.data.is_empty() {
        return 0;
    }
    if cur_p.data.len() == 1 {
        let _ = rng.next_f64();
        cur_p.data[0].p = 1.0;
        cur_p.selected = Some(0);
        return 0;
    }
    let max_l = if cur_p.sorted {
        cur_p.data[0].logit
    } else {
        cur_p
            .data
            .iter()
            .map(|d| d.logit)
            .fold(f32::NEG_INFINITY, f32::max)
    };
    let mut sum_cum = 0.0_f64;
    for item in &mut cur_p.data {
        let p = (item.logit - max_l).exp();
        item.p = p;
        sum_cum += f64::from(p);
    }
    let rnd = rng.next_f64();
    let sum_tgt = sum_cum * rnd;
    let mut sum_run = 0.0_f64;
    let mut selected = cur_p.data.len() - 1;
    let mut found = false;
    for (idx, item) in cur_p.data.iter_mut().enumerate() {
        if !found {
            sum_run += f64::from(item.p);
            if sum_run >= sum_tgt {
                selected = idx;
                found = true;
            }
        }
        if sum_cum > 0.0 {
            item.p = (f64::from(item.p) / sum_cum) as f32;
        }
    }
    cur_p.selected = Some(selected);
    selected
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_default_equals_argmax_and_breaks_ties_low() {
        let mut sampler = SamplerChain::new(SamplerConfig::default()).unwrap();
        assert_eq!(sampler.sample(&[0.5, -1.0, 3.2, 1.0]).unwrap(), 2);
        // Tie between index 1 and index 3 keeps the lower index (1).
        assert_eq!(sampler.sample(&[0.1, 2.5, 1.2, 2.5, -0.4]).unwrap(), 1);
    }

    #[test]
    fn top_k_one_equals_greedy() {
        let mut sampler = SamplerChain::new(SamplerConfig {
            temperature: 1.5,
            top_k: 1,
            seed: 123,
            ..Default::default()
        })
        .unwrap();
        let logits = [0.5, -1.0, 3.2, 1.0];
        for _ in 0..10 {
            assert_eq!(sampler.sample(&logits).unwrap(), 2);
        }
    }

    #[test]
    fn top_p_small_keeps_only_the_head() {
        let mut sampler = SamplerChain::new(SamplerConfig {
            temperature: 1.0,
            top_p: 0.5,
            min_keep: 1,
            seed: 7,
            ..Default::default()
        })
        .unwrap();
        let logits = [5.0, 0.0, -5.0];
        let (token, probs) = sampler.sample_with_probs(&logits).unwrap();
        assert_eq!(token, 0);
        assert_eq!(probs.len(), 1);
        assert_eq!(probs[0].id, 0);
    }

    #[test]
    fn min_p_filters_tokens_below_threshold() {
        let mut sampler = SamplerChain::new(SamplerConfig {
            temperature: 1.0,
            min_p: 0.1,
            min_keep: 1,
            seed: 11,
            ..Default::default()
        })
        .unwrap();
        // Ratios to max: exp(0)=1.0, exp(-1)~=0.368 (>=0.1), exp(-5)~=0.0067 (<0.1).
        let logits = [2.0, 1.0, -3.0];
        let (token, probs) = sampler.sample_with_probs(&logits).unwrap();
        assert!(token == 0 || token == 1);
        assert_eq!(probs.len(), 2);
        let ids: Vec<u32> = probs.iter().map(|d| d.id).collect();
        assert_eq!(ids, vec![0, 1]);
    }

    #[test]
    fn seed_reproducibility_and_divergence() {
        let logits = [0.0_f32, 0.0, 0.0, 0.0];
        let cfg_a = SamplerConfig {
            temperature: 1.0,
            seed: 42,
            ..Default::default()
        };
        let cfg_b = SamplerConfig {
            temperature: 1.0,
            seed: 99,
            ..Default::default()
        };

        let mut s1 = SamplerChain::new(cfg_a).unwrap();
        let mut s2 = SamplerChain::new(cfg_a).unwrap();
        let mut s3 = SamplerChain::new(cfg_b).unwrap();

        let seq1: Vec<u32> = (0..16).map(|_| s1.sample(&logits).unwrap()).collect();
        let seq2: Vec<u32> = (0..16).map(|_| s2.sample(&logits).unwrap()).collect();
        let seq3: Vec<u32> = (0..16).map(|_| s3.sample(&logits).unwrap()).collect();

        assert_eq!(seq1, seq2);
        assert_ne!(seq1, seq3);
    }

    #[test]
    fn mirostat_v2_keeps_mu_finite_and_tracks_target_surprise() {
        let logits = [2.0, 1.8, 1.6, 1.4, 1.2, 1.0, 0.8, 0.6];
        let tau = 2.5_f32;
        let mut sampler = SamplerChain::new(SamplerConfig {
            temperature: 1.0,
            mirostat: MirostatMode::V2,
            tau,
            eta: 0.1,
            seed: 7,
            ..Default::default()
        })
        .unwrap();

        let init_mu = sampler.mirostat_mu().unwrap();
        assert!((init_mu - 2.0 * tau).abs() < 1e-6);

        let mut late_surprises = Vec::new();
        for step in 0..100 {
            let (tok, probs) = sampler.sample_with_probs(&logits).unwrap();
            let mu = sampler.mirostat_mu().unwrap();
            assert!(mu.is_finite());
            if step >= 50 {
                let p = probs.iter().find(|d| d.id == tok).unwrap().p;
                late_surprises.push(-p.log2());
            }
        }
        let final_mu = sampler.mirostat_mu().unwrap();
        // Full 8-token distribution has entropy ~2.89 > tau=2.5, so mu moves down from 5.0.
        assert!(final_mu < init_mu);
        let count = late_surprises.len() as f32;
        let mean_surprise: f32 = late_surprises.iter().sum::<f32>() / count;
        assert!((mean_surprise - tau).abs() < 0.35, "mean={mean_surprise}");
    }

    #[test]
    fn mirostat_v1_keeps_mu_finite() {
        let logits = [3.0, 2.0, 1.2, 0.5, -0.2, -1.0];
        let mut sampler = SamplerChain::new(SamplerConfig {
            temperature: 1.0,
            mirostat: MirostatMode::V1,
            tau: 2.0,
            eta: 0.1,
            mirostat_m: 6,
            seed: 17,
            ..Default::default()
        })
        .unwrap();

        for _ in 0..32 {
            let tok = sampler.sample(&logits).unwrap();
            assert!((tok as usize) < logits.len());
            assert!(sampler.mirostat_mu().unwrap().is_finite());
        }
    }

    #[test]
    fn invalid_inputs_return_errors() {
        let bad_configs = [
            SamplerConfig {
                temperature: -0.1,
                ..Default::default()
            },
            SamplerConfig {
                temperature: f32::NAN,
                ..Default::default()
            },
            SamplerConfig {
                top_p: -0.01,
                ..Default::default()
            },
            SamplerConfig {
                top_p: 1.01,
                ..Default::default()
            },
            SamplerConfig {
                min_p: -0.01,
                ..Default::default()
            },
            SamplerConfig {
                min_p: 1.01,
                ..Default::default()
            },
            SamplerConfig {
                tau: 0.0,
                ..Default::default()
            },
            SamplerConfig {
                eta: 0.0,
                ..Default::default()
            },
            SamplerConfig {
                mirostat: MirostatMode::V1,
                mirostat_m: 1,
                ..Default::default()
            },
        ];
        for cfg in bad_configs {
            assert!(SamplerChain::new(cfg).is_err());
        }

        let mut sampler = SamplerChain::new(SamplerConfig::default()).unwrap();
        assert!(sampler.sample(&[]).is_err());
        assert!(sampler.sample(&[1.0, f32::NAN]).is_err());
        assert!(sampler.sample(&[1.0, f32::INFINITY]).is_err());
        assert!(sampler
            .sample(&[f32::NEG_INFINITY, f32::NEG_INFINITY])
            .is_err());
    }

    #[test]
    fn probabilities_after_chain_sum_to_one() {
        let logits = [2.5, 1.7, 0.9, 0.1, -0.8, -2.0];
        let configs = [
            SamplerConfig::default(),
            SamplerConfig {
                temperature: 0.8,
                top_k: 4,
                top_p: 0.9,
                min_p: 0.05,
                seed: 5,
                ..Default::default()
            },
            SamplerConfig {
                temperature: 1.0,
                mirostat: MirostatMode::V1,
                mirostat_m: 6,
                seed: 5,
                ..Default::default()
            },
            SamplerConfig {
                temperature: 1.0,
                mirostat: MirostatMode::V2,
                seed: 5,
                ..Default::default()
            },
        ];

        for cfg in configs {
            let mut sampler = SamplerChain::new(cfg).unwrap();
            let (_, probs) = sampler.sample_with_probs(&logits).unwrap();
            assert!(!probs.is_empty());
            assert!(probs.iter().all(|d| d.p >= 0.0 && d.p.is_finite()));
            let sum: f64 = probs.iter().map(|d| f64::from(d.p)).sum();
            assert!((sum - 1.0).abs() < 1e-6, "sum was {sum}");
        }
    }
}
