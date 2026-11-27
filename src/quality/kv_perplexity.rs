// SPDX-License-Identifier: AGPL-3.0-only
//! Perplexity with a KV cache at a chosen storage precision (THME-3082).
//!
//! `thme evaluate` scores whole windows in one pass, which never reads a stored key or value
//! back. Here every window is fed one token at a time through a fresh cache, so each later token
//! attends to the stored history, quantized when the precision is int8 or int4. The gap to the
//! f32 result is what the cache precision costs in quality.

use anyhow::{bail, Context, Result};

use crate::inference::{CausalLm, KvPrecision};

/// Result of [`kv_windowed_perplexity`].
#[derive(Debug, Clone, PartialEq)]
pub struct KvPerplexity {
    pub precision: KvPrecision,
    pub window: usize,
    pub windows: usize,
    pub tokens_scored: usize,
    pub perplexity: f64,
    /// Bytes the cache held after the last window.
    pub kv_cache_bytes: usize,
}

/// `-ln softmax(logits)[target]` in f64 with a stable log-sum-exp.
fn nll(logits: &[f32], target: u32) -> Result<f64> {
    let Some(&picked) = logits.get(target as usize) else {
        bail!(
            "token {target} is outside the vocabulary of {} logits",
            logits.len()
        );
    };
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let sum: f64 = logits.iter().map(|&l| (f64::from(l) - max).exp()).sum();
    Ok(max + sum.ln() - f64::from(picked))
}

/// Negative log-likelihood of each token after the first in `chunk`, fed one token at a time
/// through a fresh cache at `precision`; also returns the bytes the cache held at the end.
pub(crate) fn window_nlls(
    model: &dyn CausalLm,
    chunk: &[u32],
    precision: KvPrecision,
) -> Result<(Vec<f64>, usize)> {
    let mut cache = model
        .new_cache_with_precision(precision)
        .with_context(|| format!("this model has no KV cache at precision {precision:?}"))?;
    let mut out = Vec::with_capacity(chunk.len().saturating_sub(1));
    for t in 0..chunk.len().saturating_sub(1) {
        let logits = model.forward_extend(&mut cache, &chunk[t..=t])?;
        out.push(nll(&logits, chunk[t + 1])?);
    }
    Ok((out, cache.bytes()))
}

/// Perplexity over up to `max_windows` consecutive, non-overlapping windows of `window` tokens,
/// each fed one token at a time through a fresh cache at `precision`.
pub fn kv_windowed_perplexity(
    model: &dyn CausalLm,
    tokens: &[u32],
    window: usize,
    max_windows: usize,
    precision: KvPrecision,
) -> Result<KvPerplexity> {
    if window < 2 {
        bail!("window must be at least 2 tokens, got {window}");
    }
    if max_windows == 0 {
        bail!("max_windows must be at least 1");
    }
    if tokens.len() < window {
        bail!(
            "need at least one window of {window} tokens, the input has {}",
            tokens.len()
        );
    }
    let windows = (tokens.len() / window).min(max_windows);
    let mut total = 0.0f64;
    let mut scored = 0usize;
    let mut kv_cache_bytes = 0usize;
    for w in 0..windows {
        let chunk = &tokens[w * window..(w + 1) * window];
        let (nlls, bytes) = window_nlls(model, chunk, precision)?;
        total += nlls.iter().sum::<f64>();
        scored += nlls.len();
        kv_cache_bytes = bytes;
    }
    Ok(KvPerplexity {
        precision,
        window,
        windows,
        tokens_scored: scored,
        perplexity: (total / scored as f64).exp(),
        kv_cache_bytes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::llama::tests::tiny_model;

    /// A fixed token stream over the 40-token vocabulary of `tiny_model`.
    fn stream(n: usize) -> Vec<u32> {
        let mut x = 12345u64;
        (0..n)
            .map(|_| {
                x = x.wrapping_mul(6364136223846793005).wrapping_add(1);
                ((x >> 33) % 40) as u32
            })
            .collect()
    }

    /// Independent oracle: the NLL of `chunk[t]` from one full forward pass over `chunk[..t]`.
    /// The reference pass needs two tokens, so the first prediction (t = 1) has no oracle.
    fn oracle_nll(model: &dyn CausalLm, chunk: &[u32], t: usize) -> f64 {
        let logits = model.forward_logits(&chunk[..t]).unwrap();
        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
        let lse = max
            + logits
                .iter()
                .map(|&l| (f64::from(l) - max).exp())
                .sum::<f64>()
                .ln();
        lse - f64::from(logits[chunk[t] as usize])
    }

    #[test]
    fn f32_cache_matches_the_prefix_by_prefix_oracle_at_every_position() {
        let model = tiny_model(0.3);
        let tokens = stream(8);
        let (nlls, _) = window_nlls(&model, &tokens, KvPrecision::F32).unwrap();
        assert_eq!(nlls.len(), 7);
        for t in 2..8 {
            let want = oracle_nll(&model, &tokens, t);
            let got = nlls[t - 1];
            assert!(
                ((got - want) / want).abs() < 1e-4,
                "t={t} got {got} want {want}"
            );
        }
    }

    #[test]
    fn perplexity_is_exp_of_the_mean_window_nll() {
        let model = tiny_model(0.3);
        let tokens = stream(20);
        let got = kv_windowed_perplexity(&model, &tokens, 8, 2, KvPrecision::F32).unwrap();
        assert_eq!((got.windows, got.tokens_scored), (2, 14)); // 2 windows x 7 predictions
        let (a, _) = window_nlls(&model, &tokens[..8], KvPrecision::F32).unwrap();
        let (b, _) = window_nlls(&model, &tokens[8..16], KvPrecision::F32).unwrap();
        let mean = (a.iter().sum::<f64>() + b.iter().sum::<f64>()) / 14.0;
        assert!((got.perplexity - mean.exp()).abs() < 1e-9 * mean.exp());
    }

    #[test]
    fn quantized_caches_are_finite_and_smaller_than_f32() {
        let model = tiny_model(0.3);
        let tokens = stream(16);
        let f32r = kv_windowed_perplexity(&model, &tokens, 8, 1, KvPrecision::F32).unwrap();
        for p in [KvPrecision::Int8, KvPrecision::Int4] {
            let r = kv_windowed_perplexity(&model, &tokens, 8, 1, p).unwrap();
            assert!(r.perplexity.is_finite() && r.perplexity > 0.0, "{p:?}");
            assert!(r.kv_cache_bytes < f32r.kv_cache_bytes, "{p:?}");
        }
    }

    #[test]
    fn bad_arguments_are_errors() {
        let model = tiny_model(0.3);
        let tokens = stream(16);
        assert!(kv_windowed_perplexity(&model, &tokens, 1, 1, KvPrecision::F32).is_err());
        assert!(kv_windowed_perplexity(&model, &tokens, 8, 0, KvPrecision::F32).is_err());
        assert!(kv_windowed_perplexity(&model, &tokens[..5], 8, 1, KvPrecision::F32).is_err());
    }
}
