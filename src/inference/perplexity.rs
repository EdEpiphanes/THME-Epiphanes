// SPDX-License-Identifier: AGPL-3.0-only
//! Measured perplexity over a pre-tokenized corpus.
//!
//! The corpus is split into consecutive, non-overlapping windows of
//! `context_len` tokens. Every token except the first of each window is scored
//! given the tokens before it in the same window, and perplexity is
//! `exp(mean negative log-likelihood)` over all scored tokens.

use anyhow::{bail, Context, Result};
use serde::Serialize;
use std::path::Path;
use std::time::Instant;

use super::arch::CausalLm;
use crate::quality::{EvaluationCorpus, PerplexityMetrics};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PerplexityConfig {
    /// Tokens per forward pass.
    pub context_len: usize,
    /// Evaluate only this many leading corpus tokens; `None` evaluates all of them.
    pub max_tokens: Option<usize>,
}

#[derive(Debug, Clone, Serialize)]
pub struct PerplexityReport {
    pub perplexity: f64,
    pub cross_entropy_nats: f64,
    pub scored_tokens: usize,
    pub corpus_tokens_used: usize,
    pub windows: usize,
    pub context_len: usize,
    pub elapsed_secs: f64,
    pub tokens_per_sec: f64,
}

impl PerplexityReport {
    /// Converts to the quality-gate metric type, flagged as measured.
    pub fn to_metrics(&self, corpus: EvaluationCorpus) -> PerplexityMetrics {
        PerplexityMetrics {
            corpus,
            total_tokens: self.scored_tokens,
            cross_entropy_loss: self.cross_entropy_nats,
            perplexity: self.perplexity,
            latency_ms: self.elapsed_secs * 1000.0,
            tokens_per_sec: self.tokens_per_sec,
            is_simulated: false,
        }
    }
}

/// Reads a token file: consecutive little-endian `u32` token ids with no header.
pub fn read_token_file(path: &Path) -> Result<Vec<u32>> {
    let bytes =
        std::fs::read(path).with_context(|| format!("Failed to read token file {path:?}"))?;
    if bytes.is_empty() {
        bail!("Token file {path:?} is empty");
    }
    if !bytes.len().is_multiple_of(4) {
        bail!(
            "Token file {path:?} has {} bytes, which is not a whole number of u32 ids",
            bytes.len()
        );
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect())
}

/// Evaluates perplexity, calling `progress(windows_done, windows_total, running_ppl)` per window.
pub fn evaluate_perplexity(
    model: &dyn CausalLm,
    tokens: &[u32],
    config: &PerplexityConfig,
    mut progress: impl FnMut(usize, usize, f64),
) -> Result<PerplexityReport> {
    if config.context_len < 2 {
        bail!(
            "Context length must be at least 2, received {}",
            config.context_len
        );
    }
    let used = config
        .max_tokens
        .map_or(tokens.len(), |m| m.min(tokens.len()));
    let windows: Vec<&[u32]> = tokens[..used]
        .chunks(config.context_len)
        .filter(|w| w.len() >= 2)
        .collect();
    if windows.is_empty() {
        bail!("Corpus has too few tokens ({used}) to score anything");
    }

    let started = Instant::now();
    let mut total_nll = 0.0f64;
    let mut scored = 0usize;
    for (index, window) in windows.iter().enumerate() {
        let nll = model.forward_nll(window)?;
        total_nll += nll.iter().map(|&v| f64::from(v)).sum::<f64>();
        scored += nll.len();
        progress(index + 1, windows.len(), (total_nll / scored as f64).exp());
    }
    if !total_nll.is_finite() {
        bail!("Forward pass produced a non-finite loss; the weights are not usable");
    }

    let elapsed_secs = started.elapsed().as_secs_f64();
    let cross_entropy_nats = total_nll / scored as f64;
    Ok(PerplexityReport {
        perplexity: cross_entropy_nats.exp(),
        cross_entropy_nats,
        scored_tokens: scored,
        corpus_tokens_used: used,
        windows: windows.len(),
        context_len: config.context_len,
        elapsed_secs,
        tokens_per_sec: scored as f64 / elapsed_secs.max(1e-9),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::llama::tests::tiny_model;

    #[test]
    fn uniform_model_has_perplexity_equal_to_vocabulary_size() {
        let model = tiny_model(0.0);
        let tokens: Vec<u32> = (0..50).map(|i| i % 40).collect();
        let config = PerplexityConfig {
            context_len: 16,
            max_tokens: None,
        };
        let mut calls = 0;
        let report = evaluate_perplexity(&model, &tokens, &config, |_, _, _| calls += 1).unwrap();
        // 50 tokens -> windows of 16, 16, 16, 2; one unscored token per window.
        assert_eq!(report.windows, 4);
        assert_eq!(calls, 4);
        assert_eq!(report.scored_tokens, 46);
        assert!((report.perplexity - 40.0).abs() < 1e-3);
        assert!(!report.to_metrics(EvaluationCorpus::WikiText2).is_simulated);
    }

    #[test]
    fn max_tokens_truncates_the_corpus() {
        let model = tiny_model(0.3);
        let tokens: Vec<u32> = (0..64).map(|i| (i * 7) % 40).collect();
        let config = PerplexityConfig {
            context_len: 8,
            max_tokens: Some(20),
        };
        let report = evaluate_perplexity(&model, &tokens, &config, |_, _, _| {}).unwrap();
        assert_eq!(report.corpus_tokens_used, 20);
        assert_eq!(report.windows, 3);
        assert_eq!(report.scored_tokens, 17);
    }

    #[test]
    fn token_file_must_be_whole_u32_ids() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tokens.u32");
        std::fs::write(&path, [1u8, 0, 0, 0, 2, 0, 0, 0]).unwrap();
        assert_eq!(read_token_file(&path).unwrap(), vec![1, 2]);
        std::fs::write(&path, [1u8, 0, 0]).unwrap();
        assert!(read_token_file(&path).is_err());
    }
}
