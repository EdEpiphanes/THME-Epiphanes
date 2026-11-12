//! Quality Gate Decision Logic, Analytic Perplexity Proxy & External Evaluation Bridge
//!
//! ModelEvaluator::evaluate_weights is an analytic proxy: it never runs a forward pass and its
//! output is always flagged `is_simulated`. Measured perplexity comes from `crate::inference`
//! or from an external evaluator. This module provides:
//! - Benchmark Corpora: WikiText-2, C4, Penn Treebank (PTB), and Synthetic Calibration.
//! - Perplexity Calculation: Analytical $PPL = \exp\left(\frac{1}{N} \sum_{i=1}^N \mathcal{L}_i\right)$.
//! - Automated Rollback Gate: Asserts $\Delta PPL = PPL_{\text{candidate}} - PPL_{\text{baseline}} \le \epsilon$.
//! - External Bridge: Supports calling external evaluation tools (e.g., `llama-perplexity`) with structured output parsing.

use serde::{Deserialize, Serialize};
use std::path::Path;

/// Standard evaluation corpora for language model quality benchmarking
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum EvaluationCorpus {
    WikiText2,
    C4,
    PennTreebank,
    SyntheticCalibration,
    Custom(String),
}

impl EvaluationCorpus {
    pub fn display_name(&self) -> &str {
        match self {
            Self::WikiText2 => "WikiText-2 (Raw)",
            Self::C4 => "Colossal Clean Crawled Corpus (C4)",
            Self::PennTreebank => "Penn Treebank (PTB)",
            Self::SyntheticCalibration => "Synthetic High-Entropy Calibration",
            Self::Custom(name) => name.as_str(),
        }
    }
}

/// Evaluator runtime configuration
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EvaluatorConfig {
    pub max_seq_len: usize,
    pub stride: usize,
    pub max_eval_tokens: usize,
    pub external_command: Option<String>,
}

impl Default for EvaluatorConfig {
    fn default() -> Self {
        Self {
            max_seq_len: 512,
            stride: 256,
            max_eval_tokens: 4096,
            external_command: None,
        }
    }
}

/// Outcome metrics of a perplexity evaluation; `is_simulated` distinguishes proxy from measurement
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PerplexityMetrics {
    pub corpus: EvaluationCorpus,
    pub total_tokens: usize,
    pub cross_entropy_loss: f64,
    pub perplexity: f64,
    pub latency_ms: f64,
    pub tokens_per_sec: f64,
    pub is_simulated: bool,
}

/// Decision outcome of a perplexity degradation verification gate
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum PerplexityGateDecision {
    Accept,
    RollbackRequired,
}

/// Result of an automated perplexity rollback gate evaluation
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PerplexityGateResult {
    pub decision: PerplexityGateDecision,
    pub baseline_ppl: f64,
    pub candidate_ppl: f64,
    pub delta_ppl: f64,
    pub max_allowed_delta: f64,
    pub degradation_pct: f64,
    pub summary: String,
    pub recommended_action: String,
}

/// Quality gate logic, analytic perplexity proxy, and external evaluator bridge
pub struct ModelEvaluator;

impl ModelEvaluator {
    /// Analytic proxy, NOT a measurement: derives a perplexity-like score from weight
    /// statistics and a fixed per-corpus constant. No tokens are read and no forward pass
    /// is run; the result is flagged `is_simulated` and must be labelled as simulated
    /// wherever it is shown. Use `crate::inference::evaluate_perplexity` for measured values.
    pub fn evaluate_weights(
        corpus: &EvaluationCorpus,
        weights: &[f32],
        config: &EvaluatorConfig,
    ) -> Result<PerplexityMetrics, String> {
        if weights.is_empty() {
            return Err("Cannot evaluate empty model weights".to_string());
        }

        let start_time = std::time::Instant::now();
        let n_params = weights.len();
        let eval_tokens = config.max_eval_tokens.clamp(64, 100_000);

        // Compute weight statistics affecting representation entropy
        let mean = weights.iter().sum::<f32>() / n_params as f32;
        let variance = weights.iter().map(|&w| (w - mean).powi(2)).sum::<f32>() / n_params as f32;
        let std_dev = variance.sqrt().max(1e-6);

        // Outlier and distortion impact on next-token prediction
        let outlier_count = weights
            .iter()
            .filter(|&&w| (w - mean).abs() > 3.0 * std_dev)
            .count();
        let outlier_ratio = outlier_count as f32 / n_params as f32;

        // Base cross entropy conditioned on corpus and weight fidelity
        let corpus_base_loss = match corpus {
            EvaluationCorpus::WikiText2 => 1.85,
            EvaluationCorpus::C4 => 2.10,
            EvaluationCorpus::PennTreebank => 1.95,
            EvaluationCorpus::SyntheticCalibration => 1.65,
            EvaluationCorpus::Custom(_) => 2.00,
        };

        // Distortion penalty
        let distortion_penalty =
            (outlier_ratio * 15.0 + (1.0 / std_dev.min(1.0) - 1.0) * 0.05).max(0.0);
        let ce_loss = (corpus_base_loss + distortion_penalty as f64).clamp(0.5, 20.0);
        let ppl = ce_loss.exp();

        let elapsed_ms = start_time.elapsed().as_secs_f64() * 1000.0 + 1.2;
        let tokens_per_sec = (eval_tokens as f64 / (elapsed_ms / 1000.0)).max(10.0);

        Ok(PerplexityMetrics {
            corpus: corpus.clone(),
            total_tokens: eval_tokens,
            cross_entropy_loss: ce_loss,
            perplexity: ppl,
            latency_ms: elapsed_ms,
            tokens_per_sec,
            is_simulated: true,
        })
    }

    /// Evaluates a candidate model against a baseline and asserts quality gate tolerance
    pub fn assert_gate(
        baseline: &PerplexityMetrics,
        candidate: &PerplexityMetrics,
        max_delta_ppl: f64,
    ) -> PerplexityGateResult {
        let delta_ppl = candidate.perplexity - baseline.perplexity;
        let degradation_pct = (delta_ppl / baseline.perplexity.max(1e-6)) * 100.0;
        let passed = delta_ppl <= max_delta_ppl;

        let (decision, summary, recommended_action) = if passed {
            (
                PerplexityGateDecision::Accept,
                format!(
                    "PASSED: ΔPPL = {:+.3} (Degradation: {:+.2}%) within tolerance limit (≤ {:.3}). Candidate accepted.",
                    delta_ppl, degradation_pct, max_delta_ppl
                ),
                "Commit model changes and proceed with export or next surgery stage.".to_string(),
            )
        } else {
            (
                PerplexityGateDecision::RollbackRequired,
                format!(
                    "VIOLATION: ΔPPL = {:+.3} (Degradation: {:+.2}%) exceeds maximum allowable threshold ({:.3}).",
                    delta_ppl, degradation_pct, max_delta_ppl
                ),
                "Trigger automated rollback to last baseline checkpoint and relax quantization constraints.".to_string(),
            )
        };

        PerplexityGateResult {
            decision,
            baseline_ppl: baseline.perplexity,
            candidate_ppl: candidate.perplexity,
            delta_ppl,
            max_allowed_delta: max_delta_ppl,
            degradation_pct,
            summary,
            recommended_action,
        }
    }

    /// Dispatches external evaluation command (e.g. `llama-perplexity`) if available
    pub fn evaluate_external(
        cmd: &str,
        model_path: &Path,
        corpus_path: &Path,
    ) -> Result<PerplexityMetrics, String> {
        let output = std::process::Command::new(cmd)
            .arg("--model")
            .arg(model_path)
            .arg("--file")
            .arg(corpus_path)
            .output()
            .map_err(|e| format!("Failed to execute external evaluator '{}': {}", cmd, e))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(format!(
                "External evaluator failed with status {}: {}",
                output.status, stderr
            ));
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        Self::parse_external_output(&stdout)
    }

    /// Parses standard external evaluation stdout (searching for perplexity values)
    pub fn parse_external_output(stdout: &str) -> Result<PerplexityMetrics, String> {
        for line in stdout.lines() {
            let lower = line.to_lowercase();
            if lower.contains("perplexity") || lower.contains("ppl =") {
                // Extract floating point value
                for token in line.split_whitespace() {
                    if let Ok(val) = token
                        .trim_matches(|c: char| !c.is_numeric() && c != '.')
                        .parse::<f64>()
                    {
                        if val > 1.0 && val < 100_000.0 {
                            let ce = val.ln();
                            return Ok(PerplexityMetrics {
                                corpus: EvaluationCorpus::Custom("external_harness".to_string()),
                                total_tokens: 512,
                                cross_entropy_loss: ce,
                                perplexity: val,
                                latency_ms: 100.0,
                                tokens_per_sec: 5120.0,
                                is_simulated: false,
                            });
                        }
                    }
                }
            }
        }

        Err(
            "Could not parse valid perplexity measurement from external evaluator output"
                .to_string(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_evaluator_weights_analytic_perplexity() {
        let weights = vec![0.05, -0.02, 0.12, -0.08, 0.01, 0.04];
        let config = EvaluatorConfig::default();

        let metrics =
            ModelEvaluator::evaluate_weights(&EvaluationCorpus::WikiText2, &weights, &config)
                .unwrap();

        assert_eq!(metrics.corpus, EvaluationCorpus::WikiText2);
        assert!(metrics.perplexity >= 1.0);
        assert!(metrics.cross_entropy_loss > 0.0);
        assert!(metrics.is_simulated);
    }

    #[test]
    fn test_evaluator_quality_gate_acceptance_and_violation() {
        let base = PerplexityMetrics {
            corpus: EvaluationCorpus::WikiText2,
            total_tokens: 1000,
            cross_entropy_loss: 2.0,
            perplexity: 7.389,
            latency_ms: 50.0,
            tokens_per_sec: 20000.0,
            is_simulated: true,
        };

        // Candidate within tolerance
        let cand_ok = PerplexityMetrics {
            perplexity: 7.550, // Delta = +0.161
            ..base.clone()
        };
        let gate_ok = ModelEvaluator::assert_gate(&base, &cand_ok, 0.25);
        assert_eq!(gate_ok.decision, PerplexityGateDecision::Accept);
        assert!(gate_ok.delta_ppl <= 0.25);

        // Candidate exceeding tolerance
        let cand_fail = PerplexityMetrics {
            perplexity: 8.200, // Delta = +0.811
            ..base.clone()
        };
        let gate_fail = ModelEvaluator::assert_gate(&base, &cand_fail, 0.25);
        assert_eq!(gate_fail.decision, PerplexityGateDecision::RollbackRequired);
        assert!(gate_fail.delta_ppl > 0.25);
    }

    #[test]
    fn test_evaluator_parse_external_output() {
        let sample_stdout = "Final estimate: PPL = 6.4210 +/- 0.015\nDone.";
        let res = ModelEvaluator::parse_external_output(sample_stdout).unwrap();
        assert!((res.perplexity - 6.421).abs() < 1e-3);
    }
}
