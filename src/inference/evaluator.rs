//! Measured quality evaluator for the co-surgery loop.

use super::arch::ArchConfig;
use super::llama::WeightPrecision;
use super::perplexity::{evaluate_perplexity, PerplexityConfig};
use crate::ir::ModelIR;
use crate::surgery::QualityEvaluator;

/// Scores candidates with the reference forward pass on a fixed token set.
///
/// The candidate's tensors must still reference their source payloads; each tensor is
/// round-tripped through its recipe-selected format before the forward pass.
pub struct MeasuredEvaluator {
    arch: ArchConfig,
    tokens: Vec<u32>,
    config: PerplexityConfig,
}

impl MeasuredEvaluator {
    pub fn new(arch: ArchConfig, tokens: Vec<u32>, config: PerplexityConfig) -> Self {
        Self {
            arch,
            tokens,
            config,
        }
    }

    fn run(&self, model: &ModelIR, precision: WeightPrecision) -> Result<f64, String> {
        let dense = self
            .arch
            .load(model, precision)
            .map_err(|error| format!("{error:#}"))?;
        evaluate_perplexity(&*dense, &self.tokens, &self.config, |_, _, _| {})
            .map(|report| report.perplexity)
            .map_err(|error| format!("{error:#}"))
    }
}

impl QualityEvaluator for MeasuredEvaluator {
    fn baseline_perplexity(&mut self, model: &ModelIR) -> Result<f64, String> {
        self.run(model, WeightPrecision::AsStored)
    }

    fn candidate_perplexity(&mut self, model: &ModelIR) -> Result<f64, String> {
        self.run(model, WeightPrecision::SelectedFormat)
    }

    fn is_measured(&self) -> bool {
        true
    }
}
