use anyhow::Result;
use clap::Args;
use std::path::PathBuf;

use crate::quality::{
    BenchmarkScore, EvaluationCorpus, ModelEvaluator, PerplexityGateDecision, QualityGate,
};
use crate::report::ReportPrinter;

#[derive(Args, Debug)]
pub struct ValidateArgs {
    /// Baseline benchmark score (e.g. 84.5)
    #[arg(short, long)]
    pub baseline: Option<f64>,

    /// Candidate optimized benchmark score (e.g. 83.9)
    #[arg(short, long)]
    pub candidate: Option<f64>,

    /// Maximum allowed degradation in points (e.g. 1.0)
    #[arg(short, long, default_value_t = 1.0)]
    pub max_degradation: f64,

    /// Optional path to a benchmark suite JSON file
    #[arg(short, long)]
    pub suite_file: Option<PathBuf>,

    /// Evaluation corpus for perplexity verification (wikitext2, c4, ptb, synthetic)
    #[arg(long)]
    pub corpus: Option<String>,

    /// Baseline perplexity score for representation degradation verification
    #[arg(long)]
    pub baseline_ppl: Option<f64>,

    /// Candidate model perplexity score
    #[arg(long)]
    pub candidate_ppl: Option<f64>,

    /// Maximum allowable perplexity degradation (Delta PPL <= epsilon)
    #[arg(long, default_value_t = 0.25)]
    pub max_ppl_delta: f64,

    /// Output verification outcome in machine-readable JSON format
    #[arg(long)]
    pub json: bool,
}

pub fn execute(args: ValidateArgs) -> Result<()> {
    // Check if running perplexity verification
    if let (Some(base_ppl), Some(cand_ppl)) = (args.baseline_ppl, args.candidate_ppl) {
        let corpus = match args.corpus.as_deref().unwrap_or("wikitext2") {
            "c4" => EvaluationCorpus::C4,
            "ptb" => EvaluationCorpus::PennTreebank,
            "synthetic" => EvaluationCorpus::SyntheticCalibration,
            _ => EvaluationCorpus::WikiText2,
        };

        let base_metrics = crate::quality::PerplexityMetrics {
            corpus: corpus.clone(),
            total_tokens: 1024,
            cross_entropy_loss: base_ppl.ln(),
            perplexity: base_ppl,
            latency_ms: 10.0,
            tokens_per_sec: 1000.0,
            is_simulated: true,
        };

        let cand_metrics = crate::quality::PerplexityMetrics {
            corpus,
            total_tokens: 1024,
            cross_entropy_loss: cand_ppl.ln(),
            perplexity: cand_ppl,
            latency_ms: 10.0,
            tokens_per_sec: 1000.0,
            is_simulated: true,
        };

        let gate_res =
            ModelEvaluator::assert_gate(&base_metrics, &cand_metrics, args.max_ppl_delta);

        if args.json {
            println!("{}", serde_json::to_string_pretty(&gate_res)?);
        } else {
            println!("--------------------------------------------------");
            println!("  THME Empirical Perplexity Gate (Milestone v0.1.4)");
            println!("--------------------------------------------------");
            println!("  Baseline PPL   : {:.4}", gate_res.baseline_ppl);
            println!("  Candidate PPL  : {:.4}", gate_res.candidate_ppl);
            println!("  Δ PPL          : {:+.4}", gate_res.delta_ppl);
            println!("  Allowed Limit  : {:.4}", gate_res.max_allowed_delta);
            println!("  Degradation    : {:+.2}%", gate_res.degradation_pct);
            println!("  Decision       : {:?}", gate_res.decision);
            println!("  Summary        : {}", gate_res.summary);
            println!("  Action         : {}", gate_res.recommended_action);
            println!("--------------------------------------------------");
        }

        if gate_res.decision == PerplexityGateDecision::RollbackRequired {
            anyhow::bail!(
                "Empirical perplexity quality gate failed: {}",
                gate_res.summary
            );
        }
        return Ok(());
    }

    if let Some(suite_path) = args.suite_file {
        let file = std::fs::File::open(&suite_path)?;
        let scores: Vec<BenchmarkScore> = serde_json::from_reader(file)?;
        let result = QualityGate::evaluate_suite(&scores);
        if args.json {
            println!("{}", serde_json::to_string_pretty(&result)?);
        } else {
            ReportPrinter::print_gate_result(&result);
        }
        if result.decision == crate::quality::GateDecision::Reject {
            anyhow::bail!("Quality gate verification failed.");
        }
    } else if let (Some(base), Some(cand)) = (args.baseline, args.candidate) {
        let result = QualityGate::evaluate_candidate(base, cand, args.max_degradation);
        if args.json {
            println!("{}", serde_json::to_string_pretty(&result)?);
        } else {
            ReportPrinter::print_gate_result(&result);
        }
        if result.decision == crate::quality::GateDecision::Reject {
            anyhow::bail!("Quality gate verification failed.");
        }
    } else {
        anyhow::bail!("Provide either --baseline and --candidate scores, --baseline-ppl and --candidate-ppl, or a --suite-file.");
    }

    Ok(())
}
