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

    /// Gate a measured report written by the verification pipeline
    /// (`examples/production_verification.rs`) instead of supplied values
    #[arg(long, value_name = "REPORT_JSON")]
    pub report: Option<PathBuf>,

    /// Output verification outcome in machine-readable JSON format
    #[arg(long)]
    pub json: bool,
}

/// One evaluated variant from a measured verification report.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ReportVariant {
    pub id: String,
    pub perplexity: f64,
    pub delta_ppl: Option<f64>,
    pub accepted: Option<bool>,
}

/// Reads the measured evaluation from a verification report and returns its variants
/// and gate tolerance. Fails if the report holds no measured evaluation.
pub fn read_measured_report(report: &serde_json::Value) -> Result<(f64, Vec<ReportVariant>)> {
    let evaluation = report
        .get("evaluation")
        .ok_or_else(|| anyhow::anyhow!("Report has no 'evaluation' section"))?;
    if evaluation
        .get("measured")
        .and_then(serde_json::Value::as_bool)
        != Some(true)
    {
        anyhow::bail!("Report holds no measured evaluation; run the verification pipeline with --model-path and --tokens");
    }
    let tolerance = evaluation
        .get("max_ppl_delta")
        .and_then(serde_json::Value::as_f64)
        .ok_or_else(|| anyhow::anyhow!("Report has no 'max_ppl_delta'"))?;
    let variants = evaluation
        .get("variants")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| anyhow::anyhow!("Report has no 'variants' list"))?
        .iter()
        .map(|v| {
            Ok(ReportVariant {
                id: v
                    .get("id")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("unnamed")
                    .to_string(),
                perplexity: v
                    .get("perplexity")
                    .and_then(serde_json::Value::as_f64)
                    .ok_or_else(|| anyhow::anyhow!("Variant without perplexity"))?,
                delta_ppl: v.get("delta_ppl").and_then(serde_json::Value::as_f64),
                accepted: v.get("accepted").and_then(serde_json::Value::as_bool),
            })
        })
        .collect::<Result<Vec<_>>>()?;
    Ok((tolerance, variants))
}

pub fn execute(args: ValidateArgs) -> Result<()> {
    if let Some(path) = &args.report {
        let text = std::fs::read_to_string(path)
            .map_err(|e| anyhow::anyhow!("Failed to read {path:?}: {e}"))?;
        let report: serde_json::Value = serde_json::from_str(&text)?;
        let (tolerance, variants) = read_measured_report(&report)?;
        let failed: Vec<&str> = variants
            .iter()
            .filter(|v| v.accepted == Some(false))
            .map(|v| v.id.as_str())
            .collect();
        if args.json {
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "source": "measured report",
                    "max_ppl_delta": tolerance,
                    "variants": variants,
                    "passed": failed.is_empty(),
                }))?
            );
        } else {
            println!("THME Perplexity Gate (measured report, tolerance Δ ≤ {tolerance})");
            for v in &variants {
                let verdict = match v.accepted {
                    None => "baseline",
                    Some(true) => "PASSED",
                    Some(false) => "FAILED",
                };
                let delta = v
                    .delta_ppl
                    .map_or_else(String::new, |d| format!("  Δ {d:+.4}"));
                println!(
                    "  {:<16} PPL {:>12.4}{delta}  {verdict}",
                    v.id, v.perplexity
                );
            }
        }
        if !failed.is_empty() {
            anyhow::bail!("Measured quality gate failed for [{}]", failed.join(", "));
        }
        return Ok(());
    }
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
            println!("  THME Perplexity Gate (values supplied on the command line)");
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
            anyhow::bail!("Perplexity quality gate failed: {}", gate_res.summary);
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
        anyhow::bail!("Provide --report, --baseline and --candidate scores, --baseline-ppl and --candidate-ppl, or a --suite-file.");
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn measured_report_is_parsed_and_unmeasured_report_is_rejected() {
        let report = serde_json::json!({
            "evaluation": {
                "measured": true,
                "max_ppl_delta": 0.45,
                "variants": [
                    { "id": "baseline", "perplexity": 19.8, "delta_ppl": null, "accepted": null },
                    { "id": "recipe", "perplexity": 48.7, "delta_ppl": 28.9, "accepted": false }
                ]
            }
        });
        let (tolerance, variants) = read_measured_report(&report).unwrap();
        assert_eq!(tolerance, 0.45);
        assert_eq!(variants.len(), 2);
        assert_eq!(variants[1].accepted, Some(false));

        let unmeasured = serde_json::json!({ "evaluation": { "measured": false } });
        assert!(read_measured_report(&unmeasured).is_err());
        assert!(read_measured_report(&serde_json::json!({})).is_err());
    }
}
