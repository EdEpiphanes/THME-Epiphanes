use anyhow::Result;
use clap::Args;
use std::path::PathBuf;

use crate::quality::gate::{GateSpec, Measure};
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

    /// Maximum allowable perplexity degradation (Delta PPL <= epsilon); the same default as
    /// every other command
    #[arg(long, default_value_t = crate::quality::gate::DEFAULT_MAX_PPL_DELTA)]
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
    pub cross_entropy_nats: Option<f64>,
    pub delta_ppl: Option<f64>,
    pub accepted: Option<bool>,
}

/// Reads the measured evaluation of a report, rebuilds the gate it records (checking the gate's
/// hash), and re-judges every variant from its measured numbers. Older reports without a gate
/// use their `max_ppl_delta`. A stored verdict that disagrees with the gate is an error.
pub fn read_measured_report(report: &serde_json::Value) -> Result<(GateSpec, Vec<ReportVariant>)> {
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
    let gate = match evaluation.get("gate") {
        Some(recorded) => {
            let spec: GateSpec = serde_json::from_value(recorded.clone())
                .map_err(|e| anyhow::anyhow!("Report records an unreadable gate: {e}"))?;
            spec.validate()?;
            if let Some(hash) = recorded.get("hash").and_then(serde_json::Value::as_str) {
                if hash != spec.hash() {
                    anyhow::bail!(
                        "Report's gate {} does not match its recorded hash; the report was edited",
                        spec.id()
                    );
                }
            }
            spec
        }
        None => {
            let tolerance = evaluation
                .get("max_ppl_delta")
                .and_then(serde_json::Value::as_f64)
                .ok_or_else(|| anyhow::anyhow!("Report has neither a gate nor 'max_ppl_delta'"))?;
            GateSpec::perplexity(tolerance)
        }
    };
    let mut variants = evaluation
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
                cross_entropy_nats: v
                    .get("cross_entropy_nats")
                    .and_then(serde_json::Value::as_f64),
                delta_ppl: v.get("delta_ppl").and_then(serde_json::Value::as_f64),
                accepted: v.get("accepted").and_then(serde_json::Value::as_bool),
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let baseline = variants
        .iter()
        .find(|v| v.delta_ppl.is_none())
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("Report has no baseline variant"))?;
    for v in variants.iter_mut().filter(|v| v.delta_ppl.is_some()) {
        let mut measures = vec![Measure::new(
            "perplexity",
            baseline.perplexity,
            v.perplexity,
        )];
        if let (Some(b), Some(c)) = (baseline.cross_entropy_nats, v.cross_entropy_nats) {
            measures.push(Measure::new("cross_entropy_nats", b, c));
        }
        let judged = gate.judge(&measures).accepted;
        if let Some(stored) = v.accepted {
            if stored != judged {
                anyhow::bail!(
                    "Variant '{}' is recorded as {} but gate {} judges it {}",
                    v.id,
                    if stored { "accepted" } else { "rejected" },
                    gate.id(),
                    if judged { "accepted" } else { "rejected" }
                );
            }
        }
        v.accepted = Some(judged);
    }
    Ok((gate, variants))
}

pub fn execute(args: ValidateArgs) -> Result<()> {
    if let Some(path) = &args.report {
        let text = std::fs::read_to_string(path)
            .map_err(|e| anyhow::anyhow!("Failed to read {path:?}: {e}"))?;
        let report: serde_json::Value = serde_json::from_str(&text)?;
        let (gate, variants) = read_measured_report(&report)?;
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
                    "gate": gate.to_report(),
                    "variants": variants,
                    "passed": failed.is_empty(),
                }))?
            );
        } else {
            println!(
                "THME quality gate {} (measured report; verdicts re-judged from the measurements)",
                gate.id()
            );
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
        let (gate, variants) = read_measured_report(&report).unwrap();
        assert_eq!(gate.max_ppl_delta(), Some(0.45));
        assert_eq!(variants.len(), 2);
        assert_eq!(variants[1].accepted, Some(false));

        let unmeasured = serde_json::json!({ "evaluation": { "measured": false } });
        assert!(read_measured_report(&unmeasured).is_err());
        assert!(read_measured_report(&serde_json::json!({})).is_err());
    }

    #[test]
    fn recorded_gates_are_checked_and_verdicts_re_judged() {
        let gate = GateSpec::perplexity(0.45);
        let report = |accepted: bool, hash: String| {
            let mut g = gate.to_report();
            g["hash"] = serde_json::json!(hash);
            serde_json::json!({
                "evaluation": {
                    "measured": true,
                    "gate": g,
                    "variants": [
                        { "id": "baseline", "perplexity": 10.0, "delta_ppl": null },
                        { "id": "recipe", "perplexity": 10.3, "delta_ppl": 0.3, "accepted": accepted }
                    ]
                }
            })
        };
        let (g, v) = read_measured_report(&report(true, gate.hash())).unwrap();
        assert_eq!(g.id(), "thme-perplexity@v1");
        assert_eq!(v[1].accepted, Some(true));
        // A stored verdict that disagrees with the gate is caught.
        assert!(read_measured_report(&report(false, gate.hash())).is_err());
        // An edited gate no longer matches its hash.
        assert!(read_measured_report(&report(true, "0".repeat(64))).is_err());
    }

    /// Every committed measured report still validates, with its verdicts re-judged.
    #[test]
    fn committed_reports_re_judge_to_their_recorded_verdicts() {
        let mut checked = 0;
        for entry in walk(std::path::Path::new("docs/benchmarks")) {
            let Ok(text) = std::fs::read_to_string(&entry) else {
                continue;
            };
            let Ok(value) = serde_json::from_str::<serde_json::Value>(&text) else {
                continue;
            };
            if value["evaluation"]["variants"].is_array() && value["evaluation"]["measured"] == true
            {
                read_measured_report(&value).unwrap_or_else(|e| panic!("{entry:?}: {e:#}"));
                checked += 1;
            }
        }
        assert!(checked > 0, "no committed measured reports found");
    }

    fn walk(dir: &std::path::Path) -> Vec<std::path::PathBuf> {
        let mut out = Vec::new();
        if let Ok(entries) = std::fs::read_dir(dir) {
            for e in entries.flatten() {
                let p = e.path();
                if p.is_dir() {
                    out.extend(walk(&p));
                } else if p.extension().is_some_and(|x| x == "json") {
                    out.push(p);
                }
            }
        }
        out
    }
}
