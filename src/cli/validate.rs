use anyhow::Result;
use clap::Args;
use std::path::PathBuf;

use crate::quality::{BenchmarkScore, QualityGate};
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
}

pub fn execute(args: ValidateArgs) -> Result<()> {
    if let Some(suite_path) = args.suite_file {
        let file = std::fs::File::open(&suite_path)?;
        let scores: Vec<BenchmarkScore> = serde_json::from_reader(file)?;
        let result = QualityGate::evaluate_suite(&scores);
        ReportPrinter::print_gate_result(&result);
        if result.decision == crate::quality::GateDecision::Reject {
            anyhow::bail!("Quality gate verification failed.");
        }
    } else if let (Some(base), Some(cand)) = (args.baseline, args.candidate) {
        let result = QualityGate::evaluate_candidate(base, cand, args.max_degradation);
        ReportPrinter::print_gate_result(&result);
        if result.decision == crate::quality::GateDecision::Reject {
            anyhow::bail!("Quality gate verification failed.");
        }
    } else {
        anyhow::bail!("Provide either --baseline and --candidate scores, or a --suite-file.");
    }

    Ok(())
}
