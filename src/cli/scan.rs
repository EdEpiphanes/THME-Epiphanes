// SPDX-License-Identifier: AGPL-3.0-only
//! `thme scan`: static security scan of a model file or folder. Nothing is loaded or run.

use anyhow::{bail, Result};
use clap::Args;
use std::path::PathBuf;

use crate::scan::{scan, Severity};

#[derive(Args, Debug)]
pub struct ScanArgs {
    /// Model file or folder (GGUF, SafeTensors, PyTorch .bin/.pt/.pth/.ckpt, pickle, .epiph)
    #[arg(value_name = "PATH")]
    pub path: PathBuf,

    /// Also write the findings as JSON
    #[arg(long)]
    pub report: Option<PathBuf>,

    /// Exit with an error at this severity or worse: medium, high or critical
    #[arg(long, default_value = "high")]
    pub fail_on: String,
}

pub fn execute(args: ScanArgs) -> Result<()> {
    let threshold = match args.fail_on.as_str() {
        "medium" => Severity::Medium,
        "high" => Severity::High,
        "critical" => Severity::Critical,
        other => bail!("--fail-on must be medium, high or critical, not '{other}'"),
    };
    let report = scan(&args.path)?;
    println!(
        "Scanned {} files: {} ({} findings)",
        report.files_scanned,
        report.verdict(),
        report.findings.len()
    );
    for f in &report.findings {
        println!("  [{:?}] {} {}: {}", f.severity, f.rule, f.file, f.detail);
    }
    println!("Static rules only: no findings is not proof that a model is safe.");
    if let Some(path) = &args.report {
        let json = serde_json::json!({
            "tool": "thme scan",
            "thme_version": env!("CARGO_PKG_VERSION"),
            "verdict": report.verdict(),
            "files_scanned": report.files_scanned,
            "findings": report.findings,
        });
        std::fs::write(path, serde_json::to_string_pretty(&json)? + "\n")?;
    }
    if report.worst().is_some_and(|w| w >= threshold) {
        bail!("scan verdict: {}", report.verdict());
    }
    Ok(())
}
