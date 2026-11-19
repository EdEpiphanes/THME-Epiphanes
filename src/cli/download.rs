// SPDX-License-Identifier: AGPL-3.0-only
//! `thme download`: fetch a model from the Hugging Face hub, then run the static scan on it.
//! The model is never loaded. `HF_TOKEN` (optional) is sent as a bearer token and not printed.

use anyhow::Result;
use clap::Args;
use std::io::Write;
use std::path::PathBuf;

use crate::hub::{self, HttpFetch, RepoId, Request};
use crate::scan::scan;

#[derive(Args, Debug)]
pub struct DownloadArgs {
    /// Hugging Face model id, ORG/NAME
    #[arg(value_name = "ORG/NAME")]
    pub repo: String,

    /// Branch, tag or commit to fetch
    #[arg(long, default_value = "main")]
    pub revision: String,

    /// Output folder (default: models/NAME)
    #[arg(long, value_name = "DIR")]
    pub out: Option<PathBuf>,

    /// File to fetch instead of the default set (repeat for several). Pickle files need this.
    #[arg(long = "file", value_name = "NAME")]
    pub files: Vec<String>,

    /// Print the plan and the licence, then stop without downloading
    #[arg(long)]
    pub dry_run: bool,
}

pub fn execute(args: DownloadArgs) -> Result<()> {
    let repo = RepoId::parse(&args.repo)?;
    let out = args
        .out
        .unwrap_or_else(|| PathBuf::from("models").join(repo.name()));
    let token = std::env::var("HF_TOKEN")
        .ok()
        .filter(|t| !t.trim().is_empty());
    let req = Request {
        repo: &args.repo,
        revision: &args.revision,
        files: &args.files,
        out: &out,
        dry_run: args.dry_run,
    };
    let mut stdout = std::io::stdout().lock();
    hub::run(&HttpFetch::new(), &req, token.as_deref(), &mut stdout)?;
    stdout.flush()?;
    if args.dry_run {
        return Ok(());
    }
    let report = scan(&out)?;
    println!(
        "Scan of {}: {} ({} files, {} findings)",
        out.display(),
        report.verdict(),
        report.files_scanned,
        report.findings.len()
    );
    for f in &report.findings {
        println!("  [{:?}] {} {}: {}", f.severity, f.rule, f.file, f.detail);
    }
    println!("Static rules only: no findings is not proof that a model is safe.");
    Ok(())
}
