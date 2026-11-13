// SPDX-License-Identifier: AGPL-3.0-only
//! `thme bench`: the benchmark registry. Lists and checks task files and result records.
//! Running tasks is the benchmark runner (v0.2.8.3).

use anyhow::{bail, Context, Result};
use clap::{Args, Subcommand};
use std::path::{Path, PathBuf};

use crate::bench::{Registry, ResultRecord, TaskSpec, RECORD_SCHEMA, TASK_SCHEMA};

#[derive(Args, Debug)]
pub struct BenchArgs {
    #[command(subcommand)]
    pub command: BenchCommand,
}

#[derive(Subcommand, Debug)]
pub enum BenchCommand {
    /// List the task files in a folder; any malformed file or duplicate id@version fails
    List {
        #[arg(value_name = "DIR", default_value = "benchmarks/tasks")]
        dir: PathBuf,
    },
    /// Check task files and result records (detected by their `schema` field)
    Validate {
        #[arg(value_name = "FILE", required = true)]
        files: Vec<PathBuf>,
        /// Also check that every record was produced by this task
        #[arg(long)]
        task: Option<PathBuf>,
    },
    /// Check a task's dataset file against the hash in the task
    VerifyDataset {
        #[arg(value_name = "TASK")]
        task: PathBuf,
    },
}

pub fn execute(args: BenchArgs) -> Result<()> {
    match args.command {
        BenchCommand::List { dir } => {
            let registry = Registry::load(&dir)?;
            println!("{} tasks in {}", registry.len(), dir.display());
            for (_, t) in registry.iter() {
                println!(
                    "  {:<28} {:<18} {:<22} {} ({})",
                    t.key(),
                    format!("{:?}", t.family),
                    metric_name(&t.metric),
                    t.dataset.path,
                    t.dataset.split
                );
            }
            Ok(())
        }
        BenchCommand::Validate { files, task } => {
            let task = task.as_deref().map(TaskSpec::load).transpose()?;
            for file in &files {
                validate_one(file, task.as_ref())?;
            }
            Ok(())
        }
        BenchCommand::VerifyDataset { task } => {
            let spec = TaskSpec::load(&task)?;
            let dir = task.parent().unwrap_or(Path::new("."));
            let path = spec.verify_dataset(dir)?;
            println!(
                "{}: dataset {} matches its hash",
                spec.key(),
                path.display()
            );
            Ok(())
        }
    }
}

fn validate_one(file: &Path, task: Option<&TaskSpec>) -> Result<()> {
    let text = std::fs::read_to_string(file).with_context(|| format!("Failed to read {file:?}"))?;
    let value: serde_json::Value =
        serde_json::from_str(&text).with_context(|| format!("{file:?} is not JSON"))?;
    match value.get("schema").and_then(|s| s.as_str()) {
        Some(TASK_SCHEMA) => {
            let t = TaskSpec::load(file)?;
            println!("{}: task {} ok", file.display(), t.key());
        }
        Some(RECORD_SCHEMA) => {
            let r = ResultRecord::load(file)?;
            if let Some(t) = task {
                r.check_against(t)
                    .with_context(|| format!("{file:?} does not match task {}", t.key()))?;
            }
            println!(
                "{}: record {}@v{} ok, {} items, score {:.4} [{:.4}, {:.4}], {:?}",
                file.display(),
                r.benchmark.id,
                r.benchmark.version,
                r.items.len(),
                r.score,
                r.interval.low,
                r.interval.high,
                r.evidence
            );
        }
        other => {
            bail!("{file:?}: schema {other:?} is neither '{TASK_SCHEMA}' nor '{RECORD_SCHEMA}'")
        }
    }
    Ok(())
}

fn metric_name(metric: &crate::bench::Metric) -> String {
    serde_json::to_value(metric)
        .ok()
        .and_then(|v| v.get("kind").and_then(|k| k.as_str()).map(String::from))
        .unwrap_or_default()
}
