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
    /// Check a task's items against training or teacher data; fails if any item appears
    Contamination {
        /// Task file; its dataset is checked against its hash first
        #[arg(long)]
        task: PathBuf,
        /// Corpus files (any text; JSONL is fine); repeatable
        #[arg(long, required = true)]
        corpus: Vec<PathBuf>,
        /// N-gram length in words
        #[arg(long, default_value_t = crate::bench::contamination::DEFAULT_N)]
        n: usize,
        /// Also write the report as JSON
        #[arg(long)]
        report: Option<PathBuf>,
    },
    /// Judge a candidate record against a baseline record: pass, fail or inconclusive
    Gate {
        /// Gate file; criteria name benchmark metrics as `bench:<id>@v<version>`
        #[arg(long)]
        gate: PathBuf,
        #[arg(long)]
        baseline: PathBuf,
        #[arg(long)]
        candidate: PathBuf,
        /// Confidence level of the degradation interval
        #[arg(long, default_value_t = 0.95)]
        level: f64,
        /// Bootstrap resamples
        #[arg(long, default_value_t = 2000)]
        resamples: usize,
        /// Bootstrap seed, recorded in the verdict
        #[arg(long, default_value_t = 0)]
        seed: u64,
        /// Also write the verdict as JSON
        #[arg(long)]
        report: Option<PathBuf>,
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
        BenchCommand::Contamination {
            task,
            corpus,
            n,
            report,
        } => {
            use crate::bench::contamination::{read_items, ContaminationCheck};
            let spec = TaskSpec::load(&task)?;
            let dataset = spec.verify_dataset(task.parent().unwrap_or(Path::new(".")))?;
            let items = read_items(&dataset)?;
            let mut check = ContaminationCheck::new(&items, n)?;
            for file in &corpus {
                check.scan_file(file)?;
            }
            let found = check.report();
            println!(
                "{}: {} items checked against {} corpus words in {} files ({}-word n-grams)",
                spec.key(),
                found.items_checked,
                found.corpus_words,
                found.sources.len(),
                found.ngram_words
            );
            for o in &found.contaminated {
                println!(
                    "  {:<24} {:>5.1}% of its n-grams found ({} of {}) in {}",
                    o.id,
                    o.fraction * 100.0,
                    o.shared_ngrams,
                    o.item_ngrams,
                    o.sources.join(", ")
                );
            }
            if let Some(path) = report {
                let json = serde_json::json!({
                    "tool": "thme bench contamination",
                    "thme_version": env!("CARGO_PKG_VERSION"),
                    "task": spec.key(),
                    "task_sha256": spec.hash(),
                    "report": found,
                });
                std::fs::write(
                    &path,
                    serde_json::to_string_pretty(&json)?
                        + "
",
                )?;
            }
            if found.contaminated.is_empty() {
                println!("No overlap found.");
                Ok(())
            } else {
                bail!(
                    "{} of {} items appear in the corpora; scores on them are not clean",
                    found.contaminated.len(),
                    found.items_checked
                )
            }
        }
        BenchCommand::Gate {
            gate,
            baseline,
            candidate,
            level,
            resamples,
            seed,
            report,
        } => {
            let spec = crate::quality::gate::GateSpec::load(&gate)?;
            let base = ResultRecord::load(&baseline)?;
            let cand = ResultRecord::load(&candidate)?;
            let measure = crate::bench::compare::compare(&base, &cand, level, resamples, seed)?;
            let verdict = spec.judge(&[measure]);
            println!("Gate {} ({:?} evidence)", verdict.gate, base.evidence);
            for o in &verdict.outcomes {
                let interval = o
                    .degradation_interval
                    .map(|(l, h)| format!(" [{l:+.4}, {h:+.4}]"))
                    .unwrap_or_default();
                println!(
                    "  {:<28} {:<13} degradation {:+.4}{interval}",
                    o.metric,
                    o.status,
                    o.degradation.unwrap_or(f64::NAN)
                );
            }
            let decision = verdict.decision();
            println!("Decision: {decision:?}");
            if let Some(path) = report {
                let json = serde_json::json!({
                    "tool": "thme bench gate",
                    "thme_version": env!("CARGO_PKG_VERSION"),
                    "level": level,
                    "resamples": resamples,
                    "seed": seed,
                    "evidence": base.evidence,
                    "verdict": verdict,
                    "decision": decision,
                });
                std::fs::write(
                    &path,
                    serde_json::to_string_pretty(&json)?
                        + "
",
                )?;
            }
            match decision {
                crate::quality::gate::Decision::Pass => Ok(()),
                other => bail!("gate decision: {other:?}"),
            }
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
