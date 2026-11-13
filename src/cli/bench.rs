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
    /// Score outputs produced by any tool into a result record (run mode and evidence: imported)
    Score {
        /// Task file; its dataset is checked against its hash first
        #[arg(long)]
        task: PathBuf,
        /// Outputs, JSONL: one line per dataset item (see docs/benchmark-registry.md)
        #[arg(long)]
        outputs: PathBuf,
        /// SHA-256 of the model the outputs came from
        #[arg(long)]
        model_sha256: String,
        /// Recipe or conversion that produced the model, if any
        #[arg(long)]
        transformation: Option<String>,
        /// Sampling temperature the outputs were produced with (0 = greedy)
        #[arg(long, default_value_t = 0.0)]
        temperature: f64,
        /// Sampling seed; required when temperature > 0
        #[arg(long)]
        seed: Option<u64>,
        /// Generation length cap the outputs were produced with
        #[arg(long)]
        max_new_tokens: Option<u32>,
        /// Hardware the outputs were produced on, as stated by you
        #[arg(long, default_value = "not stated")]
        hardware: String,
        /// Confidence level of the interval
        #[arg(long, default_value_t = 0.95)]
        level: f64,
        /// Where to write the result record
        #[arg(long)]
        out: PathBuf,
    },
    /// Draw charts from result records: scores with intervals, paired changes, a score heatmap
    Chart {
        /// Result records; each is labelled with its file name; repeatable
        #[arg(long = "record", required = true)]
        records: Vec<PathBuf>,
        /// Baseline record for the paired-change chart
        #[arg(long)]
        baseline: Option<PathBuf>,
        /// Output folder for the SVG files
        #[arg(long)]
        out: PathBuf,
        /// Theme file
        #[arg(long, default_value = "assets/chart-themes.json")]
        theme_file: PathBuf,
        /// Confidence level for paired changes
        #[arg(long, default_value_t = 0.95)]
        level: f64,
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
        BenchCommand::Score {
            task,
            outputs,
            model_sha256,
            transformation,
            temperature,
            seed,
            max_new_tokens,
            hardware,
            level,
            out,
        } => {
            use crate::bench::score::{
                imported_record, read_dataset, read_outputs, score_outputs, today_utc, ImportedRun,
            };
            let started = std::time::Instant::now();
            let spec = TaskSpec::load(&task)?;
            let dataset = spec.verify_dataset(task.parent().unwrap_or(Path::new(".")))?;
            let items = read_dataset(&dataset)?;
            let scored = score_outputs(&spec, &items, &read_outputs(&outputs)?)?;
            let run = ImportedRun {
                model_sha256,
                transformation,
                decoding: crate::bench::Decoding {
                    temperature,
                    seed,
                    max_new_tokens,
                },
                hardware,
                run_date: today_utc(),
                level,
                seed: 0,
            };
            let record = imported_record(&spec, scored, &run, started.elapsed().as_secs_f64())?;
            std::fs::write(
                &out,
                serde_json::to_string_pretty(&record)?
                    + "
",
            )?;
            println!(
                "{}: {} items, score {:.4} [{:.4}, {:.4}] ({:?}, {:.0}%), evidence imported -> {}",
                spec.key(),
                record.items.len(),
                record.score,
                record.interval.low,
                record.interval.high,
                record.interval.method,
                record.interval.level * 100.0,
                out.display()
            );
            Ok(())
        }
        BenchCommand::Chart {
            records,
            baseline,
            out,
            theme_file,
            level,
        } => {
            use crate::bench::charts::{deltas_chart, score_heatmap, scores_chart};
            let label = |p: &Path| {
                p.file_stem()
                    .map(|s| s.to_string_lossy().into_owned())
                    .unwrap_or_default()
            };
            let loaded: Vec<(String, ResultRecord)> = records
                .iter()
                .map(|p| Ok((label(p), ResultRecord::load(p)?)))
                .collect::<Result<_>>()?;
            let labelled: Vec<(&str, &ResultRecord)> =
                loaded.iter().map(|(l, r)| (l.as_str(), r)).collect();
            std::fs::create_dir_all(&out)?;
            let mut written = Vec::new();
            for name in ["light", "dark"] {
                let theme = crate::engine::viz::Theme::load(&theme_file, name)?;
                let same_bench = labelled.iter().all(|(_, r)| {
                    crate::bench::compare::metric_name(r)
                        == crate::bench::compare::metric_name(labelled[0].1)
                });
                if same_bench {
                    written.push(write_svg(
                        &out,
                        &format!("scores-{name}.svg"),
                        &scores_chart(&labelled, &theme)?,
                    )?);
                }
                written.push(write_svg(
                    &out,
                    &format!("heatmap-{name}.svg"),
                    &score_heatmap(&labelled, &theme)?,
                )?);
                if let Some(b) = &baseline {
                    let base = ResultRecord::load(b)?;
                    let base_label = label(b);
                    let others: Vec<(&str, &ResultRecord)> = labelled
                        .iter()
                        .copied()
                        .filter(|(l, _)| *l != base_label)
                        .collect();
                    written.push(write_svg(
                        &out,
                        &format!("deltas-{name}.svg"),
                        &deltas_chart((base_label.as_str(), &base), &others, level, 0, &theme)?,
                    )?);
                }
            }
            for w in written {
                println!("wrote {w}");
            }
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

fn write_svg(dir: &Path, name: &str, svg: &str) -> Result<String> {
    let path = dir.join(name);
    std::fs::write(&path, svg)?;
    Ok(path.display().to_string())
}
