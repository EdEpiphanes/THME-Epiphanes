// SPDX-License-Identifier: AGPL-3.0-only
//! Benchmark runner, imported mode (roadmap v0.2.8.3, dossier R.4): model outputs produced by
//! any tool are scored with THME's own metric library and written as a result record with every
//! field of Table R2. The record's run mode and evidence are `imported`: THME scored the
//! outputs, it did not produce them.
//!
//! Outputs file (JSONL, one line per item, every dataset item exactly once):
//! - generated answers: `{"id": "...", "output": "..."}`
//! - likelihood choice: `{"id": "...", "log_likelihoods": [..]}` (dataset `target` = correct index)
//! - code (pass@k): `{"id": "...", "samples": n, "passed": c}` from tests run elsewhere; THME
//!   does not run code.

use anyhow::{bail, Context, Result};
use serde::Deserialize;
use std::collections::BTreeMap;
use std::path::Path;

use super::metrics::{choice_by_likelihood, pass_at_k};
use super::stats::interval_for;
use super::{
    BenchmarkRef, Decoding, Evidence, Harness, IntervalMethod, ItemResult, Metric, ModelRef,
    ResultRecord, RunMode, TaskSpec, RECORD_SCHEMA,
};

/// One dataset item (JSONL): `id`, `input`, optional `target`, optional `choices`.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct DatasetItem {
    pub id: String,
    pub input: String,
    #[serde(default)]
    pub target: Option<String>,
    #[serde(default)]
    pub choices: Option<Vec<String>>,
}

/// Reads a JSONL dataset strictly: every line an item, ids unique, at least one item.
pub fn read_dataset(path: &Path) -> Result<Vec<DatasetItem>> {
    let text =
        std::fs::read_to_string(path).with_context(|| format!("reading dataset {path:?}"))?;
    let mut items = Vec::new();
    let mut seen = std::collections::BTreeSet::new();
    for (n, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let item: DatasetItem = serde_json::from_str(line).with_context(|| {
            format!(
                "{path:?} line {}: needs string fields \"id\" and \"input\"",
                n + 1
            )
        })?;
        if !seen.insert(item.id.clone()) {
            bail!(
                "{path:?} line {}: item id '{}' appears twice",
                n + 1,
                item.id
            );
        }
        items.push(item);
    }
    if items.is_empty() {
        bail!("{path:?} holds no items");
    }
    Ok(items)
}

/// One line of an outputs file.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Output {
    pub id: String,
    #[serde(default)]
    pub output: Option<String>,
    #[serde(default)]
    pub log_likelihoods: Option<Vec<f64>>,
    #[serde(default)]
    pub samples: Option<u32>,
    #[serde(default)]
    pub passed: Option<u32>,
}

/// Reads an outputs file (JSONL).
pub fn read_outputs(path: &Path) -> Result<Vec<Output>> {
    let text =
        std::fs::read_to_string(path).with_context(|| format!("reading outputs {path:?}"))?;
    text.lines()
        .enumerate()
        .filter(|(_, l)| !l.trim().is_empty())
        .map(|(n, l)| {
            serde_json::from_str(l)
                .with_context(|| format!("{path:?} line {}: not a valid output", n + 1))
        })
        .collect()
}

/// Scores every item. Each dataset item needs exactly one output, and no output may name an
/// unknown item, so a partial or mismatched run is refused instead of scored on a subset.
pub fn score_outputs(
    task: &TaskSpec,
    items: &[DatasetItem],
    outputs: &[Output],
) -> Result<Vec<ItemResult>> {
    let mut by_id: BTreeMap<&str, &Output> = BTreeMap::new();
    for o in outputs {
        if by_id.insert(o.id.as_str(), o).is_some() {
            bail!("output for item '{}' appears twice", o.id);
        }
    }
    let known: std::collections::BTreeSet<&str> = items.iter().map(|i| i.id.as_str()).collect();
    if let Some(extra) = by_id.keys().find(|id| !known.contains(*id)) {
        bail!("output names item '{extra}', which is not in the dataset");
    }
    let mut results = Vec::with_capacity(items.len());
    for item in items {
        let Some(o) = by_id.get(item.id.as_str()) else {
            bail!(
                "no output for item '{}' ({} of {} items have outputs)",
                item.id,
                by_id.len(),
                items.len()
            );
        };
        let target = item.target.as_deref();
        let score = match &task.metric {
            Metric::ChoiceByLikelihood => {
                let lls = o.log_likelihoods.as_ref().with_context(|| {
                    format!(
                        "item '{}': likelihood tasks need \"log_likelihoods\"",
                        item.id
                    )
                })?;
                let correct: usize =
                    target
                        .and_then(|t| t.trim().parse().ok())
                        .with_context(|| {
                            format!(
                                "item '{}': target must be the index of the correct choice",
                                item.id
                            )
                        })?;
                if let Some(choices) = &item.choices {
                    if choices.len() != lls.len() {
                        bail!(
                            "item '{}': {} log-likelihoods for {} choices",
                            item.id,
                            lls.len(),
                            choices.len()
                        );
                    }
                }
                choice_by_likelihood(lls, correct).with_context(|| format!("item '{}'", item.id))?
            }
            Metric::PassAtK { k, samples } => {
                let (Some(n), Some(c)) = (o.samples, o.passed) else {
                    bail!(
                        "item '{}': pass@k tasks need \"samples\" and \"passed\"",
                        item.id
                    );
                };
                if n != *samples {
                    bail!(
                        "item '{}': {n} samples, the task asks for {samples}",
                        item.id
                    );
                }
                pass_at_k(n, c, *k).with_context(|| format!("item '{}'", item.id))?
            }
            _ => {
                let text = o.output.as_deref().with_context(|| {
                    format!("item '{}': generation tasks need \"output\"", item.id)
                })?;
                let target = target.with_context(|| format!("item '{}' has no target", item.id))?;
                task.score_output(text, target)?
            }
        };
        results.push(ItemResult {
            id: item.id.clone(),
            output: o.output.clone(),
            score,
        });
    }
    Ok(results)
}

/// What the caller states about the run; THME cannot check it.
#[derive(Debug, Clone)]
pub struct ImportedRun {
    pub model_sha256: String,
    pub transformation: Option<String>,
    pub decoding: Decoding,
    pub hardware: String,
    pub run_date: String,
    pub level: f64,
    pub seed: u64,
}

/// Builds and validates the result record for scored items.
pub fn imported_record(
    task: &TaskSpec,
    items: Vec<ItemResult>,
    run: &ImportedRun,
    duration_s: f64,
) -> Result<ResultRecord> {
    let scores: Vec<f64> = items.iter().map(|i| i.score).collect();
    let binary = scores.iter().all(|&s| s == 0.0 || s == 1.0);
    let method = if binary {
        IntervalMethod::Wilson
    } else {
        IntervalMethod::Bootstrap
    };
    let interval = interval_for(&scores, method, run.level, run.seed)?;
    let record = ResultRecord {
        schema: RECORD_SCHEMA.into(),
        benchmark: BenchmarkRef {
            id: task.id.clone(),
            version: task.version,
            task_sha256: task.hash(),
            dataset_sha256: task.dataset.sha256.clone(),
            split: task.dataset.split.clone(),
            licence: task.dataset.licence.clone(),
        },
        model: ModelRef {
            sha256: run.model_sha256.clone(),
            transformation: run.transformation.clone(),
        },
        harness: Harness {
            version: env!("CARGO_PKG_VERSION").into(),
            run_mode: RunMode::Imported,
            prompt_template_sha256: task.prompt_template_hash(),
        },
        decoding: run.decoding.clone(),
        score: scores.iter().sum::<f64>() / scores.len() as f64,
        items,
        interval,
        evidence: Evidence::Imported,
        hardware: run.hardware.clone(),
        run_date: run.run_date.clone(),
        duration_s,
    };
    record.validate()?;
    record.check_against(task)?;
    Ok(record)
}

/// Today's date (UTC) as `YYYY-MM-DD`, from the system clock.
pub fn today_utc() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    civil_date((secs / 86_400) as i64)
}

/// Days since 1970-01-01 to a calendar date (Howard Hinnant's algorithm).
fn civil_date(days: i64) -> String {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    format!("{y:04}-{m:02}-{d:02}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bench::TASK_SCHEMA;

    fn task(family: &str, extract: serde_json::Value, metric: serde_json::Value) -> TaskSpec {
        let t: TaskSpec = serde_json::from_value(serde_json::json!({
            "schema": TASK_SCHEMA, "id": "toy", "version": 1, "family": family,
            "dataset": {"path": "d.jsonl", "sha256": "a".repeat(64), "split": "test", "licence": "CC0-1.0"},
            "prompt_template": "{input}", "extract": extract, "metric": metric
        }))
        .unwrap();
        t.validate().unwrap();
        t
    }

    fn item(id: &str, target: &str) -> DatasetItem {
        DatasetItem {
            id: id.into(),
            input: "q".into(),
            target: Some(target.into()),
            choices: None,
        }
    }

    fn out(id: &str) -> Output {
        Output {
            id: id.into(),
            output: None,
            log_likelihoods: None,
            samples: None,
            passed: None,
        }
    }

    fn run() -> ImportedRun {
        ImportedRun {
            model_sha256: "b".repeat(64),
            transformation: Some("Q8_0".into()),
            decoding: Decoding {
                temperature: 0.0,
                seed: None,
                max_new_tokens: Some(64),
            },
            hardware: "supplied by the user".into(),
            run_date: "2026-11-13".into(),
            level: 0.95,
            seed: 0,
        }
    }

    #[test]
    fn generation_outputs_become_a_valid_imported_record() {
        let t = task(
            "checkable-answer",
            serde_json::json!({"kind": "after-marker", "marker": "####"}),
            serde_json::json!({"kind": "numeric-tolerance", "abs": 0.0}),
        );
        let items: Vec<DatasetItem> = (0..10)
            .map(|i| item(&format!("q{i}"), &i.to_string()))
            .collect();
        let outputs: Vec<Output> = (0..10)
            .map(|i| Output {
                output: Some(format!("work #### {}", if i < 7 { i } else { 99 })),
                ..out(&format!("q{i}"))
            })
            .collect();
        let scored = score_outputs(&t, &items, &outputs).unwrap();
        let r = imported_record(&t, scored, &run(), 0.1).unwrap();
        assert_eq!(r.score, 0.7);
        assert_eq!(r.evidence, Evidence::Imported);
        assert_eq!(r.harness.run_mode, RunMode::Imported);
        assert_eq!(r.interval.method, IntervalMethod::Wilson);
        assert!(
            (r.interval.low - 0.397).abs() < 5e-4,
            "7 of 10: Table R3 Wilson interval"
        );
    }

    #[test]
    fn likelihood_and_pass_at_k_outputs() {
        let t = task(
            "multiple-choice",
            serde_json::json!({"kind": "none"}),
            serde_json::json!({"kind": "choice-by-likelihood"}),
        );
        let items = vec![
            DatasetItem {
                choices: Some(vec!["a".into(), "b".into(), "c".into()]),
                ..item("q0", "2")
            },
            item("q1", "0"),
        ];
        let outputs = vec![
            Output {
                log_likelihoods: Some(vec![-3.0, -2.0, -1.0]),
                ..out("q0")
            },
            Output {
                log_likelihoods: Some(vec![-1.0, -1.0]),
                ..out("q1")
            },
        ];
        let s = score_outputs(&t, &items, &outputs).unwrap();
        assert_eq!(s.iter().map(|i| i.score).collect::<Vec<_>>(), [1.0, 0.0]);
        let bad = vec![
            Output {
                log_likelihoods: Some(vec![-1.0, -2.0]),
                ..out("q0")
            },
            outputs[1].clone(),
        ];
        assert!(score_outputs(&t, &items, &bad)
            .unwrap_err()
            .to_string()
            .contains("choices"));

        let t = task(
            "code",
            serde_json::json!({"kind": "none"}),
            serde_json::json!({"kind": "pass-at-k", "k": 2, "samples": 6}),
        );
        let items = vec![item("p0", ""), item("p1", "")];
        let outputs = vec![
            Output {
                samples: Some(6),
                passed: Some(2),
                ..out("p0")
            },
            Output {
                samples: Some(6),
                passed: Some(0),
                ..out("p1")
            },
        ];
        let s = score_outputs(&t, &items, &outputs).unwrap();
        assert!((s[0].score - 0.6).abs() < 1e-12 && s[1].score == 0.0);
        let r = imported_record(&t, s, &run(), 0.0).unwrap();
        assert_eq!(
            r.interval.method,
            IntervalMethod::Bootstrap,
            "fractional scores use the bootstrap"
        );
        let wrong_n = vec![
            Output {
                samples: Some(5),
                passed: Some(2),
                ..out("p0")
            },
            outputs[1].clone(),
        ];
        assert!(score_outputs(&t, &items, &wrong_n).is_err());
    }

    #[test]
    fn partial_duplicate_and_unknown_outputs_are_refused() {
        let t = task(
            "checkable-answer",
            serde_json::json!({"kind": "exact"}),
            serde_json::json!({"kind": "exact-match"}),
        );
        let items = vec![item("a", "1"), item("b", "2")];
        let one = |id: &str| Output {
            output: Some("1".into()),
            ..out(id)
        };
        assert!(score_outputs(&t, &items, &[one("a")])
            .unwrap_err()
            .to_string()
            .contains("no output for item 'b'"));
        assert!(score_outputs(&t, &items, &[one("a"), one("a"), one("b")])
            .unwrap_err()
            .to_string()
            .contains("twice"));
        assert!(score_outputs(&t, &items, &[one("a"), one("b"), one("z")])
            .unwrap_err()
            .to_string()
            .contains("not in the dataset"));
        assert!(score_outputs(&t, &items, &[out("a"), one("b")])
            .unwrap_err()
            .to_string()
            .contains("\"output\""));
    }

    #[test]
    fn civil_dates() {
        assert_eq!(civil_date(0), "1970-01-01");
        assert_eq!(civil_date(11_016), "2000-02-29");
        assert_eq!(civil_date(20_405), "2025-11-13");
        assert_eq!(today_utc().len(), 10);
    }
}
