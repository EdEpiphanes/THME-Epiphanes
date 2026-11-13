// SPDX-License-Identifier: AGPL-3.0-only
//! Comparing two result records (dossier R.6) for the three-state gate (roadmap v0.2.8.6).
//!
//! Two records are comparable only when they ran the same task (task, dataset and prompt
//! template hashes, split), with the same harness version and decoding settings, on the same
//! items, with the same evidence status. The degradation and its interval come from a paired
//! bootstrap over the shared items, so each item's difficulty cancels.

use anyhow::{bail, Result};
use std::collections::BTreeMap;

use super::stats::paired_bootstrap_difference;
use super::ResultRecord;
use crate::quality::gate::Measure;

/// Gate metric name for a benchmark score: `bench:<id>@v<version>`.
pub fn metric_name(record: &ResultRecord) -> String {
    format!(
        "bench:{}@v{}",
        record.benchmark.id, record.benchmark.version
    )
}

/// Refuses two records that cannot be compared, naming the first difference.
pub fn check_comparable(baseline: &ResultRecord, candidate: &ResultRecord) -> Result<()> {
    baseline.validate()?;
    candidate.validate()?;
    let (b, c) = (baseline, candidate);
    let pairs = [
        ("benchmark", metric_name(b), metric_name(c)),
        (
            "task hash",
            b.benchmark.task_sha256.clone(),
            c.benchmark.task_sha256.clone(),
        ),
        (
            "dataset hash",
            b.benchmark.dataset_sha256.clone(),
            c.benchmark.dataset_sha256.clone(),
        ),
        (
            "split",
            b.benchmark.split.clone(),
            c.benchmark.split.clone(),
        ),
        (
            "prompt template hash",
            b.harness.prompt_template_sha256.clone(),
            c.harness.prompt_template_sha256.clone(),
        ),
        (
            "harness version",
            b.harness.version.clone(),
            c.harness.version.clone(),
        ),
        (
            "decoding settings",
            serde_json::to_string(&b.decoding).unwrap_or_default(),
            serde_json::to_string(&c.decoding).unwrap_or_default(),
        ),
        (
            "evidence status",
            format!("{:?}", b.evidence),
            format!("{:?}", c.evidence),
        ),
    ];
    for (what, x, y) in pairs {
        if x != y {
            bail!("records are not comparable: {what} differs ('{x}' vs '{y}')");
        }
    }
    let ids_b: Vec<&str> = b.items.iter().map(|i| i.id.as_str()).collect();
    let mut ids_c: Vec<&str> = c.items.iter().map(|i| i.id.as_str()).collect();
    let mut sorted_b = ids_b.clone();
    sorted_b.sort_unstable();
    ids_c.sort_unstable();
    if sorted_b != ids_c {
        bail!("records are not comparable: they were scored on different items");
    }
    Ok(())
}

/// The candidate's degradation against the baseline (baseline score minus candidate score;
/// positive is worse) with a paired-bootstrap interval at `level`.
pub fn compare(
    baseline: &ResultRecord,
    candidate: &ResultRecord,
    level: f64,
    resamples: usize,
    seed: u64,
) -> Result<Measure> {
    check_comparable(baseline, candidate)?;
    let by_id: BTreeMap<&str, f64> = candidate
        .items
        .iter()
        .map(|i| (i.id.as_str(), i.score))
        .collect();
    let mut base = Vec::with_capacity(baseline.items.len());
    let mut cand = Vec::with_capacity(baseline.items.len());
    for item in &baseline.items {
        base.push(item.score);
        cand.push(by_id.get(item.id.as_str()).copied().unwrap_or(f64::NAN));
    }
    let (low, high) = paired_bootstrap_difference(&base, &cand, level, resamples, seed)?;
    Ok(Measure::with_interval(
        &metric_name(baseline),
        baseline.score,
        candidate.score,
        low,
        high,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bench::*;
    use crate::quality::gate::{Criterion, Decision, GateSpec};

    fn record(scores: &[f64]) -> ResultRecord {
        let items: Vec<ItemResult> = scores
            .iter()
            .enumerate()
            .map(|(i, &s)| ItemResult {
                id: format!("q{i}"),
                output: None,
                score: s,
            })
            .collect();
        let score = scores.iter().sum::<f64>() / scores.len() as f64;
        ResultRecord {
            schema: RECORD_SCHEMA.into(),
            benchmark: BenchmarkRef {
                id: "toy".into(),
                version: 1,
                task_sha256: "a".repeat(64),
                dataset_sha256: "b".repeat(64),
                split: "test".into(),
                licence: "CC0-1.0".into(),
            },
            model: ModelRef {
                sha256: "c".repeat(64),
                transformation: None,
            },
            harness: Harness {
                version: "0.2.0".into(),
                run_mode: RunMode::Native,
                prompt_template_sha256: "d".repeat(64),
            },
            decoding: Decoding {
                temperature: 0.0,
                seed: None,
                max_new_tokens: None,
            },
            items,
            score,
            interval: Interval {
                low: 0.0,
                high: 1.0,
                method: IntervalMethod::Wilson,
                level: 0.95,
            },
            evidence: Evidence::Simulated,
            hardware: "synthetic".into(),
            run_date: "2026-11-13".into(),
            duration_s: 0.0,
        }
    }

    fn gate(max_abs: f64) -> GateSpec {
        GateSpec {
            name: "toy-gate".into(),
            version: 1,
            criteria: vec![Criterion {
                metric: "bench:toy@v1".into(),
                lower_is_better: false,
                max_abs: Some(max_abs),
                max_rel: None,
            }],
        }
    }

    /// 1-in-3 items wrong on the baseline; the candidate breaks `broken` more of them.
    fn pair(n: usize, broken: usize) -> (ResultRecord, ResultRecord) {
        let base: Vec<f64> = (0..n).map(|i| if i % 3 == 0 { 0.0 } else { 1.0 }).collect();
        let mut cand = base.clone();
        for c in cand.iter_mut().filter(|c| **c == 1.0).take(broken) {
            *c = 0.0;
        }
        (record(&base), record(&cand))
    }

    #[test]
    fn identical_runs_pass_and_clear_losses_fail() {
        let (b, c) = pair(3000, 0);
        let m = compare(&b, &c, 0.95, 1000, 1).unwrap();
        assert_eq!(m.degradation_interval, Some((0.0, 0.0)));
        assert_eq!(gate(0.01).judge(&[m]).decision(), Decision::Pass);
        let (b, c) = pair(3000, 300); // 10 points worse
        let m = compare(&b, &c, 0.95, 1000, 1).unwrap();
        assert_eq!(gate(0.01).judge(&[m]).decision(), Decision::Fail);
    }

    /// Dossier R.5: small sets report inconclusive instead of passing by luck.
    #[test]
    fn a_small_set_is_inconclusive_where_a_large_one_decides() {
        let (b, c) = pair(60, 1); // 1.7 points worse on 60 items
        let m = compare(&b, &c, 0.95, 2000, 3).unwrap();
        assert_eq!(
            gate(0.02).judge(std::slice::from_ref(&m)).decision(),
            Decision::Inconclusive,
            "{m:?}"
        );
        let point = crate::quality::gate::Measure::new(&m.metric, m.baseline, m.candidate);
        assert_eq!(
            gate(0.02).judge(&[point]).decision(),
            Decision::Pass,
            "the point gate would pass it"
        );
        let (b, c) = pair(6000, 20); // 0.33 points worse on 6,000 items
        let m = compare(&b, &c, 0.95, 1000, 3).unwrap();
        assert_eq!(gate(0.02).judge(&[m]).decision(), Decision::Pass);
    }

    #[test]
    fn incomparable_records_are_refused() {
        let (b, c) = pair(30, 0);
        let mut x = c.clone();
        x.harness.version = "0.3.0".into();
        assert!(compare(&b, &x, 0.95, 10, 0)
            .unwrap_err()
            .to_string()
            .contains("harness version"));
        let mut x = c.clone();
        x.decoding.temperature = 0.5;
        x.decoding.seed = Some(1);
        assert!(compare(&b, &x, 0.95, 10, 0)
            .unwrap_err()
            .to_string()
            .contains("decoding"));
        let mut x = c.clone();
        x.items[0].id = "other".into();
        assert!(compare(&b, &x, 0.95, 10, 0)
            .unwrap_err()
            .to_string()
            .contains("different items"));
        let mut x = c;
        x.harness.run_mode = RunMode::Imported;
        x.evidence = Evidence::Imported;
        assert!(compare(&b, &x, 0.95, 10, 0)
            .unwrap_err()
            .to_string()
            .contains("evidence"));
    }
}
