// SPDX-License-Identifier: AGPL-3.0-only
//! Benchmark charts (roadmap v0.2.8.8, dossier WP37 and R.7), drawn from stored result records
//! through the visual engine. Every chart carries the records' evidence status; scores always
//! show their intervals, and differences come from paired comparisons.

use anyhow::{bail, Result};

use super::compare::{compare, metric_name};
use super::{Evidence, ResultRecord};
use crate::engine::viz::{
    heatmap, interval_plot, Frame, Heatmap, IntervalPlot, IntervalRow, Theme,
};

/// A record and the label it gets on charts (for example the model or recipe name).
pub type Labelled<'a> = (&'a str, &'a ResultRecord);

/// The evidence statuses of `records`, in first-seen order: "Imported", or "Measured; Imported".
pub fn evidence_label<'a>(records: impl IntoIterator<Item = &'a ResultRecord>) -> String {
    let mut seen: Vec<Evidence> = Vec::new();
    for r in records {
        if !seen.contains(&r.evidence) {
            seen.push(r.evidence);
        }
    }
    seen.iter()
        .map(|e| format!("{e:?}"))
        .collect::<Vec<_>>()
        .join("; ")
}

/// Scores of several records on one benchmark, each with its interval.
pub fn scores_chart(records: &[Labelled], theme: &Theme) -> Result<String> {
    let Some((_, first)) = records.first() else {
        bail!("no records to chart");
    };
    let bench = metric_name(first);
    if let Some((label, _)) = records.iter().find(|(_, r)| metric_name(r) != bench) {
        bail!("record '{label}' is not on {bench}; a scores chart shows one benchmark");
    }
    let rows = records
        .iter()
        .map(|(label, r)| IntervalRow {
            label: label.to_string(),
            value: r.score,
            low: r.interval.low,
            high: r.interval.high,
        })
        .collect();
    let level = first.interval.level;
    let plot = IntervalPlot {
        frame: Frame {
            title: format!(
                "{} scores, {:.0}% intervals",
                bench.trim_start_matches("bench:"),
                level * 100.0
            ),
            evidence: evidence_label(records.iter().map(|(_, r)| *r)),
            x_label: "Score".into(),
            y_label: String::new(),
        },
        rows,
        reference: None,
    };
    Ok(interval_plot(&plot, theme)?)
}

/// Each candidate's score minus the baseline's, with a paired-bootstrap interval; the dashed
/// line is zero. Records must be comparable (see [`super::compare::check_comparable`]).
pub fn deltas_chart(
    baseline: Labelled,
    candidates: &[Labelled],
    level: f64,
    seed: u64,
    theme: &Theme,
) -> Result<String> {
    if candidates.is_empty() {
        bail!("no candidate records to compare with the baseline");
    }
    let mut rows = Vec::new();
    for (label, cand) in candidates {
        let m = compare(baseline.1, cand, level, 2000, seed)?;
        let (lo, hi) = m.degradation_interval.unwrap_or((0.0, 0.0));
        // Degradation is baseline - candidate; the chart shows candidate - baseline.
        rows.push(IntervalRow {
            label: label.to_string(),
            value: m.candidate - m.baseline,
            low: -hi,
            high: -lo,
        });
    }
    let all = std::iter::once(baseline.1).chain(candidates.iter().map(|(_, r)| *r));
    let plot = IntervalPlot {
        frame: Frame {
            title: format!(
                "{}: change against {}, paired {:.0}% intervals",
                metric_name(baseline.1).trim_start_matches("bench:"),
                baseline.0,
                level * 100.0
            ),
            evidence: evidence_label(all),
            x_label: "Score difference (candidate - baseline)".into(),
            y_label: String::new(),
        },
        rows,
        reference: Some(0.0),
    };
    Ok(interval_plot(&plot, theme)?)
}

/// Scores of models (rows) on benchmarks (columns); a missing pair is an empty cell.
pub fn score_heatmap(records: &[Labelled], theme: &Theme) -> Result<String> {
    if records.is_empty() {
        bail!("no records to chart");
    }
    let mut models: Vec<String> = Vec::new();
    let mut benches: Vec<String> = Vec::new();
    for (label, r) in records {
        if !models.iter().any(|m| m == label) {
            models.push(label.to_string());
        }
        let b = metric_name(r).trim_start_matches("bench:").to_string();
        if !benches.contains(&b) {
            benches.push(b);
        }
    }
    let mut values = vec![vec![None; benches.len()]; models.len()];
    for (label, r) in records {
        let (Some(i), Some(j)) = (
            models.iter().position(|m| m == label),
            benches
                .iter()
                .position(|b| *b == metric_name(r).trim_start_matches("bench:")),
        ) else {
            continue;
        };
        if values[i][j].is_some() {
            bail!("two records for '{label}' on {}", benches[j]);
        }
        values[i][j] = Some(r.score);
    }
    let map = Heatmap {
        frame: Frame {
            title: "Scores by model and benchmark".into(),
            evidence: evidence_label(records.iter().map(|(_, r)| *r)),
            x_label: "Benchmark".into(),
            y_label: "Model".into(),
        },
        rows: models,
        cols: benches,
        values,
    };
    Ok(heatmap(&map, theme)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bench::*;

    fn theme() -> Theme {
        Theme::load(
            &std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/chart-themes.json"),
            "light",
        )
        .unwrap()
    }

    fn record(bench: &str, scores: &[f64], evidence: Evidence) -> ResultRecord {
        let score = scores.iter().sum::<f64>() / scores.len() as f64;
        let interval =
            crate::bench::stats::interval_for(scores, IntervalMethod::Wilson, 0.95, 0).unwrap();
        ResultRecord {
            schema: RECORD_SCHEMA.into(),
            benchmark: BenchmarkRef {
                id: bench.into(),
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
                run_mode: if evidence == Evidence::Imported {
                    RunMode::Imported
                } else {
                    RunMode::Native
                },
                prompt_template_sha256: "d".repeat(64),
            },
            decoding: Decoding {
                temperature: 0.0,
                seed: None,
                max_new_tokens: None,
            },
            items: scores
                .iter()
                .enumerate()
                .map(|(i, &s)| ItemResult {
                    id: format!("q{i}"),
                    output: None,
                    score: s,
                })
                .collect(),
            score,
            interval,
            evidence,
            hardware: "synthetic".into(),
            run_date: "2026-11-13".into(),
            duration_s: 0.0,
        }
    }

    fn scores(n: usize, wrong_every: usize) -> Vec<f64> {
        (0..n)
            .map(|i| if i % wrong_every == 0 { 0.0 } else { 1.0 })
            .collect()
    }

    #[test]
    fn scores_chart_has_one_row_per_record_and_one_benchmark() {
        let a = record("toy", &scores(100, 3), Evidence::Simulated);
        let b = record("toy", &scores(100, 4), Evidence::Simulated);
        let svg = scores_chart(&[("base", &a), ("q8", &b)], &theme()).unwrap();
        assert_eq!(svg.matches("class=\"estimate\"").count(), 2);
        assert!(
            svg.contains("Evidence: Simulated") && svg.contains("toy@v1 scores, 95% intervals")
        );
        let c = record("other", &scores(100, 3), Evidence::Simulated);
        assert!(scores_chart(&[("base", &a), ("x", &c)], &theme()).is_err());
    }

    #[test]
    fn deltas_are_paired_and_signed_candidate_minus_baseline() {
        let base = record("toy", &scores(300, 3), Evidence::Simulated);
        let mut worse = scores(300, 3);
        for s in worse.iter_mut().skip(1).step_by(3).take(30) {
            *s = 0.0;
        }
        let cand = record("toy", &worse, Evidence::Simulated);
        let svg = deltas_chart(("base", &base), &[("tq1", &cand)], 0.95, 1, &theme()).unwrap();
        assert!(svg.contains("class=\"reference\""));
        assert!(svg.contains(">-0.1 ["), "30 of 300 items lost: -0.1\n{svg}");
        let imported = record("toy", &scores(300, 3), Evidence::Imported);
        assert!(
            deltas_chart(("base", &base), &[("x", &imported)], 0.95, 1, &theme()).is_err(),
            "evidence must match"
        );
    }

    #[test]
    fn heatmap_and_evidence_labels() {
        let r1 = record("toy", &scores(10, 2), Evidence::Measured);
        let r2 = record("other", &scores(10, 5), Evidence::Imported);
        let r3 = record("toy", &scores(10, 3), Evidence::Measured);
        let svg = score_heatmap(&[("m1", &r1), ("m1", &r2), ("m2", &r3)], &theme()).unwrap();
        assert_eq!(svg.matches("class=\"cell\"").count(), 3);
        assert_eq!(svg.matches("class=\"missing\"").count(), 1);
        assert!(svg.contains("Evidence: Measured; Imported"));
        assert!(
            score_heatmap(&[("m1", &r1), ("m1", &r3)], &theme()).is_err(),
            "two records for one cell"
        );
    }
}
