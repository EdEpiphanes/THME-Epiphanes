// SPDX-License-Identifier: AGPL-3.0-only
//! Benchmark metrics library (roadmap v0.2.8.4, dossier WP33).
//!
//! Answer extraction and per-item scoring for the rules a task file names. Every score is in
//! [0, 1]. Pure functions: no model, no I/O, so each rule has hand-computed oracle tests.

use super::{Extract, Metric, TaskSpec};
use anyhow::{bail, Result};

/// Takes the answer out of a model's output according to `rule`. `None` when the rule finds
/// nothing (an empty output, no number, no choice letter, no marker) or extracts nothing by
/// design (`Extract::None`).
pub fn extract(rule: &Extract, output: &str) -> Option<String> {
    let found = match rule {
        Extract::Exact => Some(output.trim().to_string()),
        Extract::FirstLine => output
            .lines()
            .map(str::trim)
            .find(|l| !l.is_empty())
            .map(String::from),
        Extract::LastNumber => numbers(output).last().map(|(text, _)| text.clone()),
        Extract::ChoiceLetter => choice_letter(output),
        Extract::AfterMarker { marker } => output
            .rfind(marker.as_str())
            .map(|i| output[i + marker.len()..].trim().to_string()),
        Extract::None => None,
    };
    found.filter(|s| !s.is_empty())
}

/// The chosen letter: first a marked one (`(C)`, `C)`, `C.`, `C:`, `[C]`), else the first bare
/// capital letter that is not the word "I" or "A" followed by a lower-case word.
fn choice_letter(output: &str) -> Option<String> {
    let tokens: Vec<&str> = output.split_whitespace().collect();
    let letter = |t: &str| {
        let core = t.trim_matches(|c: char| "()[]{}.,:;*\"'".contains(c));
        (core.len() == 1 && core.chars().all(|c| c.is_ascii_uppercase())).then(|| core.to_string())
    };
    let marked = tokens.iter().find_map(|t| {
        let l = letter(t)?;
        let framed = (t.starts_with('(') && t.contains(')'))
            || (t.starts_with('[') && t.contains(']'))
            || [")", ".", ":", ")."]
                .iter()
                .any(|end| *t == format!("{l}{end}"));
        framed.then_some(l)
    });
    marked.or_else(|| {
        tokens.iter().enumerate().find_map(|(i, t)| {
            let l = letter(t)?;
            let next_lower = tokens
                .get(i + 1)
                .is_some_and(|n| n.starts_with(|c: char| c.is_lowercase()));
            let word = (l == "I" || l == "A") && next_lower && *t == l;
            (!word).then_some(l)
        })
    })
}

/// Every number in `text`, in order, as (text, value). Accepts a sign, thousands separators
/// (`1,234`) and decimals; a trailing full stop is not part of the number.
pub fn numbers(text: &str) -> Vec<(String, f64)> {
    let chars: Vec<char> = text.chars().collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        let starts = chars[i].is_ascii_digit()
            || (matches!(chars[i], '-' | '+' | '.')
                && chars.get(i + 1).is_some_and(|c| c.is_ascii_digit())
                && (i == 0 || !chars[i - 1].is_ascii_alphanumeric()));
        if !starts {
            i += 1;
            continue;
        }
        let begin = i;
        i += 1;
        while i < chars.len() {
            let c = chars[i];
            let next_digit = chars.get(i + 1).is_some_and(|d| d.is_ascii_digit());
            if c.is_ascii_digit() || ((c == ',' || c == '.') && next_digit) {
                i += 1;
            } else {
                break;
            }
        }
        let raw: String = chars[begin..i].iter().collect();
        let clean: String = raw.chars().filter(|&c| c != ',' && c != '+').collect();
        if let Ok(v) = clean.parse::<f64>() {
            out.push((raw, v));
        }
    }
    out
}

/// Lower case, ASCII punctuation removed, whitespace collapsed to single spaces.
pub fn normalize(text: &str) -> String {
    text.chars()
        .filter(|c| !c.is_ascii_punctuation())
        .flat_map(char::to_lowercase)
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

/// Scores one generated answer against its target. Likelihood and pass@k metrics are scored
/// with [`choice_by_likelihood`] and [`pass_at_k`] instead, and are refused here.
pub fn score(metric: &Metric, extracted: Option<&str>, target: &str) -> Result<f64> {
    let Some(answer) = extracted else {
        return match metric {
            Metric::ChoiceByLikelihood | Metric::PassAtK { .. } => {
                bail!("{metric:?} is not scored from extracted text")
            }
            _ => Ok(0.0),
        };
    };
    let hit = match metric {
        Metric::ExactMatch => answer.trim() == target.trim(),
        Metric::NormalizedMatch => normalize(answer) == normalize(target),
        Metric::NumericTolerance { abs, rel } => {
            let (Some(a), Some(t)) = (single_number(answer), single_number(target)) else {
                return Ok(0.0);
            };
            let diff = (a - t).abs();
            abs.is_some_and(|x| diff <= x) || rel.is_some_and(|r| diff <= r * t.abs())
        }
        Metric::ChoiceByLikelihood | Metric::PassAtK { .. } => {
            bail!("{metric:?} is not scored from extracted text")
        }
    };
    Ok(if hit { 1.0 } else { 0.0 })
}

/// The value of a text that is one number (after trimming), or the last number in it.
fn single_number(text: &str) -> Option<f64> {
    numbers(text).last().map(|(_, v)| *v)
}

/// 1 when the highest log-likelihood belongs to `correct`, else 0. A tie for the highest value
/// scores 0: the model did not prefer the correct choice.
pub fn choice_by_likelihood(log_likelihoods: &[f64], correct: usize) -> Result<f64> {
    if correct >= log_likelihoods.len() {
        bail!(
            "correct choice {correct} out of {} choices",
            log_likelihoods.len()
        );
    }
    if log_likelihoods.iter().any(|x| x.is_nan()) {
        bail!("a log-likelihood is NaN");
    }
    let best = log_likelihoods[correct];
    let beaten_or_tied = log_likelihoods
        .iter()
        .enumerate()
        .any(|(i, &x)| i != correct && x >= best);
    Ok(if beaten_or_tied { 0.0 } else { 1.0 })
}

/// Unbiased pass@k from `n` samples of which `c` passed: 1 - C(n-c, k) / C(n, k), computed as a
/// product so it stays exact for large `n`.
pub fn pass_at_k(n: u32, c: u32, k: u32) -> Result<f64> {
    if k == 0 || k > n || c > n {
        bail!("pass@k needs 1 <= k <= n and c <= n (n = {n}, c = {c}, k = {k})");
    }
    if n - c < k {
        return Ok(1.0);
    }
    // C(n-c, k) / C(n, k) = prod_{i = n-c+1}^{n} (1 - k / i)
    let fail: f64 = ((n - c + 1)..=n)
        .map(|i| 1.0 - f64::from(k) / f64::from(i))
        .product();
    Ok(1.0 - fail)
}

impl TaskSpec {
    /// Extracts and scores one generated output with this task's rules.
    pub fn score_output(&self, output: &str, target: &str) -> Result<f64> {
        score(
            &self.metric,
            extract(&self.extract, output).as_deref(),
            target,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn marker(m: &str) -> Extract {
        Extract::AfterMarker { marker: m.into() }
    }

    #[test]
    fn extraction_rules() {
        assert_eq!(extract(&Extract::Exact, "  42 \n"), Some("42".into()));
        assert_eq!(extract(&Extract::Exact, "   "), None);
        assert_eq!(
            extract(&Extract::FirstLine, "\n\n  B) Paris \nmore"),
            Some("B) Paris".into())
        );
        assert_eq!(
            extract(&Extract::LastNumber, "3 apples and 1,250.5 pears."),
            Some("1,250.5".into())
        );
        assert_eq!(extract(&Extract::LastNumber, "no digits"), None);
        assert_eq!(
            extract(&Extract::ChoiceLetter, "I think (C) is right"),
            Some("C".into())
        );
        assert_eq!(
            extract(&Extract::ChoiceLetter, "Answer: B."),
            Some("B".into())
        );
        assert_eq!(
            extract(&Extract::ChoiceLetter, "I am unsure"),
            None,
            "'I' starting a sentence is a word, not a choice"
        );
        assert_eq!(
            extract(&Extract::ChoiceLetter, "A good one is D"),
            Some("D".into())
        );
        assert_eq!(extract(&Extract::ChoiceLetter, "B"), Some("B".into()));
        assert_eq!(
            extract(&marker("####"), "work #### 7 #### 18"),
            Some("18".into())
        );
        assert_eq!(extract(&marker("####"), "no marker"), None);
        assert_eq!(extract(&Extract::None, "anything"), None);
    }

    #[test]
    fn numbers_are_parsed_with_signs_separators_and_decimals() {
        let vals: Vec<f64> = numbers("-3, +4, 1,000,000 and .5 then v2 and 7.")
            .into_iter()
            .map(|(_, v)| v)
            .collect();
        assert_eq!(vals, [-3.0, 4.0, 1_000_000.0, 0.5, 2.0, 7.0]);
        // A hyphen joined to a word is not a minus sign.
        assert_eq!(numbers("COVID-19")[0].1, 19.0);
    }

    #[test]
    fn match_metrics() {
        assert_eq!(
            score(&Metric::ExactMatch, Some(" Paris "), "Paris").unwrap(),
            1.0
        );
        assert_eq!(
            score(&Metric::ExactMatch, Some("paris"), "Paris").unwrap(),
            0.0
        );
        assert_eq!(
            score(&Metric::NormalizedMatch, Some("PARIS!"), "  paris ").unwrap(),
            1.0
        );
        assert_eq!(
            score(&Metric::NormalizedMatch, Some("New   York."), "new york").unwrap(),
            1.0
        );
        assert_eq!(score(&Metric::NormalizedMatch, None, "x").unwrap(), 0.0);
    }

    #[test]
    fn numeric_tolerance() {
        let exact = Metric::NumericTolerance {
            abs: Some(0.0),
            rel: None,
        };
        assert_eq!(score(&exact, Some("1,234"), "1234").unwrap(), 1.0);
        assert_eq!(score(&exact, Some("1235"), "1234").unwrap(), 0.0);
        let loose = Metric::NumericTolerance {
            abs: Some(0.5),
            rel: Some(0.01),
        };
        assert_eq!(score(&loose, Some("100.4"), "100").unwrap(), 1.0); // abs
        assert_eq!(score(&loose, Some("1009"), "1000").unwrap(), 1.0); // rel: 9 <= 10
        assert_eq!(score(&loose, Some("1011"), "1000").unwrap(), 0.0);
        assert_eq!(score(&loose, Some("no number"), "1000").unwrap(), 0.0);
    }

    #[test]
    fn likelihood_choice_and_ties() {
        assert_eq!(choice_by_likelihood(&[-3.0, -1.0, -2.0], 1).unwrap(), 1.0);
        assert_eq!(choice_by_likelihood(&[-3.0, -1.0, -2.0], 0).unwrap(), 0.0);
        assert_eq!(
            choice_by_likelihood(&[-1.0, -1.0], 0).unwrap(),
            0.0,
            "a tie is not a preference"
        );
        assert!(choice_by_likelihood(&[-1.0], 3).is_err());
        assert!(choice_by_likelihood(&[f64::NAN, -1.0], 1).is_err());
    }

    /// Dossier Table R3: n = 6, c = 2 gives 0.333, 0.600, 0.800 for k = 1, 2, 3.
    #[test]
    fn pass_at_k_reproduces_table_r3() {
        let got: Vec<f64> = (1..=3).map(|k| pass_at_k(6, 2, k).unwrap()).collect();
        for (g, want) in got.iter().zip([1.0 / 3.0, 0.6, 0.8]) {
            assert!((g - want).abs() < 1e-12, "{g} vs {want}");
        }
    }

    /// pass@k equals the share of k-subsets of the n samples that contain a passing one.
    #[test]
    fn pass_at_k_matches_enumeration() {
        fn subsets(n: u32, k: u32, c: u32) -> f64 {
            let (mut hit, mut all) = (0u64, 0u64);
            for mask in 0u32..(1 << n) {
                if mask.count_ones() == k {
                    all += 1;
                    if mask & ((1 << c) - 1) != 0 {
                        hit += 1;
                    }
                }
            }
            hit as f64 / all as f64
        }
        for n in 1..=10 {
            for c in 0..=n {
                for k in 1..=n {
                    let a = pass_at_k(n, c, k).unwrap();
                    let b = subsets(n, k, c);
                    assert!((a - b).abs() < 1e-12, "n={n} c={c} k={k}: {a} vs {b}");
                }
            }
        }
        assert!(
            pass_at_k(5, 1, 6).is_err()
                && pass_at_k(5, 6, 1).is_err()
                && pass_at_k(5, 1, 0).is_err()
        );
    }

    #[test]
    fn task_scores_an_output_end_to_end() {
        let task: TaskSpec = serde_json::from_value(serde_json::json!({
            "schema": super::super::TASK_SCHEMA, "id": "toy", "version": 1, "family": "checkable-answer",
            "dataset": {"path": "d.jsonl", "sha256": "0".repeat(64), "split": "test", "licence": "CC0-1.0"},
            "prompt_template": "{input}", "extract": {"kind": "after-marker", "marker": "####"},
            "metric": {"kind": "numeric-tolerance", "abs": 0.0}
        }))
        .unwrap();
        assert_eq!(task.score_output("2 + 2 is ... #### 4", "4").unwrap(), 1.0);
        assert_eq!(task.score_output("I do not know", "4").unwrap(), 0.0);
        assert!(score(&Metric::PassAtK { k: 1, samples: 2 }, Some("x"), "x").is_err());
    }
}
