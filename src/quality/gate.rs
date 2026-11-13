// SPDX-License-Identifier: AGPL-3.0-only
//! The quality gate (roadmap v0.2.8.1).
//!
//! A gate is versioned data: a name, a version and one or more criteria. Each criterion names a
//! metric, its direction, and the degradation it allows as an absolute margin, a relative
//! margin, or both (both must hold). Every report records the gate's name, version and content
//! hash, so a result is never re-labelled under a different gate. A criterion whose metric was
//! not measured fails as `not measured`; nothing passes by omission.
//!
//! Every gate decision in THME goes through [`GateSpec::judge`].

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::path::Path;

/// One requirement of a gate.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Criterion {
    /// Metric name as the measuring tool reports it, e.g. `perplexity`, `cross_entropy_nats`.
    pub metric: String,
    /// `true` for losses and perplexity, `false` for accuracies and scores.
    pub lower_is_better: bool,
    /// Largest allowed degradation in the metric's own units.
    #[serde(default)]
    pub max_abs: Option<f64>,
    /// Largest allowed degradation as a fraction of the baseline (0.01 = 1%).
    #[serde(default)]
    pub max_rel: Option<f64>,
}

/// A versioned quality gate.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GateSpec {
    pub name: String,
    pub version: u32,
    pub criteria: Vec<Criterion>,
}

/// One measured metric: its value on the baseline and on the candidate.
#[derive(Debug, Clone, PartialEq)]
pub struct Measure {
    pub metric: String,
    pub baseline: f64,
    pub candidate: f64,
}

impl Measure {
    pub fn new(metric: &str, baseline: f64, candidate: f64) -> Self {
        Self {
            metric: metric.to_string(),
            baseline,
            candidate,
        }
    }
}

/// How one criterion came out.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct CriterionOutcome {
    pub metric: String,
    /// `pass`, `fail` or `not measured`.
    pub status: &'static str,
    pub baseline: Option<f64>,
    pub candidate: Option<f64>,
    /// Signed so that positive is worse, in the metric's units.
    pub degradation: Option<f64>,
    pub relative_degradation: Option<f64>,
}

/// The gate's decision.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct GateVerdict {
    pub gate: String,
    pub gate_hash: String,
    pub accepted: bool,
    pub outcomes: Vec<CriterionOutcome>,
}

/// Name of THME's default gate.
pub const DEFAULT_GATE: &str = "thme-perplexity";
/// Default absolute perplexity margin, as used by every committed measured report.
pub const DEFAULT_MAX_PPL_DELTA: f64 = 0.45;

impl GateSpec {
    /// A gate with one perplexity criterion and an absolute margin.
    pub fn perplexity(max_abs: f64) -> Self {
        Self {
            name: DEFAULT_GATE.into(),
            version: 1,
            criteria: vec![Criterion {
                metric: "perplexity".into(),
                lower_is_better: true,
                max_abs: Some(max_abs),
                max_rel: None,
            }],
        }
    }

    /// Reads a gate from a JSON file and checks it.
    pub fn load(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("Failed to read gate {path:?}"))?;
        let spec: Self = serde_json::from_str(&text)
            .with_context(|| format!("Gate file {path:?} is not a valid gate"))?;
        spec.validate()?;
        Ok(spec)
    }

    /// Refuses gates that cannot decide anything or would decide nonsense.
    pub fn validate(&self) -> Result<()> {
        if self.name.trim().is_empty() {
            bail!("A gate needs a name");
        }
        if self.criteria.is_empty() {
            bail!("Gate '{}' has no criteria", self.name);
        }
        let mut seen = BTreeSet::new();
        for c in &self.criteria {
            if !seen.insert(c.metric.as_str()) {
                bail!("Gate '{}' lists metric '{}' twice", self.name, c.metric);
            }
            if c.max_abs.is_none() && c.max_rel.is_none() {
                bail!(
                    "Criterion '{}' of gate '{}' has neither max_abs nor max_rel",
                    c.metric,
                    self.name
                );
            }
            for m in [c.max_abs, c.max_rel].into_iter().flatten() {
                if !m.is_finite() || m < 0.0 {
                    bail!("Criterion '{}' has an invalid margin {m}", c.metric);
                }
            }
        }
        Ok(())
    }

    /// `name@vN`.
    pub fn id(&self) -> String {
        format!("{}@v{}", self.name, self.version)
    }

    /// SHA-256 of the gate's canonical JSON; changes with any threshold.
    pub fn hash(&self) -> String {
        let bytes = serde_json::to_vec(self).unwrap_or_default();
        crate::exporter::compute_sha256(&bytes)
    }

    /// Judges a candidate on the measured metrics. Every criterion must pass.
    pub fn judge(&self, measures: &[Measure]) -> GateVerdict {
        let outcomes: Vec<CriterionOutcome> = self
            .criteria
            .iter()
            .map(|c| {
                let Some(m) = measures.iter().find(|m| m.metric == c.metric) else {
                    return CriterionOutcome {
                        metric: c.metric.clone(),
                        status: "not measured",
                        baseline: None,
                        candidate: None,
                        degradation: None,
                        relative_degradation: None,
                    };
                };
                let degradation = if c.lower_is_better {
                    m.candidate - m.baseline
                } else {
                    m.baseline - m.candidate
                };
                let relative = (m.baseline != 0.0).then(|| degradation / m.baseline.abs());
                let finite = m.baseline.is_finite() && m.candidate.is_finite();
                let abs_ok = c.max_abs.is_none_or(|a| degradation <= a);
                let rel_ok = c.max_rel.is_none_or(|r| relative.is_some_and(|x| x <= r));
                CriterionOutcome {
                    metric: c.metric.clone(),
                    status: if finite && abs_ok && rel_ok {
                        "pass"
                    } else {
                        "fail"
                    },
                    baseline: Some(m.baseline),
                    candidate: Some(m.candidate),
                    degradation: Some(degradation),
                    relative_degradation: relative,
                }
            })
            .collect();
        GateVerdict {
            gate: self.id(),
            gate_hash: self.hash(),
            accepted: outcomes.iter().all(|o| o.status == "pass"),
            outcomes,
        }
    }

    /// Whether a candidate perplexity passes this gate against a baseline perplexity
    /// (other criteria count as not measured).
    pub fn accepts_perplexity(&self, baseline: f64, candidate: f64) -> bool {
        self.judge(&[Measure::new("perplexity", baseline, candidate)])
            .accepted
    }

    /// Fails early when the gate names a metric the calling tool cannot measure.
    pub fn require_metrics(&self, available: &[&str], tool: &str) -> Result<()> {
        for c in &self.criteria {
            if !available.contains(&c.metric.as_str()) {
                bail!(
                    "Gate {} needs metric '{}', which {tool} does not measure (it measures {})",
                    self.id(),
                    c.metric,
                    available.join(", ")
                );
            }
        }
        Ok(())
    }

    /// The gate as recorded in a report: identity, hash and criteria.
    pub fn to_report(&self) -> serde_json::Value {
        serde_json::json!({
            "name": self.name,
            "version": self.version,
            "id": self.id(),
            "hash": self.hash(),
            "criteria": self.criteria,
        })
    }

    /// The absolute perplexity margin, when the gate has one (kept in reports for older tools).
    pub fn max_ppl_delta(&self) -> Option<f64> {
        self.criteria
            .iter()
            .find(|c| c.metric == "perplexity")
            .and_then(|c| c.max_abs)
    }
}

/// Resolves a command's gate: a gate file when given, otherwise the perplexity margin.
pub fn resolve(gate_file: Option<&Path>, max_ppl_delta: f64) -> Result<GateSpec> {
    match gate_file {
        Some(path) => GateSpec::load(path),
        None => {
            let spec = GateSpec::perplexity(max_ppl_delta);
            spec.validate()?;
            Ok(spec)
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Earlier interface, now a thin layer over `GateSpec::judge`.

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum GateDecision {
    Accept,
    Reject,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkScore {
    pub task_name: String,
    pub baseline_score: f64,
    pub candidate_score: f64,
    pub allowed_degradation: f64,
    #[serde(default)]
    pub lower_is_better: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GateResult {
    pub decision: GateDecision,
    pub degradation: f64,
    pub max_allowed_degradation: f64,
    pub task_evaluations: Vec<(String, bool)>,
    pub summary: String,
}

pub struct QualityGate;

impl QualityGate {
    /// Higher-is-better metric with an absolute margin.
    pub fn evaluate_candidate(
        baseline_score: f64,
        candidate_score: f64,
        max_allowed_degradation: f64,
    ) -> GateResult {
        Self::evaluate_candidate_directional(
            baseline_score,
            candidate_score,
            max_allowed_degradation,
            false,
        )
    }

    /// One metric with an explicit direction and an absolute margin.
    pub fn evaluate_candidate_directional(
        baseline_score: f64,
        candidate_score: f64,
        max_allowed_degradation: f64,
        lower_is_better: bool,
    ) -> GateResult {
        let result = Self::evaluate_suite(&[BenchmarkScore {
            task_name: "primary_benchmark".into(),
            baseline_score,
            candidate_score,
            allowed_degradation: max_allowed_degradation,
            lower_is_better,
        }]);
        let degradation = result.degradation;
        let summary = if result.decision == GateDecision::Accept {
            format!(
                "PASSED: Observed degradation ({degradation:.2} pts) is within allowed threshold ({max_allowed_degradation:.2} pts). Model accepted."
            )
        } else {
            format!(
                "REJECTED: Observed degradation ({degradation:.2} pts) exceeds allowed threshold ({max_allowed_degradation:.2} pts). Transformation rejected."
            )
        };
        GateResult {
            max_allowed_degradation,
            summary,
            ..result
        }
    }

    /// Several tasks, each with its own direction and absolute margin; all must pass.
    pub fn evaluate_suite(scores: &[BenchmarkScore]) -> GateResult {
        let spec = GateSpec {
            name: "suite".into(),
            version: 1,
            criteria: scores
                .iter()
                .map(|s| Criterion {
                    metric: s.task_name.clone(),
                    lower_is_better: s.lower_is_better,
                    max_abs: Some(s.allowed_degradation),
                    max_rel: None,
                })
                .collect(),
        };
        let measures: Vec<Measure> = scores
            .iter()
            .map(|s| Measure::new(&s.task_name, s.baseline_score, s.candidate_score))
            .collect();
        let verdict = spec.judge(&measures);
        let task_evaluations: Vec<(String, bool)> = verdict
            .outcomes
            .iter()
            .map(|o| (o.metric.clone(), o.status == "pass"))
            .collect();
        let degradations: Vec<f64> = verdict
            .outcomes
            .iter()
            .filter_map(|o| o.degradation)
            .collect();
        let average = if degradations.is_empty() {
            0.0
        } else {
            degradations.iter().sum::<f64>() / degradations.len() as f64
        };
        let accepted = verdict.accepted && !scores.is_empty();
        let summary = if accepted || scores.is_empty() {
            format!(
                "PASSED: All {} benchmark tasks passed their individual quality gates.",
                scores.len()
            )
        } else {
            let failed: Vec<&str> = task_evaluations
                .iter()
                .filter(|(_, p)| !*p)
                .map(|(n, _)| n.as_str())
                .collect();
            format!(
                "REJECTED: Quality degradation in tasks [{}]. Transformation must be rolled back or tuned.",
                failed.join(", ")
            )
        };
        GateResult {
            decision: if accepted || scores.is_empty() {
                GateDecision::Accept
            } else {
                GateDecision::Reject
            },
            degradation: average,
            max_allowed_degradation: 0.0,
            task_evaluations,
            summary,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn criterion(metric: &str, lower: bool, abs: Option<f64>, rel: Option<f64>) -> Criterion {
        Criterion {
            metric: metric.into(),
            lower_is_better: lower,
            max_abs: abs,
            max_rel: rel,
        }
    }

    #[test]
    fn absolute_and_relative_margins_both_hold() {
        let gate = GateSpec {
            name: "demo".into(),
            version: 1,
            criteria: vec![criterion("perplexity", true, Some(0.5), Some(0.02))],
        };
        gate.validate().unwrap();
        // +0.3 on 20.0 is 1.5%: passes both.
        assert!(
            gate.judge(&[Measure::new("perplexity", 20.0, 20.3)])
                .accepted
        );
        // +0.3 on 10.0 is 3%: passes absolute, fails relative.
        let v = gate.judge(&[Measure::new("perplexity", 10.0, 10.3)]);
        assert!(!v.accepted);
        assert!((v.outcomes[0].relative_degradation.unwrap() - 0.03).abs() < 1e-12);
        // +0.6 on 100.0 is 0.6%: fails absolute.
        assert!(
            !gate
                .judge(&[Measure::new("perplexity", 100.0, 100.6)])
                .accepted
        );
        // An improvement always passes.
        assert!(
            gate.judge(&[Measure::new("perplexity", 10.0, 9.0)])
                .accepted
        );
    }

    #[test]
    fn direction_missing_metrics_and_non_finite_values() {
        let gate = GateSpec {
            name: "multi".into(),
            version: 2,
            criteria: vec![
                criterion("accuracy", false, Some(1.0), None),
                criterion("cross_entropy_nats", true, Some(0.01), None),
            ],
        };
        let v = gate.judge(&[
            Measure::new("accuracy", 70.0, 69.5),
            Measure::new("cross_entropy_nats", 3.0, 3.005),
        ]);
        assert!(v.accepted);
        assert_eq!(v.gate, "multi@v2");
        // Accuracy falling by more than a point fails (higher is better).
        assert!(
            !gate
                .judge(&[
                    Measure::new("accuracy", 70.0, 68.5),
                    Measure::new("cross_entropy_nats", 3.0, 3.0)
                ])
                .accepted
        );
        // A criterion that was not measured never passes.
        let v = gate.judge(&[Measure::new("accuracy", 70.0, 70.0)]);
        assert!(!v.accepted);
        assert_eq!(v.outcomes[1].status, "not measured");
        // NaN never passes.
        assert!(
            !gate
                .judge(&[
                    Measure::new("accuracy", 70.0, f64::NAN),
                    Measure::new("cross_entropy_nats", 3.0, 3.0)
                ])
                .accepted
        );
    }

    #[test]
    fn hash_and_identity_track_every_threshold() {
        let a = GateSpec::perplexity(0.45);
        let b = GateSpec::perplexity(0.25);
        assert_eq!(a.hash(), GateSpec::perplexity(0.45).hash(), "stable");
        assert_ne!(a.hash(), b.hash(), "a threshold change changes the hash");
        let mut c = a.clone();
        c.version = 2;
        assert_ne!(a.hash(), c.hash());
        assert_eq!(a.id(), "thme-perplexity@v1");
        assert_eq!(a.max_ppl_delta(), Some(0.45));
        let v = a.judge(&[Measure::new("perplexity", 10.0, 10.2)]);
        assert_eq!(v.gate_hash, a.hash());
    }

    #[test]
    fn invalid_gates_are_refused() {
        let bad = |criteria: Vec<Criterion>| {
            GateSpec {
                name: "x".into(),
                version: 1,
                criteria,
            }
            .validate()
            .is_err()
        };
        assert!(bad(vec![]));
        assert!(bad(vec![criterion("p", true, None, None)]));
        assert!(bad(vec![criterion("p", true, Some(-1.0), None)]));
        assert!(bad(vec![criterion("p", true, Some(f64::NAN), None)]));
        assert!(bad(vec![
            criterion("p", true, Some(1.0), None),
            criterion("p", true, Some(2.0), None)
        ]));
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("gate.json");
        std::fs::write(&path, r#"{"name": "g", "version": 1, "criteria": []}"#).unwrap();
        assert!(GateSpec::load(&path).is_err());
        std::fs::write(
            &path,
            r#"{"name": "g", "version": 3, "criteria": [{"metric": "perplexity", "lower_is_better": true, "max_rel": 0.01}]}"#,
        )
        .unwrap();
        assert_eq!(GateSpec::load(&path).unwrap().id(), "g@v3");
        assert!(resolve(None, 0.45).is_ok());
    }

    /// Every committed gate file loads; the default file is exactly the built-in default gate.
    #[test]
    fn committed_gate_files_are_valid() {
        let mut found = 0;
        for entry in std::fs::read_dir("configs/gates").unwrap().flatten() {
            let path = entry.path();
            if path.extension().is_some_and(|e| e == "json") {
                GateSpec::load(&path).unwrap_or_else(|e| panic!("{path:?}: {e:#}"));
                found += 1;
            }
        }
        assert!(found >= 2);
        let default = GateSpec::load(Path::new("configs/gates/thme-perplexity-v1.json")).unwrap();
        assert_eq!(
            default.hash(),
            GateSpec::perplexity(DEFAULT_MAX_PPL_DELTA).hash()
        );
    }

    /// The older single-metric API keeps its decisions (it now delegates to `judge`).
    #[test]
    fn legacy_interface_matches_the_spec() {
        assert_eq!(
            QualityGate::evaluate_candidate(70.0, 69.5, 1.0).decision,
            GateDecision::Accept
        );
        assert_eq!(
            QualityGate::evaluate_candidate(70.0, 68.0, 1.0).decision,
            GateDecision::Reject
        );
        assert_eq!(
            QualityGate::evaluate_candidate_directional(10.0, 10.4, 0.45, true).decision,
            GateDecision::Accept
        );
        let suite = QualityGate::evaluate_suite(&[
            BenchmarkScore {
                task_name: "a".into(),
                baseline_score: 1.0,
                candidate_score: 0.5,
                allowed_degradation: 0.1,
                lower_is_better: false,
            },
            BenchmarkScore {
                task_name: "b".into(),
                baseline_score: 1.0,
                candidate_score: 1.0,
                allowed_degradation: 0.1,
                lower_is_better: false,
            },
        ]);
        assert_eq!(suite.decision, GateDecision::Reject);
        assert!(suite.summary.contains("[a]"));
    }
}
