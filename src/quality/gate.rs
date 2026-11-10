use serde::{Deserialize, Serialize};

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
    /// Core quality gate: Q_optimized >= Q_baseline - epsilon (or <= Q_baseline + epsilon for loss/ppl)
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

    /// Evaluates candidate with explicit metric direction (higher-is-better vs lower-is-better)
    pub fn evaluate_candidate_directional(
        baseline_score: f64,
        candidate_score: f64,
        max_allowed_degradation: f64,
        lower_is_better: bool,
    ) -> GateResult {
        let degradation = if lower_is_better {
            candidate_score - baseline_score
        } else {
            baseline_score - candidate_score
        };
        let pass = degradation <= max_allowed_degradation;

        let decision = if pass {
            GateDecision::Accept
        } else {
            GateDecision::Reject
        };

        let summary = if pass {
            format!(
                "PASSED: Observed degradation ({:.2} pts) is within allowed threshold ({:.2} pts). Model accepted.",
                degradation, max_allowed_degradation
            )
        } else {
            format!(
                "REJECTED: Observed degradation ({:.2} pts) exceeds allowed threshold ({:.2} pts). Transformation rejected.",
                degradation, max_allowed_degradation
            )
        };

        GateResult {
            decision,
            degradation,
            max_allowed_degradation,
            task_evaluations: vec![("primary_benchmark".to_string(), pass)],
            summary,
        }
    }

    /// Multi-task suite evaluation
    pub fn evaluate_suite(scores: &[BenchmarkScore]) -> GateResult {
        let mut all_passed = true;
        let mut total_deg = 0.0;
        let mut task_evals = Vec::new();

        for score in scores {
            let deg = if score.lower_is_better {
                score.candidate_score - score.baseline_score
            } else {
                score.baseline_score - score.candidate_score
            };
            total_deg += deg;
            let passed = deg <= score.allowed_degradation;
            task_evals.push((score.task_name.clone(), passed));
            if !passed {
                all_passed = false;
            }
        }

        let avg_deg = if !scores.is_empty() {
            total_deg / scores.len() as f64
        } else {
            0.0
        };

        let decision = if all_passed {
            GateDecision::Accept
        } else {
            GateDecision::Reject
        };

        let summary = if all_passed {
            format!(
                "PASSED: All {} benchmark tasks passed their individual quality gates.",
                scores.len()
            )
        } else {
            let failed_tasks: Vec<&str> = task_evals
                .iter()
                .filter(|(_, p)| !*p)
                .map(|(name, _)| name.as_str())
                .collect();
            format!(
                "REJECTED: Quality degradation in tasks [{}]. Transformation must be rolled back or tuned.",
                failed_tasks.join(", ")
            )
        };

        GateResult {
            decision,
            degradation: avg_deg,
            max_allowed_degradation: 0.0,
            task_evaluations: task_evals,
            summary,
        }
    }
}
