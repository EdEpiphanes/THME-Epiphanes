use serde::{Deserialize, Serialize};

use crate::ir::ModelIR;

/// Watchdog monitoring model mutations and enforcing zero-tolerance rollback on quality degradation.
pub struct RollbackWatchdog {
    max_ppl_delta: f32,
    checkpoints: Vec<(String, ModelIR, f32)>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WatchdogDecision {
    pub allow_commit: bool,
    pub observed_delta: f32,
    pub threshold: f32,
    pub reason: String,
}

impl RollbackWatchdog {
    pub fn new(max_ppl_delta: f32) -> Self {
        Self {
            max_ppl_delta,
            checkpoints: Vec::new(),
        }
    }

    /// Records a baseline checkpoint before applying candidate surgery.
    pub fn save_checkpoint(&mut self, label: &str, model: &ModelIR, baseline_ppl: f32) {
        self.checkpoints
            .push((label.to_string(), model.clone(), baseline_ppl));
    }

    /// Evaluates candidate perplexity against baseline.
    pub fn evaluate(&self, baseline_ppl: f32, candidate_ppl: f32) -> WatchdogDecision {
        let delta = candidate_ppl - baseline_ppl;
        let gate = crate::quality::gate::GateSpec::perplexity(f64::from(self.max_ppl_delta));
        if gate.accepts_perplexity(f64::from(baseline_ppl), f64::from(candidate_ppl)) {
            WatchdogDecision {
                allow_commit: true,
                observed_delta: delta,
                threshold: self.max_ppl_delta,
                reason: format!(
                    "Quality within acceptable delta (+{:.4} <= +{:.4})",
                    delta, self.max_ppl_delta
                ),
            }
        } else {
            WatchdogDecision {
                allow_commit: false,
                observed_delta: delta,
                threshold: self.max_ppl_delta,
                reason: format!(
                    "Quality degradation breached (+{:.4} > +{:.4})",
                    delta, self.max_ppl_delta
                ),
            }
        }
    }

    /// Reverts to last known good checkpoint.
    pub fn rollback_last(&mut self) -> Option<(String, ModelIR, f32)> {
        self.checkpoints.pop()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::rebuilder::{ArchitectureBlueprint, ModelRebuilder};

    #[test]
    fn test_watchdog_acceptance_and_rejection() {
        let mut watchdog = RollbackWatchdog::new(0.15);
        let blueprint = ArchitectureBlueprint::new_hybrid_llama("test", 1, 64, 1);
        let model = ModelRebuilder::build_scaffold(&blueprint);

        watchdog.save_checkpoint("v0", &model, 5.80);

        let good = watchdog.evaluate(5.80, 5.92); // +0.12 <= 0.15
        assert!(good.allow_commit);

        let bad = watchdog.evaluate(5.80, 6.05); // +0.25 > 0.15
        assert!(!bad.allow_commit);

        let restored = watchdog.rollback_last();
        assert!(restored.is_some());
    }
}
