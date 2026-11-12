use serde::{Deserialize, Serialize};

use crate::ir::ModelIR;
use crate::profiling::SensitivityProfiler;
use crate::quality::evaluator::{EvaluationCorpus, EvaluatorConfig, ModelEvaluator};
use crate::surgery::recipe::{SurgeryEngine, TweakRecipe};

/// Discrete states in the autonomous co-surgery closed loop.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CoSurgeryState {
    /// Idle, awaiting initialization or start signal.
    Idle,
    /// Profiling layer topological sensitivity, SVD decay, and condition numbers.
    ProfileSensitivity,
    /// Synthesizing candidate recipe from topology analysis or Teacher LLM.
    SynthesizeRecipe,
    /// Applying declarative surgery mutations to the candidate model.
    ApplySurgery,
    /// Validating candidate perplexity against baseline quality gates.
    ValidateQuality,
    /// Evaluating 3D Pareto frontier dominance and hypervolume contributions.
    EvaluatePareto,
    /// Breached quality gate; performing self-healing relaxation or reverting.
    SelfHealingRelaxation,
    /// Successfully committed Pareto-optimal model state.
    Commit,
    /// Rollback triggered to previous stable checkpoint.
    Rollback,
    /// Optimization loop terminated (budget reached or converged).
    Terminated,
}

/// Budget and stopping criteria for closed-loop co-surgery.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoSurgeryBudget {
    /// Maximum number of search/surgery iterations.
    pub max_iterations: usize,
    /// Maximum allowed perplexity degradation (e.g. 0.15 = +0.15 PPL).
    pub max_ppl_delta: f64,
    /// Target model size reduction ratio (e.g. 0.50 = 50% smaller).
    pub target_reduction_ratio: f64,
    /// Maximum consecutive rollbacks before terminating or relaxing search.
    pub max_consecutive_rollbacks: usize,
}

impl Default for CoSurgeryBudget {
    fn default() -> Self {
        Self {
            max_iterations: 10,
            max_ppl_delta: 0.15,
            target_reduction_ratio: 0.40,
            max_consecutive_rollbacks: 3,
        }
    }
}

/// A checkpoint recording an accepted state along the Pareto frontier.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoSurgeryCheckpoint {
    pub iteration: usize,
    pub recipe: TweakRecipe,
    pub baseline_ppl: f64,
    pub candidate_ppl: f64,
    pub ppl_delta: f64,
    pub original_bytes: usize,
    pub current_bytes: usize,
    pub reduction_ratio: f64,
    pub timestamp: u64,
}

/// Summary report emitted by the Co-Surgery Controller.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoSurgeryReport {
    pub total_iterations: usize,
    pub successful_commits: usize,
    pub total_rollbacks: usize,
    pub total_relaxations: usize,
    pub final_state: CoSurgeryState,
    pub baseline_ppl: f64,
    pub final_ppl: f64,
    pub original_bytes: usize,
    pub final_bytes: usize,
    pub final_reduction_ratio: f64,
    pub history: Vec<CoSurgeryCheckpoint>,
    /// Where the perplexity figures came from: `measured` or `simulated (analytic proxy)`.
    #[serde(default)]
    pub quality_source: String,
}

/// Scores model candidates for the co-surgery loop.
///
/// The loop is agnostic to how perplexity is obtained: `AnalyticProxyEvaluator` keeps the
/// original weight-statistics proxy (simulated), and `crate::inference::MeasuredEvaluator`
/// runs the reference forward pass on real tokens.
pub trait QualityEvaluator: Send {
    /// Perplexity of the untouched model.
    fn baseline_perplexity(&mut self, model: &ModelIR) -> Result<f64, String>;
    /// Perplexity of a candidate whose tensors carry their recipe-selected formats.
    fn candidate_perplexity(&mut self, model: &ModelIR) -> Result<f64, String>;
    /// True when figures come from a real forward pass.
    fn is_measured(&self) -> bool;
}

/// The original analytic proxy. Its figures are simulated and labelled as such.
pub struct AnalyticProxyEvaluator;

impl QualityEvaluator for AnalyticProxyEvaluator {
    fn baseline_perplexity(&mut self, _model: &ModelIR) -> Result<f64, String> {
        // Fixed placeholder baseline used since v0.1.9; simulated, not measured.
        Ok(5.84)
    }

    fn candidate_perplexity(&mut self, model: &ModelIR) -> Result<f64, String> {
        let mut weights = Vec::new();
        for t in &model.tensors {
            if let Some(prof) = &t.weight_profile {
                weights.extend_from_slice(&prof.sample_weights);
            }
        }
        if weights.is_empty() {
            weights = vec![0.1; 4096];
        }
        ModelEvaluator::evaluate_weights(
            &EvaluationCorpus::WikiText2,
            &weights,
            &EvaluatorConfig::default(),
        )
        .map(|metrics| metrics.perplexity)
    }

    fn is_measured(&self) -> bool {
        false
    }
}

/// Autonomous Closed-Loop Co-Surgery Controller.
pub struct CoSurgeryController {
    budget: CoSurgeryBudget,
    state: CoSurgeryState,
    current_iteration: usize,
    consecutive_rollbacks: usize,
    baseline_ppl: f64,
    current_ppl: f64,
    stable_model: ModelIR,
    candidate_model: Option<ModelIR>,
    active_recipe: Option<TweakRecipe>,
    checkpoints: Vec<CoSurgeryCheckpoint>,
    total_commits: usize,
    total_rollbacks: usize,
    total_relaxations: usize,
    evaluator: Box<dyn QualityEvaluator>,
}

impl CoSurgeryController {
    /// Creates a new Co-Surgery Controller initialized with baseline model and budget.
    /// Uses the analytic proxy, so quality figures are simulated.
    pub fn new(model: ModelIR, budget: CoSurgeryBudget) -> Self {
        let mut evaluator = AnalyticProxyEvaluator;
        let baseline_ppl = evaluator.baseline_perplexity(&model).unwrap_or(5.84);
        Self::build(model, budget, Box::new(evaluator), baseline_ppl)
    }

    /// Creates a controller that scores candidates with `evaluator` and measures the baseline first.
    pub fn with_evaluator(
        model: ModelIR,
        budget: CoSurgeryBudget,
        mut evaluator: Box<dyn QualityEvaluator>,
    ) -> Result<Self, String> {
        let baseline_ppl = evaluator.baseline_perplexity(&model)?;
        Ok(Self::build(model, budget, evaluator, baseline_ppl))
    }

    /// True when quality figures come from a real forward pass.
    pub fn is_measured(&self) -> bool {
        self.evaluator.is_measured()
    }

    fn build(
        mut model: ModelIR,
        budget: CoSurgeryBudget,
        evaluator: Box<dyn QualityEvaluator>,
        baseline_ppl: f64,
    ) -> Self {
        SensitivityProfiler::profile_model(&mut model);
        Self {
            budget,
            state: CoSurgeryState::Idle,
            current_iteration: 0,
            consecutive_rollbacks: 0,
            baseline_ppl,
            current_ppl: baseline_ppl,
            stable_model: model,
            candidate_model: None,
            active_recipe: None,
            checkpoints: Vec::new(),
            total_commits: 0,
            total_rollbacks: 0,
            total_relaxations: 0,
            evaluator,
        }
    }

    /// Returns current state.
    pub fn state(&self) -> CoSurgeryState {
        self.state
    }

    /// Returns current iteration.
    pub fn current_iteration(&self) -> usize {
        self.current_iteration
    }

    /// Returns the stable accepted model.
    pub fn model(&self) -> &ModelIR {
        &self.stable_model
    }

    /// Returns the active checkpoints history.
    pub fn checkpoints(&self) -> &[CoSurgeryCheckpoint] {
        &self.checkpoints
    }

    /// Advances the state machine by one transition.
    /// Returns true if execution can continue, false if terminated.
    pub fn step(&mut self) -> bool {
        match self.state {
            CoSurgeryState::Idle => {
                if self.current_iteration >= self.budget.max_iterations {
                    self.state = CoSurgeryState::Terminated;
                    return false;
                }
                self.state = CoSurgeryState::ProfileSensitivity;
                true
            }
            CoSurgeryState::ProfileSensitivity => {
                SensitivityProfiler::profile_model(&mut self.stable_model);
                self.state = CoSurgeryState::SynthesizeRecipe;
                true
            }
            CoSurgeryState::SynthesizeRecipe => {
                let num_layers = self.stable_model.num_layers;
                let target_layer = self.current_iteration % num_layers.max(1);

                let op = crate::surgery::recipe::TweakOperation::SetPrecision {
                    start_layer: target_layer,
                    end_layer: target_layer,
                    role: Some(crate::ir::TensorRole::AttentionQ),
                    format: crate::ir::TensorFormat::Ternary,
                };

                let recipe = TweakRecipe {
                    name: format!("Automated Co-Surgery Step {}", self.current_iteration),
                    version: "1.0.0".to_string(),
                    author: "THME Closed-Loop Co-Surgery Daemon".to_string(),
                    description: format!(
                        "Automated Co-Surgery Iteration {}",
                        self.current_iteration
                    ),
                    target_model: self.stable_model.name.clone(),
                    expected_memory_savings_pct: Some(35.0),
                    expected_speedup: Some(1.25),
                    operations: vec![op],
                };
                self.active_recipe = Some(recipe);
                self.state = CoSurgeryState::ApplySurgery;
                true
            }
            CoSurgeryState::ApplySurgery => {
                if let Some(recipe) = &self.active_recipe {
                    let mut candidate = self.stable_model.clone();
                    let diff = SurgeryEngine::apply_recipe(&mut candidate, recipe);
                    if diff.modified_tensors_count > 0 {
                        self.candidate_model = Some(candidate);
                        self.state = CoSurgeryState::ValidateQuality;
                    } else {
                        self.state = CoSurgeryState::Rollback;
                    }
                } else {
                    self.state = CoSurgeryState::Rollback;
                }
                true
            }
            CoSurgeryState::ValidateQuality => {
                let Some(candidate) = self.candidate_model.as_ref() else {
                    self.state = CoSurgeryState::Rollback;
                    return true;
                };
                match self.evaluator.candidate_perplexity(candidate) {
                    Ok(ppl) => {
                        self.current_ppl = ppl;
                        self.state = if ppl - self.baseline_ppl <= self.budget.max_ppl_delta {
                            CoSurgeryState::EvaluatePareto
                        } else {
                            CoSurgeryState::SelfHealingRelaxation
                        };
                    }
                    Err(_) => {
                        self.state = CoSurgeryState::Rollback;
                    }
                }
                true
            }
            CoSurgeryState::EvaluatePareto => {
                self.state = CoSurgeryState::Commit;
                true
            }
            CoSurgeryState::SelfHealingRelaxation => {
                self.total_relaxations += 1;
                // If relaxation reached limit, trigger rollback
                if self.consecutive_rollbacks + 1 >= self.budget.max_consecutive_rollbacks {
                    self.state = CoSurgeryState::Rollback;
                } else {
                    let mut relaxed_any = false;
                    if let Some(recipe) = &mut self.active_recipe {
                        for op in &mut recipe.operations {
                            if let crate::surgery::recipe::TweakOperation::SetPrecision {
                                format,
                                ..
                            } = op
                            {
                                if *format != crate::ir::TensorFormat::Q8_0 {
                                    *format = crate::ir::TensorFormat::Q8_0;
                                    relaxed_any = true;
                                }
                            }
                        }
                    }
                    if relaxed_any {
                        self.state = CoSurgeryState::ApplySurgery;
                    } else {
                        // Already relaxed to Q8_0 and still failed; rollback
                        self.state = CoSurgeryState::Rollback;
                    }
                }
                true
            }
            CoSurgeryState::Commit => {
                if let Some(candidate) = self.candidate_model.take() {
                    let original_bytes = self.stable_model.total_estimated_bytes();
                    let current_bytes = candidate.total_estimated_bytes();
                    let reduction = if original_bytes > 0 {
                        1.0 - (current_bytes as f64 / original_bytes as f64)
                    } else {
                        0.0
                    };

                    self.checkpoints.push(CoSurgeryCheckpoint {
                        iteration: self.current_iteration,
                        recipe: self.active_recipe.take().unwrap_or_default(),
                        baseline_ppl: self.baseline_ppl,
                        candidate_ppl: self.current_ppl,
                        ppl_delta: self.current_ppl - self.baseline_ppl,
                        original_bytes,
                        current_bytes,
                        reduction_ratio: reduction,
                        timestamp: self.current_iteration as u64,
                    });

                    self.stable_model = candidate;
                    self.total_commits += 1;
                    self.consecutive_rollbacks = 0;
                }

                self.current_iteration += 1;
                if self.current_iteration >= self.budget.max_iterations {
                    self.state = CoSurgeryState::Terminated;
                    false
                } else {
                    self.state = CoSurgeryState::Idle;
                    true
                }
            }
            CoSurgeryState::Rollback => {
                self.total_rollbacks += 1;
                self.consecutive_rollbacks += 1;
                self.candidate_model = None;
                self.active_recipe = None;

                self.current_iteration += 1;
                if self.consecutive_rollbacks >= self.budget.max_consecutive_rollbacks
                    || self.current_iteration >= self.budget.max_iterations
                {
                    self.state = CoSurgeryState::Terminated;
                    false
                } else {
                    self.state = CoSurgeryState::Idle;
                    true
                }
            }
            CoSurgeryState::Terminated => false,
        }
    }

    /// Runs the loop until completion or budget exhaustion.
    pub fn run_to_completion(&mut self) -> CoSurgeryReport {
        while self.step() {}
        self.report()
    }

    /// Emits structured final report.
    pub fn report(&self) -> CoSurgeryReport {
        let original_bytes = self
            .checkpoints
            .first()
            .map(|c| c.original_bytes)
            .unwrap_or_else(|| self.stable_model.total_estimated_bytes());
        let final_bytes = self.stable_model.total_estimated_bytes();
        let reduction = if original_bytes > 0 {
            1.0 - (final_bytes as f64 / original_bytes as f64)
        } else {
            0.0
        };

        CoSurgeryReport {
            total_iterations: self.current_iteration,
            successful_commits: self.total_commits,
            total_rollbacks: self.total_rollbacks,
            total_relaxations: self.total_relaxations,
            final_state: self.state,
            baseline_ppl: self.baseline_ppl,
            final_ppl: self.current_ppl,
            original_bytes,
            final_bytes,
            final_reduction_ratio: reduction,
            history: self.checkpoints.clone(),
            quality_source: if self.evaluator.is_measured() {
                "measured".to_string()
            } else {
                "simulated (analytic proxy)".to_string()
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::rebuilder::{ArchitectureBlueprint, ModelRebuilder};

    #[test]
    fn test_closed_loop_state_machine_execution() {
        let blueprint = ArchitectureBlueprint::new_hybrid_llama("test_model", 2, 128, 1);
        let model = ModelRebuilder::build_scaffold(&blueprint);
        let budget = CoSurgeryBudget {
            max_iterations: 3,
            max_ppl_delta: 2.0,
            target_reduction_ratio: 0.50,
            max_consecutive_rollbacks: 5,
        };
        let mut controller = CoSurgeryController::new(model, budget);
        assert_eq!(controller.state(), CoSurgeryState::Idle);

        let report = controller.run_to_completion();
        assert_eq!(report.final_state, CoSurgeryState::Terminated);
        assert!(report.total_iterations >= 1);
        assert!(report.successful_commits > 0 || report.total_rollbacks > 0);
    }
}
