// SPDX-License-Identifier: AGPL-3.0-only
pub mod agent_bridge;
pub mod closed_loop;
pub mod pareto_3d;
pub mod prompt_synth;
pub mod recipe;
pub mod relaxation;
pub mod watchdog;

pub use agent_bridge::{AgentBridge, AgentBridgeConfig};
pub use closed_loop::{
    AnalyticProxyEvaluator, CoSurgeryBudget, CoSurgeryCheckpoint, CoSurgeryController,
    CoSurgeryReport, CoSurgeryState, QualityEvaluator,
};
pub use pareto_3d::{ParetoOptimizer3D, ParetoPoint3D};
pub use prompt_synth::{PromptSynthesizer, TeacherPromptPayload};
pub use recipe::{SurgeryDiff, SurgeryEngine, TweakOperation, TweakRecipe};
pub use relaxation::RelaxationEngine;
pub use watchdog::{RollbackWatchdog, WatchdogDecision};
