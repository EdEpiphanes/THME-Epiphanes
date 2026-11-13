// SPDX-License-Identifier: AGPL-3.0-only
pub mod engine;
pub mod synthesizer;
pub mod tree;

pub use engine::{
    SpeculativeEngine, SpeculativeExecutionConfig, SpeculativeStepTelemetry,
    SpeculativeTelemetryReport, SpeculativeVerificationStrategy,
};
pub use synthesizer::{
    DraftExtractionConfig, DraftExtractionStrategy, DraftSynthesisReport, DraftSynthesizer,
};
pub use tree::{SpeculativeTreeNode, SpeculativeTreeTopology};
