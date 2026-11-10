pub mod allocator;
pub mod cli;
pub mod gui;
pub mod ir;
pub mod parser;
pub mod profiling;
pub mod quality;
pub mod report;
pub mod surgery;

pub use allocator::{AllocationPlan, ParetoAllocator};
pub use gui::{launch_gui, THMEApp};
pub use ir::{ModelArchitecture, ModelIR, TensorFormat, TensorIR, TensorRole};
pub use parser::ModelImporter;
pub use profiling::{KvCacheEstimator, SensitivityProfiler};
pub use quality::{GateDecision, GateResult, QualityGate};
pub use surgery::{SurgeryDiff, SurgeryEngine, TweakOperation, TweakRecipe};
