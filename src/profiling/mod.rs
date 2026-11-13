// SPDX-License-Identifier: AGPL-3.0-only
pub mod cache_sim;
pub mod hardware;
pub mod intensity;
pub mod kv_cache;
pub mod roofline;
pub mod sensitivity;
pub mod thermal;
pub mod token_flow;
pub mod weight_stats;

pub use cache_sim::{
    CacheSimulationSummary, CacheSimulator4Tier, CacheTierLevel, FourTierCacheSimulator,
    TierAccessReport,
};
pub use hardware::{CacheTierSpec, HardwareDevice, HardwareSpec, HardwareTarget, PrecisionType};
pub use intensity::{
    ExecutionPhase, IntensityCalculator, KernelIntensityReport, KernelType,
    LayerArithmeticIntensity,
};
pub use kv_cache::{KvCacheEstimator, KvCacheProjection};
pub use roofline::{
    BoundClassification, LayerRooflinePoint, RooflineEngine, RooflineModel, RooflineProfileReport,
    RooflineReport,
};
pub use sensitivity::{LayerSensitivityReport, SensitivityProfiler};
pub use thermal::{DynamicThermalModel, ThermalConfig, ThermalModel, ThermalStepReport};
pub use token_flow::{LiveFlowSimulator, LiveFlowSnapshot, TokenFlowStatus, TransformerStage};
pub use weight_stats::WeightAnalyzer;
pub use weight_stats::{bf16_to_f32, f16_to_f32};
