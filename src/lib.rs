pub mod allocator;
pub mod cli;
pub mod dataset;
pub mod experiment;
pub mod exporter;
pub mod gui;
pub mod interop;
pub mod ir;
pub mod parser;
pub mod profiling;
pub mod quality;
pub mod report;
pub mod speculative;
pub mod surgery;
pub mod trainer;
pub mod transform;

pub use allocator::{AllocationPlan, ParetoAllocator};
pub use dataset::{Dataset, DatasetFormat, DatasetSample, DatasetSummary};
pub use experiment::{ExperimentLab, ExperimentMetrics, ExperimentTechnique};
pub use exporter::{
    compute_sha256, BitNetPacker, ExportConfig, ExportFormat, ExportQuantType, ExportReport,
    GgufMetadataSynchronizer, GgufTensor, GgufWriter, ModelExporter, PackedBitNetTensor,
    SafeTensorsEntryHeader, SafeTensorsTensor, SafeTensorsWriter,
};
pub use gui::{launch_gui, THMEApp};
pub use interop::{
    HuggingFaceConfigGenerator, PeftLoraAdapter, PyTorchCUDATrainingStrategy, PyTorchInteropConfig,
    PyTorchScriptGenerator,
};
pub use ir::{
    ModelArchitecture, ModelIR, ParadigmConversionConfig, ParadigmConversionReport,
    ParadigmConverter, SpecializedModelParadigm, TensorFormat, TensorIR, TensorRole,
};
pub use parser::ModelImporter;
pub use profiling::{KvCacheEstimator, SensitivityProfiler};
pub use quality::{
    BenchmarkScore, EvaluationCorpus, EvaluatorConfig, GateDecision, GateResult, ModelEvaluator,
    PerplexityGateDecision, PerplexityGateResult, PerplexityMetrics, QualityGate,
};
pub use speculative::{
    DraftExtractionConfig, DraftExtractionStrategy, DraftSynthesisReport, DraftSynthesizer,
    SpeculativeEngine, SpeculativeExecutionConfig, SpeculativeStepTelemetry,
    SpeculativeTelemetryReport, SpeculativeTreeNode, SpeculativeTreeTopology,
    SpeculativeVerificationStrategy,
};
pub use surgery::{SurgeryDiff, SurgeryEngine, TweakOperation, TweakRecipe};
pub use trainer::{
    MicroTrainer, OptimizerType, TrainingConfig, TrainingLossType, TrainingStepRecord,
};
pub use transform::{
    AutoDecider, MatrixAlgorithm, MatrixMath, ParameterTransformDecision, TransformPlanSummary,
};
