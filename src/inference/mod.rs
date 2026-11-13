//! Reference CPU inference path for measured quality evaluation.
//!
//! Runs a real forward pass over file-backed LLaMA-family and GPT-NeoX weights so that
//! perplexity figures are measurements rather than analytic estimates. The
//! implementation favours clarity and determinism over throughput: it is a
//! correctness reference for the quality gate, not a serving runtime.

// `as_chunks` needs Rust 1.88; kept for readability; the minimum toolchain (1.89) also has s_chunks.
#![allow(clippy::chunks_exact_to_as_chunks)]

pub mod arch;
pub mod calibration;
pub mod evaluator;
pub mod llama;
pub mod neox;
pub mod perplexity;
pub mod probe;
pub mod smoothing;
pub mod weights;

pub use arch::{ArchConfig, CausalLm};
pub use calibration::{collect_calibration, CalibrationFile, TensorCalibration};
pub use evaluator::MeasuredEvaluator;
pub use llama::{synthetic_demo, DepthExit, LlamaConfig, LlamaModel, WeightPrecision};
pub use neox::{NeoxConfig, NeoxModel};
pub use perplexity::{evaluate_perplexity, read_token_file, PerplexityConfig, PerplexityReport};
pub use probe::{ForwardTrace, InputChannelStats, LayerTrace, TensorActivity, ACTIVITY_BINS};
pub use weights::{apply_selected_format, fake_quantize, read_tensor_f32, BYOA_METADATA_KEY};
