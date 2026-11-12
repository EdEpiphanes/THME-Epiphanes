//! Reference CPU inference path for measured quality evaluation.
//!
//! Runs a real forward pass over file-backed LLaMA-family weights so that
//! perplexity figures are measurements rather than analytic estimates. The
//! implementation favours clarity and determinism over throughput: it is a
//! correctness reference for the quality gate, not a serving runtime.

// `as_chunks` needs Rust 1.88; the crate's minimum toolchain is 1.80.
#![allow(clippy::chunks_exact_to_as_chunks)]

pub mod llama;
pub mod perplexity;
pub mod weights;

pub use llama::{DepthExit, LlamaConfig, LlamaModel, WeightPrecision};
pub use perplexity::{evaluate_perplexity, read_token_file, PerplexityConfig, PerplexityReport};
pub use weights::{fake_quantize, read_tensor_f32};
