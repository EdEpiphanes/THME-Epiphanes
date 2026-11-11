//! Binary Model Exporter Subsystem (Milestone v0.1.8)
//!
//! Direct serialization of distilled and quantized models into deployable
//! binary artifacts (GGUF v3 and SafeTensors) with BitNet b1.58 ternary packed weights.

pub mod bitnet_pack;

pub use bitnet_pack::{BitNetPacker, PackedBitNetTensor};

use std::path::PathBuf;

/// Target export container format
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ExportFormat {
    /// Pure Rust GGUF v3 binary writer adhering to GGML specification
    GGUFv3,
    /// Pure Rust SafeTensors binary format with 8-byte JSON header prefix
    SafeTensors,
}

impl std::str::FromStr for ExportFormat {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "gguf" | "ggufv3" | "gguf_v3" => Ok(ExportFormat::GGUFv3),
            "safetensors" | "st" => Ok(ExportFormat::SafeTensors),
            other => Err(format!(
                "Unknown export format '{}'. Supported: gguf, safetensors",
                other
            )),
        }
    }
}

/// Quantization scheme for exported weights
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ExportQuantType {
    /// BitNet b1.58 ternary packed representation (2 bits/weight, 4 weights/byte)
    BitNetB158,
    /// 8-bit integer quantization with per-block FP16 scale
    Q8_0,
    /// Full 16-bit floating point weights
    FP16,
}

impl std::str::FromStr for ExportQuantType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "bitnet" | "bitnet-b1.58" | "bitnet_b158" | "ternary" => {
                Ok(ExportQuantType::BitNetB158)
            }
            "q8" | "q8_0" | "int8" => Ok(ExportQuantType::Q8_0),
            "fp16" | "f16" => Ok(ExportQuantType::FP16),
            other => Err(format!(
                "Unknown export quant type '{}'. Supported: bitnet-b1.58, q8_0, fp16",
                other
            )),
        }
    }
}

/// Configuration parameters for model serialization
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ExportConfig {
    pub format: ExportFormat,
    pub quant_type: ExportQuantType,
    pub output_path: PathBuf,
    pub compute_checksum: bool,
}

/// Statistical telemetry report produced following export
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ExportReport {
    pub format: ExportFormat,
    pub quant_type: ExportQuantType,
    pub total_tensors: usize,
    pub original_bytes: u64,
    pub exported_bytes: u64,
    pub compression_ratio: f32,
    pub sha256_checksum: Option<String>,
}
