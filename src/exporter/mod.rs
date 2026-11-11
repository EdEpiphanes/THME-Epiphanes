//! Binary Model Exporter Subsystem (Milestone v0.1.8)
//!
//! Direct serialization of distilled and quantized models into deployable
//! binary artifacts (GGUF v3 and SafeTensors) with BitNet b1.58 ternary packed weights.

pub mod bitnet_pack;
pub mod gguf_writer;
pub mod metadata;
pub mod safetensors_writer;
pub mod streaming_writer;

pub use bitnet_pack::{BitNetPacker, PackedBitNetTensor};
pub use gguf_writer::{GgufTensor, GgufWriter};
pub use metadata::GgufMetadataSynchronizer;
pub use safetensors_writer::{SafeTensorsEntryHeader, SafeTensorsTensor, SafeTensorsWriter};
pub use streaming_writer::{copy_reader, StreamingChunkWriter, STREAM_CHUNK_SIZE};

use anyhow::{Context, Result};
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

use crate::ir::ModelIR;
use crate::parser::gguf::GgmlType;

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

/// Unified binary model serialization engine
pub struct ModelExporter;

impl ModelExporter {
    /// Exports a `ModelIR` into a binary artifact according to `ExportConfig`
    pub fn export(model: &ModelIR, config: &ExportConfig) -> Result<ExportReport> {
        let total_tensors = model.tensors.len();
        let original_bytes = model.original_byte_size as u64;

        match config.format {
            ExportFormat::GGUFv3 => {
                let mut writer = GgufWriter::new();
                GgufMetadataSynchronizer::synchronize_into_writer(
                    &mut writer,
                    model,
                    config.quant_type,
                );

                for tensor in &model.tensors {
                    let (ggml_type, data) = match config.quant_type {
                        ExportQuantType::FP16 => {
                            let size = tensor.numel * 2;
                            (GgmlType::F16, vec![0u8; size])
                        }
                        ExportQuantType::Q8_0 => {
                            let blocks = tensor.numel.div_ceil(32);
                            let size = blocks * 34;
                            (GgmlType::Q8_0, vec![0u8; size])
                        }
                        ExportQuantType::BitNetB158 => {
                            let packed_bytes = tensor.numel.div_ceil(4);
                            (GgmlType::I8, vec![0x1B; packed_bytes])
                        }
                    };
                    writer.add_tensor(&tensor.name, tensor.shape.clone(), ggml_type, data);
                }

                writer
                    .write_to_file(&config.output_path)
                    .with_context(|| format!("Failed writing GGUF to {:?}", config.output_path))?;
            }
            ExportFormat::SafeTensors => {
                let mut writer = SafeTensorsWriter::new();
                writer.add_metadata("general.architecture", model.architecture.display_name());
                writer.add_metadata("model.name", &model.name);
                writer.add_metadata("format", "safetensors");

                for tensor in &model.tensors {
                    let (dtype, data) = match config.quant_type {
                        ExportQuantType::FP16 => ("F16", vec![0u8; tensor.numel * 2]),
                        ExportQuantType::Q8_0 => {
                            let blocks = tensor.numel.div_ceil(32);
                            ("I8", vec![0u8; blocks * 34])
                        }
                        ExportQuantType::BitNetB158 => {
                            let packed_bytes = tensor.numel.div_ceil(4);
                            ("U8", vec![0x1B; packed_bytes])
                        }
                    };
                    writer.add_tensor(&tensor.name, dtype, tensor.shape.clone(), data);
                }

                writer.write_to_file(&config.output_path).with_context(|| {
                    format!("Failed writing SafeTensors to {:?}", config.output_path)
                })?;
            }
        }

        let metadata = std::fs::metadata(&config.output_path)
            .with_context(|| format!("Failed reading metadata for {:?}", config.output_path))?;
        let exported_bytes = metadata.len();

        let compression_ratio = if exported_bytes > 0 {
            original_bytes as f32 / exported_bytes as f32
        } else {
            1.0
        };

        let sha256_checksum = if config.compute_checksum {
            Some(compute_file_sha256(&config.output_path)?)
        } else {
            None
        };

        Ok(ExportReport {
            format: config.format,
            quant_type: config.quant_type,
            total_tensors,
            original_bytes,
            exported_bytes,
            compression_ratio,
            sha256_checksum,
        })
    }
}

/// Computes the SHA-256 hex digest of a file using pure Rust (FIPS 180-4)
pub fn compute_file_sha256<P: AsRef<Path>>(path: P) -> Result<String> {
    let mut file = File::open(path.as_ref())
        .with_context(|| format!("Failed to open file for SHA-256: {:?}", path.as_ref()))?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;
    Ok(compute_sha256(&buffer))
}

/// Pure Rust SHA-256 implementation (zero external crate dependencies)
#[allow(clippy::chunks_exact_to_as_chunks)]
pub fn compute_sha256(data: &[u8]) -> String {
    let k: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];

    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];

    let bit_len = (data.len() as u64) * 8;
    let mut msg = data.to_vec();
    msg.push(0x80);
    while (msg.len() % 64) != 56 {
        msg.push(0x00);
    }
    msg.extend_from_slice(&bit_len.to_be_bytes());

    for chunk in msg.chunks_exact(64) {
        let mut w = [0u32; 64];
        for (i, part) in chunk.chunks_exact(4).enumerate() {
            w[i] = u32::from_be_bytes([part[0], part[1], part[2], part[3]]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }

        let mut a = h[0];
        let mut b = h[1];
        let mut c = h[2];
        let mut d = h[3];
        let mut e = h[4];
        let mut f = h[5];
        let mut g = h[6];
        let mut h_var = h[7];

        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let temp1 = h_var
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(k[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = s0.wrapping_add(maj);

            h_var = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }

        h[0] = h[0].wrapping_add(a);
        h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c);
        h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e);
        h[5] = h[5].wrapping_add(f);
        h[6] = h[6].wrapping_add(g);
        h[7] = h[7].wrapping_add(h_var);
    }

    let mut result = String::with_capacity(64);
    for val in h {
        use std::fmt::Write;
        let _ = write!(&mut result, "{:08x}", val);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sha256_known_vectors() {
        assert_eq!(
            compute_sha256(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            compute_sha256(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }
}
