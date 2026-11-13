//! Binary Model Exporter Subsystem (Milestone v0.1.8)
//!
//! Direct serialization of distilled and quantized models into deployable
//! binary artifacts (GGUF v3 and SafeTensors) with BitNet b1.58 ternary packed weights.

pub mod bitnet_pack;
pub mod calibration;
pub mod conversion;
pub mod gguf_writer;
pub mod kquant;
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
use std::io::{BufReader, Read};
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
    /// Standard GGML 256-value ternary block format
    TQ1_0,
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
            "tq1" | "tq1_0" => Ok(ExportQuantType::TQ1_0),
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
        if model.tensors.is_empty() {
            anyhow::bail!("Cannot export a model with no tensors");
        }

        match config.format {
            ExportFormat::GGUFv3 => {
                let target_type = match config.quant_type {
                    ExportQuantType::FP16 => ("F16", GgmlType::F16),
                    ExportQuantType::Q8_0 => ("Q8_0", GgmlType::Q8_0),
                    ExportQuantType::TQ1_0 => ("TQ1_0", GgmlType::TQ1_0),
                    ExportQuantType::BitNetB158 => anyhow::bail!(
                        "BitNet conversion is not available for source-backed export; choose a matching FP16 or Q8_0 source"
                    ),
                };
                let mut writer = GgufWriter::new();
                GgufMetadataSynchronizer::synchronize_into_writer(
                    &mut writer,
                    model,
                    config.quant_type,
                );

                for tensor in &model.tensors {
                    let source = tensor.source.clone().with_context(|| format!(
                        "Tensor '{}' has no file-backed payload; parse a model from disk before exporting", tensor.name
                    ))?;
                    if tensor.original_dtype == target_type.0 {
                        ensure_source_dtype(tensor, target_type.0)?;
                        writer.add_tensor_source(
                            &tensor.name,
                            tensor.shape.clone(),
                            target_type.1,
                            source,
                        );
                    } else if matches!(
                        config.quant_type,
                        ExportQuantType::Q8_0 | ExportQuantType::TQ1_0
                    ) && matches!(tensor.original_dtype.as_str(), "F16" | "F32")
                    {
                        let row_width = tensor.shape.last().copied().unwrap_or(1);
                        if !row_width.is_multiple_of(target_type.1.block_size()) {
                            anyhow::bail!(
                                "Tensor '{}' row width {} must be divisible by {} for {}",
                                tensor.name,
                                row_width,
                                target_type.1.block_size(),
                                target_type.0
                            );
                        }
                        if source.byte_len != u64::try_from(tensor.byte_size)? {
                            anyhow::bail!("Tensor '{}' source payload length does not match recorded byte size", tensor.name);
                        }
                        writer.add_converted_tensor_source(
                            &tensor.name,
                            tensor.shape.clone(),
                            target_type.1,
                            source,
                            tensor.original_dtype.clone(),
                        );
                    } else {
                        anyhow::bail!(
                            "Tensor '{}' has source dtype {}, cannot export as {}",
                            tensor.name,
                            tensor.original_dtype,
                            target_type.0
                        );
                    }
                }

                writer
                    .write_to_file(&config.output_path)
                    .with_context(|| format!("Failed writing GGUF to {:?}", config.output_path))?;
            }
            ExportFormat::SafeTensors => {
                if config.quant_type != ExportQuantType::FP16 {
                    anyhow::bail!(
                        "SafeTensors source-backed export currently supports same-dtype FP16 only; quantization conversions are not available"
                    );
                }
                let mut writer = SafeTensorsWriter::new();
                writer.add_metadata("general.architecture", model.architecture.display_name());
                writer.add_metadata("model.name", &model.name);
                writer.add_metadata("format", "safetensors");

                for tensor in &model.tensors {
                    ensure_source_dtype(tensor, "F16")?;
                    writer.add_tensor_source(
                        &tensor.name,
                        "F16",
                        tensor.shape.clone(),
                        tensor
                            .source
                            .clone()
                            .context("Tensor source validation unexpectedly failed")?,
                    );
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

fn ensure_source_dtype(tensor: &crate::ir::TensorIR, expected_dtype: &str) -> Result<()> {
    let source = tensor.source.as_ref().with_context(|| {
        format!(
            "Tensor '{}' has no file-backed payload; parse a model from disk before exporting",
            tensor.name
        )
    })?;
    if tensor.original_dtype != expected_dtype {
        anyhow::bail!(
            "Tensor '{}' has source dtype {}, but requested output dtype {}; cross-dtype conversion is not available",
            tensor.name,
            tensor.original_dtype,
            expected_dtype
        );
    }
    if source.byte_len != u64::try_from(tensor.byte_size)? {
        anyhow::bail!(
            "Tensor '{}' source payload length {} does not match recorded byte size {}",
            tensor.name,
            source.byte_len,
            tensor.byte_size
        );
    }
    Ok(())
}

/// Computes the SHA-256 hex digest of a file using pure Rust (FIPS 180-4)
pub fn compute_file_sha256<P: AsRef<Path>>(path: P) -> Result<String> {
    let mut file = File::open(path.as_ref())
        .with_context(|| format!("Failed to open file for SHA-256: {:?}", path.as_ref()))?;
    let mut reader = BufReader::new(&mut file);
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; STREAM_CHUNK_SIZE];
    loop {
        let read = reader
            .read(&mut buffer)
            .context("Failed reading file for SHA-256")?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(hasher.finish())
}

/// Pure Rust SHA-256 implementation (zero external crate dependencies)
pub fn compute_sha256(data: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(data);
    hasher.finish()
}

struct Sha256 {
    state: [u32; 8],
    length: u64,
    buffer: [u8; 64],
    buffered: usize,
}

impl Sha256 {
    fn new() -> Self {
        Self {
            state: [
                0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
                0x5be0cd19,
            ],
            length: 0,
            buffer: [0; 64],
            buffered: 0,
        }
    }

    #[allow(clippy::chunks_exact_to_as_chunks)]
    fn update(&mut self, mut data: &[u8]) {
        self.length = self.length.wrapping_add(data.len() as u64);

        if self.buffered > 0 {
            let take = (64 - self.buffered).min(data.len());
            self.buffer[self.buffered..self.buffered + take].copy_from_slice(&data[..take]);
            self.buffered += take;
            data = &data[take..];
            if self.buffered < 64 {
                return;
            }
            compress_sha256_block(&mut self.state, &self.buffer);
            self.buffered = 0;
        }

        let mut blocks = data.chunks_exact(64);
        for block in &mut blocks {
            let block: &[u8; 64] = block.try_into().expect("chunks_exact returns 64 bytes");
            compress_sha256_block(&mut self.state, block);
        }
        let tail = blocks.remainder();
        self.buffer[..tail.len()].copy_from_slice(tail);
        self.buffered = tail.len();
    }

    fn finish(mut self) -> String {
        let bit_len = self.length.wrapping_mul(8);
        self.buffer[self.buffered] = 0x80;
        self.buffered += 1;

        if self.buffered > 56 {
            self.buffer[self.buffered..].fill(0);
            compress_sha256_block(&mut self.state, &self.buffer);
            self.buffer = [0; 64];
        } else {
            self.buffer[self.buffered..56].fill(0);
        }
        self.buffer[56..].copy_from_slice(&bit_len.to_be_bytes());
        compress_sha256_block(&mut self.state, &self.buffer);

        let mut result = String::with_capacity(64);
        for word in self.state {
            use std::fmt::Write;
            let _ = write!(&mut result, "{word:08x}");
        }
        result
    }
}

#[allow(clippy::chunks_exact_to_as_chunks)]
fn compress_sha256_block(state: &mut [u32; 8], block: &[u8; 64]) {
    const K: [u32; 64] = [
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
    let mut schedule = [0u32; 64];
    for (index, bytes) in block.chunks_exact(4).enumerate() {
        schedule[index] = u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    }
    for i in 16..64 {
        let s0 = schedule[i - 15].rotate_right(7)
            ^ schedule[i - 15].rotate_right(18)
            ^ (schedule[i - 15] >> 3);
        let s1 = schedule[i - 2].rotate_right(17)
            ^ schedule[i - 2].rotate_right(19)
            ^ (schedule[i - 2] >> 10);
        schedule[i] = schedule[i - 16]
            .wrapping_add(s0)
            .wrapping_add(schedule[i - 7])
            .wrapping_add(s1);
    }

    let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = *state;
    for i in 0..64 {
        let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
        let ch = (e & f) ^ ((!e) & g);
        let temp1 = h
            .wrapping_add(s1)
            .wrapping_add(ch)
            .wrapping_add(K[i])
            .wrapping_add(schedule[i]);
        let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
        let maj = (a & b) ^ (a & c) ^ (b & c);
        let temp2 = s0.wrapping_add(maj);

        h = g;
        g = f;
        f = e;
        e = d.wrapping_add(temp1);
        d = c;
        c = b;
        b = a;
        a = temp1.wrapping_add(temp2);
    }
    state[0] = state[0].wrapping_add(a);
    state[1] = state[1].wrapping_add(b);
    state[2] = state[2].wrapping_add(c);
    state[3] = state[3].wrapping_add(d);
    state[4] = state[4].wrapping_add(e);
    state[5] = state[5].wrapping_add(f);
    state[6] = state[6].wrapping_add(g);
    state[7] = state[7].wrapping_add(h);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

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

    #[test]
    fn test_file_sha256_streams_across_chunk_boundaries() {
        let bytes = (0..(STREAM_CHUNK_SIZE * 2 + 91))
            .map(|index| (index % 251) as u8)
            .collect::<Vec<_>>();
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("large-payload.bin");
        std::fs::File::create(&path)
            .expect("create payload")
            .write_all(&bytes)
            .expect("write payload");

        assert_eq!(
            compute_file_sha256(path).expect("hash file"),
            compute_sha256(&bytes)
        );
    }

    #[test]
    fn test_sha256_incremental_updates_keep_partial_blocks() {
        let bytes = (0..257).map(|index| index as u8).collect::<Vec<_>>();
        let mut hasher = Sha256::new();
        hasher.update(&bytes[..3]);
        hasher.update(&bytes[3..31]);
        hasher.update(&bytes[31..130]);
        hasher.update(&bytes[130..]);

        assert_eq!(hasher.finish(), compute_sha256(&bytes));
    }
}
