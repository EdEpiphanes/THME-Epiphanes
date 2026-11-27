// SPDX-License-Identifier: AGPL-3.0-only
//! Binary Model Exporter Subsystem (Milestone v0.1.8)
//!
//! Direct serialization of distilled and quantized models into deployable
//! binary artifacts (GGUF v3 and SafeTensors) with BitNet b1.58 ternary packed weights.

pub mod bitnet_pack;
pub mod calibration;
pub mod conversion;
pub mod ggml_names;
pub mod gguf_check;
pub mod gguf_writer;
pub mod kquant;
pub mod metadata;
pub mod safetensors_writer;
pub mod streaming_writer;
pub mod tokenizer_meta;

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
    /// Each tensor in its own selected format (`TensorIR::selected_format`, set by a recipe):
    /// FP16, Q8_0, Q4_0, Q4_K, Q5_K, Q6_K or ternary (TQ1_0); FP32 and 1-D tensors keep their
    /// stored dtype, as the measured path does.
    PerTensor,
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
            "per-tensor" | "per_tensor" | "recipe" | "selected" => Ok(ExportQuantType::PerTensor),
            other => Err(format!(
                "Unknown export quant type '{}'. Supported: bitnet-b1.58, q8_0, fp16, per-tensor",
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
    #[serde(default)]
    pub llama_cpp_layout: bool,
    #[serde(default)]
    pub tokenizer_json_path: Option<PathBuf>,
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
    /// Things the file lacks that the user should know about; empty for a default export.
    #[serde(default)]
    pub warnings: Vec<String>,
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
                export_gguf(model, config)?;
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

        let mut warnings = Vec::new();
        if config.llama_cpp_layout && config.tokenizer_json_path.is_some() {
            warnings.push(
                "No tokenizer.ggml.pre was written; llama.cpp will use its default pre-tokenizer and may warn that quality is degraded (THME-3102)."
                    .to_string(),
            );
        }

        Ok(ExportReport {
            format: config.format,
            quant_type: config.quant_type,
            total_tensors,
            original_bytes,
            exported_bytes,
            compression_ratio,
            sha256_checksum,
            warnings,
        })
    }
}

fn export_gguf(model: &ModelIR, config: &ExportConfig) -> Result<()> {
    let target_type = match config.quant_type {
        ExportQuantType::PerTensor => None,
        ExportQuantType::FP16 => Some(("F16", GgmlType::F16)),
        ExportQuantType::Q8_0 => Some(("Q8_0", GgmlType::Q8_0)),
        ExportQuantType::TQ1_0 => Some(("TQ1_0", GgmlType::TQ1_0)),
        ExportQuantType::BitNetB158 => anyhow::bail!(
            "BitNet conversion is not available for source-backed export; choose a matching FP16 or Q8_0 source"
        ),
    };

    let mut writer = GgufWriter::new();
    GgufMetadataSynchronizer::synchronize_into_writer(&mut writer, model, config.quant_type);
    if config.llama_cpp_layout {
        add_missing_rope_dimension_count(&mut writer, model)?;
        add_tokenizer_metadata(&mut writer, config)?;
    }

    for tensor in &model.tensors {
        let source = tensor.source.clone().with_context(|| {
            format!(
                "Tensor '{}' has no file-backed payload; parse a model from disk before exporting",
                tensor.name
            )
        })?;
        let export_name = if config.llama_cpp_layout {
            ggml_names::ggml_tensor_name(&tensor.name).with_context(|| {
                format!(
                    "Tensor '{}' has no GGML tensor name mapping for llama.cpp layout",
                    tensor.name
                )
            })?
        } else {
            tensor.name.clone()
        };

        if config.llama_cpp_layout && ggml_names::is_norm_tensor(&export_name) {
            if source.byte_len != u64::try_from(tensor.byte_size)? {
                anyhow::bail!(
                    "Tensor '{}' source payload length does not match recorded byte size",
                    tensor.name
                );
            }
            if tensor.original_dtype == "F32" {
                writer.add_tensor_source(&export_name, tensor.shape.clone(), GgmlType::F32, source);
            } else {
                let values = crate::inference::read_tensor_f32(tensor)?;
                writer.add_tensor(
                    &export_name,
                    tensor.shape.clone(),
                    GgmlType::F32,
                    f32_slice_to_le_bytes(&values),
                );
            }
            continue;
        }

        let plan = match target_type {
            None => per_tensor_plan(tensor)?,
            Some((target_name, target_ggml)) => {
                if tensor.original_dtype == target_name {
                    ensure_source_dtype(tensor, target_name)?;
                    TensorPlan::Copy(target_ggml)
                } else if matches!(
                    config.quant_type,
                    ExportQuantType::FP16 | ExportQuantType::Q8_0 | ExportQuantType::TQ1_0
                ) && matches!(tensor.original_dtype.as_str(), "F16" | "F32" | "BF16")
                {
                    let row_width = tensor.shape.last().copied().unwrap_or(1);
                    if !row_width.is_multiple_of(target_ggml.block_size()) {
                        anyhow::bail!(
                            "Tensor '{}' row width {} must be divisible by {} for {}",
                            tensor.name,
                            row_width,
                            target_ggml.block_size(),
                            target_name
                        );
                    }
                    if source.byte_len != u64::try_from(tensor.byte_size)? {
                        anyhow::bail!(
                            "Tensor '{}' source payload length does not match recorded byte size",
                            tensor.name
                        );
                    }
                    TensorPlan::Convert(target_ggml)
                } else {
                    anyhow::bail!(
                        "Tensor '{}' has source dtype {}, cannot export as {}",
                        tensor.name,
                        tensor.original_dtype,
                        target_name
                    );
                }
            }
        };

        if config.llama_cpp_layout {
            if let Some(n_heads) = rope_head_count(&export_name, model) {
                if source.byte_len != u64::try_from(tensor.byte_size)? {
                    anyhow::bail!(
                        "Tensor '{}' source payload length does not match recorded byte size",
                        tensor.name
                    );
                }
                let target = match plan {
                    TensorPlan::Copy(ggml_type) | TensorPlan::Convert(ggml_type) => ggml_type,
                };
                let payload = encode_permuted_rope_tensor(tensor, &export_name, n_heads, target)?;
                writer.add_tensor(&export_name, tensor.shape.clone(), target, payload);
                continue;
            }
        }

        match plan {
            TensorPlan::Copy(ggml_type) => {
                ensure_source_dtype(tensor, &tensor.original_dtype)?;
                writer.add_tensor_source(&export_name, tensor.shape.clone(), ggml_type, source);
            }
            TensorPlan::Convert(ggml_type) => {
                if source.byte_len != u64::try_from(tensor.byte_size)? {
                    anyhow::bail!(
                        "Tensor '{}' source payload length does not match recorded byte size",
                        tensor.name
                    );
                }
                writer.add_converted_tensor_source(
                    &export_name,
                    tensor.shape.clone(),
                    ggml_type,
                    source,
                    tensor.original_dtype.clone(),
                );
            }
        }
    }

    writer
        .write_to_file(&config.output_path)
        .with_context(|| format!("Failed writing GGUF to {:?}", config.output_path))?;
    if config.llama_cpp_layout {
        let file = File::open(&config.output_path).with_context(|| {
            format!(
                "Failed to open exported GGUF for llama.cpp validation: {:?}",
                config.output_path
            )
        })?;
        // SAFETY: File is opened read-only immediately after flushing and not mutated during mapping.
        let mmap = unsafe { memmap2::Mmap::map(&file)? };
        let errors: Vec<String> = gguf_check::check_llama_cpp_requirements(&mmap)
            .into_iter()
            .filter(|f| f.severity == gguf_check::Severity::Error)
            .map(|f| f.to_string())
            .collect();
        if !errors.is_empty() {
            anyhow::bail!(
                "Exported GGUF failed llama.cpp compatibility check: {}",
                errors.join("; ")
            );
        }
    }
    Ok(())
}

/// Writes `{arch}.rope.dimension_count = hidden_size / head_count` when absent from metadata.
fn add_missing_rope_dimension_count(writer: &mut GgufWriter, model: &ModelIR) -> Result<()> {
    let arch = GgufMetadataSynchronizer::architecture_prefix(&model.architecture);
    let key = format!("{arch}.rope.dimension_count");
    if writer.metadata().iter().any(|(k, _)| k == &key) {
        return Ok(());
    }
    if model.num_attention_heads == 0 {
        anyhow::bail!("Cannot compute {key}: model has 0 attention heads");
    }
    let dim = u32::try_from(model.hidden_size / model.num_attention_heads)
        .with_context(|| format!("{key} exceeds u32"))?;
    writer.add_u32(key, dim);
    Ok(())
}

fn add_tokenizer_metadata(writer: &mut GgufWriter, config: &ExportConfig) -> Result<()> {
    let Some(tok_path) = config.tokenizer_json_path.as_deref() else {
        return Ok(());
    };
    let tokenizer_json = std::fs::read_to_string(tok_path)
        .with_context(|| format!("Failed to read tokenizer.json from {:?}", tok_path))?;
    let config_json = tok_path
        .parent()
        .map(|dir| dir.join("config.json"))
        .filter(|path| path.is_file())
        .map(|path| {
            std::fs::read_to_string(&path)
                .with_context(|| format!("Failed to read config.json from {:?}", path))
        })
        .transpose()?;
    for (key, value) in tokenizer_meta::tokenizer_metadata(&tokenizer_json, config_json.as_deref())?
    {
        writer.add_metadata(key, value);
    }
    Ok(())
}

fn rope_head_count(ggml_name: &str, model: &ModelIR) -> Option<usize> {
    if ggml_name.starts_with("blk.") && ggml_name.ends_with(".attn_q.weight") {
        Some(model.num_attention_heads)
    } else if ggml_name.starts_with("blk.") && ggml_name.ends_with(".attn_k.weight") {
        Some(model.num_key_value_heads)
    } else {
        None
    }
}

fn encode_permuted_rope_tensor(
    tensor: &crate::ir::TensorIR,
    ggml_name: &str,
    n_heads: usize,
    target: GgmlType,
) -> Result<Vec<u8>> {
    if tensor.shape.len() != 2 {
        anyhow::bail!(
            "Tensor '{}' must be 2-D to apply RoPE row permutation, got shape {:?}",
            tensor.name,
            tensor.shape
        );
    }
    let (rows, cols) = (tensor.shape[0], tensor.shape[1]);
    let values = crate::inference::read_tensor_f32(tensor)?;
    let permuted = ggml_names::permute_rope_rows(&values, rows, cols, n_heads).map_err(|err| {
        anyhow::anyhow!(
            "Failed to permute RoPE rows for tensor '{}' ({ggml_name}): {err}",
            tensor.name
        )
    })?;
    encode_f32_values(&permuted, target, &tensor.name)
}

fn encode_f32_values(values: &[f32], target: GgmlType, tensor_name: &str) -> Result<Vec<u8>> {
    match target {
        GgmlType::F32 => Ok(f32_slice_to_le_bytes(values)),
        GgmlType::BF16 => Ok(values
            .iter()
            .flat_map(|&v| crate::engine::minifloat::f32_to_bf16(v).to_le_bytes())
            .collect()),
        _ if conversion::is_conversion_target(target) => {
            let f32_bytes = f32_slice_to_le_bytes(values);
            let mut out = Vec::new();
            conversion::encode_float_source(
                &mut std::io::Cursor::new(f32_bytes),
                &mut out,
                "F32",
                target,
                values.len(),
            )
            .with_context(|| {
                format!(
                    "Cannot encode tensor '{}' as {}",
                    tensor_name,
                    target.type_name()
                )
            })?;
            Ok(out)
        }
        other => anyhow::bail!(
            "Tensor '{}' targets {}, which is not supported after RoPE row permutation",
            tensor_name,
            other.type_name()
        ),
    }
}

fn f32_slice_to_le_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// A tensor's payload in its selected format, by the same plan the per-tensor GGUF export uses:
/// the stored bytes when they already match (or the tensor is 1-D or FP32), otherwise the float
/// source encoded with THME's block encoders. Returns the stored data type name and the bytes.
pub fn encoded_payload(tensor: &crate::ir::TensorIR) -> Result<(String, Vec<u8>)> {
    let stored = crate::inference::read_tensor_bytes(tensor)?;
    match per_tensor_plan(tensor)? {
        TensorPlan::Copy(ggml_type) => Ok((ggml_type.type_name().to_string(), stored)),
        TensorPlan::Convert(ggml_type) => {
            let mut out = Vec::new();
            conversion::encode_float_source(
                &mut std::io::Cursor::new(stored),
                &mut out,
                &tensor.original_dtype,
                ggml_type,
                tensor.numel,
            )
            .with_context(|| {
                format!(
                    "Cannot encode tensor '{}' as {}",
                    tensor.name,
                    ggml_type.type_name()
                )
            })?;
            Ok((ggml_type.type_name().to_string(), out))
        }
    }
}

/// How one tensor is written in a per-tensor export.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TensorPlan {
    /// Copy the stored payload unchanged.
    Copy(GgmlType),
    /// Encode the float source into this block format.
    Convert(GgmlType),
}

/// GGML type of a stored dtype that can be copied into GGUF unchanged.
fn stored_ggml_type(dtype: &str) -> Option<GgmlType> {
    Some(match dtype {
        "F32" => GgmlType::F32,
        "F16" => GgmlType::F16,
        "BF16" => GgmlType::BF16,
        "Q8_0" => GgmlType::Q8_0,
        "Q4_0" => GgmlType::Q4_0,
        "TQ1_0" => GgmlType::TQ1_0,
        "Q4_K" => GgmlType::Q4_K,
        "Q5_K" => GgmlType::Q5_K,
        "Q6_K" => GgmlType::Q6_K,
        _ => return None,
    })
}

/// Decides how a tensor is written from its selected format. Mirrors the measured path
/// (`inference::load_tensor`): FP32 means "keep as stored", and 1-D tensors (norms, biases)
/// always keep their stored dtype, so an exported recipe matches what `thme evaluate` measured.
fn per_tensor_plan(tensor: &crate::ir::TensorIR) -> Result<TensorPlan> {
    use crate::ir::TensorFormat;
    let stored = || {
        stored_ggml_type(&tensor.original_dtype).with_context(|| {
            format!(
                "Tensor '{}' has stored dtype {}, which cannot be copied into GGUF",
                tensor.name, tensor.original_dtype
            )
        })
    };
    if tensor.shape.len() < 2 {
        return Ok(TensorPlan::Copy(stored()?));
    }
    let target = match &tensor.selected_format {
        TensorFormat::FP32 => return Ok(TensorPlan::Copy(stored()?)),
        TensorFormat::FP16 => GgmlType::F16,
        TensorFormat::BF16 if tensor.original_dtype == "BF16" => GgmlType::BF16,
        TensorFormat::Q8_0 => GgmlType::Q8_0,
        TensorFormat::Q4_0 => GgmlType::Q4_0,
        TensorFormat::Q4_K => GgmlType::Q4_K,
        TensorFormat::Q5_K => GgmlType::Q5_K,
        TensorFormat::Q6_K => GgmlType::Q6_K,
        TensorFormat::Ternary => GgmlType::TQ1_0,
        TensorFormat::Custom(id) => anyhow::bail!(
            "Tensor '{}' uses BYOA algorithm '{id}', which has no GGUF encoding; evaluate it with `thme evaluate`, or choose a built-in format for export",
            tensor.name
        ),
        other => anyhow::bail!(
            "Tensor '{}' is assigned {}, which the GGUF exporter cannot write (supported: FP16, Q8_0, Q4_0, Q4_K, Q5_K, Q6_K, Ternary, or FP32 to keep as stored)",
            tensor.name,
            other.display_name()
        ),
    };
    if tensor.original_dtype == target.type_name() {
        return Ok(TensorPlan::Copy(target));
    }
    if !matches!(tensor.original_dtype.as_str(), "F32" | "F16" | "BF16") {
        anyhow::bail!(
            "Tensor '{}' is stored as {}; converting to {} needs an F32, F16 or BF16 source",
            tensor.name,
            tensor.original_dtype,
            target.type_name()
        );
    }
    let row_width = tensor.shape.last().copied().unwrap_or(1);
    if !row_width.is_multiple_of(target.block_size()) {
        anyhow::bail!(
            "Tensor '{}' rows are {row_width} wide; {} needs rows divisible by {} (choose a 32-value format such as Q8_0 or Q4_0 for this tensor)",
            tensor.name,
            target.type_name(),
            target.block_size()
        );
    }
    Ok(TensorPlan::Convert(target))
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
            let mut bytes = [0u8; 64];
            bytes.copy_from_slice(block);
            compress_sha256_block(&mut self.state, &bytes);
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

    #[test]
    fn export_llama_cpp_layout_writes_tokenizer_keys_and_flag_off_keeps_output_unchanged() {
        use crate::ir::model::ModelMetadataValue as MV;
        use crate::parser::gguf::GgufParser;
        use crate::parser::ModelImporter;

        let dir = tempfile::tempdir().expect("temp dir");
        let model_dir = dir.path().join("model");
        std::fs::create_dir_all(&model_dir).expect("create model dir");

        let one = 0x3C00u16.to_le_bytes();
        let mut st = SafeTensorsWriter::new();
        st.add_tensor(
            "model.embed_tokens.weight",
            "F16",
            vec![5, 4],
            one.repeat(20),
        );
        st.add_tensor(
            "model.layers.0.self_attn.q_proj.weight",
            "F16",
            vec![4, 4],
            one.repeat(16),
        );
        st.add_tensor(
            "model.layers.0.self_attn.k_proj.weight",
            "F16",
            vec![4, 4],
            one.repeat(16),
        );
        st.add_tensor(
            "model.layers.0.input_layernorm.weight",
            "F16",
            vec![4],
            one.repeat(4),
        );
        st.add_tensor("model.norm.weight", "F16", vec![4], one.repeat(4));
        st.write_to_file(model_dir.join("model.safetensors"))
            .expect("write safetensors");

        std::fs::write(
            model_dir.join("config.json"),
            r#"{"architectures":["LlamaForCausalLM"],"hidden_size":4,"num_hidden_layers":1,"num_attention_heads":1,"num_key_value_heads":1,"vocab_size":5,"bos_token_id":0,"eos_token_id":1}"#,
        )
        .expect("write config.json");
        let tok_path = model_dir.join("tokenizer.json");
        std::fs::write(
            &tok_path,
            r#"{
                "added_tokens": [
                    {"id": 0, "content": "<s>", "special": true},
                    {"id": 1, "content": "</s>", "special": true}
                ],
                "pre_tokenizer": {"type": "ByteLevel"},
                "decoder": {"type": "ByteLevel"},
                "model": {
                    "type": "BPE",
                    "vocab": {"<s>": 0, "</s>": 1, "a": 2, "b": 3, "ab": 4},
                    "merges": ["a b"]
                }
            }"#,
        )
        .expect("write tokenizer.json");

        let model = ModelImporter::import(&model_dir).expect("import model");

        let out_off_none = dir.path().join("off_none.gguf");
        let out_off_with_path = dir.path().join("off_with_path.gguf");
        let out_on = dir.path().join("on.gguf");

        let off_report = ModelExporter::export(
            &model,
            &ExportConfig {
                format: ExportFormat::GGUFv3,
                quant_type: ExportQuantType::FP16,
                output_path: out_off_none.clone(),
                compute_checksum: false,
                llama_cpp_layout: false,
                tokenizer_json_path: None,
            },
        )
        .expect("export off none");
        ModelExporter::export(
            &model,
            &ExportConfig {
                format: ExportFormat::GGUFv3,
                quant_type: ExportQuantType::FP16,
                output_path: out_off_with_path.clone(),
                compute_checksum: false,
                llama_cpp_layout: false,
                tokenizer_json_path: Some(tok_path.clone()),
            },
        )
        .expect("export off with path");
        assert_eq!(
            std::fs::read(&out_off_none).expect("read off_none"),
            std::fs::read(&out_off_with_path).expect("read off_with_path"),
            "llama_cpp_layout=false must ignore tokenizer_json_path"
        );

        let on_report = ModelExporter::export(
            &model,
            &ExportConfig {
                format: ExportFormat::GGUFv3,
                quant_type: ExportQuantType::FP16,
                output_path: out_on.clone(),
                compute_checksum: false,
                llama_cpp_layout: true,
                tokenizer_json_path: Some(tok_path),
            },
        )
        .expect("export with llama_cpp_layout");
        assert!(
            off_report.warnings.is_empty(),
            "default export warns: {:?}",
            off_report.warnings
        );
        assert_eq!(on_report.warnings.len(), 1, "{:?}", on_report.warnings);
        assert!(
            on_report.warnings[0].contains("tokenizer.ggml.pre"),
            "{:?}",
            on_report.warnings
        );

        let bytes = std::fs::read(&out_on).expect("read exported gguf");
        let findings = gguf_check::check_gguf(&bytes);
        assert!(
            !findings
                .iter()
                .any(|f| f.severity == gguf_check::Severity::Error),
            "unexpected gguf errors: {:?}",
            findings
        );

        let parsed = GgufParser::parse_from_file(&out_on).expect("parse exported gguf");
        let meta = &parsed.source_gguf_metadata;
        assert_eq!(
            meta.get("tokenizer.ggml.model"),
            Some(&MV::String("gpt2".to_string()))
        );
        assert_eq!(
            meta.get("tokenizer.ggml.tokens"),
            Some(&MV::Array(vec![
                MV::String("<s>".to_string()),
                MV::String("</s>".to_string()),
                MV::String("a".to_string()),
                MV::String("b".to_string()),
                MV::String("ab".to_string()),
            ]))
        );
        assert_eq!(
            meta.get("tokenizer.ggml.scores"),
            Some(&MV::Array(vec![
                MV::Float32(0.0),
                MV::Float32(0.0),
                MV::Float32(0.0),
                MV::Float32(0.0),
                MV::Float32(0.0),
            ]))
        );
        assert_eq!(
            meta.get("tokenizer.ggml.token_type"),
            Some(&MV::Array(vec![
                MV::Int32(3),
                MV::Int32(3),
                MV::Int32(1),
                MV::Int32(1),
                MV::Int32(1),
            ]))
        );
        assert_eq!(
            meta.get("tokenizer.ggml.merges"),
            Some(&MV::Array(vec![MV::String("a b".to_string())]))
        );
        assert_eq!(
            meta.get("tokenizer.ggml.bos_token_id"),
            Some(&MV::Uint32(0))
        );
        assert_eq!(
            meta.get("tokenizer.ggml.eos_token_id"),
            Some(&MV::Uint32(1))
        );
    }

    #[test]
    fn export_llama_cpp_layout_fails_when_tokenizer_tokens_are_missing() {
        use crate::parser::ModelImporter;

        let dir = tempfile::tempdir().expect("temp dir");
        let model_dir = dir.path().join("model");
        std::fs::create_dir_all(&model_dir).expect("create model dir");

        let one = 0x3C00u16.to_le_bytes();
        let mut st = SafeTensorsWriter::new();
        st.add_tensor(
            "model.embed_tokens.weight",
            "F16",
            vec![5, 4],
            one.repeat(20),
        );
        st.add_tensor("model.norm.weight", "F16", vec![4], one.repeat(4));
        st.write_to_file(model_dir.join("model.safetensors"))
            .expect("write safetensors");
        std::fs::write(
            model_dir.join("config.json"),
            r#"{"architectures":["LlamaForCausalLM"],"hidden_size":4,"num_hidden_layers":1,"num_attention_heads":1,"num_key_value_heads":1,"vocab_size":5}"#,
        )
        .expect("write config.json");

        let model = ModelImporter::import(&model_dir).expect("import model");
        let err = ModelExporter::export(
            &model,
            &ExportConfig {
                format: ExportFormat::GGUFv3,
                quant_type: ExportQuantType::FP16,
                output_path: dir.path().join("missing_tok.gguf"),
                compute_checksum: false,
                llama_cpp_layout: true,
                tokenizer_json_path: None,
            },
        )
        .unwrap_err();
        let msg = format!("{err:#}");
        assert!(msg.contains("tokenizer.ggml.tokens"), "{msg}");
    }
}
