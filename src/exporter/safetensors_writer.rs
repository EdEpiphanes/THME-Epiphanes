//! Direct Pure-Rust SafeTensors Binary Serializer (Milestone v0.1.8.5)
//!
//! Serializes tensor arrays and model metadata into the standard SafeTensors
//! format: an 8-byte little-endian length prefix, a UTF-8 JSON header describing
//! shapes, dtypes, and data byte offsets, followed by contiguous raw tensor bytes.

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufWriter, Seek, SeekFrom, Write};
use std::path::Path;

use crate::exporter::streaming_writer::StreamingChunkWriter;
use crate::ir::tensor::TensorSource;

/// Tensor entry descriptor for SafeTensors JSON header
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SafeTensorsEntryHeader {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub data_offsets: [usize; 2],
}

/// In-memory tensor descriptor for SafeTensors serialization
#[derive(Debug, Clone, PartialEq)]
pub struct SafeTensorsTensor {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub data: Vec<u8>,
}

impl SafeTensorsTensor {
    /// Creates a new tensor descriptor
    pub fn new(
        name: impl Into<String>,
        dtype: impl Into<String>,
        shape: Vec<usize>,
        data: Vec<u8>,
    ) -> Self {
        Self {
            name: name.into(),
            dtype: dtype.into(),
            shape,
            data,
        }
    }
}

/// Pure-Rust SafeTensors Binary Serializer
#[derive(Debug, Clone, Default)]
pub struct SafeTensorsWriter {
    metadata: BTreeMap<String, String>,
    tensors: Vec<SafeTensorsTensor>,
    tensor_sources: Vec<Option<TensorSource>>,
}

impl SafeTensorsWriter {
    /// Creates a new empty SafeTensors writer
    pub fn new() -> Self {
        Self {
            metadata: BTreeMap::new(),
            tensors: Vec::new(),
            tensor_sources: Vec::new(),
        }
    }

    /// Adds a string key-value pair to the `__metadata__` block
    pub fn add_metadata(&mut self, key: impl Into<String>, value: impl Into<String>) -> &mut Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Appends a tensor to be written into the SafeTensors binary container
    pub fn add_tensor(
        &mut self,
        name: impl Into<String>,
        dtype: impl Into<String>,
        shape: Vec<usize>,
        data: Vec<u8>,
    ) -> &mut Self {
        self.tensors
            .push(SafeTensorsTensor::new(name, dtype, shape, data));
        self.tensor_sources.push(None);
        self
    }

    /// Registers a tensor whose unchanged payload will be copied from a source file in chunks.
    pub fn add_tensor_source(
        &mut self,
        name: impl Into<String>,
        dtype: impl Into<String>,
        shape: Vec<usize>,
        source: TensorSource,
    ) -> &mut Self {
        self.tensors
            .push(SafeTensorsTensor::new(name, dtype, shape, Vec::new()));
        self.tensor_sources.push(Some(source));
        self
    }

    /// Number of tensors queued for serialization
    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }

    /// Total bytes of raw tensor payloads
    pub fn payload_byte_size(&self) -> usize {
        self.tensors
            .iter()
            .zip(&self.tensor_sources)
            .fold(0usize, |total, (tensor, source)| {
                let size = match source {
                    Some(source) => usize::try_from(source.byte_len).unwrap_or(usize::MAX),
                    None => tensor.data.len(),
                };
                total.saturating_add(size)
            })
    }

    /// Serializes all registered metadata and tensor buffers into standard SafeTensors binary bytes
    pub fn write_to_bytes(&self) -> Result<Vec<u8>> {
        if self.tensor_sources.iter().any(Option::is_some) {
            bail!("Source-backed SafeTensors must be written with write_to_file");
        }
        let (header_bytes, payload_size) = self.build_header()?;
        let header_len = u64::try_from(header_bytes.len())?;
        let total_size = 8usize
            .checked_add(header_bytes.len())
            .and_then(|size| size.checked_add(payload_size))
            .context("SafeTensors output size overflows")?;
        let mut out = Vec::with_capacity(total_size);

        out.extend_from_slice(&header_len.to_le_bytes());
        for tensor in &self.tensors {
            out.extend_from_slice(&tensor.data);
        }

        let header_start = 8;
        out.splice(header_start..header_start, header_bytes);

        Ok(out)
    }

    /// Writes tensor sources in bounded chunks and buffers only the JSON header.
    pub fn write_to_file<P: AsRef<Path>>(&self, path: P) -> Result<usize> {
        let (header_bytes, _) = self.build_header()?;
        let path_ref = path.as_ref();
        if let Some(parent) = path_ref.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("Failed to create directory: {:?}", parent))?;
        }

        let file = File::create(path_ref)
            .with_context(|| format!("Failed to create SafeTensors file: {:?}", path_ref))?;
        let mut file = BufWriter::new(file);
        let header_len = u64::try_from(header_bytes.len())?;
        let mut total_written = 0usize;
        file.write_all(&header_len.to_le_bytes())?;
        file.write_all(&header_bytes)?;
        total_written += 8 + header_bytes.len();

        for (tensor, source) in self.tensors.iter().zip(&self.tensor_sources) {
            if let Some(source) = source {
                let mut source_file = File::open(&source.path)
                    .map_err(|_| anyhow::anyhow!("Failed to open SafeTensors source payload"))?;
                source_file
                    .seek(SeekFrom::Start(source.byte_offset))
                    .map_err(|_| anyhow::anyhow!("Failed to seek to SafeTensors source payload"))?;
                let mut chunk_writer = StreamingChunkWriter::new(&mut file);
                total_written +=
                    usize::try_from(chunk_writer.write_reader(&mut source_file, source.byte_len)?)?;
            } else {
                file.write_all(&tensor.data)?;
                total_written += tensor.data.len();
            }
        }
        file.flush()?;

        Ok(total_written)
    }

    fn build_header(&self) -> Result<(Vec<u8>, usize)> {
        if self.tensors.is_empty() && self.metadata.is_empty() {
            bail!("Cannot write empty SafeTensors: no tensors or metadata registered");
        }
        if self.tensors.len() != self.tensor_sources.len() {
            bail!("SafeTensors tensor source table is inconsistent");
        }

        let mut header_map = serde_json::Map::new();
        if !self.metadata.is_empty() {
            let meta_json = serde_json::to_value(&self.metadata)
                .context("Failed to serialize SafeTensors __metadata__ block")?;
            header_map.insert("__metadata__".to_string(), meta_json);
        }

        let mut current_offset = 0usize;
        for (tensor, source) in self.tensors.iter().zip(&self.tensor_sources) {
            let payload_len = match source {
                Some(source) => usize::try_from(source.byte_len)
                    .map_err(|_| anyhow::anyhow!("SafeTensors payload exceeds addressable size"))?,
                None => tensor.data.len(),
            };
            if source.is_some() {
                validate_source_payload(tensor, payload_len)?;
            }
            let end = current_offset
                .checked_add(payload_len)
                .context("SafeTensors payload offsets overflow")?;
            let entry = SafeTensorsEntryHeader {
                dtype: tensor.dtype.clone(),
                shape: tensor.shape.clone(),
                data_offsets: [current_offset, end],
            };
            let value = serde_json::to_value(&entry).with_context(|| {
                format!("Failed to serialize header entry for '{}'", tensor.name)
            })?;
            header_map.insert(tensor.name.clone(), value);
            current_offset = end;
        }

        let header = serde_json::to_vec(&header_map)
            .context("Failed to serialize SafeTensors header map to JSON")?;
        Ok((header, current_offset))
    }
}

fn validate_source_payload(tensor: &SafeTensorsTensor, payload_len: usize) -> Result<()> {
    let dtype: safetensors::Dtype = serde_json::from_value(serde_json::Value::String(
        tensor.dtype.clone(),
    ))
    .with_context(|| {
        format!(
            "Unsupported SafeTensors dtype '{}' for tensor '{}'",
            tensor.dtype, tensor.name
        )
    })?;
    let element_count = tensor
        .shape
        .iter()
        .try_fold(1usize, |count, &dimension| count.checked_mul(dimension))
        .with_context(|| format!("SafeTensors tensor '{}' shape overflows", tensor.name))?;
    let expected_len = element_count
        .checked_mul(dtype.size())
        .with_context(|| format!("SafeTensors tensor '{}' byte size overflows", tensor.name))?;
    if payload_len != expected_len {
        bail!(
            "SafeTensors tensor '{}' source range has {} bytes, but dtype {} and shape {:?} require {} bytes",
            tensor.name,
            payload_len,
            tensor.dtype,
            tensor.shape,
            expected_len
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::SafetensorsScanner;

    #[test]
    fn test_safetensors_writer_roundtrip_with_scanner() {
        let mut writer = SafeTensorsWriter::new();
        writer.add_metadata("format", "thme-export");
        writer.add_metadata("version", "0.1.8");

        // F32 weight: 2x2 matrix
        let f32_weights: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let mut f32_bytes = Vec::new();
        for &w in &f32_weights {
            f32_bytes.extend_from_slice(&w.to_le_bytes());
        }
        writer.add_tensor("model.embed.weight", "F32", vec![2, 2], f32_bytes);

        // F16 weight: 1x4 vector (using raw u16 representation)
        let f16_raw: Vec<u16> = vec![0x3C00, 0x4000, 0x4200, 0x4400]; // 1.0, 2.0, 3.0, 4.0 in IEEE 754 half
        let mut f16_bytes = Vec::new();
        for &w in &f16_raw {
            f16_bytes.extend_from_slice(&w.to_le_bytes());
        }
        writer.add_tensor("layers.0.mlp.weight", "F16", vec![1, 4], f16_bytes);

        let binary = writer.write_to_bytes().expect("Serialize to bytes");
        assert!(binary.len() > 8);

        // Verify 8-byte header length
        let header_len = u64::from_le_bytes(binary[0..8].try_into().unwrap()) as usize;
        assert!(binary.len() >= 8 + header_len);

        // Parse with SafetensorsScanner
        let parsed_tensors = SafetensorsScanner::parse_metadata_from_slice(&binary)
            .expect("Parse serialized SafeTensors");

        assert_eq!(parsed_tensors.len(), 2);
        let embed = parsed_tensors
            .iter()
            .find(|t| t.name == "model.embed.weight")
            .expect("Find embed");
        assert_eq!(embed.shape, vec![2, 2]);
        assert_eq!(embed.original_dtype, "F32");
        assert_eq!(embed.byte_size, 16);

        let mlp = parsed_tensors
            .iter()
            .find(|t| t.name == "layers.0.mlp.weight")
            .expect("Find mlp");
        assert_eq!(mlp.shape, vec![1, 4]);
        assert_eq!(mlp.original_dtype, "F16");
        assert_eq!(mlp.byte_size, 8);
    }

    #[test]
    fn test_safetensors_writer_empty_error() {
        let writer = SafeTensorsWriter::new();
        assert!(writer.write_to_bytes().is_err());
    }

    #[test]
    fn test_safetensors_writer_file_io() {
        let temp_dir = tempfile::tempdir().expect("Create temp dir");
        let file_path = temp_dir.path().join("model.safetensors");

        let mut writer = SafeTensorsWriter::new();
        writer.add_tensor("test_tensor", "F32", vec![1], vec![0, 0, 128, 63]); // 1.0f32
        let written_bytes = writer.write_to_file(&file_path).expect("Write file");
        assert!(written_bytes > 8);
        assert!(file_path.exists());

        let disk_tensors =
            SafetensorsScanner::parse_metadata_from_file(&file_path).expect("Parse from disk file");
        assert_eq!(disk_tensors.len(), 1);
        assert_eq!(disk_tensors[0].name, "test_tensor");
    }
}
