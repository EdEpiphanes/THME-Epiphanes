//! Direct Pure-Rust SafeTensors Binary Serializer (Milestone v0.1.8.5)
//!
//! Serializes tensor arrays and model metadata into the standard SafeTensors
//! format: an 8-byte little-endian length prefix, a UTF-8 JSON header describing
//! shapes, dtypes, and data byte offsets, followed by contiguous raw tensor bytes.

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::Write;
use std::path::Path;

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
}

impl SafeTensorsWriter {
    /// Creates a new empty SafeTensors writer
    pub fn new() -> Self {
        Self {
            metadata: BTreeMap::new(),
            tensors: Vec::new(),
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
        self
    }

    /// Number of tensors queued for serialization
    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }

    /// Total bytes of raw tensor payloads
    pub fn payload_byte_size(&self) -> usize {
        self.tensors.iter().map(|t| t.data.len()).sum()
    }

    /// Serializes all registered metadata and tensor buffers into standard SafeTensors binary bytes
    pub fn write_to_bytes(&self) -> Result<Vec<u8>> {
        if self.tensors.is_empty() && self.metadata.is_empty() {
            bail!("Cannot write empty SafeTensors: no tensors or metadata registered");
        }

        let mut header_map = serde_json::Map::new();

        // 1. Add __metadata__ block if present
        if !self.metadata.is_empty() {
            let meta_json = serde_json::to_value(&self.metadata)
                .context("Failed to serialize SafeTensors __metadata__ block")?;
            header_map.insert("__metadata__".to_string(), meta_json);
        }

        // 2. Compute data offsets and populate tensor header entries
        let mut current_offset: usize = 0;
        for tensor in &self.tensors {
            let start = current_offset;
            let end = start + tensor.data.len();
            current_offset = end;

            let entry = SafeTensorsEntryHeader {
                dtype: tensor.dtype.clone(),
                shape: tensor.shape.clone(),
                data_offsets: [start, end],
            };

            let entry_val = serde_json::to_value(&entry).with_context(|| {
                format!("Failed to serialize header entry for '{}'", tensor.name)
            })?;
            header_map.insert(tensor.name.clone(), entry_val);
        }

        // 3. Serialize JSON header
        let header_str = serde_json::to_string(&header_map)
            .context("Failed to serialize SafeTensors header map to JSON")?;
        let header_bytes = header_str.as_bytes();
        let header_len = header_bytes.len() as u64;

        // 4. Construct contiguous binary payload: [8-byte len][header_bytes][tensor_data]
        let total_size = 8 + header_bytes.len() + current_offset;
        let mut out = Vec::with_capacity(total_size);

        // 8-byte little-endian header length
        out.extend_from_slice(&header_len.to_le_bytes());
        // JSON header
        out.extend_from_slice(header_bytes);
        // Tensor data payloads
        for tensor in &self.tensors {
            out.extend_from_slice(&tensor.data);
        }

        Ok(out)
    }

    /// Writes the serialized SafeTensors container directly to disk
    pub fn write_to_file<P: AsRef<Path>>(&self, path: P) -> Result<usize> {
        let bytes = self.write_to_bytes()?;
        let path_ref = path.as_ref();
        if let Some(parent) = path_ref.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("Failed to create directory: {:?}", parent))?;
        }

        let mut file = File::create(path_ref)
            .with_context(|| format!("Failed to create SafeTensors file: {:?}", path_ref))?;
        file.write_all(&bytes)
            .with_context(|| format!("Failed to write SafeTensors bytes to {:?}", path_ref))?;

        Ok(bytes.len())
    }
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
