use anyhow::{Context, Result};
use memmap2::Mmap;
use safetensors::SafeTensors;
use std::collections::HashSet;
use std::fs::File;
use std::path::{Path, PathBuf};

use crate::ir::tensor::TensorSource;
use crate::ir::TensorIR;

pub struct SafetensorsScanner;

impl SafetensorsScanner {
    /// Discover all `.safetensors` files in a given model directory
    pub fn find_safetensors_files<P: AsRef<Path>>(dir: P) -> Result<Vec<PathBuf>> {
        let dir_path = dir.as_ref();
        if !dir_path.is_dir() {
            if dir_path.is_file() && dir_path.extension().is_some_and(|ext| ext == "safetensors") {
                return Ok(vec![dir_path.to_path_buf()]);
            }
            anyhow::bail!(
                "Path is not a valid directory or safetensors file: {:?}",
                dir_path
            );
        }

        // Check if sharded index exists
        let index_path = dir_path.join("model.safetensors.index.json");
        if index_path.exists() {
            let file = File::open(&index_path)?;
            let index: serde_json::Value = serde_json::from_reader(file)?;
            if let Some(weight_map) = index.get("weight_map").and_then(|w| w.as_object()) {
                let mut set = HashSet::new();
                for (_, filename) in weight_map {
                    if let Some(name_str) = filename.as_str() {
                        set.insert(dir_path.join(name_str));
                    }
                }
                let mut files: Vec<PathBuf> = set.into_iter().collect();
                files.sort();
                if !files.is_empty() {
                    return Ok(files);
                }
            }
        }

        // Otherwise scan directory for *.safetensors
        let mut files = Vec::new();
        for entry in std::fs::read_dir(dir_path)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_file() && path.extension().is_some_and(|ext| ext == "safetensors") {
                files.push(path);
            }
        }
        files.sort();
        Ok(files)
    }

    /// Read tensor metadata via zero-copy memory mapping without loading tensor buffers into RAM
    /// Parse SafeTensors metadata from raw in-memory bytes with bounds validation
    pub fn parse_metadata_from_slice(data: &[u8]) -> Result<Vec<TensorIR>> {
        if data.len() < 8 {
            anyhow::bail!(
                "Buffer too short for SafeTensors: {} bytes (minimum 8)",
                data.len()
            );
        }
        let header_len = u64::from_le_bytes(data[0..8].try_into().unwrap()) as usize;
        if header_len > data.len().saturating_sub(8) {
            anyhow::bail!(
                "Malformed SafeTensors header length {} exceeds buffer size {}",
                header_len,
                data.len().saturating_sub(8)
            );
        }

        let safetensors = SafeTensors::deserialize(data)
            .with_context(|| "Failed to parse SafeTensors header from memory slice")?;

        let mut tensors = Vec::new();
        for (name, view) in safetensors.tensors() {
            let shape = view.shape().to_vec();
            let dtype = format!("{:?}", view.dtype());
            let byte_size = view.data().len();

            let mut tensor_ir = TensorIR::new(name, shape, dtype.clone(), byte_size);
            tensor_ir.weight_profile = Some(crate::profiling::WeightAnalyzer::profile_tensor(
                view.data(),
                &dtype,
            ));
            tensors.push(tensor_ir);
        }

        Ok(tensors)
    }

    pub fn parse_metadata_from_file<P: AsRef<Path>>(path: P) -> Result<Vec<TensorIR>> {
        let file = File::open(path.as_ref())
            .with_context(|| format!("Failed to open safetensors file: {:?}", path.as_ref()))?;

        // SAFETY: File descriptor is open and unmodified during the lifetime of this read-only memory map.
        let mmap = unsafe { Mmap::map(&file)? };
        let source_path = path.as_ref().canonicalize().with_context(|| {
            format!(
                "Failed to resolve safetensors source path: {:?}",
                path.as_ref()
            )
        })?;
        let safetensors = SafeTensors::deserialize(&mmap).with_context(|| {
            format!(
                "Failed to parse safetensors header from {:?}",
                path.as_ref()
            )
        })?;

        let mut tensors = Vec::new();
        for (name, view) in safetensors.tensors() {
            let shape = view.shape().to_vec();
            let dtype = format!("{:?}", view.dtype());
            let byte_size = view.data().len();

            let mut tensor_ir = TensorIR::new(name, shape, dtype.clone(), byte_size);
            let source_offset = (view.data().as_ptr() as usize)
                .checked_sub(mmap.as_ptr() as usize)
                .with_context(|| {
                    format!("Invalid payload offset for tensor '{}'", tensor_ir.name)
                })?;
            let source_end = source_offset
                .checked_add(byte_size)
                .filter(|&end| end <= mmap.len())
                .with_context(|| {
                    format!(
                        "Payload range exceeds source file for tensor '{}'",
                        tensor_ir.name
                    )
                })?;
            tensor_ir.source = Some(TensorSource {
                path: source_path.clone(),
                byte_offset: u64::try_from(source_offset)
                    .context("SafeTensors source offset exceeds u64")?,
                byte_len: u64::try_from(source_end - source_offset)
                    .context("SafeTensors source length exceeds u64")?,
            });
            tensor_ir.weight_profile = Some(crate::profiling::WeightAnalyzer::profile_tensor(
                view.data(),
                &dtype,
            ));
            tensors.push(tensor_ir);
        }

        Ok(tensors)
    }

    /// Scan entire model directory (supporting single file or multi-file sharding)
    pub fn scan_model_dir<P: AsRef<Path>>(dir: P) -> Result<Vec<TensorIR>> {
        let files = Self::find_safetensors_files(dir)?;
        if files.is_empty() {
            anyhow::bail!("No safetensors weights found in model directory.");
        }

        let mut all_tensors = Vec::new();
        for file in files {
            let mut file_tensors = Self::parse_metadata_from_file(file)?;
            all_tensors.append(&mut file_tensors);
        }

        Ok(all_tensors)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exporter::safetensors_writer::SafeTensorsWriter;
    use std::io::{Read, Seek, SeekFrom};

    #[test]
    fn file_parse_retains_tensor_payload_range() {
        let payload = vec![1u8, 4, 9, 16, 25];
        let mut writer = SafeTensorsWriter::new();
        writer.add_tensor("weights", "U8", vec![5], payload.clone());

        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("source.safetensors");
        writer.write_to_file(&path).expect("write SafeTensors");

        let tensors =
            SafetensorsScanner::parse_metadata_from_file(&path).expect("parse SafeTensors file");
        let source = tensors[0].source.as_ref().expect("source range");
        assert_eq!(source.path, path.canonicalize().expect("canonical path"));
        assert_eq!(source.byte_len, payload.len() as u64);

        let mut file = File::open(&source.path).expect("open source");
        file.seek(SeekFrom::Start(source.byte_offset))
            .expect("seek to payload");
        let mut actual = vec![0; payload.len()];
        file.read_exact(&mut actual).expect("read payload");
        assert_eq!(actual, payload);
    }
}
