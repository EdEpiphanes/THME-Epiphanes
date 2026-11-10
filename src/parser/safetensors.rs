use anyhow::{Context, Result};
use memmap2::Mmap;
use safetensors::SafeTensors;
use std::collections::HashSet;
use std::fs::File;
use std::path::{Path, PathBuf};

use crate::ir::TensorIR;

pub struct SafetensorsScanner;

impl SafetensorsScanner {
    /// Discover all `.safetensors` files in a given model directory
    pub fn find_safetensors_files<P: AsRef<Path>>(dir: P) -> Result<Vec<PathBuf>> {
        let dir_path = dir.as_ref();
        if !dir_path.is_dir() {
            if dir_path.is_file() && dir_path.extension().map_or(false, |ext| ext == "safetensors") {
                return Ok(vec![dir_path.to_path_buf()]);
            }
            anyhow::bail!("Path is not a valid directory or safetensors file: {:?}", dir_path);
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
            if path.is_file() && path.extension().map_or(false, |ext| ext == "safetensors") {
                files.push(path);
            }
        }
        files.sort();
        Ok(files)
    }

    /// Read tensor metadata via zero-copy memory mapping without loading tensor buffers into RAM
    pub fn parse_metadata_from_file<P: AsRef<Path>>(path: P) -> Result<Vec<TensorIR>> {
        let file = File::open(path.as_ref())
            .with_context(|| format!("Failed to open safetensors file: {:?}", path.as_ref()))?;
        
        let mmap = unsafe { Mmap::map(&file)? };
        let safetensors = SafeTensors::deserialize(&mmap)
            .with_context(|| format!("Failed to parse safetensors header from {:?}", path.as_ref()))?;

        let mut tensors = Vec::new();
        for (name, view) in safetensors.tensors() {
            let shape = view.shape().to_vec();
            let dtype = format!("{:?}", view.dtype());
            let byte_size = view.data().len();

            let tensor_ir = TensorIR::new(name, shape, dtype, byte_size);
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
