use crate::ir::TensorFormat;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerAllocation {
    pub layer_index: usize,
    pub assigned_format: TensorFormat,
    pub original_mb: f64,
    pub optimized_mb: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AllocationPlan {
    pub model_name: String,
    pub original_size_gb: f64,
    pub planned_size_gb: f64,
    pub compression_ratio: f64,
    pub target_vram_gb: Option<f64>,
    pub estimated_quality_retention_pct: f64,
    pub layer_allocations: Vec<LayerAllocation>,
    pub tensor_allocations: HashMap<String, TensorFormat>,
}

impl AllocationPlan {
    pub fn save_to_file<P: AsRef<std::path::Path>>(&self, path: P) -> anyhow::Result<()> {
        let file = std::fs::File::create(path)?;
        serde_json::to_writer_pretty(file, self)?;
        Ok(())
    }

    pub fn load_from_file<P: AsRef<std::path::Path>>(path: P) -> anyhow::Result<Self> {
        let file = std::fs::File::open(path)?;
        let plan = serde_json::from_reader(file)?;
        Ok(plan)
    }
}
