use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct HfConfig {
    #[serde(default)]
    pub architectures: Vec<String>,
    #[serde(default)]
    pub model_type: Option<String>,
    #[serde(alias = "num_hidden_layers", alias = "n_layer", default)]
    pub num_hidden_layers: Option<usize>,
    #[serde(alias = "hidden_size", alias = "n_embd", default)]
    pub hidden_size: Option<usize>,
    #[serde(alias = "num_attention_heads", alias = "n_head", default)]
    pub num_attention_heads: Option<usize>,
    #[serde(alias = "num_key_value_heads", alias = "n_kv_heads", default)]
    pub num_key_value_heads: Option<usize>,
    #[serde(alias = "vocab_size", default)]
    pub vocab_size: Option<usize>,
    #[serde(alias = "max_position_embeddings", alias = "n_ctx", default)]
    pub max_position_embeddings: Option<usize>,
    #[serde(default)]
    pub torch_dtype: Option<String>,
}

impl HfConfig {
    pub fn load_from_dir<P: AsRef<Path>>(dir: P) -> Result<Self> {
        let dir_ref = dir.as_ref();
        let config_dir = if dir_ref.is_file() {
            dir_ref.parent().unwrap_or_else(|| Path::new("."))
        } else {
            dir_ref
        };
        let path = config_dir.join("config.json");
        if !path.exists() {
            return Ok(Self::default());
        }
        let file =
            File::open(&path).with_context(|| format!("Failed to open config file: {:?}", path))?;
        let reader = BufReader::new(file);
        let config: HfConfig = serde_json::from_reader(reader)
            .with_context(|| format!("Failed to parse config.json at: {:?}", path))?;
        Ok(config)
    }

    pub fn primary_architecture(&self) -> String {
        if let Some(arch) = self.architectures.first() {
            arch.clone()
        } else if let Some(ref m_type) = self.model_type {
            m_type.clone()
        } else {
            "Unknown".to_string()
        }
    }
}
