pub mod config;
pub mod safetensors;

use anyhow::Result;
use std::path::Path;

use crate::ir::{ModelArchitecture, ModelIR};
use config::HfConfig;
use safetensors::SafetensorsScanner;

pub struct ModelImporter;

impl ModelImporter {
    pub fn import_from_dir<P: AsRef<Path>>(dir: P) -> Result<ModelIR> {
        let dir_path = dir.as_ref();
        let model_name = dir_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unnamed_model")
            .to_string();

        let hf_config = HfConfig::load_from_dir(dir_path)?;
        let tensors = SafetensorsScanner::scan_model_dir(dir_path)?;

        let arch_str = hf_config.primary_architecture();
        let architecture = ModelArchitecture::from_str_name(&arch_str);

        // Infer parameters from config or tensor inventory
        let mut max_layer_from_tensors = 0;
        for t in &tensors {
            if let Some(l) = t.layer_index {
                if l >= max_layer_from_tensors {
                    max_layer_from_tensors = l + 1;
                }
            }
        }

        let num_layers = hf_config.num_hidden_layers.unwrap_or(max_layer_from_tensors);
        let hidden_size = hf_config.hidden_size.unwrap_or_else(|| {
            // Attempt to infer from an embedding or norm tensor
            tensors
                .iter()
                .find(|t| t.shape.len() == 2)
                .map(|t| t.shape[1])
                .unwrap_or(4096)
        });

        let num_attention_heads = hf_config.num_attention_heads.unwrap_or(32);
        let num_key_value_heads = hf_config.num_key_value_heads.unwrap_or(num_attention_heads);
        let vocab_size = hf_config.vocab_size.unwrap_or_else(|| {
            tensors
                .iter()
                .find(|t| t.name.contains("embed_tokens") || t.name.contains("lm_head"))
                .and_then(|t| t.shape.first().copied())
                .unwrap_or(32000)
        });
        let max_position_embeddings = hf_config.max_position_embeddings.unwrap_or(4096);

        let model_ir = ModelIR::new(
            model_name,
            architecture,
            num_layers,
            hidden_size,
            num_attention_heads,
            num_key_value_heads,
            vocab_size,
            max_position_embeddings,
            tensors,
        );

        Ok(model_ir)
    }
}
