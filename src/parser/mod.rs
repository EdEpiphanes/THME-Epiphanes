pub mod config;
pub mod gguf;
pub mod safetensors;

use anyhow::Result;
use std::path::Path;

use crate::ir::{ModelArchitecture, ModelIR};
use config::HfConfig;
pub use gguf::GgufParser;
pub use safetensors::SafetensorsScanner;

pub struct ModelImporter;

impl ModelImporter {
    /// Universal importer: auto-detects GGUF file, SafeTensors file, or Hugging Face model directory
    pub fn import<P: AsRef<Path>>(path: P) -> Result<ModelIR> {
        let p = path.as_ref();
        if p.is_file() {
            let ext = p
                .extension()
                .and_then(|e| e.to_str())
                .unwrap_or("")
                .to_lowercase();
            if ext == "gguf" {
                return GgufParser::parse_from_file(p);
            }
            if ext == crate::epiph::EXTENSION {
                return crate::epiph::read(p).map(|(model, _)| model);
            }
        }
        Self::import_from_dir(p)
    }

    pub fn import_from_dir<P: AsRef<Path>>(dir: P) -> Result<ModelIR> {
        let dir_path = dir.as_ref();
        let model_name = dir_path
            .file_stem()
            .and_then(|n| n.to_str())
            .unwrap_or("unnamed_model")
            .to_string();

        // Check if user pointed to a GGUF file directly
        if dir_path.is_file() {
            let ext = dir_path
                .extension()
                .and_then(|e| e.to_str())
                .unwrap_or("")
                .to_lowercase();
            if ext == "gguf" {
                return GgufParser::parse_from_file(dir_path);
            }
        }

        let hf_config = HfConfig::load_from_dir(dir_path).ok();
        let tensors = SafetensorsScanner::scan_model_dir(dir_path)?;

        let arch_str = hf_config
            .as_ref()
            .map(HfConfig::primary_architecture)
            .unwrap_or_else(|| {
                // Infer from tensor naming
                if tensors
                    .iter()
                    .any(|t| t.name.contains("llama") || t.name.contains("q_proj"))
                {
                    "LlamaForCausalLM".to_string()
                } else if tensors.iter().any(|t| t.name.contains("mistral")) {
                    "MistralForCausalLM".to_string()
                } else if tensors.iter().any(|t| t.name.contains("qwen")) {
                    "Qwen2ForCausalLM".to_string()
                } else {
                    "UnknownTransformer".to_string()
                }
            });
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

        let num_layers = hf_config
            .as_ref()
            .and_then(|c| c.num_hidden_layers)
            .unwrap_or(max_layer_from_tensors.max(1));

        let hidden_size = hf_config
            .as_ref()
            .and_then(|c| c.hidden_size)
            .unwrap_or_else(|| {
                tensors
                    .iter()
                    .find(|t| t.shape.len() == 2)
                    .map(|t| t.shape[1])
                    .unwrap_or(4096)
            });

        let num_attention_heads = hf_config
            .as_ref()
            .and_then(|c| c.num_attention_heads)
            .unwrap_or(32);
        let num_key_value_heads = hf_config
            .as_ref()
            .and_then(|c| c.num_key_value_heads)
            .unwrap_or(num_attention_heads);
        let vocab_size = hf_config
            .as_ref()
            .and_then(|c| c.vocab_size)
            .unwrap_or_else(|| {
                tensors
                    .iter()
                    .find(|t| t.name.contains("embed_tokens") || t.name.contains("lm_head"))
                    .and_then(|t| t.shape.first().copied())
                    .unwrap_or(32000)
            });
        let max_position_embeddings = hf_config
            .as_ref()
            .and_then(|c| c.max_position_embeddings)
            .unwrap_or(4096);

        let mut model_ir = ModelIR::new(
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
        model_ir.norm_eps = hf_config
            .as_ref()
            .and_then(|c| c.rms_norm_eps)
            .map(|v| v as f32);
        model_ir.rope_theta = hf_config
            .as_ref()
            .and_then(|c| c.rope_theta)
            .map(|v| v as f32);

        Ok(model_ir)
    }
}
