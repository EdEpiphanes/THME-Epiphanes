use super::tensor::{TensorIR, TensorRole};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ModelArchitecture {
    Llama,
    Mistral,
    Qwen,
    DeepSeek,
    Gemma,
    Unknown(String),
}

impl ModelArchitecture {
    pub fn from_str_name(arch: &str) -> Self {
        let lower = arch.to_lowercase();
        if lower.contains("llama") {
            Self::Llama
        } else if lower.contains("mistral") {
            Self::Mistral
        } else if lower.contains("qwen") {
            Self::Qwen
        } else if lower.contains("deepseek") {
            Self::DeepSeek
        } else if lower.contains("gemma") {
            Self::Gemma
        } else {
            Self::Unknown(arch.to_string())
        }
    }

    pub fn display_name(&self) -> &str {
        match self {
            Self::Llama => "Llama-family",
            Self::Mistral => "Mistral-family",
            Self::Qwen => "Qwen-family",
            Self::DeepSeek => "DeepSeek-family",
            Self::Gemma => "Gemma-family",
            Self::Unknown(s) => s.as_str(),
        }
    }
}

/// Comprehensive Intermediate Representation of an LLM
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelIR {
    pub name: String,
    pub architecture: ModelArchitecture,
    pub num_parameters: usize,
    pub original_byte_size: usize,
    pub num_layers: usize,
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub tensors: Vec<TensorIR>,
}

impl ModelIR {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        name: String,
        architecture: ModelArchitecture,
        num_layers: usize,
        hidden_size: usize,
        num_attention_heads: usize,
        num_key_value_heads: usize,
        vocab_size: usize,
        max_position_embeddings: usize,
        tensors: Vec<TensorIR>,
    ) -> Self {
        let num_parameters: usize = tensors.iter().map(|t| t.numel).sum();
        let original_byte_size: usize = tensors.iter().map(|t| t.byte_size).sum();
        let head_dim = hidden_size.checked_div(num_attention_heads).unwrap_or(128);

        Self {
            name,
            architecture,
            num_parameters,
            original_byte_size,
            num_layers,
            hidden_size,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            vocab_size,
            max_position_embeddings,
            tensors,
        }
    }

    /// Total memory bytes after currently selected tensor precisions
    pub fn total_estimated_bytes(&self) -> usize {
        self.tensors.iter().map(|t| t.estimated_bytes()).sum()
    }

    /// Global model compression ratio (original / compressed)
    pub fn overall_compression_ratio(&self) -> f64 {
        let est = self.total_estimated_bytes();
        if est == 0 {
            1.0
        } else {
            self.original_byte_size as f64 / est as f64
        }
    }

    /// Group tensors by layer index (None represents global embeddings / head / norm)
    pub fn tensors_by_layer(&self) -> HashMap<Option<usize>, Vec<&TensorIR>> {
        let mut map: HashMap<Option<usize>, Vec<&TensorIR>> = HashMap::new();
        for t in &self.tensors {
            map.entry(t.layer_index).or_default().push(t);
        }
        map
    }

    /// Group tensors by architectural role
    pub fn tensors_by_role(&self) -> HashMap<TensorRole, Vec<&TensorIR>> {
        let mut map: HashMap<TensorRole, Vec<&TensorIR>> = HashMap::new();
        for t in &self.tensors {
            map.entry(t.role).or_default().push(t);
        }
        map
    }
}
