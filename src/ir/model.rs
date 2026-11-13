use super::tensor::{TensorIR, TensorRole};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};

#[derive(Debug, Clone, PartialEq)]
pub enum ModelMetadataValue {
    Uint8(u8),
    Int8(i8),
    Uint16(u16),
    Int16(i16),
    Uint32(u32),
    Int32(i32),
    Float32(f32),
    Bool(bool),
    String(String),
    Array(Vec<ModelMetadataValue>),
    Uint64(u64),
    Int64(i64),
    Float64(f64),
}

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
    /// Normalization epsilon (RMSNorm or LayerNorm) from `config.json` or GGUF metadata.
    #[serde(default)]
    pub norm_eps: Option<f32>,
    /// RoPE base frequency from `config.json` or GGUF metadata.
    #[serde(default)]
    pub rope_theta: Option<f32>,
    #[serde(skip)]
    pub source_gguf_metadata: BTreeMap<String, ModelMetadataValue>,
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
            norm_eps: None,
            rope_theta: None,
            source_gguf_metadata: BTreeMap::new(),
        }
    }

    /// Total memory bytes after currently selected tensor precisions
    pub fn total_estimated_bytes(&self) -> usize {
        self.tensors.iter().map(TensorIR::estimated_bytes).sum()
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

impl Default for ModelIR {
    fn default() -> Self {
        Self::new(
            "Llama-3-8B".to_string(),
            ModelArchitecture::Llama,
            32,
            4096,
            32,
            8,
            128256,
            8192,
            Vec::new(),
        )
    }
}
