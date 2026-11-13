use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

/// Representation precision formats for weights
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[allow(non_camel_case_types)]
pub enum TensorFormat {
    FP32,
    FP16,
    BF16,
    INT8,
    Q8_0,
    Q4_0,
    Q4_K,
    Q5_K,
    Q6_K,
    Q2_K,
    Ternary, // ~1.58 bits (log2(3))
    Bit1_5,  // 1.5-bit experimental
    Custom(String),
}

impl TensorFormat {
    pub fn bits_per_weight(&self) -> f64 {
        match self {
            Self::FP32 => 32.0,
            Self::FP16 | Self::BF16 => 16.0,
            Self::INT8 | Self::Q8_0 | Self::Custom(_) => 8.0,
            Self::Q4_0 | Self::Q4_K => 4.5, // 4-bit plus scale overhead
            Self::Q5_K => 5.5,              // 176 bytes per 256 values
            Self::Q6_K => 6.5625,           // 210 bytes per 256 values
            Self::Q2_K => 2.56,
            Self::Ternary => 1.585, // log2(3)
            Self::Bit1_5 => 1.5,
        }
    }

    pub fn display_name(&self) -> &str {
        match self {
            Self::FP32 => "FP32",
            Self::FP16 => "FP16",
            Self::BF16 => "BF16",
            Self::INT8 => "INT8",
            Self::Q8_0 => "Q8_0",
            Self::Q4_0 => "Q4_0",
            Self::Q4_K => "Q4_K",
            Self::Q5_K => "Q5_K",
            Self::Q6_K => "Q6_K",
            Self::Q2_K => "Q2_K",
            Self::Ternary => "Ternary (1.58b)",
            Self::Bit1_5 => "1.5-bit",
            Self::Custom(name) => name.as_str(),
        }
    }
}

/// Architectural role of the tensor in modern Transformer architectures
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TensorRole {
    AttentionQ,
    /// Fused query/key/value projection (GPT-NeoX `query_key_value`).
    AttentionQkv,
    AttentionK,
    AttentionV,
    AttentionOut,
    MlpGate,
    MlpUp,
    MlpDown,
    Embedding,
    LayerNorm,
    FinalNorm,
    OutputHead,
    Unknown,
}

impl TensorRole {
    pub fn detect(name: &str) -> Self {
        let lower = name.to_lowercase();
        if lower.contains("lm_head")
            || lower.contains("output.weight")
            || lower.contains("embed_out")
        {
            Self::OutputHead
        } else if lower.contains("embed") || lower.contains("wte") {
            Self::Embedding
        } else if lower.contains("query_key_value") {
            Self::AttentionQkv
        } else if lower.contains("q_proj") || lower.contains("query") {
            Self::AttentionQ
        } else if lower.contains("k_proj") || lower.contains("key") {
            Self::AttentionK
        } else if lower.contains("v_proj") || lower.contains("value") {
            Self::AttentionV
        } else if lower.contains("o_proj")
            || lower.contains("out_proj")
            || lower.contains("dense") && lower.contains("attn")
            || lower.contains("attention.dense")
        {
            Self::AttentionOut
        } else if lower.contains("dense_h_to_4h") {
            Self::MlpUp
        } else if lower.contains("dense_4h_to_h") {
            Self::MlpDown
        } else if lower.contains("gate_proj") || lower.contains("w1") {
            Self::MlpGate
        } else if lower.contains("up_proj") || lower.contains("w3") {
            Self::MlpUp
        } else if lower.contains("down_proj") || lower.contains("w2") {
            Self::MlpDown
        } else if lower.contains("norm") || lower.contains("ln_") {
            if lower.contains("model.norm") || lower.contains("final_layer_norm") {
                Self::FinalNorm
            } else {
                Self::LayerNorm
            }
        } else {
            Self::Unknown
        }
    }

    /// Inherent architectural sensitivity multiplier
    pub fn base_sensitivity_weight(&self) -> f64 {
        match self {
            Self::Embedding | Self::OutputHead => 2.5, // Embeddings & Heads directly dictate logit output
            Self::FinalNorm | Self::LayerNorm => 3.0, // Normalization parameters should stay high precision
            Self::AttentionQ | Self::AttentionQkv | Self::AttentionK => 1.8, // Q & K preserve attention routing topology
            Self::AttentionV => 1.4,
            Self::AttentionOut => 1.2,
            Self::MlpGate | Self::MlpUp => 1.1,
            Self::MlpDown | Self::Unknown => 1.0, // Typically most robust to compression
        }
    }
}

/// Intermediate Representation for an individual tensor in the model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorIR {
    pub name: String,
    pub role: TensorRole,
    pub shape: Vec<usize>,
    pub original_dtype: String,
    pub numel: usize,
    pub byte_size: usize,
    pub layer_index: Option<usize>,
    pub sensitivity_score: f64,
    pub candidate_formats: Vec<TensorFormat>,
    pub selected_format: TensorFormat,
    pub metadata: HashMap<String, String>,
    pub weight_profile: Option<WeightProfile>,
    /// Runtime-only location of the original payload for zero-copy export.
    #[serde(skip)]
    pub source: Option<TensorSource>,
}

/// Byte range of a tensor payload in its original model file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorSource {
    pub path: PathBuf,
    pub byte_offset: u64,
    pub byte_len: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightProfile {
    pub min_val: f32,
    pub max_val: f32,
    pub mean_val: f32,
    pub std_val: f32,
    pub sparsity: f32,
    pub outlier_ratio: f32,
    pub histogram: [f32; 32],
    pub sample_weights: Vec<f32>,
}

impl Default for WeightProfile {
    fn default() -> Self {
        let mut hist = [0.0; 32];
        for (i, h) in hist.iter_mut().enumerate() {
            let x = (i as f32 - 15.5) / 5.0;
            *h = (-0.5 * x * x).exp();
        }
        let sum: f32 = hist.iter().sum();
        if sum > 0.0 {
            for h in hist.iter_mut() {
                *h /= sum;
            }
        }

        Self {
            min_val: -0.15,
            max_val: 0.15,
            mean_val: 0.0,
            std_val: 0.025,
            sparsity: 0.0,
            outlier_ratio: 0.003,
            histogram: hist,
            sample_weights: Vec::new(),
        }
    }
}

impl TensorIR {
    pub fn new(name: String, shape: Vec<usize>, original_dtype: String, byte_size: usize) -> Self {
        let numel = shape.iter().product();
        let role = TensorRole::detect(&name);
        let layer_index = extract_layer_index(&name);
        let base_sensitivity = role.base_sensitivity_weight();

        Self {
            name,
            role,
            shape,
            original_dtype,
            numel,
            byte_size,
            layer_index,
            sensitivity_score: base_sensitivity,
            candidate_formats: vec![
                TensorFormat::FP16,
                TensorFormat::INT8,
                TensorFormat::Q4_0,
                TensorFormat::Ternary,
            ],
            selected_format: TensorFormat::FP16,
            metadata: HashMap::new(),
            weight_profile: None,
            source: None,
        }
    }

    /// Estimated memory bytes under selected format
    pub fn estimated_bytes(&self) -> usize {
        let bits = self.selected_format.bits_per_weight();
        ((self.numel as f64 * bits) / 8.0).ceil() as usize
    }

    /// Compression ratio achieved compared to original byte size
    pub fn compression_ratio(&self) -> f64 {
        if self.estimated_bytes() == 0 {
            1.0
        } else {
            self.byte_size as f64 / self.estimated_bytes() as f64
        }
    }
}

pub fn extract_layer_index(name: &str) -> Option<usize> {
    let parts: Vec<&str> = name.split('.').collect();
    for i in 0..parts.len() {
        if (parts[i] == "layers" || parts[i] == "layer" || parts[i] == "h" || parts[i] == "blocks")
            && i + 1 < parts.len()
        {
            if let Ok(idx) = parts[i + 1].parse::<usize>() {
                return Some(idx);
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_path_is_not_serialized() {
        let mut tensor = TensorIR::new("weights".into(), vec![1], "U8".into(), 1);
        tensor.source = Some(TensorSource {
            path: PathBuf::from("local-model.safetensors"),
            byte_offset: 16,
            byte_len: 1,
        });

        let value = serde_json::to_value(tensor).expect("serialize tensor metadata");
        assert!(value.get("source").is_none());
    }
}
