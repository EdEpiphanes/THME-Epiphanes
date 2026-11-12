//! GGUF Model Architecture Metadata Synchronization Engine (Milestone v0.1.8.4)
//!
//! Synchronizes high-level neural architecture parameters and hyper-parameters
//! from `ModelIR` into compliant GGUF key-value metadata records, ensuring model
//! architectures are properly recognized by downstream runtimes (`llama.cpp`, Ollama).

use crate::exporter::gguf_writer::GgufWriter;
use crate::exporter::ExportQuantType;
use crate::ir::{ModelArchitecture, ModelIR};
use crate::parser::gguf::GgufValue;

/// Synchronizer engine for GGUF architecture metadata reconstitution
#[derive(Debug, Clone, Default)]
pub struct GgufMetadataSynchronizer;

impl GgufMetadataSynchronizer {
    /// Creates a new metadata synchronizer instance
    pub fn new() -> Self {
        Self
    }

    /// Resolves canonical lowercase architecture prefix used in GGUF keys
    pub fn architecture_prefix(arch: &ModelArchitecture) -> &'static str {
        match arch {
            ModelArchitecture::Llama | ModelArchitecture::Unknown(_) => "llama",
            ModelArchitecture::Mistral => "mistral",
            ModelArchitecture::Qwen => "qwen2",
            ModelArchitecture::DeepSeek => "deepseek2",
            ModelArchitecture::Gemma => "gemma",
        }
    }

    /// Resolves GGUF general.file_type enum for standard quantization types
    pub fn file_type_code(quant_type: ExportQuantType) -> u32 {
        match quant_type {
            ExportQuantType::FP16 => 1,       // ALL_F16
            ExportQuantType::Q8_0 => 7,       // MOSTLY_Q8_0
            ExportQuantType::TQ1_0 => 36,     // MOSTLY_TQ1_0
            ExportQuantType::BitNetB158 => 0, // ALL_F32 / Custom experimental
        }
    }

    /// Reconstitutes standard GGUF metadata key-value pairs from `ModelIR`
    pub fn build_metadata(
        model: &ModelIR,
        quant_type: ExportQuantType,
    ) -> Vec<(String, GgufValue)> {
        let arch = Self::architecture_prefix(&model.architecture);
        let mut meta = Vec::new();

        // 1. General model identity & format version
        meta.push((
            "general.architecture".to_string(),
            GgufValue::String(arch.to_string()),
        ));
        meta.push((
            "general.name".to_string(),
            GgufValue::String(model.name.clone()),
        ));
        meta.push((
            "general.quantization_version".to_string(),
            GgufValue::Uint32(2),
        ));
        meta.push((
            "general.file_type".to_string(),
            GgufValue::Uint32(Self::file_type_code(quant_type)),
        ));

        // 2. Core architectural dimensions
        meta.push((
            format!("{arch}.block_count"),
            GgufValue::Uint32(model.num_layers as u32),
        ));
        meta.push((
            format!("{arch}.context_length"),
            GgufValue::Uint32(model.max_position_embeddings as u32),
        ));
        meta.push((
            format!("{arch}.embedding_length"),
            GgufValue::Uint32(model.hidden_size as u32),
        ));

        // Calculate feed-forward intermediate dimension (heuristic from tensors if available)
        let intermediate_dim = model
            .tensors
            .iter()
            .find(|t| {
                t.name.contains("mlp.up_proj")
                    || t.name.contains("mlp.gate_proj")
                    || t.name.contains("feed_forward")
            })
            .and_then(|t| t.shape.first().copied())
            .unwrap_or_else(|| (model.hidden_size * 8) / 3);

        meta.push((
            format!("{arch}.feed_forward_length"),
            GgufValue::Uint32(intermediate_dim as u32),
        ));

        // 3. Multi-head attention & GQA parameters
        meta.push((
            format!("{arch}.attention.head_count"),
            GgufValue::Uint32(model.num_attention_heads as u32),
        ));
        meta.push((
            format!("{arch}.attention.head_count_kv"),
            GgufValue::Uint32(model.num_key_value_heads as u32),
        ));

        // 4. RoPE & Normalization constants
        meta.push((
            format!("{arch}.rope.freq_base"),
            GgufValue::Float32(10000.0),
        ));
        meta.push((
            format!("{arch}.attention.layer_norm_rms_epsilon"),
            GgufValue::Float32(1e-5),
        ));

        // 5. Vocabulary metadata
        if model.vocab_size > 0 {
            if !model
                .source_gguf_metadata
                .contains_key("tokenizer.ggml.model")
            {
                meta.push((
                    "tokenizer.ggml.model".to_string(),
                    GgufValue::String("llama".to_string()),
                ));
            }
            meta.push((
                format!("{arch}.vocab_size"),
                GgufValue::Uint32(model.vocab_size as u32),
            ));
        }

        meta
    }

    /// Synchronizes reconstituted metadata directly into a `GgufWriter`
    pub fn synchronize_into_writer(
        writer: &mut GgufWriter,
        model: &ModelIR,
        quant_type: ExportQuantType,
    ) {
        for (key, value) in &model.source_gguf_metadata {
            writer.add_metadata(
                key.clone(),
                crate::parser::gguf::GgufValue::from(value.clone()),
            );
        }
        let metadata = Self::build_metadata(model, quant_type);
        for (key, value) in metadata {
            writer.add_metadata(key, value);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::TensorIR;

    fn make_test_model() -> ModelIR {
        let tensors = vec![
            TensorIR::new(
                "model.embed_tokens.weight".to_string(),
                vec![32000, 2048],
                "F16".to_string(),
                32000 * 2048 * 2,
            ),
            TensorIR::new(
                "layers.0.mlp.up_proj.weight".to_string(),
                vec![5632, 2048],
                "F16".to_string(),
                5632 * 2048 * 2,
            ),
        ];

        ModelIR::new(
            "test-llama-3b".to_string(),
            ModelArchitecture::Llama,
            16,
            2048,
            16,
            4,
            32000,
            4096,
            tensors,
        )
    }

    #[test]
    fn test_architecture_prefix_mapping() {
        assert_eq!(
            GgufMetadataSynchronizer::architecture_prefix(&ModelArchitecture::Llama),
            "llama"
        );
        assert_eq!(
            GgufMetadataSynchronizer::architecture_prefix(&ModelArchitecture::Mistral),
            "mistral"
        );
        assert_eq!(
            GgufMetadataSynchronizer::architecture_prefix(&ModelArchitecture::Qwen),
            "qwen2"
        );
        assert_eq!(
            GgufMetadataSynchronizer::architecture_prefix(&ModelArchitecture::DeepSeek),
            "deepseek2"
        );
        assert_eq!(
            GgufMetadataSynchronizer::architecture_prefix(&ModelArchitecture::Gemma),
            "gemma"
        );
    }

    #[test]
    fn test_file_type_code_mapping() {
        assert_eq!(
            GgufMetadataSynchronizer::file_type_code(ExportQuantType::FP16),
            1
        );
        assert_eq!(
            GgufMetadataSynchronizer::file_type_code(ExportQuantType::Q8_0),
            7
        );
        assert_eq!(
            GgufMetadataSynchronizer::file_type_code(ExportQuantType::BitNetB158),
            0
        );
    }

    #[test]
    fn test_build_metadata_values() {
        let model = make_test_model();
        let meta = GgufMetadataSynchronizer::build_metadata(&model, ExportQuantType::Q8_0);

        let map: std::collections::HashMap<String, GgufValue> = meta.into_iter().collect();

        assert_eq!(
            map.get("general.architecture").and_then(GgufValue::as_str),
            Some("llama")
        );
        assert_eq!(
            map.get("llama.block_count").and_then(GgufValue::as_u64),
            Some(16)
        );
        assert_eq!(
            map.get("llama.embedding_length")
                .and_then(GgufValue::as_u64),
            Some(2048)
        );
        assert_eq!(
            map.get("llama.feed_forward_length")
                .and_then(GgufValue::as_u64),
            Some(5632)
        );
        assert_eq!(
            map.get("llama.attention.head_count")
                .and_then(GgufValue::as_u64),
            Some(16)
        );
        assert_eq!(
            map.get("llama.attention.head_count_kv")
                .and_then(GgufValue::as_u64),
            Some(4)
        );
        assert_eq!(
            map.get("general.file_type").and_then(GgufValue::as_u64),
            Some(7)
        );
    }

    #[test]
    fn test_synchronize_into_writer() {
        let model = make_test_model();
        let mut writer = GgufWriter::new();
        GgufMetadataSynchronizer::synchronize_into_writer(
            &mut writer,
            &model,
            ExportQuantType::BitNetB158,
        );

        let bytes = writer.write_to_bytes().expect("Write GGUF bytes");
        assert!(bytes.len() > 100);
        assert_eq!(&bytes[0..4], b"GGUF");
    }
}
