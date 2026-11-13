// SPDX-License-Identifier: AGPL-3.0-only
//! End-to-end source-backed exporter checks for GGUF and SafeTensors.

use std::io::{Read, Seek, SeekFrom};

use thme::exporter::gguf_writer::GgufWriter;
use thme::exporter::safetensors_writer::SafeTensorsWriter;
use thme::exporter::{
    compute_file_sha256, ExportConfig, ExportFormat, ExportQuantType, ModelExporter,
};
use thme::ir::{ModelArchitecture, ModelIR, TensorIR};
use thme::parser::gguf::{GgmlType, GgufParser, GgufValue};
use thme::parser::SafetensorsScanner;

fn q8_source_model(path: &std::path::Path) -> (ModelIR, Vec<u8>) {
    let mut payload = vec![0u8; 34];
    payload[..2].copy_from_slice(&0x3c00u16.to_le_bytes());
    for (index, value) in payload[2..].iter_mut().enumerate() {
        *value = (index as i8 - 16) as u8;
    }

    let mut writer = GgufWriter::new();
    writer
        .add_string("general.architecture", "llama")
        .add_u32("llama.block_count", 1)
        .add_u32("llama.embedding_length", 32)
        .add_u32("llama.attention.head_count", 1)
        .add_u32("llama.attention.head_count_kv", 1)
        .add_u32("llama.context_length", 128)
        .add_tensor(
            "blk.0.attn_q.weight",
            vec![32],
            GgmlType::Q8_0,
            payload.clone(),
        );
    writer.write_to_file(path).expect("write source GGUF");

    (
        GgufParser::parse_from_file(path).expect("parse source GGUF"),
        payload,
    )
}

fn f16_source_model(path: &std::path::Path) -> (ModelIR, Vec<u8>) {
    let payload = [0x00, 0x3c, 0x00, 0xc0, 0x00, 0x42, 0x00, 0x34].to_vec();
    let mut writer = SafeTensorsWriter::new();
    writer.add_tensor("weights", "F16", vec![4], payload.clone());
    writer
        .write_to_file(path)
        .expect("write source SafeTensors");

    let tensors =
        SafetensorsScanner::parse_metadata_from_file(path).expect("parse source SafeTensors");
    let model = ModelIR::new(
        "source".into(),
        ModelArchitecture::Llama,
        1,
        4,
        1,
        1,
        4,
        128,
        tensors,
    );
    (model, payload)
}

fn read_source_bytes(source: &thme::ir::tensor::TensorSource) -> Vec<u8> {
    let mut file = std::fs::File::open(&source.path).expect("open source tensor file");
    file.seek(SeekFrom::Start(source.byte_offset))
        .expect("seek to tensor payload");
    let mut bytes = vec![0; source.byte_len as usize];
    file.read_exact(&mut bytes).expect("read tensor payload");
    bytes
}

#[test]
fn gguf_q8_export_preserves_payload_metadata_and_checksum() {
    let dir = tempfile::tempdir().expect("temp dir");
    let (model, payload) = q8_source_model(&dir.path().join("source.gguf"));
    let output = dir.path().join("export.gguf");
    let config = ExportConfig {
        format: ExportFormat::GGUFv3,
        quant_type: ExportQuantType::Q8_0,
        output_path: output.clone(),
        compute_checksum: true,
    };

    let report = ModelExporter::export(&model, &config).expect("export source-backed GGUF");
    assert_eq!(report.total_tensors, 1);
    let actual_checksum = compute_file_sha256(&output).expect("hash exported file");
    assert_eq!(
        report.sha256_checksum.as_deref(),
        Some(actual_checksum.as_str())
    );

    let exported = GgufParser::parse_from_file(&output).expect("parse exported GGUF");
    assert_eq!(exported.architecture, ModelArchitecture::Llama);
    assert_eq!(exported.num_layers, 1);
    assert_eq!(exported.tensors[0].original_dtype, "Q8_0");
    assert_eq!(
        read_source_bytes(exported.tensors[0].source.as_ref().unwrap()),
        payload
    );
}

#[test]
fn gguf_export_preserves_full_tokenizer_metadata_and_normalizes_export_fields() {
    let dir = tempfile::tempdir().expect("temp dir");
    let source_path = dir.path().join("source.gguf");
    let tokens = (0..1100)
        .map(|index| GgufValue::String(format!("token_{index}")))
        .collect::<Vec<_>>();
    let scores = (0..1100)
        .map(|index| GgufValue::Float32(index as f32 / 1000.0))
        .collect::<Vec<_>>();
    let token_types = (0..1100)
        .map(|index| GgufValue::Uint32((index % 4) as u32))
        .collect::<Vec<_>>();
    let source_values = [
        ("tokenizer.ggml.tokens", GgufValue::Array(tokens.clone())),
        ("tokenizer.ggml.scores", GgufValue::Array(scores.clone())),
        (
            "tokenizer.ggml.token_type",
            GgufValue::Array(token_types.clone()),
        ),
    ];

    let mut writer = GgufWriter::new();
    writer
        .add_string("general.architecture", "llama")
        .add_string("general.name", "stale-name")
        .add_u32("general.file_type", 9)
        .add_string("tokenizer.ggml.model", "gpt2");
    for (key, value) in source_values {
        writer.add_metadata(key, value);
    }
    writer
        .add_u32("llama.block_count", 1)
        .add_u32("llama.embedding_length", 32)
        .add_u32("llama.attention.head_count", 1)
        .add_u32("llama.attention.head_count_kv", 1)
        .add_u32("llama.context_length", 128)
        .add_tensor(
            "blk.0.attn_q.weight",
            vec![32],
            GgmlType::F32,
            (0..32)
                .flat_map(|index| ((index as f32 + 1.0) / 32.0).to_le_bytes())
                .collect(),
        );
    writer
        .write_to_file(&source_path)
        .expect("write source GGUF");

    let model = GgufParser::parse_from_file(&source_path).expect("parse source GGUF");
    assert_eq!(
        model.source_gguf_metadata.get("tokenizer.ggml.model"),
        Some(&thme::ir::model::ModelMetadataValue::String(
            "gpt2".to_string()
        ))
    );
    assert!(matches!(
        model.source_gguf_metadata.get("tokenizer.ggml.tokens"),
        Some(thme::ir::model::ModelMetadataValue::Array(values)) if values.len() == 1100
    ));

    let output = dir.path().join("export.gguf");
    ModelExporter::export(
        &model,
        &ExportConfig {
            format: ExportFormat::GGUFv3,
            quant_type: ExportQuantType::Q8_0,
            output_path: output.clone(),
            compute_checksum: false,
        },
    )
    .expect("export with source tokenizer metadata");
    let exported = GgufParser::parse_from_file(&output).expect("parse exported GGUF");

    assert_eq!(
        exported.source_gguf_metadata.get("tokenizer.ggml.model"),
        Some(&thme::ir::model::ModelMetadataValue::String(
            "gpt2".to_string()
        ))
    );
    assert_eq!(
        exported.source_gguf_metadata.get("tokenizer.ggml.tokens"),
        Some(&thme::ir::model::ModelMetadataValue::Array(
            tokens.into_iter().map(Into::into).collect()
        ))
    );
    assert_eq!(
        exported.source_gguf_metadata.get("tokenizer.ggml.scores"),
        Some(&thme::ir::model::ModelMetadataValue::Array(
            scores.into_iter().map(Into::into).collect()
        ))
    );
    assert_eq!(
        exported
            .source_gguf_metadata
            .get("tokenizer.ggml.token_type"),
        Some(&thme::ir::model::ModelMetadataValue::Array(
            token_types.into_iter().map(Into::into).collect()
        ))
    );
    assert_eq!(
        exported.source_gguf_metadata.get("general.name"),
        Some(&thme::ir::model::ModelMetadataValue::String(
            "source".to_string()
        ))
    );
    assert_eq!(
        exported.source_gguf_metadata.get("general.file_type"),
        Some(&thme::ir::model::ModelMetadataValue::Uint32(7))
    );
}

#[test]
fn safetensors_fp16_export_preserves_payload_and_roundtrips() {
    let dir = tempfile::tempdir().expect("temp dir");
    let (model, payload) = f16_source_model(&dir.path().join("source.safetensors"));
    let output = dir.path().join("export.safetensors");
    let config = ExportConfig {
        format: ExportFormat::SafeTensors,
        quant_type: ExportQuantType::FP16,
        output_path: output.clone(),
        compute_checksum: true,
    };

    let report = ModelExporter::export(&model, &config).expect("export source-backed SafeTensors");
    assert_eq!(report.total_tensors, 1);
    assert!(report.sha256_checksum.is_some());

    let tensors =
        SafetensorsScanner::parse_metadata_from_file(&output).expect("parse exported SafeTensors");
    assert_eq!(tensors.len(), 1);
    assert_eq!(tensors[0].shape, vec![4]);
    assert_eq!(tensors[0].original_dtype, "F16");
    assert_eq!(
        read_source_bytes(tensors[0].source.as_ref().unwrap()),
        payload
    );
}

#[test]
fn unsupported_quantization_and_missing_sources_fail_closed() {
    let dir = tempfile::tempdir().expect("temp dir");
    let (model, _) = f16_source_model(&dir.path().join("source.safetensors"));
    let output = dir.path().join("unsupported.safetensors");
    let config = ExportConfig {
        format: ExportFormat::SafeTensors,
        quant_type: ExportQuantType::BitNetB158,
        output_path: output.clone(),
        compute_checksum: false,
    };
    let error = ModelExporter::export(&model, &config)
        .unwrap_err()
        .to_string();
    assert!(error.contains("same-dtype FP16"));
    assert!(!output.exists());

    let source_less_model = ModelIR::new(
        "metadata-only".into(),
        ModelArchitecture::Llama,
        1,
        4,
        1,
        1,
        4,
        128,
        vec![TensorIR::new("weights".into(), vec![4], "F16".into(), 8)],
    );
    let missing_output = dir.path().join("missing.safetensors");
    let config = ExportConfig {
        format: ExportFormat::SafeTensors,
        quant_type: ExportQuantType::FP16,
        output_path: missing_output.clone(),
        compute_checksum: false,
    };
    let error = ModelExporter::export(&source_less_model, &config)
        .unwrap_err()
        .to_string();
    assert!(error.contains("no file-backed payload"));
    assert!(!missing_output.exists());
}
