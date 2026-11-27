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
        llama_cpp_layout: false,

        tokenizer_json_path: None,
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
            llama_cpp_layout: false,

            tokenizer_json_path: None,
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
        llama_cpp_layout: false,

        tokenizer_json_path: None,
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
        llama_cpp_layout: false,

        tokenizer_json_path: None,
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
        llama_cpp_layout: false,

        tokenizer_json_path: None,
    };
    let error = ModelExporter::export(&source_less_model, &config)
        .unwrap_err()
        .to_string();
    assert!(error.contains("no file-backed payload"));
    assert!(!missing_output.exists());
}

fn bf16_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|&v| thme::engine::minifloat::f32_to_bf16(v).to_le_bytes())
        .collect()
}

fn f16_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|&v| thme::engine::minifloat::f32_to_f16(v).to_le_bytes())
        .collect()
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

#[test]
fn llama_cpp_layout_renames_converts_bf16_norms_permutes_qk_and_preserves_flag_off() {
    use thme::exporter::gguf_check::check_gguf;
    use thme::inference::read_tensor_f32;
    use thme::ir::model::ModelMetadataValue;

    let dir = tempfile::tempdir().expect("temp dir");
    let src_path = dir.path().join("tiny_llama.safetensors");

    let attn_norm_in = [1.25_f32, -0.5, 2.0, 0.125];
    let ffn_norm_in = [0.5_f32, 1.5, -1.0, 3.0];
    let output_norm_in = [2.5_f32, -1.25, 0.25, 0.75];

    let q_in: Vec<f32> = (1..=16).map(|i| i as f32).collect();
    let k_in: Vec<f32> = (21..=36).map(|i| i as f32).collect();
    let embed_in: Vec<f32> = (0..16).map(|i| 100.0 + i as f32).collect();
    let v_in: Vec<f32> = (0..16).map(|i| 200.0 + i as f32).collect();
    let o_in: Vec<f32> = (0..16).map(|i| 300.0 + i as f32).collect();
    let gate_in: Vec<f32> = (0..32).map(|i| 400.0 + i as f32).collect();
    let up_in: Vec<f32> = (0..32).map(|i| 500.0 + i as f32).collect();
    let down_in: Vec<f32> = (0..32).map(|i| 600.0 + i as f32).collect();
    let head_in: Vec<f32> = (0..16).map(|i| 700.0 + i as f32).collect();

    let mut writer = SafeTensorsWriter::new();
    writer.add_tensor(
        "model.embed_tokens.weight",
        "F32",
        vec![4, 4],
        f32_bytes(&embed_in),
    );
    writer.add_tensor(
        "model.layers.0.input_layernorm.weight",
        "BF16",
        vec![4],
        bf16_bytes(&attn_norm_in),
    );
    writer.add_tensor(
        "model.layers.0.self_attn.q_proj.weight",
        "F32",
        vec![4, 4],
        f32_bytes(&q_in),
    );
    writer.add_tensor(
        "model.layers.0.self_attn.k_proj.weight",
        "F32",
        vec![4, 4],
        f32_bytes(&k_in),
    );
    writer.add_tensor(
        "model.layers.0.self_attn.v_proj.weight",
        "F32",
        vec![4, 4],
        f32_bytes(&v_in),
    );
    writer.add_tensor(
        "model.layers.0.self_attn.o_proj.weight",
        "F32",
        vec![4, 4],
        f32_bytes(&o_in),
    );
    writer.add_tensor(
        "model.layers.0.post_attention_layernorm.weight",
        "BF16",
        vec![4],
        bf16_bytes(&ffn_norm_in),
    );
    writer.add_tensor(
        "model.layers.0.mlp.gate_proj.weight",
        "F32",
        vec![8, 4],
        f32_bytes(&gate_in),
    );
    writer.add_tensor(
        "model.layers.0.mlp.up_proj.weight",
        "F32",
        vec![8, 4],
        f32_bytes(&up_in),
    );
    writer.add_tensor(
        "model.layers.0.mlp.down_proj.weight",
        "F32",
        vec![4, 8],
        f32_bytes(&down_in),
    );
    writer.add_tensor(
        "model.norm.weight",
        "BF16",
        vec![4],
        bf16_bytes(&output_norm_in),
    );
    writer.add_tensor("lm_head.weight", "F32", vec![4, 4], f32_bytes(&head_in));
    writer
        .write_to_file(&src_path)
        .expect("write source SafeTensors");

    let tensors =
        SafetensorsScanner::parse_metadata_from_file(&src_path).expect("parse source metadata");
    let model = ModelIR::new(
        "tiny-llama".into(),
        ModelArchitecture::Llama,
        1,
        4,
        1,
        1,
        4,
        16,
        tensors,
    );

    // Flag off: preserves HF names, BF16 norms, unpermuted q/k, no rope.dimension_count.
    let off_path = dir.path().join("flag_off.gguf");
    let off_cfg = ExportConfig {
        format: ExportFormat::GGUFv3,
        quant_type: ExportQuantType::PerTensor,
        output_path: off_path.clone(),
        compute_checksum: false,
        llama_cpp_layout: false,

        tokenizer_json_path: None,
    };
    ModelExporter::export(&model, &off_cfg).expect("export flag-off GGUF");
    let off_bytes = std::fs::read(&off_path).expect("read flag-off GGUF");

    let off_again_path = dir.path().join("flag_off_again.gguf");
    ModelExporter::export(
        &model,
        &ExportConfig {
            output_path: off_again_path.clone(),
            ..off_cfg
        },
    )
    .expect("re-export flag-off GGUF");
    assert_eq!(
        std::fs::read(&off_again_path).expect("read second flag-off GGUF"),
        off_bytes
    );

    let off_model = GgufParser::parse_from_file(&off_path).expect("parse flag-off GGUF");
    let off_names: Vec<&str> = off_model.tensors.iter().map(|t| t.name.as_str()).collect();
    assert!(off_names.contains(&"model.embed_tokens.weight"));
    assert!(off_names.contains(&"model.layers.0.input_layernorm.weight"));
    assert!(off_names.contains(&"model.layers.0.self_attn.q_proj.weight"));
    assert!(off_names.contains(&"model.norm.weight"));
    assert_eq!(
        off_model
            .tensors
            .iter()
            .find(|t| t.name == "model.layers.0.input_layernorm.weight")
            .unwrap()
            .original_dtype,
        "BF16"
    );
    assert_eq!(
        off_model
            .source_gguf_metadata
            .get("llama.rope.dimension_count"),
        None
    );

    // Flag on: GGML names, F32 norms, RoPE-permuted Q/K, rope.dimension_count = 4.
    let on_path = dir.path().join("flag_on.gguf");
    ModelExporter::export(
        &model,
        &ExportConfig {
            format: ExportFormat::GGUFv3,
            quant_type: ExportQuantType::PerTensor,
            output_path: on_path.clone(),
            compute_checksum: false,
            llama_cpp_layout: true,

            tokenizer_json_path: None,
        },
    )
    .expect("export llama.cpp layout GGUF");

    let on_bytes = std::fs::read(&on_path).expect("read flag-on GGUF");
    let findings = check_gguf(&on_bytes);
    assert!(
        findings.is_empty(),
        "unexpected gguf_check findings: {:?}",
        findings
    );

    let on_model = GgufParser::parse_from_file(&on_path).expect("parse flag-on GGUF");
    let mut on_names: Vec<&str> = on_model.tensors.iter().map(|t| t.name.as_str()).collect();
    on_names.sort_unstable();
    let mut expected_names = vec![
        "blk.0.attn_k.weight",
        "blk.0.attn_norm.weight",
        "blk.0.attn_output.weight",
        "blk.0.attn_q.weight",
        "blk.0.attn_v.weight",
        "blk.0.ffn_down.weight",
        "blk.0.ffn_gate.weight",
        "blk.0.ffn_norm.weight",
        "blk.0.ffn_up.weight",
        "output.weight",
        "output_norm.weight",
        "token_embd.weight",
    ];
    expected_names.sort_unstable();
    assert_eq!(on_names, expected_names);

    let find_on = |name: &str| {
        on_model
            .tensors
            .iter()
            .find(|t| t.name == name)
            .unwrap_or_else(|| panic!("missing tensor {name}"))
    };
    let find_off = |name: &str| {
        off_model
            .tensors
            .iter()
            .find(|t| t.name == name)
            .unwrap_or_else(|| panic!("missing flag-off tensor {name}"))
    };

    for (ggml_name, expected_vals) in [
        ("blk.0.attn_norm.weight", attn_norm_in.as_slice()),
        ("blk.0.ffn_norm.weight", ffn_norm_in.as_slice()),
        ("output_norm.weight", output_norm_in.as_slice()),
    ] {
        let t = find_on(ggml_name);
        assert_eq!(t.original_dtype, "F32", "{ggml_name}");
        assert_eq!(read_tensor_f32(t).unwrap(), expected_vals, "{ggml_name}");
    }

    // Input rows [0, 1, 2, 3] become [0, 2, 1, 3].
    let expected_q = vec![
        1.0, 2.0, 3.0, 4.0, 9.0, 10.0, 11.0, 12.0, 5.0, 6.0, 7.0, 8.0, 13.0, 14.0, 15.0, 16.0,
    ];
    let expected_k = vec![
        21.0, 22.0, 23.0, 24.0, 29.0, 30.0, 31.0, 32.0, 25.0, 26.0, 27.0, 28.0, 33.0, 34.0, 35.0,
        36.0,
    ];
    assert_eq!(
        read_tensor_f32(find_on("blk.0.attn_q.weight")).unwrap(),
        expected_q
    );
    assert_eq!(
        read_tensor_f32(find_on("blk.0.attn_k.weight")).unwrap(),
        expected_k
    );

    assert_eq!(
        on_model
            .source_gguf_metadata
            .get("llama.rope.dimension_count"),
        Some(&ModelMetadataValue::Uint32(4))
    );

    // Every other tensor keeps identical payload bytes to the flag-off path.
    for (ggml_name, hf_name) in [
        ("token_embd.weight", "model.embed_tokens.weight"),
        (
            "blk.0.attn_v.weight",
            "model.layers.0.self_attn.v_proj.weight",
        ),
        (
            "blk.0.attn_output.weight",
            "model.layers.0.self_attn.o_proj.weight",
        ),
        (
            "blk.0.ffn_gate.weight",
            "model.layers.0.mlp.gate_proj.weight",
        ),
        ("blk.0.ffn_up.weight", "model.layers.0.mlp.up_proj.weight"),
        (
            "blk.0.ffn_down.weight",
            "model.layers.0.mlp.down_proj.weight",
        ),
        ("output.weight", "lm_head.weight"),
    ] {
        assert_eq!(
            read_source_bytes(find_on(ggml_name).source.as_ref().unwrap()),
            read_source_bytes(find_off(hf_name).source.as_ref().unwrap()),
            "{ggml_name} payload bytes changed"
        );
    }

    // Unmappable tensor name errors and names the offending tensor.
    let mut bad_model = model.clone();
    let mut extra = bad_model.tensors[0].clone();
    extra.name = "rotary_emb.inv_freq".to_string();
    bad_model.tensors.push(extra);
    let err = ModelExporter::export(
        &bad_model,
        &ExportConfig {
            format: ExportFormat::GGUFv3,
            quant_type: ExportQuantType::PerTensor,
            output_path: dir.path().join("unmappable.gguf"),
            compute_checksum: false,
            llama_cpp_layout: true,

            tokenizer_json_path: None,
        },
    )
    .unwrap_err();
    assert!(
        format!("{err:#}").contains("rotary_emb.inv_freq"),
        "{err:#}"
    );
}

#[test]
fn llama_cpp_layout_quantizes_qk_after_permuting_and_preserves_existing_rope_dim() {
    use thme::exporter::ggml_names::permute_rope_rows;
    use thme::inference::{fake_quantize, read_tensor_f32};
    use thme::ir::model::ModelMetadataValue;
    use thme::ir::TensorFormat;

    let dir = tempfile::tempdir().expect("temp dir");
    let src_path = dir.path().join("q8_llama.safetensors");

    let norm_in: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) * 0.125).collect();
    let q_in: Vec<f32> = (0..32 * 32)
        .map(|i| ((i % 97) as f32 - 48.0) * 0.05)
        .collect();
    let k_in: Vec<f32> = (0..16 * 32)
        .map(|i| ((i % 53) as f32 - 26.0) * 0.05)
        .collect();

    let mut writer = SafeTensorsWriter::new();
    writer.add_tensor(
        "model.layers.0.input_layernorm.weight",
        "F16",
        vec![32],
        f16_bytes(&norm_in),
    );
    writer.add_tensor(
        "model.layers.0.self_attn.q_proj.weight",
        "F32",
        vec![32, 32],
        f32_bytes(&q_in),
    );
    writer.add_tensor(
        "model.layers.0.self_attn.k_proj.weight",
        "F32",
        vec![16, 32],
        f32_bytes(&k_in),
    );
    writer
        .write_to_file(&src_path)
        .expect("write source SafeTensors");

    let tensors =
        SafetensorsScanner::parse_metadata_from_file(&src_path).expect("parse source metadata");
    let mut model = ModelIR::new(
        "tiny-q8".into(),
        ModelArchitecture::Llama,
        1,
        32,
        2,
        1,
        32,
        64,
        tensors,
    );

    let out_path = dir.path().join("q8_llama_cpp.gguf");
    ModelExporter::export(
        &model,
        &ExportConfig {
            format: ExportFormat::GGUFv3,
            quant_type: ExportQuantType::Q8_0,
            output_path: out_path.clone(),
            compute_checksum: false,
            llama_cpp_layout: true,

            tokenizer_json_path: None,
        },
    )
    .expect("export Q8_0 llama.cpp layout GGUF");

    let exported = GgufParser::parse_from_file(&out_path).expect("parse Q8_0 GGUF");
    assert_eq!(
        exported
            .source_gguf_metadata
            .get("llama.rope.dimension_count"),
        Some(&ModelMetadataValue::Uint32(16))
    );

    let norm_t = exported
        .tensors
        .iter()
        .find(|t| t.name == "blk.0.attn_norm.weight")
        .unwrap();
    assert_eq!(norm_t.original_dtype, "F32");
    let expected_norm: Vec<f32> = norm_in
        .iter()
        .map(|&v| thme::engine::minifloat::f16_to_f32(thme::engine::minifloat::f32_to_f16(v)))
        .collect();
    assert_eq!(read_tensor_f32(norm_t).unwrap(), expected_norm);

    let q_t = exported
        .tensors
        .iter()
        .find(|t| t.name == "blk.0.attn_q.weight")
        .unwrap();
    assert_eq!(q_t.original_dtype, "Q8_0");
    let mut expected_q = permute_rope_rows(&q_in, 32, 32, 2).unwrap();
    fake_quantize(&mut expected_q, &TensorFormat::Q8_0).unwrap();
    assert_eq!(read_tensor_f32(q_t).unwrap(), expected_q);

    let k_t = exported
        .tensors
        .iter()
        .find(|t| t.name == "blk.0.attn_k.weight")
        .unwrap();
    assert_eq!(k_t.original_dtype, "Q8_0");
    let mut expected_k = permute_rope_rows(&k_in, 16, 32, 1).unwrap();
    fake_quantize(&mut expected_k, &TensorFormat::Q8_0).unwrap();
    assert_eq!(read_tensor_f32(k_t).unwrap(), expected_k);

    // Existing rope.dimension_count in source_gguf_metadata is preserved when already present.
    model.source_gguf_metadata.insert(
        "llama.rope.dimension_count".into(),
        ModelMetadataValue::Uint32(99),
    );
    let preserve_path = dir.path().join("preserve_rope.gguf");
    ModelExporter::export(
        &model,
        &ExportConfig {
            format: ExportFormat::GGUFv3,
            quant_type: ExportQuantType::Q8_0,
            output_path: preserve_path.clone(),
            compute_checksum: false,
            llama_cpp_layout: true,

            tokenizer_json_path: None,
        },
    )
    .expect("export with existing rope.dimension_count");
    let preserved = GgufParser::parse_from_file(&preserve_path).expect("parse preserved GGUF");
    assert_eq!(
        preserved
            .source_gguf_metadata
            .get("llama.rope.dimension_count"),
        Some(&ModelMetadataValue::Uint32(99))
    );
}
