#![allow(clippy::chunks_exact_to_as_chunks)]
// SPDX-License-Identifier: AGPL-3.0-only

use std::io::{Read, Seek, SeekFrom};

use thme::exporter::calibration::optimal_ternary_scale;
use thme::exporter::gguf_check;
use thme::exporter::gguf_writer::GgufWriter;
use thme::exporter::{ExportConfig, ExportFormat, ExportQuantType, ModelExporter};
use thme::parser::gguf::{GgmlType, GgufBlockDecoder, GgufParser, GgufValue};

fn float_source(path: &std::path::Path, elements: usize) -> Vec<u8> {
    let payload: Vec<u8> = (0..elements)
        .map(|index| ((index as f32 % 31.0) - 15.0) / 7.0)
        .flat_map(f32::to_le_bytes)
        .collect();
    let mut writer = GgufWriter::new();
    writer.add_tensor("weights", vec![elements], GgmlType::F32, payload.clone());
    writer.write_to_file(path).expect("write source GGUF");
    payload
}

fn tensor_payload(path: &std::path::Path, source: &thme::ir::tensor::TensorSource) -> Vec<u8> {
    let mut file = std::fs::File::open(path).expect("open file");
    file.seek(SeekFrom::Start(source.byte_offset))
        .expect("seek payload");
    let mut bytes = vec![0; source.byte_len as usize];
    file.read_exact(&mut bytes).expect("read payload");
    bytes
}

#[test]
fn source_backed_f32_exports_to_standard_q8_0_blocks() {
    let dir = tempfile::tempdir().expect("temp dir");
    let source_path = dir.path().join("source.gguf");
    let input = float_source(&source_path, 64);
    let model = GgufParser::parse_from_file(&source_path).expect("parse source");
    let output = dir.path().join("q8.gguf");
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
    .expect("export Q8_0");
    let exported = GgufParser::parse_from_file(&output).expect("parse export");
    assert_eq!(exported.tensors[0].original_dtype, "Q8_0");
    let payload = tensor_payload(&output, exported.tensors[0].source.as_ref().unwrap());
    assert_eq!(payload.len(), 68);
    let input_values: Vec<f32> = input
        .chunks_exact(4)
        .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
        .collect();
    let decoded: Vec<f32> = payload
        .chunks_exact(34)
        .flat_map(|block| GgufBlockDecoder::unpack_q8_0(block).expect("decode Q8_0"))
        .collect();
    assert_eq!(decoded.len(), input_values.len());
    assert!(decoded
        .iter()
        .zip(input_values)
        .all(|(actual, expected)| (actual - expected).abs() < 0.05));
}

#[test]
fn source_backed_f32_exports_to_standard_tq1_0_blocks() {
    let dir = tempfile::tempdir().expect("temp dir");
    let source_path = dir.path().join("source.gguf");
    let input = float_source(&source_path, 256);
    let model = GgufParser::parse_from_file(&source_path).expect("parse source");
    let output = dir.path().join("tq1.gguf");
    ModelExporter::export(
        &model,
        &ExportConfig {
            format: ExportFormat::GGUFv3,
            quant_type: ExportQuantType::TQ1_0,
            output_path: output.clone(),
            compute_checksum: false,

            llama_cpp_layout: false,

            tokenizer_json_path: None,
        },
    )
    .expect("export TQ1_0");
    let exported = GgufParser::parse_from_file(&output).expect("parse export");
    assert_eq!(exported.tensors[0].original_dtype, "TQ1_0");
    assert_eq!(exported.tensors[0].byte_size, 54);
    let payload = tensor_payload(&output, exported.tensors[0].source.as_ref().unwrap());
    assert_eq!(payload.len(), 54);
    assert_eq!(GgmlType::from_u32(34), GgmlType::TQ1_0);
    let decoded = GgufBlockDecoder::unpack_tq1_0(&payload).expect("decode TQ1_0");
    assert_eq!(decoded.len(), 256);
    let input_values: Vec<f32> = input
        .chunks_exact(4)
        .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
        .collect();
    let calibrated_scale = optimal_ternary_scale(&input_values).unwrap().scale;
    let stored_scale = thme::profiling::f16_to_f32(u16::from_le_bytes([payload[52], payload[53]]));
    for (index, (actual, input)) in decoded.iter().zip(input_values).enumerate() {
        let trit = (input / calibrated_scale).round().clamp(-1.0, 1.0);
        assert!(
            (actual - trit * stored_scale).abs() < 1e-6,
            "TQ1_0 trit order mismatch at element {index}"
        );
    }
}

#[test]
fn positive_tq1_scale_underflowing_fp16_fails_export() {
    let dir = tempfile::tempdir().expect("temp dir");
    let source_path = dir.path().join("tiny.gguf");
    let payload: Vec<u8> = vec![1.0e-10f32; 256]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect();
    let mut writer = GgufWriter::new();
    writer.add_tensor("weights", vec![256], GgmlType::F32, payload);
    writer
        .write_to_file(&source_path)
        .expect("write source GGUF");
    let model = GgufParser::parse_from_file(&source_path).expect("parse source");
    let output = dir.path().join("tiny-tq1.gguf");
    let error = ModelExporter::export(
        &model,
        &ExportConfig {
            format: ExportFormat::GGUFv3,
            quant_type: ExportQuantType::TQ1_0,
            output_path: output,
            compute_checksum: false,

            llama_cpp_layout: false,

            tokenizer_json_path: None,
        },
    )
    .unwrap_err();
    assert!(format!("{error:#}").contains("underflows to zero in FP16"));
}

#[test]
fn quantized_rows_must_not_cross_ggml_block_boundaries() {
    let dir = tempfile::tempdir().expect("temp dir");
    let source_path = dir.path().join("unaligned.gguf");
    float_source(&source_path, 33);
    let model = GgufParser::parse_from_file(&source_path).expect("parse source");
    let output = dir.path().join("invalid.gguf");
    let error = ModelExporter::export(
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
    .unwrap_err();
    assert!(error.to_string().contains("row width 33"));
    assert!(!output.exists());
}

fn bf16_payload_and_expected(values: &[f32]) -> (Vec<u8>, Vec<f32>) {
    let mut bytes = Vec::with_capacity(values.len() * 2);
    let mut expected = Vec::with_capacity(values.len());
    for &value in values {
        let bits = thme::engine::minifloat::f32_to_bf16(value);
        bytes.extend_from_slice(&bits.to_le_bytes());
        expected.push(thme::profiling::bf16_to_f32(bits));
    }
    (bytes, expected)
}

fn tiny_bf16_source(path: &std::path::Path) -> (thme::ir::ModelIR, Vec<f32>, Vec<f32>, Vec<f32>) {
    let embed_raw: Vec<f32> = (0..128).map(|i| ((i % 15) as f32 - 7.0) * 0.25).collect();
    let linear_raw: Vec<f32> = (0..64)
        .map(|i| (((i + 5) % 15) as f32 - 7.0) * 0.25)
        .collect();
    let norm_raw: Vec<f32> = (0..32).map(|i| ((i % 7) as f32 + 1.0) * 0.25).collect();

    let (embed_bytes, embed_expected) = bf16_payload_and_expected(&embed_raw);
    let (linear_bytes, linear_expected) = bf16_payload_and_expected(&linear_raw);
    let (norm_bytes, norm_expected) = bf16_payload_and_expected(&norm_raw);

    let mut writer = GgufWriter::new();
    writer.add_string("general.architecture", "llama");
    writer.add_u32("llama.block_count", 1);
    writer.add_u32("llama.embedding_length", 32);
    writer.add_u32("llama.attention.head_count", 2);
    writer.add_u32("llama.attention.head_count_kv", 2);
    writer.add_string("tokenizer.ggml.model", "llama");
    writer.add_metadata(
        "tokenizer.ggml.tokens",
        GgufValue::Array(vec![
            GgufValue::String("<s>".to_string()),
            GgufValue::String("</s>".to_string()),
            GgufValue::String("a".to_string()),
            GgufValue::String("b".to_string()),
        ]),
    );
    writer.add_tensor(
        "model.embed_tokens.weight",
        vec![4, 32],
        GgmlType::BF16,
        embed_bytes,
    );
    writer.add_tensor(
        "model.layers.0.mlp.up_proj.weight",
        vec![2, 32],
        GgmlType::BF16,
        linear_bytes,
    );
    writer.add_tensor("model.norm.weight", vec![32], GgmlType::BF16, norm_bytes);
    writer.write_to_file(path).expect("write BF16 source GGUF");

    let model = GgufParser::parse_from_file(path).expect("parse BF16 source");
    (model, embed_expected, linear_expected, norm_expected)
}

fn decode_f16_payload(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|b| thme::profiling::f16_to_f32(u16::from_le_bytes([b[0], b[1]])))
        .collect()
}

fn decode_f32_payload(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

fn decode_q8_0_payload(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(34)
        .flat_map(|block| GgufBlockDecoder::unpack_q8_0(block).expect("decode Q8_0"))
        .collect()
}

fn assert_q8_0_within_tolerance(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (a_block, e_block) in actual.chunks_exact(32).zip(expected.chunks_exact(32)) {
        let max_abs = e_block.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let tol = max_abs / 254.0;
        for (a, e) in a_block.iter().zip(e_block) {
            let diff = (a - e).abs();
            assert!(
                diff <= tol,
                "Q8_0 diff {diff} exceeds max|x|/254 ({tol}) for actual {a}, expected {e}"
            );
        }
    }
}

#[test]
fn bf16_sources_export_to_fp16_and_q8_0_including_llama_cpp_layout() {
    let dir = tempfile::tempdir().expect("temp dir");
    let source_path = dir.path().join("bf16_source.gguf");
    let (model, embed_expected, linear_expected, norm_expected) = tiny_bf16_source(&source_path);

    let cases = [
        ("fp16_raw.gguf", ExportQuantType::FP16, false),
        ("q8_raw.gguf", ExportQuantType::Q8_0, false),
        ("fp16_llama.gguf", ExportQuantType::FP16, true),
        ("q8_llama.gguf", ExportQuantType::Q8_0, true),
    ];

    for (filename, quant_type, llama_cpp_layout) in cases {
        let output = dir.path().join(filename);
        ModelExporter::export(
            &model,
            &ExportConfig {
                format: ExportFormat::GGUFv3,
                quant_type,
                output_path: output.clone(),
                compute_checksum: false,
                llama_cpp_layout,
                tokenizer_json_path: None,
            },
        )
        .unwrap_or_else(|err| panic!("export {filename} failed: {err:#}"));

        let exported = GgufParser::parse_from_file(&output).expect("parse exported GGUF");
        let (embed_name, linear_name, norm_name) = if llama_cpp_layout {
            (
                "token_embd.weight",
                "blk.0.ffn_up.weight",
                "output_norm.weight",
            )
        } else {
            (
                "model.embed_tokens.weight",
                "model.layers.0.mlp.up_proj.weight",
                "model.norm.weight",
            )
        };
        let expected_linear_dtype = match quant_type {
            ExportQuantType::FP16 => "F16",
            ExportQuantType::Q8_0 => "Q8_0",
            _ => unreachable!(),
        };
        let expected_norm_dtype = if llama_cpp_layout {
            "F32"
        } else {
            expected_linear_dtype
        };

        let find = |name: &str| {
            exported
                .tensors
                .iter()
                .find(|t| t.name == name)
                .unwrap_or_else(|| panic!("missing tensor {name} in {filename}"))
        };
        let embed = find(embed_name);
        let linear = find(linear_name);
        let norm = find(norm_name);

        assert_eq!(embed.original_dtype, expected_linear_dtype);
        assert_eq!(linear.original_dtype, expected_linear_dtype);
        assert_eq!(norm.original_dtype, expected_norm_dtype);

        let embed_bytes = tensor_payload(&output, embed.source.as_ref().unwrap());
        let linear_bytes = tensor_payload(&output, linear.source.as_ref().unwrap());
        let norm_bytes = tensor_payload(&output, norm.source.as_ref().unwrap());

        match quant_type {
            ExportQuantType::FP16 => {
                assert_eq!(decode_f16_payload(&embed_bytes), embed_expected);
                assert_eq!(decode_f16_payload(&linear_bytes), linear_expected);
            }
            ExportQuantType::Q8_0 => {
                assert_q8_0_within_tolerance(&decode_q8_0_payload(&embed_bytes), &embed_expected);
                assert_q8_0_within_tolerance(&decode_q8_0_payload(&linear_bytes), &linear_expected);
            }
            _ => unreachable!(),
        }

        if llama_cpp_layout {
            assert_eq!(decode_f32_payload(&norm_bytes), norm_expected);
            let raw = std::fs::read(&output).expect("read exported bytes");
            assert!(gguf_check::check_llama_cpp_requirements(&raw).is_empty());
        } else {
            match quant_type {
                ExportQuantType::FP16 => {
                    assert_eq!(decode_f16_payload(&norm_bytes), norm_expected);
                }
                ExportQuantType::Q8_0 => {
                    assert_q8_0_within_tolerance(&decode_q8_0_payload(&norm_bytes), &norm_expected);
                }
                _ => unreachable!(),
            }
        }
    }
}
