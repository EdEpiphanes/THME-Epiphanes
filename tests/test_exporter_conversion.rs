#![allow(clippy::chunks_exact_to_as_chunks)]
// SPDX-License-Identifier: AGPL-3.0-only

use std::io::{Read, Seek, SeekFrom};

use thme::exporter::calibration::optimal_ternary_scale;
use thme::exporter::gguf_writer::GgufWriter;
use thme::exporter::{ExportConfig, ExportFormat, ExportQuantType, ModelExporter};
use thme::parser::gguf::{GgmlType, GgufParser};

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
        },
    )
    .unwrap_err();
    assert!(error.to_string().contains("row width 33"));
    assert!(!output.exists());
}

use thme::parser::gguf::GgufBlockDecoder;
