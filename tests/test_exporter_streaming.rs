use std::io::{Read, Seek, SeekFrom};

use thme::exporter::gguf_writer::GgufWriter;
use thme::exporter::safetensors_writer::SafeTensorsWriter;
use thme::exporter::{
    ExportConfig, ExportFormat, ExportQuantType, ModelExporter, StreamingChunkWriter,
    STREAM_CHUNK_SIZE,
};
use thme::ir::tensor::TensorSource;
use thme::ir::{ModelArchitecture, ModelIR};
use thme::parser::gguf::{GgmlType, GgufParser};
use thme::parser::SafetensorsScanner;

fn read_source(source: &TensorSource) -> Vec<u8> {
    let mut file = std::fs::File::open(&source.path).expect("open source tensor");
    file.seek(SeekFrom::Start(source.byte_offset))
        .expect("seek to source tensor");
    let mut bytes = vec![0; source.byte_len as usize];
    file.read_exact(&mut bytes).expect("read source tensor");
    bytes
}

#[test]
fn source_backed_gguf_export_preserves_tensor_bytes() {
    let payload = [0x00, 0x3c, 0x00, 0xc0, 0x00, 0x42, 0x00, 0x34];
    let dir = tempfile::tempdir().expect("temp dir");
    let source_path = dir.path().join("source.gguf");
    let output_path = dir.path().join("exported.gguf");

    let mut source_writer = GgufWriter::new();
    source_writer.add_tensor(
        "blk.0.attn_q.weight",
        vec![4],
        GgmlType::F16,
        payload.to_vec(),
    );
    source_writer
        .write_to_file(&source_path)
        .expect("write source GGUF");

    let model = GgufParser::parse_from_file(&source_path).expect("parse source GGUF");
    let config = ExportConfig {
        format: ExportFormat::GGUFv3,
        quant_type: ExportQuantType::FP16,
        output_path: output_path.clone(),
        compute_checksum: true,
    };
    let report = ModelExporter::export(&model, &config).expect("stream GGUF export");
    assert_eq!(report.total_tensors, 1);
    assert!(report.sha256_checksum.is_some());

    let exported = GgufParser::parse_from_file(&output_path).expect("parse exported GGUF");
    assert_eq!(
        read_source(exported.tensors[0].source.as_ref().unwrap()),
        payload
    );
}

#[test]
fn source_backed_q8_gguf_export_preserves_quantized_blocks() {
    let payload = vec![0x5au8; 34];
    let dir = tempfile::tempdir().expect("temp dir");
    let source_path = dir.path().join("source-q8.gguf");
    let output_path = dir.path().join("exported-q8.gguf");
    let mut source_writer = GgufWriter::new();
    source_writer.add_tensor("weights", vec![32], GgmlType::Q8_0, payload.clone());
    source_writer
        .write_to_file(&source_path)
        .expect("write source GGUF");

    let model = GgufParser::parse_from_file(&source_path).expect("parse source GGUF");
    let config = ExportConfig {
        format: ExportFormat::GGUFv3,
        quant_type: ExportQuantType::Q8_0,
        output_path: output_path.clone(),
        compute_checksum: false,
    };
    ModelExporter::export(&model, &config).expect("stream Q8_0 export");

    let exported = GgufParser::parse_from_file(&output_path).expect("parse exported GGUF");
    assert_eq!(
        read_source(exported.tensors[0].source.as_ref().unwrap()),
        payload
    );
}

#[test]
fn source_backed_safetensors_export_preserves_tensor_bytes() {
    let payload = [0x00, 0x3c, 0x00, 0xc0, 0x00, 0x42, 0x00, 0x34];
    let dir = tempfile::tempdir().expect("temp dir");
    let source_path = dir.path().join("source.safetensors");
    let output_path = dir.path().join("exported.safetensors");

    let mut source_writer = SafeTensorsWriter::new();
    source_writer.add_tensor("weights", "F16", vec![4], payload.to_vec());
    source_writer
        .write_to_file(&source_path)
        .expect("write source SafeTensors");

    let tensors = SafetensorsScanner::parse_metadata_from_file(&source_path)
        .expect("parse source SafeTensors");
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
    let config = ExportConfig {
        format: ExportFormat::SafeTensors,
        quant_type: ExportQuantType::FP16,
        output_path: output_path.clone(),
        compute_checksum: false,
    };
    ModelExporter::export(&model, &config).expect("stream SafeTensors export");

    let exported = SafetensorsScanner::parse_metadata_from_file(&output_path)
        .expect("parse exported SafeTensors");
    assert_eq!(read_source(exported[0].source.as_ref().unwrap()), payload);
}

#[test]
fn model_export_rejects_missing_sources_and_unsupported_conversion() {
    let dir = tempfile::tempdir().expect("temp dir");
    let source_path = dir.path().join("source.safetensors");
    let mut source_writer = SafeTensorsWriter::new();
    source_writer.add_tensor("weights", "F16", vec![4], vec![0; 8]);
    source_writer
        .write_to_file(&source_path)
        .expect("write source SafeTensors");
    let tensors = SafetensorsScanner::parse_metadata_from_file(&source_path)
        .expect("parse source SafeTensors");
    let sourced_model = ModelIR::new(
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
    let mut config = ExportConfig {
        format: ExportFormat::SafeTensors,
        quant_type: ExportQuantType::Q8_0,
        output_path: dir.path().join("unsupported.safetensors"),
        compute_checksum: false,
    };
    assert!(ModelExporter::export(&sourced_model, &config)
        .unwrap_err()
        .to_string()
        .contains("same-dtype"));
    assert!(!config.output_path.exists());

    let source_less_model = ModelIR::new(
        "metadata-only".into(),
        ModelArchitecture::Llama,
        1,
        4,
        1,
        1,
        4,
        128,
        vec![thme::ir::TensorIR::new(
            "weights".into(),
            vec![4],
            "F16".into(),
            8,
        )],
    );
    config.quant_type = ExportQuantType::FP16;
    config.output_path = dir.path().join("missing.safetensors");
    assert!(ModelExporter::export(&source_less_model, &config)
        .unwrap_err()
        .to_string()
        .contains("no file-backed payload"));
    assert!(!config.output_path.exists());
}

#[test]
fn streaming_helper_uses_fixed_size_chunks() {
    let input = vec![42u8; STREAM_CHUNK_SIZE * 3 + 17];
    let mut reader = std::io::Cursor::new(input.clone());
    let mut output = Vec::new();
    let mut writer = StreamingChunkWriter::new(&mut output);
    let written = writer
        .write_reader(&mut reader, input.len() as u64)
        .expect("stream payload");
    assert_eq!(written, input.len() as u64);
    assert_eq!(output, input);
}

#[test]
fn source_backed_safetensors_rejects_invalid_dtype_shape_ranges() {
    let dir = tempfile::tempdir().expect("temp dir");
    let missing_source = dir.path().join("not-opened.bin");

    let mut wrong_length = SafeTensorsWriter::new();
    wrong_length.add_tensor_source(
        "weights",
        "F16",
        vec![4],
        TensorSource {
            path: missing_source.clone(),
            byte_offset: 0,
            byte_len: 7,
        },
    );
    let output = dir.path().join("wrong-length.safetensors");
    let error = wrong_length.write_to_file(&output).unwrap_err().to_string();
    assert!(error.contains("7 bytes"));
    assert!(error.contains("require 8 bytes"));
    assert!(!output.exists());

    let mut overflow_shape = SafeTensorsWriter::new();
    overflow_shape.add_tensor_source(
        "weights",
        "F32",
        vec![usize::MAX, 2],
        TensorSource {
            path: missing_source.clone(),
            byte_offset: 0,
            byte_len: 8,
        },
    );
    let output = dir.path().join("overflow-shape.safetensors");
    let error = overflow_shape
        .write_to_file(&output)
        .unwrap_err()
        .to_string();
    assert!(error.contains("shape overflows"));
    assert!(!output.exists());

    let mut unsupported_dtype = SafeTensorsWriter::new();
    unsupported_dtype.add_tensor_source(
        "weights",
        "Q8_0",
        vec![32],
        TensorSource {
            path: missing_source,
            byte_offset: 0,
            byte_len: 34,
        },
    );
    let output = dir.path().join("unsupported-dtype.safetensors");
    let error = unsupported_dtype
        .write_to_file(&output)
        .unwrap_err()
        .to_string();
    assert!(error.contains("Unsupported SafeTensors dtype"));
    assert!(!output.exists());
}
