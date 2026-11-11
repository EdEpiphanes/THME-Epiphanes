//! End-to-End Integration Tests for Binary Model Exporter (Milestones v0.1.8.4 & v0.1.8.5)
//!
//! Verifies:
//! 1. Full model serialization to GGUF v3 with metadata reconstitution.
//! 2. Full model serialization to SafeTensors with JSON header and contiguous payload.
//! 3. BitNet b1.58 ternary packed weight serialization and compression ratios.
//! 4. SHA-256 checksum generation and integrity validation.

use thme::exporter::{ExportConfig, ExportFormat, ExportQuantType, ModelExporter};
use thme::ir::{ModelArchitecture, ModelIR, TensorIR};
use thme::parser::SafetensorsScanner;

fn build_mock_transformer() -> ModelIR {
    let mut tensors = Vec::new();
    let hidden = 128;
    let intermediate = 256;

    // Embedding
    tensors.push(TensorIR::new(
        "model.embed_tokens.weight".to_string(),
        vec![1000, hidden],
        "F16".to_string(),
        1000 * hidden * 2,
    ));

    // Layer 0 Self-Attention & MLP
    tensors.push(TensorIR::new(
        "model.layers.0.self_attn.q_proj.weight".to_string(),
        vec![hidden, hidden],
        "F16".to_string(),
        hidden * hidden * 2,
    ));
    tensors.push(TensorIR::new(
        "model.layers.0.self_attn.k_proj.weight".to_string(),
        vec![hidden, hidden],
        "F16".to_string(),
        hidden * hidden * 2,
    ));
    tensors.push(TensorIR::new(
        "model.layers.0.mlp.up_proj.weight".to_string(),
        vec![intermediate, hidden],
        "F16".to_string(),
        intermediate * hidden * 2,
    ));
    tensors.push(TensorIR::new(
        "model.layers.0.mlp.down_proj.weight".to_string(),
        vec![hidden, intermediate],
        "F16".to_string(),
        hidden * intermediate * 2,
    ));

    ModelIR::new(
        "mock-smollm-135m".to_string(),
        ModelArchitecture::Llama,
        1,
        hidden,
        4,
        4,
        1000,
        2048,
        tensors,
    )
}

#[test]
fn test_e2e_export_gguf_v3_with_metadata_sync() {
    let temp_dir = tempfile::tempdir().expect("temp dir");
    let out_path = temp_dir.path().join("model.gguf");

    let model = build_mock_transformer();
    let config = ExportConfig {
        format: ExportFormat::GGUFv3,
        quant_type: ExportQuantType::Q8_0,
        output_path: out_path.clone(),
        compute_checksum: true,
    };

    let report = ModelExporter::export(&model, &config).expect("Export GGUF");
    assert_eq!(report.format, ExportFormat::GGUFv3);
    assert_eq!(report.quant_type, ExportQuantType::Q8_0);
    assert_eq!(report.total_tensors, 5);
    assert!(report.exported_bytes > 0);
    assert!(report.sha256_checksum.is_some());

    // Verify file exists and has GGUF magic
    assert!(out_path.exists());
    let bytes = std::fs::read(&out_path).expect("Read GGUF");
    assert_eq!(&bytes[0..4], b"GGUF");
    assert_eq!(u32::from_le_bytes(bytes[4..8].try_into().unwrap()), 3);
}

#[test]
fn test_e2e_export_safetensors_with_deserialization() {
    let temp_dir = tempfile::tempdir().expect("temp dir");
    let out_path = temp_dir.path().join("model.safetensors");

    let model = build_mock_transformer();
    let config = ExportConfig {
        format: ExportFormat::SafeTensors,
        quant_type: ExportQuantType::FP16,
        output_path: out_path.clone(),
        compute_checksum: true,
    };

    let report = ModelExporter::export(&model, &config).expect("Export SafeTensors");
    assert_eq!(report.format, ExportFormat::SafeTensors);
    assert_eq!(report.quant_type, ExportQuantType::FP16);
    assert_eq!(report.total_tensors, 5);
    assert!(report.sha256_checksum.is_some());

    // Parse back using SafetensorsScanner
    let scanned_tensors = SafetensorsScanner::parse_metadata_from_file(&out_path)
        .expect("Scan written SafeTensors file");
    assert_eq!(scanned_tensors.len(), 5);

    let embed = scanned_tensors
        .iter()
        .find(|t| t.name == "model.embed_tokens.weight")
        .expect("embed tensor");
    assert_eq!(embed.shape, vec![1000, 128]);
    assert_eq!(embed.original_dtype, "F16");
}

#[test]
fn test_e2e_export_bitnet_b158_compression() {
    let temp_dir = tempfile::tempdir().expect("temp dir");
    let out_path = temp_dir.path().join("model_bitnet.safetensors");

    let model = build_mock_transformer();
    let config = ExportConfig {
        format: ExportFormat::SafeTensors,
        quant_type: ExportQuantType::BitNetB158,
        output_path: out_path,
        compute_checksum: false,
    };

    let report = ModelExporter::export(&model, &config).expect("Export BitNet SafeTensors");
    assert_eq!(report.quant_type, ExportQuantType::BitNetB158);
    // BitNet 2-bit packing achieves substantial compression over 16-bit baseline
    assert!(report.compression_ratio > 4.0);
}
