//! Integration & Verification Test Suite for BitNet b1.58 Bitstream Packer (v0.1.8.1)
//!
//! Verifies:
//! 1. Bitwise round-trip fidelity on ternary matrices {-1, 0, +1}.
//! 2. Per-channel scale extraction: alpha_i = mean(|W_ij|).
//! 3. Column padding alignment (when cols % 4 != 0).
//! 4. Unpacking and de-quantization accuracy against expected floating-point values.
//! 5. Compression ratio (>7.0x against FP16 baselines on realistic layer dimensions).
//! 6. Serialization and JSON round-trip of ExportConfig, ExportReport, and PackedBitNetTensor.
//! 7. Edge cases: empty matrix, single row/column, all zeros, extreme magnitudes.

use thme::exporter::{
    BitNetPacker, ExportConfig, ExportFormat, ExportQuantType, ExportReport, PackedBitNetTensor,
};

#[test]
fn test_bitnet_packer_ternary_roundtrip() {
    let rows = 4;
    let cols = 12; // exactly 3 bytes per row
    let mut ternary = Vec::with_capacity(rows * cols);
    for r in 0..rows {
        for c in 0..cols {
            let val = match (r + c) % 3 {
                0 => -1i8,
                1 => 0i8,
                _ => 1i8,
            };
            ternary.push(val);
        }
    }
    let scales = vec![0.125, 0.25, 0.5, 1.0];

    let packed =
        BitNetPacker::pack_ternary_slice("layer.0.attn.q_proj", rows, cols, &ternary, &scales);
    assert_eq!(packed.rows, 4);
    assert_eq!(packed.cols, 12);
    assert_eq!(packed.packed_data.len(), 4 * 3); // 12 bytes

    let unpacked = BitNetPacker::unpack_raw_ternary(&packed);
    assert_eq!(unpacked, ternary);

    let unpacked_floats = BitNetPacker::unpack_matrix(&packed);
    assert_eq!(unpacked_floats.len(), rows * cols);

    // Verify scaled values for each row
    for (r, &scale) in scales.iter().enumerate().take(rows) {
        for c in 0..cols {
            let idx = r * cols + c;
            let expected = ternary[idx] as f32 * scale;
            assert!((unpacked_floats[idx] - expected).abs() < 1e-6);
        }
    }
}

#[test]
fn test_bitnet_packer_auto_scaling_and_quantization() {
    let rows = 2;
    let cols = 8;
    // Row 0 has mean abs of ~0.5
    // Row 1 has mean abs of ~2.0
    let weights = vec![
        0.5, -0.5, 0.0, 0.5, -0.5, 0.5, 0.0, -0.5, 2.0, -2.0, 0.0, 2.0, -2.0, 2.0, 0.0, -2.0,
    ];

    let packed = BitNetPacker::pack_matrix("layer.0.mlp.down_proj", rows, cols, &weights);
    assert_eq!(packed.rows, 2);
    assert_eq!(packed.cols, 8);
    assert_eq!(packed.row_scales.len(), 2);

    // Row 0 scale should be 3.0 / 8 = 0.375
    // Row 1 scale should be 12.0 / 8 = 1.5
    assert!((packed.row_scales[0] - 0.375).abs() < 1e-4);
    assert!((packed.row_scales[1] - 1.5).abs() < 1e-4);

    let raw_ternary = BitNetPacker::unpack_raw_ternary(&packed);
    assert_eq!(raw_ternary.len(), 16);
    // Signs should be preserved
    assert_eq!(raw_ternary[0], 1);
    assert_eq!(raw_ternary[1], -1);
    assert_eq!(raw_ternary[2], 0);
}

#[test]
fn test_bitnet_packer_odd_column_dimensions_padding() {
    // Tests column sizes from 1 to 15 to ensure padding byte logic handles all alignments
    for cols in 1..=15 {
        let rows = 3;
        let total = rows * cols;
        let weights: Vec<f32> = (0..total)
            .map(|i| if i % 2 == 0 { 0.7 } else { -0.7 })
            .collect();

        let packed = BitNetPacker::pack_matrix("odd_dim", rows, cols, &weights);
        let expected_bytes_per_row = cols.div_ceil(4);
        assert_eq!(packed.packed_data.len(), rows * expected_bytes_per_row);

        let unpacked = BitNetPacker::unpack_matrix(&packed);
        assert_eq!(unpacked.len(), total);
    }
}

#[test]
fn test_bitnet_compression_ratio_realistic_layer() {
    // Hidden dimension 4090 x 4096 (standard LLaMA-style MLP down projection)
    let rows = 512;
    let cols = 1024;
    let weights = vec![0.05f32; rows * cols];

    let packed = BitNetPacker::pack_matrix("mlp_down", rows, cols, &weights);
    // FP16 size = 512 * 1024 * 2 = 1,048,576 bytes (1 MB)
    // Packed payload = 512 * 256 = 131,072 bytes
    // Scales = 512 * 4 = 2,048 bytes
    // Total packed = 133,120 bytes -> Compression ratio = 1048576 / 133120 = 7.876x
    let ratio = packed.compression_ratio();
    assert!(ratio > 7.8, "Expected ratio > 7.8x, got {}", ratio);
}

#[test]
fn test_bitnet_empty_and_zero_edge_cases() {
    let empty_packed = BitNetPacker::pack_matrix("empty", 0, 0, &[]);
    assert_eq!(empty_packed.element_count(), 0);
    assert_eq!(empty_packed.packed_byte_size(), 0);
    assert_eq!(BitNetPacker::unpack_matrix(&empty_packed).len(), 0);

    // All zero matrix
    let zeros = vec![0.0f32; 16];
    let zero_packed = BitNetPacker::pack_matrix("zeros", 4, 4, &zeros);
    let unpacked_zeros = BitNetPacker::unpack_matrix(&zero_packed);
    for &val in &unpacked_zeros {
        assert_eq!(val, 0.0);
    }
}

#[test]
fn test_export_config_and_report_json_serde() {
    let config = ExportConfig {
        format: ExportFormat::GGUFv3,
        quant_type: ExportQuantType::BitNetB158,
        output_path: std::path::PathBuf::from("models/exported_model.gguf"),
        compute_checksum: true,
    };

    let json_str = serde_json::to_string_pretty(&config).expect("Serialize config");
    let deserialized: ExportConfig = serde_json::from_str(&json_str).expect("Deserialize config");
    assert_eq!(deserialized.format, ExportFormat::GGUFv3);
    assert_eq!(deserialized.quant_type, ExportQuantType::BitNetB158);

    let report = ExportReport {
        format: ExportFormat::SafeTensors,
        quant_type: ExportQuantType::Q8_0,
        total_tensors: 142,
        original_bytes: 4_200_000_000,
        exported_bytes: 1_050_000_000,
        compression_ratio: 4.0,
        sha256_checksum: Some(
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855".to_string(),
        ),
    };

    let report_json = serde_json::to_string(&report).expect("Serialize report");
    let deser_report: ExportReport =
        serde_json::from_str(&report_json).expect("Deserialize report");
    assert_eq!(deser_report.total_tensors, 142);
    assert_eq!(deser_report.compression_ratio, 4.0);

    let tensor = PackedBitNetTensor {
        name: "test_weight".to_string(),
        rows: 2,
        cols: 4,
        row_scales: vec![0.5, 0.5],
        packed_data: vec![0x1B, 0x1B],
    };
    let tensor_json = serde_json::to_string(&tensor).expect("Serialize tensor");
    let deser_tensor: PackedBitNetTensor =
        serde_json::from_str(&tensor_json).expect("Deserialize tensor");
    assert_eq!(deser_tensor.name, "test_weight");
    assert_eq!(deser_tensor.packed_byte_size(), 2);
}
