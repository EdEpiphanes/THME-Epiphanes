// SPDX-License-Identifier: AGPL-3.0-only
//! Integration Tests for Pure-Rust GGUF v3 Binary Writer (Milestone v0.1.8.2)

use std::path::Path;
use tempfile::tempdir;
use thme::exporter::bitnet_pack::BitNetPacker;
use thme::exporter::gguf_writer::{GgufTensor, GgufWriter};
use thme::ir::ModelArchitecture;
use thme::parser::gguf::{GgmlType, GgufBlockDecoder, GgufParser, GgufReader};
use thme::parser::ModelImporter;

#[test]
fn test_e2e_mock_llama_gguf_export_and_import() {
    let dir = tempdir().expect("Failed to create temporary directory");
    let model_path = dir.path().join("mock_llama.gguf");

    let mut writer = GgufWriter::new();
    writer
        .add_string("general.architecture", "llama")
        .add_u32("llama.block_count", 2)
        .add_u32("llama.embedding_length", 64)
        .add_u32("llama.attention.head_count", 4)
        .add_u32("llama.attention.head_count_kv", 4)
        .add_u32("llama.context_length", 512)
        .add_u32("llama.vocab_size", 1000);

    // Register 6 mock tensors
    writer.add_tensor(
        "token_embd.weight",
        vec![1000, 64],
        GgmlType::F32,
        vec![0u8; 1000 * 64 * 4],
    );
    writer.add_tensor(
        "blk.0.attn_q.weight",
        vec![64, 64],
        GgmlType::F32,
        vec![0u8; 64 * 64 * 4],
    );
    writer.add_tensor(
        "blk.0.attn_k.weight",
        vec![64, 64],
        GgmlType::F32,
        vec![0u8; 64 * 64 * 4],
    );
    writer.add_tensor(
        "blk.0.attn_v.weight",
        vec![64, 64],
        GgmlType::F32,
        vec![0u8; 64 * 64 * 4],
    );
    writer.add_tensor(
        "blk.0.attn_output.weight",
        vec![64, 64],
        GgmlType::F32,
        vec![0u8; 64 * 64 * 4],
    );
    writer.add_tensor(
        "output_norm.weight",
        vec![64],
        GgmlType::F32,
        vec![0u8; 64 * 4],
    );

    let written_bytes = writer
        .write_to_file(&model_path)
        .expect("Failed to write GGUF model to file");
    assert!(written_bytes > 0);

    // 1. Validate with direct GgufParser
    let model = GgufParser::parse_from_file(&model_path).expect("GgufParser failed to read file");
    assert_eq!(model.architecture, ModelArchitecture::Llama);
    assert_eq!(model.num_layers, 2);
    assert_eq!(model.hidden_size, 64);
    assert_eq!(model.num_attention_heads, 4);
    assert_eq!(model.num_key_value_heads, 4);
    assert_eq!(model.max_position_embeddings, 512);
    assert_eq!(model.vocab_size, 1000);
    assert_eq!(model.tensors.len(), 6);

    // 2. Validate with universal ModelImporter
    let imported = ModelImporter::import(&model_path).expect("ModelImporter failed to import GGUF");
    assert_eq!(imported.architecture, ModelArchitecture::Llama);
    assert_eq!(imported.num_layers, 2);
    assert_eq!(imported.tensors.len(), 6);
    assert_eq!(imported.tensors[0].name, "token_embd.weight");
    assert_eq!(imported.tensors[0].shape, vec![1000, 64]);
}

#[test]
fn test_e2e_gguf_writer_quant_q8_0_roundtrip() {
    let mut writer = GgufWriter::new();
    writer.add_string("general.architecture", "llama");

    // Construct valid Q8_0 block: 2 bytes FP16 scale + 32 signed int8s = 34 bytes
    let scale_fp16 = 0x3C00_u16; // 1.0 in FP16
    let mut q8_payload = Vec::with_capacity(34);
    q8_payload.extend_from_slice(&scale_fp16.to_le_bytes());

    let original_i8s: Vec<i8> = (0..32).map(|i| (i as i8) - 16).collect();
    for &val in &original_i8s {
        q8_payload.push(val as u8);
    }

    writer.add_tensor(
        "blk.0.mlp.weight",
        vec![32],
        GgmlType::Q8_0,
        q8_payload.clone(),
    );

    let bytes = writer.write_to_bytes().expect("Failed to write Q8_0 GGUF");

    let model = GgufParser::parse_from_bytes(&bytes, Path::new("q8_model.gguf"))
        .expect("Parser failed to parse Q8_0 model");
    assert_eq!(model.tensors.len(), 1);
    assert_eq!(model.tensors[0].original_dtype, "Q8_0");
    assert_eq!(model.tensors[0].byte_size, 34);

    // Locate payload slice from parsed offset and dequantize with GgufBlockDecoder
    let mut reader = GgufReader::new(&bytes);
    reader.read_bytes(24).unwrap(); // Header
    let _key = reader.read_string().unwrap();
    let vtype = reader.read_u32().unwrap();
    let _val = reader.read_value(vtype).unwrap();

    let _t_name = reader.read_string().unwrap();
    let _n_dims = reader.read_u32().unwrap();
    let _dim = reader.read_u64().unwrap();
    let _gtype = reader.read_u32().unwrap();
    let offset = reader.read_u64().unwrap();

    let header_end = reader.cursor;
    let pad = (32 - (header_end % 32)) % 32;
    let data_start = header_end + pad;

    let payload_slice = &bytes[data_start + offset as usize..data_start + offset as usize + 34];
    assert_eq!(payload_slice, &q8_payload[..]);

    let unpacked = GgufBlockDecoder::unpack_q8_0(payload_slice).expect("Unpack Q8_0 failed");
    assert_eq!(unpacked.len(), 32);
    for i in 0..32 {
        let expected = (original_i8s[i] as f32) * 1.0;
        assert!(
            (unpacked[i] - expected).abs() < 1e-5,
            "Mismatch at index {}: got {}, expected {}",
            i,
            unpacked[i],
            expected
        );
    }
}

#[test]
fn test_e2e_bitnet_packed_tensor_export() {
    // 4 rows x 8 columns = 32 ternary weights
    let rows = 4;
    let cols = 8;
    let weights: Vec<f32> = vec![
        -1.0, 0.0, 1.0, 0.0, 1.0, -1.0, 0.0, 1.0, 0.0, 1.0, -1.0, -1.0, 0.0, 0.0, 1.0, -1.0, 1.0,
        1.0, 0.0, -1.0, 0.0, 1.0, -1.0, 0.0, -1.0, -1.0, 1.0, 0.0, 1.0, 0.0, 0.0, -1.0,
    ];

    let packed = BitNetPacker::pack_matrix("model.layers.0.mlp.gate", rows, cols, &weights);
    assert_eq!(packed.packed_byte_size(), 8); // 32 weights / 4 weights/byte = 8 bytes

    let mut writer = GgufWriter::new();
    writer.add_string("general.architecture", "bitnet");

    // Export packed ternary bitstream as I8 tensor
    let packed_cols = packed.packed_byte_size() / rows;
    let weight_tensor = GgufTensor::new(
        format!("{}.weight", packed.name),
        vec![rows, packed_cols],
        GgmlType::I8,
        packed.packed_data.clone(),
    );
    writer.add_tensor_entry(weight_tensor);

    // Export row scales as F32 tensor
    let mut scale_bytes = Vec::with_capacity(rows * 4);
    for &scale in &packed.row_scales {
        scale_bytes.extend_from_slice(&scale.to_le_bytes());
    }
    writer.add_tensor(
        format!("{}.scale", packed.name),
        vec![rows],
        GgmlType::F32,
        scale_bytes,
    );

    let bytes = writer
        .write_to_bytes()
        .expect("Failed to write BitNet GGUF");

    let model = GgufParser::parse_from_bytes(&bytes, Path::new("bitnet_test.gguf"))
        .expect("Failed to parse BitNet GGUF");
    assert_eq!(model.tensors.len(), 2);
    assert_eq!(model.tensors[0].name, "model.layers.0.mlp.gate.weight");
    assert_eq!(model.tensors[0].shape, vec![rows, packed_cols]);
    assert_eq!(model.tensors[0].byte_size, 8);

    // Unpack from parsed GGUF payload and verify exact ternary fidelity
    let mut reader = GgufReader::new(&bytes);
    reader.read_bytes(24).unwrap();
    let _key = reader.read_string().unwrap();
    let vtype = reader.read_u32().unwrap();
    let _val = reader.read_value(vtype).unwrap();

    let _t0_name = reader.read_string().unwrap();
    let _n0_dims = reader.read_u32().unwrap();
    let _d0_0 = reader.read_u64().unwrap();
    let _d0_1 = reader.read_u64().unwrap();
    let _g0_type = reader.read_u32().unwrap();
    let offset0 = reader.read_u64().unwrap();

    let _t1_name = reader.read_string().unwrap();
    let _n1_dims = reader.read_u32().unwrap();
    let _d1_0 = reader.read_u64().unwrap();
    let _g1_type = reader.read_u32().unwrap();
    let _offset1 = reader.read_u64().unwrap();

    let header_end = reader.cursor;
    let pad = (32 - (header_end % 32)) % 32;
    let data_start = header_end + pad;

    let extracted_packed = &bytes[data_start + offset0 as usize..data_start + offset0 as usize + 8];
    assert_eq!(extracted_packed, &packed.packed_data[..]);

    let recovered_packed = thme::exporter::bitnet_pack::PackedBitNetTensor {
        name: packed.name,
        rows,
        cols,
        row_scales: packed.row_scales,
        packed_data: extracted_packed.to_vec(),
    };

    let unpacked_ternary = BitNetPacker::unpack_raw_ternary(&recovered_packed);
    let original_ternary: Vec<i8> = weights.iter().map(|&w| w as i8).collect();
    assert_eq!(unpacked_ternary, original_ternary);
}

#[test]
fn test_e2e_fuzz_prefix_truncation_on_exported_gguf() {
    let mut writer = GgufWriter::new();
    writer.add_string("general.architecture", "llama");
    writer.add_u32("llama.block_count", 1);
    writer.add_tensor("w0", vec![16], GgmlType::I8, vec![42u8; 16]);

    let buffer = writer
        .write_to_bytes()
        .expect("Failed to serialize model for fuzzing");
    assert!(buffer.len() >= 32);

    // Truncate at every single byte index from 0 to buffer.len()
    for len in 0..buffer.len() {
        let truncated = &buffer[..len];
        // Parser must return Err on truncated input and must NEVER panic
        let _ = GgufParser::parse_from_bytes(truncated, Path::new("fuzz.gguf"));
    }
}
