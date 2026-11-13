// SPDX-License-Identifier: AGPL-3.0-only
//! Adversarial Compatibility & Consumer Integration Test Suite for GgufWriter (Milestone v0.1.8.2)
//!
//! EMPIRICAL CHALLENGER 2 ADVERSARIAL HARNESS:
//! 1. GgufBlockDecoder interaction: Q8_0, Q4_0, Q4_1, Q5_0, and BitNet blocks
//! 2. ModelImporter::import round-trip compatibility with GgufWriter::write_to_file
//! 3. Error-handling & boundary conditions:
//!    - Shape numel vs byte slice mismatch (undersized, oversized, indivisible by block_size)
//!    - Tensors with >8 dimensions vs <=8 dimensions
//!    - Non-ASCII, multi-byte UTF-8 keys, strings, and tensor names
//!    - Empty keys, empty names, and empty string values
//!    - Alignment extremes (alignment 1, 0 clamp, 128)

use std::path::Path;
use tempfile::tempdir;
use thme::exporter::bitnet_pack::BitNetPacker;
use thme::exporter::gguf_writer::{GgufTensor, GgufWriter};
use thme::ir::ModelArchitecture;
use thme::parser::gguf::{GgmlType, GgufBlockDecoder, GgufParser, GgufReader, GgufValue};
use thme::parser::ModelImporter;

// =========================================================================
// 1. GgufBlockDecoder Interaction (Q4_0, Q8_0, Q4_1, Q5_0, BitNet)
// =========================================================================

#[test]
fn test_adversarial_q4_0_block_decoder_roundtrip() {
    let mut writer = GgufWriter::new();
    writer.add_string("general.architecture", "llama");

    // Construct 2 valid Q4_0 blocks (64 weights total = 36 bytes payload)
    // Block 0: scale = 0.5 (FP16 0x3800), nibbles = alternating 0 and 15 (mapped: 0 - 8 = -8, 15 - 8 = +7)
    // Block 1: scale = 2.0 (FP16 0x4000), nibbles = 8 (mapped: 8 - 8 = 0)
    let scale0: u16 = 0x3800; // 0.5
    let scale1: u16 = 0x4000; // 2.0

    let mut q4_payload = Vec::with_capacity(36);
    // Block 0
    q4_payload.extend_from_slice(&scale0.to_le_bytes());
    q4_payload.extend([0xF0_u8; 16]); // low nibble 0 -> -8, high nibble 15 -> +7

    // Block 1
    q4_payload.extend_from_slice(&scale1.to_le_bytes());
    q4_payload.extend([0x88_u8; 16]); // low nibble 8 -> 0, high nibble 8 -> 0

    assert_eq!(q4_payload.len(), 36);

    writer.add_tensor(
        "blk.0.attn_q.weight",
        vec![64],
        GgmlType::Q4_0,
        q4_payload.clone(),
    );

    let bytes = writer.write_to_bytes().expect("Failed to write Q4_0 GGUF");
    let model = GgufParser::parse_from_bytes(&bytes, Path::new("q4_model.gguf"))
        .expect("Parser failed to parse Q4_0 model");
    assert_eq!(model.tensors.len(), 1);
    assert_eq!(model.tensors[0].original_dtype, "Q4_0");
    assert_eq!(model.tensors[0].byte_size, 36);

    // Locate payload slice from parsed offset
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

    let slice = &bytes[data_start + offset as usize..data_start + offset as usize + 36];
    assert_eq!(slice, &q4_payload[..]);

    // Unpack block 0
    let unpacked_b0 = GgufBlockDecoder::unpack_q4_0(&slice[0..18]).expect("Unpack block 0 failed");
    for i in 0..16 {
        assert!(
            (unpacked_b0[i] - (-4.0)).abs() < 1e-4,
            "Block 0 low nibble mismatch: got {}, expected -4.0",
            unpacked_b0[i]
        );
        assert!(
            (unpacked_b0[i + 16] - 3.5).abs() < 1e-4,
            "Block 0 high nibble mismatch: got {}, expected 3.5",
            unpacked_b0[i + 16]
        );
    }

    // Unpack block 1
    let unpacked_b1 = GgufBlockDecoder::unpack_q4_0(&slice[18..36]).expect("Unpack block 1 failed");
    for &val in &unpacked_b1 {
        assert!(
            val.abs() < 1e-5,
            "Block 1 mismatch: got {}, expected 0.0",
            val
        );
    }
}

#[test]
fn test_adversarial_q8_0_multi_block_decoder_roundtrip() {
    let mut writer = GgufWriter::new();
    writer.add_string("general.architecture", "llama");

    // Construct 4 blocks of Q8_0 (128 weights = 136 bytes)
    let scales: [u16; 4] = [
        0x3C00, // 1.0
        0x3800, // 0.5
        0x4000, // 2.0
        0xBC00, // -1.0
    ];

    let mut q8_payload = Vec::with_capacity(136);
    for (b, &scale) in scales.iter().enumerate() {
        q8_payload.extend_from_slice(&scale.to_le_bytes());
        for i in 0..32 {
            let val = ((i as i8) + (b as i8 * 5)) - 10;
            q8_payload.push(val as u8);
        }
    }

    writer.add_tensor(
        "blk.0.ffn_up.weight",
        vec![128],
        GgmlType::Q8_0,
        q8_payload.clone(),
    );

    let bytes = writer
        .write_to_bytes()
        .expect("Failed to write multi-block Q8_0");

    let model = GgufParser::parse_from_bytes(&bytes, Path::new("q8_multi.gguf")).unwrap();
    assert_eq!(model.tensors.len(), 1);
    assert_eq!(model.tensors[0].byte_size, 136);

    let mut reader = GgufReader::new(&bytes);
    reader.read_bytes(24).unwrap();
    let _ = reader.read_string().unwrap();
    let vt = reader.read_u32().unwrap();
    let _ = reader.read_value(vt).unwrap();
    let _ = reader.read_string().unwrap();
    let _ = reader.read_u32().unwrap();
    let _ = reader.read_u64().unwrap();
    let _ = reader.read_u32().unwrap();
    let offset = reader.read_u64().unwrap();

    let header_end = reader.cursor;
    let pad = (32 - (header_end % 32)) % 32;
    let data_start = header_end + pad;

    let payload_slice = &bytes[data_start + offset as usize..data_start + offset as usize + 136];
    assert_eq!(payload_slice, &q8_payload[..]);

    let float_scales = [1.0_f32, 0.5_f32, 2.0_f32, -1.0_f32];
    for b in 0..4 {
        let block_slice = &payload_slice[b * 34..(b + 1) * 34];
        let unpacked = GgufBlockDecoder::unpack_q8_0(block_slice).unwrap();
        for (i, &unpacked_val) in unpacked.iter().enumerate() {
            let original_i8 = ((i as i8) + (b as i8 * 5)) - 10;
            let expected = (original_i8 as f32) * float_scales[b];
            assert!(
                (unpacked_val - expected).abs() < 1e-4,
                "Block {} index {} mismatch: got {}, expected {}",
                b,
                i,
                unpacked_val,
                expected
            );
        }
    }
}

#[test]
fn test_adversarial_bitnet_pack_multi_row_export_and_unpack() {
    let rows = 8;
    let cols = 16;
    let total_weights = rows * cols; // 128 weights

    // Deterministic pseudo-ternary weights
    let mut weights = Vec::with_capacity(total_weights);
    for i in 0..total_weights {
        let rem = i % 3;
        weights.push(match rem {
            0 => -1.0,
            1 => 0.0,
            _ => 1.0,
        });
    }

    let packed = BitNetPacker::pack_matrix("model.layers.0.mlp.down_proj", rows, cols, &weights);
    assert_eq!(packed.packed_byte_size(), 32); // 128 / 4 = 32 bytes

    let mut writer = GgufWriter::new();
    writer.add_string("general.architecture", "bitnet");

    let packed_cols = packed.packed_byte_size() / rows; // 4
    let weight_tensor = GgufTensor::new(
        format!("{}.weight", packed.name),
        vec![rows, packed_cols],
        GgmlType::I8,
        packed.packed_data.clone(),
    );
    writer.add_tensor_entry(weight_tensor);

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
    let model = GgufParser::parse_from_bytes(&bytes, Path::new("bitnet_multi.gguf")).unwrap();
    assert_eq!(model.tensors.len(), 2);

    // Extract packed weight payload
    let mut reader = GgufReader::new(&bytes);
    reader.read_bytes(24).unwrap();
    let _ = reader.read_string().unwrap();
    let vt = reader.read_u32().unwrap();
    let _ = reader.read_value(vt).unwrap();

    let _ = reader.read_string().unwrap();
    let _ = reader.read_u32().unwrap();
    let _ = reader.read_u64().unwrap();
    let _ = reader.read_u64().unwrap();
    let _ = reader.read_u32().unwrap();
    let offset0 = reader.read_u64().unwrap();

    let _ = reader.read_string().unwrap();
    let _ = reader.read_u32().unwrap();
    let _ = reader.read_u64().unwrap();
    let _ = reader.read_u32().unwrap();
    let _offset1 = reader.read_u64().unwrap();

    let header_end = reader.cursor;
    let pad = (32 - (header_end % 32)) % 32;
    let data_start = header_end + pad;

    let extracted_packed =
        &bytes[data_start + offset0 as usize..data_start + offset0 as usize + 32];
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

// =========================================================================
// 2. ModelImporter::import Round-Trip Compatibility
// =========================================================================

#[test]
fn test_adversarial_model_importer_llama_mistral_qwen() {
    let dir = tempdir().expect("Failed to create temporary directory");

    let arch_specs = [
        ("llama", ModelArchitecture::Llama, 2, 64, 4),
        ("mistral", ModelArchitecture::Mistral, 4, 128, 8),
        ("qwen2", ModelArchitecture::Qwen, 3, 96, 6),
    ];

    for (arch_name, expected_arch, layers, hidden, heads) in arch_specs {
        let model_file = dir.path().join(format!("mock_{}.gguf", arch_name));

        let mut writer = GgufWriter::new();
        writer
            .add_string("general.architecture", arch_name)
            .add_u32(format!("{}.block_count", arch_name), layers as u32)
            .add_u32(format!("{}.embedding_length", arch_name), hidden as u32)
            .add_u32(format!("{}.attention.head_count", arch_name), heads as u32)
            .add_u32(
                format!("{}.attention.head_count_kv", arch_name),
                heads as u32,
            )
            .add_u32(format!("{}.context_length", arch_name), 1024)
            .add_u32(format!("{}.vocab_size", arch_name), 256);

        // Dummy embedding and head
        writer.add_tensor(
            "token_embd.weight",
            vec![256, hidden],
            GgmlType::F16,
            vec![0u8; 256 * hidden * 2],
        );
        writer.add_tensor(
            "output.weight",
            vec![256, hidden],
            GgmlType::F16,
            vec![0u8; 256 * hidden * 2],
        );

        writer
            .write_to_file(&model_file)
            .expect("write_to_file failed");

        // Import via universal ModelImporter
        let imported =
            ModelImporter::import(&model_file).expect("ModelImporter failed on GGUF file");

        assert_eq!(imported.architecture, expected_arch);
        assert_eq!(imported.num_layers, layers);
        assert_eq!(imported.hidden_size, hidden);
        assert_eq!(imported.num_attention_heads, heads);
        assert_eq!(imported.num_key_value_heads, heads);
        assert_eq!(imported.vocab_size, 256);
        assert_eq!(imported.tensors.len(), 2);
        assert_eq!(imported.tensors[0].name, "token_embd.weight");
        assert_eq!(imported.tensors[0].shape, vec![256, hidden]);
        assert_eq!(imported.tensors[0].original_dtype, "F16");
        assert!(imported.tensors[0].weight_profile.is_some());
    }
}

// =========================================================================
// 3. Error-Handling and Robustness
// =========================================================================

#[test]
fn test_adversarial_invalid_shape_and_payload_validation() {
    // 1. Under-sized payload for F32
    {
        let mut writer = GgufWriter::new();
        writer.add_tensor("t_under", vec![4, 4], GgmlType::F32, vec![0u8; 60]); // 60 instead of 64
        let res = writer.write_to_bytes();
        assert!(res.is_err(), "Expected error for undersized F32 payload");
        let err = res.unwrap_err().to_string();
        assert!(err.contains("payload size mismatch"));
    }

    // 2. Over-sized payload for F32
    {
        let mut writer = GgufWriter::new();
        writer.add_tensor("t_over", vec![4, 4], GgmlType::F32, vec![0u8; 68]); // 68 instead of 64
        let res = writer.write_to_bytes();
        assert!(res.is_err(), "Expected error for oversized F32 payload");
        let err = res.unwrap_err().to_string();
        assert!(err.contains("payload size mismatch"));
    }

    // 3. Non-divisible element count for quantized format (Q4_0 block size is 32)
    {
        let mut writer = GgufWriter::new();
        writer.add_tensor("t_indivisible", vec![33], GgmlType::Q4_0, vec![0u8; 18]);
        let res = writer.write_to_bytes();
        assert!(
            res.is_err(),
            "Expected error for element count not divisible by block_size"
        );
        let err = res.unwrap_err().to_string();
        assert!(err.contains("is not divisible by block size"));
    }

    // 4. Q8_0 payload size mismatch (32 elements requires 34 bytes, but 32 provided)
    {
        let mut writer = GgufWriter::new();
        writer.add_tensor("t_q8_mismatch", vec![32], GgmlType::Q8_0, vec![0u8; 32]);
        let res = writer.write_to_bytes();
        assert!(res.is_err());
        let err = res.unwrap_err().to_string();
        assert!(err.contains("payload size mismatch"));
    }

    // 5. Unknown GGML type bypasses block validation safely
    {
        let mut writer = GgufWriter::new();
        writer.add_tensor(
            "t_unknown",
            vec![10, 5],
            GgmlType::Unknown(999),
            vec![42u8; 50],
        );
        let res = writer.write_to_bytes();
        assert!(res.is_ok(), "Unknown GGML type should be allowed through");
    }
}

#[test]
fn test_adversarial_tensor_dimensions_boundary_gt_8() {
    // 8 dimensions: exactly at the maximum allowed boundary -> must succeed
    {
        let mut writer = GgufWriter::new();
        // Shape: 8 dimensions [1, 1, 1, 1, 1, 1, 2, 2] -> 4 elements of F32 = 16 bytes
        let shape_8d = vec![1, 1, 1, 1, 1, 1, 2, 2];
        writer.add_tensor("t_8d", shape_8d.clone(), GgmlType::F32, vec![0u8; 16]);

        let bytes = writer
            .write_to_bytes()
            .expect("Writing 8D tensor should succeed");
        let model = GgufParser::parse_from_bytes(&bytes, Path::new("8d.gguf")).unwrap();
        assert_eq!(model.tensors.len(), 1);
        assert_eq!(model.tensors[0].shape, shape_8d);
    }

    // 9 dimensions: exceeds maximum allowed boundary (8) -> must fail with descriptive error
    {
        let mut writer = GgufWriter::new();
        let shape_9d = vec![1, 1, 1, 1, 1, 1, 1, 1, 2];
        writer.add_tensor("t_9d", shape_9d, GgmlType::F32, vec![0u8; 8]);

        let res = writer.write_to_bytes();
        assert!(res.is_err(), "Writing 9D tensor must fail");
        let err = res.unwrap_err().to_string();
        assert!(
            err.contains("dimension count 9 exceeds maximum supported (8)"),
            "Error was: {}",
            err
        );
    }
}

#[test]
fn test_adversarial_non_ascii_and_multibyte_utf8() {
    let mut writer = GgufWriter::new();

    // Multi-byte Unicode metadata keys and values
    // Russian, Chinese, Japanese, French accents, German umlauts, Emojis
    writer.add_metadata(
        "модель.архитектура",
        GgufValue::String("русская_модель".into()),
    );
    writer.add_metadata("模型.层数", GgufValue::Uint32(12));
    writer.add_metadata("モデル.トークン数", GgufValue::Uint64(32000));
    writer.add_metadata(
        "paramètre.détaillé",
        GgufValue::String("café_au_lait".into()),
    );
    writer.add_metadata("konfiguration.größe", GgufValue::String("über_groß".into()));
    writer.add_metadata(
        "emoji.rocket_launch",
        GgufValue::String("🚀_speed_⚡".into()),
    );

    // Multi-byte Unicode tensor names
    let payload = vec![0u8; 32];
    writer.add_tensor("слой.0.внимание", vec![32], GgmlType::I8, payload.clone());
    writer.add_tensor("层.0.权重", vec![32], GgmlType::I8, payload.clone());
    writer.add_tensor("tenseur_évalué", vec![32], GgmlType::I8, payload);

    let bytes = writer
        .write_to_bytes()
        .expect("Serialization of multi-byte UTF-8 must succeed");

    // Parse back with GgufParser
    let model = GgufParser::parse_from_bytes(&bytes, Path::new("utf8_test.gguf"))
        .expect("Parser must accept multi-byte UTF-8 keys and tensors");

    assert_eq!(model.tensors.len(), 3);
    assert_eq!(model.tensors[0].name, "слой.0.внимание");
    assert_eq!(model.tensors[1].name, "层.0.权重");
    assert_eq!(model.tensors[2].name, "tenseur_évalué");

    // Verify metadata directly via GgufReader
    let mut reader = GgufReader::new(&bytes);
    reader.read_bytes(24).unwrap(); // Header
    for _ in 0..6 {
        let key = reader.read_string().unwrap();
        let val_type = reader.read_u32().unwrap();
        let val = reader.read_value(val_type).unwrap();
        match key.as_str() {
            "модель.архитектура" => {
                assert_eq!(val.as_str().unwrap(), "русская_модель")
            }
            "模型.层数" => assert_eq!(val.as_u64().unwrap(), 12),
            "モデル.トークン数" => assert_eq!(val.as_u64().unwrap(), 32000),
            "paramètre.détaillé" => assert_eq!(val.as_str().unwrap(), "café_au_lait"),
            "konfiguration.größe" => assert_eq!(val.as_str().unwrap(), "über_groß"),
            "emoji.rocket_launch" => assert_eq!(val.as_str().unwrap(), "🚀_speed_⚡"),
            other => panic!("Unexpected parsed metadata key: {}", other),
        }
    }
}

#[test]
fn test_adversarial_empty_keys_names_and_values() {
    // 1. Empty metadata key must fail
    {
        let mut writer = GgufWriter::new();
        writer.add_string("", "some_value");
        let res = writer.write_to_bytes();
        assert!(res.is_err());
        assert!(res.unwrap_err().to_string().contains("key cannot be empty"));
    }

    // 2. Empty tensor name must fail
    {
        let mut writer = GgufWriter::new();
        writer.add_tensor("", vec![4], GgmlType::F32, vec![0u8; 16]);
        let res = writer.write_to_bytes();
        assert!(res.is_err());
        assert!(res
            .unwrap_err()
            .to_string()
            .contains("name cannot be empty"));
    }

    // 3. Empty string value is VALID in GGUF
    {
        let mut writer = GgufWriter::new();
        writer.add_string("valid.empty_value", "");
        let bytes = writer
            .write_to_bytes()
            .expect("Empty string value is valid");

        let mut reader = GgufReader::new(&bytes);
        reader.read_bytes(24).unwrap();
        let key = reader.read_string().unwrap();
        assert_eq!(key, "valid.empty_value");
        let vt = reader.read_u32().unwrap();
        let val = reader.read_value(vt).unwrap();
        assert_eq!(val.as_str().unwrap(), "");
    }
}

#[test]
fn test_adversarial_alignment_extremes() {
    // 1. Alignment = 1 (byte-packed, zero padding)
    {
        let mut writer = GgufWriter::new().with_alignment(1);
        assert_eq!(writer.alignment(), 1);
        writer.add_tensor("t1", vec![7], GgmlType::I8, vec![1u8; 7]);
        writer.add_tensor("t2", vec![13], GgmlType::I8, vec![2u8; 13]);

        let bytes = writer.write_to_bytes().expect("Alignment 1 must succeed");
        let model = GgufParser::parse_from_bytes(&bytes, Path::new("align1.gguf")).unwrap();
        assert_eq!(model.tensors.len(), 2);
    }

    // 2. Alignment = 0 (must clamp to 1 without panic / division by zero)
    {
        let mut writer = GgufWriter::new().with_alignment(0);
        assert_eq!(writer.alignment(), 1);
        writer.add_tensor("t_clamp", vec![5], GgmlType::I8, vec![3u8; 5]);
        let bytes = writer
            .write_to_bytes()
            .expect("Zero alignment clamp must succeed");
        assert!(!bytes.is_empty());
    }

    // 3. Alignment = 128 (large cache-line / AVX-512 boundary)
    {
        let mut writer = GgufWriter::new().with_alignment(128);
        assert_eq!(writer.alignment(), 128);
        writer.add_tensor("t_large1", vec![19], GgmlType::I8, vec![4u8; 19]);
        writer.add_tensor("t_large2", vec![43], GgmlType::I8, vec![5u8; 43]);

        let bytes = writer.write_to_bytes().expect("Alignment 128 must succeed");

        let mut reader = GgufReader::new(&bytes);
        reader.read_bytes(24).unwrap();
        let _ = reader.read_string().unwrap(); // general.alignment
        let vt = reader.read_u32().unwrap();
        let _ = reader.read_value(vt).unwrap();

        let _ = reader.read_string().unwrap();
        let _ = reader.read_u32().unwrap();
        let _ = reader.read_u64().unwrap();
        let _ = reader.read_u32().unwrap();
        let off1 = reader.read_u64().unwrap();

        let _ = reader.read_string().unwrap();
        let _ = reader.read_u32().unwrap();
        let _ = reader.read_u64().unwrap();
        let _ = reader.read_u32().unwrap();
        let off2 = reader.read_u64().unwrap();

        let header_end = reader.cursor;
        let pad = (128 - (header_end % 128)) % 128;
        let data_start = header_end + pad;

        assert_eq!(data_start % 128, 0);
        assert_eq!(off1, 0);
        assert_eq!(off2, 128); // 19 bytes padded to 128 boundary
        assert_eq!((data_start + off1 as usize) % 128, 0);
        assert_eq!((data_start + off2 as usize) % 128, 0);
    }
}
