use std::path::Path;
use thme::parser::gguf::{GgmlType, GgufBlockDecoder, GgufParser};

#[test]
fn test_ggml_quant_type_metadata() {
    assert_eq!(GgmlType::from_u32(0), GgmlType::F32);
    assert_eq!(GgmlType::from_u32(1), GgmlType::F16);
    assert_eq!(GgmlType::from_u32(2), GgmlType::Q4_0);
    assert_eq!(GgmlType::from_u32(3), GgmlType::Q4_1);
    assert_eq!(GgmlType::from_u32(6), GgmlType::Q5_0);
    assert_eq!(GgmlType::from_u32(8), GgmlType::Q8_0);
    assert_eq!(GgmlType::from_u32(12), GgmlType::Q4_K);
    assert_eq!(GgmlType::from_u32(23), GgmlType::IQ4_XS);

    assert_eq!(GgmlType::Q4_0.block_size(), 32);
    assert_eq!(GgmlType::Q4_0.type_size(), 18);
    assert!((GgmlType::Q4_0.bytes_per_weight() - (18.0 / 32.0)).abs() < 1e-6);

    assert_eq!(GgmlType::Q8_0.block_size(), 32);
    assert_eq!(GgmlType::Q8_0.type_size(), 34);
    assert!((GgmlType::Q8_0.bytes_per_weight() - (34.0 / 32.0)).abs() < 1e-6);

    assert_eq!(GgmlType::Q4_K.block_size(), 256);
    assert_eq!(GgmlType::Q4_K.type_size(), 144);

    assert_eq!(GgmlType::IQ4_XS.block_size(), 256);
    assert_eq!(GgmlType::IQ4_XS.type_size(), 136);

    assert!(GgmlType::Q4_0.is_quantized());
    assert!(!GgmlType::F32.is_quantized());
    assert!(!GgmlType::BF16.is_quantized());
}

#[test]
fn test_q8_0_block_unpacking() {
    let mut block = [0u8; 34];
    // scale = 1.0 (fp16 0x3C00)
    let scale_fp16 = 0x3C00_u16;
    block[0..2].copy_from_slice(&scale_fp16.to_le_bytes());

    for i in 0..32 {
        block[2 + i] = (i as i8 - 16) as u8;
    }

    let weights = GgufBlockDecoder::unpack_q8_0(&block).expect("Unpacking failed");
    assert_eq!(weights.len(), 32);
    for (i, &w) in weights.iter().enumerate() {
        let expected = (i as i8 - 16) as f32 * 1.0;
        assert!(
            (w - expected).abs() < 1e-5,
            "Mismatch at index {}: got {}, expected {}",
            i,
            w,
            expected
        );
    }
}

#[test]
fn test_q4_0_block_unpacking() {
    let mut block = [0u8; 18];
    // scale = 2.0 (fp16 0x4000)
    let scale_fp16 = 0x4000_u16;
    block[0..2].copy_from_slice(&scale_fp16.to_le_bytes());

    // Byte pattern: low nibble = 8 (val = 0), high nibble = 12 (val = 4)
    // byte = 0xC8
    for i in 0..16 {
        block[2 + i] = 0xC8;
    }

    let weights = GgufBlockDecoder::unpack_q4_0(&block).expect("Unpacking failed");
    assert_eq!(weights.len(), 32);
    for i in 0..16 {
        assert!((weights[i] - 0.0).abs() < 1e-5);
        assert!((weights[i + 16] - 8.0).abs() < 1e-5); // (12 - 8) * 2.0 = 8.0
    }
}

#[test]
fn test_q4_1_block_unpacking() {
    let mut block = [0u8; 20];
    // scale = 0.5 (fp16 0x3800), min = -2.0 (fp16 0xC000)
    block[0..2].copy_from_slice(&0x3800_u16.to_le_bytes());
    block[2..4].copy_from_slice(&0xC000_u16.to_le_bytes());

    for i in 0..16 {
        // low = 4, high = 10 -> byte = 0xA4
        block[4 + i] = 0xA4;
    }

    let weights = GgufBlockDecoder::unpack_q4_1(&block).expect("Unpacking failed");
    assert_eq!(weights.len(), 32);
    for i in 0..16 {
        assert!((weights[i] - (0.5 * 4.0 - 2.0)).abs() < 1e-4);
        assert!((weights[i + 16] - (0.5 * 10.0 - 2.0)).abs() < 1e-4);
    }
}

#[test]
fn test_q5_0_block_unpacking() {
    let mut block = [0u8; 22];
    // scale = 1.0 (fp16 0x3C00)
    block[0..2].copy_from_slice(&0x3C00_u16.to_le_bytes());
    // qh (high bits) = 0
    block[2..6].copy_from_slice(&0u32.to_le_bytes());
    // low 4 bits = 16 (0x10) -> (16 - 16) * 1.0 = 0.0
    for i in 0..16 {
        block[6 + i] = 0x00; // nibbles 0, so (0 - 16) = -16
    }

    let weights = GgufBlockDecoder::unpack_q5_0(&block).expect("Unpacking failed");
    assert_eq!(weights.len(), 32);
    for &w in &weights {
        assert!((w - (-16.0)).abs() < 1e-4);
    }
}

#[test]
fn test_quantized_gguf_parsing_with_weight_profile() {
    let mut buf = Vec::new();
    buf.extend_from_slice(b"GGUF");
    buf.extend_from_slice(&3u32.to_le_bytes()); // Version 3
    buf.extend_from_slice(&1u64.to_le_bytes()); // 1 tensor
    buf.extend_from_slice(&1u64.to_le_bytes()); // 1 metadata

    let key = "general.architecture";
    buf.extend_from_slice(&(key.len() as u64).to_le_bytes());
    buf.extend_from_slice(key.as_bytes());
    buf.extend_from_slice(&8u32.to_le_bytes());
    let val = "llama";
    buf.extend_from_slice(&(val.len() as u64).to_le_bytes());
    buf.extend_from_slice(val.as_bytes());

    // Tensor: Q4_0 quantized, shape [32]
    let t_name = "model.layers.0.mlp.down_proj.weight";
    buf.extend_from_slice(&(t_name.len() as u64).to_le_bytes());
    buf.extend_from_slice(t_name.as_bytes());
    buf.extend_from_slice(&1u32.to_le_bytes()); // 1 dim
    buf.extend_from_slice(&32u64.to_le_bytes());
    buf.extend_from_slice(&2u32.to_le_bytes()); // Type 2 = Q4_0
    buf.extend_from_slice(&0u64.to_le_bytes()); // offset 0

    // Pad to 32 bytes alignment
    let header_len = buf.len();
    let pad = (32 - (header_len % 32)) % 32;
    buf.extend(vec![0u8; pad]);

    // Q4_0 payload (18 bytes)
    let scale_fp16 = 0x3C00_u16; // 1.0
    buf.extend_from_slice(&scale_fp16.to_le_bytes());
    buf.extend(vec![0x88u8; 16]); // all nibbles 8 -> values 0.0

    let model =
        GgufParser::parse_from_bytes(&buf, Path::new("test_q4.gguf")).expect("GGUF parse failed");
    assert_eq!(model.tensors.len(), 1);
    assert_eq!(model.tensors[0].original_dtype, "Q4_0");
    assert_eq!(model.tensors[0].byte_size, 18);
    let wp = model.tensors[0].weight_profile.as_ref().unwrap();
    // validated sample size
    assert!(wp.mean_val.abs() < 1e-4);
}
