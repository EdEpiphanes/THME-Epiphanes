// SPDX-License-Identifier: AGPL-3.0-only
use std::path::Path;
use thme::parser::gguf::GgufParser;
use thme::parser::safetensors::SafetensorsScanner;

#[test]
fn test_fuzz_gguf_truncated_stream_safety() {
    // Generate valid minimal GGUF buffer
    let mut valid_gguf = Vec::new();
    valid_gguf.extend_from_slice(b"GGUF");
    valid_gguf.extend_from_slice(&3u32.to_le_bytes());
    valid_gguf.extend_from_slice(&2u64.to_le_bytes()); // 2 tensors
    valid_gguf.extend_from_slice(&1u64.to_le_bytes()); // 1 metadata

    let key = "general.architecture";
    valid_gguf.extend_from_slice(&(key.len() as u64).to_le_bytes());
    valid_gguf.extend_from_slice(key.as_bytes());
    valid_gguf.extend_from_slice(&8u32.to_le_bytes());
    let val = "llama";
    valid_gguf.extend_from_slice(&(val.len() as u64).to_le_bytes());
    valid_gguf.extend_from_slice(val.as_bytes());

    // Tensor 1
    let t1 = "blk.0.q.weight";
    valid_gguf.extend_from_slice(&(t1.len() as u64).to_le_bytes());
    valid_gguf.extend_from_slice(t1.as_bytes());
    valid_gguf.extend_from_slice(&1u32.to_le_bytes());
    valid_gguf.extend_from_slice(&32u64.to_le_bytes());
    valid_gguf.extend_from_slice(&0u32.to_le_bytes()); // F32
    valid_gguf.extend_from_slice(&0u64.to_le_bytes());

    // Tensor 2
    let t2 = "blk.0.k.weight";
    valid_gguf.extend_from_slice(&(t2.len() as u64).to_le_bytes());
    valid_gguf.extend_from_slice(t2.as_bytes());
    valid_gguf.extend_from_slice(&1u32.to_le_bytes());
    valid_gguf.extend_from_slice(&32u64.to_le_bytes());
    valid_gguf.extend_from_slice(&0u32.to_le_bytes());
    valid_gguf.extend_from_slice(&128u64.to_le_bytes());

    let full_len = valid_gguf.len();

    // 1. Prefix Truncation Fuzzing: Feed every possible slice [0..k]
    for k in 0..=full_len {
        let truncated = &valid_gguf[0..k];
        // Parser MUST NOT panic under any prefix truncation
        let _ = GgufParser::parse_from_bytes(truncated, Path::new("truncated.gguf"));
    }

    // 2. Deterministic Bit Mutation Fuzzing (1,000 iterations)
    let mut mutated = valid_gguf.clone();
    let mut prng_state = 0x12345678_u64;
    for i in 0..1000 {
        // Simple LCG PRNG
        prng_state = prng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let byte_idx = (prng_state as usize) % full_len;
        let bit_flip = 1u8 << ((prng_state >> 32) % 8);
        mutated[byte_idx] ^= bit_flip;

        // Parser MUST NOT panic on bit flips
        let _ = GgufParser::parse_from_bytes(&mutated, Path::new(&format!("mutated_{}.gguf", i)));
    }
}

#[test]
fn test_fuzz_safetensors_malformed_slice_safety() {
    // 1. Extreme short lengths
    for k in 0..8 {
        let short_data = vec![0u8; k];
        let res = SafetensorsScanner::parse_metadata_from_slice(&short_data);
        assert!(res.is_err(), "Short buffer must return an error gracefully");
    }

    // 2. Giant header length claiming more than file size
    let mut giant_header = vec![0u8; 128];
    let bad_len: u64 = 0xFFFFFFFFFFFFFFFF;
    giant_header[0..8].copy_from_slice(&bad_len.to_le_bytes());
    let res = SafetensorsScanner::parse_metadata_from_slice(&giant_header);
    assert!(
        res.is_err(),
        "Unbounded header length must be rejected safely"
    );

    // 3. Corrupt JSON payload
    let mut invalid_json = vec![0u8; 64];
    let header_len = 32_u64;
    invalid_json[0..8].copy_from_slice(&header_len.to_le_bytes());
    invalid_json[8..40].fill(b'x');
    let res = SafetensorsScanner::parse_metadata_from_slice(&invalid_json);
    assert!(
        res.is_err(),
        "Garbage header must return structured error without panic"
    );
}
