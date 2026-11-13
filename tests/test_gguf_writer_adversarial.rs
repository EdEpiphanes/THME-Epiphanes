// SPDX-License-Identifier: AGPL-3.0-only
//! Adversarial Stress & Empirical Challenge Test Suite for Pure-Rust GGUF v3 Binary Writer (v0.1.8.2 / v0.1.8.3)
//!
//! EMPIRICAL CHALLENGER 1 ADVERSARIAL HARNESS:
//! 1. Byte alignment boundaries & 0x00 padding verification
//! 2. Magic bytes (b"GGUF" / 0x46554747 LE) and version (3u32) constants
//! 3. Column-major dimension reversal on 1D, 2D, 3D, and 4D tensors via GgufParser roundtrip
//! 4. Comprehensive roundtrip of all 13 metadata types with boundary values
//! 5. Boundary conditions: zero tensors, zero metadata, custom alignments (16, 64, 128, 256), and failure cases

use std::path::Path;
use tempfile::tempdir;
use thme::exporter::gguf_writer::{gguf_value_type_id, GgufWriter};
use thme::parser::gguf::{GgmlType, GgufParser, GgufReader, GgufValue};

fn assert_gguf_values_identical(a: &GgufValue, b: &GgufValue) {
    match (a, b) {
        (GgufValue::Uint8(x), GgufValue::Uint8(y)) => assert_eq!(x, y),
        (GgufValue::Int8(x), GgufValue::Int8(y)) => assert_eq!(x, y),
        (GgufValue::Uint16(x), GgufValue::Uint16(y)) => assert_eq!(x, y),
        (GgufValue::Int16(x), GgufValue::Int16(y)) => assert_eq!(x, y),
        (GgufValue::Uint32(x), GgufValue::Uint32(y)) => assert_eq!(x, y),
        (GgufValue::Int32(x), GgufValue::Int32(y)) => assert_eq!(x, y),
        (GgufValue::Float32(x), GgufValue::Float32(y)) => {
            if x.is_nan() && y.is_nan() {
                // Both NaN
            } else {
                assert!(
                    (x - y).abs() < 1e-6,
                    "Float32 mismatch: got {x}, expected {y}"
                );
            }
        }
        (GgufValue::Bool(x), GgufValue::Bool(y)) => assert_eq!(x, y),
        (GgufValue::String(x), GgufValue::String(y)) => assert_eq!(x, y),
        (GgufValue::Array(x), GgufValue::Array(y)) => {
            assert_eq!(
                x.len(),
                y.len(),
                "Array lengths differ: {} vs {}",
                x.len(),
                y.len()
            );
            for (xi, yi) in x.iter().zip(y.iter()) {
                assert_gguf_values_identical(xi, yi);
            }
        }
        (GgufValue::Uint64(x), GgufValue::Uint64(y)) => assert_eq!(x, y),
        (GgufValue::Int64(x), GgufValue::Int64(y)) => assert_eq!(x, y),
        (GgufValue::Float64(x), GgufValue::Float64(y)) => {
            if x.is_nan() && y.is_nan() {
                // Both NaN
            } else {
                assert!(
                    (x - y).abs() < 1e-12,
                    "Float64 mismatch: got {x}, expected {y}"
                );
            }
        }
        _ => panic!("Type mismatch: {:?} vs {:?}", a, b),
    }
}

// -----------------------------------------------------------------------------
// CHALLENGE 1: Byte alignment boundaries & 0x00 padding verification
// -----------------------------------------------------------------------------
#[test]
fn test_adversarial_byte_alignment_and_zero_padding() {
    for alignment in [16, 32, 64, 128] {
        let mut writer = GgufWriter::new().with_alignment(alignment);
        writer.add_string("general.architecture", "llama");
        writer.add_u32("llama.block_count", 1);

        // Three tensors with irregular / prime-numbered byte counts using I8 (1 byte per element)
        // Sizes: 11 bytes, 29 bytes, 43 bytes
        let t1_payload: Vec<u8> = (0..11).map(|x| (x * 7 + 1) as u8).collect();
        let t2_payload: Vec<u8> = (0..29).map(|x| (x * 13 + 3) as u8).collect();
        let t3_payload: Vec<u8> = (0..43).map(|x| (x * 17 + 5) as u8).collect();

        writer.add_tensor("t1_prime11", vec![11], GgmlType::I8, t1_payload.clone());
        writer.add_tensor("t2_prime29", vec![29], GgmlType::I8, t2_payload.clone());
        writer.add_tensor("t3_prime43", vec![43], GgmlType::I8, t3_payload.clone());

        let bytes = writer
            .write_to_bytes()
            .unwrap_or_else(|e| panic!("Failed to serialize with alignment {alignment}: {e}"));

        // Parse with raw GgufReader to find header_end exactly
        let mut reader = GgufReader::new(&bytes);
        reader.read_bytes(24).expect("Base header read failed");

        // Skip 3 metadata entries (general.architecture, llama.block_count, general.alignment)
        let kv_count = if alignment == 32 { 2 } else { 3 };
        for _ in 0..kv_count {
            let _k = reader.read_string().expect("Read key failed");
            let vtype = reader.read_u32().expect("Read val type failed");
            let _v = reader.read_value(vtype).expect("Read val failed");
        }

        // Read 3 tensor headers
        let mut tensor_offsets = Vec::new();
        for _ in 0..3 {
            let _name = reader.read_string().expect("Tensor name failed");
            let n_dims = reader.read_u32().expect("Tensor ndims failed");
            for _ in 0..n_dims {
                reader.read_u64().expect("Tensor dim failed");
            }
            reader.read_u32().expect("Tensor gtype failed");
            let offset = reader.read_u64().expect("Tensor offset failed");
            tensor_offsets.push(offset);
        }

        let header_end = reader.cursor;
        let expected_header_pad = (alignment - (header_end % alignment)) % alignment;
        let data_start = header_end + expected_header_pad;

        // 1. Invariant: data_start MUST be aligned to `alignment`
        assert_eq!(
            data_start % alignment,
            0,
            "data_start {data_start} not aligned to {alignment}"
        );

        // 2. Invariant: Header padding bytes MUST be all strictly 0x00
        if expected_header_pad > 0 {
            let header_pad_slice = &bytes[header_end..data_start];
            assert_eq!(
                header_pad_slice.len(),
                expected_header_pad,
                "Header padding length mismatch"
            );
            assert!(
                header_pad_slice.iter().all(|&b| b == 0x00),
                "Header padding contains non-zero bytes: {:?}",
                header_pad_slice
            );
        }

        // 3. Invariant: Every tensor offset relative to data_start MUST be aligned to `alignment`
        for (i, &offset) in tensor_offsets.iter().enumerate() {
            assert_eq!(
                offset as usize % alignment,
                0,
                "Tensor {i} offset {offset} not aligned to {alignment}"
            );
            let abs_tensor_start = data_start + offset as usize;
            assert_eq!(
                abs_tensor_start % alignment,
                0,
                "Tensor {i} absolute start {abs_tensor_start} not aligned to {alignment}"
            );
        }

        // 4. Invariant: Inter-tensor padding bytes MUST be all strictly 0x00
        let payloads = [&t1_payload, &t2_payload, &t3_payload];
        for i in 0..3 {
            let start = data_start + tensor_offsets[i] as usize;
            let len = payloads[i].len();
            assert_eq!(
                &bytes[start..start + len],
                payloads[i].as_slice(),
                "Tensor {i} payload corrupt"
            );

            let pad_len = (alignment - (len % alignment)) % alignment;
            let pad_slice = &bytes[start + len..start + len + pad_len];
            assert_eq!(
                pad_slice.len(),
                pad_len,
                "Inter-tensor pad length mismatch for tensor {i}"
            );
            assert!(
                pad_slice.iter().all(|&b| b == 0x00),
                "Tensor {i} trailing padding contains non-zero byte: {:?}",
                pad_slice
            );
        }

        // 5. Invariant: Total file size MUST be a multiple of `alignment`
        assert_eq!(
            bytes.len() % alignment,
            0,
            "Total file size {} is not a multiple of alignment {}",
            bytes.len(),
            alignment
        );

        // 6. Full round-trip parsing through GgufParser
        let model = GgufParser::parse_from_bytes(&bytes, Path::new("align_test.gguf"))
            .expect("GgufParser failed to parse aligned model");
        assert_eq!(model.tensors.len(), 3);
        assert_eq!(model.tensors[0].byte_size, 11);
        assert_eq!(model.tensors[1].byte_size, 29);
        assert_eq!(model.tensors[2].byte_size, 43);
    }
}

// -----------------------------------------------------------------------------
// CHALLENGE 2: Magic bytes and Version constants verification
// -----------------------------------------------------------------------------
#[test]
fn test_adversarial_magic_and_version_constants() {
    let mut writer = GgufWriter::new();
    writer.add_string("general.architecture", "llama");
    let bytes = writer.write_to_bytes().expect("Failed to write");

    // 1. Direct byte inspection of first 8 bytes
    assert_eq!(
        &bytes[0..4],
        b"GGUF",
        "Magic bytes must match ASCII 'GGUF' exactly"
    );

    // 2. Exact little-endian numerical value: 'G'=0x47, 'G'=0x47, 'U'=0x55, 'F'=0x46 -> 0x46554747
    let magic_u32_le = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    assert_eq!(
        magic_u32_le, 0x46554747,
        "Magic 32-bit LE word must be 0x46554747"
    );

    // 3. Version must be 3u32 in little-endian format
    assert_eq!(
        &bytes[4..8],
        &[0x03, 0x00, 0x00, 0x00],
        "Version must be 3u32 in little-endian"
    );
    let version_u32 = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
    assert_eq!(version_u32, 3, "Version must equal 3");

    // 4. Adversarial check: GgufParser must bail on corrupted magic or unsupported version
    let mut bad_magic = bytes.clone();
    bad_magic[0] = b'T';
    bad_magic[1] = b'H';
    bad_magic[2] = b'M';
    bad_magic[3] = b'E';
    let res = GgufParser::parse_from_bytes(&bad_magic, Path::new("bad_magic.gguf"));
    assert!(res.is_err(), "Parser must reject invalid magic b'THME'");

    let mut bad_version = bytes;
    bad_version[4..8].copy_from_slice(&4u32.to_le_bytes());
    let res2 = GgufParser::parse_from_bytes(&bad_version, Path::new("bad_version.gguf"));
    assert!(
        res2.is_err(),
        "Parser must reject unsupported GGUF version 4"
    );
}

// -----------------------------------------------------------------------------
// CHALLENGE 3: Column-major dimension reversal on 1D, 2D, 3D, and 4D tensors
// -----------------------------------------------------------------------------
#[test]
fn test_adversarial_dimension_reversal_1d_to_4d() {
    let mut writer = GgufWriter::new();
    writer.add_string("general.architecture", "llama");

    // Define 1D, 2D, 3D, and 4D tensor shapes (row-major)
    let shape_1d = vec![37];
    let shape_2d = vec![11, 23];
    let shape_3d = vec![5, 7, 13];
    let shape_4d = vec![2, 3, 5, 7];

    writer.add_tensor("tensor_1d", shape_1d.clone(), GgmlType::I8, vec![1u8; 37]);
    writer.add_tensor(
        "tensor_2d",
        shape_2d.clone(),
        GgmlType::I8,
        vec![2u8; 11 * 23],
    );
    writer.add_tensor(
        "tensor_3d",
        shape_3d.clone(),
        GgmlType::I8,
        vec![3u8; 5 * 7 * 13],
    );
    writer.add_tensor(
        "tensor_4d",
        shape_4d.clone(),
        GgmlType::I8,
        vec![4u8; 2 * 3 * 5 * 7],
    );

    let bytes = writer
        .write_to_bytes()
        .expect("Failed to write multi-dim tensors");

    // 1. Raw GgufReader inspection: verify disk encoding is strictly reversed
    let mut reader = GgufReader::new(&bytes);
    reader.read_bytes(24).unwrap(); // base header
    let _key = reader.read_string().unwrap();
    let vtype = reader.read_u32().unwrap();
    let _val = reader.read_value(vtype).unwrap();

    // Check 1D tensor on disk: [37]
    assert_eq!(reader.read_string().unwrap(), "tensor_1d");
    assert_eq!(reader.read_u32().unwrap(), 1);
    assert_eq!(reader.read_u64().unwrap(), 37);
    let _ = reader.read_u32().unwrap(); // gtype
    let _ = reader.read_u64().unwrap(); // offset

    // Check 2D tensor on disk: [23, 11] (fastest dim first)
    assert_eq!(reader.read_string().unwrap(), "tensor_2d");
    assert_eq!(reader.read_u32().unwrap(), 2);
    assert_eq!(reader.read_u64().unwrap(), 23);
    assert_eq!(reader.read_u64().unwrap(), 11);
    let _ = reader.read_u32().unwrap();
    let _ = reader.read_u64().unwrap();

    // Check 3D tensor on disk: [13, 7, 5]
    assert_eq!(reader.read_string().unwrap(), "tensor_3d");
    assert_eq!(reader.read_u32().unwrap(), 3);
    assert_eq!(reader.read_u64().unwrap(), 13);
    assert_eq!(reader.read_u64().unwrap(), 7);
    assert_eq!(reader.read_u64().unwrap(), 5);
    let _ = reader.read_u32().unwrap();
    let _ = reader.read_u64().unwrap();

    // Check 4D tensor on disk: [7, 5, 3, 2]
    assert_eq!(reader.read_string().unwrap(), "tensor_4d");
    assert_eq!(reader.read_u32().unwrap(), 4);
    assert_eq!(reader.read_u64().unwrap(), 7);
    assert_eq!(reader.read_u64().unwrap(), 5);
    assert_eq!(reader.read_u64().unwrap(), 3);
    assert_eq!(reader.read_u64().unwrap(), 2);
    let _ = reader.read_u32().unwrap();
    let _ = reader.read_u64().unwrap();

    // 2. High-level parser roundtrip: GgufParser must invert back to original row-major shapes!
    let model = GgufParser::parse_from_bytes(&bytes, Path::new("multidim.gguf"))
        .expect("Failed to parse multidim model");
    assert_eq!(model.tensors.len(), 4);
    assert_eq!(
        model.tensors[0].shape, shape_1d,
        "1D shape mismatch after roundtrip"
    );
    assert_eq!(
        model.tensors[1].shape, shape_2d,
        "2D shape mismatch after roundtrip"
    );
    assert_eq!(
        model.tensors[2].shape, shape_3d,
        "3D shape mismatch after roundtrip"
    );
    assert_eq!(
        model.tensors[3].shape, shape_4d,
        "4D shape mismatch after roundtrip"
    );
}

// -----------------------------------------------------------------------------
// CHALLENGE 4: Comprehensive roundtrip of all 13 metadata types with boundary values
// -----------------------------------------------------------------------------
#[test]
fn test_adversarial_all_13_metadata_types_exhaustive() {
    let mut writer = GgufWriter::new();

    let test_kvs: Vec<(&str, GgufValue)> = vec![
        // 0: Uint8
        ("test.u8.min", GgufValue::Uint8(0)),
        ("test.u8.max", GgufValue::Uint8(u8::MAX)),
        // 1: Int8
        ("test.i8.min", GgufValue::Int8(i8::MIN)),
        ("test.i8.max", GgufValue::Int8(i8::MAX)),
        // 2: Uint16
        ("test.u16.min", GgufValue::Uint16(0)),
        ("test.u16.max", GgufValue::Uint16(u16::MAX)),
        // 3: Int16
        ("test.i16.min", GgufValue::Int16(i16::MIN)),
        ("test.i16.max", GgufValue::Int16(i16::MAX)),
        // 4: Uint32
        ("test.u32.min", GgufValue::Uint32(0)),
        ("test.u32.max", GgufValue::Uint32(u32::MAX)),
        // 5: Int32
        ("test.i32.min", GgufValue::Int32(i32::MIN)),
        ("test.i32.max", GgufValue::Int32(i32::MAX)),
        // 6: Float32
        ("test.f32.zero", GgufValue::Float32(0.0)),
        ("test.f32.pi", GgufValue::Float32(std::f32::consts::PI)),
        ("test.f32.min_pos", GgufValue::Float32(f32::MIN_POSITIVE)),
        // 7: Bool
        ("test.bool.true", GgufValue::Bool(true)),
        ("test.bool.false", GgufValue::Bool(false)),
        // 8: String
        ("test.str.empty", GgufValue::String(String::new())),
        (
            "test.str.unicode",
            GgufValue::String("Topological äöü Heterogeneous αβγδε 日本語 Engine ∑∫√".into()),
        ),
        // 9: Array
        ("test.arr.empty", GgufValue::Array(Vec::new())),
        (
            "test.arr.u32",
            GgufValue::Array(vec![
                GgufValue::Uint32(1),
                GgufValue::Uint32(100),
                GgufValue::Uint32(u32::MAX),
            ]),
        ),
        (
            "test.arr.str",
            GgufValue::Array(vec![
                GgufValue::String("alpha".into()),
                GgufValue::String("omega".into()),
            ]),
        ),
        (
            "test.arr.f32",
            GgufValue::Array(vec![GgufValue::Float32(1.25), GgufValue::Float32(-3.5)]),
        ),
        // 10: Uint64
        ("test.u64.min", GgufValue::Uint64(0)),
        ("test.u64.max", GgufValue::Uint64(u64::MAX)),
        // 11: Int64
        ("test.i64.min", GgufValue::Int64(i64::MIN)),
        ("test.i64.max", GgufValue::Int64(i64::MAX)),
        // 12: Float64
        ("test.f64.zero", GgufValue::Float64(0.0)),
        ("test.f64.e", GgufValue::Float64(std::f64::consts::E)),
        ("test.f64.min_pos", GgufValue::Float64(f64::MIN_POSITIVE)),
    ];

    for (k, v) in &test_kvs {
        writer.add_metadata(*k, v.clone());
    }

    let bytes = writer
        .write_to_bytes()
        .expect("Failed to write metadata suite");

    let mut reader = GgufReader::new(&bytes);
    reader.read_bytes(4).unwrap(); // magic
    assert_eq!(reader.read_u32().unwrap(), 3); // version
    assert_eq!(reader.read_u64().unwrap(), 0); // tensor_count
    let kv_count = reader.read_u64().unwrap() as usize;
    assert_eq!(kv_count, test_kvs.len());

    for (expected_key, expected_val) in &test_kvs {
        let key = reader.read_string().expect("Failed to read string key");
        assert_eq!(&key, expected_key);
        let val_type = reader.read_u32().expect("Failed to read type id");
        assert_eq!(
            val_type,
            gguf_value_type_id(expected_val),
            "Type ID mismatch for key {key}"
        );
        let parsed_val = reader.read_value(val_type).expect("Failed to read value");
        assert_gguf_values_identical(&parsed_val, expected_val);
    }
}

// -----------------------------------------------------------------------------
// CHALLENGE 5: Boundary conditions: zero tensors, zero metadata, custom alignment, and errors
// -----------------------------------------------------------------------------
#[test]
fn test_adversarial_boundary_conditions() {
    // 1. Boundary: Zero tensors, zero metadata
    {
        let writer = GgufWriter::new();
        let bytes = writer
            .write_to_bytes()
            .expect("Zero/zero model should serialize");
        assert_eq!(
            bytes.len(),
            32,
            "Minimal empty container must be exactly 32 bytes (24 header + 8 pad)"
        );
        assert_eq!(&bytes[0..4], b"GGUF");
        assert_eq!(u32::from_le_bytes(bytes[4..8].try_into().unwrap()), 3);
        assert_eq!(u64::from_le_bytes(bytes[8..16].try_into().unwrap()), 0); // tensors
        assert_eq!(u64::from_le_bytes(bytes[16..24].try_into().unwrap()), 0); // metadata
        assert_eq!(&bytes[24..32], &[0u8; 8]); // 8 zero padding bytes

        let model = GgufParser::parse_from_bytes(&bytes, Path::new("empty.gguf"))
            .expect("Parser should parse empty container");
        assert_eq!(model.tensors.len(), 0);
    }

    // 2. Boundary: Zero metadata, but valid tensors
    {
        let mut writer = GgufWriter::new();
        writer.add_tensor("t_raw", vec![4], GgmlType::F32, vec![0u8; 16]);
        let bytes = writer
            .write_to_bytes()
            .expect("Zero metadata model should serialize");
        let model = GgufParser::parse_from_bytes(&bytes, Path::new("no_meta.gguf"))
            .expect("Parser should parse model without metadata");
        assert_eq!(model.tensors.len(), 1);
        assert_eq!(model.tensors[0].name, "t_raw");
    }

    // 3. Boundary: Custom alignment values (16, 64, 128, 256)
    for align in [16, 64, 128, 256] {
        let mut writer = GgufWriter::new().with_alignment(align);
        writer.add_tensor("t", vec![align], GgmlType::I8, vec![42u8; align]);
        let bytes = writer
            .write_to_bytes()
            .unwrap_or_else(|e| panic!("Failed for alignment {align}: {e}"));
        assert_eq!(
            bytes.len() % align,
            0,
            "File length {} must align to {}",
            bytes.len(),
            align
        );

        let model = GgufParser::parse_from_bytes(&bytes, Path::new("custom_align.gguf"))
            .expect("Parser failed on custom alignment");
        assert_eq!(model.tensors.len(), 1);
    }

    // 4. Adversarial Error paths
    // 4a. Empty metadata key
    {
        let mut writer = GgufWriter::new();
        writer.add_string("", "empty_key_value");
        let res = writer.write_to_bytes();
        assert!(res.is_err(), "Writer must reject empty metadata key");
    }

    // 4b. Empty tensor name
    {
        let mut writer = GgufWriter::new();
        writer.add_tensor("", vec![1], GgmlType::I8, vec![0]);
        let res = writer.write_to_bytes();
        assert!(res.is_err(), "Writer must reject empty tensor name");
    }

    // 4c. Unsupported dimension count > 8
    {
        let mut writer = GgufWriter::new();
        writer.add_tensor(
            "hyper_dim",
            vec![1, 1, 1, 1, 1, 1, 1, 1, 1], // 9 dimensions
            GgmlType::I8,
            vec![0],
        );
        let res = writer.write_to_bytes();
        assert!(
            res.is_err(),
            "Writer must reject tensors with > 8 dimensions"
        );
    }

    // 4d. Inhomogeneous array elements in metadata
    {
        let mut writer = GgufWriter::new();
        writer.add_metadata(
            "inhomogeneous.array",
            GgufValue::Array(vec![
                GgufValue::Uint32(10),
                GgufValue::String("bad_mixed_string".into()),
            ]),
        );
        let res = writer.write_to_bytes();
        assert!(
            res.is_err(),
            "Writer must reject inhomogeneous array metadata"
        );
    }

    // 4e. Element count not divisible by block size for quantized type (Q4_0 block size = 32)
    {
        let mut writer = GgufWriter::new();
        // Shape [17] is not divisible by block size 32
        writer.add_tensor("bad_q4", vec![17], GgmlType::Q4_0, vec![0u8; 18]);
        let res = writer.write_to_bytes();
        assert!(
            res.is_err(),
            "Writer must reject quantized tensor not divisible by block size"
        );
    }
}

// -----------------------------------------------------------------------------
// CHALLENGE 6: High-density stress test with 50 tensors and 100 metadata pairs
// -----------------------------------------------------------------------------
#[test]
fn test_adversarial_high_density_stress() {
    let dir = tempdir().expect("Failed to create tempdir");
    let file_path = dir.path().join("dense_stress.gguf");

    let mut writer = GgufWriter::new();
    writer.add_string("general.architecture", "llama");
    writer.add_u32("llama.block_count", 50);

    // 100 metadata key-value pairs
    for i in 0..100 {
        writer.add_string(format!("meta.key.stress_{i:03}"), format!("value_{i:04}"));
    }

    // 50 tensors of diverse shapes, types, and byte lengths
    for i in 0..50 {
        let (gtype, shape, data) = match i % 4 {
            0 => (GgmlType::F32, vec![8, 8], vec![0xAA; 8 * 8 * 4]),
            1 => (GgmlType::F16, vec![16, 4], vec![0xBB; 16 * 4 * 2]),
            2 => (GgmlType::I8, vec![33], vec![0xCC; 33]),
            _ => (GgmlType::I32, vec![5], vec![0xDD; 5 * 4]),
        };
        writer.add_tensor(format!("stress.tensor.{i:03}"), shape, gtype, data);
    }

    let written = writer
        .write_to_file(&file_path)
        .expect("High-density write failed");
    assert!(written > 0);
    assert_eq!(
        written % 32,
        0,
        "Dense model file size must be 32-byte aligned"
    );

    // Parse back from file using GgufParser
    let model = GgufParser::parse_from_file(&file_path).expect("Dense model parse failed");
    assert_eq!(model.tensors.len(), 50);

    for (i, tensor) in model.tensors.iter().enumerate() {
        assert_eq!(tensor.name, format!("stress.tensor.{i:03}"));
        match i % 4 {
            0 => {
                assert_eq!(tensor.shape, vec![8, 8]);
                assert_eq!(tensor.byte_size, 256);
            }
            1 => {
                assert_eq!(tensor.shape, vec![16, 4]);
                assert_eq!(tensor.byte_size, 128);
            }
            2 => {
                assert_eq!(tensor.shape, vec![33]);
                assert_eq!(tensor.byte_size, 33);
            }
            _ => {
                assert_eq!(tensor.shape, vec![5]);
                assert_eq!(tensor.byte_size, 20);
            }
        }
    }
}
