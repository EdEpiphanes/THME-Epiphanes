// SPDX-License-Identifier: AGPL-3.0-only
//! Portability golden test for the `.epiph` format (`docs/formats/epiph-v1.md`, section 9).
//!
//! The model is built from fixed integers. Its description holds no clock, no random value, no
//! host path and no hash-map order, so the packed bytes depend only on the writer. CI runs this
//! test on Linux x86-64, Windows x86-64 and macOS arm64. Big-endian hosts and other chips are
//! not tested.

use std::path::Path;
use thme::epiph::{read, verify, write, Encoding, PackOptions};
use thme::exporter::compute_sha256;
use thme::inference::read_tensor_bytes;
use thme::ir::tensor::{SourceEncoding, TensorSource};
use thme::ir::{ModelArchitecture, ModelIR, TensorIR};

/// SHA-256 of the whole file with identity encoding (`compress: None` writes the same bytes).
/// The manifest records `tool_version`, so this hash changes with every crate version bump.
const GOLDEN_IDENTITY_FILE: &str =
    "9e6ea4736046fb873d730cd27183119c0ae6130f5e7cefe67ea322d4eddb8e48";
/// SHA-256 of the three decoded payloads concatenated in tensor order. The DEFLATE stream bytes
/// are not pinned: RFC 1951 allows many streams for one payload, and the stream depends on the
/// compressor backend. Only the decoded bytes are the contract.
const GOLDEN_DECODED_PAYLOADS: &str =
    "b8fbecc9ee0f4ae79ffc1c504d161051a0e473264005090d0a638d47520e470a";

/// Tensors as (name, shape, payload). Payloads are F32 words from fixed integers.
fn golden_tensors() -> Vec<(&'static str, Vec<usize>, Vec<u8>)> {
    // A repeating ramp of 1 KiB: DEFLATE shrinks it, so the compressed path is exercised.
    let ramp: Vec<u8> = (0..256)
        .flat_map(|i| ((i % 7) as f32).to_le_bytes())
        .collect();
    // Small hashed integers, 12 bits each, from wrapping integer arithmetic only.
    let hashed = |words: u32, salt: u32| -> Vec<u8> {
        (0..words)
            .flat_map(|i| {
                let v = i.wrapping_add(salt).wrapping_mul(2_654_435_761) >> 20;
                (v as f32).to_le_bytes()
            })
            .collect()
    };
    vec![
        ("token_embd.weight", vec![32, 8], ramp),
        ("blk.0.attn_q.weight", vec![4, 4], hashed(16, 1)),
        ("output_norm.weight", vec![8], hashed(8, 7)),
    ]
}

/// Writes each payload beside `dir` under its tensor name and describes it as a file-backed
/// F32 tensor. The writer needs a file to read from; the file name is fixed, so it is portable.
fn golden_model(dir: &Path) -> ModelIR {
    let mut tensors: Vec<TensorIR> = golden_tensors()
        .into_iter()
        .map(|(name, shape, bytes)| {
            let path = dir.join(format!("{name}.raw"));
            std::fs::write(&path, &bytes).expect("write source payload");
            let mut tensor = TensorIR::new(name.into(), shape, "F32".into(), bytes.len());
            tensor.source = Some(TensorSource {
                path,
                byte_offset: 0,
                byte_len: bytes.len() as u64,
                encoding: SourceEncoding::Identity,
            });
            tensor
        })
        .collect();
    // Two keys in a HashMap: the writer must order them the same way on every run.
    tensors[0].metadata.insert("zeta".into(), "1".into());
    tensors[0].metadata.insert("alpha".into(), "2".into());
    ModelIR::new(
        "golden".into(),
        ModelArchitecture::Llama,
        1,
        8,
        2,
        2,
        8,
        16,
        tensors,
    )
}

/// Packs `model` with a fixed creation time and returns the file bytes.
fn pack(model: &ModelIR, compress: Option<Encoding>, out: &Path) -> Vec<u8> {
    let options = PackOptions {
        compress,
        created_unix: Some(1_800_000_000),
        ..PackOptions::default()
    };
    write(model, &options, out).expect("pack golden model");
    std::fs::read(out).expect("read packed file")
}

#[test]
fn packed_bytes_match_the_golden_hashes_on_every_host() {
    let dir = tempfile::tempdir().expect("temporary directory");
    let model = golden_model(dir.path());
    let expected = golden_tensors();

    let identity_path = dir.path().join("identity.epiph");
    let identity = pack(&model, Some(Encoding::Identity), &identity_path);
    assert_eq!(
        compute_sha256(&identity),
        GOLDEN_IDENTITY_FILE,
        "identity encoding file bytes changed"
    );
    let none = pack(&model, None, &dir.path().join("none.epiph"));
    assert_eq!(none, identity, "no encoding and identity encoding differ");
    assert_eq!(
        verify(&identity_path)
            .expect("verify identity file")
            .tensors,
        expected.len()
    );

    let deflate_path = dir.path().join("deflate.epiph");
    let summary = write(
        &model,
        &PackOptions {
            compress: Some(Encoding::Deflate),
            created_unix: Some(1_800_000_000),
            ..PackOptions::default()
        },
        &deflate_path,
    )
    .expect("pack golden model with deflate");
    assert!(summary.compressed_tensors >= 1, "no payload was compressed");
    verify(&deflate_path).expect("verify deflate file");

    for path in [&identity_path, &deflate_path] {
        let (back, _) = read(path).expect("read packed file");
        assert_eq!(back.tensors.len(), expected.len());
        let mut decoded = Vec::new();
        for (tensor, (name, shape, bytes)) in back.tensors.iter().zip(&expected) {
            assert_eq!(tensor.name, *name);
            assert_eq!(&tensor.shape, shape);
            let payload = read_tensor_bytes(tensor).expect("decode payload");
            assert_eq!(&payload, bytes, "{name} differs after reading back");
            decoded.extend(payload);
        }
        assert_eq!(
            compute_sha256(&decoded),
            GOLDEN_DECODED_PAYLOADS,
            "decoded payloads changed in {path:?}"
        );
    }
}
