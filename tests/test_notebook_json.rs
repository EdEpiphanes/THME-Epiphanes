// SPDX-License-Identifier: AGPL-3.0-only
//! The `--json` output that docs/notebooks.md documents. Each test runs the real `thme` binary on a
//! tiny GGUF written to a temp folder, then checks that stdout parses as JSON and has the keys the
//! guide lists.

use serde_json::Value;
use std::path::{Path, PathBuf};
use std::process::Command;
use thme::exporter::gguf_writer::GgufWriter;
use thme::parser::gguf::GgmlType;

fn tiny_gguf(dir: &Path) -> PathBuf {
    let path = dir.join("tiny.gguf");
    let mut writer = GgufWriter::new();
    writer
        .add_string("general.architecture", "llama")
        .add_u32("llama.block_count", 1)
        .add_u32("llama.embedding_length", 4)
        .add_u32("llama.attention.head_count", 1)
        .add_u32("llama.attention.head_count_kv", 1)
        .add_u32("llama.context_length", 128)
        .add_tensor(
            "weights",
            vec![4],
            GgmlType::F16,
            vec![0x00, 0x3c, 0x00, 0xc0, 0x00, 0x42, 0x00, 0x34],
        );
    writer.write_to_file(&path).expect("write tiny gguf");
    path
}

fn thme_json(args: &[&str]) -> Value {
    let out = Command::new(env!("CARGO_BIN_EXE_thme"))
        .args(args)
        .output()
        .expect("run thme");
    assert!(
        out.status.success(),
        "thme {args:?} failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    serde_json::from_slice(&out.stdout)
        .unwrap_or_else(|e| panic!("stdout of thme {args:?} is not pure JSON: {e}"))
}

fn assert_keys(value: &Value, keys: &[&str]) {
    for key in keys {
        assert!(value.get(key).is_some(), "missing key '{key}' in {value}");
    }
}

#[test]
fn inspect_json_has_documented_keys() {
    let dir = tempfile::tempdir().expect("temp dir");
    let model = tiny_gguf(dir.path());
    let value = thme_json(&["inspect", model.to_str().expect("utf8 path"), "--json"]);
    assert_keys(
        &value,
        &[
            "name",
            "architecture",
            "num_parameters",
            "num_layers",
            "hidden_size",
            "vocab_size",
            "tensors",
        ],
    );
}

#[test]
fn profile_json_has_documented_keys() {
    let dir = tempfile::tempdir().expect("temp dir");
    let model = tiny_gguf(dir.path());
    let value = thme_json(&["profile", model.to_str().expect("utf8 path"), "--json"]);
    assert_keys(&value, &["model", "layers", "kv_cache"]);
    assert!(value["layers"].is_array(), "layers must be a list: {value}");
    assert!(
        value["kv_cache"].is_array(),
        "kv_cache must be a list: {value}"
    );
}
