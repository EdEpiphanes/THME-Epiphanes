// SPDX-License-Identifier: AGPL-3.0-only
//! Scan on import (roadmap v0.2.7.3). Its own test binary, because the override is process-wide.
//! The model is a tiny synthetic checkpoint written here; the hostile fixture is a harmless byte
//! pattern (an `os.system` pickle import). Nothing is run.

use std::path::Path;
use thme::exporter::safetensors_writer::SafeTensorsWriter;
use thme::parser::{allow_unsafe_models, ModelImporter};

/// A two-tensor LLaMA-style checkpoint with its config.
fn write_model(dir: &Path) {
    let d = 8usize;
    let mut w = SafeTensorsWriter::new();
    w.add_tensor(
        "model.embed_tokens.weight",
        "F32",
        vec![16, d],
        0.5f32.to_le_bytes().repeat(16 * d),
    );
    w.add_tensor(
        "model.norm.weight",
        "F32",
        vec![d],
        1.0f32.to_le_bytes().repeat(d),
    );
    w.write_to_file(dir.join("model.safetensors")).unwrap();
    std::fs::write(
        dir.join("config.json"),
        r#"{"architectures":["LlamaForCausalLM"],"model_type":"llama","hidden_size":8,"num_hidden_layers":0,"num_attention_heads":2,"num_key_value_heads":2,"intermediate_size":16,"vocab_size":16}"#,
    )
    .unwrap();
}

#[test]
fn dangerous_models_are_refused_unless_explicitly_allowed() {
    let clean = tempfile::tempdir().unwrap();
    write_model(clean.path());
    ModelImporter::import(clean.path()).unwrap();

    // The same model with a code-importing pickle beside it is refused, naming the finding.
    let dir = tempfile::tempdir().unwrap();
    write_model(dir.path());
    std::fs::write(
        dir.path().join("extra.pkl"),
        b"cos\nsystem\n(S'echo fixture'\ntR.",
    )
    .unwrap();
    let err = format!("{:#}", ModelImporter::import(dir.path()).unwrap_err());
    assert!(
        err.contains("refusing to load") && err.contains("extra.pkl"),
        "{err}"
    );
    assert!(err.contains("--allow-unsafe-model"), "{err}");

    // The explicit override loads it (with a warning on stderr); the pickle is never executed.
    allow_unsafe_models(true);
    let model = ModelImporter::import(dir.path()).unwrap();
    assert!(!model.tensors.is_empty());
    allow_unsafe_models(false);
    assert!(
        ModelImporter::import(dir.path()).is_err(),
        "the override is not sticky once cleared"
    );
}
