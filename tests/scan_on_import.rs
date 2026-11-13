// SPDX-License-Identifier: AGPL-3.0-only
//! Scan on import (roadmap v0.2.7.3). Its own test binary, because the override is process-wide.
//! The fixture is a harmless byte pattern (an `os.system` pickle import); nothing is run.

use thme::parser::{allow_unsafe_models, ModelImporter};

#[test]
fn dangerous_models_are_refused_unless_explicitly_allowed() {
    let mock = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/mock_llama");
    // A clean model loads.
    ModelImporter::import(&mock).unwrap();

    // The same model with a code-importing pickle beside it is refused, naming the finding.
    let dir = tempfile::tempdir().unwrap();
    for f in ["config.json", "model.safetensors"] {
        std::fs::copy(mock.join(f), dir.path().join(f)).unwrap();
    }
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
