// SPDX-License-Identifier: AGPL-3.0-only
//! Shared helpers for the opt-in real-model tests on the two small open-weights models.
//!
//! Real models are never searched for in a fixed or user-specific place. The tests look in
//! `THME_TEST_MODELS_DIR` (a folder holding `pythia-14m` and `smollm-135m`), or in a relative
//! `models/` folder, and only when `THME_REAL_MODEL_TESTS=1` is set. Otherwise they print a SKIP
//! reason and return.

// Each test crate includes this module and uses a different subset of it.
#![allow(dead_code)]

use std::ffi::OsStr;
use std::path::{Path, PathBuf};

/// A real model the tests may use, with the shape its `config.json` declares.
pub struct TestModel {
    pub dir_name: &'static str,
    /// The family string `ArchConfig::family` must return.
    pub family: &'static str,
    pub num_layers: usize,
    pub hidden_size: usize,
    pub num_heads: usize,
    pub vocab_size: usize,
}

/// Pythia-14M: GPT-NeoX, 6 layers, hidden 128 (EleutherAI, Apache-2.0).
pub const PYTHIA_14M: TestModel = TestModel {
    dir_name: "pythia-14m",
    family: "GPT-NeoX",
    num_layers: 6,
    hidden_size: 128,
    num_heads: 4,
    vocab_size: 50304,
};

/// SmolLM-135M: LLaMA, 30 layers, hidden 576 (Hugging Face, Apache-2.0).
pub const SMOLLM_135M: TestModel = TestModel {
    dir_name: "smollm-135m",
    family: "LLaMA",
    num_layers: 30,
    hidden_size: 576,
    num_heads: 9,
    vocab_size: 49152,
};

/// Returns `<models_dir>/<dir_name>` when it is a complete Hugging Face folder
/// (`config.json` and `model.safetensors`). `models_dir` defaults to `models`.
pub fn resolve_model_dir(models_dir: Option<&OsStr>, dir_name: &str) -> Option<PathBuf> {
    let base = models_dir.map_or_else(|| PathBuf::from("models"), PathBuf::from);
    let dir = base.join(dir_name);
    let complete = dir.join("config.json").is_file() && dir.join("model.safetensors").is_file();
    complete.then_some(dir)
}

/// Returns the model folder when the real-model gate is open and the folder is present.
/// Otherwise prints a SKIP line that says why and returns `None`.
pub fn real_model_dir(model: &TestModel) -> Option<PathBuf> {
    if std::env::var_os("THME_REAL_MODEL_TESTS").is_none() {
        eprintln!(
            "SKIP {}: real-model tests are off; set THME_REAL_MODEL_TESTS=1",
            model.dir_name
        );
        return None;
    }
    let models_dir = std::env::var_os("THME_TEST_MODELS_DIR");
    let dir = resolve_model_dir(models_dir.as_deref(), model.dir_name);
    if dir.is_none() {
        eprintln!(
            "SKIP {}: no folder with config.json and model.safetensors under THME_TEST_MODELS_DIR (or ./models)",
            model.dir_name
        );
    }
    dir
}

/// Path as a string for JSON-RPC arguments.
pub fn path_str(path: &Path) -> String {
    path.to_string_lossy().into_owned()
}
