// SPDX-License-Identifier: AGPL-3.0-only
//! Incremental decoding against real checkpoints (THME-3070).
//!
//! Opt-in: set `THME_REAL_MODEL_TESTS=1` and `THME_TEST_MODELS_DIR` to a folder that holds
//! `pythia-14m` and `smollm-135m`. Without them each test prints why it skipped and passes.
//! The synthetic-model checks live beside the forward pass in `src/inference/`.

use std::path::PathBuf;
use thme::inference::{ArchConfig, CausalLm, WeightPrecision};
use thme::parser::ModelImporter;

/// Largest allowed absolute gap between incremental and full-sequence logits.
const TOLERANCE: f32 = 1e-4;

fn model_path(name: &str) -> Option<PathBuf> {
    if std::env::var_os("THME_REAL_MODEL_TESTS").is_none() {
        eprintln!("skipped {name}: set THME_REAL_MODEL_TESTS=1 to run real-model tests");
        return None;
    }
    let Some(root) = std::env::var_os("THME_TEST_MODELS_DIR") else {
        eprintln!("skipped {name}: THME_TEST_MODELS_DIR is not set");
        return None;
    };
    let path = PathBuf::from(root).join(name);
    if !path.exists() {
        eprintln!("skipped {name}: not found under THME_TEST_MODELS_DIR");
        return None;
    }
    Some(path)
}

/// Feeds a fixed token sequence through the cache in two chunkings and checks every position
/// against the full-sequence forward pass.
fn check_incremental(name: &str) {
    let Some(path) = model_path(name) else {
        return;
    };
    let source = ModelImporter::import(&path).expect("import model");
    let arch = ArchConfig::detect(&source, &path).expect("detect architecture");
    let model: Box<dyn CausalLm> = arch
        .load(&source, WeightPrecision::AsStored)
        .expect("load weights");
    let vocab = model.vocab_size() as u64;
    let tokens: Vec<u32> = (0..24u64)
        .map(|i| ((i * 7919 + 13) % vocab) as u32)
        .collect();
    for prefill in [1, 8] {
        let mut cache = model.new_cache().expect("architecture has a KV cache");
        let mut worst = 0.0f32;
        let mut end = 0;
        while end < tokens.len() {
            let step = if end == 0 { prefill } else { 1 };
            let next = (end + step).min(tokens.len());
            let logits = model
                .forward_extend(&mut cache, &tokens[end..next])
                .expect("incremental step");
            end = next;
            if end < 2 {
                continue; // forward_logits needs two tokens; nothing to compare yet
            }
            let full = model.forward_logits(&tokens[..end]).expect("full pass");
            for (a, b) in logits.iter().zip(&full) {
                worst = worst.max((a - b).abs());
            }
        }
        assert_eq!(cache.len(), tokens.len());
        println!(
            "{name} prefill={prefill}: max |incremental - full| = {worst:e} over {} positions",
            tokens.len()
        );
        assert!(worst < TOLERANCE, "{name} prefill={prefill}: gap {worst}");
    }
}

#[test]
fn pythia_14m_incremental_logits_match_full_forward() {
    check_incremental("pythia-14m");
}

#[test]
fn smollm_135m_incremental_logits_match_full_forward() {
    check_incremental("smollm-135m");
}
