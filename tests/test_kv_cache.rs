// SPDX-License-Identifier: AGPL-3.0-only
//! Incremental decoding against the two real checkpoints (THME-3070), CPU only.
//!
//! Opt-in through the shared gate in `tests/common`: set `THME_REAL_MODEL_TESTS=1` and
//! `THME_TEST_MODELS_DIR` to the folder that holds `pythia-14m` and `smollm-135m`. Without both,
//! each test prints a SKIP line and passes. The synthetic-model checks live in `src/inference/`.

mod common;

use common::{real_model_dir, TestModel, PYTHIA_14M, SMOLLM_135M};
use thme::inference::{ArchConfig, CausalLm, WeightPrecision};
use thme::parser::ModelImporter;

/// Largest allowed absolute gap between incremental and full-sequence logits.
const TOLERANCE: f32 = 1e-4;

/// Feeds a fixed token sequence through the KV cache, first as a prefill chunk and then one
/// token at a time, and checks every position against the full-sequence forward pass.
fn check_incremental(model_def: &TestModel) {
    let Some(path) = real_model_dir(model_def) else {
        return;
    };
    let name = model_def.dir_name;
    let source = ModelImporter::import(&path).expect("import model");
    let arch = ArchConfig::detect(&source, &path).expect("detect architecture");
    assert_eq!(arch.family(), model_def.family);
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
    check_incremental(&PYTHIA_14M);
}

#[test]
fn smollm_135m_incremental_logits_match_full_forward() {
    check_incremental(&SMOLLM_135M);
}
