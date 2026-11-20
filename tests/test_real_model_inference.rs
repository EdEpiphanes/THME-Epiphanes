// SPDX-License-Identifier: AGPL-3.0-only
//! Real-model tests on Pythia-14M (GPT-NeoX) and SmolLM-135M (LLaMA), CPU only.
//!
//! Opt-in: set `THME_REAL_MODEL_TESTS=1` and `THME_TEST_MODELS_DIR` to the folder that holds
//! `pythia-14m` and `smollm-135m`. Without both, each test prints a SKIP line and passes.
//! The assertions use only what both families share, plus each model's own config values.

mod common;

use common::{real_model_dir, resolve_model_dir, TestModel, PYTHIA_14M, SMOLLM_135M};
use std::ffi::OsStr;
use thme::inference::arch::ArchConfig;
use thme::parser::ModelImporter;

const MODELS: [&TestModel; 2] = [&PYTHIA_14M, &SMOLLM_135M];

// Token ids are valid in both vocabularies (50304 and 49152).
const TOKENS: [u32; 2] = [12, 345];

#[test]
fn resolver_needs_config_and_weights_in_the_models_folder() {
    let root = tempfile::tempdir().expect("temp dir");
    let base = root.path();

    // Missing folder: nothing found.
    assert!(resolve_model_dir(Some(base.as_os_str()), "pythia-14m").is_none());

    // Config only: incomplete, nothing found.
    let partial = base.join("pythia-14m");
    std::fs::create_dir_all(&partial).expect("create partial");
    std::fs::write(partial.join("config.json"), "{}").expect("write config");
    assert!(resolve_model_dir(Some(base.as_os_str()), "pythia-14m").is_none());

    // Config and weights: found.
    std::fs::write(partial.join("model.safetensors"), b"").expect("write weights");
    assert_eq!(
        resolve_model_dir(Some(base.as_os_str()), "pythia-14m"),
        Some(partial)
    );

    // Without THME_TEST_MODELS_DIR the relative models/ folder is used; this name is absent.
    assert!(resolve_model_dir(None::<&OsStr>, "no-such-model-folder").is_none());
}

#[test]
fn real_model_import_detects_family_and_shape() {
    for model in MODELS {
        let Some(dir) = real_model_dir(model) else {
            continue;
        };
        let ir = ModelImporter::import(&dir).expect("import real model folder");
        assert_eq!(ir.num_layers, model.num_layers, "{}", model.dir_name);
        assert_eq!(ir.hidden_size, model.hidden_size, "{}", model.dir_name);
        assert_eq!(
            ir.num_attention_heads, model.num_heads,
            "{}",
            model.dir_name
        );
        assert_eq!(ir.vocab_size, model.vocab_size, "{}", model.dir_name);

        let arch = ArchConfig::detect(&ir, &dir).expect("detect architecture");
        assert_eq!(arch.family(), model.family, "{}", model.dir_name);

        let runnable = arch
            .load(&ir, thme::inference::WeightPrecision::AsStored)
            .expect("load runnable model");
        assert_eq!(
            runnable.num_layers(),
            model.num_layers,
            "{}",
            model.dir_name
        );
        assert_eq!(
            runnable.vocab_size(),
            model.vocab_size,
            "{}",
            model.dir_name
        );
    }
}

#[test]
fn real_model_forward_logits_and_full_depth_exit_are_finite() {
    for model in MODELS {
        let Some(dir) = real_model_dir(model) else {
            continue;
        };
        let ir = ModelImporter::import(&dir).expect("import real model folder");
        let arch = ArchConfig::detect(&ir, &dir).expect("detect architecture");
        let runnable = arch
            .load(&ir, thme::inference::WeightPrecision::AsStored)
            .expect("load runnable model");

        let logits = runnable.forward_logits(&TOKENS).expect("forward_logits");
        assert_eq!(logits.len(), model.vocab_size, "{}", model.dir_name);
        assert!(
            logits.iter().all(|v| v.is_finite()),
            "{}: all logits must be finite",
            model.dir_name
        );

        let schedule: Vec<usize> = (0..model.num_layers).collect();
        let exits = runnable
            .forward_scheduled(&TOKENS, &schedule, &[model.num_layers])
            .expect("forward_scheduled over all layers");
        assert_eq!(exits.len(), 1);
        assert_eq!(exits[0].layers_applied, model.num_layers);
        assert_eq!(exits[0].nll.len(), 1);
        assert!(exits[0].nll[0].is_finite(), "{}: NLL", model.dir_name);
        println!(
            "{} ({}): full-depth exit NLL {:.4}",
            model.dir_name, model.family, exits[0].nll[0]
        );
    }
}

#[test]
fn real_model_epiph_round_trip_keeps_logits() {
    for model in MODELS {
        let Some(dir) = real_model_dir(model) else {
            continue;
        };
        let out_root = tempfile::tempdir().expect("temp dir");
        let epiph_path = out_root.path().join(format!("{}.epiph", model.dir_name));

        let pack_args = thme::cli::epiph::PackArgs {
            model: dir.clone(),
            out: epiph_path.clone(),
            recipe: None,
            calibration: None,
            quality: Vec::new(),
            parent: Vec::new(),
            annotate: Vec::new(),
            quantize: false,
            allow_partial: false,
            placement: None,
            compress: None,
        };
        thme::cli::epiph::pack(&pack_args).expect("pack epiph container");

        let report = thme::epiph::verify(&epiph_path).expect("verify epiph container");
        let source_ir = ModelImporter::import(&dir).expect("import source");
        assert!(report.tensors > 0, "{}: no tensors", model.dir_name);
        assert_eq!(
            report.tensors,
            source_ir.tensors.len(),
            "{}",
            model.dir_name
        );

        let source_arch = ArchConfig::detect(&source_ir, &dir).expect("detect source");
        let source_logits = source_arch
            .load(&source_ir, thme::inference::WeightPrecision::AsStored)
            .expect("load source")
            .forward_logits(&TOKENS)
            .expect("source forward_logits");

        let epiph_ir = ModelImporter::import(&epiph_path).expect("import epiph model");
        assert_eq!(epiph_ir.num_layers, model.num_layers, "{}", model.dir_name);
        let epiph_arch = ArchConfig::detect(&epiph_ir, &epiph_path).expect("detect epiph");
        assert_eq!(epiph_arch.family(), model.family, "{}", model.dir_name);
        let epiph_logits = epiph_arch
            .load(&epiph_ir, thme::inference::WeightPrecision::AsStored)
            .expect("load epiph")
            .forward_logits(&TOKENS)
            .expect("epiph forward_logits");

        assert_eq!(source_logits.len(), epiph_logits.len());
        let max_abs_diff = source_logits
            .iter()
            .zip(&epiph_logits)
            .map(|(s, e)| (s - e).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_abs_diff < 1e-4,
            "{}: source vs .epiph logit max diff {max_abs_diff} must be < 1e-4",
            model.dir_name
        );
    }
}
