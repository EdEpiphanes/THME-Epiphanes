// SPDX-License-Identifier: AGPL-3.0-only
use std::path::{Path, PathBuf};
use thme::inference::arch::ArchConfig;
use thme::parser::ModelImporter;

fn resolve_medgemma_path() -> Option<PathBuf> {
    // Tests that open a real model file are opt-in: set THME_REAL_MODEL_TESTS=1 to run them.
    if std::env::var_os("THME_REAL_MODEL_TESTS").is_none() {
        return None;
    }
    let path = std::env::var("THME_MEDGEMMA_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            let relative = PathBuf::from("models/medgemma-4b-it-UD-Q4_K_XL.gguf");
            if relative.exists() {
                relative
            } else if let Ok(local_app_data) = std::env::var("LOCALAPPDATA") {
                PathBuf::from(local_app_data).join("hermes/models/medgemma-4b-it-UD-Q4_K_XL.gguf")
            } else {
                relative
            }
        });
    if path.exists() {
        Some(path)
    } else {
        None
    }
}

fn resolve_or_pack_epiph_path(gguf_path: &Path) -> PathBuf {
    let candidates = [
        PathBuf::from("tmp/medgemma.epiph"),
        PathBuf::from("THME-Epiphanes/tmp/medgemma.epiph"),
    ];
    for p in &candidates {
        if p.exists() {
            return p.clone();
        }
    }
    let target = PathBuf::from("tmp/medgemma.epiph");
    let _ = std::fs::create_dir_all("tmp");
    let pack_args = thme::cli::epiph::PackArgs {
        model: gguf_path.to_path_buf(),
        out: target.clone(),
        recipe: None,
        calibration: None,
        quality: Vec::new(),
        parent: Vec::new(),
        annotate: Vec::new(),
        quantize: false,
        allow_partial: false,
    };
    thme::cli::epiph::pack(&pack_args).expect("pack epiph container");
    target
}

#[test]
fn test_medgemma_model_structure_and_tensor_resolution() {
    let Some(path) = resolve_medgemma_path() else {
        eprintln!("MedGemma model file not found on host disk; skipping test");
        return;
    };

    let model = ModelImporter::import(&path).expect("import medgemma model");
    assert_eq!(model.num_layers, 34);
    assert_eq!(model.hidden_size, 2560);
    assert_eq!(model.num_attention_heads, 8);
    assert_eq!(model.num_key_value_heads, 4);

    let arch_cfg = ArchConfig::detect(&model, &path).expect("detect architecture for medgemma");
    assert_eq!(arch_cfg.family(), "Gemma");

    match arch_cfg {
        ArchConfig::Gemma(ref cfg) => {
            assert_eq!(cfg.num_layers, 34);
            assert_eq!(cfg.hidden_size, 2560);
            assert_eq!(cfg.num_heads, 8);
            assert_eq!(cfg.num_kv_heads, 4);
            assert_eq!(cfg.head_dim, 256);
        }
        _ => panic!("Expected Gemma family architecture for medgemma"),
    }

    // Verify all 34 layer tensors are mapped by find_tensor aliases
    assert!(thme::inference::gemma::find_tensor(&model, "token_embd.weight").is_some());
    assert!(thme::inference::gemma::find_tensor(&model, "output_norm.weight").is_some());

    for layer in 0..1 {
        let q_tensor =
            thme::inference::gemma::find_tensor(&model, &format!("blk.{layer}.attn_q.weight"))
                .expect("attn_q");
        println!(
            "Layer 0 Q tensor shape: {:?}, numel: {}, dtype: {}",
            q_tensor.shape, q_tensor.numel, q_tensor.original_dtype
        );
        let k_tensor =
            thme::inference::gemma::find_tensor(&model, &format!("blk.{layer}.attn_k.weight"))
                .expect("attn_k");
        println!(
            "Layer 0 K tensor shape: {:?}, numel: {}, dtype: {}",
            k_tensor.shape, k_tensor.numel, k_tensor.original_dtype
        );
        let norm_tensor =
            thme::inference::gemma::find_tensor(&model, &format!("blk.{layer}.attn_norm.weight"));
        println!(
            "Layer 0 attn_norm tensor: {:?}",
            norm_tensor.map(|t| (&t.name, &t.shape, t.numel, &t.original_dtype))
        );
    }

    // Load runnable model and verify forward pass over 1 layer
    let runnable = arch_cfg
        .load(&model, thme::inference::WeightPrecision::AsStored)
        .expect("load runnable gemma model");
    assert_eq!(runnable.num_layers(), 34);
    assert_eq!(runnable.vocab_size(), 262208);

    // Run scheduled forward pass across layer 0 with exit after layer 0
    let tokens = [2u32, 106u32];
    let exits = runnable
        .forward_scheduled(&tokens, &[0], &[1])
        .expect("forward_scheduled layer 0");
    assert_eq!(exits.len(), 1);
    assert_eq!(exits[0].layers_applied, 1);
    assert_eq!(exits[0].nll.len(), 1);
    assert!(exits[0].nll[0].is_finite());
    println!(
        "Layer 0 exit NLL: {:.4}, top1: {:?}",
        exits[0].nll[0], exits[0].top1
    );
    if let Some(thme::ir::model::ModelMetadataValue::Array(tokens_meta)) =
        model.source_gguf_metadata.get("tokenizer.ggml.tokens")
    {
        let next_id = exits[0].top1[0] as usize;
        if let Some(thme::ir::model::ModelMetadataValue::String(tok_str)) = tokens_meta.get(next_id)
        {
            println!(
                "Predicted next token for [2, 106]: ID {} = {:?}",
                next_id, tok_str
            );
        }
    }

    // 2-step autoregressive generation loop on MedGemma weights
    let mut seq = vec![2u32, 106u32];
    for step in 0..2 {
        let step_exits = runnable
            .forward_scheduled(&seq, &[0], &[1])
            .unwrap_or_else(|e| panic!("forward_scheduled at step {step} failed: {e}"));
        assert_eq!(step_exits.len(), 1);
        let top1 = step_exits[0]
            .top1
            .last()
            .copied()
            .expect("next token prediction");
        println!(
            "Autoregressive step {step}: seq_len={}, predicted_token_id={top1}",
            seq.len()
        );
        seq.push(top1);
    }
    assert_eq!(seq.len(), 4);
}

#[test]
fn test_medgemma_full_34_layer_forward() {
    let Some(path) = resolve_medgemma_path() else {
        eprintln!("MedGemma model file not found on host disk; skipping test");
        return;
    };

    let model = ModelImporter::import(&path).expect("import medgemma model");
    assert_eq!(model.num_layers, 34);
    assert_eq!(model.hidden_size, 2560);
    assert_eq!(model.num_attention_heads, 8);
    assert_eq!(model.num_key_value_heads, 4);

    let arch_cfg = ArchConfig::detect(&model, &path).expect("detect architecture for medgemma");
    assert_eq!(arch_cfg.family(), "Gemma");

    let runnable = arch_cfg
        .load(&model, thme::inference::WeightPrecision::AsStored)
        .expect("load runnable gemma model");
    assert_eq!(runnable.num_layers(), 34);
    assert_eq!(runnable.vocab_size(), 262208);

    let tokens = [2u32, 106u32];

    // 1. Full 34-layer forward pass producing unnormalized logits for the final token position
    let logits = runnable
        .forward_logits(&tokens)
        .expect("forward_logits over all 34 layers");
    assert_eq!(
        logits.len(),
        262208,
        "Vocabulary size must match MedGemma 262,208"
    );

    // Verify all logits are finite (no NaN, no Inf)
    let non_finite_count = logits
        .iter()
        .filter(|v| v.is_nan() || v.is_infinite())
        .count();
    assert_eq!(non_finite_count, 0, "All output logits must be finite");

    // Top-1 predicted token
    let (top1_id, top1_logit) = logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(id, &v)| (id as u32, v))
        .expect("logits non-empty");

    println!(
        "MedGemma full 34-layer forward: top1_token_id={}, logit={:.4}",
        top1_id, top1_logit
    );
    assert_eq!(
        top1_id, 107,
        "Top-1 token after [2, 106] (<start_of_turn>) must be 107 (<end_of_turn>)"
    );
    assert!(top1_logit.is_finite());
    assert!(
        top1_logit > 0.0,
        "Top-1 logit should be positive and high confidence"
    );

    // 2. Scheduled forward pass across all 34 layers with depth exit at layer 34
    let schedule: Vec<usize> = (0..34).collect();
    let exits = runnable
        .forward_scheduled(&tokens, &schedule, &[34])
        .expect("forward_scheduled all 34 layers");
    assert_eq!(exits.len(), 1);
    assert_eq!(exits[0].layers_applied, 34);
    assert_eq!(exits[0].nll.len(), 1);
    assert!(
        exits[0].nll[0].is_finite(),
        "34-layer exit NLL must be finite"
    );
    println!("MedGemma full 34-layer exit NLL: {:.4}", exits[0].nll[0]);
}

#[test]
fn test_medgemma_epiph_loading_and_inference_parity() {
    let Some(gguf_path) = resolve_medgemma_path() else {
        eprintln!("MedGemma model file not found on host disk; skipping test");
        return;
    };

    let epiph_path = resolve_or_pack_epiph_path(&gguf_path);
    assert!(
        epiph_path.exists(),
        ".epiph container must exist at {:?}",
        epiph_path
    );

    // 1. Verify .epiph container format, 444 tensors and attachment digests
    let report = thme::epiph::verify(&epiph_path).expect("verify epiph container");
    assert_eq!(
        report.tensors, 444,
        "MedGemma .epiph container must verify exactly 444 tensors"
    );
    assert_eq!(
        report.attachments, 1,
        "MedGemma .epiph container must contain GGUF metadata attachment"
    );
    assert!(
        report.payload_bytes > 2_500_000_000,
        "Payload bytes must exceed 2.5 GB"
    );
    println!(
        "Verified .epiph container: {} tensors, {} attachments, {} bytes",
        report.tensors, report.attachments, report.payload_bytes
    );

    // 2. Load model from GGUF and run forward pass
    let gguf_ir = ModelImporter::import(&gguf_path).expect("import gguf model");
    let gguf_arch = ArchConfig::detect(&gguf_ir, &gguf_path).expect("detect gguf architecture");
    let gguf_runnable = gguf_arch
        .load(&gguf_ir, thme::inference::WeightPrecision::AsStored)
        .expect("load gguf runnable");

    let tokens = [2u32, 106u32];
    let gguf_logits = gguf_runnable
        .forward_logits(&tokens)
        .expect("gguf forward_logits");
    assert_eq!(gguf_logits.len(), 262208);

    // 3. Load model from .epiph container via ModelImporter::import
    let epiph_ir = ModelImporter::import(&epiph_path).expect("import epiph model");
    assert_eq!(epiph_ir.num_layers, 34);
    assert_eq!(epiph_ir.hidden_size, 2560);
    assert_eq!(epiph_ir.vocab_size, 262208);

    let epiph_arch = ArchConfig::detect(&epiph_ir, &epiph_path).expect("detect epiph architecture");
    assert_eq!(epiph_arch.family(), "Gemma");

    let epiph_runnable = epiph_arch
        .load(&epiph_ir, thme::inference::WeightPrecision::AsStored)
        .expect("load epiph runnable");
    assert_eq!(epiph_runnable.num_layers(), 34);
    assert_eq!(epiph_runnable.vocab_size(), 262208);

    let epiph_logits = epiph_runnable
        .forward_logits(&tokens)
        .expect("epiph forward_logits");
    assert_eq!(epiph_logits.len(), 262208);

    // 4. Verify numerical parity between GGUF and .epiph logits (< 1e-4 tolerance)
    let mut max_abs_diff = 0.0f32;
    for (idx, (&g_val, &e_val)) in gguf_logits.iter().zip(&epiph_logits).enumerate() {
        let diff = (g_val - e_val).abs();
        if diff > max_abs_diff {
            max_abs_diff = diff;
        }
        assert!(
            diff < 1e-4,
            "Logit parity mismatch at index {idx}: GGUF={g_val}, EPIPH={e_val}, diff={diff}"
        );
    }

    println!(
        "MedGemma GGUF vs .epiph parity confirmed: max_abs_diff = {:.6e}",
        max_abs_diff
    );
    assert!(
        max_abs_diff < 1e-4,
        "Logit max difference ({max_abs_diff}) must be < 1e-4"
    );

    // Compare top-1 predictions
    let gguf_top1 = gguf_logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(id, &v)| (id as u32, v))
        .expect("gguf top1");

    let epiph_top1 = epiph_logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(id, &v)| (id as u32, v))
        .expect("epiph top1");

    assert_eq!(
        gguf_top1.0, epiph_top1.0,
        "Top-1 predicted token ID must match between GGUF and .epiph"
    );
    assert_eq!(gguf_top1.0, 107);
    println!(
        "GGUF top-1: ID {}, logit {:.4}; .epiph top-1: ID {}, logit {:.4}",
        gguf_top1.0, gguf_top1.1, epiph_top1.0, epiph_top1.1
    );
}
