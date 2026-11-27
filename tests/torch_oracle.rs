// SPDX-License-Identifier: AGPL-3.0-only
//! Independent oracle for the hand-written CPU backward pass (THME-3036 follow-up).
//!
//! `scripts/make_torch_gradients.py` runs the same tiny LLaMA in PyTorch (float64, CPU) and
//! writes `tests/fixtures/torch_gradients_tiny.json`: the loss and, per tensor, the gradient norm
//! and its first eight values. The tests here rebuild the same weights from `tiny_oracle` and
//! compare. The weights come from a fixed-seed generator in `tests/common/tiny_oracle.rs`, shared
//! with `examples/dump_tiny_gradients.rs`, which the Python script reads.

#[path = "common/tiny_oracle.rs"]
mod tiny_oracle;

use anyhow::{Context, Result};
use serde_json::Value;
use thme::trainer::backward::loss_and_grads;
use tiny_oracle::*;

const FIXTURE: &str = "tests/fixtures/torch_gradients_tiny.json";
/// Relative tolerance for the Rust loss and gradients against the PyTorch fixture (f64).
const TOL: f64 = 1e-6;

/// `|got - want| / max(|want|, 1e-9) <= TOL`, with the absolute floor the oracle script uses.
fn assert_close(what: &str, got: f64, want: f64) {
    let rel = (got - want).abs() / want.abs().max(1e-9);
    assert!(
        rel <= TOL,
        "{what}: rust {got:e}, pytorch {want:e}, relative error {rel:e}"
    );
}

#[test]
fn tiny_weights_follow_the_hf_layout() {
    let w = tiny_weights();
    let names = w.tensor_names();
    let tensors = w.tensors();
    assert_eq!(names.len(), tensors.len());
    // embed, nine tensors per layer, final norm; the head is tied, so there is no lm_head.
    assert_eq!(names.len(), 2 * 9 + 2);
    for (name, t) in names.iter().zip(tensors) {
        let hf = hf_name(name);
        let shape = hf_shape(&hf, &w.config, w.vocab, w.inter);
        assert_eq!(t.len(), shape.iter().product::<usize>(), "{hf}");
    }
    assert_eq!(
        hf_name("layers.1.down"),
        "model.layers.1.mlp.down_proj.weight"
    );
    assert_eq!(hf_name("embed"), "model.embed_tokens.weight");
}

#[test]
fn rust_loss_and_gradients_match_pytorch_fixture() -> Result<()> {
    // Skips, loudly, until scripts/make_torch_gradients.py has written the fixture.
    let Ok(text) = std::fs::read_to_string(FIXTURE) else {
        eprintln!("SKIP torch oracle: {FIXTURE} is absent, run scripts/make_torch_gradients.py");
        return Ok(());
    };
    let fixture: Value = serde_json::from_str(&text).context("parsing the fixture")?;

    let w = tiny_weights();
    let c = &w.config;
    let cfg = &fixture["config"];
    assert_eq!(cfg["num_hidden_layers"].as_u64(), Some(c.num_layers as u64));
    assert_eq!(cfg["hidden_size"].as_u64(), Some(c.hidden_size as u64));
    assert_eq!(
        cfg["num_attention_heads"].as_u64(),
        Some(c.num_heads as u64)
    );
    assert_eq!(
        cfg["num_key_value_heads"].as_u64(),
        Some(c.num_kv_heads as u64)
    );
    assert_eq!(cfg["intermediate_size"].as_u64(), Some(w.inter as u64));
    assert_eq!(cfg["vocab_size"].as_u64(), Some(w.vocab as u64));
    assert_eq!(cfg["tie_word_embeddings"].as_bool(), Some(true));
    assert_eq!(cfg["rope_theta"].as_f64(), Some(f64::from(c.rope_theta)));
    assert_eq!(
        cfg["rms_norm_eps"].as_f64(),
        Some(f64::from(c.rms_norm_eps))
    );
    let tokens: Vec<u64> = fixture["tokens"]
        .as_array()
        .context("tokens")?
        .iter()
        .filter_map(Value::as_u64)
        .collect();
    assert_eq!(tokens, TOKENS.map(u64::from));

    let (loss, grads) = loss_and_grads(&w, None, &TOKENS, true)?;
    let torch_loss = fixture["loss"].as_f64().context("loss")?;
    assert_close("loss", loss, torch_loss);

    let entries = fixture["tensors"].as_object().context("tensors")?;
    let names = w.tensor_names();
    assert_eq!(entries.len(), names.len(), "tensor count");
    for (name, grad) in names.iter().zip(grads.weights.tensors()) {
        let hf = hf_name(name);
        let entry = &fixture["tensors"][hf.as_str()];
        let want_norm = entry["norm"].as_f64().context("norm")?;
        let norm = grad.iter().map(|g| g * g).sum::<f64>().sqrt();
        assert_close(&format!("{hf} norm"), norm, want_norm);
        let first8 = entry["first8"].as_array().context("first8")?;
        assert_eq!(first8.len(), 8.min(grad.len()), "{hf} first8 length");
        for (i, want) in first8.iter().enumerate() {
            let want = want.as_f64().context("first8 value")?;
            assert_close(&format!("{hf}[{i}]"), grad[i], want);
        }
    }
    Ok(())
}
