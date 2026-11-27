// SPDX-License-Identifier: AGPL-3.0-only
//! Dumps the torch-oracle tiny model as one JSON file (THME-3036 follow-up): the config, the
//! token ids, every weight tensor under its Hugging Face name with the Rust loss and the Rust
//! gradient of that tensor. `scripts/make_torch_gradients.py` reads it. The model comes from the
//! fixed-seed generator in `tests/common/tiny_oracle.rs`; the math is `f64`.
//!
//! ```text
//! cargo run --example dump_tiny_gradients -- <out.json>
//! ```

#[path = "../tests/common/tiny_oracle.rs"]
mod tiny_oracle;

use anyhow::{Context, Result};
use serde_json::{json, Map};
use thme::trainer::backward::loss_and_grads;
use tiny_oracle::{hf_name, hf_shape, tiny_weights, TOKENS};

fn main() -> Result<()> {
    let out = std::env::args()
        .nth(1)
        .context("usage: dump_tiny_gradients <out.json>")?;
    let w = tiny_weights();
    let (loss, grads) = loss_and_grads(&w, None, &TOKENS, true)?;

    let mut tensors = Map::new();
    for ((name, values), grad) in w
        .tensor_names()
        .iter()
        .zip(w.tensors())
        .zip(grads.weights.tensors())
    {
        let hf = hf_name(name);
        let shape = hf_shape(&hf, &w.config, w.vocab, w.inter);
        tensors.insert(
            hf,
            json!({ "shape": shape, "values": values, "grad": grad }),
        );
    }

    let c = &w.config;
    let doc = json!({
        "config": {
            "num_hidden_layers": c.num_layers,
            "hidden_size": c.hidden_size,
            "num_attention_heads": c.num_heads,
            "num_key_value_heads": c.num_kv_heads,
            "intermediate_size": w.inter,
            "vocab_size": w.vocab,
            // The f32 config value widened to f64, exactly what the Rust forward uses.
            "rms_norm_eps": f64::from(c.rms_norm_eps),
            "rope_theta": f64::from(c.rope_theta),
            "tie_word_embeddings": w.lm_head.is_none(),
        },
        "tokens": TOKENS,
        "loss": loss,
        "tensors": tensors,
    });
    std::fs::write(&out, serde_json::to_vec(&doc)?).with_context(|| format!("writing {out}"))?;
    println!(
        "wrote {} tensors and loss {loss:.12} to {out}",
        w.tensor_names().len()
    );
    Ok(())
}
