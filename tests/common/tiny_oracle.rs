// SPDX-License-Identifier: AGPL-3.0-only
//! Fixed-seed tiny LLaMA shared by the torch-oracle example and test (THME-3036 follow-up).
//!
//! shortcut: a splitmix64 generator instead of the inference test helper `tiny_model`, which is
//! `cfg(test)` and cannot be reached from an example. The numbers differ from `tiny_model`.
//! Upgrade if the oracle has to share the inference test model.

use thme::inference::llama::LlamaConfig;
use thme::trainer::backward::{LayerWeights, TrainWeights};

pub const TOKENS: [u32; 7] = [3, 17, 5, 9, 30, 2, 11];
const VOCAB: usize = 40;
const INTER: usize = 32;
const LAYERS: usize = 2;

pub fn tiny_config() -> LlamaConfig {
    LlamaConfig {
        num_layers: LAYERS,
        hidden_size: 16,
        num_heads: 4,
        num_kv_heads: 2,
        rms_norm_eps: 1e-5,
        rope_theta: 10_000.0,
    }
}

/// Deterministic uniform values in [-1, 1) from splitmix64.
struct Rng(u64);

impl Rng {
    fn values(&mut self, len: usize, scale: f64) -> Vec<f64> {
        (0..len)
            .map(|_| {
                self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
                let mut z = self.0;
                z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
                z ^= z >> 31;
                ((z >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0) * scale
            })
            .collect()
    }

    /// Norm gains near one, so the gain gradients are not trivially zero.
    fn gains(&mut self, len: usize) -> Vec<f64> {
        self.values(len, 0.1).into_iter().map(|v| 1.0 + v).collect()
    }
}

/// The tiny model with a tied head, every weight drawn from the fixed seed.
pub fn tiny_weights() -> TrainWeights<f64> {
    let config = tiny_config();
    let d = config.hidden_size;
    let kv = d / config.num_heads * config.num_kv_heads;
    let mut rng = Rng(7);
    let embed = rng.values(VOCAB * d, 0.25);
    let layers = (0..LAYERS)
        .map(|_| LayerWeights {
            attn_norm: rng.gains(d),
            q: rng.values(d * d, 0.25),
            k: rng.values(kv * d, 0.25),
            v: rng.values(kv * d, 0.25),
            o: rng.values(d * d, 0.25),
            mlp_norm: rng.gains(d),
            gate: rng.values(INTER * d, 0.25),
            up: rng.values(INTER * d, 0.25),
            down: rng.values(d * INTER, 0.25),
        })
        .collect();
    TrainWeights {
        config,
        vocab: VOCAB,
        inter: INTER,
        embed,
        layers,
        final_norm: rng.gains(d),
        lm_head: None,
    }
}

/// Hugging Face tensor name of an internal name such as `layers.0.q` (`embed`, `final_norm`).
pub fn hf_name(internal: &str) -> String {
    match internal {
        "embed" => "model.embed_tokens.weight".to_string(),
        "final_norm" => "model.norm.weight".to_string(),
        "lm_head" => "lm_head.weight".to_string(),
        _ => {
            let mut parts = internal.split('.');
            let (Some("layers"), Some(layer), Some(part), None) =
                (parts.next(), parts.next(), parts.next(), parts.next())
            else {
                return format!("unknown.{internal}");
            };
            let suffix = match part {
                "attn_norm" => "input_layernorm.weight",
                "q" => "self_attn.q_proj.weight",
                "k" => "self_attn.k_proj.weight",
                "v" => "self_attn.v_proj.weight",
                "o" => "self_attn.o_proj.weight",
                "mlp_norm" => "post_attention_layernorm.weight",
                "gate" => "mlp.gate_proj.weight",
                "up" => "mlp.up_proj.weight",
                "down" => "mlp.down_proj.weight",
                other => return format!("unknown.{other}"),
            };
            format!("model.layers.{layer}.{suffix}")
        }
    }
}

/// Row-major shape of a Hugging Face tensor (`[out, in]` for linear layers).
pub fn hf_shape(hf: &str, c: &LlamaConfig, vocab: usize, inter: usize) -> Vec<usize> {
    let d = c.hidden_size;
    let kv = d / c.num_heads * c.num_kv_heads;
    if hf.ends_with("embed_tokens.weight") || hf == "lm_head.weight" {
        vec![vocab, d]
    } else if hf.ends_with("norm.weight") {
        vec![d]
    } else if hf.ends_with("k_proj.weight") || hf.ends_with("v_proj.weight") {
        vec![kv, d]
    } else if hf.ends_with("gate_proj.weight") || hf.ends_with("up_proj.weight") {
        vec![inter, d]
    } else if hf.ends_with("down_proj.weight") {
        vec![d, inter]
    } else {
        vec![d, d]
    }
}
