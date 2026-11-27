// SPDX-License-Identifier: AGPL-3.0-only
//! Clean-room GGML tensor naming and RoPE layout helpers for LLaMA-family GGUF export.

/// Maps a Hugging Face LLaMA-family tensor name to its documented GGML tensor name.
///
/// Returns `None` for any unrecognised name or non-numeric layer index.
pub fn ggml_tensor_name(hf_name: &str) -> Option<String> {
    match hf_name {
        "model.embed_tokens.weight" => Some("token_embd.weight".to_string()),
        "model.norm.weight" => Some("output_norm.weight".to_string()),
        "lm_head.weight" => Some("output.weight".to_string()),
        _ => {
            let rest = hf_name.strip_prefix("model.layers.")?;
            let (layer_str, suffix) = rest.split_once('.')?;
            if layer_str.is_empty() || !layer_str.bytes().all(|b| b.is_ascii_digit()) {
                return None;
            }
            let layer: usize = layer_str.parse().ok()?;
            let ggml_suffix = match suffix {
                "input_layernorm.weight" => "attn_norm.weight",
                "post_attention_layernorm.weight" => "ffn_norm.weight",
                "self_attn.q_proj.weight" => "attn_q.weight",
                "self_attn.k_proj.weight" => "attn_k.weight",
                "self_attn.v_proj.weight" => "attn_v.weight",
                "self_attn.o_proj.weight" => "attn_output.weight",
                "mlp.gate_proj.weight" => "ffn_gate.weight",
                "mlp.up_proj.weight" => "ffn_up.weight",
                "mlp.down_proj.weight" => "ffn_down.weight",
                _ => return None,
            };
            Some(format!("blk.{layer}.{ggml_suffix}"))
        }
    }
}

/// Returns `true` when `ggml_name` denotes a norm tensor (`attn_norm`, `ffn_norm`, or `output_norm`).
pub fn is_norm_tensor(ggml_name: &str) -> bool {
    ggml_name
        .split('.')
        .any(|part| matches!(part, "attn_norm" | "ffn_norm" | "output_norm"))
}

/// Permutes rows of a row-major `rows x cols` Q/K projection from split-half RoPE pairs
/// `(i, i + head_dim / 2)` into interleaved pairs `(2i, 2i + 1)` per head.
pub fn permute_rope_rows(
    data: &[f32],
    rows: usize,
    cols: usize,
    n_heads: usize,
) -> Result<Vec<f32>, String> {
    remap_rope_rows(data, rows, cols, n_heads, true)
}

/// Inverse of [`permute_rope_rows`], restoring split-half RoPE row order from interleaved pairs.
pub fn unpermute_rope_rows(
    data: &[f32],
    rows: usize,
    cols: usize,
    n_heads: usize,
) -> Result<Vec<f32>, String> {
    remap_rope_rows(data, rows, cols, n_heads, false)
}

fn remap_rope_rows(
    data: &[f32],
    rows: usize,
    cols: usize,
    n_heads: usize,
    to_interleaved: bool,
) -> Result<Vec<f32>, String> {
    let expected_len = rows
        .checked_mul(cols)
        .ok_or_else(|| "RoPE matrix dimensions overflow usize".to_string())?;
    if data.len() != expected_len {
        return Err(format!(
            "RoPE matrix data length {} does not match {}x{} ({expected_len})",
            data.len(),
            rows,
            cols
        ));
    }
    if n_heads == 0 {
        return Err("RoPE n_heads must be non-zero".to_string());
    }
    if !rows.is_multiple_of(n_heads) {
        return Err(format!(
            "RoPE rows ({rows}) must be divisible by n_heads ({n_heads})"
        ));
    }
    let head_dim = rows / n_heads;
    if head_dim == 0 || !head_dim.is_multiple_of(2) {
        return Err(format!(
            "RoPE head_dim ({head_dim}) must be a non-zero even integer"
        ));
    }
    let half = head_dim / 2;
    let mut out = vec![0.0f32; expected_len];
    for head in 0..n_heads {
        let base = head * head_dim;
        for i in 0..half {
            let split_a = base + i;
            let split_b = base + i + half;
            let inter_even = base + 2 * i;
            let inter_odd = base + 2 * i + 1;
            if to_interleaved {
                copy_row(&mut out, data, inter_even, split_a, cols);
                copy_row(&mut out, data, inter_odd, split_b, cols);
            } else {
                copy_row(&mut out, data, split_a, inter_even, cols);
                copy_row(&mut out, data, split_b, inter_odd, cols);
            }
        }
    }
    Ok(out)
}

fn copy_row(dst: &mut [f32], src: &[f32], dst_row: usize, src_row: usize, cols: usize) {
    let dst_start = dst_row * cols;
    let src_start = src_row * cols;
    dst[dst_start..dst_start + cols].copy_from_slice(&src[src_start..src_start + cols]);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ggml_tensor_name_llama_mappings() {
        let cases = [
            ("model.embed_tokens.weight", Some("token_embd.weight")),
            ("model.norm.weight", Some("output_norm.weight")),
            ("lm_head.weight", Some("output.weight")),
            (
                "model.layers.0.input_layernorm.weight",
                Some("blk.0.attn_norm.weight"),
            ),
            (
                "model.layers.12.post_attention_layernorm.weight",
                Some("blk.12.ffn_norm.weight"),
            ),
            (
                "model.layers.3.self_attn.q_proj.weight",
                Some("blk.3.attn_q.weight"),
            ),
            (
                "model.layers.3.self_attn.k_proj.weight",
                Some("blk.3.attn_k.weight"),
            ),
            (
                "model.layers.3.self_attn.v_proj.weight",
                Some("blk.3.attn_v.weight"),
            ),
            (
                "model.layers.3.self_attn.o_proj.weight",
                Some("blk.3.attn_output.weight"),
            ),
            (
                "model.layers.7.mlp.gate_proj.weight",
                Some("blk.7.ffn_gate.weight"),
            ),
            (
                "model.layers.7.mlp.up_proj.weight",
                Some("blk.7.ffn_up.weight"),
            ),
            (
                "model.layers.7.mlp.down_proj.weight",
                Some("blk.7.ffn_down.weight"),
            ),
        ];
        for (hf, expected) in cases {
            assert_eq!(ggml_tensor_name(hf).as_deref(), expected, "for {hf}");
        }
    }

    #[test]
    fn test_ggml_tensor_name_rejects_unknown_and_non_numeric_layers() {
        let rejected = [
            "model.layers.x.mlp.up_proj.weight",
            "rotary_emb.inv_freq",
            "model.layers.0.self_attn.rotary_emb.inv_freq",
            "model.layers..mlp.up_proj.weight",
            "model.layers.-1.mlp.up_proj.weight",
            "model.layers.+1.mlp.up_proj.weight",
            "model.layers.0.self_attn.q_proj.bias",
            "model.layers.0.unknown.weight",
            "transformer.h.0.attn.c_attn.weight",
            "",
        ];
        for name in rejected {
            assert_eq!(ggml_tensor_name(name), None, "expected None for {name}");
        }
    }

    #[test]
    fn test_is_norm_tensor() {
        assert!(is_norm_tensor("blk.0.attn_norm.weight"));
        assert!(is_norm_tensor("blk.15.ffn_norm.weight"));
        assert!(is_norm_tensor("output_norm.weight"));

        assert!(!is_norm_tensor("blk.0.attn_q.weight"));
        assert!(!is_norm_tensor("blk.0.attn_output.weight"));
        assert!(!is_norm_tensor("blk.0.ffn_up.weight"));
        assert!(!is_norm_tensor("output.weight"));
        assert!(!is_norm_tensor("token_embd.weight"));
        assert!(!is_norm_tensor("blk.0.attn_norm_extra.weight"));
    }

    #[test]
    fn test_permute_rope_rows_4x1_single_head() {
        let input = [10.0f32, 11.0, 12.0, 13.0];
        let permuted = permute_rope_rows(&input, 4, 1, 1).expect("valid 4x1 matrix");
        assert_eq!(permuted, vec![10.0, 12.0, 11.0, 13.0]);

        let restored = unpermute_rope_rows(&permuted, 4, 1, 1).expect("valid unpermute");
        assert_eq!(restored, input);
    }

    #[test]
    fn test_permute_rope_rows_8x2_two_heads() {
        let input = [
            0.0f32, 100.0, 1.0, 101.0, 2.0, 102.0, 3.0, 103.0, 4.0, 104.0, 5.0, 105.0, 6.0, 106.0,
            7.0, 107.0,
        ];
        let expected = [
            0.0f32, 100.0, 2.0, 102.0, 1.0, 101.0, 3.0, 103.0, 4.0, 104.0, 6.0, 106.0, 5.0, 105.0,
            7.0, 107.0,
        ];
        let permuted = permute_rope_rows(&input, 8, 2, 2).expect("valid 8x2 matrix");
        assert_eq!(permuted, expected);

        let restored = unpermute_rope_rows(&permuted, 8, 2, 2).expect("valid unpermute");
        assert_eq!(restored, input);
    }

    #[test]
    fn test_permute_rope_rows_round_trip_lcg_1000_values() {
        let mut state: u64 = 0x9e37_79b9_7f4a_7c15;
        let mut input = Vec::with_capacity(1000);
        for _ in 0..1000 {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let val = ((state >> 40) as u32) as f32 / 16_777_216.0;
            input.push(val);
        }

        let permuted = permute_rope_rows(&input, 40, 25, 5).expect("valid 40x25 matrix");
        assert_ne!(permuted, input);
        let restored = unpermute_rope_rows(&permuted, 40, 25, 5).expect("valid unpermute");
        assert_eq!(restored, input);
    }

    #[test]
    fn test_permute_rope_rows_error_cases() {
        assert!(permute_rope_rows(&[1.0, 2.0, 3.0], 4, 1, 1).is_err());
        assert!(permute_rope_rows(&[0.0; 6], 6, 1, 4).is_err());
        assert!(permute_rope_rows(&[0.0; 6], 6, 1, 2).is_err());
        assert!(permute_rope_rows(&[], 0, 4, 2).is_err());
        assert!(permute_rope_rows(&[0.0; 4], 4, 1, 0).is_err());
        assert!(permute_rope_rows(&[0.0; 4], usize::MAX, 2, 1).is_err());

        assert!(unpermute_rope_rows(&[1.0, 2.0, 3.0], 4, 1, 1).is_err());
        assert!(unpermute_rope_rows(&[0.0; 6], 6, 1, 2).is_err());
        assert!(unpermute_rope_rows(&[], 0, 4, 2).is_err());
    }
}
