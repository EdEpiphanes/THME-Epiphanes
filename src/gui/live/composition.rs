// SPDX-License-Identifier: AGPL-3.0-only
//! Model composition extraction, report summaries, and token helpers for the live view.

use std::path::PathBuf;

use super::{Composition, Family};
use crate::ir::{ModelIR, TensorFormat, TensorRole};

pub(super) fn composition_from_ir(ir: &ModelIR) -> Composition {
    let mut layers: Composition = vec![Vec::new(); ir.num_layers];
    for tensor in &ir.tensors {
        let (Some(layer), true) = (tensor.layer_index, tensor.shape.len() == 2) else {
            continue;
        };
        let name = match tensor.role {
            TensorRole::AttentionQ => "q",
            TensorRole::AttentionK => "k",
            TensorRole::AttentionV => "v",
            TensorRole::AttentionOut => "o",
            TensorRole::AttentionQkv => "qkv",
            TensorRole::MlpGate => "gate",
            TensorRole::MlpUp => "up",
            TensorRole::MlpDown => "down",
            _ => continue,
        };
        if let Some(slot) = layers.get_mut(layer) {
            slot.push((name.to_string(), Family::from_dtype(&tensor.original_dtype)));
        }
    }
    layers
}

/// Composition of a recipe-applied model: assigned formats, BYOA algorithms by their declared
/// bits, and stored formats for untouched tensors.
pub(super) fn composition_from_candidate(ir: &ModelIR) -> Composition {
    let mut layers = composition_from_ir(ir);
    for tensor in &ir.tensors {
        let (Some(layer), true) = (tensor.layer_index, tensor.shape.len() == 2) else {
            continue;
        };
        let family = match &tensor.selected_format {
            TensorFormat::FP32 => continue, // untouched: keep the stored family
            TensorFormat::Custom(_) => {
                let bits = tensor
                    .metadata
                    .get(crate::inference::BYOA_METADATA_KEY)
                    .and_then(|json| {
                        crate::transform::byoa::AlgorithmDefinition::from_json(json).ok()
                    })
                    .map_or(16.0, |a| a.effective_bits_per_param);
                match bits {
                    b if b <= 1.2 => Family::Binary,
                    b if b <= 2.0 => Family::Ternary,
                    b if b <= 4.5 => Family::Int4,
                    b if b <= 8.5 => Family::Int8,
                    _ => Family::Dense,
                }
            }
            other => Family::of(other),
        };
        let name = role_short_name(tensor.role);
        if let (Some(slot), Some(name)) = (layers.get_mut(layer), name) {
            if let Some(entry) = slot.iter_mut().find(|(n, _)| n == name) {
                entry.1 = family;
            }
        }
    }
    layers
}

fn role_short_name(role: TensorRole) -> Option<&'static str> {
    Some(match role {
        TensorRole::AttentionQ => "q",
        TensorRole::AttentionK => "k",
        TensorRole::AttentionV => "v",
        TensorRole::AttentionOut => "o",
        TensorRole::AttentionQkv => "qkv",
        TensorRole::MlpGate => "gate",
        TensorRole::MlpUp => "up",
        TensorRole::MlpDown => "down",
        _ => return None,
    })
}

pub(super) fn optional_path(text: &str) -> Option<PathBuf> {
    let trimmed = text.trim();
    (!trimmed.is_empty()).then(|| PathBuf::from(trimmed))
}

/// Short lines for the side panel from a `thme evaluate` report.
pub(super) fn gate_summary(report: &serde_json::Value, path: &std::path::Path) -> Vec<String> {
    let variants = report["evaluation"]["variants"]
        .as_array()
        .cloned()
        .unwrap_or_default();
    let ppl = |i: usize| {
        variants
            .get(i)
            .and_then(|v| v["perplexity"].as_f64())
            .unwrap_or(f64::NAN)
    };
    let delta = ppl(1) - ppl(0);
    let passed = report["evaluation"]["gate_passed"]
        .as_bool()
        .unwrap_or(false);
    let mut lines = vec![
        format!(
            "{} Δ PPL {delta:+.4}",
            if passed { "PASSED" } else { "FAILED" }
        ),
        format!("Source {:.4} -> recipe {:.4}", ppl(0), ppl(1)),
    ];
    if let Some(skipped) = report["recipe"]["operations_not_evaluated"].as_array() {
        if !skipped.is_empty() {
            let names: Vec<&str> = skipped.iter().filter_map(|s| s.as_str()).collect();
            lines.push(format!("Not evaluated: {}", names.join(", ")));
        }
    }
    lines.push(format!("Report: {}", path.display()));
    lines
}

pub(super) fn pseudo_random_tokens(vocab: usize, count: usize, seed: u64) -> Vec<u32> {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
    (0..count)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) % vocab.max(1) as u64) as u32
        })
        .collect()
}
