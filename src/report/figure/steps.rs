// SPDX-License-Identifier: AGPL-3.0-only
//! Forward-pass ordering, animation frame stepping, and layer indices for the model figure.

use super::{FORWARD_MAX_FRAMES, FORWARD_STEPS};
use crate::ir::model::ModelIR;
use crate::ir::tensor::TensorRole;

fn role_rank(role: &TensorRole) -> usize {
    use TensorRole::*;
    match role {
        LayerNorm => 0,
        AttentionQ => 1,
        AttentionQkv => 1,
        AttentionK => 2,
        AttentionV => 3,
        AttentionOut => 4,
        MlpGate => 5,
        MlpUp => 6,
        MlpDown => 7,
        _ => 8,
    }
}

/// What to light in one animation frame.
#[derive(Debug, Clone, PartialEq)]
pub struct Highlight {
    /// Index into the model's tensors of the tensor the splash is thrown from (the first lit).
    pub active: usize,
    /// Indices into the model's tensors of every lit tensor (one, or a run of consecutive ones).
    pub lit: Vec<usize>,
    /// 0 at the start of this frame's turn, 1 at the end.
    pub phase: f64,
}

/// Tensor indices in forward-pass order: tensors outside the layers that come first (the
/// embedding), then each layer's tensors in role order, then the rest (final norm, head).
pub fn forward_order(model: &ModelIR) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..model.tensors.len()).collect();
    let key = |i: &usize| {
        let t = &model.tensors[*i];
        let stage = match (t.layer_index, &t.role) {
            (None, TensorRole::Embedding) => 0,
            (Some(_), _) => 1,
            _ => 2,
        };
        (
            stage,
            t.layer_index.unwrap_or(0),
            role_rank(&t.role),
            t.name.clone(),
        )
    };
    idx.sort_by_key(key);
    idx
}

/// The forward-pass animation frames, at most [`FORWARD_MAX_FRAMES`] in total. Each tensor gets
/// up to [`FORWARD_STEPS`] frames, fewer for a bigger model (never below one). Only when there
/// are more tensors than frames do consecutive tensors light together, in forward order.
pub fn forward_steps(model: &ModelIR) -> Vec<Highlight> {
    let order = forward_order(model);
    let mut out = Vec::new();
    if order.len() <= FORWARD_MAX_FRAMES {
        // As many frames per tensor as the cap allows: at most FORWARD_STEPS, never below 1.
        let per = (FORWARD_MAX_FRAMES / order.len().max(1)).clamp(1, FORWARD_STEPS);
        for &active in &order {
            for step in 0..per {
                out.push(Highlight {
                    active,
                    lit: vec![active],
                    phase: (step as f64 + 0.5) / per as f64,
                });
            }
        }
    } else {
        let run = order.len().div_ceil(FORWARD_MAX_FRAMES);
        let runs: Vec<&[usize]> = order.chunks(run).collect();
        let steps = (FORWARD_MAX_FRAMES / runs.len()).max(1);
        for chunk in runs {
            let Some(&active) = chunk.first() else {
                continue;
            };
            for step in 0..steps {
                out.push(Highlight {
                    active,
                    lit: chunk.to_vec(),
                    phase: (step as f64 + 0.5) / steps as f64,
                });
            }
        }
    }
    out
}

/// The layer indices that occur among the model's tensors, sorted (the real structure, not
/// `num_layers`).
pub fn layer_indices(model: &ModelIR) -> Vec<usize> {
    let set: std::collections::BTreeSet<usize> =
        model.tensors.iter().filter_map(|t| t.layer_index).collect();
    set.into_iter().collect()
}
