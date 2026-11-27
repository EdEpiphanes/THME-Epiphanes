// SPDX-License-Identifier: AGPL-3.0-only
//! Exact symmetries of the LLaMA-family forward pass (v0.2.9.2, WP15).
//!
//! Each transform rewrites the weights of a dense [`LlamaModel`] so that the forward pass gives
//! the same logits up to floating-point rounding. Every transform here is exact: a permutation
//! of FFN units or key/value groups, folding RMS-norm scales into the matrices that read them,
//! and a rotation of the residual stream after folding. Bounded and empirical transforms (for
//! example after block quantization) are not in this slice. Tests check each transform against
//! the reference forward pass on the tiny model, and a control shows that the check can fail.

use anyhow::{bail, Result};

use super::llama::LlamaModel;

/// An orthogonal `dim x dim` matrix, stored row-major and built from Householder reflections.
#[derive(Debug, Clone, PartialEq)]
pub struct Rotation {
    dim: usize,
    matrix: Vec<f32>,
}

impl Rotation {
    /// Builds `H_k ... H_1` where `H = I - 2 u u^T` and `u` is each given vector normalised.
    /// Every vector must have length `dim` and be non-zero.
    pub fn from_reflectors(dim: usize, vectors: &[Vec<f32>]) -> Result<Self> {
        let mut q = vec![0.0f64; dim * dim];
        for i in 0..dim {
            q[i * dim + i] = 1.0;
        }
        for v in vectors {
            if v.len() != dim {
                bail!("reflector has length {}, expected {dim}", v.len());
            }
            let norm2: f64 = v.iter().map(|x| f64::from(*x) * f64::from(*x)).sum();
            if norm2 == 0.0 {
                bail!("reflector must be non-zero");
            }
            let u: Vec<f64> = v.iter().map(|x| f64::from(*x) / norm2.sqrt()).collect();
            // q <- (I - 2 u u^T) q, computed column by column.
            for col in 0..dim {
                let dot: f64 = (0..dim).map(|r| u[r] * q[r * dim + col]).sum();
                for r in 0..dim {
                    q[r * dim + col] -= 2.0 * u[r] * dot;
                }
            }
        }
        Ok(Self {
            dim,
            matrix: q.into_iter().map(|x| x as f32).collect(),
        })
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn matrix(&self) -> &[f32] {
        &self.matrix
    }

    /// `Q v`.
    fn apply(&self, v: &[f32]) -> Vec<f32> {
        (0..self.dim)
            .map(|i| {
                (0..self.dim)
                    .map(|j| self.matrix[i * self.dim + j] * v[j])
                    .sum()
            })
            .collect()
    }
}

/// Checks that `perm` is a permutation of `0..n`.
fn check_permutation(perm: &[usize], n: usize) -> Result<()> {
    if perm.len() != n {
        bail!("permutation has {} entries, expected {n}", perm.len());
    }
    let mut seen = vec![false; n];
    for &p in perm {
        if p >= n || seen[p] {
            bail!("{perm:?} is not a permutation of 0..{n}");
        }
        seen[p] = true;
    }
    Ok(())
}

/// Reorders the rows of a `rows x row_len` matrix: new row `i` is old row `perm[i]`.
fn permute_rows(w: &mut [f32], row_len: usize, perm: &[usize]) {
    let old = w.to_vec();
    for (i, &p) in perm.iter().enumerate() {
        w[i * row_len..(i + 1) * row_len].copy_from_slice(&old[p * row_len..(p + 1) * row_len]);
    }
}

/// Reorders the columns of a `rows x cols` matrix: new column `j` is old column `perm[j]`.
fn permute_cols(w: &mut [f32], cols: usize, perm: &[usize]) {
    let old = w.to_vec();
    let rows = w.len() / cols;
    for r in 0..rows {
        for (j, &p) in perm.iter().enumerate() {
            w[r * cols + j] = old[r * cols + p];
        }
    }
}

/// Multiplies each column `j` of a `rows x cols` matrix by `scale[j]`.
fn scale_cols(w: &mut [f32], cols: usize, scale: &[f32]) {
    for row in w.chunks_exact_mut(cols) {
        for (x, s) in row.iter_mut().zip(scale) {
            *x *= s;
        }
    }
}

/// Replaces each row `r` of a matrix whose rows have length `dim` by `Q r`.
fn rotate_rows(w: &mut [f32], rot: &Rotation) {
    for row in w.chunks_exact_mut(rot.dim) {
        let rotated = rot.apply(row);
        row.copy_from_slice(&rotated);
    }
}

/// Replaces each column `c` of a `dim x cols` matrix by `Q c`.
fn rotate_cols(w: &mut [f32], rot: &Rotation) {
    let cols = w.len() / rot.dim;
    for j in 0..cols {
        let column: Vec<f32> = (0..rot.dim).map(|r| w[r * cols + j]).collect();
        let rotated = rot.apply(&column);
        for (r, value) in rotated.into_iter().enumerate() {
            w[r * cols + j] = value;
        }
    }
}

/// Permutes the FFN units of one layer: gate and up rows, and down columns, in the same order.
pub fn permute_ffn_units(model: &mut LlamaModel, layer: usize, perm: &[usize]) -> Result<()> {
    let d = model.config.hidden_size;
    let inter = model.intermediate_size;
    check_permutation(perm, inter)?;
    let Some(l) = model.layers.get_mut(layer) else {
        bail!("layer {layer} does not exist");
    };
    permute_rows(&mut l.gate, d, perm);
    permute_rows(&mut l.up, d, perm);
    permute_cols(&mut l.down, inter, perm);
    Ok(())
}

/// Permutes key/value groups of one layer. Key and value heads move by `perm`, and each query
/// head moves with the group it attends to, so grouped-query attention still pairs them.
pub fn permute_kv_groups(model: &mut LlamaModel, layer: usize, perm: &[usize]) -> Result<()> {
    let cfg = &model.config;
    let (d, heads, kv_heads) = (cfg.hidden_size, cfg.num_heads, cfg.num_kv_heads);
    check_permutation(perm, kv_heads)?;
    let hd = d / heads;
    let group = heads / kv_heads;
    // New query head i * group + j takes old query head perm[i] * group + j.
    let query_perm: Vec<usize> = (0..heads)
        .map(|h| perm[h / group] * group + h % group)
        .collect();
    let Some(l) = model.layers.get_mut(layer) else {
        bail!("layer {layer} does not exist");
    };
    permute_row_blocks(&mut l.k, hd, d, perm);
    permute_row_blocks(&mut l.v, hd, d, perm);
    permute_row_blocks(&mut l.q, hd, d, &query_perm);
    permute_col_blocks(&mut l.o, hd, d, &query_perm);
    Ok(())
}

/// Reorders blocks of `block` rows (each row `row_len` long) by `perm`.
fn permute_row_blocks(w: &mut [f32], block: usize, row_len: usize, perm: &[usize]) {
    let old = w.to_vec();
    for (i, &p) in perm.iter().enumerate() {
        let src = p * block * row_len;
        let dst = i * block * row_len;
        w[dst..dst + block * row_len].copy_from_slice(&old[src..src + block * row_len]);
    }
}

/// Reorders blocks of `block` columns of a `rows x (perm.len() * block)` matrix by `perm`.
fn permute_col_blocks(w: &mut [f32], block: usize, rows: usize, perm: &[usize]) {
    let width = perm.len() * block;
    let old = w.to_vec();
    for r in 0..rows {
        for (i, &p) in perm.iter().enumerate() {
            for c in 0..block {
                w[r * width + i * block + c] = old[r * width + p * block + c];
            }
        }
    }
}

/// Folds every RMS-norm scale into the matrices that read the normalised vector, and sets the
/// scales to one. The output head is untied from the embedding first (a copy), so the final
/// scale can fold into it. Shortcut: untying copies the embedding, which doubles that memory
/// for tied models.
pub fn fold_norms(model: &mut LlamaModel) -> Result<()> {
    let d = model.config.hidden_size;
    for l in &mut model.layers {
        let attn = std::mem::replace(&mut l.attn_norm, vec![1.0; d]);
        scale_cols(&mut l.q, d, &attn);
        scale_cols(&mut l.k, d, &attn);
        scale_cols(&mut l.v, d, &attn);
        let mlp = std::mem::replace(&mut l.mlp_norm, vec![1.0; d]);
        scale_cols(&mut l.gate, d, &mlp);
        scale_cols(&mut l.up, d, &mlp);
    }
    let embed = model.embed.clone();
    let head = model.lm_head.get_or_insert(embed);
    let final_norm = std::mem::replace(&mut model.final_norm, vec![1.0; d]);
    scale_cols(head, d, &final_norm);
    Ok(())
}

/// Rotates the residual stream by `rot`. Every norm scale must already be one (see
/// [`fold_norms`]), because a per-channel scale does not commute with a rotation.
pub fn rotate_residual(model: &mut LlamaModel, rot: &Rotation) -> Result<()> {
    let norms_are_one = |v: &[f32]| v.iter().all(|x| *x == 1.0);
    if !model
        .layers
        .iter()
        .all(|l| norms_are_one(&l.attn_norm) && norms_are_one(&l.mlp_norm))
        || !norms_are_one(&model.final_norm)
    {
        bail!("fold the RMS-norm scales before rotating the residual stream");
    }
    if rot.dim != model.config.hidden_size {
        bail!(
            "rotation has dimension {}, the residual stream is {}",
            rot.dim,
            model.config.hidden_size
        );
    }
    rotate_rows(&mut model.embed, rot);
    for l in &mut model.layers {
        // Readers of the normalised residual: rows transform as Q w.
        for w in [&mut l.q, &mut l.k, &mut l.v, &mut l.gate, &mut l.up] {
            rotate_rows(w, rot);
        }
        // Writers into the residual: columns transform as Q c.
        for w in [&mut l.o, &mut l.down] {
            rotate_cols(w, rot);
        }
    }
    if let Some(head) = model.lm_head.as_mut() {
        rotate_rows(head, rot);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::llama::tests::tiny_model;

    const TOKENS: [u32; 7] = [1, 5, 9, 2, 30, 11, 4];

    /// The tiny model with non-trivial norm scales, so folding has something to fold.
    fn model_with_norms() -> LlamaModel {
        let mut m = tiny_model(0.3);
        let d = m.config.hidden_size;
        let scale =
            |k: usize| -> Vec<f32> { (0..d).map(|i| 0.6 + 0.05 * ((i + k) % 9) as f32).collect() };
        for (k, l) in m.layers.iter_mut().enumerate() {
            l.attn_norm = scale(k);
            l.mlp_norm = scale(k + 3);
        }
        m.final_norm = scale(7);
        m
    }

    /// Largest difference in the NLL of every position and in the final logits.
    fn max_gap(a: &LlamaModel, b: &LlamaModel) -> f32 {
        let nll = |m: &LlamaModel| m.forward_nll(&TOKENS).unwrap();
        let logits = |m: &LlamaModel| m.forward_logits(&TOKENS).unwrap();
        let nll_gap = nll(a)
            .iter()
            .zip(nll(b))
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        let logit_gap = logits(a)
            .iter()
            .zip(logits(b))
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        nll_gap.max(logit_gap)
    }

    fn some_rotation(d: usize) -> Rotation {
        let mut a = vec![0.0f32; d];
        let mut b = vec![0.0f32; d];
        let mut c = vec![0.0f32; d];
        for i in 0..d {
            a[i] = 1.0 + (i % 3) as f32;
            b[i] = ((i * 7) % 5) as f32 - 2.0;
            c[i] = if i % 4 == 0 { 1.0 } else { 0.25 };
        }
        Rotation::from_reflectors(d, &[a, b, c]).unwrap()
    }

    #[test]
    fn one_reflection_is_a_hand_checked_diagonal() {
        // I - 2 e1 e1^T flips the first coordinate and nothing else.
        let r = Rotation::from_reflectors(3, &[vec![2.0, 0.0, 0.0]]).unwrap();
        assert_eq!(r.matrix(), &[-1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
        let r = Rotation::from_reflectors(3, &[vec![1.0, 0.0, 0.0], vec![0.0, 0.0, 5.0]]).unwrap();
        assert_eq!(r.matrix(), &[-1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0]);
    }

    #[test]
    fn rotation_is_orthogonal() {
        let d = 16;
        let r = some_rotation(d);
        for i in 0..d {
            for j in 0..d {
                let dot: f32 = (0..d)
                    .map(|k| r.matrix()[i * d + k] * r.matrix()[j * d + k])
                    .sum();
                let want = if i == j { 1.0 } else { 0.0 };
                assert!((dot - want).abs() < 1e-5, "row {i}.{j} = {dot}");
            }
        }
    }

    #[test]
    fn rotation_rejects_bad_reflectors() {
        assert!(Rotation::from_reflectors(3, &[vec![0.0, 0.0, 0.0]]).is_err());
        assert!(Rotation::from_reflectors(3, &[vec![1.0, 0.0]]).is_err());
    }

    #[test]
    fn ffn_unit_permutation_is_exact() {
        let before = model_with_norms();
        let mut after = model_with_norms();
        let inter = after.intermediate_size;
        let perm: Vec<usize> = (0..inter).map(|i| (i * 5 + 3) % inter).collect();
        permute_ffn_units(&mut after, 0, &perm).unwrap();
        assert_ne!(
            before.layers[0].gate, after.layers[0].gate,
            "transform must change weights"
        );
        assert!(max_gap(&before, &after) < 1e-4);
    }

    #[test]
    fn ffn_permutation_without_its_down_columns_is_detected() {
        // Control: moving gate and up rows alone changes the output, so the oracle has teeth.
        let before = model_with_norms();
        let mut broken = model_with_norms();
        let inter = broken.intermediate_size;
        let d = broken.config.hidden_size;
        let perm: Vec<usize> = (0..inter).rev().collect();
        permute_rows(&mut broken.layers[0].gate, d, &perm);
        let gap = max_gap(&before, &broken);
        assert!(gap > 1e-3, "control gap {gap}");
    }

    #[test]
    fn kv_group_permutation_is_exact_and_rejects_non_permutations() {
        let before = model_with_norms();
        let mut after = model_with_norms();
        permute_kv_groups(&mut after, 1, &[1, 0]).unwrap();
        assert_ne!(before.layers[1].k, after.layers[1].k);
        assert!(max_gap(&before, &after) < 1e-4);
        assert!(permute_kv_groups(&mut after, 1, &[0, 0]).is_err());
        assert!(permute_kv_groups(&mut after, 1, &[0]).is_err());
        assert!(permute_kv_groups(&mut after, 9, &[1, 0]).is_err());
    }

    #[test]
    fn norm_folding_is_exact_and_leaves_unit_scales() {
        let before = model_with_norms();
        let mut after = model_with_norms();
        fold_norms(&mut after).unwrap();
        assert!(after
            .layers
            .iter()
            .all(|l| l.attn_norm.iter().all(|x| *x == 1.0)));
        assert!(after.final_norm.iter().all(|x| *x == 1.0));
        assert!(after.lm_head.is_some());
        assert!(max_gap(&before, &after) < 1e-4);
    }

    #[test]
    fn residual_rotation_is_exact_after_folding() {
        let before = model_with_norms();
        let mut after = model_with_norms();
        fold_norms(&mut after).unwrap();
        let rot = some_rotation(after.config.hidden_size);
        rotate_residual(&mut after, &rot).unwrap();
        assert_ne!(before.embed, after.embed, "transform must change weights");
        assert!(max_gap(&before, &after) < 1e-4);
    }

    #[test]
    fn residual_rotation_without_folding_is_refused_and_changes_the_output() {
        // Control: rotating through unfolded norm scales would change the logits, so the
        // transform refuses; the same rotation applied by hand shows the gap is real.
        let before = model_with_norms();
        let mut refused = model_with_norms();
        let rot = some_rotation(refused.config.hidden_size);
        assert!(rotate_residual(&mut refused, &rot).is_err());

        let mut by_hand = model_with_norms();
        rotate_rows(&mut by_hand.embed, &rot);
        let gap = max_gap(&before, &by_hand);
        assert!(gap > 1e-2, "control gap {gap}");
    }

    #[test]
    fn all_transforms_together_are_exact() {
        let before = model_with_norms();
        let mut after = model_with_norms();
        let perm: Vec<usize> = (0..32).map(|i| (i * 5 + 3) % 32).collect();
        permute_ffn_units(&mut after, 0, &perm).unwrap();
        fold_norms(&mut after).unwrap();
        permute_kv_groups(&mut after, 0, &[1, 0]).unwrap();
        let rot = some_rotation(after.config.hidden_size);
        rotate_residual(&mut after, &rot).unwrap();
        assert!(max_gap(&before, &after) < 1e-4);
    }
}
