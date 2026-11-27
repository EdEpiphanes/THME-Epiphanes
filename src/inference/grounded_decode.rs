// SPDX-License-Identifier: AGPL-3.0-only
//! Epiphanes Anti-Hallucination, Grounded Decoding & Degeneration Guard (`THME_Research_Dossier_v3.3` §7.9).

use crate::engine::error::{EngineError, EngineResult};
use crate::quality::epiphanes::asymmetry_adaptive_renyi;
use std::collections::VecDeque;

/// Selects the premature contrast layer `l* = argmax J_alpha^AA(p_N || p_l)`
/// and computes `DoLa` contrasted log-probabilities strictly inside the
/// Adaptive Plausibility Constraint (`APC`, `p_N >= alpha_apc * max(p_N)`).
pub fn epiphanes_dola_contrast(
    p_final: &[f32],
    candidate_layers: &[&[f32]],
    alpha_apc: f32,
    beta_dola: f32,
) -> EngineResult<(usize, Vec<f32>)> {
    if p_final.is_empty() || candidate_layers.is_empty() {
        return Err(EngineError::InvalidInput(
            "epiphanes_dola_contrast requires non-empty distributions".to_string(),
        ));
    }
    let mut best_layer = 0_usize;
    let mut best_div = f64::NEG_INFINITY;
    for (idx, layer_probs) in candidate_layers.iter().enumerate() {
        if layer_probs.len() != p_final.len() {
            return Err(EngineError::InvalidInput(
                "candidate layer vocab size mismatch".to_string(),
            ));
        }
        let div = asymmetry_adaptive_renyi(p_final, 0.0, layer_probs, 0.0, 2.0, 1e-3)?;
        if div > best_div {
            best_div = div;
            best_layer = idx;
        }
    }

    let max_p = p_final.iter().copied().fold(0.0_f32, f32::max);
    let cutoff = alpha_apc * max_p;
    let p_premature = candidate_layers[best_layer];
    let mut out_logits = Vec::with_capacity(p_final.len());

    for (&pf, &pp) in p_final.iter().zip(p_premature.iter()) {
        if pf < cutoff {
            out_logits.push(f32::NEG_INFINITY);
        } else {
            let lp_f = pf.max(1e-12).ln();
            let lp_p = pp.max(1e-12).ln();
            out_logits.push(lp_f - beta_dola * lp_p);
        }
    }
    Ok((best_layer, out_logits))
}

/// Context-Aware Decoding (`CAD`) contrast for RAG faithfulness (`B=2` co-execution).
pub fn cad_contrast_logits(
    logits_with_ctx: &[f32],
    logits_no_ctx: &[f32],
    gamma_cad: f32,
) -> EngineResult<Vec<f32>> {
    if logits_with_ctx.len() != logits_no_ctx.len() || logits_with_ctx.is_empty() {
        return Err(EngineError::InvalidInput(
            "cad_contrast_logits requires equal non-empty logit slices".to_string(),
        ));
    }
    Ok(logits_with_ctx
        .iter()
        .zip(logits_no_ctx.iter())
        .map(|(&zc, &zn)| (1.0 + gamma_cad) * zc - gamma_cad * zn)
        .collect())
}

/// Zero-NLI Semantic Entropy over internal `RMSNorm` final-layer embeddings (`R102`).
/// Clusters embeddings by cosine similarity `>= cos_thresh` and returns Shannon entropy (nats).
pub fn semantic_entropy_from_embeddings(
    embeddings: &[&[f32]],
    cos_thresh: f32,
) -> EngineResult<f64> {
    if embeddings.is_empty() {
        return Err(EngineError::InvalidInput(
            "semantic_entropy_from_embeddings requires at least one sample".to_string(),
        ));
    }
    let dim = embeddings[0].len();
    let mut clusters: Vec<(Vec<f32>, usize)> = Vec::new();

    for &emb in embeddings {
        if emb.len() != dim {
            return Err(EngineError::InvalidInput(
                "embedding dimension mismatch in semantic_entropy".to_string(),
            ));
        }
        let norm: f32 = emb.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-8);
        let unit: Vec<f32> = emb.iter().map(|v| v / norm).collect();

        let mut assigned = false;
        for (center, count) in &mut clusters {
            let cos: f32 = center.iter().zip(unit.iter()).map(|(a, b)| a * b).sum();
            if cos >= cos_thresh {
                *count += 1;
                assigned = true;
                break;
            }
        }
        if !assigned {
            clusters.push((unit, 1));
        }
    }

    let total = embeddings.len() as f64;
    let mut se = 0.0_f64;
    for (_, count) in clusters {
        let p = (count as f64) / total;
        se -= p * p.ln();
    }
    Ok(se)
}

/// 4-gram hash ring buffer to detect repetitive attractor loops (`>= 3` repeats).
pub struct NgramLoopBreaker {
    recent_tokens: VecDeque<u32>,
    history_hashes: VecDeque<u64>,
    max_history: usize,
}

impl NgramLoopBreaker {
    pub fn new(max_history: usize) -> Self {
        Self {
            recent_tokens: VecDeque::with_capacity(4),
            history_hashes: VecDeque::with_capacity(max_history),
            max_history: max_history.max(16),
        }
    }

    /// Pushes `token` and returns `true` if the trailing 4-gram has appeared `>= 3` times.
    pub fn push_and_check_loop(&mut self, token: u32) -> bool {
        if self.recent_tokens.len() == 4 {
            self.recent_tokens.pop_front();
        }
        self.recent_tokens.push_back(token);
        if self.recent_tokens.len() < 4 {
            return false;
        }
        let hash = self
            .recent_tokens
            .iter()
            .fold(0xcbf29ce484222325_u64, |h, &t| {
                (h ^ u64::from(t)).wrapping_mul(0x100000001b3)
            });
        if self.history_hashes.len() == self.max_history {
            self.history_hashes.pop_front();
        }
        self.history_hashes.push_back(hash);
        self.history_hashes.iter().filter(|&&h| h == hash).count() >= 3
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dola_masks_below_apc_and_contrasts_divergent_layer() {
        let p_final = [0.7_f32, 0.25, 0.05];
        let l0 = [0.6_f32, 0.3, 0.1];
        let l1 = [0.2_f32, 0.7, 0.1];
        let (best, logits) = epiphanes_dola_contrast(&p_final, &[&l0, &l1], 0.1, 0.5).unwrap();
        assert_eq!(best, 1);
        // 0.05 < 0.1 * 0.7 = 0.07 -> masked to -inf.
        assert!(logits[2].is_infinite() && logits[2].is_sign_negative());
        assert!(logits[0].is_finite() && logits[1].is_finite());
    }

    #[test]
    fn loop_breaker_fires_on_the_third_repeat_of_a_four_gram_and_not_before() {
        // Tokens 1,2,3,4 repeated. The 4-gram (1,2,3,4) ends at positions 4, 8 and 12; every
        // other 4-gram (2,3,4,1), (3,4,1,2), (4,1,2,3) ends at 5/9, 6/10 and 7/11, so only
        // position 12 sees a gram for the third time.
        let mut breaker = NgramLoopBreaker::new(256);
        let fired: Vec<usize> = (0..12)
            .filter(|i| breaker.push_and_check_loop([1u32, 2, 3, 4][i % 4]))
            .map(|i| i + 1)
            .collect();
        assert_eq!(fired, vec![12]);
    }

    #[test]
    fn loop_breaker_ignores_a_sequence_with_no_repeated_four_gram() {
        let mut breaker = NgramLoopBreaker::new(256);
        assert!((1u32..=40).all(|t| !breaker.push_and_check_loop(t)));
    }

    #[test]
    fn semantic_entropy_is_zero_when_all_agree_and_positive_when_split() {
        let a = [1.0_f32, 0.0];
        let b = [0.0_f32, 1.0];
        assert!(semantic_entropy_from_embeddings(&[&a, &a, &a], 0.9).unwrap() < 1e-12);
        let se = semantic_entropy_from_embeddings(&[&a, &b], 0.9).unwrap();
        assert!((se - std::f64::consts::LN_2).abs() < 1e-6);
    }

    #[test]
    fn loop_breaker_triggers_on_third_4gram_repeat() {
        let mut lb = NgramLoopBreaker::new(64);
        for rep in 0..3 {
            for t in [10_u32, 20, 30, 40] {
                let tripped = lb.push_and_check_loop(t);
                if rep == 2 && t == 40 {
                    assert!(tripped);
                } else {
                    assert!(!tripped);
                }
            }
        }
    }
}
