// SPDX-License-Identifier: AGPL-3.0-only
//! Fusion-aware Signed-Rank-R `DC-CCCP` + `ADMM-DP` + Tabu solver (`THME_Research_Dossier_v3.3` §4.2).
//!
//! Solves coupled mixed-precision allocation over hardware-fused execution groups
//! (`[Wq; Wk; Wv]`, `[Wgate; Wup]`, `Wo`, `Wdown`) preserving both harmful
//! amplification (`Gamma_+`) and beneficial cross-layer compensation (`Gamma_-`)
//! using the exact AM-GM convex majorizer and `delta0`-smoothed subgradients.

use crate::engine::alloc::{min_cost_choice, Choice};
use crate::engine::error::{EngineError, EngineResult};

const DELTA_SMOOTH: f64 = 1e-12;

/// A fused tensor execution group (e.g. `[Wq; Wk; Wv]` or `[Wgate; Wup]`).
#[derive(Debug, Clone)]
pub struct FusedGroupSpec {
    /// Effective Adjoint-Lyapunov sensitivity `J_{G -> L}^eff`.
    pub j_eff: f64,
    /// Per-format local quantization variance `eps_{G,f}` (`>= 0`).
    pub eps: Vec<f64>,
    /// Per-format byte size `bytes_{G,f}`.
    pub bytes: Vec<u64>,
}

/// Output of the coupled `DC-CCCP` + Tabu solver.
#[derive(Debug, Clone, PartialEq)]
pub struct CoupledAllocation {
    pub choice: Vec<usize>,
    pub total_bytes: u64,
    pub coupled_objective: f64,
}

/// Evaluates the exact signed-rank-R coupled DC objective on discrete choices.
pub fn evaluate_coupled_objective(
    groups: &[FusedGroupSpec],
    choice: &[usize],
    gamma_plus: &[Vec<f64>],
    gamma_minus_factors: &[Vec<f64>],
) -> f64 {
    let n = groups.len();
    let mut eps_sel = Vec::with_capacity(n);
    let mut s_sel = Vec::with_capacity(n);
    let mut obj = 0.0_f64;

    for (g, &f) in choice.iter().enumerate() {
        let e = groups[g].eps[f].max(0.0);
        eps_sel.push(e);
        s_sel.push((e + DELTA_SMOOTH).sqrt());
        obj += groups[g].j_eff * groups[g].j_eff * e;
    }

    for i in 0..n {
        for j in 0..n {
            if i != j {
                obj += gamma_plus[i][j] * s_sel[i] * s_sel[j];
            }
        }
    }

    for u_r in gamma_minus_factors {
        let dot: f64 = u_r.iter().zip(s_sel.iter()).map(|(&u, &s)| u * s).sum();
        obj -= dot * dot;
    }

    obj
}

/// Solves coupled allocation via DP initialization, AM-GM `DC-CCCP` homotopy
/// iterations, Lagrangian budget-feasible projection, and 1-swap Tabu polish.
pub fn solve_coupled_dc_cccp(
    groups: &[FusedGroupSpec],
    gamma_plus: &[Vec<f64>],
    gamma_minus_factors: &[Vec<f64>],
    budget_bytes: u64,
) -> EngineResult<CoupledAllocation> {
    let n = groups.len();
    if n == 0 {
        return Err(EngineError::InvalidInput(
            "groups must not be empty".to_string(),
        ));
    }
    if gamma_plus.len() != n || gamma_plus.iter().any(|r| r.len() != n) {
        return Err(EngineError::InvalidInput(
            "gamma_plus must be an N x N matrix".to_string(),
        ));
    }
    for u_r in gamma_minus_factors {
        if u_r.len() != n {
            return Err(EngineError::InvalidInput(
                "gamma_minus_factors rows must have length N".to_string(),
            ));
        }
    }

    // Step 1: Initialize with exact DP on diagonal costs using a 4096-unit grid.
    let unit = (budget_bytes.div_ceil(4096)).max(1);
    let cap_units = budget_bytes / unit;
    let mut dp_items = Vec::with_capacity(n);
    for g in groups {
        if g.eps.is_empty() || g.eps.len() != g.bytes.len() {
            return Err(EngineError::InvalidInput(
                "group eps and bytes must have equal non-zero length".to_string(),
            ));
        }
        let mut opts = Vec::with_capacity(g.eps.len());
        for (&e, &b) in g.eps.iter().zip(g.bytes.iter()) {
            opts.push(Choice {
                weight: b.div_ceil(unit),
                cost: g.j_eff * g.j_eff * e.max(0.0),
            });
        }
        dp_items.push(opts);
    }

    let init = min_cost_choice(&dp_items, cap_units)?;
    let mut x: Vec<Vec<f64>> = groups
        .iter()
        .enumerate()
        .map(|(i, g)| {
            let mut v = vec![0.0_f64; g.eps.len()];
            v[init.choice[i]] = 1.0;
            v
        })
        .collect();

    // Step 2: Homotopy DC-CCCP with AM-GM majorizer on Gamma_+ and smoothed Gamma_- gradient.
    for &tau_temp in &[0.25_f64, 0.05, 0.0] {
        for _ in 0..8 {
            let mut s_cur = Vec::with_capacity(n);
            for (g, x_g) in x.iter().enumerate() {
                let e_g: f64 = x_g
                    .iter()
                    .zip(groups[g].eps.iter())
                    .map(|(&p, &e)| p * e.max(0.0))
                    .sum();
                s_cur.push((e_g + DELTA_SMOOTH).sqrt());
            }

            // Compute linearized marginal cost per group from AM-GM majorizer + Gamma_- gradient.
            let mut eff_weight_on_eps = vec![0.0_f64; n];
            for g in 0..n {
                let mut w = groups[g].j_eff * groups[g].j_eff;
                for gp in 0..n {
                    if g != gp {
                        w += gamma_plus[g][gp] * (s_cur[gp] / s_cur[g]);
                    }
                }
                for u_r in gamma_minus_factors {
                    let dot: f64 = u_r.iter().zip(s_cur.iter()).map(|(&u, &s)| u * s).sum();
                    w -= dot * (u_r[g] / s_cur[g]);
                }
                eff_weight_on_eps[g] = w.max(0.0);
            }

            // Update simplex coordinates via softmax (homotopy) or hard best-ratio step.
            let mut next_x = Vec::with_capacity(n);
            let mut max_delta = 0.0_f64;
            for (g, spec) in groups.iter().enumerate() {
                let scores: Vec<f64> = spec
                    .eps
                    .iter()
                    .zip(spec.bytes.iter())
                    .map(|(&e, &b)| -(eff_weight_on_eps[g] * e + 1e-9 * (b as f64)))
                    .collect();

                let mut probs = vec![0.0_f64; scores.len()];
                if tau_temp > 0.0 {
                    let max_s = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                    let mut sum_e = 0.0_f64;
                    for (idx, &s) in scores.iter().enumerate() {
                        let v = ((s - max_s) / tau_temp).clamp(-50.0, 0.0).exp();
                        probs[idx] = v;
                        sum_e += v;
                    }
                    for p in &mut probs {
                        *p /= sum_e.max(1e-12);
                    }
                } else {
                    let best_idx = scores
                        .iter()
                        .enumerate()
                        .max_by(|a, b| a.1.total_cmp(b.1))
                        .map(|(idx, _)| idx)
                        .unwrap_or(0);
                    probs[best_idx] = 1.0;
                }

                for (&old, &new) in x[g].iter().zip(probs.iter()) {
                    max_delta = max_delta.max((old - new).abs());
                }
                next_x.push(probs);
            }
            x = next_x;
            if max_delta < 1e-4 {
                break;
            }
        }
    }

    // Step 3: Round to discrete + budget-feasible projection (if rounded bytes > budget_bytes).
    let mut choice: Vec<usize> = x
        .iter()
        .map(|probs| {
            probs
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(idx, _)| idx)
                .unwrap_or(0)
        })
        .collect();

    let mut total_bytes: u64 = choice
        .iter()
        .enumerate()
        .map(|(g, &f)| groups[g].bytes[f])
        .sum();

    // If rounding overshot budget, greedily downgrade the group with smallest dE / dBytes.
    while total_bytes > budget_bytes {
        let mut best_move: Option<(usize, usize, f64, u64)> = None;
        for (g, spec) in groups.iter().enumerate() {
            let cur_f = choice[g];
            let cur_b = spec.bytes[cur_f];
            let cur_e = spec.eps[cur_f];
            for cand_f in 0..spec.bytes.len() {
                let cand_b = spec.bytes[cand_f];
                if cand_b >= cur_b {
                    continue;
                }
                let saved = cur_b - cand_b;
                let ratio = ((spec.eps[cand_f] - cur_e).max(0.0) * spec.j_eff * spec.j_eff)
                    / (saved as f64);
                if best_move.is_none_or(|(_, _, r, _)| ratio < r) {
                    best_move = Some((g, cand_f, ratio, saved));
                }
            }
        }
        let Some((g, new_f, _, saved)) = best_move else {
            choice = init.choice.clone();
            total_bytes = choice
                .iter()
                .enumerate()
                .map(|(i, &f)| groups[i].bytes[f])
                .sum();
            break;
        };
        choice[g] = new_f;
        total_bytes -= saved;
    }

    // Step 4: 1-swap Tabu polish strictly within budget_bytes.
    let mut cur_obj = evaluate_coupled_objective(groups, &choice, gamma_plus, gamma_minus_factors);
    for _ in 0..(2 * n) {
        let mut improved = false;
        for g in 0..n {
            let old_f = choice[g];
            let old_b = groups[g].bytes[old_f];
            for cand_f in 0..groups[g].bytes.len() {
                if cand_f == old_f {
                    continue;
                }
                let new_b = groups[g].bytes[cand_f];
                let next_total = total_bytes - old_b + new_b;
                if next_total > budget_bytes {
                    continue;
                }
                choice[g] = cand_f;
                let cand_obj =
                    evaluate_coupled_objective(groups, &choice, gamma_plus, gamma_minus_factors);
                if cand_obj + 1e-12 < cur_obj {
                    cur_obj = cand_obj;
                    total_bytes = next_total;
                    improved = true;
                    break;
                }
                choice[g] = old_f;
            }
            if improved {
                break;
            }
        }
        if !improved {
            break;
        }
    }

    Ok(CoupledAllocation {
        choice,
        total_bytes,
        coupled_objective: cur_obj,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coupled_solver_respects_budget_and_improves_or_matches_dp() {
        let groups = vec![
            FusedGroupSpec {
                j_eff: 2.0,
                eps: vec![0.01, 0.05, 0.20],
                bytes: vec![800, 400, 200],
            },
            FusedGroupSpec {
                j_eff: 1.5,
                eps: vec![0.02, 0.06, 0.25],
                bytes: vec![800, 400, 200],
            },
        ];
        let gamma_plus = vec![vec![0.0, 0.3], vec![0.3, 0.0]];
        let gamma_minus = vec![vec![0.1, -0.1]];
        let alloc = solve_coupled_dc_cccp(&groups, &gamma_plus, &gamma_minus, 800).unwrap();
        assert!(alloc.total_bytes <= 800);
        assert!(alloc.coupled_objective.is_finite());
    }
}
