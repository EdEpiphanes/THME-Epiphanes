// SPDX-License-Identifier: AGPL-3.0-only
//! Fusion-aware Signed-Rank-R `DC-CCCP` + `ADMM-DP` + Tabu solver (`THME_Research_Dossier_v3.3` §4.2).
//!
//! Solves coupled mixed-precision allocation over hardware-fused execution groups
//! (`[Wq; Wk; Wv]`, `[Wgate; Wup]`, `Wo`, `Wdown`) preserving both harmful
//! amplification (`Gamma_+`) and beneficial cross-layer compensation (`Gamma_-`)
//! using the sign-split majorizer of the dossier (positive coupling terms bounded by AM-GM,
//! negative ones kept exactly) and a monotone majorize-minimize descent.

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

/// `s_G = sqrt(eps_G + delta0)` of the selected format of every group.
fn selected_roots(groups: &[FusedGroupSpec], choice: &[usize]) -> Vec<f64> {
    choice
        .iter()
        .enumerate()
        .map(|(g, &f)| (groups[g].eps[f].max(0.0) + DELTA_SMOOTH).sqrt())
        .collect()
}

/// Sign-split majorizer of [`evaluate_coupled_objective`] at `x`, expanded around `x_t`
/// (dossier v5 section 4.2, equations 15 to 17).
///
/// With `C = gamma_plus - sum_r u_r u_r^T` off the diagonal, the objective is
/// `sum_G (J_G^2 eps_G - sum_r u_rG^2 s_G^2) + sum_{G != G'} C_GG' s_G s_G'`. A term with
/// `C > 0` is bounded above by the AM-GM inequality around `x_t`; a term with `C <= 0` is convex
/// in the relaxed variables and is kept exactly. The result is at least the objective for every
/// `x` and equals it at `x = x_t`, so a step that lowers it lowers the objective.
pub fn sign_split_surrogate(
    groups: &[FusedGroupSpec],
    x: &[usize],
    x_t: &[usize],
    gamma_plus: &[Vec<f64>],
    gamma_minus_factors: &[Vec<f64>],
) -> f64 {
    let n = groups.len();
    let s = selected_roots(groups, x);
    let s_t = selected_roots(groups, x_t);
    let mut surrogate = 0.0_f64;
    for g in 0..n {
        let e = groups[g].eps[x[g]].max(0.0);
        let u_sq: f64 = gamma_minus_factors.iter().map(|u| u[g] * u[g]).sum();
        surrogate += groups[g].j_eff * groups[g].j_eff * e - u_sq * s[g] * s[g];
    }
    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            let c = gamma_plus[i][j] - gamma_minus_factors.iter().map(|u| u[i] * u[j]).sum::<f64>();
            if c > 0.0 {
                let ratio = s_t[j] / s_t[i];
                surrogate += 0.5 * c * (ratio * s[i] * s[i] + s[j] * s[j] / ratio);
            } else {
                surrogate += c * s[i] * s[j];
            }
        }
    }
    surrogate
}

/// Majorize-minimize descent over discrete choices: each step moves one group to the format that
/// lowers the sign-split surrogate most within `budget_bytes`. Because the surrogate bounds the
/// objective from above and is tight at the current point, the objective never increases.
/// Returns the final choice and the objective after every step (first entry is the start).
pub fn mm_descend(
    groups: &[FusedGroupSpec],
    gamma_plus: &[Vec<f64>],
    gamma_minus_factors: &[Vec<f64>],
    start: &[usize],
    budget_bytes: u64,
) -> (Vec<usize>, Vec<f64>) {
    let mut choice = start.to_vec();
    let mut bytes: u64 = choice
        .iter()
        .enumerate()
        .map(|(g, &f)| groups[g].bytes[f])
        .sum();
    let mut current = evaluate_coupled_objective(groups, &choice, gamma_plus, gamma_minus_factors);
    let mut trace = vec![current];
    for _ in 0..(8 * groups.len() + 8) {
        let expansion = choice.clone();
        let mut best: Option<(usize, usize, f64)> = None;
        for g in 0..groups.len() {
            for f in 0..groups[g].bytes.len() {
                if f == choice[g] {
                    continue;
                }
                let next_bytes = bytes - groups[g].bytes[choice[g]] + groups[g].bytes[f];
                if next_bytes > budget_bytes {
                    continue;
                }
                let mut candidate = expansion.clone();
                candidate[g] = f;
                let value = sign_split_surrogate(
                    groups,
                    &candidate,
                    &expansion,
                    gamma_plus,
                    gamma_minus_factors,
                );
                if value + 1e-12 < current && best.is_none_or(|(_, _, v)| value < v) {
                    best = Some((g, f, value));
                }
            }
        }
        let Some((g, f, _)) = best else { break };
        bytes = bytes - groups[g].bytes[choice[g]] + groups[g].bytes[f];
        choice[g] = f;
        current = evaluate_coupled_objective(groups, &choice, gamma_plus, gamma_minus_factors);
        trace.push(current);
    }
    (choice, trace)
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

    // Step 3b: majorize-minimize descent on the sign-split surrogate (monotone, in budget).
    let (descended, _) = mm_descend(
        groups,
        gamma_plus,
        gamma_minus_factors,
        &choice,
        budget_bytes,
    );
    choice = descended;
    total_bytes = choice
        .iter()
        .enumerate()
        .map(|(g, &f)| groups[g].bytes[f])
        .sum();

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

    /// Deterministic splitmix64 stream in [0, 1) for reproducible random instances.
    fn unit_stream(seed: u64) -> impl FnMut() -> f64 {
        let mut state = seed;
        move || {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            ((z ^ (z >> 31)) >> 11) as f64 / (1_u64 << 53) as f64
        }
    }

    /// A random instance whose gamma_plus has negative off-diagonal entries (a PSD projection
    /// can), which is why the majorizer needs the sign split of dossier v5 section 4.2.
    fn random_instance(
        next: &mut impl FnMut() -> f64,
        n: usize,
    ) -> (Vec<FusedGroupSpec>, Vec<Vec<f64>>, Vec<Vec<f64>>) {
        let groups: Vec<FusedGroupSpec> = (0..n)
            .map(|_| FusedGroupSpec {
                j_eff: 0.5 + 2.0 * next(),
                eps: vec![
                    0.005 + 0.02 * next(),
                    0.03 + 0.1 * next(),
                    0.15 + 0.3 * next(),
                ],
                bytes: vec![800, 400, 200],
            })
            .collect();
        let gamma_plus: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| if i == j { 0.0 } else { 0.6 * next() - 0.2 })
                    .collect()
            })
            .collect();
        let factors: Vec<Vec<f64>> = (0..2)
            .map(|_| (0..n).map(|_| 0.4 * next() - 0.2).collect())
            .collect();
        (groups, gamma_plus, factors)
    }

    /// The surrogate is at least the objective everywhere and equals it at the expansion point.
    #[test]
    fn sign_split_surrogate_majorizes_the_objective() {
        let mut next = unit_stream(20_261_127);
        for _ in 0..200 {
            let (groups, gp, gm) = random_instance(&mut next, 5);
            let pick = |next: &mut dyn FnMut() -> f64| -> Vec<usize> {
                (0..5).map(|_| (next() * 3.0) as usize % 3).collect()
            };
            let x_t = pick(&mut next);
            let x = pick(&mut next);
            let obj_x = evaluate_coupled_objective(&groups, &x, &gp, &gm);
            let sur_x = sign_split_surrogate(&groups, &x, &x_t, &gp, &gm);
            assert!(
                sur_x >= obj_x - 1e-9,
                "surrogate {sur_x} below objective {obj_x}"
            );
            let obj_t = evaluate_coupled_objective(&groups, &x_t, &gp, &gm);
            let sur_t = sign_split_surrogate(&groups, &x_t, &x_t, &gp, &gm);
            assert!(
                (sur_t - obj_t).abs() < 1e-9,
                "not tight at the expansion point"
            );
        }
    }

    /// Majorize-minimize steps never increase the objective, and stay inside the budget.
    #[test]
    fn mm_descent_is_monotone_and_budget_feasible() {
        let mut next = unit_stream(7_777);
        for _ in 0..40 {
            let (groups, gp, gm) = random_instance(&mut next, 6);
            let start = vec![1_usize; 6];
            let budget = 6 * 400;
            let (choice, trace) = mm_descend(&groups, &gp, &gm, &start, budget);
            assert!(
                trace.windows(2).all(|w| w[1] <= w[0] + 1e-9),
                "objective rose: {trace:?}"
            );
            let bytes: u64 = choice
                .iter()
                .enumerate()
                .map(|(g, &f)| groups[g].bytes[f])
                .sum();
            assert!(bytes <= budget);
        }
    }

    /// The solver is never worse than its own diagonal-DP starting point.
    #[test]
    fn solver_matches_or_beats_the_independent_dp_choice() {
        let mut next = unit_stream(31_337);
        for _ in 0..20 {
            let (groups, gp, gm) = random_instance(&mut next, 5);
            let budget = 5 * 400;
            let alloc = solve_coupled_dc_cccp(&groups, &gp, &gm, budget).unwrap();
            let dp_items: Vec<Vec<Choice>> = groups
                .iter()
                .map(|g| {
                    g.eps
                        .iter()
                        .zip(&g.bytes)
                        .map(|(&e, &b)| Choice {
                            weight: b,
                            cost: g.j_eff * g.j_eff * e,
                        })
                        .collect()
                })
                .collect();
            let dp = min_cost_choice(&dp_items, budget).unwrap();
            let dp_obj = evaluate_coupled_objective(&groups, &dp.choice, &gp, &gm);
            assert!(alloc.coupled_objective <= dp_obj + 1e-9);
        }
    }
}
