// SPDX-License-Identifier: AGPL-3.0-only
//! Optimal transport (dossier WP4): the northwest-corner starting plan and the
//! plain-domain Sinkhorn algorithm, with marginal validation and plan bookkeeping
//! shared by the log-domain variant that follows.

use super::error::{ensure_finite, EngineError, EngineResult};

/// A transport plan on `a.len()` rows and `b.len()` columns.
#[derive(Debug, Clone, PartialEq)]
pub struct Transport {
    /// Row-major plan, `rows * cols` entries.
    pub plan: Vec<f64>,
    /// Total cost, the sum of `plan[i][j] * cost[i][j]`.
    pub cost: f64,
    /// Iterations the solver used.
    pub iterations: usize,
    /// Largest absolute deviation of a row or column sum from `a` and `b`.
    pub marginal_error: f64,
}

fn validate_marginals(a: &[f64], b: &[f64]) -> EngineResult<()> {
    if a.is_empty() {
        return Err(EngineError::InvalidInput(
            "the row marginal a must not be empty".to_string(),
        ));
    }
    if b.is_empty() {
        return Err(EngineError::InvalidInput(
            "the column marginal b must not be empty".to_string(),
        ));
    }
    ensure_finite("a", a)?;
    ensure_finite("b", b)?;
    if a.iter().any(|&v| v < 0.0) {
        return Err(EngineError::InvalidInput(
            "the row marginal a has a negative entry".to_string(),
        ));
    }
    if b.iter().any(|&v| v < 0.0) {
        return Err(EngineError::InvalidInput(
            "the column marginal b has a negative entry".to_string(),
        ));
    }
    let sum_a = a.iter().sum::<f64>();
    let sum_b = b.iter().sum::<f64>();
    if sum_a <= 0.0 {
        return Err(EngineError::InvalidInput(format!(
            "the row marginal a must sum to a positive number, got {sum_a}"
        )));
    }
    if sum_b <= 0.0 {
        return Err(EngineError::InvalidInput(format!(
            "the column marginal b must sum to a positive number, got {sum_b}"
        )));
    }
    let scale = sum_a.max(sum_b);
    if (sum_a - sum_b).abs() > 1.0e-9 * scale {
        return Err(EngineError::InvalidInput(format!(
            "the marginals are not conserved: sum(a) = {sum_a}, sum(b) = {sum_b}; \
             mass is never renormalized silently"
        )));
    }
    Ok(())
}

pub fn northwest_corner(a: &[f64], b: &[f64]) -> EngineResult<Vec<f64>> {
    validate_marginals(a, b)?;
    let rows = a.len();
    let cols = b.len();
    let mut plan = vec![0.0; rows * cols];
    let mut remaining_b = b.to_vec();
    for (i, &row_mass) in a.iter().enumerate() {
        let mut remaining = row_mass;
        for (j, col_mass) in remaining_b.iter_mut().enumerate() {
            if remaining == 0.0 {
                break;
            }
            let take = remaining.min(*col_mass);
            plan[i * cols + j] = take;
            remaining -= take;
            *col_mass -= take;
        }
    }
    Ok(plan)
}

/// Validates a Sinkhorn problem: the marginals, the row-major cost matrix (one
/// finite entry per plan cell), and the three scientific parameters, none of which
/// has a default. Shared with the log-domain variant.
fn validate_problem(
    cost: &[f64],
    a: &[f64],
    b: &[f64],
    epsilon: f64,
    tol: f64,
    max_iter: usize,
) -> EngineResult<()> {
    validate_marginals(a, b)?;
    if cost.len() != a.len() * b.len() {
        return Err(EngineError::InvalidInput(format!(
            "the cost matrix has {} entries, the marginals imply {} ({} rows x {} columns)",
            cost.len(),
            a.len() * b.len(),
            a.len(),
            b.len()
        )));
    }
    ensure_finite("cost", cost)?;
    if !epsilon.is_finite() || epsilon <= 0.0 {
        return Err(EngineError::InvalidInput(format!(
            "epsilon must be finite and positive, got {epsilon}"
        )));
    }
    if !tol.is_finite() || tol <= 0.0 {
        return Err(EngineError::InvalidInput(format!(
            "tol must be finite and positive, got {tol}"
        )));
    }
    if max_iter == 0 {
        return Err(EngineError::InvalidInput(
            "max_iter must be at least 1".to_string(),
        ));
    }
    Ok(())
}

/// The shared bookkeeping: the total cost and the largest absolute row or column
/// deviation, and the [`Transport`] value. Shared with the log-domain variant.
fn finish(plan: Vec<f64>, cost: &[f64], a: &[f64], b: &[f64], iterations: usize) -> Transport {
    let cols = b.len();
    let mut marginal_error = 0.0f64;
    for (i, &want) in a.iter().enumerate() {
        let got: f64 = plan.iter().skip(i * cols).take(cols).copied().sum();
        marginal_error = marginal_error.max((got - want).abs());
    }
    for (j, &want) in b.iter().enumerate() {
        let got: f64 = plan.iter().skip(j).step_by(cols).copied().sum();
        marginal_error = marginal_error.max((got - want).abs());
    }
    let total_cost = plan.iter().zip(cost.iter()).map(|(&p, &c)| p * c).sum();
    Transport {
        plan,
        cost: total_cost,
        iterations,
        marginal_error,
    }
}

/// Plain-domain Sinkhorn on `cost` with marginals `a` and `b`. The kernel is
/// `K[i][j] = exp(-cost[i][j] / epsilon)`; the plan is `diag(u) K diag(v)`, and `u`
/// and `v` are rescaled to the row and column marginals alternately until the
/// largest absolute row or column deviation is at most `tol`.
///
/// Small `epsilon` drives the kernel entries below the f64 floor, so the plain
/// domain is refused the moment a row or column of the plan is identically zero or a
/// scaling is not finite: `NonFinite` or `Infeasible`, both naming the log-domain
/// variant. A finite answer is never a NaN.
pub fn sinkhorn(
    cost: &[f64],
    a: &[f64],
    b: &[f64],
    epsilon: f64,
    tol: f64,
    max_iter: usize,
) -> EngineResult<Transport> {
    validate_problem(cost, a, b, epsilon, tol, max_iter)?;
    let rows = a.len();
    let cols = b.len();
    // K = exp(-C/epsilon), row-major.
    let kernel: Vec<f64> = cost.iter().map(|&c| (-c / epsilon).exp()).collect();
    let mut u = vec![1.0; rows];
    let mut v = vec![1.0; cols];
    let mut iterations = 0usize;
    loop {
        // Rows: u_i <- a_i / sum_j K[i][j] v_j.
        for i in 0..rows {
            let mut sum = 0.0f64;
            for j in 0..cols {
                sum += kernel[i * cols + j] * v[j];
            }
            if !sum.is_finite() || sum <= 0.0 {
                return Err(EngineError::Infeasible(format!(
                    "the kernel row {i} scaled to {sum} at iteration {iterations}; \
                     epsilon {epsilon} is too small for the plain domain, use sinkhorn_log"
                )));
            }
            let next = a[i] / sum;
            if !next.is_finite() {
                return Err(EngineError::NonFinite {
                    what: "row scaling u".to_string(),
                    index: i,
                });
            }
            u[i] = next;
        }
        // Columns: v_j <- b_j / sum_i K[i][j] u_i.
        for j in 0..cols {
            let mut sum = 0.0f64;
            for i in 0..rows {
                sum += kernel[i * cols + j] * u[i];
            }
            if !sum.is_finite() || sum <= 0.0 {
                return Err(EngineError::Infeasible(format!(
                    "the kernel column {j} scaled to {sum} at iteration {iterations}; \
                     epsilon {epsilon} is too small for the plain domain, use sinkhorn_log"
                )));
            }
            let next = b[j] / sum;
            if !next.is_finite() {
                return Err(EngineError::NonFinite {
                    what: "column scaling v".to_string(),
                    index: j,
                });
            }
            v[j] = next;
        }
        iterations += 1;
        let mut plan = vec![0.0; rows * cols];
        for i in 0..rows {
            for j in 0..cols {
                plan[i * cols + j] = u[i] * kernel[i * cols + j] * v[j];
            }
        }
        let out = finish(plan, cost, a, b, iterations);
        if out.marginal_error <= tol {
            return Ok(out);
        }
        if iterations >= max_iter {
            return Err(EngineError::NoConvergence {
                iterations,
                residual: out.marginal_error,
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The classic 2x2 start: a = [0.3, 0.7], b = [0.5, 0.5] fills (0,0) with 0.3,
    /// (1,0) with the remaining 0.2 of column 0 and (1,1) with 0.5. The task's
    /// `[0.3, 0.0, 0.2, 0.5]` is the exact-math answer; in f64 the third entry is one
    /// ulp off (0.5 - 0.3), and both vectors are exactly feasible, so the entries are
    /// compared with a tolerance and the marginals to 1e-15.
    #[test]
    fn northwest_corner_starts_in_the_top_left() {
        let plan = northwest_corner(&[0.3, 0.7], &[0.5, 0.5]).unwrap();
        let want = [0.3, 0.0, 0.2, 0.5];
        for (i, &w) in want.iter().enumerate() {
            assert!(
                (plan[i] - w).abs() <= 1.0e-15,
                "entry {i}: got {}, want {}",
                plan[i],
                w
            );
        }
        // Feasible: the row and column sums are the marginals up to rounding (the
        // third entry is one ulp off the exact-math 0.2, which shifts one column
        // sum by one ulp too).
        assert!((plan[0] + plan[1] - 0.3).abs() <= 1.0e-15);
        assert!((plan[2] + plan[3] - 0.7).abs() <= 1.0e-15);
        assert!((plan[0] + plan[2] - 0.5).abs() <= 1.0e-15);
        assert!((plan[1] + plan[3] - 0.5).abs() <= 1.0e-15);
    }

    /// Seeded 5x7 problem; the row and column sums reproduce the marginals `a` and
    /// `b` to rounding.
    #[test]
    fn northwest_corner_is_exactly_feasible() {
        let (a, b) = marginals(5, 7, 2026);
        let plan = northwest_corner(&a, &b).unwrap();
        let cols = b.len();
        for (i, &want) in a.iter().enumerate() {
            let got: f64 = plan.iter().skip(i * cols).take(cols).copied().sum();
            assert!(
                (got - want).abs() <= 1.0e-12,
                "row {i}: sum {got}, marginal {want}"
            );
        }
        for (j, &want) in b.iter().enumerate() {
            let got: f64 = plan.iter().skip(j).step_by(cols).copied().sum();
            assert!(
                (got - want).abs() <= 1.0e-12,
                "column {j}: sum {got}, marginal {want}"
            );
        }
        // The start is a feasible flow, so its entries are non-negative.
        assert!(plan.iter().all(|&v| v >= 0.0));
    }

    /// Every refusal is a structured error: unequal mass, a negative entry and an
    /// empty `a` are `InvalidInput`; a NaN entry is `NonFinite`, because the shared
    /// `ensure_finite` guard reports it that way (the batch's other steps do the
    /// same). The task line says "all InvalidInput"; `ensure_finite` is mandated by
    /// the same task and returns `NonFinite`, so the guard wins.
    #[test]
    fn bad_marginals_are_refused() {
        assert!(matches!(
            northwest_corner(&[0.3, 0.8], &[0.5, 0.5]),
            Err(EngineError::InvalidInput(_))
        ));
        assert!(matches!(
            northwest_corner(&[0.3, -0.1], &[0.2, 0.0]),
            Err(EngineError::InvalidInput(_))
        ));
        assert!(matches!(
            northwest_corner(&[0.3, f64::NAN], &[0.5, 0.5]),
            Err(EngineError::NonFinite { .. })
        ));
        assert!(matches!(
            northwest_corner(&[], &[0.5, 0.5]),
            Err(EngineError::InvalidInput(_))
        ));
        // A mass that is not positive is refused too.
        assert!(matches!(
            northwest_corner(&[0.0, 0.0], &[0.5, 0.5]),
            Err(EngineError::InvalidInput(_))
        ));
        // Mass just inside the 1e-9 relative tolerance is accepted.
        let mut b = vec![0.5, 0.5];
        b[0] += 1.0e-12;
        b[1] -= 1.0e-12;
        assert!(northwest_corner(&[0.3, 0.7], &b).is_ok());
    }

    /// A seeded 5x7 pair of marginals with equal positive mass, like `seeded` in the
    /// `basis` tests.
    fn marginals(rows: usize, cols: usize, seed: u64) -> (Vec<f64>, Vec<f64>) {
        let mut s = seed;
        let mut next = || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (s >> 40) as f64 / 16_777_216.0
        };
        let a: Vec<f64> = (0..rows).map(|_| next() + 0.05).collect();
        let mut b: Vec<f64> = (0..cols).map(|_| next() + 0.05).collect();
        let sum = a.iter().sum::<f64>();
        let scale = sum / b.iter().sum::<f64>();
        for v in b.iter_mut() {
            *v *= scale;
        }
        (a, b)
    }

    /// A row-major cost matrix from the seeded generator, like `marginals`.
    fn cost(rows: usize, cols: usize, seed: u64) -> Vec<f64> {
        let mut s = seed;
        let mut next = || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (s >> 40) as f64 / 16_777_216.0
        };
        (0..rows * cols).map(|_| next()).collect()
    }

    /// The task's 2x2: a = b = [0.5, 0.5], cost [[0, 1], [1, 0]]. The closed form is
    /// plan = [[p, 0.5-p], [0.5-p, p]] with p = 0.5 / (1 + k), k = exp(-1/eps), and
    /// cost = 1 / (exp(1/eps) + 1), for eps = 1.0, 0.5, 0.2.
    #[test]
    fn sinkhorn_matches_the_two_by_two_closed_form() {
        let cost = [0.0, 1.0, 1.0, 0.0];
        for eps in [1.0, 0.5, 0.2] {
            let plan = sinkhorn(&cost, &[0.5, 0.5], &[0.5, 0.5], eps, 1.0e-12, 100_000).unwrap();
            let k = (-1.0 / eps).exp();
            let p = 0.5 / (1.0 + k);
            let want = [p, 0.5 - p, 0.5 - p, p];
            for (i, &w) in want.iter().enumerate() {
                assert!(
                    (plan.plan[i] - w).abs() <= 1.0e-9,
                    "eps {eps}: entry {i}: got {}, want {}",
                    plan.plan[i],
                    w
                );
            }
            let exact = 1.0 / ((1.0 / eps).exp() + 1.0);
            assert!(
                (plan.cost - exact).abs() <= 1.0e-9,
                "eps {eps}: cost {}, want {exact}",
                plan.cost
            );
            assert!(plan.iterations <= 100_000);
            assert!(plan.marginal_error <= 1.0e-12);
        }
    }

    /// Mass conservation on a fixed 5x7 problem (seeded LCG, non-uniform marginals of
    /// equal sums): the reported error and an independent recomputation from `plan`
    /// are both within `tol`, the total mass equals `sum(a)` and every entry is
    /// non-negative.
    #[test]
    fn sinkhorn_conserves_mass_on_a_seeded_problem() {
        let (a, b) = marginals(5, 7, 2026);
        let cost = cost(5, 7, 4242);
        let tol = 1.0e-10;
        let plan = sinkhorn(&cost, &a, &b, 1.0, tol, 100_000).unwrap();
        let cols = b.len();
        let mut error = 0.0f64;
        for (i, &want) in a.iter().enumerate() {
            let got: f64 = plan.plan.iter().skip(i * cols).take(cols).copied().sum();
            error = error.max((got - want).abs());
        }
        for (j, &want) in b.iter().enumerate() {
            let got: f64 = plan.plan.iter().skip(j).step_by(cols).copied().sum();
            error = error.max((got - want).abs());
        }
        assert!(
            plan.marginal_error <= tol,
            "reported error {} exceeds {tol}",
            plan.marginal_error
        );
        assert!(error <= tol, "recomputed error {error} exceeds {tol}");
        let total: f64 = plan.plan.iter().sum();
        assert!(
            (total - a.iter().sum::<f64>()).abs() <= tol,
            "total mass {total} is not sum(a)"
        );
        assert!(
            plan.plan.iter().all(|&v| v >= 0.0),
            "the plan has a negative entry"
        );
        assert!(
            (plan.cost
                - plan
                    .plan
                    .iter()
                    .zip(cost.iter())
                    .map(|(&p, &c)| p * c)
                    .sum::<f64>())
            .abs()
                <= 1.0e-9,
            "the reported cost does not match plan * cost"
        );
    }

    /// Errors: `epsilon = 0` and a cost matrix of the wrong length give
    /// `InvalidInput`; a NaN cost gives the variant `ensure_finite` returns
    /// (`NonFinite`); `max_iter = 1` on the 5x7 problem, whose one pass leaves a
    /// deviation above `tol`, gives `NoConvergence`.
    #[test]
    fn sinkhorn_refuses_bad_input_and_reports_non_convergence() {
        assert!(matches!(
            sinkhorn(
                &[0.0, 1.0, 1.0, 0.0],
                &[0.5, 0.5],
                &[0.5, 0.5],
                0.0,
                1.0e-9,
                100
            ),
            Err(EngineError::InvalidInput(_))
        ));
        assert!(matches!(
            sinkhorn(&[0.0, 1.0, 1.0], &[0.5, 0.5], &[0.5, 0.5], 1.0, 1.0e-9, 100),
            Err(EngineError::InvalidInput(_))
        ));
        assert!(matches!(
            sinkhorn(
                &[0.0, 1.0, 1.0, f64::NAN],
                &[0.5, 0.5],
                &[0.5, 0.5],
                1.0,
                1.0e-9,
                100
            ),
            Err(EngineError::NonFinite { .. })
        ));
        let (a, b) = marginals(5, 7, 2026);
        let cost = cost(5, 7, 4242);
        let err = sinkhorn(&cost, &a, &b, 1.0, 1.0e-12, 1).unwrap_err();
        match err {
            EngineError::NoConvergence {
                iterations,
                residual,
            } => {
                assert_eq!(iterations, 1);
                assert!(residual > 1.0e-12);
            }
            other => panic!("expected NoConvergence, got {other}"),
        }
    }
}
