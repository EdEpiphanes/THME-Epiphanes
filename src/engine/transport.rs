// SPDX-License-Identifier: AGPL-3.0-only
//! Optimal transport (dossier WP4): the northwest-corner starting plan, with marginal
//! validation shared by the Sinkhorn variants that follow.

use super::error::{ensure_finite, EngineError, EngineResult};

/// A transport plan on `a.len()` rows and `b.len()` columns.
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
}
