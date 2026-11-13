// SPDX-License-Identifier: AGPL-3.0-only
//! Errors shared by every engine primitive (work package WP1).

use std::fmt;

/// Why an engine computation could not return a result.
#[derive(Debug, Clone, PartialEq)]
pub enum EngineError {
    /// An argument violates the documented interface (shape, range, empty input).
    InvalidInput(String),
    /// The problem is well formed but has no solution (for example a budget no
    /// allocation can meet).
    Infeasible(String),
    /// An iterative method stopped at its iteration limit before meeting its tolerance.
    NoConvergence { iterations: usize, residual: f64 },
    /// A value that must be finite is NaN or infinite.
    NonFinite { what: String, index: usize },
}

impl fmt::Display for EngineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInput(msg) => write!(f, "invalid input: {msg}"),
            Self::Infeasible(msg) => write!(f, "infeasible: {msg}"),
            Self::NoConvergence {
                iterations,
                residual,
            } => write!(
                f,
                "no convergence after {iterations} iterations (residual {residual:e})"
            ),
            Self::NonFinite { what, index } => {
                write!(f, "{what} has a non-finite value at index {index}")
            }
        }
    }
}

impl std::error::Error for EngineError {}

pub type EngineResult<T> = Result<T, EngineError>;

/// Fails with [`EngineError::NonFinite`] at the first NaN or infinity in `values`.
pub fn ensure_finite(what: &str, values: &[f64]) -> EngineResult<()> {
    match values.iter().position(|v| !v.is_finite()) {
        Some(index) => Err(EngineError::NonFinite {
            what: what.to_string(),
            index,
        }),
        None => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_variant_is_reachable_and_readable() {
        let cases = [
            (
                EngineError::InvalidInput("empty".into()),
                "invalid input: empty",
            ),
            (
                EngineError::Infeasible("budget 1 < 4".into()),
                "infeasible: budget 1 < 4",
            ),
            (
                EngineError::NoConvergence {
                    iterations: 50,
                    residual: 0.5,
                },
                "no convergence after 50 iterations",
            ),
            (
                EngineError::NonFinite {
                    what: "x".into(),
                    index: 2,
                },
                "x has a non-finite value at index 2",
            ),
        ];
        for (err, text) in cases {
            assert!(err.to_string().contains(text), "{err}");
        }
    }

    #[test]
    fn finite_guard_finds_the_first_bad_value() {
        assert_eq!(ensure_finite("x", &[1.0, -2.0, 0.0]), Ok(()));
        assert_eq!(ensure_finite("x", &[]), Ok(()));
        assert_eq!(
            ensure_finite("w", &[1.0, f64::NAN, f64::INFINITY]),
            Err(EngineError::NonFinite {
                what: "w".into(),
                index: 1
            })
        );
        assert!(ensure_finite("w", &[f64::NEG_INFINITY]).is_err());
        // Converts into the crate's anyhow errors at call sites.
        let as_anyhow: anyhow::Error = EngineError::Infeasible("x".into()).into();
        assert!(as_anyhow.to_string().contains("infeasible"));
    }
}
