// SPDX-License-Identifier: AGPL-3.0-only
//! ODE steppers (dossier WP6): explicit fixed-step methods, starting with Euler.

use super::error::{ensure_finite, EngineError, EngineResult};

/// Right-hand side of `y' = f(t, y)`: the derivative at time `t` of the state `y`.
pub type Rhs<'a> = dyn Fn(f64, &[f64]) -> Vec<f64> + 'a;

/// A fixed-step integration method. Step 02 adds `Rk4`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FixedMethod {
    /// Forward Euler: `y + h f(t, y)`, first order.
    Euler,
}

/// Evaluates `f(t, y)` and checks that the result is a derivative of `y`: one finite
/// entry per state component. Shared by every stepper.
fn eval(f: &Rhs, t: f64, y: &[f64]) -> EngineResult<Vec<f64>> {
    let d = f(t, y);
    if d.len() != y.len() {
        return Err(EngineError::InvalidInput(format!(
            "f(t, y) returned {} entries, the state has {}",
            d.len(),
            y.len()
        )));
    }
    ensure_finite("f(t,y)", &d)?;
    Ok(d)
}

/// One step of `method` from `(t, y)` with step size `h`. A negative `h` integrates
/// backwards. The step size, the state and the derivative are validated; nothing
/// defaults.
pub fn step(method: FixedMethod, f: &Rhs, t: f64, y: &[f64], h: f64) -> EngineResult<Vec<f64>> {
    if !h.is_finite() || h == 0.0 {
        return Err(EngineError::InvalidInput(format!(
            "step size must be finite and non-zero, got {h}"
        )));
    }
    if y.is_empty() {
        return Err(EngineError::InvalidInput(
            "the state y must not be empty".to_string(),
        ));
    }
    ensure_finite("y", y)?;
    let d = eval(f, t, y)?;
    let next: Vec<f64> = match method {
        FixedMethod::Euler => y
            .iter()
            .zip(d.iter())
            .map(|(&yi, &di)| yi + h * di)
            .collect(),
    };
    ensure_finite("y_next", &next)?;
    Ok(next)
}

/// Integrates `y' = f(t, y)` from `(t0, y0)` to `t1` in `steps` equal steps of
/// `h = (t1 - t0) / steps`. The time is `t0 + i * h`, never accumulated, so the
/// last step starts exactly on `t1 - h` and lands on `t1`.
pub fn integrate_fixed(
    method: FixedMethod,
    f: &Rhs,
    t0: f64,
    y0: &[f64],
    t1: f64,
    steps: usize,
) -> EngineResult<Vec<f64>> {
    if steps == 0 {
        return Err(EngineError::InvalidInput(
            "steps must be at least 1".to_string(),
        ));
    }
    if !t0.is_finite() || !t1.is_finite() {
        return Err(EngineError::InvalidInput(format!(
            "t0 and t1 must be finite, got {t0} and {t1}"
        )));
    }
    let h = (t1 - t0) / steps as f64;
    let mut t = t0;
    let mut y = y0.to_vec();
    for i in 0..steps {
        y = step(method, f, t, &y, h)?;
        t = t0 + (i as f64 + 1.0) * h;
    }
    Ok(y)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn decay(_t: f64, y: &[f64]) -> Vec<f64> {
        vec![-y[0]]
    }

    #[test]
    fn euler_order_is_one_on_the_decay_equation() {
        let exact = (-1.0f64).exp();
        let mut previous: Option<f64> = None;
        for n in [100usize, 200, 400, 800] {
            let got = integrate_fixed(FixedMethod::Euler, &decay, 0.0, &[1.0], 1.0, n).unwrap()[0];
            let error = (got - exact).abs();
            if let Some(prev) = previous {
                let ratio = prev / error;
                assert!(
                    (ratio - 2.0).abs() <= 0.05 * 2.0,
                    "error ratio {ratio} at n = {n} is not within 5% of 2"
                );
                let order = ratio.log2();
                assert!(
                    (order - 1.0).abs() <= 0.1,
                    "observed order {order} at n = {n} is not within 0.1 of 1"
                );
            }
            previous = Some(error);
        }
    }

    #[test]
    fn bad_input_is_refused() {
        let wrong_length = |_t: f64, _y: &[f64]| vec![1.0, 2.0];
        let infinite = |_t: f64, y: &[f64]| vec![f64::INFINITY * y[0]];
        assert!(matches!(
            integrate_fixed(FixedMethod::Euler, &decay, 0.0, &[1.0], 1.0, 0),
            Err(EngineError::InvalidInput(_))
        ));
        assert!(matches!(
            step(FixedMethod::Euler, &decay, 0.0, &[1.0], f64::NAN),
            Err(EngineError::InvalidInput(_))
        ));
        assert!(matches!(
            step(FixedMethod::Euler, &decay, 0.0, &[1.0], 0.0),
            Err(EngineError::InvalidInput(_))
        ));
        assert!(matches!(
            step(FixedMethod::Euler, &decay, 0.0, &[], 1.0),
            Err(EngineError::InvalidInput(_))
        ));
        assert!(matches!(
            step(FixedMethod::Euler, &decay, 0.0, &[f64::NAN], 1.0),
            Err(EngineError::NonFinite { .. })
        ));
        assert!(matches!(
            step(FixedMethod::Euler, &wrong_length, 0.0, &[1.0], 1.0),
            Err(EngineError::InvalidInput(_))
        ));
        // f returns infinity: refused as NonFinite, never propagated as a NaN state.
        assert!(matches!(
            step(FixedMethod::Euler, &infinite, 0.0, &[1.0], 1.0),
            Err(EngineError::NonFinite { .. })
        ));
    }

    #[test]
    fn one_step_and_the_integrator_agree_bit_for_bit() {
        let one = integrate_fixed(FixedMethod::Euler, &decay, 0.0, &[1.0], 0.5, 1).unwrap();
        let direct = step(FixedMethod::Euler, &decay, 0.0, &[1.0], 0.5).unwrap();
        assert_eq!(one, direct);
    }
}
