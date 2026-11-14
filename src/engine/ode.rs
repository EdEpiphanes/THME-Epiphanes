// SPDX-License-Identifier: AGPL-3.0-only
//! ODE steppers (dossier WP6): explicit fixed-step methods, Euler and RK4.

use super::error::{ensure_finite, EngineError, EngineResult};

/// Right-hand side of `y' = f(t, y)`: the derivative at time `t` of the state `y`.
pub type Rhs<'a> = dyn Fn(f64, &[f64]) -> Vec<f64> + 'a;

/// A fixed-step integration method.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FixedMethod {
    /// Forward Euler: `y + h f(t, y)`, first order.
    Euler,
    /// Classic 4-stage Runge-Kutta, fourth order.
    Rk4,
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
    let next: Vec<f64> = match method {
        FixedMethod::Euler => {
            let d = eval(f, t, y)?;
            y.iter()
                .zip(d.iter())
                .map(|(&yi, &di)| yi + h * di)
                .collect()
        }
        FixedMethod::Rk4 => {
            let k1 = eval(f, t, y)?;
            let k2 = eval(f, t + 0.5 * h, &add_scaled(y, &k1, 0.5 * h))?;
            let k3 = eval(f, t + 0.5 * h, &add_scaled(y, &k2, 0.5 * h))?;
            let k4 = eval(f, t + h, &add_scaled(y, &k3, h))?;
            y.iter()
                .zip(k1.iter())
                .zip(k2.iter())
                .zip(k3.iter())
                .zip(k4.iter())
                .map(|((((&yi, &k1i), &k2i), &k3i), &k4i)| {
                    yi + (h / 6.0) * (k1i + 2.0 * k2i + 2.0 * k3i + k4i)
                })
                .collect()
        }
    };
    ensure_finite("y_next", &next)?;
    Ok(next)
}

/// `y + c * d`, component-wise. The stage states of a stepper are not validated here:
/// `eval` has already checked the state it was handed, and `y_next` is checked after.
fn add_scaled(y: &[f64], d: &[f64], c: f64) -> Vec<f64> {
    y.iter()
        .zip(d.iter())
        .map(|(&yi, &di)| yi + c * di)
        .collect()
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

    /// RK4 with an infinite-valued `f` gives `NonFinite`; a wrong-length `f` gives
    /// `InvalidInput`.
    #[test]
    fn rk4_refuses_bad_derivatives() {
        let wrong_length = |_t: f64, _y: &[f64]| vec![1.0, 2.0];
        let infinite = |_t: f64, y: &[f64]| vec![f64::INFINITY * y[0]];
        assert!(matches!(
            step(FixedMethod::Rk4, &infinite, 0.0, &[1.0], 1.0),
            Err(EngineError::NonFinite { .. })
        ));
        assert!(matches!(
            step(FixedMethod::Rk4, &wrong_length, 0.0, &[1.0], 1.0),
            Err(EngineError::InvalidInput(_))
        ));
    }

    /// `step(Rk4, ..)` and `integrate_fixed(Rk4, .., 1)` agree bit for bit.
    #[test]
    fn rk4_one_step_and_the_integrator_agree_bit_for_bit() {
        let one = integrate_fixed(FixedMethod::Rk4, &decay, 0.0, &[1.0], 0.5, 1).unwrap();
        let direct = step(FixedMethod::Rk4, &decay, 0.0, &[1.0], 0.5).unwrap();
        assert_eq!(one, direct);
    }

    /// RK4 on y' = -y, y(0) = 1, [0, 1], exact exp(-1): n = 4, 8, 16, 32.
    ///
    /// The ratio and order assertions use n = 8, 16, 32, where the asymptotic
    /// regime has taken over: at n = 4 the error still has an O(h^5) share that
    /// inflates the first ratio (about 17.8 rather than 16). The n = 4 error is
    /// pinned separately, still about 1.5e-5, far above the ~1e-16 rounding floor,
    /// so it is not blurred by floating-point noise.
    #[test]
    fn rk4_order_is_four_on_the_decay_equation() {
        let exact = (-1.0f64).exp();
        let error_at = |n: usize| {
            let got = integrate_fixed(FixedMethod::Rk4, &decay, 0.0, &[1.0], 1.0, n).unwrap()[0];
            (got - exact).abs()
        };
        let first = error_at(4);
        assert!(
            (1.0e-6..=1.0e-4).contains(&first),
            "error at n = 4 is {first}, expected a value well above the rounding floor"
        );
        let mut previous: Option<f64> = None;
        for n in [8usize, 16, 32] {
            let error = error_at(n);
            if let Some(prev) = previous {
                let ratio = prev / error;
                assert!(
                    (ratio - 16.0).abs() <= 0.1 * 16.0,
                    "error ratio {ratio} at n = {n} is not within 10% of 16"
                );
                let order = ratio.log2();
                assert!(
                    (order - 4.0).abs() <= 0.1,
                    "observed order {order} at n = {n} is not within 0.1 of 4"
                );
            }
            previous = Some(error);
        }
    }

    #[test]
    fn one_step_and_the_integrator_agree_bit_for_bit() {
        let one = integrate_fixed(FixedMethod::Euler, &decay, 0.0, &[1.0], 0.5, 1).unwrap();
        let direct = step(FixedMethod::Euler, &decay, 0.0, &[1.0], 0.5).unwrap();
        assert_eq!(one, direct);
    }
}
