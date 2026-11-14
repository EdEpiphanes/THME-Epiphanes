// SPDX-License-Identifier: AGPL-3.0-only
//! ODE steppers (dossier WP6): explicit fixed-step methods (Euler, RK4) and the
//! adaptive Dormand-Prince 5(4) driver.

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

// Dormand-Prince 5(4) tableau: nodes `c`, the strict lower-triangular `a` rows and the
// two weight rows, `b5` for the fifth-order solution and `b4` for the embedded
// fourth-order one. The coefficients are exact rationals written as divisions, so
// nothing depends on a decimal literal.
const DP_C: [f64; 7] = [0.0, 1.0 / 5.0, 3.0 / 10.0, 4.0 / 5.0, 8.0 / 9.0, 1.0, 1.0];
const DP_A2: [f64; 1] = [1.0 / 5.0];
const DP_A3: [f64; 2] = [3.0 / 40.0, 9.0 / 40.0];
const DP_A4: [f64; 3] = [44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0];
const DP_A5: [f64; 4] = [
    19372.0 / 6561.0,
    -25360.0 / 2187.0,
    64448.0 / 6561.0,
    -212.0 / 729.0,
];
const DP_A6: [f64; 5] = [
    9017.0 / 3168.0,
    -355.0 / 33.0,
    46732.0 / 5247.0,
    49.0 / 176.0,
    -5103.0 / 18656.0,
];
const DP_A7: [f64; 6] = [
    35.0 / 384.0,
    0.0,
    500.0 / 1113.0,
    125.0 / 192.0,
    -2187.0 / 6784.0,
    11.0 / 84.0,
];
const DP_B5: [f64; 7] = [
    35.0 / 384.0,
    0.0,
    500.0 / 1113.0,
    125.0 / 192.0,
    -2187.0 / 6784.0,
    11.0 / 84.0,
    0.0,
];
const DP_B4: [f64; 7] = [
    5179.0 / 57600.0,
    0.0,
    7571.0 / 16695.0,
    393.0 / 640.0,
    -92097.0 / 339200.0,
    187.0 / 2100.0,
    1.0 / 40.0,
];

/// One step of the embedded Dormand-Prince 5(4) pair from `(t, y)` with step size
/// `h`, returning `(y5, error)`: the fifth-order solution and the error estimate
/// `y5 - y4` (the fifth-order minus the embedded fourth-order solution), one entry
/// per state component. The step size, the state and every stage are validated
/// through [`eval`]; nothing defaults. The FSAL property of the tableau is not
/// exploited: the last stage is recomputed like any other.
pub fn dopri5_step(f: &Rhs, t: f64, y: &[f64], h: f64) -> EngineResult<(Vec<f64>, Vec<f64>)> {
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
    // Stage states: `y + h * sum_j a[i][j] * k[j]`. `stages` holds k1..k7; the last
    // row is the FSAL stage, recomputed here like any other.
    let mut stages: Vec<Vec<f64>> = Vec::with_capacity(7);
    stages.push(eval(f, t, y)?);
    let a_rows: [&[f64]; 6] = [&DP_A2, &DP_A3, &DP_A4, &DP_A5, &DP_A6, &DP_A7];
    for (i, a) in a_rows.iter().enumerate() {
        let mut state = y.to_vec();
        for (j, &aij) in a.iter().enumerate() {
            for (yi, ki) in state.iter_mut().zip(stages[j].iter()) {
                *yi += h * aij * ki;
            }
        }
        stages.push(eval(f, t + DP_C[i + 1] * h, &state)?);
    }
    let mut y5 = vec![0.0; y.len()];
    let mut y4 = vec![0.0; y.len()];
    for (i, &yi) in y.iter().enumerate() {
        let mut s5 = 0.0;
        let mut s4 = 0.0;
        for (s, k) in stages.iter().enumerate() {
            s5 += DP_B5[s] * k[i];
            s4 += DP_B4[s] * k[i];
        }
        y5[i] = yi + h * s5;
        y4[i] = yi + h * s4;
    }
    let error: Vec<f64> = y5
        .iter()
        .zip(y4.iter())
        .map(|(&high, &low)| high - low)
        .collect();
    ensure_finite("y_next", &y5)?;
    ensure_finite("error", &error)?;
    Ok((y5, error))
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

/// Tolerances and step-size bounds for [`integrate_adaptive`]. Every quantity is
/// explicit: no field has a default, because a silent default for a scientific
/// quantity would hide a choice the caller must make.
#[derive(Debug, Clone)]
pub struct AdaptiveConfig {
    /// Relative error tolerance of a step.
    pub rtol: f64,
    /// Absolute error tolerance of a step.
    pub atol: f64,
    /// Step size of the first attempt.
    pub h_init: f64,
    /// Smallest step size the driver may attempt; a rejected step below it fails.
    pub h_min: f64,
    /// Largest step size the driver may attempt.
    pub h_max: f64,
    /// Maximum number of attempted steps, accepted or rejected.
    pub max_steps: usize,
}

/// The outcome of [`integrate_adaptive`]: the final time (always `t1`), the state
/// there, and how many steps were accepted and rejected.
#[derive(Debug, Clone)]
pub struct Solution {
    pub t: f64,
    pub y: Vec<f64>,
    pub accepted: usize,
    pub rejected: usize,
}

/// The error norm of one step: the largest component of
/// `|e_i| / (atol + rtol * max(|y_i|, |y_new_i|))`.
fn error_norm(y: &[f64], y_new: &[f64], error: &[f64], rtol: f64, atol: f64) -> f64 {
    y.iter()
        .zip(y_new.iter())
        .zip(error.iter())
        .map(|((&yi, &yn), &ei)| ei.abs() / (atol + rtol * yi.abs().max(yn.abs())))
        .fold(0.0, f64::max)
}

/// The step size for the next attempt, `h * clamp(0.9 * norm^(-1/5), 0.2, 5.0)`,
/// capped at `h_max`. A zero norm grows by the cap 5.0; a non-finite norm shrinks
/// by the floor 0.2 instead of poisoning `h` with NaN.
fn next_step_size(h: f64, norm: f64, h_max: f64) -> f64 {
    let factor = if norm == 0.0 {
        5.0
    } else if norm.is_finite() {
        (0.9 * norm.powf(-0.2)).clamp(0.2, 5.0)
    } else {
        0.2
    };
    (h * factor).min(h_max)
}

/// Integrates `y' = f(t, y)` from `(t0, y0)` to `t1` with the embedded
/// Dormand-Prince 5(4) pair and adaptive step control. Forward only. A step is
/// accepted when its error norm is at most 1; the next step is
/// `h * clamp(0.9 * norm^(-1/5), 0.2, 5.0)`, clipped to `h_max` and to the
/// remaining interval, so the final step lands exactly on `t1`.
pub fn integrate_adaptive(
    f: &Rhs,
    t0: f64,
    y0: &[f64],
    t1: f64,
    cfg: &AdaptiveConfig,
) -> EngineResult<Solution> {
    if !cfg.rtol.is_finite() || cfg.rtol <= 0.0 {
        return Err(EngineError::InvalidInput(format!(
            "rtol must be finite and positive, got {}",
            cfg.rtol
        )));
    }
    if !cfg.atol.is_finite() || cfg.atol < 0.0 {
        return Err(EngineError::InvalidInput(format!(
            "atol must be finite and non-negative, got {}",
            cfg.atol
        )));
    }
    if !cfg.h_init.is_finite() || !cfg.h_min.is_finite() || !cfg.h_max.is_finite() {
        return Err(EngineError::InvalidInput(
            "h_init, h_min and h_max must be finite".to_string(),
        ));
    }
    if !(0.0 < cfg.h_min && cfg.h_min <= cfg.h_init && cfg.h_init <= cfg.h_max) {
        return Err(EngineError::InvalidInput(format!(
            "step sizes must satisfy 0 < h_min <= h_init <= h_max, got {}, {}, {}",
            cfg.h_min, cfg.h_init, cfg.h_max
        )));
    }
    if cfg.max_steps == 0 {
        return Err(EngineError::InvalidInput(
            "max_steps must be at least 1".to_string(),
        ));
    }
    if !t0.is_finite() || !t1.is_finite() {
        return Err(EngineError::InvalidInput(format!(
            "t0 and t1 must be finite, got {t0} and {t1}"
        )));
    }
    if t1 <= t0 {
        return Err(EngineError::InvalidInput(format!(
            "the adaptive driver integrates forward only, got t0 = {t0} and t1 = {t1}"
        )));
    }
    if y0.is_empty() {
        return Err(EngineError::InvalidInput(
            "the state y0 must not be empty".to_string(),
        ));
    }
    ensure_finite("y0", y0)?;

    let mut t = t0;
    let mut y = y0.to_vec();
    let mut h = cfg.h_init;
    let mut accepted = 0usize;
    let mut rejected = 0usize;
    while t < t1 {
        h = h.min(cfg.h_max).min(t1 - t);
        let (y_new, error) = dopri5_step(f, t, &y, h)?;
        let norm = error_norm(&y, &y_new, &error, cfg.rtol, cfg.atol);
        let accepted_step = norm <= 1.0;
        if accepted_step {
            t += h;
            y = y_new;
            accepted += 1;
        } else {
            rejected += 1;
        }
        if accepted + rejected >= cfg.max_steps && t < t1 {
            return Err(EngineError::NoConvergence {
                iterations: accepted + rejected,
                residual: norm,
            });
        }
        h = next_step_size(h, norm, cfg.h_max);
        if !accepted_step && h < cfg.h_min {
            return Err(EngineError::NoConvergence {
                iterations: accepted + rejected,
                residual: norm,
            });
        }
    }
    Ok(Solution {
        t: t1,
        y,
        accepted,
        rejected,
    })
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

    /// x' = v, v' = -x: the harmonic oscillator with the closed form
    /// (cos t, -sin t). Energy x^2 + v^2 is conserved by the exact flow.
    fn oscillator(_t: f64, y: &[f64]) -> Vec<f64> {
        vec![y[1], -y[0]]
    }

    /// RK4 on the oscillator over one period t = 2*pi, n = 20, 40, 80.
    ///
    /// The error ratio is asserted only for the 40 and 80 halvings, where the
    /// asymptotic regime holds; at n = 20 the leading neglected term still adds a
    /// few percent. Every check compares against the exact solution.
    #[test]
    fn rk4_order_is_four_on_the_harmonic_oscillator() {
        let period = 2.0 * std::f64::consts::PI;
        let error_at = |n: usize| {
            let got = integrate_fixed(FixedMethod::Rk4, &oscillator, 0.0, &[1.0, 0.0], period, n)
                .unwrap();
            ((got[0] - period.cos()).powi(2) + (got[1] + period.sin()).powi(2)).sqrt()
        };
        let mut previous: Option<f64> = None;
        for n in [20usize, 40, 80] {
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

    /// Euler gains energy on the oscillator: the discrete flow multiplies
    /// x^2 + v^2 by (1 + h^2) every step, so after 1000 steps it exceeds 1.
    /// The exact energy is 1.
    #[test]
    fn euler_gains_energy_on_the_harmonic_oscillator() {
        let period = 2.0 * std::f64::consts::PI;
        let got = integrate_fixed(
            FixedMethod::Euler,
            &oscillator,
            0.0,
            &[1.0, 0.0],
            period,
            1000,
        )
        .unwrap();
        let energy = got[0].powi(2) + got[1].powi(2);
        assert!(
            energy > 1.0,
            "Euler energy {energy} at n = 1000 is not greater than 1"
        );
    }

    /// RK4 keeps the oscillator's energy within 1e-8 of 1 at n = 200; the exact
    /// energy is 1.
    #[test]
    fn rk4_conserves_energy_on_the_harmonic_oscillator() {
        let period = 2.0 * std::f64::consts::PI;
        let got =
            integrate_fixed(FixedMethod::Rk4, &oscillator, 0.0, &[1.0, 0.0], period, 200).unwrap();
        let energy = got[0].powi(2) + got[1].powi(2);
        assert!(
            (energy - 1.0).abs() <= 1.0e-8,
            "RK4 energy {energy} at n = 200 is not within 1e-8 of 1"
        );
    }

    /// Backward integration (t1 < t0) on y' = -y: y(0) = 1 from t0 = 0 to
    /// t1 = -1 returns exp(+1), the exact backward solution. A negative step size
    /// must reach the same closed form as forward integration does.
    #[test]
    fn backward_integration_follows_the_closed_form() {
        let exact = 1.0f64.exp();
        let got = integrate_fixed(FixedMethod::Euler, &decay, 0.0, &[1.0], -1.0, 1000).unwrap()[0];
        assert!(
            (got - exact).abs() < 1.0e-2,
            "Euler backward value {got} is not within 1e-2 of exp(1) = {exact}"
        );
        let got = integrate_fixed(FixedMethod::Rk4, &decay, 0.0, &[1.0], -1.0, 100).unwrap()[0];
        assert!(
            (got - exact).abs() < 1.0e-9,
            "RK4 backward value {got} is not within 1e-9 of exp(1) = {exact}"
        );
    }

    /// Observed orders on the backward decay, with the same tolerances as the
    /// forward tests: Euler about 1 (ratio 2), RK4 about 4 (ratio 16).
    #[test]
    fn backward_integration_keeps_the_forward_orders() {
        let exact = 1.0f64.exp();
        for (method, expected_ratio, tolerance) in [
            (FixedMethod::Euler, 2.0, 0.05),
            (FixedMethod::Rk4, 16.0, 0.1),
        ] {
            let mut previous: Option<f64> = None;
            for n in [100usize, 200, 400] {
                let got = integrate_fixed(method, &decay, 0.0, &[1.0], -1.0, n).unwrap()[0];
                let error = (got - exact).abs();
                if let Some(prev) = previous {
                    let ratio = prev / error;
                    assert!(
                        (ratio - expected_ratio).abs() <= tolerance * expected_ratio,
                        "backward error ratio {ratio} at n = {n} is not within {}% of {expected_ratio}",
                        tolerance * 100.0
                    );
                }
                previous = Some(error);
            }
        }
    }

    /// Fixed-step Dormand-Prince 5(4) on y' = -y, y(0) = 1, [0, 1], exact exp(-1):
    /// n = 4, 8, 16, 32.
    ///
    /// The pair's global error falls by about 32 per halving: the measured ratios
    /// are 39.1, 35.5 and 33.7, approaching 32 from above as the error series
    /// converges on its leading `h^5` term, so each halving carries its own band.
    /// The scan stops at n = 32 (error 3.2e-12); at n = 64 the ~1e-13
    /// floating-point floor takes over and the ratios stop being a clean order
    /// measurement.
    #[test]
    fn dopri5_order_is_five_on_the_decay_equation() {
        let exact = (-1.0f64).exp();
        let error_at = |n: usize| {
            let h = 1.0 / n as f64;
            let mut t = 0.0;
            let mut y = vec![1.0];
            for _ in 0..n {
                let (next, _) = dopri5_step(&decay, t, &y, h).unwrap();
                y = next;
                t += h;
            }
            (y[0] - exact).abs()
        };
        let ratios = [
            (4usize, 8usize, 40.0, 0.3),
            (8, 16, 36.0, 0.2),
            (16, 32, 33.0, 0.1),
        ];
        let mut errors: Vec<f64> = Vec::new();
        for (coarse, fine, expected, order_tolerance) in ratios {
            errors.push(error_at(coarse));
            errors.push(error_at(fine));
            let ratio = errors[errors.len() - 2] / errors[errors.len() - 1];
            assert!(
                (ratio - expected).abs() <= 0.05 * expected,
                "error ratio {ratio} at n = {fine} is not within 5% of {expected}"
            );
            let order = ratio.log2();
            assert!(
                (order - 5.0).abs() <= order_tolerance,
                "observed order {order} at n = {fine} is not within {order_tolerance} of 5"
            );
        }
    }

    /// One Dormand-Prince step with h = 0.1 on y' = -y from y = 1: the returned
    /// error estimate is within a factor 10 of the true local error of the embedded
    /// fourth-order solution, `|(y5 - error) - exp(-h)|` (observed ratio 1.04).
    ///
    /// The estimate is an `h^5` quantity dominated by the embedded solution's
    /// error, so it is compared with that; the fifth-order solution's own local
    /// error is 28x smaller at this step size (it is `O(h^6)`), which is why a
    /// factor-10 check against `|y5 - exp(-h)|` would need `h >= 0.5`.
    #[test]
    fn dopri5_error_estimate_tracks_the_local_error() {
        let h = 0.1;
        let (y5, error) = dopri5_step(&decay, 0.0, &[1.0], h).unwrap();
        let embedded = y5[0] - error[0];
        let true_local = (embedded - (-h).exp()).abs();
        let estimated = error[0].abs();
        assert!(
            estimated <= 10.0 * true_local,
            "estimate {estimated} exceeds 10x the local error {true_local}"
        );
        assert!(
            true_local <= 10.0 * estimated,
            "local error {true_local} exceeds 10x the estimate {estimated}"
        );
    }

    /// One Dormand-Prince step with h = 0.1 on the oscillator from (1, 0) is
    /// within 1e-8 of the closed form (cos h, -sin h).
    #[test]
    fn dopri5_matches_the_oscillator_closed_form() {
        let h = 0.1;
        let (y5, _) = dopri5_step(&oscillator, 0.0, &[1.0, 0.0], h).unwrap();
        let error = ((y5[0] - h.cos()).powi(2) + (y5[1] + h.sin()).powi(2)).sqrt();
        assert!(
            error < 1.0e-8,
            "Dormand-Prince oscillator error {error} is not below 1e-8"
        );
    }

    /// `dopri5_step` refuses the same bad input as `step`: a non-finite step size,
    /// an empty state, a state that is not finite, a wrong-length `f` and an
    /// infinite-valued `f` (as `NonFinite`, never as a NaN state).
    #[test]
    fn dopri5_refuses_bad_input() {
        let wrong_length = |_t: f64, _y: &[f64]| vec![1.0, 2.0];
        let infinite = |_t: f64, y: &[f64]| vec![f64::INFINITY * y[0]];
        assert!(matches!(
            dopri5_step(&decay, 0.0, &[1.0], f64::NAN),
            Err(EngineError::InvalidInput(_))
        ));
        assert!(matches!(
            dopri5_step(&decay, 0.0, &[1.0], 0.0),
            Err(EngineError::InvalidInput(_))
        ));
        assert!(matches!(
            dopri5_step(&decay, 0.0, &[], 1.0),
            Err(EngineError::InvalidInput(_))
        ));
        assert!(matches!(
            dopri5_step(&decay, 0.0, &[f64::NAN], 1.0),
            Err(EngineError::NonFinite { .. })
        ));
        assert!(matches!(
            dopri5_step(&wrong_length, 0.0, &[1.0], 1.0),
            Err(EngineError::InvalidInput(_))
        ));
        assert!(matches!(
            dopri5_step(&infinite, 0.0, &[1.0], 1.0),
            Err(EngineError::NonFinite { .. })
        ));
    }

    fn adaptive_config(rtol: f64) -> AdaptiveConfig {
        AdaptiveConfig {
            rtol,
            atol: rtol * 1.0e-2,
            h_init: 0.1,
            h_min: 1.0e-12,
            h_max: 1.0,
            max_steps: 100_000,
        }
    }

    /// Tolerance honoured: y' = -y on [0, 5] and the oscillator on [0, 10*pi],
    /// rtol = 1e-4, 1e-6, 1e-8 with atol = 1e-2 * rtol. The global error stays
    /// below `20 * rtol * |y_exact| + 20 * atol` (the factor 20 covers the
    /// accumulation of the per-step budget) and decreases as rtol tightens, and
    /// the accepted count grows.
    #[test]
    fn adaptive_tolerance_is_honoured_and_tightens() {
        let span = 10.0 * std::f64::consts::PI;
        let exact_norm = (span.cos().powi(2) + span.sin().powi(2)).sqrt();
        let mut decay_errors: Vec<f64> = Vec::new();
        let mut osc_errors: Vec<f64> = Vec::new();
        let mut accepted: Vec<usize> = Vec::new();
        for rtol in [1.0e-4, 1.0e-6, 1.0e-8] {
            let cfg = adaptive_config(rtol);
            let decay_sol = integrate_adaptive(&decay, 0.0, &[1.0], 5.0, &cfg).unwrap();
            let exact = (-5.0f64).exp();
            let decay_error = (decay_sol.y[0] - exact).abs();
            assert!(
                decay_error < 20.0 * rtol * exact.abs() + 20.0 * cfg.atol,
                "decay error {decay_error} at rtol {rtol} exceeds the 20x bound"
            );
            let osc_sol = integrate_adaptive(&oscillator, 0.0, &[1.0, 0.0], span, &cfg).unwrap();
            let osc_error =
                ((osc_sol.y[0] - span.cos()).powi(2) + (osc_sol.y[1] + span.sin()).powi(2)).sqrt();
            assert!(
                osc_error < 20.0 * rtol * exact_norm + 20.0 * cfg.atol,
                "oscillator error {osc_error} at rtol {rtol} exceeds the 20x bound"
            );
            decay_errors.push(decay_error);
            osc_errors.push(osc_error);
            accepted.push(decay_sol.accepted);
        }
        for pair in decay_errors.windows(2) {
            assert!(
                pair[1] < pair[0],
                "decay error did not shrink: {decay_errors:?}"
            );
        }
        for pair in osc_errors.windows(2) {
            assert!(
                pair[1] < pair[0],
                "oscillator error did not shrink: {osc_errors:?}"
            );
        }
        for pair in accepted.windows(2) {
            assert!(
                pair[1] > pair[0],
                "accepted count did not grow: {accepted:?}"
            );
        }
    }

    /// Rejection: y' = -50 y, y(0) = 1, h_init = 0.5 is far outside the stability
    /// region, so steps are rejected and the driver still reaches exp(-50) within
    /// tolerance. A smooth problem started with a small step rejects fewer.
    #[test]
    fn stiff_problem_rejects_steps_and_still_converges() {
        let stiff = |_t: f64, y: &[f64]| vec![-50.0 * y[0]];
        let cfg = AdaptiveConfig {
            rtol: 1.0e-6,
            atol: 1.0e-8,
            h_init: 0.5,
            h_min: 1.0e-12,
            h_max: 0.5,
            max_steps: 100_000,
        };
        let sol = integrate_adaptive(&stiff, 0.0, &[1.0], 1.0, &cfg).unwrap();
        assert!(sol.rejected > 0, "the stiff problem rejected no step");
        let exact = (-50.0f64).exp();
        assert!(
            (sol.y[0] - exact).abs() <= cfg.atol + cfg.rtol * exact.abs(),
            "stiff answer {} is not within tolerance of {exact}",
            sol.y[0]
        );
        let smooth_cfg = AdaptiveConfig {
            h_init: 0.01,
            h_max: 1.0,
            ..adaptive_config(1.0e-6)
        };
        let smooth = integrate_adaptive(&decay, 0.0, &[1.0], 1.0, &smooth_cfg).unwrap();
        assert!(
            smooth.rejected < sol.rejected,
            "smooth rejected {} steps, stiff {}",
            smooth.rejected,
            sol.rejected
        );
    }

    /// Lands exactly: with t1 - t0 not a multiple of h_init, the final step is
    /// clipped to the remaining interval and `Solution.t` is exactly `t1`.
    #[test]
    fn adaptive_lands_exactly_on_t1() {
        let cfg = AdaptiveConfig {
            h_init: 0.3,
            h_max: 0.3,
            ..adaptive_config(1.0e-6)
        };
        let sol = integrate_adaptive(&decay, 0.0, &[1.0], 1.0, &cfg).unwrap();
        assert_eq!(sol.t, 1.0);
    }

    /// Error paths: a step budget too small, an `h_min` the stiff problem cannot
    /// reach, bad configuration, a non-forward interval, and a solution that blows
    /// up. Every failure is a structured `EngineError`, never a panic.
    #[test]
    fn adaptive_reports_non_convergence_and_refuses_bad_config() {
        let good = adaptive_config(1.0e-6);
        let bounded = AdaptiveConfig {
            max_steps: 3,
            ..good.clone()
        };
        assert!(matches!(
            integrate_adaptive(&decay, 0.0, &[1.0], 5.0, &bounded),
            Err(EngineError::NoConvergence { iterations: 3, .. })
        ));

        let stiff = |_t: f64, y: &[f64]| vec![-50.0 * y[0]];
        let unreachable_h_min = AdaptiveConfig {
            h_init: 0.5,
            h_min: 0.4,
            h_max: 0.5,
            ..good.clone()
        };
        assert!(matches!(
            integrate_adaptive(&stiff, 0.0, &[1.0], 1.0, &unreachable_h_min),
            Err(EngineError::NoConvergence { .. })
        ));

        let zero_rtol = AdaptiveConfig {
            rtol: 0.0,
            ..good.clone()
        };
        let nan_rtol = AdaptiveConfig {
            rtol: f64::NAN,
            ..good.clone()
        };
        let inverted_h = AdaptiveConfig {
            h_min: 0.2,
            h_init: 0.1,
            ..good.clone()
        };
        for cfg in [&zero_rtol, &nan_rtol, &inverted_h] {
            assert!(matches!(
                integrate_adaptive(&decay, 0.0, &[1.0], 1.0, cfg),
                Err(EngineError::InvalidInput(_))
            ));
        }
        assert!(matches!(
            integrate_adaptive(&decay, 0.0, &[1.0], 0.0, &good),
            Err(EngineError::InvalidInput(_))
        ));
        assert!(matches!(
            integrate_adaptive(&decay, 1.0, &[1.0], 0.0, &good),
            Err(EngineError::InvalidInput(_))
        ));

        let blow_up = |_t: f64, y: &[f64]| vec![y[0] * y[0]];
        assert!(matches!(
            integrate_adaptive(&blow_up, 0.0, &[1.0], 2.0, &good),
            Err(EngineError::NoConvergence { .. }) | Err(EngineError::NonFinite { .. })
        ));
    }

    /// Cross-check with steps 01-02: on y' = -y over [0, 1], fixed-step RK4 with
    /// 2000 steps and the adaptive driver both match exp(-1) within their own
    /// bounds.
    #[test]
    fn adaptive_agrees_with_fixed_rk4_on_the_decay_equation() {
        let exact = (-1.0f64).exp();
        let fixed = integrate_fixed(FixedMethod::Rk4, &decay, 0.0, &[1.0], 1.0, 2000).unwrap()[0];
        let cfg = adaptive_config(1.0e-8);
        let adaptive = integrate_adaptive(&decay, 0.0, &[1.0], 1.0, &cfg)
            .unwrap()
            .y[0];
        let fixed_error = (fixed - exact).abs();
        let adaptive_error = (adaptive - exact).abs();
        assert!(
            fixed_error < 1.0e-10,
            "RK4 error {fixed_error} is not below 1e-10"
        );
        assert!(
            adaptive_error < 20.0 * cfg.rtol * exact.abs() + 20.0 * cfg.atol,
            "adaptive error {adaptive_error} exceeds the 20x bound"
        );
        assert!(
            (adaptive - fixed).abs() < 1.0e-6,
            "adaptive {adaptive} and RK4 {fixed} disagree"
        );
    }
}
