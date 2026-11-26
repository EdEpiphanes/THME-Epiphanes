// SPDX-License-Identifier: AGPL-3.0-only
//! Epiphanes Math non-Gaussian quality-gate primitives (`ADR-0001`).
//!
//! Implements distribution-free, anytime-valid quality-gate statistics from
//! `THME_Research_Dossier_v3.3`: Geyer's integrated autocorrelation time,
//! asymmetric two-sided Mixture-GKL + Fan-Howard-Ramdas E-processes,
//! Kusuoka-coherent Weighted CVaR, Asymmetry-Adaptive Jeffreys-Renyi
//! divergence, and pure-content Bits-Per-Byte (BPB).

use crate::engine::error::{ensure_finite, EngineError, EngineResult};
use crate::quality::gate::Decision;

/// Estimates integrated autocorrelation time `tau_int >= 1` via Geyer's
/// initial positive sequence over lag-`k` sample autocorrelations `autocorrs`
/// (starting at lag 1).
pub fn geyer_tau_int(autocorrs: &[f64]) -> usize {
    let mut sum = 0.0_f64;
    let mut idx = 0_usize;
    while idx + 1 < autocorrs.len() {
        let pair = autocorrs[idx] + autocorrs[idx + 1];
        if !pair.is_finite() || pair <= 0.0 {
            break;
        }
        sum += pair;
        idx += 2;
    }
    let tau = (1.0 + 2.0 * sum).ceil();
    if tau.is_finite() && tau >= 1.0 {
        tau as usize
    } else {
        1
    }
}

/// Pure-content Bits Per Byte (BPB) excluding BOS/EOS/template framing bytes.
pub fn pure_content_bpb(
    mean_loss_nats: f64,
    scored_tokens: usize,
    content_utf8_bytes: usize,
) -> EngineResult<f64> {
    if !mean_loss_nats.is_finite() || mean_loss_nats < 0.0 {
        return Err(EngineError::InvalidInput(format!(
            "mean_loss_nats must be finite and non-negative, got {mean_loss_nats}"
        )));
    }
    if content_utf8_bytes == 0 {
        return Err(EngineError::InvalidInput(
            "content_utf8_bytes must be positive".to_string(),
        ));
    }
    let bits = (scored_tokens as f64) * mean_loss_nats / std::f64::consts::LN_2;
    Ok(bits / (content_utf8_bytes as f64))
}

/// Evaluates the exact Fan-Howard-Ramdas supermartingale testing `H0: mu >= mu0`
/// (rejecting large `mu0` when `delta < mu0`; strictly positive for all `delta >= -1.0`).
fn upper_bound_eprocess(drifts: &[f64], mu0: f64) -> f64 {
    let c0 = (1.0 + mu0).max(1e-6);
    let mut log_wealth = 0.0_f64;
    let mut mean_prev = 0.0_f64;
    let mut m2_prev = 0.0_f64;

    for (step, &delta) in drifts.iter().enumerate() {
        if step > 0 {
            let var_prev = if step > 1 {
                (m2_prev / ((step - 1) as f64)).max(1e-6)
            } else {
                0.25
            };
            let gamma = (0.95 / (var_prev.sqrt() + c0)).min(0.92 / c0);
            let gc = (gamma * c0).min(0.95);
            let psi_e = (-(1.0 - gc).ln() - gc) / (c0 * c0);
            let centered = delta - mean_prev;
            // Grows when delta < mu0, proving true mean < mu0.
            log_wealth += gamma * (mu0 - delta) - psi_e * centered * centered;
        }
        let n = (step + 1) as f64;
        let diff = delta - mean_prev;
        mean_prev += diff / n;
        m2_prev += diff * (delta - mean_prev);
    }

    log_wealth.clamp(-700.0, 700.0).exp()
}

/// Evaluates the Mixture-GKL wealth process testing `H0: mu <= mu0`
/// (rejecting small `mu0` when `delta > mu0`; non-negative for all `delta >= -1.0`).
fn lower_bound_eprocess(drifts: &[f64], mu0: f64) -> f64 {
    let mut running_max = 0.0_f64;
    let mut mean_prev = 0.0_f64;
    let mut m2_prev = 0.0_f64;
    let mut wealth = [1.0_f64; 9];
    let c_trunc = 0.5_f64;
    let eps = 1e-8_f64;

    for (step, &delta) in drifts.iter().enumerate() {
        let var_prev = if step > 1 {
            m2_prev / ((step - 1) as f64)
        } else {
            1.0
        };
        let denom_bound = (1.0 + mu0).max(running_max).max(1e-6);
        let var_cap = c_trunc / (var_prev + eps).sqrt();

        for (j, w) in wealth.iter_mut().enumerate() {
            let grid_lam = 0.5_f64.powi(j as i32) / denom_bound;
            let lam = grid_lam.min(var_cap).min(1.0 / (1.0 + mu0).max(1e-6));
            let factor = (1.0 + lam * (delta - mu0)).max(0.0);
            *w *= factor;
        }

        let n = (step + 1) as f64;
        let diff = delta - mean_prev;
        mean_prev += diff / n;
        m2_prev += diff * (delta - mean_prev);
        if delta > running_max {
            running_max = delta;
        }
    }

    wealth.iter().sum::<f64>() / (wealth.len() as f64)
}

/// Computes anytime-valid two-sided bounds `[L_t, U_t]` and the three-state
/// Epiphanes `Decision` (`Pass`, `Fail`, `Inconclusive`) against `tau_ra`
/// at significance `eta` (e.g., `0.05` for 95% confidence).
pub fn hedged_epiphanes_bounds(
    drifts: &[f64],
    eta: f64,
    tau_ra: f64,
) -> EngineResult<(f64, f64, Decision)> {
    if drifts.is_empty() {
        return Err(EngineError::InvalidInput(
            "drifts must contain at least one block".to_string(),
        ));
    }
    if !(0.0 < eta && eta < 1.0) {
        return Err(EngineError::InvalidInput(format!(
            "eta must be in (0, 1), got {eta}"
        )));
    }
    if !tau_ra.is_finite() || tau_ra < 0.0 {
        return Err(EngineError::InvalidInput(format!(
            "tau_ra must be finite and non-negative, got {tau_ra}"
        )));
    }
    ensure_finite("drifts", drifts)?;
    for (i, &d) in drifts.iter().enumerate() {
        if d < -1.0 {
            return Err(EngineError::InvalidInput(format!(
                "normalized drift[{i}] = {d} < -1.0 violates cross-entropy non-negativity"
            )));
        }
    }

    let threshold = 1.0 / eta;
    let sample_mean = drifts.iter().sum::<f64>() / (drifts.len() as f64);

    // Bisection for upper bound U_t = inf { mu0 >= sample_mean : upper_bound_eprocess(mu0) >= 1/eta }.
    // `upper_bound_eprocess` is monotonically increasing in mu0.
    let mut u_lo = sample_mean.max(-1.0);
    let mut u_hi = drifts
        .iter()
        .copied()
        .fold(sample_mean + 1.0, f64::max)
        .max(1.0);
    while upper_bound_eprocess(drifts, u_hi) < threshold && u_hi < 1e6 {
        u_hi *= 2.0;
    }
    for _ in 0..48 {
        let mid = 0.5 * (u_lo + u_hi);
        if upper_bound_eprocess(drifts, mid) >= threshold {
            u_hi = mid;
        } else {
            u_lo = mid;
        }
    }
    let upper = u_hi;

    // Bisection for lower bound L_t = sup { mu0 <= sample_mean : lower_bound_eprocess(mu0) >= 1/eta }.
    // `lower_bound_eprocess` is monotonically decreasing in mu0.
    let mut l_lo = -1.0_f64;
    let mut l_hi = sample_mean.max(-1.0);
    for _ in 0..48 {
        let mid = 0.5 * (l_lo + l_hi);
        if lower_bound_eprocess(drifts, mid) >= threshold {
            l_lo = mid;
        } else {
            l_hi = mid;
        }
    }
    let lower = l_lo;

    let decision = if upper <= tau_ra {
        Decision::Pass
    } else if lower > tau_ra {
        Decision::Fail
    } else {
        Decision::Inconclusive
    };

    Ok((lower, upper, decision))
}

/// Kusuoka-coherent Weighted CVaR (`WCVaR`) with entropic tilting `phi(beta) = 2*beta`.
///
/// Satisfies subadditivity, monotonicity, translation invariance, and positive
/// homogeneity across composed repair stages.
pub fn coherent_wcvar(drifts: &[f64]) -> EngineResult<f64> {
    if drifts.is_empty() {
        return Err(EngineError::InvalidInput(
            "drifts must not be empty for WCVaR".to_string(),
        ));
    }
    ensure_finite("wcvar drift", drifts)?;
    let mut sorted = drifts.to_vec();
    sorted.sort_by(f64::total_cmp);

    let n = sorted.len();
    // Exact discrete integration of int_0^1 CVaR_beta(X) * 2*beta d_beta:
    // For order statistics x_(1) <= ... <= x_(n), weight on x_(k) is (2k - 1) / n^2.
    let n2 = (n as f64) * (n as f64);
    let mut wcvar = 0.0_f64;
    for (idx, &val) in sorted.iter().enumerate() {
        let weight = ((2 * idx + 1) as f64) / n2;
        wcvar += weight * val;
    }
    Ok(wcvar)
}

/// Computes `D_alpha^(lambda)(p || q)` in FP64 over a coarsened Top-K + Tail partition.
fn skewed_renyi(p: &[f64], q: &[f64], alpha: f64, lambda: f64) -> f64 {
    let mut sum = 0.0_f64;
    for (&pv, &qv) in p.iter().zip(q.iter()) {
        if pv <= 0.0 {
            continue;
        }
        let mix = (1.0 - lambda) * qv + lambda * pv;
        sum += pv.powf(alpha) * mix.powf(1.0 - alpha);
    }
    (sum.max(1e-300).ln() / (alpha - 1.0)).max(0.0)
}

/// Asymmetry-Adaptive Jeffreys-Renyi divergence `J_alpha^AA(p0, p_theta)` on the
/// FP32 Top-K + collapsed tail partition (`K = 256` recommended).
pub fn asymmetry_adaptive_renyi(
    p0_topk: &[f32],
    p0_tail: f32,
    pt_topk: &[f32],
    pt_tail: f32,
    alpha: f64,
    lambda: f64,
) -> EngineResult<f64> {
    if p0_topk.len() != pt_topk.len() || p0_topk.is_empty() {
        return Err(EngineError::InvalidInput(
            "p0_topk and pt_topk must have equal non-zero length".to_string(),
        ));
    }
    if !(alpha > 1.0 && alpha.is_finite()) {
        return Err(EngineError::InvalidInput(format!(
            "Renyi order alpha must be > 1.0, got {alpha}"
        )));
    }
    if !(0.0 < lambda && lambda < 0.5) {
        return Err(EngineError::InvalidInput(format!(
            "skew lambda must be in (0, 0.5), got {lambda}"
        )));
    }

    let mut p0 = Vec::with_capacity(p0_topk.len() + 1);
    let mut pt = Vec::with_capacity(pt_topk.len() + 1);
    for &v in p0_topk {
        p0.push(f64::from(v).max(0.0));
    }
    p0.push(f64::from(p0_tail).max(0.0));
    for &v in pt_topk {
        pt.push(f64::from(v).max(0.0));
    }
    pt.push(f64::from(pt_tail).max(0.0));

    let s0: f64 = p0.iter().sum();
    let st: f64 = pt.iter().sum();
    if s0 <= 0.0 || st <= 0.0 {
        return Err(EngineError::InvalidInput(
            "probability distributions must have positive mass".to_string(),
        ));
    }
    for v in &mut p0 {
        *v /= s0;
    }
    for v in &mut pt {
        *v /= st;
    }

    let d_fwd = skewed_renyi(&p0, &pt, alpha, lambda);
    let d_rev = skewed_renyi(&pt, &p0, alpha, lambda);
    let s_star = 1.0 / (1.0 + (-(d_fwd - d_rev)).exp());
    Ok((1.0 - s_star) * d_fwd + s_star * d_rev)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn geyer_tau_stops_on_negative_pair() {
        let acf = [0.6, 0.3, 0.1, -0.2, 0.5];
        // Pair 0: 0.9 > 0; Pair 1: -0.1 <= 0 -> stops. tau_int = ceil(1 + 2*0.9) = 3.
        assert_eq!(geyer_tau_int(&acf), 3);
    }

    #[test]
    fn wcvar_satisfies_translation_and_subadditivity() {
        let x = [0.01, 0.02, 0.05, 0.10, 0.30];
        let y = [0.04, 0.01, 0.03, 0.08, 0.02];
        let xy: Vec<f64> = x.iter().zip(y.iter()).map(|(a, b)| a + b).collect();
        let wx = coherent_wcvar(&x).unwrap();
        let wy = coherent_wcvar(&y).unwrap();
        let wxy = coherent_wcvar(&xy).unwrap();
        assert!(wxy <= wx + wy + 1e-12, "subadditivity violated");

        let x_shift: Vec<f64> = x.iter().map(|v| v + 0.5).collect();
        let w_shift = coherent_wcvar(&x_shift).unwrap();
        assert!((w_shift - (wx + 0.5)).abs() < 1e-12, "translation violated");
    }

    #[test]
    fn hedged_bounds_classify_pass_fail_inconclusive() {
        let low_drift = vec![0.002; 80];
        let (l, u, dec) = hedged_epiphanes_bounds(&low_drift, 0.05, 0.05).unwrap();
        assert!(l <= u);
        assert_eq!(dec, Decision::Pass);

        let high_drift = vec![0.20; 80];
        let (l2, u2, dec2) = hedged_epiphanes_bounds(&high_drift, 0.05, 0.05).unwrap();
        assert!(l2 <= u2);
        assert_eq!(dec2, Decision::Fail);

        let tiny_sample = vec![0.04; 3];
        let (_l3, _u3, dec3) = hedged_epiphanes_bounds(&tiny_sample, 0.05, 0.05).unwrap();
        assert_eq!(dec3, Decision::Inconclusive);
    }

    #[test]
    fn asymmetry_adaptive_renyi_is_zero_on_identical_and_finite_on_zero_support() {
        let p = [0.5_f32, 0.3, 0.1];
        let j0 = asymmetry_adaptive_renyi(&p, 0.1, &p, 0.1, 2.0, 1e-3).unwrap();
        assert!(j0.abs() < 1e-12);

        let q = [0.5_f32, 0.0, 0.4];
        let j1 = asymmetry_adaptive_renyi(&p, 0.1, &q, 0.1, 2.0, 1e-3).unwrap();
        assert!(j1.is_finite() && j1 > 0.0);
    }
}
