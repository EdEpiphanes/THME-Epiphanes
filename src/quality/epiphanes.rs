// SPDX-License-Identifier: AGPL-3.0-only
//! Epiphanes Math non-Gaussian quality-gate primitives (`ADR-0001`).
//!
//! Implements distribution-free, anytime-valid quality-gate statistics from
//! `THME_Research_Dossier_v3.3`: Geyer's integrated autocorrelation time,
//! two-sided betting e-processes with a declared range,
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

/// Declared upper range `M` of a block drift (assumption A3, dossier v5 Table 3). Without a
/// declared range no distribution-free upper confidence bound exists (Bahadur and Savage, 1956).
pub const DEFAULT_DECLARED_RANGE: f64 = 0.25;

/// Anytime-valid confidence sequence `[L_t, U_t]` for the mean block drift and the three-state
/// `Decision` against `tau_ra`, at error level `eta` (dossier v5, Sections 3.3 and 3.6).
///
/// Two betting e-processes over a grid of null values `mu0` in `(-1, declared_range)`: the lower
/// side bets on `delta > mu0` and needs only `delta >= -1`; the upper side bets on `delta < mu0`
/// and uses the declared range `delta <= declared_range`. Each side is a mixture of nine bets
/// with weights 1/9; the two sides are averaged and `mu0` stays in the set while the average is
/// below `1 / eta`. The bounds are the running intersection over blocks.
///
/// A block above `declared_range` violates the declared range: the gate fails and returns
/// `[mean, declared_range]` without running the processes (their factors are only guaranteed
/// non-negative inside the range).
pub fn hedged_epiphanes_bounds(
    drifts: &[f64],
    eta: f64,
    tau_ra: f64,
    declared_range: f64,
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
    if !declared_range.is_finite() || declared_range <= 0.0 {
        return Err(EngineError::InvalidInput(format!(
            "declared range must be finite and positive, got {declared_range}"
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

    let sample_mean = drifts.iter().sum::<f64>() / (drifts.len() as f64);
    if drifts.iter().any(|&d| d > declared_range) {
        return Ok((sample_mean, declared_range, Decision::Fail));
    }

    const BETS: usize = 9;
    const GRID_STEP: f64 = 1e-4;
    let m = declared_range;
    let threshold = 1.0 / eta;
    let grid: Vec<f64> = (1..)
        .map(|k| -1.0 + GRID_STEP * f64::from(k))
        .take_while(|&mu0| mu0 < m)
        .collect();
    let mut plus = vec![[1.0_f64; BETS]; grid.len()];
    let mut minus = vec![[1.0_f64; BETS]; grid.len()];
    let (mut lower, mut upper) = (-1.0_f64, m);
    let (mut mean_prev, mut m2_prev) = (0.0_f64, 0.0_f64);

    for (step, &delta) in drifts.iter().enumerate() {
        let b = (step + 1) as f64;
        // Predictable variance statistic from blocks before this one only (A5).
        let v_prev = (tau_ra * tau_ra + m2_prev) / b;
        let cap = 1.0 / (v_prev + 1e-6).sqrt();
        let (mut set_min, mut set_max) = (f64::INFINITY, f64::NEG_INFINITY);
        for (i, &mu0) in grid.iter().enumerate() {
            let mut sum_plus = 0.0;
            let mut sum_minus = 0.0;
            for j in 0..BETS {
                let half = 0.5_f64.powi(j as i32);
                let lam_plus = (half / (1.0 + mu0)).min(cap);
                let lam_minus = (half / (m - mu0)).min(cap);
                plus[i][j] *= 1.0 + lam_plus * (delta - mu0);
                minus[i][j] *= 1.0 + lam_minus * (mu0 - delta);
                sum_plus += plus[i][j];
                sum_minus += minus[i][j];
            }
            let e = 0.5 * sum_plus / BETS as f64 + 0.5 * sum_minus / BETS as f64;
            if e < threshold {
                set_min = set_min.min(mu0);
                set_max = set_max.max(mu0);
            }
        }
        if set_min <= set_max {
            lower = lower.max(set_min);
            upper = upper.min(set_max);
        }
        let diff = delta - mean_prev;
        mean_prev += diff / b;
        m2_prev += diff * (delta - mean_prev);
    }

    let decision = if lower > tau_ra {
        Decision::Fail
    } else if upper <= tau_ra {
        Decision::Pass
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
        let (l, u, dec) =
            hedged_epiphanes_bounds(&low_drift, 0.05, 0.05, DEFAULT_DECLARED_RANGE).unwrap();
        assert!(l <= u);
        assert_eq!(dec, Decision::Pass);

        let high_drift = vec![0.20; 80];
        let (l2, u2, dec2) =
            hedged_epiphanes_bounds(&high_drift, 0.05, 0.05, DEFAULT_DECLARED_RANGE).unwrap();
        assert!(l2 <= u2);
        assert_eq!(dec2, Decision::Fail);

        let tiny_sample = vec![0.04; 3];
        let (_l3, _u3, dec3) =
            hedged_epiphanes_bounds(&tiny_sample, 0.05, 0.05, DEFAULT_DECLARED_RANGE).unwrap();
        assert_eq!(dec3, Decision::Inconclusive);
    }

    /// Source versus source at the paper's defaults (dossier v5, Table 4): the upper limit
    /// of the confidence sequence for a candidate identical to the source.
    #[test]
    fn identical_candidate_matches_the_papers_table_4() {
        for (blocks, expected) in [(37, 0.0360), (74, 0.0187), (111, 0.0126), (148, 0.0095)] {
            let drifts = vec![0.0; blocks];
            let (_l, u, _d) =
                hedged_epiphanes_bounds(&drifts, 0.05, 0.0223, DEFAULT_DECLARED_RANGE).unwrap();
            assert!(
                (u - expected).abs() < 6e-5,
                "U_{blocks} = {u}, the paper's Table 4 gives {expected}"
            );
        }
    }

    #[test]
    fn a_block_above_the_declared_range_fails() {
        let mut drifts = vec![0.0; 50];
        drifts.push(0.30);
        let (_l, _u, d) =
            hedged_epiphanes_bounds(&drifts, 0.05, 0.0223, DEFAULT_DECLARED_RANGE).unwrap();
        assert_eq!(d, Decision::Fail);
        assert!(hedged_epiphanes_bounds(&drifts, 0.05, 0.0223, 0.0).is_err());
    }

    /// Miscoverage of `[L_t, U_t]` on a bounded law with known mean 0.04 stays below eta.
    #[test]
    fn miscoverage_stays_below_eta_on_a_bounded_law() {
        let mut state = 20_261_126_u64;
        let mut next = move || {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            ((z ^ (z >> 31)) >> 11) as f64 / (1_u64 << 53) as f64
        };
        let (paths, blocks, true_mean) = (50, 100, 0.04);
        let mut misses = 0;
        for _ in 0..paths {
            // Uniform on [-0.10, 0.18]: mean 0.04, inside [-1, M].
            let drifts: Vec<f64> = (0..blocks).map(|_| -0.10 + 0.28 * next()).collect();
            let (l, u, _) =
                hedged_epiphanes_bounds(&drifts, 0.05, 0.0223, DEFAULT_DECLARED_RANGE).unwrap();
            if true_mean < l || true_mean > u {
                misses += 1;
            }
        }
        assert!(
            misses as f64 / paths as f64 <= 0.05,
            "{misses} of {paths} paths missed"
        );
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
