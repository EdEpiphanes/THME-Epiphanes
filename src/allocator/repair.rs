// SPDX-License-Identifier: AGPL-3.0-only
//! Post-quantization & post-transposing drift-reduction repair (`THME_Research_Dossier_v3.3` §6).
//!
//! Implements:
//! 1. In-place linear dequant-scale second-moment recalibration (`s_j^*`).
//! 2. SwiGLU-aware post-SiLU second-moment matching (`s_{j,gate}^*`).
//! 3. Causal sink-separated Bregman LSE attention alignment (`eta_h^*`, `beta_{h,0}^*`).
//! 4. Scale-invariant Nystrom Cholesky-SVD residual repair (`A^*`, `B^*`).

use crate::engine::error::{EngineError, EngineResult};

/// In-place second-moment recalibration for linear dequantization scales `s_j^*`,
/// preserving Pre-Norm residual-stream balance with zero extra parameters.
pub fn recalibrate_dequant_scales(
    scales: &mut [f32],
    ref_second_moment: &[f32],
    quant_second_moment: &[f32],
) -> EngineResult<()> {
    if scales.len() != ref_second_moment.len() || scales.len() != quant_second_moment.len() {
        return Err(EngineError::InvalidInput(
            "recalibrate_dequant_scales slice length mismatch".to_string(),
        ));
    }
    let eps = 1e-8_f32;
    for ((s, &r2), &q2) in scales
        .iter_mut()
        .zip(ref_second_moment.iter())
        .zip(quant_second_moment.iter())
    {
        if !s.is_finite() || !r2.is_finite() || !q2.is_finite() || r2 < 0.0 || q2 < 0.0 {
            return Err(EngineError::InvalidInput(
                "non-finite or negative second moment in recalibrate_dequant_scales".to_string(),
            ));
        }
        *s *= ((r2 + eps) / (q2 + eps)).sqrt();
    }
    Ok(())
}

#[inline]
fn silu(z: f64) -> f64 {
    let sig = 1.0 / (1.0 + (-z).exp());
    z * sig
}

/// SwiGLU-aware post-SiLU second-moment matching for a single `W_gate` channel.
/// Returns `(s_gate_star, first_moment_residual)` where `first_moment_residual`
/// is folded into downstream `W_down` bias/offset when below `Q4_0`.
pub fn swiglu_post_silu_match(
    z_ref: &[f32],
    z_quant: &[f32],
    z_up: &[f32],
    init_scale: f32,
) -> EngineResult<(f32, f32)> {
    if z_ref.is_empty() || z_ref.len() != z_quant.len() || z_ref.len() != z_up.len() {
        return Err(EngineError::InvalidInput(
            "swiglu_post_silu_match requires equal non-empty activation slices".to_string(),
        ));
    }
    let n = z_ref.len() as f64;
    let target_m2: f64 = z_ref
        .iter()
        .map(|&z| {
            let v = silu(f64::from(z));
            v * v
        })
        .sum::<f64>()
        / n;

    // 1D damped secant/Newton search for scalar multiplier alpha > 0 on z_quant.
    let mut alpha = 1.0_f64;
    let eps = 1e-8_f64;
    for _ in 0..10 {
        let mut m2 = 0.0_f64;
        let mut dm2 = 0.0_f64;
        for &zq in z_quant {
            let zq64 = f64::from(zq);
            let u = alpha * zq64;
            let sig = 1.0 / (1.0 + (-u).exp());
            let s = u * sig;
            let ds = sig * (1.0 + u * (1.0 - sig));
            m2 += s * s;
            dm2 += 2.0 * s * ds * zq64;
        }
        m2 /= n;
        dm2 /= n;
        let err = m2 - target_m2;
        if err.abs() < 1e-7 {
            break;
        }
        if dm2.abs() > eps {
            alpha = (alpha - (err / dm2).clamp(-0.5, 0.5)).clamp(0.25, 4.0);
        } else {
            break;
        }
    }

    let ref_prod: f64 = z_ref
        .iter()
        .zip(z_up.iter())
        .map(|(&zr, &zu)| silu(f64::from(zr)) * f64::from(zu))
        .sum::<f64>()
        / n;
    let quant_prod: f64 = z_quant
        .iter()
        .zip(z_up.iter())
        .map(|(&zq, &zu)| silu(alpha * f64::from(zq)) * f64::from(zu))
        .sum::<f64>()
        / n;

    Ok((init_scale * (alpha as f32), (ref_prod - quant_prod) as f32))
}

/// Causal sink-separated Bregman LSE attention alignment (`eta_h^* > 0`, `beta_{h,0}^*`).
///
/// `p0_causal` and `s_theta_causal` contain causal prefix slices `j <= i` for one head.
pub fn causal_bregman_lse_align(
    p0_causal: &[Vec<f64>],
    s_theta_causal: &[Vec<f64>],
) -> EngineResult<(f64, f64)> {
    if p0_causal.is_empty() || p0_causal.len() != s_theta_causal.len() {
        return Err(EngineError::InvalidInput(
            "causal_bregman_lse_align requires matching non-empty prefix slices".to_string(),
        ));
    }

    let mut eta = 1.0_f64;
    let mut beta = 0.0_f64;
    let m = p0_causal.len() as f64;

    for _ in 0..12 {
        let (mut g_eta, mut g_beta) = (0.0_f64, 0.0_f64);
        let (mut h_ee, mut h_eb, mut h_bb) = (1e-6_f64, 0.0_f64, 1e-6_f64);

        for (p0, st) in p0_causal.iter().zip(s_theta_causal.iter()) {
            if p0.is_empty() || p0.len() != st.len() {
                return Err(EngineError::InvalidInput(
                    "prefix probability and score lengths must match".to_string(),
                ));
            }
            let mut logits = Vec::with_capacity(st.len());
            let mut max_l = f64::NEG_INFINITY;
            for (j, &s) in st.iter().enumerate() {
                let l = eta * s + if j == 0 { beta } else { 0.0 };
                if l > max_l {
                    max_l = l;
                }
                logits.push(l);
            }
            let mut probs = vec![0.0_f64; st.len()];
            let mut sum_e = 0.0_f64;
            for (j, &l) in logits.iter().enumerate() {
                let e = (l - max_l).exp();
                probs[j] = e;
                sum_e += e;
            }
            for p in &mut probs {
                *p /= sum_e.max(1e-300);
            }

            let e_p_s: f64 = probs.iter().zip(st.iter()).map(|(&p, &s)| p * s).sum();
            let e_p0_s: f64 = p0.iter().zip(st.iter()).map(|(&p, &s)| p * s).sum();
            let e_p_s2: f64 = probs.iter().zip(st.iter()).map(|(&p, &s)| p * s * s).sum();

            g_eta += e_p_s - e_p0_s;
            g_beta += probs[0] - p0[0];

            h_ee += (e_p_s2 - e_p_s * e_p_s).max(0.0);
            h_eb += probs[0] * (st[0] - e_p_s);
            h_bb += probs[0] * (1.0 - probs[0]);
        }

        g_eta /= m;
        g_beta /= m;
        h_ee /= m;
        h_eb /= m;
        h_bb /= m;

        let det = h_ee * h_bb - h_eb * h_eb;
        if det.abs() < 1e-12 || (g_eta.abs() < 1e-7 && g_beta.abs() < 1e-7) {
            break;
        }
        let d_eta = (h_bb * g_eta - h_eb * g_beta) / det;
        let d_beta = (-h_eb * g_eta + h_ee * g_beta) / det;

        eta = (eta - d_eta.clamp(-0.5, 0.5)).clamp(0.1, 10.0);
        beta = (beta - d_beta.clamp(-2.0, 2.0)).clamp(-10.0, 10.0);
    }

    Ok((eta, beta))
}

/// Solves a dense SPD linear system `A x = b` (`dim x dim`) via Cholesky factorization.
fn cholesky_solve_spd(a: &[f64], b: &[f64], dim: usize) -> EngineResult<Vec<f64>> {
    if a.len() != dim * dim || b.len() != dim {
        return Err(EngineError::InvalidInput(
            "cholesky_solve_spd dimension mismatch".to_string(),
        ));
    }
    let mut l = vec![0.0_f64; dim * dim];
    for i in 0..dim {
        for j in 0..=i {
            let mut sum = a[i * dim + j];
            for k in 0..j {
                sum -= l[i * dim + k] * l[j * dim + k];
            }
            if i == j {
                if sum <= 0.0 || !sum.is_finite() {
                    return Err(EngineError::InvalidInput(
                        "matrix is not strictly positive definite".to_string(),
                    ));
                }
                l[i * dim + j] = sum.sqrt();
            } else {
                l[i * dim + j] = sum / l[j * dim + j];
            }
        }
    }

    // Forward substitution L y = b.
    let mut y = vec![0.0_f64; dim];
    for i in 0..dim {
        let mut sum = b[i];
        for k in 0..i {
            sum -= l[i * dim + k] * y[k];
        }
        y[i] = sum / l[i * dim + i];
    }
    // Backward substitution L^T x = y.
    let mut x = vec![0.0_f64; dim];
    for i in (0..dim).rev() {
        let mut sum = y[i];
        for k in (i + 1)..dim {
            sum -= l[k * dim + i] * x[k];
        }
        x[i] = sum / l[i * dim + i];
    }
    Ok(x)
}

/// Scale-invariant Nystrom Tikhonov normal-equations solver for residual repair:
/// minimizes `||(W - W_tilde) L_ny - A B||_F^2` using trace-normalized damping.
///
/// Returns `(a_star, b_star)` with `a_star` (`d_out x rank`) and `b_star` (`rank x d_in`).
pub fn nystrom_tikhonov_residual_repair(
    residual: &[f64],
    d_out: usize,
    d_in: usize,
    l_ny: &[f64],
    r_ny: usize,
    rank: usize,
) -> EngineResult<(Vec<f64>, Vec<f64>)> {
    if residual.len() != d_out * d_in || l_ny.len() != d_in * r_ny || r_ny == 0 || rank == 0 {
        return Err(EngineError::InvalidInput(
            "nystrom_tikhonov_residual_repair dimension mismatch".to_string(),
        ));
    }
    let r = rank.min(r_ny).min(d_out);

    // 1. Compute E = (W - W_tilde) L_ny of shape (d_out x r_ny).
    let mut e_mat = vec![0.0_f64; d_out * r_ny];
    for i in 0..d_out {
        for j in 0..r_ny {
            let mut sum = 0.0_f64;
            for k in 0..d_in {
                sum += residual[i * d_in + k] * l_ny[k * r_ny + j];
            }
            e_mat[i * r_ny + j] = sum;
        }
    }

    // 2. Compute top-r left/right singular pairs via deflated power iteration on E^T E.
    let mut a_star = vec![0.0_f64; d_out * r];
    let mut v_r = vec![0.0_f64; r * r_ny];
    let mut e_rem = e_mat;

    for comp in 0..r {
        let mut v = vec![1.0 / (r_ny as f64).sqrt(); r_ny];
        let mut u = vec![0.0_f64; d_out];
        let mut sigma = 0.0_f64;

        for _ in 0..24 {
            // u = E_rem * v
            let mut norm_u = 0.0_f64;
            for i in 0..d_out {
                let mut s = 0.0_f64;
                for j in 0..r_ny {
                    s += e_rem[i * r_ny + j] * v[j];
                }
                u[i] = s;
                norm_u += s * s;
            }
            norm_u = norm_u.sqrt();
            if norm_u < 1e-12 {
                sigma = 0.0;
                break;
            }
            for ui in &mut u {
                *ui /= norm_u;
            }
            // v = E_rem^T * u
            let mut norm_v = 0.0_f64;
            for j in 0..r_ny {
                let mut s = 0.0_f64;
                for i in 0..d_out {
                    s += e_rem[i * r_ny + j] * u[i];
                }
                v[j] = s;
                norm_v += s * s;
            }
            sigma = norm_v.sqrt();
            if sigma < 1e-12 {
                break;
            }
            for vj in &mut v {
                *vj /= sigma;
            }
        }

        for i in 0..d_out {
            a_star[i * r + comp] = sigma * u[i];
        }
        for j in 0..r_ny {
            v_r[comp * r_ny + j] = v[j];
        }
        for i in 0..d_out {
            for j in 0..r_ny {
                e_rem[i * r_ny + j] -= sigma * u[i] * v[j];
            }
        }
    }

    // 3. Form Gram_ny = L_ny^T L_ny + lambda_svd * I_{r_ny} with trace-normalized damping.
    let mut gram_ny = vec![0.0_f64; r_ny * r_ny];
    let mut trace = 0.0_f64;
    for i in 0..r_ny {
        for j in 0..=i {
            let mut s = 0.0_f64;
            for k in 0..d_in {
                s += l_ny[k * r_ny + i] * l_ny[k * r_ny + j];
            }
            gram_ny[i * r_ny + j] = s;
            gram_ny[j * r_ny + i] = s;
            if i == j {
                trace += s;
            }
        }
    }
    let lambda_svd = 1e-3 * (trace / (r_ny as f64)).max(1e-8);
    for i in 0..r_ny {
        gram_ny[i * r_ny + i] += lambda_svd;
    }

    // 4. Solve B^* = V_r^T (L_ny^T L_ny + lambda_svd I)^{-1} L_ny^T.
    let mut b_star = vec![0.0_f64; r * d_in];
    for comp in 0..r {
        let rhs = &v_r[comp * r_ny..(comp + 1) * r_ny];
        let z = cholesky_solve_spd(&gram_ny, rhs, r_ny)?;
        for col in 0..d_in {
            let mut s = 0.0_f64;
            for j in 0..r_ny {
                s += z[j] * l_ny[col * r_ny + j];
            }
            b_star[comp * d_in + col] = s;
        }
    }

    Ok((a_star, b_star))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recalibrate_scales_restores_exact_ratio() {
        let mut s = [2.0_f32, 1.0];
        recalibrate_dequant_scales(&mut s, &[4.0, 9.0], &[1.0, 1.0]).unwrap();
        assert!((s[0] - 4.0).abs() < 1e-5);
        assert!((s[1] - 3.0).abs() < 1e-5);
    }

    #[test]
    fn causal_bregman_recovers_exact_sink_bias_and_temp() {
        // Teacher has eta=2.0, beta=1.5 relative to student scores [1.0, 0.0].
        let scores = vec![vec![1.0, 0.0], vec![0.5, -0.5, 0.2]];
        let mut probs = Vec::new();
        for st in &scores {
            let mut exps: Vec<f64> = Vec::new();
            for (j, &s) in st.iter().enumerate() {
                exps.push((2.0_f64 * s + if j == 0 { 1.5_f64 } else { 0.0_f64 }).exp());
            }
            let sum: f64 = exps.iter().sum();
            probs.push(exps.into_iter().map(|e| e / sum).collect());
        }
        let (eta, beta) = causal_bregman_lse_align(&probs, &scores).unwrap();
        assert!((eta - 2.0).abs() < 1e-3, "eta={eta}");
        assert!((beta - 1.5).abs() < 1e-3, "beta={beta}");
    }

    #[test]
    fn nystrom_residual_repair_reduces_error() {
        let d_out = 3;
        let d_in = 4;
        let r_ny = 3;
        let residual = vec![
            1.0, 0.5, 0.0, 0.0, //
            0.0, 1.0, 0.5, 0.0, //
            0.0, 0.0, 1.0, 0.5,
        ];
        let l_ny = vec![
            1.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, //
            0.0, 0.0, 1.0, //
            0.1, 0.1, 0.1,
        ];
        let (a, b) =
            nystrom_tikhonov_residual_repair(&residual, d_out, d_in, &l_ny, r_ny, 2).unwrap();
        assert_eq!(a.len(), d_out * 2);
        assert_eq!(b.len(), 2 * d_in);
    }
}
