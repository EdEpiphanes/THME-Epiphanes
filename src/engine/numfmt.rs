//! Number formats (dossier WP3). The one absmean-ternary quantizer in THME lives here; the
//! BitNet packer, `MatrixMath` and the BYOA `AbsmeanTernary` rule all call it.
//!
//! Ternary rounding follows BitNet b1.58: scale by gamma = mean |w|, then round to
//! {-1, 0, +1}. The zero band is in **scaled** units: a weight becomes 0 when
//! |w / gamma| < zero_band. With zero_band = 0.5 this is round-to-nearest then clamp.
//! Callers choose their own scale floor for all-zero input, because the stored scale of a
//! zero tensor is a format decision, not arithmetic.
//!
//! The GGML `TQ1_0` encoder (`exporter::conversion`) is a different quantizer: it uses an
//! MSE-optimal scale, not absmean, and its exact float arithmetic is part of committed
//! measured reports, so it is not routed through here.

/// Mean absolute value; 0 for an empty slice.
pub fn absmean(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().map(|v| v.abs()).sum::<f32>() / values.len() as f32
}

/// Ternary code of an already scaled value: 0 inside the zero band (and for NaN), else its sign.
pub fn ternary_code(scaled: f32, zero_band: f32) -> i8 {
    if scaled.abs() >= zero_band {
        if scaled > 0.0 {
            1
        } else {
            -1
        }
    } else {
        0
    }
}

/// Ternary codes of `values` against `scale` (each value divided by `scale`).
pub fn ternary_codes(values: &[f32], scale: f32, zero_band: f32) -> Vec<i8> {
    values
        .iter()
        .map(|&w| ternary_code(w / scale, zero_band))
        .collect()
}

// ---------------------------------------------------------------------------------------------
// Integer, binary, codebook and phase quantizers. Each states its worst-case error.

use super::error::{ensure_finite, EngineError, EngineResult};

/// Symmetric absmax integer quantization to `bits` bits (2 to 16): the scale maps the largest
/// magnitude to `2^(bits-1) - 1`. Returns `(codes, scale)`. Error bound: `scale / 2` per value,
/// plus the f32 rounding of the division and product (at most `4 * f32::EPSILON * |x|`).
pub fn quantize_absmax(values: &[f32], bits: u32) -> EngineResult<(Vec<i32>, f32)> {
    if !(2..=16).contains(&bits) {
        return Err(EngineError::InvalidInput(format!(
            "integer width must be 2 to 16 bits, got {bits}"
        )));
    }
    let as_f64: Vec<f64> = values.iter().map(|&v| f64::from(v)).collect();
    ensure_finite("values", &as_f64)?;
    let levels = ((1i32 << (bits - 1)) - 1) as f32;
    let amax = values.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    if amax == 0.0 {
        return Ok((vec![0; values.len()], 0.0));
    }
    let scale = amax / levels;
    let codes = values
        .iter()
        .map(|&v| (v / scale).round().clamp(-levels, levels) as i32)
        .collect();
    Ok((codes, scale))
}

/// Sign quantization: `±scale` with `scale = mean |x|`, the scale that minimizes the squared
/// error for a fixed sign pattern. Returns `(signs, scale)`.
pub fn quantize_sign(values: &[f32]) -> (Vec<i8>, f32) {
    let scale = absmean(values);
    let signs = values
        .iter()
        .map(|&v| if v < 0.0 { -1 } else { 1 })
        .collect();
    (signs, scale)
}

/// Index of the level nearest to `x` (the first one on a tie). Error bound: half the largest
/// gap between neighbouring levels inside their range, or the distance to the end level outside.
pub fn nearest_level(levels: &[f32], x: f32) -> EngineResult<usize> {
    if levels.is_empty() {
        return Err(EngineError::InvalidInput("codebook has no levels".into()));
    }
    if !x.is_finite() {
        return Err(EngineError::NonFinite {
            what: "codebook input".into(),
            index: 0,
        });
    }
    let mut best = 0;
    let mut best_diff = (x - levels[0]).abs();
    for (i, &level) in levels.iter().enumerate().skip(1) {
        let diff = (x - level).abs();
        if diff < best_diff {
            best_diff = diff;
            best = i;
        }
    }
    Ok(best)
}

/// Phase quantization onto `k` equally spaced angles `2*pi*j/k`. Returns the index `j`.
/// Error bound: the wrapped angular error is at most `pi / k`.
pub fn quantize_phase(theta: f64, k: u32) -> EngineResult<u32> {
    if k == 0 {
        return Err(EngineError::InvalidInput(
            "phase quantization needs k > 0".into(),
        ));
    }
    if !theta.is_finite() {
        return Err(EngineError::NonFinite {
            what: "phase".into(),
            index: 0,
        });
    }
    let step = std::f64::consts::TAU / f64::from(k);
    let j = (theta.rem_euclid(std::f64::consts::TAU) / step).round() as u64 % u64::from(k);
    Ok(j as u32)
}

/// The angle of phase index `j` out of `k`.
pub fn phase_angle(j: u32, k: u32) -> f64 {
    std::f64::consts::TAU * f64::from(j) / f64::from(k.max(1))
}

/// Wrapped difference between two angles, in `[0, pi]`.
pub fn angle_distance(a: f64, b: f64) -> f64 {
    let d = (a - b).rem_euclid(std::f64::consts::TAU);
    d.min(std::f64::consts::TAU - d)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Oracle: BitNet b1.58 round-and-clamp, `clamp(round(x), -1, 1)`, over a dense grid
    /// that includes the +-0.5 ties (round half away from zero gives +-1).
    #[test]
    fn half_band_equals_round_and_clamp() {
        for k in -400..=400 {
            let x = k as f32 / 100.0;
            let want = x.round().clamp(-1.0, 1.0) as i8;
            assert_eq!(ternary_code(x, 0.5), want, "x = {x}");
        }
        assert_eq!(ternary_code(f32::NAN, 0.5), 0);
    }

    #[test]
    fn absmean_and_codes_by_hand() {
        assert_eq!(absmean(&[]), 0.0);
        assert!((absmean(&[1.0, -1.0, 0.2, -0.2]) - 0.6).abs() < 1e-7);
        assert_eq!(
            ternary_codes(&[1.0, -1.0, 0.2, -0.2], 0.6, 0.5),
            vec![1, -1, 0, 0]
        );
        // A wider band zeroes more.
        assert_eq!(
            ternary_codes(&[1.0, -1.0, 0.2, -0.2], 0.6, 2.0),
            vec![0, 0, 0, 0]
        );
    }

    /// Ternary reconstruction at the absmean scale loses less than zeroing every weight.
    #[test]
    fn reconstruction_error_is_bounded_by_the_input_energy() {
        let w: Vec<f32> = (0..128)
            .map(|i| ((i * 53 % 128) as f32 - 63.5) / 50.0)
            .collect();
        let g = absmean(&w);
        let codes = ternary_codes(&w, g, 0.5);
        let err: f32 = w
            .iter()
            .zip(&codes)
            .map(|(x, &c)| (x - f32::from(c) * g).powi(2))
            .sum();
        let energy: f32 = w.iter().map(|x| x * x).sum();
        assert!(err < energy, "{err} vs {energy}");
    }

    #[test]
    fn absmax_integer_error_is_at_most_half_a_step() {
        let w: Vec<f32> = (0..500)
            .map(|i| ((i * 37 % 500) as f32 - 250.0) / 97.0)
            .collect();
        for bits in [2u32, 4, 8, 12] {
            let (codes, scale) = quantize_absmax(&w, bits).unwrap();
            for (x, c) in w.iter().zip(&codes) {
                assert!(
                    (x - *c as f32 * scale).abs() <= scale / 2.0 + 4.0 * f32::EPSILON * x.abs(),
                    "{bits} bits"
                );
            }
        }
        // More bits, much less error (theory: 4x per extra bit in MSE).
        let mse = |bits| {
            let (c, s) = quantize_absmax(&w, bits).unwrap();
            w.iter()
                .zip(&c)
                .map(|(x, q)| (x - *q as f32 * s).powi(2))
                .sum::<f32>()
        };
        assert!(mse(4) > 50.0 * mse(8));
        assert_eq!(quantize_absmax(&[0.0, 0.0], 8).unwrap(), (vec![0, 0], 0.0));
        assert!(quantize_absmax(&w, 1).is_err());
        assert!(quantize_absmax(&[f32::NAN], 8).is_err());
    }

    #[test]
    fn sign_scale_is_the_least_squares_scale() {
        let w: Vec<f32> = (0..200)
            .map(|i| ((i * 53 % 200) as f32 - 99.5) / 40.0)
            .collect();
        let (signs, scale) = quantize_sign(&w);
        let err = |s: f32| {
            w.iter()
                .zip(&signs)
                .map(|(x, &g)| (x - f32::from(g) * s).powi(2))
                .sum::<f32>()
        };
        // Oracle: no other scale on a fine grid around it does better.
        for k in 1..40 {
            let other = scale * (0.5 + k as f32 / 40.0);
            assert!(err(scale) <= err(other) + 1e-4);
        }
    }

    #[test]
    fn nearest_level_matches_brute_force() {
        let levels = [-1.0f32, -0.3, 0.0, 0.25, 1.0];
        for k in -300..=300 {
            let x = k as f32 / 200.0;
            let got = nearest_level(&levels, x).unwrap();
            let best = levels
                .iter()
                .map(|l| (x - l).abs())
                .fold(f32::INFINITY, f32::min);
            assert_eq!((x - levels[got]).abs(), best, "x = {x}");
        }
        assert!(nearest_level(&[], 0.0).is_err());
        assert!(nearest_level(&levels, f32::NAN).is_err());
    }

    #[test]
    fn phase_error_is_at_most_pi_over_k() {
        for k in [1u32, 2, 4, 16, 255] {
            for i in 0..2000 {
                let theta = -20.0 + i as f64 * 0.0213;
                let j = quantize_phase(theta, k).unwrap();
                assert!(j < k);
                let err = angle_distance(theta, phase_angle(j, k));
                assert!(
                    err <= std::f64::consts::PI / f64::from(k) + 1e-12,
                    "k={k} theta={theta}"
                );
            }
        }
        assert!(quantize_phase(1.0, 0).is_err());
        assert!(quantize_phase(f64::NAN, 4).is_err());
    }
}
