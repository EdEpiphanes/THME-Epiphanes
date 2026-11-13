// SPDX-License-Identifier: AGPL-3.0-only
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
}
