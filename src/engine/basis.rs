// SPDX-License-Identifier: AGPL-3.0-only
//! Orthonormal bases (dossier WP5). The one Walsh-Hadamard transform in THME lives here;
//! every caller (BYOA rotated quantization, `MatrixMath`) uses it, so two code paths that
//! must rotate identically cannot drift apart.

use super::error::{ensure_finite, EngineError, EngineResult};

/// Normalized in-place Walsh-Hadamard transform applied to each consecutive block of
/// `block` values. `block` must be a power of two and divide `data.len()`.
///
/// The normalized Hadamard matrix `H / sqrt(block)` is orthonormal and symmetric, so it
/// preserves the L2 norm and is its own inverse: applying it twice returns the input up to
/// rounding.
pub fn walsh_hadamard_blocks(data: &mut [f32], block: usize) -> EngineResult<()> {
    if block == 0 || !block.is_power_of_two() {
        return Err(EngineError::InvalidInput(format!(
            "Walsh-Hadamard block size must be a power of two, got {block}"
        )));
    }
    if !data.len().is_multiple_of(block) {
        return Err(EngineError::InvalidInput(format!(
            "Walsh-Hadamard block size {block} does not divide length {}",
            data.len()
        )));
    }
    let norm = 1.0 / (block as f32).sqrt();
    for chunk in data.chunks_exact_mut(block) {
        let mut step = 1;
        while step < block {
            let mut i = 0;
            while i < block {
                for j in 0..step {
                    let (u, v) = (chunk[i + j], chunk[i + step + j]);
                    chunk[i + j] = u + v;
                    chunk[i + step + j] = u - v;
                }
                i += 2 * step;
            }
            step *= 2;
        }
        for x in chunk.iter_mut() {
            *x *= norm;
        }
    }
    Ok(())
}

/// Normalized Walsh-Hadamard transform of the whole slice (length a power of two; empty
/// and single-element slices are returned unchanged).
pub fn walsh_hadamard(data: &mut [f32]) -> EngineResult<()> {
    if data.len() < 2 {
        return Ok(());
    }
    walsh_hadamard_blocks(data, data.len())
}

/// In-place radix-2 FFT, forward sign convention: `X_k = sum_j x_j exp(-2 pi i j k / n)`,
/// unnormalized. `re` and `im` hold the real and imaginary parts and must have the same
/// length, a nonzero power of two. The naive DFT is the test oracle, not a second code path.
pub fn fft(re: &mut [f64], im: &mut [f64]) -> EngineResult<()> {
    let n = re.len();
    if n == 0 || !n.is_power_of_two() || im.len() != n {
        return Err(EngineError::InvalidInput(format!(
            "FFT needs equal lengths that are a nonzero power of two, got {n} and {}",
            im.len()
        )));
    }
    ensure_finite("FFT real part", re)?;
    ensure_finite("FFT imaginary part", im)?;
    let bits = n.trailing_zeros();
    for i in 0..n {
        let j = (0..bits).fold(0, |acc, b| acc | (((i >> b) & 1) << (bits - 1 - b)));
        if j > i {
            re.swap(i, j);
            im.swap(i, j);
        }
    }
    let mut len = 2;
    while len <= n {
        let half = len / 2;
        let angle = -2.0 * std::f64::consts::PI / len as f64;
        for start in (0..n).step_by(len) {
            for k in 0..half {
                let (s, c) = (angle * k as f64).sin_cos();
                let (a, b) = (start + k, start + k + half);
                let tr = c * re[b] - s * im[b];
                let ti = c * im[b] + s * re[b];
                re[b] = re[a] - tr;
                im[b] = im[a] - ti;
                re[a] += tr;
                im[a] += ti;
            }
        }
        len <<= 1;
    }
    Ok(())
}

/// Givens rotation `(c, s)` with `c*a + s*b = r` and `-s*a + c*b = 0`, where `r = hypot(a, b)`.
/// For `a = b = 0` the identity `(1, 0)` is returned.
pub fn givens(a: f64, b: f64) -> EngineResult<(f64, f64)> {
    ensure_finite("Givens input", &[a, b])?;
    let r = a.hypot(b);
    if r == 0.0 {
        return Ok((1.0, 0.0));
    }
    Ok((a / r, b / r))
}

/// Applies the Givens rotation `(c, s)` to coordinates `i` and `j` of `x` in place:
/// `x_i <- c x_i + s x_j` and `x_j <- -s x_i + c x_j`. Orthogonal, so the norm is kept.
pub fn apply_givens(x: &mut [f64], i: usize, j: usize, (c, s): (f64, f64)) -> EngineResult<()> {
    if i == j || i >= x.len() || j >= x.len() {
        return Err(EngineError::InvalidInput(format!(
            "Givens indices {i} and {j} must differ and be below {}",
            x.len()
        )));
    }
    let (xi, xj) = (x[i], x[j]);
    x[i] = c * xi + s * xj;
    x[j] = -s * xi + c * xj;
    Ok(())
}

/// Unit Householder vector `v` with `(I - 2 v v^T) x = alpha e_1`, `alpha = -sign(x_0) |x|`.
/// The sign choice keeps `x_0 - alpha` free of cancellation. Errors on the zero vector.
pub fn householder_vector(x: &[f64]) -> EngineResult<Vec<f64>> {
    if x.is_empty() {
        return Err(EngineError::InvalidInput(
            "Householder vector of an empty slice".to_string(),
        ));
    }
    ensure_finite("Householder input", x)?;
    let norm = x.iter().map(|v| v * v).sum::<f64>().sqrt();
    ensure_finite("Householder norm", &[norm])?;
    if norm == 0.0 {
        return Err(EngineError::InvalidInput(
            "the zero vector has no Householder reflector".to_string(),
        ));
    }
    let alpha = if x[0] < 0.0 { norm } else { -norm };
    let mut v = x.to_vec();
    v[0] -= alpha;
    let len = v.iter().map(|t| t * t).sum::<f64>().sqrt();
    for t in &mut v {
        *t /= len;
    }
    Ok(v)
}

/// Applies the reflector `I - 2 v v^T` to `y` in place. `v` must be unit length (as returned
/// by [`householder_vector`]); a non-unit `v` is an error rather than a silent non-reflection.
pub fn apply_householder(v: &[f64], y: &mut [f64]) -> EngineResult<()> {
    if v.len() != y.len() {
        return Err(EngineError::InvalidInput(format!(
            "Householder vector length {} differs from data length {}",
            v.len(),
            y.len()
        )));
    }
    let norm_sq: f64 = v.iter().map(|t| t * t).sum();
    if (norm_sq - 1.0).abs() > 1e-9 {
        return Err(EngineError::InvalidInput(format!(
            "Householder vector must be unit length, squared norm is {norm_sq}"
        )));
    }
    let dot: f64 = v.iter().zip(y.iter()).map(|(a, b)| a * b).sum();
    for (yi, vi) in y.iter_mut().zip(v) {
        *yi -= 2.0 * dot * vi;
    }
    Ok(())
}

/// Computes diagonal column-side Sinkhorn equilibrium scales `S` for row-major
/// `w` (`rows x cols`) preserving zero entries (`w_ij == 0 => (W*S)_ij == 0`)
/// for the `BITCOS` sparse ternary branch (`ADR-0002`).
pub fn sinkhorn_column_scales(
    w: &[f32],
    rows: usize,
    cols: usize,
    iters: usize,
) -> EngineResult<Vec<f32>> {
    if rows == 0 || cols == 0 || w.len() != rows * cols {
        return Err(EngineError::InvalidInput(format!(
            "sinkhorn shape mismatch: len={} for {rows}x{cols}",
            w.len()
        )));
    }
    let mut r_scale = vec![1.0_f32; rows];
    let mut c_scale = vec![1.0_f32; cols];
    let eps = 1e-6_f32;

    for _ in 0..iters.max(1) {
        for r in 0..rows {
            let row = &w[r * cols..(r + 1) * cols];
            let mut acc = 0.0_f32;
            for (c, &val) in row.iter().enumerate() {
                let v = (val * c_scale[c]).abs();
                acc += v * v;
            }
            r_scale[r] = 1.0 / (acc / (cols as f32) + eps).sqrt();
        }
        for c in 0..cols {
            let mut acc = 0.0_f32;
            for r in 0..rows {
                let v = (w[r * cols + c] * r_scale[r]).abs();
                acc += v * v;
            }
            c_scale[c] = 1.0 / (acc / (rows as f32) + eps).sqrt();
        }
    }
    // Normalize geometric mean of column scales to 1.0 for numerical stability.
    let log_mean: f64 = c_scale
        .iter()
        .map(|&s| f64::from(s.max(eps)).ln())
        .sum::<f64>()
        / (cols as f64);
    let norm = (-log_mean).exp() as f32;
    for s in &mut c_scale {
        *s *= norm;
    }
    Ok(c_scale)
}

/// Applies a 3-factor Kronecker preconditioner `(P1 x P2 x P3)` in-place to a
/// vector `x` of length `d1 * d2 * d3` (supporting non-power-of-2 FFN widths
/// such as `5632 = 32 * 16 * 11`).
pub fn kronecker_3factor_apply(
    x: &mut [f32],
    p1: &[f32],
    p2: &[f32],
    p3: &[f32],
    dims: (usize, usize, usize),
) -> EngineResult<()> {
    let (d1, d2, d3) = dims;
    if d1 == 0 || d2 == 0 || d3 == 0 || x.len() != d1 * d2 * d3 {
        return Err(EngineError::InvalidInput(format!(
            "kronecker_3factor_apply length {} != {d1}*{d2}*{d3}",
            x.len()
        )));
    }
    if p1.len() != d1 * d1 || p2.len() != d2 * d2 || p3.len() != d3 * d3 {
        return Err(EngineError::InvalidInput(
            "kronecker factor matrix dimension mismatch".to_string(),
        ));
    }

    let mut tmp = vec![0.0_f32; x.len()];
    // Mode-3: multiply each d3-fiber by P3.
    for i in 0..(d1 * d2) {
        let off = i * d3;
        for r in 0..d3 {
            let mut sum = 0.0_f32;
            for c in 0..d3 {
                sum += p3[r * d3 + c] * x[off + c];
            }
            tmp[off + r] = sum;
        }
    }
    // Mode-2: multiply along d2.
    x.copy_from_slice(&tmp);
    for i1 in 0..d1 {
        for i3 in 0..d3 {
            for r in 0..d2 {
                let mut sum = 0.0_f32;
                for c in 0..d2 {
                    sum += p2[r * d2 + c] * tmp[(i1 * d2 + c) * d3 + i3];
                }
                x[(i1 * d2 + r) * d3 + i3] = sum;
            }
        }
    }
    // Mode-1: multiply along d1.
    tmp.copy_from_slice(x);
    for i23 in 0..(d2 * d3) {
        for r in 0..d1 {
            let mut sum = 0.0_f32;
            for c in 0..d1 {
                sum += p1[r * d1 + c] * tmp[c * (d2 * d3) + i23];
            }
            x[r * (d2 * d3) + i23] = sum;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seeded(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((s >> 40) as f32 / 16_777_216.0) - 0.5
            })
            .collect()
    }

    fn fnv(values: &[f32]) -> u64 {
        values.iter().fold(0xcbf29ce484222325u64, |h, x| {
            (h ^ x.to_bits() as u64).wrapping_mul(0x100000001b3)
        })
    }

    /// Oracle: the Sylvester Hadamard matrix, H[i][j] = (-1)^popcount(i & j) / sqrt(n),
    /// multiplied out in f64.
    #[test]
    fn matches_the_explicit_hadamard_matrix() {
        for n in [2usize, 4, 8, 16, 64] {
            let x = seeded(n, n as u64);
            let mut fast = x.clone();
            walsh_hadamard(&mut fast).unwrap();
            let scale = 1.0 / (n as f64).sqrt();
            for (i, got) in fast.iter().enumerate() {
                let want: f64 = (0..n)
                    .map(|j| {
                        let sign = if (i & j).count_ones() % 2 == 0 {
                            1.0
                        } else {
                            -1.0
                        };
                        sign * f64::from(x[j])
                    })
                    .sum::<f64>()
                    * scale;
                assert!((f64::from(*got) - want).abs() < 1e-5, "n={n} i={i}");
            }
        }
    }

    #[test]
    fn preserves_norm_and_is_its_own_inverse() {
        let x = seeded(512, 7);
        let mut y = x.clone();
        walsh_hadamard_blocks(&mut y, 128).unwrap();
        let (nx, ny): (f32, f32) = (x.iter().map(|v| v * v).sum(), y.iter().map(|v| v * v).sum());
        assert!((nx - ny).abs() < 1e-3 * nx);
        walsh_hadamard_blocks(&mut y, 128).unwrap();
        assert!(x.iter().zip(&y).all(|(a, b)| (a - b).abs() < 1e-5));
        // Parseval on an impulse: all energy spreads evenly.
        let mut e = vec![0.0f32; 16];
        e[0] = 1.0;
        walsh_hadamard(&mut e).unwrap();
        assert!(e.iter().all(|v| (v - 0.25).abs() < 1e-7));
    }

    /// Bit-identical to the two implementations this module replaced (hashes recorded
    /// from `byoa::wht_blocks` and `MatrixMath::fast_walsh_hadamard_transform` before the merge).
    #[test]
    fn bit_identical_to_the_replaced_copies() {
        let mut w = seeded(256, 12345);
        walsh_hadamard_blocks(&mut w, 64).unwrap();
        assert_eq!(fnv(&w), 0xa11216529b28661d);
        let mut v: Vec<f32> = (0..16).map(|i| i as f32 * 0.37 - 2.0).collect();
        walsh_hadamard(&mut v).unwrap();
        assert_eq!(fnv(&v), 0x5d5ebe90cff64b77);
    }

    #[test]
    fn bad_shapes_are_errors() {
        assert!(walsh_hadamard_blocks(&mut [0.0; 8], 3).is_err());
        assert!(walsh_hadamard_blocks(&mut [0.0; 6], 4).is_err());
        assert!(walsh_hadamard_blocks(&mut [0.0; 8], 0).is_err());
        assert!(walsh_hadamard(&mut [0.0; 3]).is_err());
        assert!(walsh_hadamard(&mut []).is_ok());
    }

    #[test]
    fn sinkhorn_preserves_zeros_and_balances_columns() {
        let w = [
            10.0_f32, 0.0, 0.1, //
            0.0, 5.0, 0.0, //
            10.0, 0.0, 0.1,
        ];
        let s = sinkhorn_column_scales(&w, 3, 3, 8).unwrap();
        assert_eq!(s.len(), 3);
        // Zero entries remain exactly zero after column scaling W * S.
        for r in 0..3 {
            for c in 0..3 {
                let scaled = w[r * 3 + c] * s[c];
                if w[r * 3 + c] == 0.0 {
                    assert_eq!(scaled, 0.0);
                }
            }
        }
        // Column 2 (small entries 0.1) receives a larger scale than column 0 (10.0).
        assert!(s[2] > s[0]);
    }

    #[test]
    fn kronecker_3factor_matches_explicit_product() {
        let (d1, d2, d3) = (2, 3, 2);
        let p1 = [1.0_f32, 0.5, -0.5, 2.0];
        let p2 = [
            1.0_f32, 0.0, 0.2, //
            0.1, 1.5, 0.0, //
            0.0, -0.3, 1.0,
        ];
        let p3 = [0.8_f32, 0.6, -0.6, 0.8];
        let x: Vec<f32> = (0..(d1 * d2 * d3))
            .map(|i| (i as f32) * 0.25 - 1.0)
            .collect();

        let mut got = x.clone();
        kronecker_3factor_apply(&mut got, &p1, &p2, &p3, (d1, d2, d3)).unwrap();

        // Explicit Kronecker matrix-vector oracle.
        let n = d1 * d2 * d3;
        let mut want = vec![0.0_f32; n];
        for r1 in 0..d1 {
            for r2 in 0..d2 {
                for r3 in 0..d3 {
                    let row = (r1 * d2 + r2) * d3 + r3;
                    let mut sum = 0.0_f32;
                    for c1 in 0..d1 {
                        for c2 in 0..d2 {
                            for c3 in 0..d3 {
                                let col = (c1 * d2 + c2) * d3 + c3;
                                let k = p1[r1 * d1 + c1] * p2[r2 * d2 + c2] * p3[r3 * d3 + c3];
                                sum += k * x[col];
                            }
                        }
                    }
                    want[row] = sum;
                }
            }
        }
        for (a, b) in got.iter().zip(want.iter()) {
            assert!((a - b).abs() < 1e-5, "got={a} want={b}");
        }
    }

    fn seeded64(n: usize, seed: u64) -> Vec<f64> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((s >> 11) as f64 / 9_007_199_254_740_992.0) - 0.5
            })
            .collect()
    }

    /// Oracle: the naive O(n^2) DFT, X_k = sum_j x_j exp(-2 pi i j k / n), in f64.
    fn naive_dft(re: &[f64], im: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let n = re.len();
        let mut out_re = vec![0.0; n];
        let mut out_im = vec![0.0; n];
        for k in 0..n {
            for j in 0..n {
                let angle = -2.0 * std::f64::consts::PI * ((j * k) % n) as f64 / n as f64;
                let (s, c) = angle.sin_cos();
                out_re[k] += re[j] * c - im[j] * s;
                out_im[k] += re[j] * s + im[j] * c;
            }
        }
        (out_re, out_im)
    }

    #[test]
    fn fft_matches_the_naive_dft() {
        for n in [1usize, 2, 4, 8, 16, 32, 64] {
            let re = seeded64(n, n as u64);
            let im = seeded64(n, 1000 + n as u64);
            let (want_re, want_im) = naive_dft(&re, &im);
            let (mut got_re, mut got_im) = (re.clone(), im.clone());
            fft(&mut got_re, &mut got_im).unwrap();
            for k in 0..n {
                assert!((got_re[k] - want_re[k]).abs() < 1e-10, "n={n} re k={k}");
                assert!((got_im[k] - want_im[k]).abs() < 1e-10, "n={n} im k={k}");
            }
        }
    }

    /// Hand-worked: an impulse has a flat spectrum; a constant has one spike at bin 0; and
    /// [1, 0, -1, 0] has spectrum [0, 2, 0, 2].
    #[test]
    fn fft_hand_worked_cases() {
        let (mut re, mut im) = ([1.0, 0.0, 0.0, 0.0], [0.0; 4]);
        fft(&mut re, &mut im).unwrap();
        assert_eq!(re, [1.0; 4]);
        assert_eq!(im, [0.0; 4]);

        let (mut re, mut im) = ([1.0; 4], [0.0; 4]);
        fft(&mut re, &mut im).unwrap();
        assert_eq!(re, [4.0, 0.0, 0.0, 0.0]);

        let (mut re, mut im) = ([1.0, 0.0, -1.0, 0.0], [0.0; 4]);
        fft(&mut re, &mut im).unwrap();
        for (got, want) in re.iter().zip([0.0, 2.0, 0.0, 2.0]) {
            assert!((got - want).abs() < 1e-12);
        }
        assert!(im.iter().all(|v| v.abs() < 1e-12));
    }

    #[test]
    fn fft_rejects_bad_shapes_and_non_finite_values() {
        assert!(fft(&mut [0.0; 3], &mut [0.0; 3]).is_err());
        assert!(fft(&mut [], &mut []).is_err());
        assert!(fft(&mut [0.0; 4], &mut [0.0; 2]).is_err());
        assert!(matches!(
            fft(&mut [f64::NAN, 0.0], &mut [0.0; 2]),
            Err(EngineError::NonFinite { .. })
        ));
    }

    /// Hand-worked: (3, 4) has r = 5, so (c, s) = (0.6, 0.8) and the rotated pair is (5, 0).
    #[test]
    fn givens_zeroes_the_second_component() {
        let (c, s) = givens(3.0, 4.0).unwrap();
        assert!((c - 0.6).abs() < 1e-15 && (s - 0.8).abs() < 1e-15);
        assert!((c * 3.0 + s * 4.0 - 5.0).abs() < 1e-12);
        assert!((-s * 3.0 + c * 4.0).abs() < 1e-12);
        assert_eq!(givens(2.0, 0.0).unwrap(), (1.0, 0.0));
        assert_eq!(givens(0.0, -2.0).unwrap(), (0.0, -1.0));
        assert_eq!(givens(0.0, 0.0).unwrap(), (1.0, 0.0));
        assert!(givens(f64::INFINITY, 1.0).is_err());
    }

    #[test]
    fn givens_rotation_in_a_plane_preserves_the_norm() {
        let mut x = vec![1.0, 2.0, 3.0, 4.0];
        let g = givens(x[1], x[3]).unwrap();
        apply_givens(&mut x, 1, 3, g).unwrap();
        assert!(x[3].abs() < 1e-12);
        assert!((x[1] - 20f64.sqrt()).abs() < 1e-12);
        assert_eq!((x[0], x[2]), (1.0, 3.0));
        let norm: f64 = x.iter().map(|v| v * v).sum();
        assert!((norm - 30.0).abs() < 1e-12);
        assert!(apply_givens(&mut x, 1, 1, g).is_err());
        assert!(apply_givens(&mut x, 1, 4, g).is_err());
    }

    /// Hand-worked: x = (3, 4), alpha = -5, v = (8, 4) / sqrt(80), and H x = (-5, 0).
    #[test]
    fn householder_maps_a_vector_onto_the_first_axis() {
        let x = [3.0, 4.0];
        let v = householder_vector(&x).unwrap();
        let root = 80f64.sqrt();
        assert!((v[0] - 8.0 / root).abs() < 1e-12 && (v[1] - 4.0 / root).abs() < 1e-12);
        let mut y = x.to_vec();
        apply_householder(&v, &mut y).unwrap();
        assert!((y[0] + 5.0).abs() < 1e-12 && y[1].abs() < 1e-12);
        // Reflecting twice returns the input.
        apply_householder(&v, &mut y).unwrap();
        assert!((y[0] - 3.0).abs() < 1e-12 && (y[1] - 4.0).abs() < 1e-12);
    }

    /// Oracle: the explicit matrix H = I - 2 v v^T, built and multiplied in the test.
    #[test]
    fn householder_matches_the_explicit_matrix() {
        for n in [2usize, 3, 5, 8] {
            let x = seeded64(n, 40 + n as u64);
            let v = householder_vector(&x).unwrap();
            let mut got = seeded64(n, 90 + n as u64);
            let want: Vec<f64> = (0..n)
                .map(|i| {
                    (0..n)
                        .map(|j| {
                            let h = f64::from(u8::from(i == j)) - 2.0 * v[i] * v[j];
                            h * got[j]
                        })
                        .sum()
                })
                .collect();
            apply_householder(&v, &mut got).unwrap();
            for (a, b) in got.iter().zip(&want) {
                assert!((a - b).abs() < 1e-12, "n={n}");
            }
            let norm_v: f64 = v.iter().map(|t| t * t).sum();
            assert!((norm_v - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn householder_rejects_zero_mismatch_and_non_unit_input() {
        assert!(matches!(
            householder_vector(&[0.0, 0.0]),
            Err(EngineError::InvalidInput(_))
        ));
        assert!(householder_vector(&[]).is_err());
        assert!(householder_vector(&[1.0, f64::NAN]).is_err());
        assert!(apply_householder(&[1.0, 0.0], &mut [1.0]).is_err());
        assert!(apply_householder(&[2.0, 0.0], &mut [1.0, 1.0]).is_err());
    }
}
