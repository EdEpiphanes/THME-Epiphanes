// SPDX-License-Identifier: AGPL-3.0-only
//! Orthonormal bases (dossier WP5). The one Walsh-Hadamard transform in THME lives here;
//! every caller (BYOA rotated quantization, `MatrixMath`) uses it, so two code paths that
//! must rotate identically cannot drift apart.

use super::error::{EngineError, EngineResult};

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
}
