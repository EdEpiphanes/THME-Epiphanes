// SPDX-License-Identifier: AGPL-3.0-only
//! AVX2 + FMA kernels for the CPU reference forward pass. Every function requires the caller to have verified both features.

use super::{BLOCK, LANES};
use std::arch::x86_64::{
    __m256, _mm256_add_ps, _mm256_fmadd_ps, _mm256_loadu_ps, _mm256_set1_ps, _mm256_setzero_ps,
    _mm256_storeu_ps,
};

#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn horizontal_sum(v: __m256) -> f32 {
    let mut lanes = [0.0f32; LANES];
    // SAFETY: `lanes` holds exactly eight f32 values.
    unsafe { _mm256_storeu_ps(lanes.as_mut_ptr(), v) };
    lanes.iter().sum()
}

#[inline]
#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn dot(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().min(b.len());
    let vector_len = len / LANES * LANES;
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut i = 0;
    while i + 2 * LANES <= vector_len {
        // SAFETY: `i + 16 <= vector_len <= len`, so all four loads are in bounds.
        unsafe {
            acc0 = _mm256_fmadd_ps(
                _mm256_loadu_ps(a.as_ptr().add(i)),
                _mm256_loadu_ps(b.as_ptr().add(i)),
                acc0,
            );
            acc1 = _mm256_fmadd_ps(
                _mm256_loadu_ps(a.as_ptr().add(i + LANES)),
                _mm256_loadu_ps(b.as_ptr().add(i + LANES)),
                acc1,
            );
        }
        i += 2 * LANES;
    }
    if i < vector_len {
        // SAFETY: `i + 8 <= vector_len <= len`.
        unsafe {
            acc0 = _mm256_fmadd_ps(
                _mm256_loadu_ps(a.as_ptr().add(i)),
                _mm256_loadu_ps(b.as_ptr().add(i)),
                acc0,
            );
        }
    }
    // SAFETY: the caller verified AVX2 and FMA.
    let mut sum = unsafe { horizontal_sum(_mm256_add_ps(acc0, acc1)) };
    for (x, y) in a[vector_len..len].iter().zip(&b[vector_len..len]) {
        sum += x * y;
    }
    sum
}

/// `y += alpha * x`
#[inline]
#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn axpy(alpha: f32, x: &[f32], y: &mut [f32]) {
    let len = x.len().min(y.len());
    let vector_len = len / LANES * LANES;
    let scale = _mm256_set1_ps(alpha);
    let mut i = 0;
    while i < vector_len {
        // SAFETY: `i + 8 <= vector_len <= len` for both slices.
        unsafe {
            let updated = _mm256_fmadd_ps(
                scale,
                _mm256_loadu_ps(x.as_ptr().add(i)),
                _mm256_loadu_ps(y.as_ptr().add(i)),
            );
            _mm256_storeu_ps(y.as_mut_ptr().add(i), updated);
        }
        i += LANES;
    }
    for (o, v) in y[vector_len..len].iter_mut().zip(&x[vector_len..len]) {
        *o += alpha * v;
    }
}

#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn matmul(w: &[f32], xs: &[f32], ys: &mut [f32], in_dim: usize, out_dim: usize) {
    let vector_len = in_dim / LANES * LANES;
    let mut x_blocks = xs.chunks_exact(BLOCK * in_dim);
    let mut y_blocks = ys.chunks_exact_mut(BLOCK * out_dim);
    for (xb, yb) in (&mut x_blocks).zip(&mut y_blocks) {
        for (o, w_row) in w.chunks_exact(in_dim).enumerate() {
            let mut acc = [_mm256_setzero_ps(); BLOCK];
            let mut i = 0;
            while i < vector_len {
                // SAFETY: `w_row` has `in_dim` values and `xb` has `BLOCK * in_dim`;
                // `i + 8 <= vector_len <= in_dim` keeps every load in bounds.
                unsafe {
                    let wv = _mm256_loadu_ps(w_row.as_ptr().add(i));
                    for (b, a) in acc.iter_mut().enumerate() {
                        let xv = _mm256_loadu_ps(xb.as_ptr().add(b * in_dim + i));
                        *a = _mm256_fmadd_ps(wv, xv, *a);
                    }
                }
                i += LANES;
            }
            for (b, a) in acc.iter().enumerate() {
                // SAFETY: the caller verified AVX2 and FMA.
                let mut sum = unsafe { horizontal_sum(*a) };
                let x_row = &xb[b * in_dim..(b + 1) * in_dim];
                for (wv, xv) in w_row[vector_len..].iter().zip(&x_row[vector_len..]) {
                    sum += wv * xv;
                }
                yb[b * out_dim + o] = sum;
            }
        }
    }
    let x_rest = x_blocks.remainder();
    let y_rest = y_blocks.into_remainder();
    for (x, y) in x_rest
        .chunks_exact(in_dim)
        .zip(y_rest.chunks_exact_mut(out_dim))
    {
        for (w_row, out) in w.chunks_exact(in_dim).zip(y.iter_mut()) {
            // SAFETY: the caller verified AVX2 and FMA.
            *out = unsafe { dot(w_row, x) };
        }
    }
}
