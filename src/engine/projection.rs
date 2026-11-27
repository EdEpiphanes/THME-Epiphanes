// SPDX-License-Identifier: AGPL-3.0-only
//! Projections that always report the structure they keep (dossier WP8).
//!
//! A projection that silently drops directions is a bug report waiting to happen, so every
//! function here returns the number that says how much was kept next to the coordinates:
//! PCA reports retained variance, classical MDS reports its goodness of fit, and the
//! stereographic map reports its local distance scale. The eigen-decomposition behind PCA
//! and MDS is one cyclic Jacobi solver, shared by both.

use super::error::{ensure_finite, EngineError, EngineResult};

/// Largest matrix side the eigensolver accepts (each sweep costs n^3 operations).
pub const MAX_EIGEN_SIZE: usize = 256;
/// Sweep limit of the cyclic Jacobi method. It converges quadratically, so this is generous.
const MAX_SWEEPS: usize = 64;
/// Off-diagonal mass, relative to the Frobenius norm, at which the Jacobi sweeps stop.
const OFF_DIAGONAL_TOL: f64 = 1e-14;

/// Eigen-decomposition of a symmetric matrix.
#[derive(Debug, Clone, PartialEq)]
pub struct SymEigen {
    /// Eigenvalues, largest first.
    pub values: Vec<f64>,
    /// Row-major n x n: column `k` is the unit eigenvector of `values[k]`.
    pub vectors: Vec<f64>,
}

fn off_diagonal_norm(a: &[f64], n: usize) -> f64 {
    let mut sum = 0.0;
    for i in 0..n {
        for j in 0..n {
            if i != j {
                sum += a[i * n + j] * a[i * n + j];
            }
        }
    }
    sum.sqrt()
}

/// Eigenvalues and orthonormal eigenvectors of a symmetric `n x n` matrix (row-major), by
/// cyclic Jacobi rotations. Errors on bad shapes, non-finite values, asymmetry or no convergence.
pub fn symmetric_eigen(matrix: &[f64], n: usize) -> EngineResult<SymEigen> {
    if n == 0 || n > MAX_EIGEN_SIZE || matrix.len() != n * n {
        return Err(EngineError::InvalidInput(format!(
            "eigen-decomposition needs a square matrix of side 1 to {MAX_EIGEN_SIZE}, got {} values for side {n}",
            matrix.len()
        )));
    }
    ensure_finite("matrix", matrix)?;
    let scale = matrix.iter().fold(1.0f64, |m, v| m.max(v.abs()));
    for i in 0..n {
        for j in i + 1..n {
            if (matrix[i * n + j] - matrix[j * n + i]).abs() > 1e-12 * scale {
                return Err(EngineError::InvalidInput(
                    "matrix is not symmetric".to_string(),
                ));
            }
        }
    }
    let mut a = matrix.to_vec();
    let mut v = vec![0.0; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }
    let frobenius = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let mut off = off_diagonal_norm(&a, n);
    let mut sweeps = 0;
    while off > OFF_DIAGONAL_TOL * frobenius {
        if sweeps == MAX_SWEEPS {
            return Err(EngineError::NoConvergence {
                iterations: sweeps,
                residual: off,
            });
        }
        for p in 0..n {
            for q in p + 1..n {
                let apq = a[p * n + q];
                if apq == 0.0 {
                    continue;
                }
                // Rotation angle that zeroes a_pq (Numerical Recipes convention).
                let theta = (a[q * n + q] - a[p * n + p]) / (2.0 * apq);
                let sign = if theta >= 0.0 { 1.0 } else { -1.0 };
                let t = sign / (theta.abs() + (theta * theta + 1.0).sqrt());
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                // A <- J^T A J, then V <- V J, with J the rotation in the (p, q) plane.
                for k in 0..n {
                    let (akp, akq) = (a[k * n + p], a[k * n + q]);
                    a[k * n + p] = c * akp - s * akq;
                    a[k * n + q] = s * akp + c * akq;
                }
                for k in 0..n {
                    let (apk, aqk) = (a[p * n + k], a[q * n + k]);
                    a[p * n + k] = c * apk - s * aqk;
                    a[q * n + k] = s * apk + c * aqk;
                }
                for k in 0..n {
                    let (vkp, vkq) = (v[k * n + p], v[k * n + q]);
                    v[k * n + p] = c * vkp - s * vkq;
                    v[k * n + q] = s * vkp + c * vkq;
                }
            }
        }
        sweeps += 1;
        off = off_diagonal_norm(&a, n);
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| a[j * n + j].total_cmp(&a[i * n + i]));
    let values = order.iter().map(|&k| a[k * n + k]).collect();
    let mut vectors = vec![0.0; n * n];
    for (col, &k) in order.iter().enumerate() {
        for i in 0..n {
            vectors[i * n + col] = v[i * n + k];
        }
    }
    Ok(SymEigen { values, vectors })
}

/// Principal component analysis of `rows` samples with `cols` features (row-major data).
#[derive(Debug, Clone, PartialEq)]
pub struct Pca {
    /// Row-major rows x k: each sample's coordinates on the kept components.
    pub scores: Vec<f64>,
    /// Row-major k x cols: unit principal directions, largest variance first.
    pub components: Vec<f64>,
    /// Variance along every principal direction, largest first (all `cols` of them).
    pub explained_variance: Vec<f64>,
    /// Share of the total variance kept by the k components, in [0, 1].
    pub retained_variance: f64,
    /// Effective number of directions carrying the variance (see [`participation_ratio`]).
    pub participation_ratio: f64,
}

/// PCA with `k` components on the population covariance (divided by `rows`). Always reports
/// the retained variance; errors when the data has no variance at all.
pub fn pca(data: &[f64], rows: usize, cols: usize, k: usize) -> EngineResult<Pca> {
    if rows == 0 || cols == 0 || data.len() != rows * cols || k == 0 || k > cols {
        return Err(EngineError::InvalidInput(format!(
            "PCA needs rows x cols data and 1 <= k <= cols, got {} values for {rows}x{cols}, k={k}",
            data.len()
        )));
    }
    ensure_finite("data", data)?;
    let mut mean = vec![0.0; cols];
    for r in 0..rows {
        for c in 0..cols {
            mean[c] += data[r * cols + c] / rows as f64;
        }
    }
    let centered: Vec<f64> = (0..rows * cols)
        .map(|idx| data[idx] - mean[idx % cols])
        .collect();
    let mut cov = vec![0.0; cols * cols];
    for r in 0..rows {
        for i in 0..cols {
            for j in 0..cols {
                cov[i * cols + j] += centered[r * cols + i] * centered[r * cols + j] / rows as f64;
            }
        }
    }
    let eig = symmetric_eigen(&cov, cols)?;
    // The covariance is positive semidefinite; rounding may leave tiny negatives, clamped here.
    let explained_variance: Vec<f64> = eig.values.iter().map(|&l| l.max(0.0)).collect();
    let total: f64 = explained_variance.iter().sum();
    if total <= 0.0 {
        return Err(EngineError::InvalidInput(
            "the data has zero variance, so there is no structure to project".to_string(),
        ));
    }
    let retained_variance = explained_variance[..k].iter().sum::<f64>() / total;
    let participation_ratio = participation_ratio(&explained_variance)?;
    let mut components = vec![0.0; k * cols];
    for c in 0..k {
        for i in 0..cols {
            components[c * cols + i] = eig.vectors[i * cols + c];
        }
    }
    let mut scores = vec![0.0; rows * k];
    for r in 0..rows {
        for c in 0..k {
            scores[r * k + c] = (0..cols)
                .map(|i| centered[r * cols + i] * components[c * cols + i])
                .sum();
        }
    }
    Ok(Pca {
        scores,
        components,
        explained_variance,
        retained_variance,
        participation_ratio,
    })
}

/// Participation ratio `(sum l)^2 / sum l^2` of non-negative spectrum values: the effective
/// number of directions. Equals the count for a flat spectrum and 1 for a single direction.
pub fn participation_ratio(values: &[f64]) -> EngineResult<f64> {
    ensure_finite("eigenvalues", values)?;
    if values.iter().any(|&v| v < 0.0) {
        return Err(EngineError::InvalidInput(
            "participation ratio needs non-negative eigenvalues".to_string(),
        ));
    }
    let sum: f64 = values.iter().sum();
    let sum_sq: f64 = values.iter().map(|v| v * v).sum();
    if sum_sq == 0.0 {
        return Err(EngineError::InvalidInput(
            "participation ratio of an all-zero spectrum is undefined".to_string(),
        ));
    }
    Ok(sum * sum / sum_sq)
}

/// Classical (Torgerson) multidimensional scaling output.
#[derive(Debug, Clone, PartialEq)]
pub struct Mds {
    /// Row-major n x k coordinates.
    pub coords: Vec<f64>,
    /// Sum of the k largest eigenvalues over the sum of all |eigenvalues| (Mardia's goodness of
    /// fit). It is 1 when the k coordinates reproduce the distances exactly.
    pub goodness_of_fit: f64,
    /// Every eigenvalue of the double-centred matrix, largest first.
    pub eigenvalues: Vec<f64>,
}

/// Classical MDS of an `n x n` distance matrix into `k` dimensions.
/// shortcut: classical (Torgerson) MDS only; no SMACOF stress refinement, so non-Euclidean
/// distance tables are reported through `goodness_of_fit` rather than optimised.
pub fn classical_mds(distances: &[f64], n: usize, k: usize) -> EngineResult<Mds> {
    if n == 0 || distances.len() != n * n || k == 0 || k > n {
        return Err(EngineError::InvalidInput(format!(
            "MDS needs an n x n distance matrix and 1 <= k <= n, got {} values for n={n}, k={k}",
            distances.len()
        )));
    }
    ensure_finite("distances", distances)?;
    for i in 0..n {
        if distances[i * n + i] != 0.0 {
            return Err(EngineError::InvalidInput(format!(
                "distance from point {i} to itself must be zero"
            )));
        }
        for j in 0..n {
            let d = distances[i * n + j];
            if d < 0.0 {
                return Err(EngineError::InvalidInput(format!(
                    "negative distance at ({i}, {j})"
                )));
            }
            if (d - distances[j * n + i]).abs() > 1e-12 * (1.0 + d.abs()) {
                return Err(EngineError::InvalidInput(
                    "distance matrix is not symmetric".to_string(),
                ));
            }
        }
    }
    let squared: Vec<f64> = distances.iter().map(|d| d * d).collect();
    let row_mean: Vec<f64> = (0..n)
        .map(|i| squared[i * n..(i + 1) * n].iter().sum::<f64>() / n as f64)
        .collect();
    let grand_mean = row_mean.iter().sum::<f64>() / n as f64;
    // B = -1/2 J D^2 J with J = I - 11^T / n; D is symmetric, so column means are row means.
    let mut b = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            b[i * n + j] = -0.5 * (squared[i * n + j] - row_mean[i] - row_mean[j] + grand_mean);
        }
    }
    let eig = symmetric_eigen(&b, n)?;
    let abs_sum: f64 = eig.values.iter().map(|v| v.abs()).sum();
    if abs_sum == 0.0 {
        return Err(EngineError::InvalidInput(
            "all distances are zero: the points coincide and there is nothing to embed".to_string(),
        ));
    }
    let kept: f64 = eig.values[..k].iter().sum();
    let mut coords = vec![0.0; n * k];
    for i in 0..n {
        for c in 0..k {
            coords[i * k + c] = eig.vectors[i * n + c] * eig.values[c].max(0.0).sqrt();
        }
    }
    Ok(Mds {
        coords,
        goodness_of_fit: kept / abs_sum,
        eigenvalues: eig.values,
    })
}

/// A point of the plane from the sphere, with the local scale that says what was distorted.
#[derive(Debug, Clone, PartialEq)]
pub struct StereographicPoint {
    /// Coordinates in R^n, where the sphere lives in R^(n+1).
    pub coords: Vec<f64>,
    /// Plane distance per unit of sphere distance at this point, `1 / (1 - x_last)`. The map is
    /// conformal, so every direction is scaled by this factor and angles are kept.
    pub distance_scale: f64,
}

/// Stereographic projection of a unit vector of R^(n+1) from the north pole `(0, .., 0, 1)`
/// onto the plane `x_last = 0`. The north pole itself has no image and is an error.
pub fn stereographic_from_north(point: &[f64]) -> EngineResult<StereographicPoint> {
    if point.len() < 2 {
        return Err(EngineError::InvalidInput(
            "a point on a sphere needs at least two coordinates".to_string(),
        ));
    }
    ensure_finite("point", point)?;
    let norm = point.iter().map(|v| v * v).sum::<f64>().sqrt();
    if (norm - 1.0).abs() > 1e-9 {
        return Err(EngineError::InvalidInput(format!(
            "point must lie on the unit sphere, its norm is {norm}"
        )));
    }
    let last = point[point.len() - 1];
    let denom = 1.0 - last;
    if denom <= 1e-12 {
        return Err(EngineError::InvalidInput(
            "the north pole has no stereographic image".to_string(),
        ));
    }
    Ok(StereographicPoint {
        coords: point[..point.len() - 1].iter().map(|x| x / denom).collect(),
        distance_scale: 1.0 / denom,
    })
}

/// Inverse stereographic map: plane coordinates back to the unit sphere of R^(n+1).
pub fn stereographic_to_sphere(coords: &[f64]) -> EngineResult<Vec<f64>> {
    ensure_finite("coordinates", coords)?;
    let r2: f64 = coords.iter().map(|y| y * y).sum();
    let den = r2 + 1.0;
    let mut point: Vec<f64> = coords.iter().map(|y| 2.0 * y / den).collect();
    point.push((r2 - 1.0) / den);
    Ok(point)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seeded(n: usize, seed: u64) -> Vec<f64> {
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

    /// Oracle: A v_k = lambda_k v_k, checked from the definition, and V^T V = I.
    #[test]
    fn eigen_pairs_satisfy_the_definition() {
        let n = 5;
        let m = seeded(n * n, 3);
        let a: Vec<f64> = (0..n * n)
            .map(|idx| {
                let (i, j) = (idx / n, idx % n);
                0.5 * (m[i * n + j] + m[j * n + i])
            })
            .collect();
        let e = symmetric_eigen(&a, n).unwrap();
        assert!(e.values.windows(2).all(|w| w[0] >= w[1]), "descending");
        for k in 0..n {
            for i in 0..n {
                let av: f64 = (0..n).map(|j| a[i * n + j] * e.vectors[j * n + k]).sum();
                let lv = e.values[k] * e.vectors[i * n + k];
                assert!((av - lv).abs() < 1e-10, "k={k} i={i}");
            }
            for l in 0..n {
                let dot: f64 = (0..n)
                    .map(|i| e.vectors[i * n + k] * e.vectors[i * n + l])
                    .sum();
                let want = if k == l { 1.0 } else { 0.0 };
                assert!((dot - want).abs() < 1e-10, "orthonormal k={k} l={l}");
            }
        }
    }

    /// Hand-worked: [[2, 1], [1, 2]] has eigenvalues 3 and 1, with eigenvectors (1, 1)/sqrt(2)
    /// and (1, -1)/sqrt(2).
    #[test]
    fn symmetric_two_by_two_by_hand() {
        let e = symmetric_eigen(&[2.0, 1.0, 1.0, 2.0], 2).unwrap();
        assert!((e.values[0] - 3.0).abs() < 1e-12 && (e.values[1] - 1.0).abs() < 1e-12);
        let r = 0.5f64.sqrt();
        assert!((e.vectors[0].abs() - r).abs() < 1e-12);
        assert!((e.vectors[2].abs() - r).abs() < 1e-12);
        // Column 0 of V is the eigenvector for lambda 3: (vectors[0], vectors[2]).
        assert!(
            e.vectors[0] * e.vectors[2] > 0.0,
            "(1, 1) direction for lambda 3"
        );
    }

    #[test]
    fn eigen_rejects_bad_input() {
        assert!(symmetric_eigen(&[], 0).is_err());
        assert!(symmetric_eigen(&[1.0, 2.0, 3.0], 2).is_err());
        assert!(
            symmetric_eigen(&[1.0, 2.0, 0.0, 1.0], 2).is_err(),
            "not symmetric"
        );
        assert!(matches!(
            symmetric_eigen(&[f64::NAN, 0.0, 0.0, 1.0], 2),
            Err(EngineError::NonFinite { .. })
        ));
    }

    /// Hand-worked: points on the line y = 2x have one direction, (1, 2)/sqrt(5), and keep
    /// all their variance in one component.
    #[test]
    fn pca_of_a_line_keeps_everything_in_one_component() {
        let data = [1.0, 2.0, 2.0, 4.0, 3.0, 6.0, -1.0, -2.0];
        let p = pca(&data, 4, 2, 1).unwrap();
        assert!((p.retained_variance - 1.0).abs() < 1e-12);
        let s = 5f64.sqrt();
        let (c0, c1) = (p.components[0], p.components[1]);
        assert!((c0.abs() - 1.0 / s).abs() < 1e-12 && (c1.abs() - 2.0 / s).abs() < 1e-12);
        assert!(c0 * c1 > 0.0);
        assert!((p.participation_ratio - 1.0).abs() < 1e-12);
        assert_eq!(p.scores.len(), 4);
    }

    /// Hand-worked: the four points (1,0), (-1,0), (0,1), (0,-1) have covariance diag(1/2, 1/2).
    /// One component keeps half the variance; the participation ratio is 2.
    #[test]
    fn pca_of_a_cross_keeps_half_the_variance() {
        let data = [1.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0, -1.0];
        let p = pca(&data, 4, 2, 1).unwrap();
        assert!((p.retained_variance - 0.5).abs() < 1e-12);
        assert!((p.participation_ratio - 2.0).abs() < 1e-12);
        let full = pca(&data, 4, 2, 2).unwrap();
        assert!((full.retained_variance - 1.0).abs() < 1e-12);
    }

    #[test]
    fn pca_rejects_zero_variance_and_bad_k() {
        let constant = [1.0, 1.0, 1.0, 1.0];
        assert!(matches!(
            pca(&constant, 2, 2, 1),
            Err(EngineError::InvalidInput(_))
        ));
        let data = [1.0, 0.0, 0.0, 1.0];
        assert!(pca(&data, 2, 2, 0).is_err());
        assert!(pca(&data, 2, 2, 3).is_err());
        assert!(pca(&data, 3, 2, 1).is_err());
    }

    /// Hand-worked: equal eigenvalues give the full count; one dominant eigenvalue gives 1.
    /// [4, 1] gives 25 / 17.
    #[test]
    fn participation_ratio_by_hand() {
        assert!((participation_ratio(&[1.0, 1.0, 1.0, 1.0]).unwrap() - 4.0).abs() < 1e-12);
        assert!((participation_ratio(&[1.0, 0.0]).unwrap() - 1.0).abs() < 1e-12);
        assert!((participation_ratio(&[4.0, 1.0]).unwrap() - 25.0 / 17.0).abs() < 1e-12);
        assert!(participation_ratio(&[1.0, -0.5]).is_err());
        assert!(participation_ratio(&[0.0, 0.0]).is_err());
    }

    /// Oracle: the distances of the output coordinates equal the input distances. The points
    /// (0, 1, 3) lie on a line, so one dimension reproduces them and the fit is exact.
    #[test]
    fn mds_reproduces_distances_of_points_on_a_line() {
        let d = [0.0, 1.0, 3.0, 1.0, 0.0, 2.0, 3.0, 2.0, 0.0];
        let m = classical_mds(&d, 3, 1).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                let got = (m.coords[i] - m.coords[j]).abs();
                assert!((got - d[i * 3 + j]).abs() < 1e-9, "i={i} j={j}");
            }
        }
        assert!((m.goodness_of_fit - 1.0).abs() < 1e-9);
    }

    /// Hand-worked: the unit square has side 1 and diagonal sqrt(2). Two dimensions reproduce
    /// it exactly.
    #[test]
    fn mds_of_the_unit_square_in_two_dimensions() {
        let r2 = 2f64.sqrt();
        let d = [
            0.0, 1.0, r2, 1.0, //
            1.0, 0.0, 1.0, r2, //
            r2, 1.0, 0.0, 1.0, //
            1.0, r2, 1.0, 0.0,
        ];
        let m = classical_mds(&d, 4, 2).unwrap();
        for i in 0..4 {
            for j in 0..4 {
                let (a, b) = (&m.coords[i * 2..i * 2 + 2], &m.coords[j * 2..j * 2 + 2]);
                let got = ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt();
                assert!((got - d[i * 4 + j]).abs() < 1e-9, "i={i} j={j}");
            }
        }
        assert!((m.goodness_of_fit - 1.0).abs() < 1e-9);
        // One dimension of the square keeps only part of the structure, and says so.
        let one = classical_mds(&d, 4, 1).unwrap();
        assert!(one.goodness_of_fit < 1.0 - 1e-6);
    }

    #[test]
    fn mds_rejects_invalid_distances() {
        assert!(classical_mds(&[0.0, -1.0, -1.0, 0.0], 2, 1).is_err());
        assert!(
            classical_mds(&[0.0, 1.0, 2.0, 0.0], 2, 1).is_err(),
            "not symmetric"
        );
        assert!(
            classical_mds(&[1.0, 1.0, 1.0, 1.0], 2, 1).is_err(),
            "diagonal"
        );
        assert!(classical_mds(&[0.0, 1.0, 1.0, 0.0], 2, 3).is_err());
        assert!(
            classical_mds(&[0.0; 4], 2, 1).is_err(),
            "all points coincide"
        );
    }

    /// Hand-worked: the south pole maps to the origin with scale 1/2; the equator maps to
    /// itself with scale 1; (0.6, 0, 0.8) maps to (3, 0) with scale 5.
    #[test]
    fn stereographic_by_hand() {
        let south = stereographic_from_north(&[0.0, 0.0, -1.0]).unwrap();
        assert_eq!(south.coords, vec![0.0, 0.0]);
        assert!((south.distance_scale - 0.5).abs() < 1e-12);
        let equator = stereographic_from_north(&[1.0, 0.0, 0.0]).unwrap();
        assert!((equator.coords[0] - 1.0).abs() < 1e-12 && equator.coords[1].abs() < 1e-12);
        assert!((equator.distance_scale - 1.0).abs() < 1e-12);
        let off = stereographic_from_north(&[0.6, 0.0, 0.8]).unwrap();
        assert!((off.coords[0] - 3.0).abs() < 1e-12 && off.coords[1].abs() < 1e-12);
        assert!((off.distance_scale - 5.0).abs() < 1e-12);
    }

    /// Oracle: the inverse map returns the original point on the sphere.
    #[test]
    fn stereographic_round_trips_on_random_sphere_points() {
        for k in 0..50u64 {
            let raw = seeded(3, 500 + k);
            let norm = raw.iter().map(|v| v * v).sum::<f64>().sqrt();
            let point: Vec<f64> = raw.iter().map(|v| v / norm).collect();
            if point[2] > 0.99 {
                continue;
            }
            let p = stereographic_from_north(&point).unwrap();
            let back = stereographic_to_sphere(&p.coords).unwrap();
            for (a, b) in back.iter().zip(&point) {
                assert!((a - b).abs() < 1e-10, "k={k}");
            }
        }
    }

    #[test]
    fn stereographic_rejects_the_pole_and_off_sphere_points() {
        assert!(stereographic_from_north(&[0.0, 0.0, 1.0]).is_err());
        assert!(
            stereographic_from_north(&[1.0, 1.0, 0.0]).is_err(),
            "not unit length"
        );
        assert!(stereographic_from_north(&[1.0]).is_err());
        assert!(stereographic_from_north(&[f64::NAN, 0.0, 1.0]).is_err());
        assert!(stereographic_to_sphere(&[f64::INFINITY]).is_err());
    }
}
