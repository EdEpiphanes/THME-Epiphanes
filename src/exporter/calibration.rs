//! MSE-optimal ternary scales for row-wise channels and fixed-size groups.

use anyhow::{bail, Result};

/// Scale and mean squared reconstruction error for one channel or group.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScaleEstimate {
    pub scale: f32,
    pub mse: f64,
}

/// Finds the MSE-optimal scale for ternary quantization of one value slice.
///
/// Values are quantized as `scale * round(clamp(value / scale, -1, 1))`.
pub fn optimal_ternary_scale(values: &[f32]) -> Result<ScaleEstimate> {
    if values.is_empty() {
        bail!("Cannot calibrate an empty tensor channel or group");
    }
    if values.iter().any(|value| !value.is_finite()) {
        bail!("Cannot calibrate weights containing NaN or infinity");
    }

    let mut magnitudes: Vec<f64> = values.iter().map(|value| f64::from(value.abs())).collect();
    magnitudes.sort_by(f64::total_cmp);

    let mut prefix_sum = Vec::with_capacity(magnitudes.len() + 1);
    let mut prefix_squared_sum = Vec::with_capacity(magnitudes.len() + 1);
    prefix_sum.push(0.0);
    prefix_squared_sum.push(0.0);
    for magnitude in &magnitudes {
        prefix_sum.push(prefix_sum.last().copied().unwrap_or_default() + magnitude);
        prefix_squared_sum
            .push(prefix_squared_sum.last().copied().unwrap_or_default() + magnitude * magnitude);
    }

    let total_squared_error = *prefix_squared_sum.last().unwrap_or(&0.0);
    let mut best = ScaleEstimate {
        scale: 0.0,
        mse: total_squared_error / values.len() as f64,
    };

    let mut consider = |scale: f32| {
        if !scale.is_finite() || scale < 0.0 {
            return;
        }
        let scale64 = f64::from(scale);
        let inactive = magnitudes.partition_point(|magnitude| *magnitude < scale64 * 0.5);
        let active_count = magnitudes.len() - inactive;
        let active_sum = prefix_sum[magnitudes.len()] - prefix_sum[inactive];
        let squared_error = (total_squared_error - 2.0 * scale64 * active_sum
            + scale64 * scale64 * active_count as f64)
            .max(0.0);
        let mse = squared_error / values.len() as f64;
        if mse < best.mse {
            best = ScaleEstimate { scale, mse };
        }
    };

    // Within each fixed ternary partition, the least-squares scale is the mean
    // magnitude of nonzero values. Check each possible partition.
    for active_count in 1..=magnitudes.len() {
        let active_sum = prefix_sum[magnitudes.len()] - prefix_sum[magnitudes.len() - active_count];
        consider((active_sum / active_count as f64) as f32);
    }

    // The active set changes when alpha reaches twice an input magnitude.
    // Checking these boundaries covers minima that fall at a partition edge.
    for magnitude in &magnitudes {
        consider((2.0 * magnitude).min(f64::from(f32::MAX)) as f32);
    }

    Ok(best)
}

/// Calibrates one ternary scale for each row of a row-major matrix.
pub fn calibrate_per_channel(
    weights: &[f32],
    rows: usize,
    columns: usize,
) -> Result<Vec<ScaleEstimate>> {
    if rows == 0 || columns == 0 {
        bail!("Channel calibration requires nonzero row and column counts");
    }
    let expected_len = rows
        .checked_mul(columns)
        .ok_or_else(|| anyhow::anyhow!("Channel dimensions overflow usize"))?;
    if weights.len() != expected_len {
        bail!(
            "Channel calibration expected {expected_len} weights for {rows}x{columns}, received {}",
            weights.len()
        );
    }

    weights
        .chunks_exact(columns)
        .map(optimal_ternary_scale)
        .collect()
}

/// Calibrates scales for contiguous groups; the final group may be shorter.
pub fn calibrate_per_group(weights: &[f32], group_size: usize) -> Result<Vec<ScaleEstimate>> {
    if weights.is_empty() {
        bail!("Cannot calibrate an empty weight slice");
    }
    if group_size == 0 {
        bail!("Group size must be greater than zero");
    }

    weights
        .chunks(group_size)
        .map(optimal_ternary_scale)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_optimal_ternary_scale_exact_ternary() {
        // Values that are exact multiples of 0.5: -0.5, 0.0, 0.5
        let values = vec![-0.5, 0.0, 0.5, 0.5, -0.5];
        let estimate = optimal_ternary_scale(&values).expect("calibrate exact ternary");
        assert!((estimate.scale - 0.5).abs() < 1e-4);
        assert!(estimate.mse < 1e-6);
    }

    #[test]
    fn test_optimal_ternary_scale_beats_naive_absmean() {
        // Skewed outlier distribution where optimal L2 scale differs significantly from L1 mean
        let values = vec![0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 2.0];
        let estimate = optimal_ternary_scale(&values).expect("calibrate outliers");

        // Naive absmean: (7 * 0.05 + 2.0) / 8 = 2.35 / 8 ≈ 0.29375
        let naive_scale: f32 = values.iter().map(|v| v.abs()).sum::<f32>() / values.len() as f32;
        let naive_mse: f64 = values
            .iter()
            .map(|&v| {
                let q = (v / naive_scale).round().clamp(-1.0, 1.0);
                let diff = (v - naive_scale * q) as f64;
                diff * diff
            })
            .sum::<f64>()
            / values.len() as f64;

        assert!(
            estimate.mse <= naive_mse,
            "Optimal MSE {} must be <= naive MSE {}",
            estimate.mse,
            naive_mse
        );
    }

    #[test]
    fn test_calibrate_per_channel_dimensions() {
        let rows = 4;
        let cols = 8;
        let weights = vec![0.25f32; rows * cols];
        let estimates = calibrate_per_channel(&weights, rows, cols).expect("calibrate channels");
        assert_eq!(estimates.len(), rows);
        for est in &estimates {
            assert!((est.scale - 0.25).abs() < 1e-4);
            assert!(est.mse < 1e-6);
        }
    }

    #[test]
    fn test_calibrate_per_group_grouping() {
        let weights = vec![0.1f32; 64];
        let estimates = calibrate_per_group(&weights, 16).expect("calibrate groups");
        assert_eq!(estimates.len(), 4); // 64 / 16 = 4 groups
        for est in &estimates {
            assert!((est.scale - 0.1).abs() < 1e-4);
        }
    }

    #[test]
    fn test_calibration_rejection_of_invalid_inputs() {
        // Empty
        assert!(optimal_ternary_scale(&[]).is_err());
        // NaN
        assert!(optimal_ternary_scale(&[1.0, f32::NAN, -1.0]).is_err());
        // Infinity
        assert!(optimal_ternary_scale(&[1.0, f32::INFINITY]).is_err());
        // Zero dimensions in channel calibration
        assert!(calibrate_per_channel(&[1.0], 0, 1).is_err());
        // Mismatched length in channel calibration
        assert!(calibrate_per_channel(&[1.0, 2.0], 2, 2).is_err());
        // Zero group size
        assert!(calibrate_per_group(&[1.0, 2.0], 0).is_err());
    }
}
