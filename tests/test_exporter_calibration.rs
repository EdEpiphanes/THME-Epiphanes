#[path = "../src/exporter/calibration.rs"]
mod calibration;

use calibration::{calibrate_per_channel, calibrate_per_group, optimal_ternary_scale};

#[test]
fn optimal_scale_reduces_error_for_mixed_magnitudes() {
    let weights = [2.0, -2.0, 0.25, -0.25];
    let estimate = optimal_ternary_scale(&weights).expect("scale calibration");

    assert!((estimate.scale - 2.0).abs() < 1e-6);
    assert!((estimate.mse - 0.03125).abs() < 1e-9);
    let baseline_mse = weights
        .iter()
        .map(|weight| {
            let reconstructed = 1.0 * (weight / 1.0).clamp(-1.0, 1.0).round();
            f64::from((weight - reconstructed).powi(2))
        })
        .sum::<f64>()
        / weights.len() as f64;
    assert!(estimate.mse < baseline_mse);
}

#[test]
fn per_channel_and_group_calibration_respect_boundaries() {
    let channels = calibrate_per_channel(&[1.0, -1.0, 1.0, 4.0, -4.0, 0.0], 2, 3)
        .expect("channel calibration");
    assert_eq!(channels.len(), 2);
    assert!((channels[0].scale - 1.0).abs() < 1e-6);
    assert!((channels[1].scale - 4.0).abs() < 1e-6);
    assert_eq!(channels[0].mse, 0.0);
    assert_eq!(channels[1].mse, 0.0);

    let groups = calibrate_per_group(&[2.0, -2.0, 0.25, 0.25, 1.0], 2).expect("group calibration");
    assert_eq!(groups.len(), 3);
    assert!((groups[0].scale - 2.0).abs() < 1e-6);
    assert!((groups[1].scale - 0.25).abs() < 1e-6);
    assert!((groups[2].scale - 1.0).abs() < 1e-6);
}

#[test]
fn zero_weights_calibrate_to_zero_scale_and_error() {
    let estimate = optimal_ternary_scale(&[0.0, -0.0, 0.0]).expect("zero calibration");
    assert_eq!(
        estimate,
        calibration::ScaleEstimate {
            scale: 0.0,
            mse: 0.0
        }
    );
}

#[test]
fn invalid_calibration_inputs_return_errors() {
    assert!(optimal_ternary_scale(&[]).is_err());
    assert!(optimal_ternary_scale(&[1.0, f32::NAN]).is_err());
    assert!(calibrate_per_channel(&[1.0, 2.0], 1, 3).is_err());
    assert!(calibrate_per_channel(&[1.0], 0, 1).is_err());
    assert!(calibrate_per_group(&[], 4).is_err());
    assert!(calibrate_per_group(&[1.0], 0).is_err());
}
