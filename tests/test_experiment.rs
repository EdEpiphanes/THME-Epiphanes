use thme::experiment::{ExperimentLab, ExperimentTechnique};

#[test]
fn test_experiment_noise_injection() {
    let mut weights = vec![1.0, 2.0, -1.0, -2.0, 0.5];
    let technique = ExperimentTechnique::NoiseInjection {
        variance: 0.05,
        distribution: "Gaussian".to_string(),
    };

    let metrics = ExperimentLab::execute(&technique, &mut weights);
    assert_eq!(metrics.affected_elements, 5);
    assert!(metrics.mean_squared_error > 0.0);
    assert!(metrics.signal_to_noise_ratio_db.is_finite());
}

#[test]
fn test_experiment_spectral_reshaping() {
    let mut weights = vec![10.0, 5.0, 2.0, 1.0, 0.5, 0.1];
    let technique = ExperimentTechnique::SpectralReshaping {
        gamma_exponent: 0.8,
    };

    let metrics = ExperimentLab::execute(&technique, &mut weights);
    assert_eq!(metrics.affected_elements, 6);
    assert!(metrics.transformed_l2_norm > 0.0);
}

#[test]
fn test_experiment_phase_modulation() {
    let mut weights = vec![0.5; 8];
    let technique = ExperimentTechnique::PhaseModulation {
        frequency_rad: 0.25,
        phase_shift_rad: 0.1,
    };

    let metrics = ExperimentLab::execute(&technique, &mut weights);
    assert_eq!(metrics.affected_elements, 8);
    assert!(metrics.mean_squared_error > 0.0);
}

#[test]
fn test_experiment_activation_warping() {
    let mut weights = vec![0.1, 0.2, 0.3, 0.4];
    let technique = ExperimentTechnique::ActivationWarping { blend_factor: 0.5 };

    let metrics = ExperimentLab::execute(&technique, &mut weights);
    assert_eq!(metrics.affected_elements, 4);
}

#[test]
fn test_experiment_stochastic_dropblock() {
    let mut weights = vec![1.0; 16];
    let technique = ExperimentTechnique::StochasticDropBlock {
        drop_probability: 0.5,
        block_size: 4,
    };

    let metrics = ExperimentLab::execute(&technique, &mut weights);
    assert!(metrics.sparsity_delta_pct >= 0.0);
}
