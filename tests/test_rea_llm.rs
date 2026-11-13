//! Integration Tests for REA-LLM & EWHA Engine

use thme::transform::rea::{EwhaConstraints, LayerReachProfile, LayerRepresentationMix, ReaEngine};

#[test]
fn test_layer_reach_profile_80_20() {
    let num_layers = 10;
    let checkpoint = 2; // Checkpoint at 30% depth (L0, L1, L2)
    let reach = LayerReachProfile::new_two_path(num_layers, checkpoint, 80.0);

    assert_eq!(reach.num_layers, 10);
    assert_eq!(reach.checkpoint_layer, 2);
    assert!((reach.p_continue - 0.20).abs() < 1e-4);

    // Early prefix layers (L0..=2) must have q_l = 1.0
    for &q in &reach.q_values[..=checkpoint] {
        assert_eq!(q, 1.0);
    }

    // Remaining layers (L3..=9) must have q_l = 0.20
    for &q in &reach.q_values[(checkpoint + 1)..num_layers] {
        assert!((q - 0.20).abs() < 1e-4);
    }

    // Expected depth fraction: (3 * 1.0 + 7 * 0.2) / 10 = 4.4 / 10 = 0.44
    let expected_f = reach.expected_depth_fraction();
    assert!((expected_f - 0.44).abs() < 1e-3);
}

#[test]
fn test_layer_representation_mix_properties() {
    let mix = LayerRepresentationMix::new(0.20, 0.70, 0.10);
    assert!((mix.alpha_binary - 0.20).abs() < 1e-4);
    assert!((mix.beta_ternary - 0.70).abs() < 1e-4);
    assert!((mix.gamma_phase - 0.10).abs() < 1e-4);

    // Bits per param: 0.2 * 1 + 0.7 * 2 + 0.1 * 4 = 0.2 + 1.4 + 0.4 = 2.0
    let bits = mix.effective_bits_per_param();
    assert!((bits - 2.0).abs() < 1e-4);

    // Compression ratio: 16.0 / 2.0 = 8.0
    assert!((mix.compression_ratio() - 8.0).abs() < 1e-4);
}

#[test]
fn test_memory_traffic_and_energy_calculation() {
    let reach = LayerReachProfile::new_two_path(4, 1, 80.0);
    // Layers 0, 1: q = 1.0
    // Layers 2, 3: q = 0.2
    let layer_bytes = vec![
        10 * 1024 * 1024,
        10 * 1024 * 1024,
        10 * 1024 * 1024,
        10 * 1024 * 1024,
    ];

    // Expected bytes: (1.0 * 10 + 1.0 * 10 + 0.2 * 10 + 0.2 * 10) MB = 24.0 MB
    let traffic_mb = ReaEngine::compute_expected_memory_traffic(&reach, &layer_bytes);
    assert!((traffic_mb - 24.0).abs() < 1e-3);

    let layer_energy_uj = vec![1000.0, 1000.0, 1000.0, 1000.0];
    // Expected energy: (1.0 * 1000 + 1.0 * 1000 + 0.2 * 1000 + 0.2 * 1000) uJ = 2400 uJ = 2.4 mJ
    let energy_mj = ReaEngine::compute_expected_energy_mj(&reach, &layer_energy_uj);
    assert!((energy_mj - 2.4).abs() < 1e-3);
}

#[test]
fn test_ewha_solution_generation() {
    let reach = LayerReachProfile::new_two_path(16, 4, 80.0);
    let constraints = EwhaConstraints::default();

    let solution = ReaEngine::solve_ewha("test-llama-16l", reach, &constraints, 2048);

    assert_eq!(solution.model_name, "test-llama-16l");
    assert_eq!(solution.num_layers, 16);
    assert_eq!(solution.layer_allocations.len(), 16);
    assert!(solution.total_static_storage_mb > 0.0);
    assert!(solution.expected_traffic_mb_per_token > 0.0);
    assert!(solution.expected_energy_mj_per_token > 0.0);
    assert!(solution.expected_latency_ms > 0.0);
    assert!(solution.quality_penalty_total >= 0.0);

    // Expected depth fraction for 16 layers with checkpoint at layer 4 (5 prefix layers: 0..=4):
    // (5 * 1.0 + 11 * 0.2) / 16 = (5 + 2.2) / 16 = 7.2 / 16 = 0.45
    assert!((solution.expected_depth_fraction - 0.45).abs() < 1e-2);
}
