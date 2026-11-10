use thme::gui::TensorInfo;
use thme::transform::{AutoDecider, MatrixAlgorithm, MatrixMath};

#[test]
fn test_svd_power_iteration_reconstruction() {
    // 4x4 low-rank synthetic matrix (rank-1 outer product u * v^T)
    let u = [1.0, 2.0, -1.0, 0.5];
    let v = [0.5, -1.0, 2.0, 1.5];
    let mut matrix = vec![0.0; 16];
    for r in 0..4 {
        for c in 0..4 {
            matrix[r * 4 + c] = u[r] * v[c];
        }
    }

    let (sigma, u_est, v_est) = MatrixMath::power_iteration_svd(&matrix, 4, 4, 30);
    assert!(sigma > 0.0);
    assert_eq!(u_est.len(), 4);
    assert_eq!(v_est.len(), 4);

    // Verify reconstruction error is low
    let mut recon_err = 0.0;
    for r in 0..4 {
        for c in 0..4 {
            let recon = sigma * u_est[r] * v_est[c];
            recon_err += (matrix[r * 4 + c] - recon).abs();
        }
    }
    let mean_err = recon_err / 16.0;
    assert!(mean_err < 0.1, "Mean reconstruction error was {}", mean_err);
}

#[test]
fn test_hadamard_orthogonality_and_energy() {
    let mut v = vec![0.5, -1.2, 3.4, -0.8, 2.1, -0.3, 1.5, -2.5];
    let orig_energy: f32 = v.iter().map(|x| x * x).sum();

    MatrixMath::fast_walsh_hadamard_transform(&mut v);
    let trans_energy: f32 = v.iter().map(|x| x * x).sum();

    // Energy must be conserved under orthonormal WHT
    assert!((orig_energy - trans_energy).abs() < 1e-3);
}

#[test]
fn test_smoothquant_scaling_bounds() {
    let weights = vec![
        0.01, 1.50, -0.02, 0.04, 0.03, -2.10, 0.01, -0.05, -0.02, 1.80, 0.05, 0.02,
    ];
    let scales = MatrixMath::compute_smoothquant_scales(&weights, 4, 0.5);
    assert_eq!(scales.len(), 4);
    // Channel 1 has high outliers (1.5, -2.1, 1.8) -> scale should reflect it
    assert!(scales[1] > 0.1 && scales[1] < 10.0);
}

#[test]
fn test_distillation_similarity_bounds() {
    let (cosine_sim, kl) = MatrixMath::simulate_distillation_alignment(4096, 2048, 2.0);
    assert!((0.85..=1.0).contains(&cosine_sim));
    assert!(kl > 0.0 && kl < 1.0);
}

#[test]
fn test_auto_decider_layer_specialization() {
    let default_wp = thme::ir::WeightProfile::default();
    let tensors = vec![
        TensorInfo {
            name: "model.layers.0.self_attn.q_proj.weight".to_string(),
            shape: vec![2048, 2048],
            dtype: "F16".to_string(),
            num_elements: 2048 * 2048,
            min_val: -0.15,
            max_val: 0.15,
            mean_val: 0.0,
            std_val: 0.025,
            sparsity: 0.02,
            outlier_ratio: 0.002,
            histogram: default_wp.histogram,
            sample_weights: vec![0.05; 32],
            layer_index: Some(0),
            role: "AttentionQ".to_string(),
        },
        TensorInfo {
            name: "model.layers.6.mlp.gate_proj.weight".to_string(),
            shape: vec![8192, 2048],
            dtype: "F16".to_string(),
            num_elements: 8192 * 2048,
            min_val: -0.22,
            max_val: 0.21,
            mean_val: 0.0,
            std_val: 0.03,
            sparsity: 0.05,
            outlier_ratio: 0.0015,
            histogram: default_wp.histogram,
            sample_weights: vec![0.02; 32],
            layer_index: Some(6),
            role: "MlpGate".to_string(),
        },
        TensorInfo {
            name: "model.layers.6.mlp.down_proj.weight".to_string(),
            shape: vec![2048, 8192],
            dtype: "F16".to_string(),
            num_elements: 2048 * 8192,
            min_val: -0.45,
            max_val: 0.42,
            mean_val: 0.0,
            std_val: 0.04,
            sparsity: 0.03,
            outlier_ratio: 0.006, // Heavy outlier
            histogram: default_wp.histogram,
            sample_weights: vec![0.08; 32],
            layer_index: Some(6),
            role: "MlpDown".to_string(),
        },
        TensorInfo {
            name: "model.layers.6.input_layernorm.weight".to_string(),
            shape: vec![2048],
            dtype: "F16".to_string(),
            num_elements: 2048,
            min_val: 0.9,
            max_val: 1.1,
            mean_val: 1.0,
            std_val: 0.04,
            sparsity: 0.0,
            outlier_ratio: 0.0001,
            histogram: default_wp.histogram,
            sample_weights: vec![1.0; 8],
            layer_index: Some(6),
            role: "LayerNorm".to_string(),
        },
    ];

    let (decisions, summary) = AutoDecider::build_model_plan(&tensors, 12, 0.025);
    assert_eq!(decisions.len(), 4);
    assert!(summary.overall_compression_ratio > 1.5);
    assert!(summary.optimized_storage_mb < summary.original_storage_mb);

    // LayerNorm must remain dense FP16
    assert_eq!(decisions[3].algorithm, MatrixAlgorithm::DenseBaselineFP16);

    // Layer 0 AttentionQ boundary must be protected with Truncated SVD
    assert!(matches!(
        decisions[0].algorithm,
        MatrixAlgorithm::TruncatedSVD { .. }
    ));

    // Layer 6 MlpDown with high outliers (0.006) should use Walsh-Hadamard rotation or SmoothQuant
    assert!(
        matches!(
            decisions[2].algorithm,
            MatrixAlgorithm::HadamardRotation { .. }
        ) || matches!(decisions[2].algorithm, MatrixAlgorithm::SmoothQuant { .. })
    );
}
