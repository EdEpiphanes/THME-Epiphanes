// SPDX-License-Identifier: AGPL-3.0-only
//! Integration Tests for Bring Your Own Algorithm (BYOA) Engine

use thme::ir::TensorRole;
use thme::transform::byoa::{
    AlgorithmCategory, AlgorithmDefinition, AlgorithmRegistry, ScaleMode, TransformationRule,
};
use thme::transform::{AutoDecider, MatrixAlgorithm};

#[test]
fn test_algorithm_registry_builtins() {
    let registry = AlgorithmRegistry::new();
    let algs = registry.list();

    assert!(algs.len() >= 5);
    assert!(registry.get("thme.bitnet_b158").is_some());
    assert!(registry.get("thme.sparse_2to4").is_some());
    assert!(registry.get("thme.hadamard_wht").is_some());
    assert!(registry.get("thme.smoothquant").is_some());
    assert!(registry.get("thme.gated_deltanet").is_some());
}

#[test]
fn test_custom_algorithm_registration_and_json_roundtrip() {
    let mut registry = AlgorithmRegistry::new();

    let custom_alg = AlgorithmDefinition {
        id: "user.quaternary_quant".to_string(),
        name: "Quaternary 2-Bit Quantizer".to_string(),
        author: "Lab Research Team".to_string(),
        version: "0.2.1".to_string(),
        category: AlgorithmCategory::Quantization,
        description:
            "Discretizes weights into 4 discrete levels [-2, -1, 1, 2] with absmean scaling."
                .to_string(),
        effective_bits_per_param: 2.0,
        compression_ratio: 8.0,
        hardware_speedup_factor: 2.4,
        target_roles: vec![TensorRole::MlpGate, TensorRole::MlpUp],
        rule: TransformationRule::QuantizeCodebook {
            levels: vec![-2.0, -1.0, 1.0, 2.0],
            scale_mode: ScaleMode::AbsMean,
        },
    };

    registry.register(custom_alg);
    assert!(registry.get("user.quaternary_quant").is_some());

    // Test JSON export
    let json_export = registry
        .export_to_json("user.quaternary_quant")
        .expect("Failed to export JSON");
    assert!(json_export.contains("Quaternary 2-Bit Quantizer"));

    // Remove and re-import from JSON
    registry.remove("user.quaternary_quant");
    assert!(registry.get("user.quaternary_quant").is_none());

    let imported_id = registry
        .import_from_json(&json_export)
        .expect("Failed to import JSON");
    assert_eq!(imported_id, "user.quaternary_quant");
    assert!(registry.get("user.quaternary_quant").is_some());
}

#[test]
fn test_custom_algorithm_weight_transformation() {
    let custom_ternary = AlgorithmDefinition {
        id: "user.custom_ternary".to_string(),
        name: "Custom Ternary".to_string(),
        author: "Test".to_string(),
        version: "1.0".to_string(),
        category: AlgorithmCategory::Quantization,
        description: "Ternary test".to_string(),
        effective_bits_per_param: 1.585,
        compression_ratio: 10.1,
        hardware_speedup_factor: 2.8,
        target_roles: vec![TensorRole::MlpGate],
        rule: TransformationRule::AbsmeanTernary {
            scale_gamma: 1.0,
            zero_band: 0.5,
        },
    };

    let mut weights = vec![0.1, -0.8, 1.2, -0.2, 0.9, -1.5];
    let result = custom_ternary
        .apply_to_weights(&mut weights)
        .expect("Transformation failed");

    assert_eq!(result.original_elements, 6);
    assert!(result.mean_squared_error >= 0.0);
    assert!(result.sparsity_ratio >= 0.0);

    // Each element should be discretized to -1.0, 0.0, or 1.0
    for &w in &weights {
        assert!(
            w == -1.0 || w == 0.0 || w == 1.0,
            "Unexpected ternary value: {}",
            w
        );
    }
}

#[test]
fn test_autodecider_with_custom_registry() {
    let mut registry = AlgorithmRegistry::new();

    // Register a custom algorithm specifically targeting AttentionK with high compression
    registry.register(AlgorithmDefinition {
        id: "user.aggressive_k_proj".to_string(),
        name: "Aggressive K-Projection Compactor".to_string(),
        author: "Research Lead".to_string(),
        version: "1.0".to_string(),
        category: AlgorithmCategory::Quantization,
        description: "Specialized for Key projections".to_string(),
        effective_bits_per_param: 2.0,
        compression_ratio: 8.0,
        hardware_speedup_factor: 2.5,
        target_roles: vec![TensorRole::AttentionK],
        rule: TransformationRule::StructuredSparsity { m: 2, n: 4 },
    });

    let decision = AutoDecider::decide_for_parameter_with_registry(
        "model.layers.5.self_attn.k_proj.weight",
        5,
        32,
        "AttentionK",
        &[4096, 4096],
        0.0,
        0.002,
        0.03,
        0.03, // Quality tolerance >= 0.02
        Some(&registry),
    );

    match decision.algorithm {
        MatrixAlgorithm::Custom {
            id,
            name,
            effective_bits,
        } => {
            assert_eq!(id, "user.aggressive_k_proj");
            assert_eq!(name, "Aggressive K-Projection Compactor");
            assert_eq!(effective_bits, 2.0);
        }
        other => panic!("Expected Custom algorithm, got {:?}", other),
    }
}
