// SPDX-License-Identifier: AGPL-3.0-only
//! Integration tests for Layer Deconstruction & Topological Component Disassembly Engine

use thme::ir::layer_decomp::{
    DeconstructedTensor, LayerDeconstruction, LayerTopologyStyle, SubBlockCategory,
};
use thme::ir::{ArchitectureBlueprint, ModelRebuilder, TensorRole};

#[test]
fn test_layer_deconstruction_structure() {
    let blueprint = ArchitectureBlueprint::new_hybrid_llama("test-deconstruct-model", 4, 1024, 1);
    let model_ir = ModelRebuilder::build_scaffold(&blueprint);

    // Extract tensors for Layer 0
    let l0_tensors: Vec<DeconstructedTensor> = model_ir
        .tensors
        .iter()
        .filter(|t| t.layer_index == Some(0))
        .map(|t| DeconstructedTensor {
            name: t.name.clone(),
            role: t.role,
            shape: t.shape.clone(),
            dtype: t.original_dtype.clone(),
            num_elements: t.numel,
            byte_size: t.byte_size,
            is_active: true,
            selected_format: "FP16".to_string(),
            effective_bits: 16.0,
            routing_destination: "Forward Propagation".to_string(),
            min_val: -0.1,
            max_val: 0.1,
            mean_val: 0.0,
            std_val: 0.02,
            sparsity: 0.0,
            outlier_ratio: 0.0,
        })
        .collect();

    assert!(!l0_tensors.is_empty(), "Layer 0 should contain tensors");

    // Deconstruct Layer 0
    let decomp =
        LayerDeconstruction::analyze(0, "gqa_attention", 1.0, false, 0.0, 1.0, 0.0, &l0_tensors);

    assert_eq!(decomp.layer_index, 0);
    assert_eq!(
        decomp.topology_style,
        LayerTopologyStyle::StandardMonolithic
    );
    assert!(decomp.total_parameters > 0);
    assert!(decomp.total_bytes > 0);
    assert_eq!(decomp.reach_probability, 1.0);
    assert!(!decomp.is_exit_checkpoint);

    // Verify sub-blocks
    assert!(decomp.input_norm.is_some(), "Input norm sub-block missing");
    let norm = decomp.input_norm.as_ref().unwrap();
    assert_eq!(norm.category, SubBlockCategory::InputNorm);
    assert!(!norm.tensors.is_empty());

    assert_eq!(decomp.mixer_block.category, SubBlockCategory::SequenceMixer);
    assert!(!decomp.mixer_block.tensors.is_empty());
    // Attention should contain Q, K, V, O
    let has_q = decomp
        .mixer_block
        .tensors
        .iter()
        .any(|t| t.role == TensorRole::AttentionQ);
    let has_k = decomp
        .mixer_block
        .tensors
        .iter()
        .any(|t| t.role == TensorRole::AttentionK);
    let has_v = decomp
        .mixer_block
        .tensors
        .iter()
        .any(|t| t.role == TensorRole::AttentionV);
    let has_o = decomp
        .mixer_block
        .tensors
        .iter()
        .any(|t| t.role == TensorRole::AttentionOut);
    assert!(
        has_q && has_k && has_v && has_o,
        "Mixer missing Q, K, V, or Out projections"
    );

    assert!(
        decomp.ffn_block.is_some(),
        "Feed-forward MLP sub-block missing"
    );
    let ffn = decomp.ffn_block.as_ref().unwrap();
    assert_eq!(ffn.category, SubBlockCategory::FeedForwardMlp);
    let has_gate = ffn.tensors.iter().any(|t| t.role == TensorRole::MlpGate);
    let has_up = ffn.tensors.iter().any(|t| t.role == TensorRole::MlpUp);
    let has_down = ffn.tensors.iter().any(|t| t.role == TensorRole::MlpDown);
    assert!(
        has_gate && has_up && has_down,
        "MLP missing Gate, Up, or Down projections"
    );
}

#[test]
fn test_layer_deconstruction_hybrid_and_checkpoint() {
    let blueprint = ArchitectureBlueprint::new_hybrid_llama("test-hybrid-cp-model", 4, 1024, 2);
    let model_ir = ModelRebuilder::build_scaffold(&blueprint);

    // Deconstruct Layer 2 (Checkpoint layer)
    let l2_tensors: Vec<DeconstructedTensor> = model_ir
        .tensors
        .iter()
        .filter(|t| t.layer_index == Some(2))
        .map(|t| DeconstructedTensor {
            name: t.name.clone(),
            role: t.role,
            shape: t.shape.clone(),
            dtype: t.original_dtype.clone(),
            num_elements: t.numel,
            byte_size: t.byte_size,
            is_active: true,
            selected_format: "BitNet b1.58".to_string(),
            effective_bits: 2.0,
            routing_destination: "Forward Propagation".to_string(),
            min_val: -0.1,
            max_val: 0.1,
            mean_val: 0.0,
            std_val: 0.02,
            sparsity: 0.0,
            outlier_ratio: 0.0,
        })
        .collect();

    let decomp_tribrid = LayerDeconstruction::analyze(
        2,
        "gated_deltanet",
        0.80,
        true,
        0.20,
        0.50,
        0.30,
        &l2_tensors,
    );

    assert_eq!(decomp_tribrid.layer_index, 2);
    assert_eq!(
        decomp_tribrid.topology_style,
        LayerTopologyStyle::TribridTriMixer
    );
    assert!(decomp_tribrid.is_exit_checkpoint);
    assert!(decomp_tribrid
        .routing_block
        .description
        .contains("Early-Exit Checkpoint"));
}
