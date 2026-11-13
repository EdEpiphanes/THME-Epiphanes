// SPDX-License-Identifier: AGPL-3.0-only
use thme::ir::{ModelArchitecture, ModelIR, TensorFormat, TensorIR};
use thme::speculative::{
    DraftExtractionConfig, DraftExtractionStrategy, DraftSynthesizer, SpeculativeEngine,
    SpeculativeExecutionConfig, SpeculativeTreeTopology, SpeculativeVerificationStrategy,
};
use thme::surgery::{SurgeryEngine, TweakOperation, TweakRecipe};

fn create_mock_target_model(num_layers: usize, hidden_size: usize) -> ModelIR {
    let mut tensors = Vec::new();
    let intermediate = (hidden_size * 8) / 3;

    tensors.push(TensorIR::new(
        "model.embed_tokens.weight".to_string(),
        vec![32000, hidden_size],
        "F16".to_string(),
        32000 * hidden_size * 2,
    ));

    for i in 0..num_layers {
        tensors.push(TensorIR::new(
            format!("layers.{}.self_attn.q_proj.weight", i),
            vec![hidden_size, hidden_size],
            "F16".to_string(),
            hidden_size * hidden_size * 2,
        ));
        tensors.push(TensorIR::new(
            format!("layers.{}.self_attn.k_proj.weight", i),
            vec![hidden_size, hidden_size],
            "F16".to_string(),
            hidden_size * hidden_size * 2,
        ));
        tensors.push(TensorIR::new(
            format!("layers.{}.self_attn.v_proj.weight", i),
            vec![hidden_size, hidden_size],
            "F16".to_string(),
            hidden_size * hidden_size * 2,
        ));
        tensors.push(TensorIR::new(
            format!("layers.{}.self_attn.o_proj.weight", i),
            vec![hidden_size, hidden_size],
            "F16".to_string(),
            hidden_size * hidden_size * 2,
        ));
        tensors.push(TensorIR::new(
            format!("layers.{}.mlp.gate_proj.weight", i),
            vec![intermediate, hidden_size],
            "F16".to_string(),
            intermediate * hidden_size * 2,
        ));
        tensors.push(TensorIR::new(
            format!("layers.{}.mlp.up_proj.weight", i),
            vec![intermediate, hidden_size],
            "F16".to_string(),
            intermediate * hidden_size * 2,
        ));
        tensors.push(TensorIR::new(
            format!("layers.{}.mlp.down_proj.weight", i),
            vec![hidden_size, intermediate],
            "F16".to_string(),
            hidden_size * intermediate * 2,
        ));
    }

    tensors.push(TensorIR::new(
        "lm_head.weight".to_string(),
        vec![32000, hidden_size],
        "F16".to_string(),
        32000 * hidden_size * 2,
    ));

    ModelIR::new(
        "target-llama-16l".to_string(),
        ModelArchitecture::Llama,
        num_layers,
        hidden_size,
        32,
        8,
        32000,
        4096,
        tensors,
    )
}

#[test]
fn test_draft_model_prefix_extraction() {
    let target = create_mock_target_model(16, 2048);
    let cfg = DraftExtractionConfig {
        strategy: DraftExtractionStrategy::PrefixLayers,
        num_layers: 4,
        target_hidden_size: None,
        apply_bitnet: false,
        stride_step: 4,
    };

    let (draft, report) = DraftSynthesizer::extract_draft_model(&target, &cfg).unwrap();

    assert_eq!(draft.num_layers, 4);
    assert_eq!(draft.hidden_size, 2048);
    assert!(draft.num_parameters < target.num_parameters);
    assert!(report.parameter_reduction_pct > 60.0);
    assert!(report.latency_cost_ratio < 0.35);

    // Ensure all 4 layers are present and indexed 0..4
    for l in 0..4 {
        let has_q = draft
            .tensors
            .iter()
            .any(|t| t.name == format!("layers.{}.self_attn.q_proj.weight", l));
        assert!(has_q, "Layer {} missing in draft", l);
    }
}

#[test]
fn test_draft_model_ternary_bitnet() {
    let target = create_mock_target_model(12, 2048);
    let cfg = DraftExtractionConfig {
        strategy: DraftExtractionStrategy::TernaryBitNet,
        num_layers: 3,
        target_hidden_size: None,
        apply_bitnet: true,
        stride_step: 4,
    };

    let (draft, report) = DraftSynthesizer::extract_draft_model(&target, &cfg).unwrap();

    assert_eq!(draft.num_layers, 3);
    assert!(report.memory_savings_pct > 70.0);

    let ternary_count = draft
        .tensors
        .iter()
        .filter(|t| matches!(t.selected_format, TensorFormat::Ternary))
        .count();
    assert!(ternary_count > 0);
}

#[test]
fn test_speculative_execution_speedup() {
    let cfg = SpeculativeExecutionConfig {
        lookahead_horizon: 5,
        temperature: 0.7,
        strategy: SpeculativeVerificationStrategy::RejectionSampling,
        base_draft_fidelity: 0.85,
        latency_cost_ratio: 0.12,
    };

    let telemetry = SpeculativeEngine::simulate_speculative_run(&cfg, 20, 10.0);

    assert_eq!(telemetry.lookahead_horizon, 5);
    assert_eq!(telemetry.total_steps_simulated, 20);
    assert!(telemetry.mean_acceptance_rate > 0.20);
    assert!(telemetry.mean_tokens_per_pass > 2.0);
    assert!(telemetry.effective_speedup_multiplier > 1.35);
    assert!(telemetry.memory_traffic_savings_pct > 30.0);

    // Verify optimal horizon search
    let (best_h, best_s) = SpeculativeEngine::find_optimal_horizon(0.85, 0.12);
    assert!(best_h >= 2);
    assert!(best_s > 1.5);
}

#[test]
fn test_tree_speculative_topology() {
    // 3 at depth 1, 2 children each at depth 2 = 1 + 3 + 6 = 10 nodes
    let tree = SpeculativeTreeTopology::build_canonical(&[3, 2]);

    assert_eq!(tree.max_depth, 2);
    assert_eq!(tree.total_nodes, 10);
    assert_eq!(tree.num_paths, 6);

    // Test causal tree attention mask:
    // Root attends to root
    assert!(tree.can_attend(0, 0));
    // Child 1 (node 1) attends to root (node 0) and itself
    assert!(tree.can_attend(1, 0));
    assert!(tree.can_attend(1, 1));
    // Child 1 cannot attend to sibling Child 2 (node 2)
    assert!(!tree.can_attend(1, 2));

    // Evaluate tree acceptance
    let (best_path, accepted, rate) = tree.evaluate_tree_acceptance(0.90);
    assert!(!best_path.is_empty());
    assert!(accepted <= 2);
    assert!(rate >= 0.0);
}

#[test]
fn test_declarative_recipe_extract_draft() {
    let mut model = create_mock_target_model(8, 2048);

    let recipe = TweakRecipe {
        name: "Extract-Draft-Recipe".to_string(),
        author: "SpeculativeSynthesizer".to_string(),
        target_model: "target-llama-16l".to_string(),
        version: "1.0".to_string(),
        description: "Extract 2-layer draft model".to_string(),
        operations: vec![TweakOperation::ExtractDraftModel {
            num_layers: 2,
            target_hidden_size: None,
            apply_bitnet: Some(true),
        }],
        expected_speedup: Some(2.2),
        expected_memory_savings_pct: Some(75.0),
    };

    let diff = SurgeryEngine::apply_recipe(&mut model, &recipe);

    assert_eq!(model.num_layers, 2);
    assert!(diff.modified_tensors_count > 0);
    assert!(diff.savings_percent > 50.0);
}
