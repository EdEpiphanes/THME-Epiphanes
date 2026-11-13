// SPDX-License-Identifier: AGPL-3.0-only
//! Integration Tests for Open Weights Architecture Rebuilder

use thme::ir::model::{ModelArchitecture, ModelIR};
use thme::ir::rebuilder::{ArchitectureBlueprint, ModelRebuilder, RemapStrategy};
use thme::ir::tensor::{TensorFormat, TensorRole};
use thme::surgery::recipe::{SurgeryEngine, TweakOperation, TweakRecipe};

#[test]
fn test_build_scaffold_from_blueprint() {
    let blueprint = ArchitectureBlueprint::new_hybrid_llama("rebuilt-hybrid-8l", 8, 2048, 2);

    assert_eq!(blueprint.num_layers, 8);
    assert_eq!(blueprint.hidden_size, 2048);
    assert_eq!(blueprint.layers.len(), 8);

    let scaffold: ModelIR = ModelRebuilder::build_scaffold(&blueprint);

    assert_eq!(scaffold.name, "rebuilt-hybrid-8l");
    assert_eq!(scaffold.architecture, ModelArchitecture::Llama);
    assert_eq!(scaffold.num_layers, 8);
    assert_eq!(scaffold.hidden_size, 2048);
    assert!(scaffold.num_parameters > 0);
    assert!(scaffold.original_byte_size > 0);

    // Verify key tensors exist
    let has_embed = scaffold
        .tensors
        .iter()
        .any(|t| t.name == "model.embed_tokens.weight");
    let has_norm = scaffold
        .tensors
        .iter()
        .any(|t| t.name == "model.norm.weight");
    let has_head = scaffold.tensors.iter().any(|t| t.name == "lm_head.weight");
    let has_l0_q = scaffold
        .tensors
        .iter()
        .any(|t| t.name == "model.layers.0.self_attn.q_proj.weight");

    assert!(has_embed, "Embedding tensor missing");
    assert!(has_norm, "Norm tensor missing");
    assert!(has_head, "LM Head tensor missing");
    assert!(has_l0_q, "Layer 0 Q tensor missing");
}

#[test]
fn test_rebuild_from_open_weights_remapping() {
    // 1. Create a source open-weights model with customized formats
    let source_blueprint =
        ArchitectureBlueprint::new_hybrid_llama("source-open-weights", 4, 1024, 1);
    let mut source_model = ModelRebuilder::build_scaffold(&source_blueprint);

    // Apply Ternary format to Layer 2 Gate projection
    for t in &mut source_model.tensors {
        if t.name == "model.layers.2.mlp.gate_proj.weight" {
            t.selected_format = TensorFormat::Ternary;
        }
    }

    // 2. Rebuild target model with 6 layers
    let target_blueprint =
        ArchitectureBlueprint::new_hybrid_llama("rebuilt-student-6l", 6, 1024, 2);
    let rebuilt_model = ModelRebuilder::rebuild_from_open_weights(
        &target_blueprint,
        &source_model,
        RemapStrategy::ZeroPad,
    );

    assert_eq!(rebuilt_model.num_layers, 6);
    let l2_gate = rebuilt_model
        .tensors
        .iter()
        .find(|t| t.name == "model.layers.2.mlp.gate_proj.weight")
        .expect("Layer 2 Gate missing in rebuilt model");

    assert_eq!(l2_gate.selected_format, TensorFormat::Ternary);
}

#[test]
fn test_surgery_custom_algorithm_and_rebuild_operations() {
    let blueprint = ArchitectureBlueprint::new_hybrid_llama("surgery-test-model", 4, 1024, 1);
    let mut model = ModelRebuilder::build_scaffold(&blueprint);

    let recipe = TweakRecipe {
        name: "Test-BYOA-Recipe".to_string(),
        author: "Agent-X".to_string(),
        target_model: "surgery-test-model".to_string(),
        version: "1.0".to_string(),
        description: "Applies custom BYOA algorithm and records rebuild op".to_string(),
        operations: vec![
            TweakOperation::ApplyCustomAlgorithm {
                start_layer: 0,
                end_layer: 2,
                role: Some(TensorRole::AttentionQ),
                algorithm_id: "user.custom_q_rotator".to_string(),
            },
            TweakOperation::RebuildArchitecture {
                num_layers: 4,
                hidden_size: 1024,
                default_mixer: "gated_deltanet".to_string(),
            },
        ],
        expected_speedup: Some(2.0),
        expected_memory_savings_pct: Some(50.0),
    };

    let diff = SurgeryEngine::apply_recipe(&mut model, &recipe);

    assert_eq!(diff.recipe_name, "Test-BYOA-Recipe");
    assert!(diff.modified_tensors_count > 0);
    assert!(diff
        .audit_log
        .iter()
        .any(|l| l.contains("user.custom_q_rotator")));
    assert!(diff
        .audit_log
        .iter()
        .any(|l| l.contains("Rebuilt model topology")));

    // Check that target tensor has the custom format assigned
    let l0_q = model
        .tensors
        .iter()
        .find(|t| t.name == "model.layers.0.self_attn.q_proj.weight")
        .unwrap();

    assert_eq!(
        l0_q.selected_format,
        TensorFormat::Custom("user.custom_q_rotator".to_string())
    );
}
