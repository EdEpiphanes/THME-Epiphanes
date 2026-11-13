// SPDX-License-Identifier: AGPL-3.0-only
use thme::ir::{
    ModelArchitecture, ModelIR, ParadigmConversionConfig, ParadigmConverter,
    SpecializedModelParadigm, TensorFormat, TensorIR,
};
use thme::surgery::{SurgeryEngine, TweakOperation, TweakRecipe};

fn create_mock_llm(num_layers: usize, hidden_size: usize) -> ModelIR {
    let mut tensors = Vec::new();

    // Embeddings
    tensors.push(TensorIR::new(
        "model.embed_tokens.weight".to_string(),
        vec![32000, hidden_size],
        "F16".to_string(),
        32000 * hidden_size * 2,
    ));

    // Transformer layers
    for l in 0..num_layers {
        // Attention Q, K, V, O
        tensors.push(TensorIR::new(
            format!("layers.{}.self_attn.q_proj.weight", l),
            vec![hidden_size, hidden_size],
            "F16".to_string(),
            hidden_size * hidden_size * 2,
        ));
        tensors.push(TensorIR::new(
            format!("layers.{}.self_attn.k_proj.weight", l),
            vec![hidden_size, hidden_size],
            "F16".to_string(),
            hidden_size * hidden_size * 2,
        ));
        tensors.push(TensorIR::new(
            format!("layers.{}.self_attn.v_proj.weight", l),
            vec![hidden_size, hidden_size],
            "F16".to_string(),
            hidden_size * hidden_size * 2,
        ));
        tensors.push(TensorIR::new(
            format!("layers.{}.self_attn.o_proj.weight", l),
            vec![hidden_size, hidden_size],
            "F16".to_string(),
            hidden_size * hidden_size * 2,
        ));

        // MLP Gate, Up, Down
        tensors.push(TensorIR::new(
            format!("layers.{}.mlp.gate_proj.weight", l),
            vec![hidden_size * 2, hidden_size],
            "F16".to_string(),
            hidden_size * 2 * hidden_size * 2,
        ));
        tensors.push(TensorIR::new(
            format!("layers.{}.mlp.up_proj.weight", l),
            vec![hidden_size * 2, hidden_size],
            "F16".to_string(),
            hidden_size * 2 * hidden_size * 2,
        ));
        tensors.push(TensorIR::new(
            format!("layers.{}.mlp.down_proj.weight", l),
            vec![hidden_size, hidden_size * 2],
            "F16".to_string(),
            hidden_size * 2 * hidden_size * 2,
        ));
    }

    // LM Head
    tensors.push(TensorIR::new(
        "lm_head.weight".to_string(),
        vec![32000, hidden_size],
        "F16".to_string(),
        32000 * hidden_size * 2,
    ));

    ModelIR::new(
        "Test-LLM-8B".to_string(),
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
fn test_paradigm_detection() {
    let llm = create_mock_llm(4, 4096);
    assert_eq!(
        ParadigmConverter::detect_paradigm(&llm),
        SpecializedModelParadigm::LLM
    );

    // Convert to MoE
    let (moe, _) = ParadigmConverter::convert(
        &llm,
        &ParadigmConversionConfig {
            target_paradigm: SpecializedModelParadigm::MoE,
            num_experts: 4,
            top_k: 2,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(
        ParadigmConverter::detect_paradigm(&moe),
        SpecializedModelParadigm::MoE
    );

    // Convert to VLM
    let (vlm, _) = ParadigmConverter::convert(
        &llm,
        &ParadigmConversionConfig {
            target_paradigm: SpecializedModelParadigm::VLM,
            vision_hidden_dim: 1024,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(
        ParadigmConverter::detect_paradigm(&vlm),
        SpecializedModelParadigm::VLM
    );

    // Convert to LAM
    let (lam, _) = ParadigmConverter::convert(
        &llm,
        &ParadigmConversionConfig {
            target_paradigm: SpecializedModelParadigm::LAM,
            num_action_outputs: 16,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(
        ParadigmConverter::detect_paradigm(&lam),
        SpecializedModelParadigm::LAM
    );

    // Convert to LCM
    let (lcm, _) = ParadigmConverter::convert(
        &llm,
        &ParadigmConversionConfig {
            target_paradigm: SpecializedModelParadigm::LCM,
            concept_dim: 512,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(
        ParadigmConverter::detect_paradigm(&lcm),
        SpecializedModelParadigm::LCM
    );

    // Convert to SAM
    let (sam, _) = ParadigmConverter::convert(
        &llm,
        &ParadigmConversionConfig {
            target_paradigm: SpecializedModelParadigm::SAM,
            vision_hidden_dim: 768,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(
        ParadigmConverter::detect_paradigm(&sam),
        SpecializedModelParadigm::SAM
    );

    // Convert to MLM
    let (mlm, _) = ParadigmConverter::convert(
        &llm,
        &ParadigmConversionConfig {
            target_paradigm: SpecializedModelParadigm::MLM,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(
        ParadigmConverter::detect_paradigm(&mlm),
        SpecializedModelParadigm::MLM
    );
}

#[test]
fn test_conversion_llm_to_moe_and_back() {
    let llm = create_mock_llm(4, 2048);
    let original_params = llm.num_parameters;

    // Upcycle to MoE (4 experts)
    let (moe, report) = ParadigmConverter::convert(
        &llm,
        &ParadigmConversionConfig {
            target_paradigm: SpecializedModelParadigm::MoE,
            num_experts: 4,
            top_k: 2,
            ..Default::default()
        },
    )
    .unwrap();

    assert!(moe.num_parameters > original_params);
    assert_eq!(report.target_paradigm, SpecializedModelParadigm::MoE);

    // Check that routers exist
    let has_routers = moe.tensors.iter().any(|t| t.name.contains("router"));
    assert!(has_routers);

    // Check that experts exist
    let expert_count = moe
        .tensors
        .iter()
        .filter(|t| t.name.contains("experts.0.gate_proj"))
        .count();
    assert_eq!(expert_count, 4);

    // Convert back to standard LLM
    let (restored_llm, report_back) = ParadigmConverter::convert(
        &moe,
        &ParadigmConversionConfig {
            target_paradigm: SpecializedModelParadigm::LLM,
            ..Default::default()
        },
    )
    .unwrap();

    assert_eq!(
        ParadigmConverter::detect_paradigm(&restored_llm),
        SpecializedModelParadigm::LLM
    );
    assert!(restored_llm.num_parameters < moe.num_parameters);
    assert_eq!(report_back.target_paradigm, SpecializedModelParadigm::LLM);
}

#[test]
fn test_conversion_llm_to_slm_with_bitnet() {
    let llm = create_mock_llm(8, 4096);

    // Convert to SLM: 4 layers, hidden dim 1024, BitNet b1.58
    let (slm, report) = ParadigmConverter::convert(
        &llm,
        &ParadigmConversionConfig {
            target_paradigm: SpecializedModelParadigm::SLM,
            target_layers: Some(4),
            target_hidden_size: Some(1024),
            apply_bitnet: true,
            ..Default::default()
        },
    )
    .unwrap();

    assert_eq!(slm.num_layers, 4);
    assert_eq!(slm.hidden_size, 1024);
    assert!(slm.num_parameters < llm.num_parameters);
    assert!(report.storage_savings_pct > 50.0);

    // Verify BitNet ternary format applied to MLP
    let ternary_count = slm
        .tensors
        .iter()
        .filter(|t| matches!(t.selected_format, TensorFormat::Ternary))
        .count();
    assert!(ternary_count > 0);
}

#[test]
fn test_declarative_recipe_paradigm_conversion() {
    let mut model = create_mock_llm(4, 2048);

    let recipe = TweakRecipe {
        name: "Upcycle-To-MoE-Recipe".to_string(),
        author: "Teacher-LLM".to_string(),
        target_model: "Test-LLM".to_string(),
        version: "1.0".to_string(),
        description: "Sparse upcycling to 4 experts with top-2 routing".to_string(),
        operations: vec![TweakOperation::ConvertParadigm {
            target_paradigm: SpecializedModelParadigm::MoE,
            num_experts: Some(4),
            top_k: Some(2),
            target_hidden_size: None,
            target_layers: None,
            apply_bitnet: None,
        }],
        expected_speedup: None,
        expected_memory_savings_pct: None,
    };

    let diff = SurgeryEngine::apply_recipe(&mut model, &recipe);
    assert!(diff.modified_tensors_count > 0);
    assert_eq!(
        ParadigmConverter::detect_paradigm(&model),
        SpecializedModelParadigm::MoE
    );
}

#[test]
fn test_all_eight_paradigms_round_trip() {
    let base_llm = create_mock_llm(4, 2048);

    let all_paradigms = SpecializedModelParadigm::all();
    assert_eq!(all_paradigms.len(), 8);

    for target in all_paradigms {
        let mut cfg = ParadigmConversionConfig::new(target);
        cfg.num_experts = 4;
        cfg.top_k = 2;
        cfg.target_hidden_size = Some(1024);
        cfg.target_layers = Some(2);
        cfg.apply_bitnet = true;

        let (converted, report) = ParadigmConverter::convert(&base_llm, &cfg).unwrap();
        assert_eq!(report.target_paradigm, target);
        assert_eq!(ParadigmConverter::detect_paradigm(&converted), target);

        // Convert back to standard LLM
        let back_cfg = ParadigmConversionConfig::new(SpecializedModelParadigm::LLM);
        let (restored, back_report) = ParadigmConverter::convert(&converted, &back_cfg).unwrap();
        assert_eq!(back_report.target_paradigm, SpecializedModelParadigm::LLM);
        assert_eq!(
            ParadigmConverter::detect_paradigm(&restored),
            SpecializedModelParadigm::LLM
        );
    }
}
