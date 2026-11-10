use thme::allocator::ParetoAllocator;
use thme::ir::{
    extract_layer_index, ModelArchitecture, ModelIR, TensorFormat, TensorIR, TensorRole,
};
use thme::profiling::{KvCacheEstimator, SensitivityProfiler};
use thme::quality::{BenchmarkScore, GateDecision, QualityGate};

#[test]
fn test_tensor_role_detection() {
    assert_eq!(
        TensorRole::detect("model.layers.0.self_attn.q_proj.weight"),
        TensorRole::AttentionQ
    );
    assert_eq!(
        TensorRole::detect("model.layers.0.self_attn.k_proj.weight"),
        TensorRole::AttentionK
    );
    assert_eq!(
        TensorRole::detect("model.layers.0.mlp.down_proj.weight"),
        TensorRole::MlpDown
    );
    assert_eq!(
        TensorRole::detect("model.embed_tokens.weight"),
        TensorRole::Embedding
    );
    assert_eq!(
        TensorRole::detect("model.norm.weight"),
        TensorRole::FinalNorm
    );
}

#[test]
fn test_layer_index_extraction() {
    assert_eq!(
        extract_layer_index("model.layers.15.self_attn.q_proj.weight"),
        Some(15)
    );
    assert_eq!(
        extract_layer_index("transformer.h.3.mlp.c_proj.weight"),
        Some(3)
    );
    assert_eq!(extract_layer_index("model.embed_tokens.weight"), None);
}

#[test]
fn test_kv_cache_formula() {
    // 32 layers, batch size 1, context 2048, 8 KV heads, head dim 128, FP16 (2 bytes)
    // Elements = 2 * 32 * 1 * 2048 * 8 * 128 = 134,217,728 elements
    // In bytes = 134,217,728 * 2 = 268,435,456 bytes (256 MB)
    let bytes = KvCacheEstimator::calculate_bytes(32, 1, 2048, 8, 128, 2.0);
    assert_eq!(bytes, 268_435_456);

    let mb = bytes as f64 / (1024.0 * 1024.0);
    assert_eq!(mb, 256.0);
}

#[test]
fn test_quality_gate() {
    // Within tolerance (higher is better)
    let res_pass = QualityGate::evaluate_candidate(85.0, 84.5, 1.0);
    assert_eq!(res_pass.decision, GateDecision::Accept);

    // Exceeds tolerance (higher is better)
    let res_fail = QualityGate::evaluate_candidate(85.0, 82.0, 1.0);
    assert_eq!(res_fail.decision, GateDecision::Reject);

    // Lower is better (Perplexity / loss: baseline 6.5, candidate 6.8 with 0.5 allowed degradation)
    let res_ppl_pass = QualityGate::evaluate_candidate_directional(6.5, 6.8, 0.5, true);
    assert_eq!(res_ppl_pass.decision, GateDecision::Accept);

    // Lower is better failure (baseline 6.5, candidate 7.5 with 0.5 allowed degradation)
    let res_ppl_fail = QualityGate::evaluate_candidate_directional(6.5, 7.5, 0.5, true);
    assert_eq!(res_ppl_fail.decision, GateDecision::Reject);

    // Multi-task suite with mixed directions
    let suite = vec![
        BenchmarkScore {
            task_name: "mmlu".to_string(),
            baseline_score: 80.0,
            candidate_score: 79.5,
            allowed_degradation: 1.0,
            lower_is_better: false,
        },
        BenchmarkScore {
            task_name: "perplexity".to_string(),
            baseline_score: 5.2,
            candidate_score: 5.3,
            allowed_degradation: 0.2,
            lower_is_better: true,
        },
    ];
    let suite_res = QualityGate::evaluate_suite(&suite);
    assert_eq!(suite_res.decision, GateDecision::Accept);
}

#[test]
fn test_model_ir_and_pareto_planner() {
    let mut tensors = Vec::new();
    // Simulate a miniature 2-layer model
    tensors.push(TensorIR::new(
        "model.embed_tokens.weight".to_string(),
        vec![32000, 2048],
        "F16".to_string(),
        131_072_000,
    ));

    for l in 0..2 {
        tensors.push(TensorIR::new(
            format!("model.layers.{}.self_attn.q_proj.weight", l),
            vec![2048, 2048],
            "F16".to_string(),
            8_388_608,
        ));
        tensors.push(TensorIR::new(
            format!("model.layers.{}.mlp.down_proj.weight", l),
            vec![2048, 8192],
            "F16".to_string(),
            33_554_432,
        ));
    }

    tensors.push(TensorIR::new(
        "model.norm.weight".to_string(),
        vec![2048],
        "F16".to_string(),
        4_096,
    ));

    let mut model = ModelIR::new(
        "test-mini-llama".to_string(),
        ModelArchitecture::Llama,
        2,
        2048,
        16,
        16,
        32000,
        2048,
        tensors,
    );

    // Profile sensitivity
    SensitivityProfiler::profile_model(&mut model);
    assert!(model.tensors[0].sensitivity_score > 0.1);

    // Plan Pareto allocation
    let plan = ParetoAllocator::plan_allocation(&model, None, 1.5);
    assert!(plan.planned_size_gb < plan.original_size_gb);
    assert!(plan.compression_ratio > 1.0);
    assert_eq!(
        plan.tensor_allocations.get("model.norm.weight"),
        Some(&TensorFormat::FP16)
    );

    // Test zero target budget edge case (should not panic or produce NaN)
    let zero_budget_plan = ParetoAllocator::plan_allocation(&model, Some(0.0), 2.0);
    assert!(zero_budget_plan.compression_ratio >= 1.0);
}
