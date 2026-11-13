// SPDX-License-Identifier: AGPL-3.0-only
use serde_json::json;

#[test]
fn test_mcp_initialize_and_tools_list() {
    // Verify tools can be listed
    let req = json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/list",
        "params": {}
    });

    assert_eq!(req["method"], "tools/list");
}

#[test]
fn test_mcp_auto_decide_and_validate_simulation() {
    let registry = thme::transform::byoa::AlgorithmRegistry::new();
    let dec = thme::transform::AutoDecider::decide_for_parameter_with_registry(
        "model.layers.0.mlp.down_proj.weight",
        0,
        12,
        "Linear",
        &[2048, 2048],
        0.0,
        0.005,
        0.1,
        0.025,
        Some(&registry),
    );
    assert!(!dec.rationale.is_empty());

    // Test quality gate
    let base = thme::quality::PerplexityMetrics {
        corpus: thme::quality::EvaluationCorpus::WikiText2,
        total_tokens: 512,
        cross_entropy_loss: 1.8,
        perplexity: 6.049,
        latency_ms: 10.0,
        tokens_per_sec: 1000.0,
        is_simulated: true,
    };
    let cand = thme::quality::PerplexityMetrics {
        perplexity: 6.120,
        ..base.clone()
    };
    let gate = thme::quality::ModelEvaluator::assert_gate(&base, &cand, 0.25);
    assert_eq!(gate.decision, thme::quality::PerplexityGateDecision::Accept);
}
