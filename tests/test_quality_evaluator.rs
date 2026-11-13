// SPDX-License-Identifier: AGPL-3.0-only
use thme::quality::{
    EvaluationCorpus, EvaluatorConfig, ModelEvaluator, PerplexityGateDecision, PerplexityMetrics,
};

#[test]
fn test_quality_evaluator_corpora_perplexity() {
    let weights = vec![0.1, -0.2, 0.05, 0.3, -0.15, 0.08, -0.04, 0.22];
    let config = EvaluatorConfig::default();

    for corpus in [
        EvaluationCorpus::WikiText2,
        EvaluationCorpus::C4,
        EvaluationCorpus::PennTreebank,
        EvaluationCorpus::SyntheticCalibration,
    ] {
        let metrics = ModelEvaluator::evaluate_weights(&corpus, &weights, &config).unwrap();
        assert_eq!(metrics.corpus, corpus);
        assert!(metrics.perplexity >= 1.0);
        assert!(metrics.cross_entropy_loss > 0.0);
        assert!(metrics.tokens_per_sec > 0.0);
        assert!(metrics.is_simulated);
    }
}

#[test]
fn test_quality_evaluator_rollback_gate_strict() {
    let baseline = PerplexityMetrics {
        corpus: EvaluationCorpus::WikiText2,
        total_tokens: 2048,
        cross_entropy_loss: 1.85,
        perplexity: 6.3598,
        latency_ms: 12.5,
        tokens_per_sec: 16000.0,
        is_simulated: true,
    };

    // Candidate 1: Minor degradation within 0.25 tolerance -> Accept
    let candidate_acceptable = PerplexityMetrics {
        perplexity: 6.4500, // Delta = +0.0902
        ..baseline.clone()
    };
    let gate_accept = ModelEvaluator::assert_gate(&baseline, &candidate_acceptable, 0.25);
    assert_eq!(gate_accept.decision, PerplexityGateDecision::Accept);
    assert!(gate_accept.delta_ppl <= 0.25);
    assert!(gate_accept.summary.contains("PASSED"));

    // Candidate 2: Major degradation exceeding 0.25 tolerance -> RollbackRequired
    let candidate_bad = PerplexityMetrics {
        perplexity: 7.1000, // Delta = +0.7402
        ..baseline.clone()
    };
    let gate_rollback = ModelEvaluator::assert_gate(&baseline, &candidate_bad, 0.25);
    assert_eq!(
        gate_rollback.decision,
        PerplexityGateDecision::RollbackRequired
    );
    assert!(gate_rollback.delta_ppl > 0.25);
    assert!(gate_rollback.summary.contains("VIOLATION"));
    assert!(gate_rollback.recommended_action.contains("rollback"));
}

#[test]
fn test_quality_evaluator_empty_weights_error() {
    let weights: Vec<f32> = vec![];
    let config = EvaluatorConfig::default();
    let res = ModelEvaluator::evaluate_weights(&EvaluationCorpus::WikiText2, &weights, &config);
    assert!(res.is_err());
}
