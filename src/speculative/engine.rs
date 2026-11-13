// SPDX-License-Identifier: AGPL-3.0-only
use serde::{Deserialize, Serialize};

/// Verification sampling strategy for speculative candidate validation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SpeculativeVerificationStrategy {
    /// Greedy exact-match verification (arg max_x p(x) == arg max_x q(x))
    GreedyMatch,
    /// Standard speculative rejection sampling (u <= min(1, p(x)/q(x)))
    RejectionSampling,
    /// Entropy-adaptive thresholding based on sequence uncertainty
    EntropyAdaptive,
}

impl SpeculativeVerificationStrategy {
    pub fn display_name(&self) -> &'static str {
        match self {
            Self::GreedyMatch => "Greedy Exact Match",
            Self::RejectionSampling => "Speculative Rejection Sampling",
            Self::EntropyAdaptive => "Entropy-Adaptive Gated Sampling",
        }
    }
}

/// Parameters controlling speculative decoding execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeculativeExecutionConfig {
    /// Speculative lookahead horizon gamma (number of speculative draft tokens)
    pub lookahead_horizon: usize,
    /// Sampling temperature tau (0.0 for greedy, 0.7 for standard sampling)
    pub temperature: f32,
    /// Verification strategy to use
    pub strategy: SpeculativeVerificationStrategy,
    /// Base acceptance probability calibration multiplier (0.5 to 1.0)
    pub base_draft_fidelity: f32,
    /// Relative latency cost ratio r = T_draft / T_target
    pub latency_cost_ratio: f64,
}

impl Default for SpeculativeExecutionConfig {
    fn default() -> Self {
        Self {
            lookahead_horizon: 5,
            temperature: 0.7,
            strategy: SpeculativeVerificationStrategy::RejectionSampling,
            base_draft_fidelity: 0.82,
            latency_cost_ratio: 0.12,
        }
    }
}

/// Telemetry record for a single speculative execution step
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeculativeStepTelemetry {
    /// Step index
    pub step_index: usize,
    /// Number of speculative tokens drafted (gamma)
    pub drafted_tokens: usize,
    /// Number of speculative tokens accepted by the target model (k)
    pub accepted_tokens: usize,
    /// Total tokens emitted in this step (k + 1 bonus/recovery token)
    pub emitted_tokens: usize,
    /// Step acceptance rate: k / gamma
    pub acceptance_rate: f64,
    /// Step latency speedup relative to non-speculative autoregression
    pub step_speedup: f64,
    /// Target verification latency ms
    pub target_latency_ms: f64,
    /// Draft execution latency ms
    pub draft_latency_ms: f64,
}

/// Comprehensive telemetry report for speculative decoding evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeculativeTelemetryReport {
    pub lookahead_horizon: usize,
    pub total_steps_simulated: usize,
    pub total_tokens_emitted: usize,
    pub total_drafted_tokens: usize,
    pub total_accepted_tokens: usize,
    /// Overall empirical acceptance rate alpha = sum(accepted) / sum(drafted)
    pub mean_acceptance_rate: f64,
    /// Average accepted tokens per step: E[k]
    pub mean_accepted_per_step: f64,
    /// Average emitted tokens per target forward pass: E[k] + 1
    pub mean_tokens_per_pass: f64,
    /// Latency cost ratio r = T_draft / T_target
    pub latency_cost_ratio: f64,
    /// Effective wall-clock latency speedup multiplier S
    pub effective_speedup_multiplier: f64,
    /// Memory bandwidth traffic reduction percentage
    pub memory_traffic_savings_pct: f64,
    /// Per-step telemetry breakdown
    pub steps: Vec<SpeculativeStepTelemetry>,
}

impl SpeculativeTelemetryReport {
    pub fn format_summary(&self) -> String {
        format!(
            "Speculative Decoding Telemetry Report (Horizon gamma={})\n\
            - Total Tokens Emitted       : {}\n\
            - Mean Acceptance Rate alpha : {:.1}%\n\
            - Tokens Generated / Pass    : {:.2} tok/pass\n\
            - Draft Latency Cost Ratio r : {:.3}\n\
            - Effective Latency Speedup  : {:.2}x\n\
            - Memory Bandwidth Savings   : {:.1}%\n\
            - Status                     : {}",
            self.lookahead_horizon,
            self.total_tokens_emitted,
            self.mean_acceptance_rate * 100.0,
            self.mean_tokens_per_pass,
            self.latency_cost_ratio,
            self.effective_speedup_multiplier,
            self.memory_traffic_savings_pct,
            if self.effective_speedup_multiplier > 1.25 {
                "High Acceleration Gain (Production Recommended)"
            } else if self.effective_speedup_multiplier >= 1.0 {
                "Moderate Acceleration Gain"
            } else {
                "Draft Latency Overhead Exceeds Acceptance Gain"
            }
        )
    }
}

/// Pure Rust speculative decoding execution and simulation engine
pub struct SpeculativeEngine;

impl SpeculativeEngine {
    /// Simulates speculative decoding across multiple forward passes and calculates exact speedup
    pub fn simulate_speculative_run(
        config: &SpeculativeExecutionConfig,
        num_steps: usize,
        target_baseline_latency_ms: f64,
    ) -> SpeculativeTelemetryReport {
        let gamma = config.lookahead_horizon.max(1);
        let steps_count = num_steps.max(1);
        let r = config.latency_cost_ratio.clamp(0.01, 0.90);
        let draft_step_latency_ms = target_baseline_latency_ms * r;

        let mut step_records = Vec::with_capacity(steps_count);
        let mut total_drafted = 0;
        let mut total_accepted = 0;
        let mut total_emitted = 0;

        for step in 0..steps_count {
            // Compute token acceptance according to strategy
            let mut accepted_in_step = 0;

            // Temperature decay factor: lower temp increases draft alignment
            let temp_factor = if config.temperature <= 0.05 {
                1.15 // Near-greedy has higher agreement
            } else {
                1.0 / (1.0 + 0.20 * config.temperature as f64)
            };

            let base_p = (config.base_draft_fidelity as f64 * temp_factor).clamp(0.40, 0.98);

            // Rejection sampling cascade: sequentially verify tokens until first rejection
            for token_pos in 0..gamma {
                // Progressive positional entropy penalty: later draft tokens diverge more
                let pos_decay = 1.0 - (0.04 * token_pos as f64);
                let p_accept = (base_p * pos_decay).clamp(0.15, 0.99);

                // Deterministic pseudo-random sequence for reproducible telemetry
                let pseudo_rand = (((step + 1) * 73 + (token_pos + 1) * 31) % 100) as f64 / 100.0;

                if pseudo_rand <= p_accept {
                    accepted_in_step += 1;
                } else {
                    // Rejection: speculation stops at first failed candidate
                    break;
                }
            }

            // Total emitted tokens: accepted draft tokens + 1 target token (bonus or recovery)
            let emitted_in_step = accepted_in_step + 1;
            total_drafted += gamma;
            total_accepted += accepted_in_step;
            total_emitted += emitted_in_step;

            // Non-speculative time to emit this many tokens: emitted_in_step * target_baseline_latency_ms
            let non_spec_time = emitted_in_step as f64 * target_baseline_latency_ms;
            // Speculative time: gamma * draft_step_latency_ms + 1 * target_baseline_latency_ms
            let spec_time = (gamma as f64 * draft_step_latency_ms) + target_baseline_latency_ms;
            let step_speedup = non_spec_time / spec_time.max(0.001);

            step_records.push(SpeculativeStepTelemetry {
                step_index: step,
                drafted_tokens: gamma,
                accepted_tokens: accepted_in_step,
                emitted_tokens: emitted_in_step,
                acceptance_rate: accepted_in_step as f64 / gamma as f64,
                step_speedup,
                target_latency_ms: target_baseline_latency_ms,
                draft_latency_ms: gamma as f64 * draft_step_latency_ms,
            });
        }

        let mean_acceptance = total_accepted as f64 / total_drafted.max(1) as f64;
        let mean_accepted_per_step = total_accepted as f64 / steps_count as f64;
        let mean_tokens_per_pass = total_emitted as f64 / steps_count as f64;

        // Classical Leviathan formula: S = (E[k] + 1) / (1 + gamma * r)
        let effective_speedup = mean_tokens_per_pass / (1.0 + gamma as f64 * r);

        // Memory bandwidth reduction calculation
        let non_spec_traffic = total_emitted as f64 * 1.0;
        let spec_traffic = (steps_count as f64 * 1.0) + (total_drafted as f64 * r);
        let traffic_savings = (1.0 - (spec_traffic / non_spec_traffic.max(1.0))) * 100.0;

        SpeculativeTelemetryReport {
            lookahead_horizon: gamma,
            total_steps_simulated: steps_count,
            total_tokens_emitted: total_emitted,
            total_drafted_tokens: total_drafted,
            total_accepted_tokens: total_accepted,
            mean_acceptance_rate: mean_acceptance,
            mean_accepted_per_step,
            mean_tokens_per_pass,
            latency_cost_ratio: r,
            effective_speedup_multiplier: effective_speedup,
            memory_traffic_savings_pct: traffic_savings.clamp(0.0, 95.0),
            steps: step_records,
        }
    }

    /// Evaluates optimal speculative horizon gamma for a given draft latency ratio
    pub fn find_optimal_horizon(base_fidelity: f32, latency_ratio: f64) -> (usize, f64) {
        let mut best_horizon = 1;
        let mut best_speedup = 1.0;

        for gamma in 1..=12 {
            let config = SpeculativeExecutionConfig {
                lookahead_horizon: gamma,
                base_draft_fidelity: base_fidelity,
                latency_cost_ratio: latency_ratio,
                ..Default::default()
            };
            let report = Self::simulate_speculative_run(&config, 20, 10.0);
            if report.effective_speedup_multiplier > best_speedup {
                best_speedup = report.effective_speedup_multiplier;
                best_horizon = gamma;
            }
        }

        (best_horizon, best_speedup)
    }
}
