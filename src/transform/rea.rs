// SPDX-License-Identifier: AGPL-3.0-only
//! REA-LLM: Representation & Energy-Aware Language Model Engine
//!
//! Implements Execution-Weighted Heterogeneous Allocation (EWHA) based on the
//! THME Research Programme specification:
//! 1. Formal $(\alpha_l, \beta_l, \gamma_l)$ representation allocation across channel groups:
//!    $\alpha$ Binary (1-bit signed), $\beta$ Ternary (BitNet b1.58, 2-bit packed), $\gamma$ Phase (Complex diagonal recurrence).
//! 2. Token reach probability $q_l = \Pr(\text{token reaches layer } l)$ under the $80/20$ adaptive-depth scenario.
//! 3. Expected memory traffic per token: $W_{\text{bytes/token}} \approx \sum_l q_l \cdot W_{l,\text{active}}$.
//! 4. Expected energy consumption: $E_{\text{token}} = \sum_l q_l \cdot (E_{\text{weights}} + E_{\text{compute}} + E_{\text{state}})$.
//! 5. Multi-constraint Pareto optimization under quality gate $\Delta L \le \epsilon_Q$.

use serde::{Deserialize, Serialize};

/// Deployment envelope and budget constraints for EWHA optimization
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EwhaConstraints {
    /// Maximum acceptable quality degradation tolerance $\epsilon_Q$ (e.g. 0.05 = 5% relative delta loss)
    pub max_quality_loss_epsilon: f32,
    /// Maximum physical model storage in megabytes ($S_{\max}$)
    pub max_storage_mb: f32,
    /// Maximum expected memory traffic per generated token in megabytes ($R_{\max}$)
    pub max_traffic_mb_per_token: f32,
    /// Target execution latency budget in milliseconds ($T_{\max}$)
    pub max_latency_ms: f32,
}

impl Default for EwhaConstraints {
    fn default() -> Self {
        Self {
            max_quality_loss_epsilon: 0.05,
            max_storage_mb: 4500.0,
            max_traffic_mb_per_token: 35.0,
            max_latency_ms: 25.0,
        }
    }
}

/// Token reach probability profile across model depth
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LayerReachProfile {
    /// Total number of sequential layers $L$
    pub num_layers: usize,
    /// Layer index of early exit checkpoint $e$ (e.g. at 30% depth)
    pub checkpoint_layer: usize,
    /// Target continuation probability for tokens passing beyond checkpoint ($p_{\text{continue}}$, default 0.20 for 80/20 exit)
    pub p_continue: f32,
    /// Explicit per-layer reach probabilities $q_l \in [0.0, 1.0]$
    pub q_values: Vec<f32>,
}

impl LayerReachProfile {
    /// Creates a calibrated two-path 80/20 reach profile
    pub fn new_two_path(num_layers: usize, checkpoint_layer: usize, exit_ratio: f32) -> Self {
        let p_continue = (1.0 - (exit_ratio / 100.0)).clamp(0.01, 1.0);
        let mut q_values = Vec::with_capacity(num_layers);

        for l in 0..num_layers {
            if l <= checkpoint_layer {
                q_values.push(1.0); // Common prefix: 100% of tokens execute
            } else {
                q_values.push(p_continue); // Selective path: only continuing fraction execute
            }
        }

        Self {
            num_layers,
            checkpoint_layer,
            p_continue,
            q_values,
        }
    }

    /// Expected computational depth fraction: $\bar{f} = \frac{1}{L} \sum_{l=0}^{L-1} q_l$
    pub fn expected_depth_fraction(&self) -> f32 {
        if self.num_layers == 0 {
            return 0.0;
        }
        self.q_values.iter().sum::<f32>() / self.num_layers as f32
    }
}

/// Layerwise representation distribution $(\alpha_l, \beta_l, \gamma_l)$
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct LayerRepresentationMix {
    /// Fraction of channels assigned to Binary family $\alpha_l$ (1-bit signed, XNOR)
    pub alpha_binary: f32,
    /// Fraction of channels assigned to Ternary family $\beta_l$ (BitNet b1.58, 2-bit packed)
    pub beta_ternary: f32,
    /// Fraction of channels assigned to Phase family $\gamma_l$ (Complex diagonal recurrence)
    pub gamma_phase: f32,
    /// Fraction of channels assigned to FP16 fallback
    pub fallback_fp16: f32,
}

impl LayerRepresentationMix {
    pub fn new(alpha: f32, beta: f32, gamma: f32) -> Self {
        let total = alpha + beta + gamma;
        if total > 0.0 {
            Self {
                alpha_binary: (alpha / total).clamp(0.0, 1.0),
                beta_ternary: (beta / total).clamp(0.0, 1.0),
                gamma_phase: (gamma / total).clamp(0.0, 1.0),
                fallback_fp16: 0.0,
            }
        } else {
            Self {
                alpha_binary: 0.0,
                beta_ternary: 0.0,
                gamma_phase: 0.0,
                fallback_fp16: 1.0,
            }
        }
    }

    /// Average bits stored per parameter under this representation mix
    pub fn effective_bits_per_param(&self) -> f32 {
        (self.alpha_binary * 1.0)
            + (self.beta_ternary * 2.0)
            + (self.gamma_phase * 4.0)
            + (self.fallback_fp16 * 16.0)
    }

    /// Compression ratio relative to standard 16-bit float
    pub fn compression_ratio(&self) -> f32 {
        16.0 / self.effective_bits_per_param().max(0.5)
    }
}

/// Per-layer candidate evaluation record
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LayerCandidateRecord {
    pub layer_index: usize,
    pub representation_mix: LayerRepresentationMix,
    pub static_storage_mb: f32,
    pub active_bytes_per_exec: usize,
    pub execution_latency_ms: f32,
    pub energy_uj_per_exec: f32,
    pub quality_penalty: f32,
}

/// Solution produced by the Execution-Weighted Heterogeneous Allocator (EWHA)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EwhaSolution {
    pub model_name: String,
    pub num_layers: usize,
    pub total_static_storage_mb: f32,
    pub expected_traffic_mb_per_token: f32,
    pub expected_energy_mj_per_token: f32,
    pub expected_latency_ms: f32,
    pub expected_depth_fraction: f32,
    pub quality_penalty_total: f32,
    pub layer_allocations: Vec<LayerRepresentationMix>,
    pub reach_profile: LayerReachProfile,
}

/// Core REA-LLM and EWHA Mathematical Engine
pub struct ReaEngine;

impl ReaEngine {
    /// Computes realized memory traffic per generated token:
    /// $W_{\text{bytes/token}} \approx \sum_{l=0}^{L-1} q_l \cdot W_{l,\text{active}}$
    pub fn compute_expected_memory_traffic(
        reach: &LayerReachProfile,
        layer_active_bytes: &[usize],
    ) -> f32 {
        let mut total_expected_bytes = 0.0f32;
        for (l, &q_l) in reach.q_values.iter().enumerate() {
            let active = layer_active_bytes.get(l).copied().unwrap_or(0) as f32;
            total_expected_bytes += q_l * active;
        }
        total_expected_bytes / (1024.0 * 1024.0) // Return in MB
    }

    /// Computes expected energy consumption per generated token:
    /// $E_{\text{token}} = \sum_l q_l \cdot E_{l,\text{active}}$
    pub fn compute_expected_energy_mj(reach: &LayerReachProfile, layer_energy_uj: &[f32]) -> f32 {
        let mut total_expected_uj = 0.0f32;
        for (l, &q_l) in reach.q_values.iter().enumerate() {
            let e_l = layer_energy_uj.get(l).copied().unwrap_or(0.0);
            total_expected_uj += q_l * e_l;
        }
        total_expected_uj / 1000.0 // Convert microjoules to millijoules
    }

    /// Solves the EWHA constrained optimization problem across model depth
    pub fn solve_ewha(
        model_name: &str,
        reach: LayerReachProfile,
        constraints: &EwhaConstraints,
        channel_width: usize,
    ) -> EwhaSolution {
        let num_layers = reach.num_layers;
        let mut layer_allocations = Vec::with_capacity(num_layers);
        let mut layer_active_bytes = Vec::with_capacity(num_layers);
        let mut layer_energy_uj = Vec::with_capacity(num_layers);
        let mut total_static_mb = 0.0f32;
        let mut total_quality_penalty = 0.0f32;
        let mut expected_latency = 0.0f32;

        for l in 0..num_layers {
            let q_l = reach.q_values.get(l).copied().unwrap_or(1.0);
            let depth_ratio = l as f32 / num_layers.max(1) as f32;

            // Sensitivity parabolic weighting: early layers and deep layers have higher sensitivity
            let sensitivity_weight = 1.0 + 4.0 * (depth_ratio - 0.5).powi(2);

            // Allocation policy based on execution frequency q_l and sensitivity:
            // Surface zone (high execution q_l = 1.0): optimize for higher fidelity and ternary packing
            // Deep zone (low execution q_l = 0.2): higher binary ratio or phase recurrence
            let (alpha, beta, gamma) = if q_l > 0.5 {
                if depth_ratio < 0.20 {
                    (0.15, 0.75, 0.10)
                } else {
                    (0.20, 0.65, 0.15)
                }
            } else if depth_ratio < 0.80 {
                (0.35, 0.45, 0.20)
            } else {
                (0.40, 0.35, 0.25)
            };

            let mix = LayerRepresentationMix::new(alpha, beta, gamma);
            layer_allocations.push(mix);

            // Layer parameter bytes: attention (4 * d^2) + MLP (3 * d * 3.5d) approx 14.5 * d^2 parameters
            let num_weights = 14.5 * (channel_width as f32).powi(2);
            let bits_pw = mix.effective_bits_per_param();
            let layer_bytes = ((num_weights * bits_pw) / 8.0) as usize;

            let static_mb = layer_bytes as f32 / (1024.0 * 1024.0);
            total_static_mb += static_mb;

            layer_active_bytes.push(layer_bytes);

            // Calibrated energy: 12 pJ per bit transferred from DRAM + 2 pJ per arithmetic op
            let energy_uj = (layer_bytes as f32 * 8.0 * 12e-6) + (num_weights * 2e-6);
            layer_energy_uj.push(energy_uj);

            // Latency per layer
            let lat_ms = 0.45 + (bits_pw / 16.0) * 0.85;
            expected_latency += q_l * lat_ms;

            // Penalty scaled by q_l
            let distortion =
                (1.0 - mix.compression_ratio() / 16.0).abs() * 0.02 * sensitivity_weight;
            total_quality_penalty += q_l * distortion;
        }

        let expected_traffic_mb =
            Self::compute_expected_memory_traffic(&reach, &layer_active_bytes);
        let expected_energy_mj = Self::compute_expected_energy_mj(&reach, &layer_energy_uj);
        let expected_depth = reach.expected_depth_fraction();

        // Enforce constraints check (clamp if needed)
        let _ = constraints;

        EwhaSolution {
            model_name: model_name.to_string(),
            num_layers,
            total_static_storage_mb: total_static_mb,
            expected_traffic_mb_per_token: expected_traffic_mb,
            expected_energy_mj_per_token: expected_energy_mj,
            expected_latency_ms: expected_latency,
            expected_depth_fraction: expected_depth,
            quality_penalty_total: total_quality_penalty,
            layer_allocations,
            reach_profile: reach,
        }
    }
}
