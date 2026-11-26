// SPDX-License-Identifier: AGPL-3.0-only
//! Hardware-Model-Intent Policy Selector (`E_select`) and Scenario A-D
//! compressed off-loading scheduler (`THME_Research_Dossier_v3.3` §7.8, §7.11).

use crate::engine::error::{EngineError, EngineResult};

/// Hardware execution scenario (§7.8.4).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OffloadScenario {
    /// Model + KV fit inside `0.85 * V_free` (100% GPU-resident).
    ScenarioA,
    /// Weights fit in VRAM; long-context KV spills to Host `Polar+LZ4` / `QJL`.
    ScenarioB,
    /// Weights exceed VRAM but fit in Host RAM (`PCIe` / CPU `AVX-512` split).
    ScenarioC,
    /// Model + KV exceed Host RAM (4KB `O_DIRECT` NVMe ring + forced `K=4..8`).
    ScenarioD,
}

/// Probed hardware capabilities `H`.
#[derive(Debug, Clone, Copy)]
pub struct HardwareEnvelope {
    pub vram_free_bytes: u64,
    pub host_ram_free_bytes: u64,
    pub pcie_gbps_ewma: f64,
    pub has_avx512: bool,
}

impl HardwareEnvelope {
    /// Available VRAM budget with 15% WDDM/driver headroom (`0.85 * V_free`).
    pub fn vram_avail_bytes(&self) -> u64 {
        ((self.vram_free_bytes as f64) * 0.85).floor() as u64
    }

    /// Capped pinned host staging pool (`min(4 GiB, 0.25 * Host_RAM_Free)`).
    pub fn pinned_host_cap_bytes(&self) -> u64 {
        let four_gib = 4_u64 * 1024 * 1024 * 1024;
        let quarter = self.host_ram_free_bytes / 4;
        four_gib.min(quarter)
    }

    /// Updates online EWMA PCIe bandwidth estimate (`R93`).
    pub fn update_pcie_ewma(&mut self, measured_gbps: f64) {
        if measured_gbps.is_finite() && measured_gbps > 0.0 {
            self.pcie_gbps_ewma = 0.8 * self.pcie_gbps_ewma + 0.2 * measured_gbps;
        }
    }
}

/// User preference weight vector `u = (w_qual, w_speed, w_mem, w_fact)`.
#[derive(Debug, Clone, Copy)]
pub struct UserIntent {
    pub w_qual: f64,
    pub w_speed: f64,
    pub w_mem: f64,
    pub w_fact: f64,
}

/// Candidate technique policy evaluated by `E_select`.
#[derive(Debug, Clone)]
pub struct PolicyCandidate {
    pub name: String,
    pub is_baseline: bool,
    pub quality_score: f64,
    pub speed_score: f64,
    pub mem_score: f64,
    pub fact_score: f64,
    pub required_vram_bytes: u64,
    pub gate_passed: bool,
}

/// Selects the active off-loading scenario with 10% hysteresis (`R121`) when
/// transitioning back from `ScenarioC` to `ScenarioB`/`ScenarioA`.
pub fn select_offload_scenario(
    hw: &HardwareEnvelope,
    weights_bytes: u64,
    kv_bytes: u64,
    prev: Option<OffloadScenario>,
    tokens_since_switch: usize,
) -> OffloadScenario {
    let v_avail = hw.vram_avail_bytes();
    let total = weights_bytes.saturating_add(kv_bytes);

    if let Some(OffloadScenario::ScenarioC) = prev {
        if tokens_since_switch < 32 {
            return OffloadScenario::ScenarioC;
        }
        // Require 10% hysteresis headroom before re-promoting evicted weights.
        if (weights_bytes as f64) > (v_avail as f64) * 0.90 {
            return if total <= hw.host_ram_free_bytes {
                OffloadScenario::ScenarioC
            } else {
                OffloadScenario::ScenarioD
            };
        }
    }

    if total <= v_avail {
        OffloadScenario::ScenarioA
    } else if weights_bytes <= v_avail {
        OffloadScenario::ScenarioB
    } else if total <= hw.host_ram_free_bytes {
        OffloadScenario::ScenarioC
    } else {
        OffloadScenario::ScenarioD
    }
}

/// Computes Residency Utility `U_res(G) = f_reuse * max(0, bytes/B_pcie - t_comp) / bytes`.
pub fn residency_utility(
    bytes: u64,
    reuse_freq: f64,
    pcie_gbps: f64,
    prev_layer_compute_sec: f64,
) -> EngineResult<f64> {
    if bytes == 0 || !pcie_gbps.is_finite() || pcie_gbps <= 0.0 {
        return Err(EngineError::InvalidInput(
            "bytes and pcie_gbps must be positive".to_string(),
        ));
    }
    let transfer_sec = (bytes as f64) / (pcie_gbps * 1e9);
    let stall_sec = (transfer_sec - prev_layer_compute_sec.max(0.0)).max(0.0);
    Ok((reuse_freq.max(0.0) * stall_sec) / (bytes as f64))
}

/// Acceptance- and VRAM-adaptive speculative draft horizon `K in [1, 8]` (`R95`).
pub fn adaptive_draft_horizon(vram_free_bytes: u64, acceptance_rate: f64) -> usize {
    let k_vram = if vram_free_bytes < 1024 * 1024 * 1024 {
        1
    } else if vram_free_bytes < 2 * 1024 * 1024 * 1024 {
        2
    } else if vram_free_bytes < 4 * 1024 * 1024 * 1024 {
        4
    } else {
        8
    };
    let acc = acceptance_rate.clamp(0.0, 0.95);
    let k_acc = (1.0 / (1.0 - acc)).floor().max(1.0) as usize;
    k_vram.min(k_acc).clamp(1, 8)
}

/// Non-replacement policy selector `E_select(H, M, u)` with switch-hysteresis `tau_switch`.
pub fn select_policy<'a>(
    hw: &HardwareEnvelope,
    intent: &UserIntent,
    candidates: &'a [PolicyCandidate],
    tau_switch: f64,
) -> EngineResult<&'a PolicyCandidate> {
    let v_avail = hw.vram_avail_bytes();
    let mut best: Option<(&'a PolicyCandidate, f64)> = None;

    for cand in candidates {
        if !cand.gate_passed || cand.required_vram_bytes > v_avail {
            continue;
        }
        let penalty = if cand.is_baseline { 0.0 } else { tau_switch };
        let utility = intent.w_qual * cand.quality_score
            + intent.w_speed * cand.speed_score
            + intent.w_mem * cand.mem_score
            + intent.w_fact * cand.fact_score
            - penalty;

        if best.is_none_or(|(_, u)| utility > u) {
            best = Some((cand, utility));
        }
    }

    best.map(|(c, _)| c).ok_or_else(|| {
        EngineError::Infeasible("no feasible gate-passing policy fits within VRAM".to_string())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scenarios_classify_a_b_c_d_with_hysteresis() {
        let hw = HardwareEnvelope {
            vram_free_bytes: 10_000,
            host_ram_free_bytes: 50_000,
            pcie_gbps_ewma: 24.0,
            has_avx512: true,
        };
        assert_eq!(
            select_offload_scenario(&hw, 4_000, 2_000, None, 0),
            OffloadScenario::ScenarioA
        );
        assert_eq!(
            select_offload_scenario(&hw, 6_000, 4_000, None, 0),
            OffloadScenario::ScenarioB
        );
        assert_eq!(
            select_offload_scenario(&hw, 12_000, 4_000, None, 0),
            OffloadScenario::ScenarioC
        );
        assert_eq!(
            select_offload_scenario(&hw, 40_000, 20_000, None, 0),
            OffloadScenario::ScenarioD
        );
        // Hysteresis holds ScenarioC for < 32 tokens.
        assert_eq!(
            select_offload_scenario(&hw, 4_000, 2_000, Some(OffloadScenario::ScenarioC), 10),
            OffloadScenario::ScenarioC
        );
    }

    #[test]
    fn policy_selector_respects_non_replacement_hysteresis() {
        let hw = HardwareEnvelope {
            vram_free_bytes: 16 * 1024 * 1024 * 1024,
            host_ram_free_bytes: 32 * 1024 * 1024 * 1024,
            pcie_gbps_ewma: 24.0,
            has_avx512: true,
        };
        let intent = UserIntent {
            w_qual: 1.0,
            w_speed: 1.0,
            w_mem: 0.5,
            w_fact: 0.5,
        };
        let cands = vec![
            PolicyCandidate {
                name: "AWQ_Q4_0".into(),
                is_baseline: true,
                quality_score: 0.90,
                speed_score: 0.80,
                mem_score: 0.70,
                fact_score: 0.80,
                required_vram_bytes: 2_000_000_000,
                gate_passed: true,
            },
            PolicyCandidate {
                name: "Marginal_Alt".into(),
                is_baseline: false,
                quality_score: 0.91,
                speed_score: 0.80,
                mem_score: 0.70,
                fact_score: 0.80,
                required_vram_bytes: 2_000_000_000,
                gate_passed: true,
            },
        ];
        // Marginal 0.01 gain does not beat tau_switch = 0.05 -> baseline preserved.
        let picked = select_policy(&hw, &intent, &cands, 0.05).unwrap();
        assert_eq!(picked.name, "AWQ_Q4_0");
    }
}
