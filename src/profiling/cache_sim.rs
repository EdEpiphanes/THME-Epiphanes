use crate::profiling::hardware::HardwareDevice;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CacheTierLevel {
    L1,
    L2,
    L3,
    DRAM,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TierAccessReport {
    pub level: CacheTierLevel,
    pub name: String,
    pub capacity_bytes: usize,
    pub working_set_bytes: usize,
    pub hit_ratio: f64,
    pub miss_ratio: f64,
    pub latency_ns: f64,
    pub bandwidth_gbps: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheSimulationSummary {
    pub tiers: Vec<TierAccessReport>,
    pub effective_latency_ns: f64,
    pub effective_bandwidth_gbps: f64,
    pub total_working_set_bytes: usize,
    pub dram_spill_bytes: usize,
}

pub struct FourTierCacheSimulator {
    device: HardwareDevice,
}

impl FourTierCacheSimulator {
    pub fn new(device: HardwareDevice) -> Self {
        Self { device }
    }

    pub fn simulate_layer_working_set(&self, working_set_bytes: usize) -> CacheSimulationSummary {
        let mut tiers_report = Vec::new();
        let num_caches = self.device.caches.len();

        let mut tier_hit_ratios = Vec::new();
        let mut max_onchip_cache = 0;

        for (idx, cache) in self.device.caches.iter().enumerate() {
            let cap = cache.capacity_bytes;
            let level = match idx {
                0 => CacheTierLevel::L1,
                1 => CacheTierLevel::L2,
                2 => {
                    if num_caches > 3 {
                        CacheTierLevel::L3
                    } else {
                        CacheTierLevel::DRAM
                    }
                }
                _ => CacheTierLevel::DRAM,
            };

            if level != CacheTierLevel::DRAM && cap > max_onchip_cache {
                max_onchip_cache = cap;
            }

            let hit_ratio = if cap == 0 || working_set_bytes == 0 {
                0.0
            } else if working_set_bytes <= cap {
                0.95
            } else {
                ((cap as f64 / working_set_bytes as f64) * 0.95).clamp(0.0, 0.95)
            };

            let miss_ratio = (1.0 - hit_ratio).max(0.0);

            tiers_report.push(TierAccessReport {
                level,
                name: cache.name.clone(),
                capacity_bytes: cap,
                working_set_bytes,
                hit_ratio,
                miss_ratio,
                latency_ns: cache.latency_ns,
                bandwidth_gbps: cache.bandwidth_gbps,
            });

            tier_hit_ratios.push((hit_ratio, cache.latency_ns, cache.bandwidth_gbps));
        }

        let mut eff_latency = 0.0;
        let mut sum_inv_bw = 0.0;
        let mut miss_prob = 1.0;

        for (idx, (hit_r, lat, bw)) in tier_hit_ratios.iter().enumerate() {
            let is_last = idx == tier_hit_ratios.len() - 1;
            let serviced_frac = if is_last {
                miss_prob
            } else {
                miss_prob * hit_r
            };

            eff_latency += serviced_frac * lat;
            if *bw > 0.0 {
                sum_inv_bw += serviced_frac / bw;
            }

            if !is_last {
                miss_prob *= 1.0 - hit_r;
            }
        }

        let eff_bw = if sum_inv_bw > 0.0 {
            1.0 / sum_inv_bw
        } else {
            self.device.dram_bandwidth_gbps
        };

        let dram_spill = (working_set_bytes as f64 * miss_prob).round() as usize;

        CacheSimulationSummary {
            tiers: tiers_report,
            effective_latency_ns: eff_latency,
            effective_bandwidth_gbps: eff_bw,
            total_working_set_bytes: working_set_bytes,
            dram_spill_bytes: dram_spill,
        }
    }

    pub fn simulate_working_set(&self, working_set_bytes: usize) -> CacheSimulationSummary {
        self.simulate_layer_working_set(working_set_bytes)
    }
}

pub type CacheSimulator4Tier = FourTierCacheSimulator;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_four_tier_cache_simulator_spill() {
        let h100 = HardwareDevice::predefined_h100();
        let sim = FourTierCacheSimulator::new(h100);

        let small_summary = sim.simulate_working_set(1_000_000);
        assert_eq!(small_summary.tiers[0].level, CacheTierLevel::L1);
        assert_eq!(small_summary.tiers[0].hit_ratio, 0.95);
        assert_eq!(small_summary.dram_spill_bytes, 2500);

        let large_working_set = 100_000_000;
        let large_summary = sim.simulate_working_set(large_working_set);
        assert!(large_summary.tiers[0].hit_ratio < 0.95);
        assert!(large_summary.tiers[0].miss_ratio > 0.05);
        assert!(large_summary.dram_spill_bytes > 30_000_000);
        assert!(large_summary.effective_latency_ns > small_summary.effective_latency_ns);
    }
}
