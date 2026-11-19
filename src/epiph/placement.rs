// SPDX-License-Identifier: AGPL-3.0-only
//! GPU / CPU / stream placement of `.epiph` tensors under a memory budget (THME-3016).
//!
//! Pure functions: sizes in bytes in, a placement out. Two exact selections with the
//! multiple-choice allocator: first the most bytes that fit VRAM go to the GPU, then the
//! most bytes of the rest that fit host RAM stay in CPU memory; the remainder is streamed
//! from the file on demand.

use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};

use crate::engine::alloc::{min_cost_choice, Choice};
use crate::engine::error::EngineError;

const GPU_REASON: &str =
    "on the GPU: part of the selection with the most bytes that fits the VRAM budget";
const CPU_REASON: &str = "did not fit the VRAM selection; fits the host RAM budget";
const STREAM_REASON: &str =
    "fits neither the VRAM nor the host RAM selection; streamed from the file";

/// Where a tensor is read from during inference.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Tier {
    Gpu,
    Cpu,
    /// Read from the file on demand.
    Stream,
}

/// Memory available for weights, in bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Budget {
    pub vram_bytes: u64,
    pub host_bytes: u64,
}

impl Budget {
    /// "8192:32768" -> VRAM 8192 MiB, host RAM 32768 MiB (bytes = MiB * 1_048_576, checked).
    pub fn parse_mib(text: &str) -> Result<Budget> {
        let mib = |part: &str| {
            part.parse::<u64>()
                .ok()
                .and_then(|n| n.checked_mul(1 << 20))
        };
        text.split_once(':')
            .and_then(|(vram, host)| {
                Some(Budget {
                    vram_bytes: mib(vram)?,
                    host_bytes: mib(host)?,
                })
            })
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "budget '{text}' is not VRAM_MIB:HOST_MIB (two whole numbers of MiB, such as 8192:32768)"
                )
            })
    }
}

/// One tensor's tier, size and the reason for that tier.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Placed {
    pub name: String,
    pub bytes: u64,
    pub tier: Tier,
    pub reason: String,
}

/// The tier of every tensor (same order as the input) and the bytes per tier.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Placement {
    pub budget: Budget,
    pub gpu_bytes: u64,
    pub cpu_bytes: u64,
    pub stream_bytes: u64,
    pub tensors: Vec<Placed>,
}

/// Size grid for a budget: 4096 units, or one byte per unit for small budgets.
fn grid_unit(budget: u64) -> u64 {
    budget.div_ceil(4096).max(1)
}

/// Places every tensor. `tensors` is `(name, bytes)` in order.
pub fn plan(tensors: &[(String, u64)], budget: Budget) -> Result<Placement, EngineError> {
    let vram_unit = grid_unit(budget.vram_bytes);
    let host_unit = grid_unit(budget.host_bytes);

    let gpu_items: Vec<Vec<Choice>> = tensors
        .iter()
        .map(|&(_, bytes)| {
            vec![
                Choice {
                    weight: bytes.div_ceil(vram_unit),
                    cost: 0.0,
                },
                Choice {
                    weight: 0,
                    cost: bytes as f64,
                },
            ]
        })
        .collect();
    let gpu = min_cost_choice(&gpu_items, budget.vram_bytes / vram_unit)?;

    // Option 1 of the GPU pass means "not on the GPU".
    let off_gpu: Vec<(usize, u64)> = tensors
        .iter()
        .zip(&gpu.choice)
        .enumerate()
        .filter(|(_, (_, &option))| option == 1)
        .map(|(i, ((_, bytes), _))| (i, *bytes))
        .collect();
    let host_items: Vec<Vec<Choice>> = off_gpu
        .iter()
        .map(|&(_, bytes)| {
            vec![
                Choice {
                    weight: bytes.div_ceil(host_unit),
                    cost: 0.0,
                },
                Choice {
                    weight: 0,
                    cost: bytes as f64,
                },
            ]
        })
        .collect();
    let host = min_cost_choice(&host_items, budget.host_bytes / host_unit)?;

    let mut tiers: Vec<Tier> = gpu
        .choice
        .iter()
        .map(|&option| if option == 0 { Tier::Gpu } else { Tier::Stream })
        .collect();
    for (&(i, _), &option) in off_gpu.iter().zip(&host.choice) {
        if let Some(tier) = tiers.get_mut(i) {
            *tier = if option == 0 { Tier::Cpu } else { Tier::Stream };
        }
    }

    let (mut gpu_bytes, mut cpu_bytes, mut stream_bytes) = (0u64, 0u64, 0u64);
    let mut placed = Vec::with_capacity(tensors.len());
    for ((name, bytes), tier) in tensors.iter().zip(tiers) {
        let total = match tier {
            Tier::Gpu => &mut gpu_bytes,
            Tier::Cpu => &mut cpu_bytes,
            Tier::Stream => &mut stream_bytes,
        };
        *total = total
            .checked_add(*bytes)
            .ok_or_else(|| EngineError::InvalidInput("tensor sizes overflow u64".to_string()))?;
        let reason = match tier {
            Tier::Gpu => GPU_REASON,
            Tier::Cpu => CPU_REASON,
            Tier::Stream => STREAM_REASON,
        };
        placed.push(Placed {
            name: name.clone(),
            bytes: *bytes,
            tier,
            reason: reason.to_string(),
        });
    }
    Ok(Placement {
        budget,
        gpu_bytes,
        cpu_bytes,
        stream_bytes,
        tensors: placed,
    })
}

impl Placement {
    /// Checks this placement against the tensors it was made from and against its budget.
    pub fn check(&self, tensors: &[(String, u64)]) -> Result<()> {
        if self.tensors.len() != tensors.len() {
            bail!(
                "placement holds {} tensors, expected {}",
                self.tensors.len(),
                tensors.len()
            );
        }
        for (placed, (name, bytes)) in self.tensors.iter().zip(tensors) {
            if placed.name != *name || placed.bytes != *bytes {
                bail!(
                    "tensor '{name}' ({bytes} bytes) is placed as '{}' ({} bytes)",
                    placed.name,
                    placed.bytes
                );
            }
        }
        let sum = |tier: Tier| {
            self.tensors
                .iter()
                .filter(|t| t.tier == tier)
                .fold(0u64, |acc, t| acc.saturating_add(t.bytes))
        };
        for (total, tier, name) in [
            (self.gpu_bytes, Tier::Gpu, "gpu_bytes"),
            (self.cpu_bytes, Tier::Cpu, "cpu_bytes"),
            (self.stream_bytes, Tier::Stream, "stream_bytes"),
        ] {
            if total != sum(tier) {
                bail!(
                    "{name} is {total}, but the tensors on that tier sum to {}",
                    sum(tier)
                );
            }
        }
        if self.gpu_bytes > self.budget.vram_bytes {
            bail!(
                "gpu_bytes {} exceeds the VRAM budget {}",
                self.gpu_bytes,
                self.budget.vram_bytes
            );
        }
        if self.cpu_bytes > self.budget.host_bytes {
            bail!(
                "cpu_bytes {} exceeds the host budget {}",
                self.cpu_bytes,
                self.budget.host_bytes
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            self.0 >> 33
        }
    }

    fn named(sizes: &[u64]) -> Vec<(String, u64)> {
        sizes
            .iter()
            .enumerate()
            .map(|(i, &bytes)| (format!("t{i}"), bytes))
            .collect()
    }

    fn budget(vram_bytes: u64, host_bytes: u64) -> Budget {
        Budget {
            vram_bytes,
            host_bytes,
        }
    }

    fn tiers(p: &Placement) -> Vec<Tier> {
        p.tensors.iter().map(|t| t.tier).collect()
    }

    #[test]
    fn everything_fits_on_the_gpu() {
        let p = plan(&named(&[10, 20, 30]), budget(100, 100)).unwrap();
        assert_eq!(tiers(&p), vec![Tier::Gpu; 3]);
        assert_eq!(p.gpu_bytes, 60);
        assert_eq!(p.cpu_bytes, 0);
        assert_eq!(p.stream_bytes, 0);
    }

    #[test]
    fn exact_choice_beats_greedy_by_order() {
        let p = plan(&named(&[60, 50, 40, 30]), budget(100, 60)).unwrap();
        assert_eq!(
            tiers(&p),
            vec![Tier::Gpu, Tier::Cpu, Tier::Gpu, Tier::Stream]
        );
        assert_eq!(p.gpu_bytes, 100);
        assert_eq!(p.cpu_bytes, 50);
        assert_eq!(p.stream_bytes, 30);
        assert_eq!(p.tensors[0].reason, GPU_REASON);
        assert_eq!(p.tensors[1].reason, CPU_REASON);
        assert_eq!(p.tensors[3].reason, STREAM_REASON);
        let names: Vec<&str> = p.tensors.iter().map(|t| t.name.as_str()).collect();
        assert_eq!(names, vec!["t0", "t1", "t2", "t3"]);
    }

    #[test]
    fn zero_budgets_stream_everything() {
        let p = plan(&named(&[10, 20, 30]), budget(0, 0)).unwrap();
        assert_eq!(tiers(&p), vec![Tier::Stream; 3]);
        assert_eq!(p.stream_bytes, 60);

        let p = plan(&named(&[500]), budget(10, 0)).unwrap();
        assert_eq!(tiers(&p), vec![Tier::Stream]);
    }

    #[test]
    fn tensor_larger_than_both_budgets_streams_alone() {
        let p = plan(&named(&[500, 40, 30]), budget(100, 50)).unwrap();
        assert_eq!(tiers(&p), vec![Tier::Stream, Tier::Gpu, Tier::Gpu]);
        assert_eq!(p.gpu_bytes, 70);
        assert_eq!(p.stream_bytes, 500);
    }

    #[test]
    fn random_instances_respect_budgets_and_totals() {
        let mut rng = Lcg(3016);
        for _ in 0..200 {
            let n = 1 + (rng.next() % 12) as usize;
            let sizes: Vec<u64> = (0..n).map(|_| rng.next() % 5001).collect();
            let tensors = named(&sizes);
            let b = budget(rng.next() % 40_001, rng.next() % 40_001);
            let p = plan(&tensors, b).unwrap();
            assert!(p.gpu_bytes <= b.vram_bytes);
            assert!(p.cpu_bytes <= b.host_bytes);
            let total: u64 = sizes.iter().sum();
            assert_eq!(p.gpu_bytes + p.cpu_bytes + p.stream_bytes, total);
            assert_eq!(plan(&tensors, b).unwrap(), p);
            assert!(p.check(&tensors).is_ok());
        }
    }

    #[test]
    fn check_accepts_a_fresh_plan_and_rejects_edits() {
        let tensors = named(&[60, 50, 40, 30]);
        let fresh = plan(&tensors, budget(100, 60)).unwrap();
        assert!(fresh.check(&tensors).is_ok());

        let mut bad = fresh.clone();
        bad.cpu_bytes += 1;
        assert!(bad.check(&tensors).is_err());

        let mut bad = fresh.clone();
        bad.tensors[0].name = "renamed".to_string();
        assert!(bad.check(&tensors).is_err());

        let mut bad = fresh.clone();
        bad.tensors.swap(0, 1);
        assert!(bad.check(&tensors).is_err());

        let mut bad = fresh.clone();
        bad.budget.vram_bytes = 10;
        assert!(bad.check(&tensors).is_err());
    }

    #[test]
    fn empty_input_gives_empty_plan() {
        let p = plan(&[], budget(100, 100)).unwrap();
        assert!(p.tensors.is_empty());
        assert_eq!((p.gpu_bytes, p.cpu_bytes, p.stream_bytes), (0, 0, 0));
    }

    #[test]
    fn json_round_trip_uses_lowercase_tiers() {
        let p = plan(&named(&[60, 50, 40, 30]), budget(100, 60)).unwrap();
        let json = serde_json::to_string(&p).unwrap();
        assert!(json.contains("\"tier\":\"gpu\""));
        assert!(json.contains("\"tier\":\"cpu\""));
        assert!(json.contains("\"tier\":\"stream\""));
        assert_eq!(serde_json::from_str::<Placement>(&json).unwrap(), p);
    }

    #[test]
    fn parse_mib_reads_vram_and_host_in_mib() {
        assert_eq!(
            Budget::parse_mib("8192:32768").unwrap(),
            budget(8192 * 1_048_576, 32768 * 1_048_576)
        );
        assert_eq!(Budget::parse_mib("0:0").unwrap(), budget(0, 0));
    }

    #[test]
    fn parse_mib_refuses_bad_text_without_panicking() {
        for text in [
            "",
            "5",
            "1:2:3",
            "a:1",
            "-1:1",
            "1:",
            "18446744073709551615:1",
        ] {
            assert!(Budget::parse_mib(text).is_err(), "{text:?}");
        }
    }
}
