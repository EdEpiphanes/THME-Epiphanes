// SPDX-License-Identifier: AGPL-3.0-only
//! Automatic resident weight budget and format selection for GPU inference.

#![cfg_attr(not(test), allow(dead_code))]

/// Device-side resident weight storage format, ordered from largest to smallest.
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ResidentFormat {
    F32,
    F16,
    Q8_0,
    Q4_0,
}

impl ResidentFormat {
    const ALL: [Self; 4] = [Self::F32, Self::F16, Self::Q8_0, Self::Q4_0];

    fn bytes_for(self, model_f32_bytes: u64) -> u64 {
        let num: u128 = match self {
            Self::F32 => 128,
            Self::F16 => 64,
            Self::Q8_0 => 34,
            Self::Q4_0 => 18,
        };
        let scaled = (u128::from(model_f32_bytes) * num).div_ceil(128);
        u64::try_from(scaled).unwrap_or(u64::MAX)
    }
}

/// Device memory query and probe-allocation interface.
pub(super) trait MemoryProbe {
    /// Free device memory in bytes, or `None` when the backend does not report VRAM.
    fn free_bytes(&self) -> Option<u64>;
    /// Attempts a probe allocation of `bytes`; returns `true` when the device accepts it.
    fn try_alloc(&self, bytes: u64) -> bool;
}

/// Selected resident storage format and cache budget for a model on a GPU device.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct WeightPlan {
    pub(super) format: ResidentFormat,
    pub(super) budget_bytes: u64,
    pub(super) resident: bool,
    pub(super) reason: String,
}

fn probe_halving(model_f32_bytes: u64, probe: &dyn MemoryProbe) -> u64 {
    if model_f32_bytes == 0 {
        return 0;
    }
    let min_bytes = ResidentFormat::Q4_0.bytes_for(model_f32_bytes);
    let mut size = model_f32_bytes;
    loop {
        if probe.try_alloc(size) {
            return size;
        }
        if size <= min_bytes {
            return 0;
        }
        size = (size / 2).max(min_bytes);
    }
}

/// Chooses the highest-precision resident format and budget that fit available device memory.
pub(super) fn plan_weight_budget(model_f32_bytes: u64, probe: &dyn MemoryProbe) -> WeightPlan {
    let (available, source) = match probe.free_bytes() {
        Some(free) => (
            free.saturating_sub(free / 10),
            "free_bytes minus 10% headroom",
        ),
        None => (
            probe_halving(model_f32_bytes, probe),
            "halving try_alloc probe",
        ),
    };

    for format in ResidentFormat::ALL {
        let needed = format.bytes_for(model_f32_bytes);
        if needed <= available && probe.try_alloc(needed) {
            return WeightPlan {
                format,
                budget_bytes: needed,
                resident: true,
                reason: format!(
                    "{format:?} fits ({needed} bytes needed, {available} bytes available via {source})"
                ),
            };
        }
    }

    let q4_bytes = ResidentFormat::Q4_0.bytes_for(model_f32_bytes);
    WeightPlan {
        format: ResidentFormat::Q4_0,
        budget_bytes: 0,
        resident: false,
        reason: format!(
            "cannot keep weights resident (re-upload every window): model F32 size is {model_f32_bytes} bytes, smallest Q4_0 size is {q4_bytes} bytes, available budget is {available} bytes ({source})"
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct FakeProbe {
        free: Option<u64>,
        alloc_ok_up_to: u64,
    }

    impl MemoryProbe for FakeProbe {
        fn free_bytes(&self) -> Option<u64> {
            self.free
        }

        fn try_alloc(&self, bytes: u64) -> bool {
            bytes <= self.alloc_ok_up_to
        }
    }

    #[test]
    fn plenty_of_memory_keeps_f32() {
        let probe = FakeProbe {
            free: Some(200_000),
            alloc_ok_up_to: 200_000,
        };
        let plan = plan_weight_budget(128_000, &probe);
        assert_eq!(plan.format, ResidentFormat::F32);
        assert_eq!(plan.budget_bytes, 128_000);
        assert!(plan.resident);
    }

    #[test]
    fn f32_does_not_fit_falls_back_to_f16() {
        let probe = FakeProbe {
            free: Some(130_000),
            alloc_ok_up_to: 130_000,
        };
        let plan = plan_weight_budget(128_000, &probe);
        assert_eq!(plan.format, ResidentFormat::F16);
        assert_eq!(plan.budget_bytes, 64_000);
        assert!(plan.resident);
    }

    #[test]
    fn only_q4_0_fits() {
        let probe = FakeProbe {
            free: Some(30_000),
            alloc_ok_up_to: 30_000,
        };
        let plan = plan_weight_budget(128_000, &probe);
        assert_eq!(plan.format, ResidentFormat::Q4_0);
        assert_eq!(plan.budget_bytes, 18_000);
        assert!(plan.resident);
    }

    #[test]
    fn nothing_fits_returns_non_resident_with_sizes_in_reason() {
        let probe = FakeProbe {
            free: Some(10_000),
            alloc_ok_up_to: 10_000,
        };
        let plan = plan_weight_budget(128_000, &probe);
        assert!(!plan.resident);
        assert_eq!(plan.format, ResidentFormat::Q4_0);
        assert_eq!(plan.budget_bytes, 0);
        assert!(plan.reason.contains("128000"));
        assert!(plan.reason.contains("18000"));
        assert!(plan.reason.contains("9000"));
    }

    #[test]
    fn free_bytes_none_falls_back_to_try_alloc_probing() {
        let f16_probe = FakeProbe {
            free: None,
            alloc_ok_up_to: 70_000,
        };
        let f16_plan = plan_weight_budget(128_000, &f16_probe);
        assert_eq!(f16_plan.format, ResidentFormat::F16);
        assert_eq!(f16_plan.budget_bytes, 64_000);
        assert!(f16_plan.resident);

        let q4_probe = FakeProbe {
            free: None,
            alloc_ok_up_to: 20_000,
        };
        let q4_plan = plan_weight_budget(128_000, &q4_probe);
        assert_eq!(q4_plan.format, ResidentFormat::Q4_0);
        assert_eq!(q4_plan.budget_bytes, 18_000);
        assert!(q4_plan.resident);

        let none_probe = FakeProbe {
            free: None,
            alloc_ok_up_to: 10_000,
        };
        let none_plan = plan_weight_budget(128_000, &none_probe);
        assert!(!none_plan.resident);
        assert_eq!(none_plan.format, ResidentFormat::Q4_0);
        assert_eq!(none_plan.budget_bytes, 0);
    }

    #[test]
    fn try_alloc_failure_moves_to_next_format() {
        let probe = FakeProbe {
            free: Some(200_000),
            alloc_ok_up_to: 40_000,
        };
        let plan = plan_weight_budget(128_000, &probe);
        assert_eq!(plan.format, ResidentFormat::Q8_0);
        assert_eq!(plan.budget_bytes, 34_000);
        assert!(plan.resident);
    }
}
