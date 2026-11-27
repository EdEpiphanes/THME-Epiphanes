// SPDX-License-Identifier: AGPL-3.0-only
//! Pure budget-shrinking planner for run settings (THME-3058).
//!
//! Reduces [`RunSettings`] in a fixed decision order until [`run_memory`] fits within a byte
//! budget, respecting user-fixed flags and a context floor.

use serde::{Deserialize, Serialize};
use std::fmt;

use super::run_memory::{run_memory, KvType, ModelShape};

const WEIGHT_LEVELS: [f64; 4] = [4.0, 2.0, 1.0625, 0.5625];

/// Runtime settings adjusted by [`shrink_to_budget`].
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RunSettings {
    pub context: usize,
    pub kv: KvType,
    pub weight_bytes_per_param: f64,
}

/// Flags marking user-locked settings that [`shrink_to_budget`] must leave unchanged.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct Fixed {
    pub context: bool,
    pub kv: bool,
    pub weights: bool,
}

/// Lower bounds enforced during shrinking.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShrinkLimits {
    pub context_floor: usize,
}

/// Error returned when [`shrink_to_budget`] cannot fit within the requested byte budget.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ShrinkError {
    DoesNotFit {
        needed: u64,
        budget: u64,
        tried: Vec<String>,
    },
}

impl fmt::Display for ShrinkError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DoesNotFit {
                needed,
                budget,
                tried,
            } => write!(
                f,
                "needed {needed} bytes, exceeds budget {budget} bytes (tried {} step(s))",
                tried.len()
            ),
        }
    }
}

impl std::error::Error for ShrinkError {}

fn measure(shape: &ModelShape, settings: RunSettings) -> u64 {
    run_memory(
        shape,
        settings.weight_bytes_per_param,
        settings.context,
        settings.kv,
    )
    .total()
}

fn next_kv(kv: KvType) -> Option<KvType> {
    match kv {
        KvType::F32 => Some(KvType::F16),
        KvType::F16 => Some(KvType::Int8),
        KvType::Int8 => Some(KvType::Int4),
        KvType::Int4 => None,
    }
}

fn next_weight_bytes_per_param(current: f64) -> Option<f64> {
    WEIGHT_LEVELS.iter().copied().find(|&level| level < current)
}

fn format_attempt(label: &str, settings: RunSettings, total: u64) -> String {
    format!(
        "{label}: kv={:?}, context={}, weight_bytes_per_param={} -> {total} bytes",
        settings.kv, settings.context, settings.weight_bytes_per_param
    )
}

/// Shrinks `start` settings until [`run_memory`] total bytes is at most `budget_bytes`.
///
/// Decision order:
/// 1. If `run_memory(shape, start.weight_bytes_per_param, start.context, start.kv).total() <= budget_bytes`,
///    return `start` unchanged.
/// 2. Otherwise, when `fixed.kv` is `false`, reduce `kv` one step at a time along
///    `F32 -> F16 -> Int8 -> Int4`.
/// 3. Then, when `fixed.context` is `false`, halve `context` (rounded down, never below
///    `limits.context_floor`; if `start.context` itself is below `limits.context_floor`, it is
///    kept as-is and never raised).
/// 4. Last, when `fixed.weights` is `false`, step `weight_bytes_per_param` down the list
///    `[4.0, 2.0, 1.0625, 0.5625]` one entry at a time (if the current value is not on the list,
///    use the next lower list entry).
///
/// Stops as soon as `run_memory(...).total() <= budget_bytes`.
///
/// # Errors
///
/// Returns [`ShrinkError::DoesNotFit`] with the final `needed` bytes and a list of what was tried
/// if nothing more can change and the total still exceeds `budget_bytes`.
pub fn shrink_to_budget(
    shape: &ModelShape,
    start: RunSettings,
    fixed: Fixed,
    limits: ShrinkLimits,
    budget_bytes: u64,
) -> Result<RunSettings, ShrinkError> {
    let mut current = start;
    let mut current_total = measure(shape, current);

    if current_total <= budget_bytes {
        return Ok(start);
    }

    let mut tried = vec![format_attempt("start", current, current_total)];

    if !fixed.kv {
        while let Some(kv) = next_kv(current.kv) {
            current.kv = kv;
            current_total = measure(shape, current);
            tried.push(format_attempt("kv", current, current_total));
            if current_total <= budget_bytes {
                return Ok(current);
            }
        }
    }

    if !fixed.context {
        let context_floor = start.context.min(limits.context_floor);
        loop {
            let halved = (current.context / 2).max(context_floor);
            if halved >= current.context {
                break;
            }
            current.context = halved;
            current_total = measure(shape, current);
            tried.push(format_attempt("context", current, current_total));
            if current_total <= budget_bytes {
                return Ok(current);
            }
        }
    }

    if !fixed.weights {
        while let Some(weight_bytes) = next_weight_bytes_per_param(current.weight_bytes_per_param) {
            current.weight_bytes_per_param = weight_bytes;
            current_total = measure(shape, current);
            tried.push(format_attempt("weights", current, current_total));
            if current_total <= budget_bytes {
                return Ok(current);
            }
        }
    }

    Err(ShrinkError::DoesNotFit {
        needed: current_total,
        budget: budget_bytes,
        tried,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_llama() -> ModelShape {
        ModelShape {
            layers: 2,
            hidden: 8,
            intermediate: 16,
            heads: 2,
            kv_heads: 1,
            vocab: 10,
            gated_mlp: true,
            tied_embeddings: false,
        }
    }

    #[test]
    fn tiny_llama_budget_6024_returns_start_unchanged() {
        // Tiny LLaMA: 1352 params * 4.0 = 5408 weight bytes.
        // context=5, head_dim=4, kv_dim=4:
        //   F32 KV row = 16 B -> kv = 2 * 2 * 5 * 16 = 320 B.
        //   scratch = (8*4 + 16*2 + 5*2) * 4 = 74 * 4 = 296 B.
        //   total = 5408 + 320 + 296 = 6024 B.
        let shape = tiny_llama();
        let start = RunSettings {
            context: 5,
            kv: KvType::F32,
            weight_bytes_per_param: 4.0,
        };
        let limits = ShrinkLimits { context_floor: 1 };
        let res = shrink_to_budget(&shape, start, Fixed::default(), limits, 6024);
        assert_eq!(res, Ok(start));
    }

    #[test]
    fn tiny_llama_budget_between_f16_and_f32_picks_f16() {
        // F32 KV at context=5, 4.0 bpp: 5408 + 320 + 296 = 6024 B (exceeds 5900).
        // Step 1 reduces KV from F32 to F16:
        //   F16 KV row = 4 * 2 = 8 B -> kv = 2 * 2 * 5 * 8 = 160 B.
        //   total = 5408 + 160 + 296 = 5864 B <= 5900 B -> stops at F16.
        let shape = tiny_llama();
        let start = RunSettings {
            context: 5,
            kv: KvType::F32,
            weight_bytes_per_param: 4.0,
        };
        let limits = ShrinkLimits { context_floor: 1 };
        let res = shrink_to_budget(&shape, start, Fixed::default(), limits, 5900);
        let expected = RunSettings {
            context: 5,
            kv: KvType::F16,
            weight_bytes_per_param: 4.0,
        };
        assert_eq!(res, Ok(expected));
    }

    #[test]
    fn tiny_llama_below_weights_with_weights_fixed_returns_does_not_fit() {
        // Weights at 4.0 bpp = 1352 * 4 = 5408 B (fixed.weights = true).
        // KV steps F32(6024) -> F16(5864) -> Int8(5864) -> Int4(24*5=120 -> 5824).
        // Context halves 5 -> 2 (Int4 kv=48, scratch=(64+4)*4=272 -> 5408+48+272=5728)
        //   -> 1 (Int4 kv=24, scratch=(64+2)*4=264 -> 5408+24+264=5696), stops at floor=1.
        // Budget 5000 < 5408 (and < 5696), so DoesNotFit { needed: 5696, budget: 5000 }.
        let shape = tiny_llama();
        let start = RunSettings {
            context: 5,
            kv: KvType::F32,
            weight_bytes_per_param: 4.0,
        };
        let fixed = Fixed {
            context: false,
            kv: false,
            weights: true,
        };
        let limits = ShrinkLimits { context_floor: 1 };
        let res = shrink_to_budget(&shape, start, fixed, limits, 5000);
        match res {
            Err(ShrinkError::DoesNotFit {
                needed,
                budget,
                tried,
            }) => {
                assert_eq!(needed, 5696);
                assert_eq!(budget, 5000);
                // start + 3 KV steps + 2 context steps = 6 entries
                assert_eq!(tried.len(), 6);
            }
            Ok(other) => panic!("expected DoesNotFit, got {other:?}"),
        }
    }

    #[test]
    fn tiny_llama_fixed_kv_leaves_kv_unchanged() {
        // F32 KV is fixed (16 B/row -> kv = 64 * context).
        // At context=5: 5408 + 320 + 296 = 6024 B (> 5800).
        // Context halves 5 -> 2: 5408 + 128 + 272 = 5808 B (> 5800).
        // Context halves 2 -> 1: 5408 + 64 + 264 = 5736 B (<= 5800).
        let shape = tiny_llama();
        let start = RunSettings {
            context: 5,
            kv: KvType::F32,
            weight_bytes_per_param: 4.0,
        };
        let fixed = Fixed {
            context: false,
            kv: true,
            weights: false,
        };
        let limits = ShrinkLimits { context_floor: 1 };
        let res = shrink_to_budget(&shape, start, fixed, limits, 5800);
        let expected = RunSettings {
            context: 1,
            kv: KvType::F32,
            weight_bytes_per_param: 4.0,
        };
        assert_eq!(res, Ok(expected));
    }

    #[test]
    fn tiny_llama_start_context_below_floor_is_never_raised_and_unlisted_weights_step_down() {
        // start.context=2 < context_floor=4: context must remain 2 (neither raised nor lowered).
        // start.weight_bytes_per_param=3.0 (not in [4.0, 2.0, 1.0625, 0.5625]): next lower is 2.0.
        // At Int4 KV, context=2: kv=48, scratch=272.
        //   At 3.0 bpp: weights = 1352 * 3 = 4056 -> total = 4056 + 48 + 272 = 4376 (> 3100).
        //   At 2.0 bpp: weights = 1352 * 2 = 2704 -> total = 2704 + 48 + 272 = 3024 (<= 3100).
        let shape = tiny_llama();
        let start = RunSettings {
            context: 2,
            kv: KvType::Int4,
            weight_bytes_per_param: 3.0,
        };
        let limits = ShrinkLimits { context_floor: 4 };
        let res = shrink_to_budget(&shape, start, Fixed::default(), limits, 3100);
        let expected = RunSettings {
            context: 2,
            kv: KvType::Int4,
            weight_bytes_per_param: 2.0,
        };
        assert_eq!(res, Ok(expected));
    }

    #[test]
    fn tiny_llama_weights_step_to_1_0625_and_0_5625_match_literals() {
        // At Int4 KV, context=1: kv = 24, scratch = 264 (sum = 288).
        //   2.0 bpp:    weights = 2704 -> total = 2992.
        //   1.0625 bpp: weights = ceil(1352 * 1.0625) = ceil(1436.5) = 1437 -> total = 1725.
        //   0.5625 bpp: weights = ceil(1352 * 0.5625) = ceil(760.5) = 761 -> total = 1049.
        let shape = tiny_llama();
        let start = RunSettings {
            context: 1,
            kv: KvType::Int4,
            weight_bytes_per_param: 2.0,
        };
        let limits = ShrinkLimits { context_floor: 1 };

        let res_10625 = shrink_to_budget(&shape, start, Fixed::default(), limits, 1725);
        let expected_10625 = RunSettings {
            context: 1,
            kv: KvType::Int4,
            weight_bytes_per_param: 1.0625,
        };
        assert_eq!(res_10625, Ok(expected_10625));

        let res_05625 = shrink_to_budget(&shape, start, Fixed::default(), limits, 1049);
        let expected_05625 = RunSettings {
            context: 1,
            kv: KvType::Int4,
            weight_bytes_per_param: 0.5625,
        };
        assert_eq!(res_05625, Ok(expected_05625));

        let res_fail = shrink_to_budget(&shape, start, Fixed::default(), limits, 1048);
        match res_fail {
            Err(ShrinkError::DoesNotFit {
                needed,
                budget,
                tried,
            }) => {
                assert_eq!(needed, 1049);
                assert_eq!(budget, 1048);
                assert_eq!(tried.len(), 3);
            }
            Ok(other) => panic!("expected DoesNotFit, got {other:?}"),
        }
    }

    struct Lcg(u64);

    impl Lcg {
        fn next_u64(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            self.0
        }

        fn gen_range(&mut self, upper: usize) -> usize {
            (self.next_u64() % (upper as u64)) as usize
        }
    }

    #[test]
    fn property_500_lcg_cases_preserve_invariants() {
        const KVS: [KvType; 4] = [KvType::F32, KvType::F16, KvType::Int8, KvType::Int4];
        const WEIGHTS: [f64; 9] = [8.0, 4.0, 3.0, 2.0, 1.5, 1.0625, 0.75, 0.5625, 0.25];

        let mut rng = Lcg(0x5EED_3058_CAFE_BABE);

        for case_idx in 0..500 {
            let heads = 1 + rng.gen_range(8);
            let kv_heads = 1 + rng.gen_range(heads);
            let head_dim = 1 + rng.gen_range(40);
            let shape = ModelShape {
                layers: 1 + rng.gen_range(12),
                hidden: heads * head_dim,
                intermediate: 4 + rng.gen_range(64),
                heads,
                kv_heads,
                vocab: 4 + rng.gen_range(64),
                gated_mlp: (rng.next_u64() & 1) == 1,
                tied_embeddings: (rng.next_u64() & 1) == 1,
            };

            let start = RunSettings {
                context: rng.gen_range(2048),
                kv: KVS[rng.gen_range(KVS.len())],
                weight_bytes_per_param: WEIGHTS[rng.gen_range(WEIGHTS.len())],
            };

            let fixed_bits = rng.next_u64();
            let fixed = Fixed {
                context: (fixed_bits & 1) != 0,
                kv: (fixed_bits & 2) != 0,
                weights: (fixed_bits & 4) != 0,
            };

            let limits = ShrinkLimits {
                context_floor: rng.gen_range(512),
            };

            let start_total = measure(&shape, start);
            let budget_bytes = match rng.gen_range(5) {
                0 => 0,
                1 => start_total / 4,
                2 => start_total / 2,
                3 => start_total.saturating_add(1),
                _ => rng.next_u64() % start_total.saturating_mul(2).max(1),
            };

            let min_allowed_context = start.context.min(limits.context_floor);

            match shrink_to_budget(&shape, start, fixed, limits, budget_bytes) {
                Ok(res) => {
                    let res_total = measure(&shape, res);
                    assert!(
                        res_total <= budget_bytes,
                        "case {case_idx}: total {res_total} exceeded budget {budget_bytes}"
                    );
                    assert!(
                        res.context <= start.context,
                        "case {case_idx}: context {} exceeded start {}",
                        res.context,
                        start.context
                    );
                    assert!(
                        res.context >= min_allowed_context,
                        "case {case_idx}: context {} fell below floor {}",
                        res.context,
                        min_allowed_context
                    );
                    if fixed.context {
                        assert_eq!(res.context, start.context, "case {case_idx}: context");
                    }
                    if fixed.kv {
                        assert_eq!(res.kv, start.kv, "case {case_idx}: kv");
                    }
                    if fixed.weights {
                        assert_eq!(
                            res.weight_bytes_per_param.to_bits(),
                            start.weight_bytes_per_param.to_bits(),
                            "case {case_idx}: weights"
                        );
                    }
                }
                Err(ShrinkError::DoesNotFit {
                    needed,
                    budget,
                    tried,
                }) => {
                    assert_eq!(budget, budget_bytes, "case {case_idx}: budget mismatch");
                    assert!(
                        needed > budget_bytes,
                        "case {case_idx}: needed {needed} <= budget {budget_bytes}"
                    );
                    assert!(!tried.is_empty(), "case {case_idx}: tried list was empty");
                }
            }
        }
    }
}
