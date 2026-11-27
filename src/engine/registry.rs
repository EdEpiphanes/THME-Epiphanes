// SPDX-License-Identifier: AGPL-3.0-only
//! Typed conversion registry with accumulated error bounds (roadmap v0.2.5.9, dossier WP10).
//!
//! Values move between typed number formats only through registered conversions. Every
//! conversion either widens exactly (to F32) or rounds once to its target (through
//! [`super::minifloat`], the one rounding routine). A value that is not already a value of its
//! source type is refused, so a conversion never hides a rounding it did not declare.
//!
//! A chain of conversions accumulates its bound. With `E_0 = 0` and `E_(i+1) = E_i +
//! b_i(|x| + E_i)`, where `b_i` is the monotone bound of step `i`, the final error against the
//! exact input is at most `E_n`. Each step's bound is evaluated at the largest magnitude the
//! intermediate value can have, so the total is an upper bound, not an estimate.

use super::error::{ensure_finite, EngineError, EngineResult};
use super::minifloat::{Format, BF16, FP16, FP8_E4M3, FP8_E5M2};

/// A number type the registry knows. `F32` is the carrier every conversion widens into; the
/// others are the minifloat formats of [`super::minifloat`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Dtype {
    F32,
    F16,
    Bf16,
    Fp8E4M3,
    Fp8E5M2,
}

impl Dtype {
    pub const ALL: [Dtype; 5] = [
        Dtype::F32,
        Dtype::F16,
        Dtype::Bf16,
        Dtype::Fp8E4M3,
        Dtype::Fp8E5M2,
    ];

    /// The minifloat layout of this type; `None` for F32.
    pub fn format(self) -> Option<Format> {
        match self {
            Dtype::F32 => None,
            Dtype::F16 => Some(FP16),
            Dtype::Bf16 => Some(BF16),
            Dtype::Fp8E4M3 => Some(FP8_E4M3),
            Dtype::Fp8E5M2 => Some(FP8_E5M2),
        }
    }

    pub fn name(self) -> &'static str {
        self.format().map_or("F32", |f| f.name)
    }
}

/// One registered step between two types of different kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Conversion {
    pub from: Dtype,
    pub to: Dtype,
}

impl Conversion {
    /// Worst-case absolute error of this step for an input of magnitude at most `magnitude`.
    /// Non-decreasing in `magnitude`, which the chain accumulation relies on.
    fn bound(self, magnitude: f64) -> f64 {
        self.to.format().map_or(0.0, |f| f.error_bound(magnitude))
    }

    /// Applies the step. The input must already be a value of `from`; a finite result that
    /// overflows the target is an error, never a silent infinity.
    fn apply(self, x: f32) -> EngineResult<f32> {
        ensure_finite("conversion input", &[f64::from(x)])?;
        if let Some(f) = self.from.format() {
            if f.round(x) != x {
                return Err(EngineError::InvalidInput(format!(
                    "{x} is not a value of {}",
                    f.name
                )));
            }
        }
        match self.to.format() {
            None => Ok(x),
            Some(f) => {
                let y = f.round(x);
                if y.is_finite() {
                    Ok(y)
                } else {
                    Err(EngineError::Infeasible(format!("{x} overflows {}", f.name)))
                }
            }
        }
    }
}

/// A converted value and the worst-case distance from the exact input it may have.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Converted {
    pub value: f32,
    pub error_bound: f64,
}

/// The registered conversions, keyed by typed source and target.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ConversionRegistry {
    entries: Vec<Conversion>,
}

impl ConversionRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Every ordered pair of distinct types, each registered as one step.
    pub fn standard() -> Self {
        let mut entries = Vec::new();
        for from in Dtype::ALL {
            for to in Dtype::ALL {
                if from != to {
                    entries.push(Conversion { from, to });
                }
            }
        }
        Self { entries }
    }

    /// Adds a step. Refuses the identity and a second step for the same pair.
    pub fn register(&mut self, from: Dtype, to: Dtype) -> EngineResult<()> {
        if from == to {
            return Err(EngineError::InvalidInput(format!(
                "{} to {} is the identity, not a conversion",
                from.name(),
                to.name()
            )));
        }
        if self.get(from, to).is_some() {
            return Err(EngineError::InvalidInput(format!(
                "a conversion from {} to {} is already registered",
                from.name(),
                to.name()
            )));
        }
        self.entries.push(Conversion { from, to });
        Ok(())
    }

    pub fn get(&self, from: Dtype, to: Dtype) -> Option<Conversion> {
        self.entries
            .iter()
            .copied()
            .find(|c| c.from == from && c.to == to)
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// One registered step from `from` to `to`.
    pub fn convert(&self, from: Dtype, to: Dtype, x: f32) -> EngineResult<Converted> {
        self.convert_path(&[from, to], x)
    }

    /// Walks a path of types (at least two), each consecutive pair a registered step, and
    /// returns the final value with its accumulated bound. `x` must be a value of `path[0]`.
    pub fn convert_path(&self, path: &[Dtype], x: f32) -> EngineResult<Converted> {
        if path.len() < 2 {
            return Err(EngineError::InvalidInput(
                "a conversion path needs at least two types".to_string(),
            ));
        }
        ensure_finite("conversion input", &[f64::from(x)])?;
        let magnitude = f64::from(x).abs();
        let mut value = x;
        let mut bound = 0.0;
        for pair in path.windows(2) {
            let step = self.get(pair[0], pair[1]).ok_or_else(|| {
                EngineError::InvalidInput(format!(
                    "no registered conversion from {} to {}",
                    pair[0].name(),
                    pair[1].name()
                ))
            })?;
            bound += step.bound(magnitude + bound);
            value = step.apply(value)?;
        }
        Ok(Converted {
            value,
            error_bound: bound,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seeded(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                // Log-uniform magnitude in [2^-6, 2^6) with a random sign.
                let u = (s >> 11) as f64 / 9_007_199_254_740_992.0;
                let mag = (2f64.powf(-6.0 + 12.0 * u)) as f32;
                if s & 1 == 0 {
                    mag
                } else {
                    -mag
                }
            })
            .collect()
    }

    #[test]
    fn standard_registry_has_every_ordered_pair_of_distinct_types() {
        let reg = ConversionRegistry::standard();
        assert_eq!(reg.len(), 5 * 4);
        for from in Dtype::ALL {
            for to in Dtype::ALL {
                assert_eq!(
                    reg.get(from, to).is_some(),
                    from != to,
                    "{from:?} to {to:?}"
                );
            }
        }
    }

    #[test]
    fn register_refuses_duplicates_and_identity() {
        let mut reg = ConversionRegistry::new();
        assert!(reg.register(Dtype::F32, Dtype::F32).is_err());
        reg.register(Dtype::F32, Dtype::F16).unwrap();
        assert!(matches!(
            reg.register(Dtype::F32, Dtype::F16),
            Err(EngineError::InvalidInput(_))
        ));
        assert_eq!(reg.len(), 1);
    }

    /// Hand-worked: rounding 1.0 to FP16 has a bound of 2^-11 (half a unit of the 10-bit
    /// mantissa at 1.0). The value 1.0 is exact.
    #[test]
    fn one_rounding_step_by_hand() {
        let reg = ConversionRegistry::standard();
        let got = reg.convert(Dtype::F32, Dtype::F16, 1.0).unwrap();
        assert_eq!(got.value, 1.0);
        assert_eq!(got.error_bound, 2f64.powi(-11));
    }

    #[test]
    fn widening_to_f32_is_exact_with_zero_bound() {
        let reg = ConversionRegistry::standard();
        for (from, x) in [
            (Dtype::F16, 0.5f32),
            (Dtype::Bf16, 1.5),
            (Dtype::Fp8E4M3, -3.0),
        ] {
            let got = reg.convert(from, Dtype::F32, x).unwrap();
            assert_eq!(got.value, x);
            assert_eq!(got.error_bound, 0.0);
        }
    }

    #[test]
    fn input_must_already_be_a_value_of_its_source_type() {
        let reg = ConversionRegistry::standard();
        // 0.1 is not an FP16 value, so a conversion from FP16 must refuse it.
        assert!(matches!(
            reg.convert(Dtype::F16, Dtype::F32, 0.1),
            Err(EngineError::InvalidInput(_))
        ));
    }

    #[test]
    fn overflow_and_non_finite_inputs_are_errors() {
        let reg = ConversionRegistry::standard();
        // FP16 overflows to infinity above 65504; E4M3 has no infinity and its maximum is 448.
        assert!(matches!(
            reg.convert(Dtype::F32, Dtype::F16, 70_000.0),
            Err(EngineError::Infeasible(_))
        ));
        assert!(matches!(
            reg.convert(Dtype::F32, Dtype::Fp8E4M3, 1000.0),
            Err(EngineError::Infeasible(_))
        ));
        assert!(matches!(
            reg.convert(Dtype::F32, Dtype::F16, f32::NAN),
            Err(EngineError::NonFinite { .. })
        ));
        assert!(reg.convert(Dtype::F32, Dtype::Bf16, f32::INFINITY).is_err());
    }

    /// Hand-worked chain F32 -> BF16 -> FP8 E4M3 on x = 1.0. Step one: 2^-8. Step two is
    /// evaluated at magnitude 1 + 2^-8: 2^-4 (1 + 2^-8). Total 2^-8 + 2^-4 + 2^-12, which is
    /// 0.066650390625, against 0.0625 for the single rounding F32 -> FP8 E4M3.
    #[test]
    fn chain_bounds_accumulate_by_hand() {
        let reg = ConversionRegistry::standard();
        let path = [Dtype::F32, Dtype::Bf16, Dtype::Fp8E4M3];
        let got = reg.convert_path(&path, 1.0).unwrap();
        assert_eq!(got.value, 1.0);
        assert_eq!(got.error_bound, 0.066_650_390_625);
        let direct = reg.convert(Dtype::F32, Dtype::Fp8E4M3, 1.0).unwrap();
        assert_eq!(direct.error_bound, 0.0625);
        assert!(got.error_bound > direct.error_bound);
    }

    /// Oracle: for each random input, the chained value differs from the input by no more
    /// than the accumulated bound. The path rounds, widens, and rounds again.
    #[test]
    fn chain_bound_holds_on_random_inputs() {
        let reg = ConversionRegistry::standard();
        let path = [
            Dtype::F32,
            Dtype::F16,
            Dtype::F32,
            Dtype::Bf16,
            Dtype::F32,
            Dtype::Fp8E4M3,
        ];
        for x in seeded(2000, 77) {
            let got = reg.convert_path(&path, x).unwrap();
            assert!(got.error_bound.is_finite());
            let actual = f64::from(got.value - x).abs();
            assert!(
                actual <= got.error_bound,
                "x={x} value={} actual={actual} bound={}",
                got.value,
                got.error_bound
            );
        }
    }

    #[test]
    fn paths_need_registered_steps_and_at_least_two_types() {
        let reg = ConversionRegistry::standard();
        assert!(matches!(
            reg.convert_path(&[Dtype::F32], 1.0),
            Err(EngineError::InvalidInput(_))
        ));
        let empty = ConversionRegistry::new();
        assert!(matches!(
            empty.convert(Dtype::F32, Dtype::F16, 1.0),
            Err(EngineError::InvalidInput(_))
        ));
        assert!(empty.convert_path(&[Dtype::F32, Dtype::F16], 1.0).is_err());
    }
}
