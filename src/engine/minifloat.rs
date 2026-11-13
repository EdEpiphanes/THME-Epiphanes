// SPDX-License-Identifier: AGPL-3.0-only
//! Small floating-point formats (dossier WP3): FP16, BF16, FP8 E4M3 and FP8 E5M2.
//!
//! One rounding routine serves every format: round to nearest, ties to even, the IEEE 754
//! default. Every other place in THME that stores a half-precision value calls this module.
//!
//! **Error bound.** For a finite input `x` whose rounded value is finite, the error is at most
//! half a unit in the last place: `|x - round(x)| <= max(2^-(M+1) * |x|, 2^(emin-M-1))`, where `M`
//! is the number of mantissa bits and `emin = 1 - bias`. [`Format::error_bound`] returns it.
//!
//! Overflow: formats with infinities (FP16, BF16, E5M2) round to infinity; E4M3, which has no
//! infinity (OCP FP8 "FN" variant), returns its NaN code. NaN stays NaN.

/// A binary floating-point format with `exp_bits` exponent and `man_bits` mantissa bits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Format {
    pub name: &'static str,
    pub exp_bits: u32,
    pub man_bits: u32,
    /// Infinity exists (exponent field all ones, mantissa zero).
    pub has_inf: bool,
}

pub const FP16: Format = Format {
    name: "FP16",
    exp_bits: 5,
    man_bits: 10,
    has_inf: true,
};
pub const BF16: Format = Format {
    name: "BF16",
    exp_bits: 8,
    man_bits: 7,
    has_inf: true,
};
/// OCP FP8 E4M3 ("E4M3FN"): no infinities; only all-ones exponent and mantissa is NaN.
pub const FP8_E4M3: Format = Format {
    name: "FP8-E4M3",
    exp_bits: 4,
    man_bits: 3,
    has_inf: false,
};
pub const FP8_E5M2: Format = Format {
    name: "FP8-E5M2",
    exp_bits: 5,
    man_bits: 2,
    has_inf: true,
};

impl Format {
    fn bias(self) -> i32 {
        (1 << (self.exp_bits - 1)) - 1
    }

    fn emin(self) -> i32 {
        1 - self.bias()
    }

    fn sign_shift(self) -> u32 {
        self.exp_bits + self.man_bits
    }

    fn exp_mask(self) -> u32 {
        (1 << self.exp_bits) - 1
    }

    fn man_mask(self) -> u32 {
        (1 << self.man_bits) - 1
    }

    /// Largest finite value.
    pub fn max_finite(self) -> f64 {
        let top_exp = if self.has_inf {
            self.exp_mask() as i32 - 1
        } else {
            self.exp_mask() as i32
        };
        // Without infinities the all-ones mantissa at the top exponent is NaN.
        let top_man = if self.has_inf {
            self.man_mask()
        } else {
            self.man_mask() - 1
        };
        (1.0 + f64::from(top_man) / f64::from(1u32 << self.man_bits))
            * 2f64.powi(top_exp - self.bias())
    }

    /// Bit pattern of a quiet NaN.
    pub fn nan(self) -> u32 {
        if self.has_inf {
            (self.exp_mask() << self.man_bits) | (1 << (self.man_bits - 1))
        } else {
            (self.exp_mask() << self.man_bits) | self.man_mask()
        }
    }

    /// The worst-case rounding error for an input of magnitude `x` (section bound above).
    pub fn error_bound(self, x: f64) -> f64 {
        let relative = 2f64.powi(-(self.man_bits as i32) - 1) * x.abs();
        let absolute = 2f64.powi(self.emin() - self.man_bits as i32 - 1);
        relative.max(absolute)
    }

    /// Rounds `x` to this format (nearest, ties to even) and returns the bit pattern.
    pub fn encode(self, x: f32) -> u32 {
        let sign = u32::from(x.is_sign_negative()) << self.sign_shift();
        if x.is_nan() {
            return self.nan();
        }
        let a = f64::from(x.abs());
        if a == 0.0 {
            return sign;
        }
        if a.is_infinite() {
            return if self.has_inf {
                sign | (self.exp_mask() << self.man_bits)
            } else {
                sign | self.nan()
            };
        }
        // Exponent of `a` from its f64 bits (exact; every f32 is a normal f64).
        let e = ((a.to_bits() >> 52) & 0x7ff) as i32 - 1023;
        let e = e.max(self.emin());
        let quantum = 2f64.powi(e - self.man_bits as i32);
        let rounded = (a / quantum).round_ties_even() * quantum;
        if rounded > self.max_finite() {
            return if self.has_inf {
                sign | (self.exp_mask() << self.man_bits)
            } else {
                sign | self.nan()
            };
        }
        let min_normal = 2f64.powi(self.emin());
        if rounded < min_normal {
            let man = (rounded / 2f64.powi(self.emin() - self.man_bits as i32)) as u32;
            return sign | man;
        }
        let e = ((rounded.to_bits() >> 52) & 0x7ff) as i32 - 1023;
        let man = ((rounded / 2f64.powi(e) - 1.0) * f64::from(1u32 << self.man_bits)) as u32;
        sign | (((e + self.bias()) as u32) << self.man_bits) | man
    }

    /// The value of a bit pattern.
    pub fn decode(self, bits: u32) -> f32 {
        let negative = (bits >> self.sign_shift()) & 1 == 1;
        let exp = (bits >> self.man_bits) & self.exp_mask();
        let man = bits & self.man_mask();
        let scale = f64::from(1u32 << self.man_bits);
        let magnitude = if exp == self.exp_mask() && self.has_inf {
            if man == 0 {
                f64::INFINITY
            } else {
                f64::NAN
            }
        } else if exp == self.exp_mask() && !self.has_inf && man == self.man_mask() {
            f64::NAN
        } else if exp == 0 {
            f64::from(man) / scale * 2f64.powi(self.emin())
        } else {
            (1.0 + f64::from(man) / scale) * 2f64.powi(exp as i32 - self.bias())
        };
        let value = if negative { -magnitude } else { magnitude };
        value as f32
    }

    /// `x` after a round trip through this format.
    pub fn round(self, x: f32) -> f32 {
        self.decode(self.encode(x))
    }
}

/// IEEE half precision: f32 to bits (nearest, ties to even).
pub fn f32_to_f16(x: f32) -> u16 {
    FP16.encode(x) as u16
}

/// IEEE half precision: bits to f32 (exact).
pub fn f16_to_f32(bits: u16) -> f32 {
    FP16.decode(u32::from(bits))
}

/// Brain float 16: f32 to bits (nearest, ties to even; NaN stays NaN).
pub fn f32_to_bf16(x: f32) -> u16 {
    BF16.encode(x) as u16
}

/// Brain float 16: bits to f32 (exact: the upper half of an f32).
pub fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Oracle 1: every finite FP16 value survives decode then encode unchanged (65,536 codes).
    #[test]
    fn every_fp16_code_round_trips() {
        for bits in 0u32..=0xffff {
            let v = FP16.decode(bits);
            if v.is_nan() {
                assert!(FP16.decode(FP16.encode(v)).is_nan());
                continue;
            }
            assert_eq!(FP16.encode(v), bits, "code {bits:#06x} value {v}");
        }
    }

    /// Oracle 2: the exact midpoint between two neighbours rounds to the even one, and a value
    /// a hair either side rounds to the nearer one. Together with oracle 1 this pins down
    /// round-to-nearest-even completely.
    fn check_midpoints(format: Format, limit: u32) {
        for bits in 0..limit {
            let lo = format.decode(bits);
            let hi = format.decode(bits + 1);
            if !lo.is_finite() || !hi.is_finite() || hi <= lo {
                continue;
            }
            let mid = (f64::from(lo) + f64::from(hi)) / 2.0;
            let mid32 = mid as f32;
            if f64::from(mid32) != mid {
                continue; // the midpoint is not an f32
            }
            let even = if bits % 2 == 0 { bits } else { bits + 1 };
            assert_eq!(format.encode(mid32), even, "{} tie at {mid}", format.name);
            assert_eq!(
                format.encode(mid32.next_down()),
                bits,
                "{} below {mid}",
                format.name
            );
            assert_eq!(
                format.encode(mid32.next_up()),
                bits + 1,
                "{} above {mid}",
                format.name
            );
        }
    }

    #[test]
    fn ties_go_to_even_in_every_format() {
        check_midpoints(FP16, 0x7bff); // all positive finite FP16
        check_midpoints(BF16, 0x7f7f); // all positive finite BF16
        check_midpoints(FP8_E4M3, 0x7e);
        check_midpoints(FP8_E5M2, 0x7b);
    }

    /// The bug that motivated this module: values just below a power of two lost the rounding
    /// carry and came out at half their size.
    #[test]
    fn carry_into_the_exponent_is_kept() {
        assert_eq!(f16_to_f32(f32_to_f16(1.9999)), 2.0);
        assert_eq!(f16_to_f32(f32_to_f16(-0.031_247_877)), -0.031_25);
        assert_eq!(f16_to_f32(f32_to_f16(65_519.0)), 65_504.0);
        assert!(f16_to_f32(f32_to_f16(65_520.0)).is_infinite());
        assert!(f16_to_f32(f32_to_f16(f32::NAN)).is_nan());
    }

    #[test]
    fn known_values_of_each_format() {
        assert_eq!(FP16.max_finite(), 65_504.0);
        assert_eq!(BF16.max_finite(), f64::from(f32::from_bits(0x7f7f_0000)));
        assert_eq!(FP8_E4M3.max_finite(), 448.0);
        assert_eq!(FP8_E5M2.max_finite(), 57_344.0);
        assert_eq!(
            FP8_E4M3.decode(0x01),
            2f32.powi(-9),
            "smallest E4M3 subnormal"
        );
        assert!(FP8_E4M3.decode(0x7f).is_nan());
        assert!(
            FP8_E4M3.decode(FP8_E4M3.encode(1000.0)).is_nan(),
            "E4M3 has no infinity"
        );
        assert!(FP8_E5M2.decode(FP8_E5M2.encode(1e6)).is_infinite());
        assert_eq!(bf16_to_f32(f32_to_bf16(1.0)), 1.0);
        assert_eq!(f16_to_f32(f32_to_f16(-0.0)).to_bits(), (-0.0f32).to_bits());
    }

    /// The stated bound holds on random values across every format's range.
    #[test]
    fn error_stays_within_the_stated_bound() {
        let mut state = 7u64;
        for format in [FP16, BF16, FP8_E4M3, FP8_E5M2] {
            let span = format.max_finite().log2() + 20.0;
            for _ in 0..20_000 {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let u = (state >> 11) as f64 / (1u64 << 53) as f64;
                let magnitude = 2f64.powf(u * span - 20.0).min(format.max_finite());
                let x = if state & 1 == 0 {
                    magnitude
                } else {
                    -magnitude
                } as f32;
                let r = format.round(x);
                let err = (f64::from(x) - f64::from(r)).abs();
                assert!(
                    err <= format.error_bound(f64::from(x)) * (1.0 + 1e-12),
                    "{} x={x} r={r} err={err}",
                    format.name
                );
            }
        }
    }

    /// The old decoder was correct; the shared one agrees with it on every code.
    #[test]
    fn decode_matches_the_bit_shift_definition_of_bf16() {
        for bits in 0u32..=0xffff {
            let a = BF16.decode(bits);
            let b = bf16_to_f32(bits as u16);
            assert!(a == b || (a.is_nan() && b.is_nan()), "{bits:#06x}");
        }
    }
}
