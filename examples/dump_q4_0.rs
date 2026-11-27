// SPDX-License-Identifier: AGPL-3.0-only
//! Oracle for the PyTorch bridge (`scripts/train/quant.py`): builds a fixed pseudo-random matrix,
//! round-trips it through THME's own Q4_0 (`fake_quantize`, the function `thme evaluate` uses) and
//! writes the input and the result as raw f32 bit patterns, so the comparison is exact.
//!
//! ```text
//! cargo run --example dump_q4_0 -- tests/fixtures/q4_0_roundtrip.json
//! ```

use anyhow::{Context, Result};
use serde_json::json;
use thme::inference::fake_quantize;
use thme::ir::TensorFormat;

const ROWS: usize = 4;
const COLS: usize = 128;

/// xorshift64*: a fixed, dependency-free generator; the output is a uniform value in [-1, 1).
fn next(state: &mut u64) -> f32 {
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    let r = state.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 40;
    (r as f32 / (1u64 << 23) as f32) - 1.0
}

/// 16 blocks of 32: zeros, a tie between +max and -max, one outlier, FP16-subnormal scales, then random
/// blocks whose scale grows from 1e-3 to about 3.
fn input() -> Vec<f32> {
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    let mut values = Vec::with_capacity(ROWS * COLS);
    for block in 0..(ROWS * COLS / 32) {
        let scale = match block {
            3 => 3e-5, // scale lands in the FP16 subnormal range
            b if b >= 4 => 1e-3 * 1.6f32.powi(b as i32 - 3),
            _ => 0.05,
        };
        for _ in 0..32 {
            values.push(match block {
                0 => 0.0,
                _ => next(&mut state) * scale,
            });
        }
        let start = block * 32;
        if block == 1 {
            values[start + 5] = 0.4;
            values[start + 20] = -0.4;
        } else if block == 2 {
            values[start + 7] = -3.0;
        }
    }
    values
}

fn main() -> Result<()> {
    let out = std::env::args()
        .nth(1)
        .context("usage: dump_q4_0 <out.json>")?;
    let original = input();
    let mut rounded = original.clone();
    fake_quantize(&mut rounded, &TensorFormat::Q4_0)?;
    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<u32>>();
    let report = json!({
        "format": "Q4_0",
        "shape": [ROWS, COLS],
        "note": "f32 bit patterns; output is THME fake_quantize(Q4_0) of input",
        "input_bits": bits(&original),
        "output_bits": bits(&rounded),
    });
    std::fs::write(&out, serde_json::to_string(&report)?)
        .with_context(|| format!("Failed to write {out}"))?;
    Ok(())
}
