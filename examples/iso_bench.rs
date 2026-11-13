// SPDX-License-Identifier: AGPL-3.0-only
//! Times the isometric kit workload of tests/iso_golden.rs in Rust, for comparison with
//! `node scripts/iso_golden.mjs <iso-kit.mjs> --bench N` (the same cases in the anatomy kit).
//! Usage: cargo run --release --example iso_bench -- N

#[path = "../tests/iso_golden.rs"]
#[allow(dead_code)]
mod golden;

fn main() {
    let n: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(1000);
    let _ = golden::cases(); // warm up
    let t0 = std::time::Instant::now();
    let mut bytes = 0usize;
    for _ in 0..n {
        for (_, svg) in golden::cases() {
            bytes += svg.len();
        }
    }
    let ms = t0.elapsed().as_secs_f64() * 1000.0;
    println!(
        "{{\"runtime\":\"rust release\",\"iterations\":{n},\"ms\":{ms:.1},\"bytes\":{bytes}}}"
    );
}
