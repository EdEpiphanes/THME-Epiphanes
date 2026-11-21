// SPDX-License-Identifier: AGPL-3.0-only
//! Feature audit of every real GPU adapter: which optional features wgpu exposes and whether a
//! tiny kernel for each one runs and gives the right numbers.
//!
//! ```text
//! cargo run --release --example gpu_features -- --out docs/benchmarks/_adapters/rtx-4060-ti_features.json
//! ```

use anyhow::{Context, Result};
use serde_json::{json, Value};
use thme::inference::gpu::features::audit;

fn main() -> Result<()> {
    let out = std::env::args()
        .skip_while(|a| a != "--out")
        .nth(1)
        .map(std::path::PathBuf::from);
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::PRIMARY,
        flags: wgpu::InstanceFlags::empty(),
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });
    let reports: Vec<Value> =
        pollster::block_on(instance.enumerate_adapters(wgpu::Backends::PRIMARY))
            .iter()
            .filter(|a| a.get_info().device_type != wgpu::DeviceType::Cpu)
            .map(audit)
            .collect();
    let text = serde_json::to_string_pretty(&json!({ "adapters": reports }))?;
    match out {
        Some(p) => std::fs::write(&p, text + "\n").with_context(|| p.display().to_string())?,
        None => println!("{text}"),
    }
    Ok(())
}
