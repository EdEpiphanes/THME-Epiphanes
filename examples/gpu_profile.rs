// SPDX-License-Identifier: AGPL-3.0-only
//! Repeatable profile of the GPU path against the CPU path on one small model.
//!
//! For each pass shape it runs CPU and GPU forward passes interleaved (warm-up first),
//! then reports medians with min and max, plus where the host spent its time on the GPU
//! runs (weight upload, input writes, encoding, submit, wait, read-back). The first GPU
//! run is reported apart because it pays the one-off weight upload.
//!
//! The "decode" shape here is a 2-token pass (the smallest the reference accepts): the
//! reference path has no KV cache yet, so a real incremental decode step is not measured.
//!
//! ```text
//! cargo run --release --example gpu_profile -- \
//!     --model-path models/pythia-14m --label pythia-14m --runs 7 \
//!     --load-note "AWQ job running on CPU" \
//!     --out docs/benchmarks/pythia-14m/data/gpu_profile.json
//! ```

use anyhow::{Context, Result};
use clap::Parser;
use serde_json::{json, Value};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use thme::inference::gpu::{GpuBackend, GpuContext, GpuStats};
use thme::inference::{ArchConfig, WeightPrecision};
use thme::parser::ModelImporter;

#[derive(Parser, Debug)]
#[command(name = "gpu_profile", about = "CPU vs GPU forward-pass profile")]
struct Args {
    /// Model directory (never copied or committed)
    #[arg(long)]
    model_path: PathBuf,
    /// Short model label used in the report, e.g. pythia-14m
    #[arg(long)]
    label: String,
    /// Timed runs per shape and device (at least 7 for a stated spread)
    #[arg(long, default_value_t = 7)]
    runs: usize,
    /// Free-text note on what else was loading the machine
    #[arg(long, default_value = "not recorded")]
    load_note: String,
    /// Output JSON path; printed to stdout when absent
    #[arg(long)]
    out: Option<PathBuf>,
}

fn ms(ns: u128) -> f64 {
    ns as f64 / 1e6
}

fn spread(samples: &[f64]) -> Value {
    let mut sorted = samples.to_vec();
    sorted.sort_by(f64::total_cmp);
    let n = sorted.len();
    let median = if n % 2 == 1 {
        sorted[n / 2]
    } else {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    };
    json!({ "median": median, "min": sorted[0], "max": sorted[n - 1], "samples": samples })
}

fn delta(after: GpuStats, before: GpuStats) -> GpuStats {
    GpuStats {
        dispatches: after.dispatches - before.dispatches,
        submits: after.submits - before.submits,
        weight_uploads: after.weight_uploads - before.weight_uploads,
        weight_upload_ns: after.weight_upload_ns - before.weight_upload_ns,
        input_write_ns: after.input_write_ns - before.input_write_ns,
        encode_ns: after.encode_ns - before.encode_ns,
        submit_ns: after.submit_ns - before.submit_ns,
        wait_ns: after.wait_ns - before.wait_ns,
        readback_ns: after.readback_ns - before.readback_ns,
    }
}

fn stats_json(s: &GpuStats) -> Value {
    json!({
        "dispatches": s.dispatches,
        "submits": s.submits,
        "weight_uploads": s.weight_uploads,
        "weight_upload_ms": ms(s.weight_upload_ns.into()),
        "input_write_ms": ms(s.input_write_ns.into()),
        "encode_ms": ms(s.encode_ns.into()),
        "submit_ms": ms(s.submit_ns.into()),
        "wait_ms": ms(s.wait_ns.into()),
        "readback_ms": ms(s.readback_ns.into()),
    })
}

fn median_stats(all: &[GpuStats]) -> Value {
    let med = |f: fn(&GpuStats) -> u64| -> u64 {
        let mut v: Vec<u64> = all.iter().map(f).collect();
        v.sort_unstable();
        v[v.len() / 2]
    };
    stats_json(&GpuStats {
        dispatches: med(|s| s.dispatches),
        submits: med(|s| s.submits),
        weight_uploads: med(|s| s.weight_uploads),
        weight_upload_ns: med(|s| s.weight_upload_ns),
        input_write_ns: med(|s| s.input_write_ns),
        encode_ns: med(|s| s.encode_ns),
        submit_ns: med(|s| s.submit_ns),
        wait_ns: med(|s| s.wait_ns),
        readback_ns: med(|s| s.readback_ns),
    })
}

fn git(args: &[&str]) -> String {
    std::process::Command::new("git")
        .args(args)
        .output()
        .ok()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_default()
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

fn main() -> Result<()> {
    let args = Args::parse();
    anyhow::ensure!(args.runs >= 1, "--runs must be at least 1");

    let t_init = Instant::now();
    let gpu = Arc::new(GpuContext::init().context("GPU adapter init failed")?);
    let adapter_init_ms = ms(t_init.elapsed().as_nanos());
    let info = gpu.adapter_info.clone();

    let ir = ModelImporter::import(&args.model_path).context("model import failed")?;
    let config = ArchConfig::detect(&ir, &args.model_path)?;
    let cpu_model = config.load(&ir, WeightPrecision::AsStored)?;
    let backend: Arc<dyn GpuBackend> = gpu.clone();
    let gpu_model = config.load_with_gpu(&ir, WeightPrecision::AsStored, Some(backend))?;
    let vocab = cpu_model.vocab_size();

    let shapes = [("prefill_128", 128usize), ("short_pass_2", 2usize)];
    let mut phases = Vec::new();
    for (name, len) in shapes {
        let tokens: Vec<u32> = (0..len).map(|i| ((i * 7919 + 13) % vocab) as u32).collect();

        // Warm-up: the first GPU pass pays pipeline use and the one-off weight upload.
        let before = gpu.stats();
        let t = Instant::now();
        let gpu_ref = gpu_model.forward_logits(&tokens)?;
        let first_ms = ms(t.elapsed().as_nanos());
        let first_stats = delta(gpu.stats(), before);
        let cpu_ref = cpu_model.forward_logits(&tokens)?;
        let diff = max_abs_diff(&gpu_ref, &cpu_ref);

        let (mut cpu_ms, mut gpu_ms, mut per_run) = (Vec::new(), Vec::new(), Vec::new());
        for _ in 0..args.runs {
            let t = Instant::now();
            cpu_model.forward_logits(&tokens)?;
            cpu_ms.push(ms(t.elapsed().as_nanos()));

            let before = gpu.stats();
            let t = Instant::now();
            gpu_model.forward_logits(&tokens)?;
            gpu_ms.push(ms(t.elapsed().as_nanos()));
            per_run.push(delta(gpu.stats(), before));
        }
        phases.push(json!({
            "name": name,
            "tokens": len,
            "cpu_ms": spread(&cpu_ms),
            "gpu_ms": spread(&gpu_ms),
            "gpu_first_run_ms": first_ms,
            "gpu_first_run_breakdown": stats_json(&first_stats),
            "gpu_median_breakdown": median_stats(&per_run),
            "max_abs_logit_diff_gpu_vs_cpu": diff,
        }));
    }

    let report = json!({
        "schema": 1,
        "label": args.label,
        "device": {
            "name": info.name,
            "driver": info.driver,
            "driver_info": info.driver_info,
            "backend": format!("{:?}", info.backend),
            "device_type": format!("{:?}", info.device_type),
            "timestamp_query_used": false,
        },
        "build_profile": if cfg!(debug_assertions) { "debug" } else { "release" },
        "threads": std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1),
        "git_commit": git(&["rev-parse", "--short", "HEAD"]),
        "git_dirty": !git(&["status", "--porcelain", "--untracked-files=no"]).is_empty(),
        "machine_load_note": args.load_note,
        "runs": args.runs,
        "adapter_init_ms": adapter_init_ms,
        "method": "CPU and GPU forward_logits interleaved per shape after one warm-up pass each; wall clock; synthetic token ids (timing only). Breakdown fields are host-side waits, not kernel time; wait_ms bounds kernel time from above.",
        "phases": phases,
    });
    let text = serde_json::to_string_pretty(&report)?;
    match args.out {
        Some(path) => std::fs::write(&path, format!("{text}\n"))?,
        None => println!("{text}"),
    }
    Ok(())
}
