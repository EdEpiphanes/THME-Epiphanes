// SPDX-License-Identifier: AGPL-3.0-only
//! Full-split perplexity pass timed on CPU and on GPU, paired and order-randomized.
//!
//! Before timing it waits for a quiet machine (total CPU below `--max-cpu` percent on every
//! one of a few samples, up to `--wait-minutes`), records the samples, and repeats the check
//! before each pair. Each repeat runs one full CPU pass and one full GPU pass in a random
//! order. The report carries both perplexities and their difference.
//!
//! ```text
//! cargo run --release --example gpu_perplexity -- \
//!     --model-path models/pythia-14m --label pythia-14m \
//!     --tokens target/eval/wikitext2_test.pythia-14m.u32 --runs 3 \
//!     --out docs/benchmarks/pythia-14m/data/gpu_perplexity.json
//! ```

use anyhow::{bail, Context, Result};
use clap::Parser;
use serde_json::{json, Value};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};
use sysinfo::System;

use thme::inference::{
    evaluate_perplexity, read_token_file, ArchConfig, CausalLm, DeviceArg, GpuBackend,
    PerplexityConfig, PerplexityReport, WeightPrecision,
};
use thme::parser::ModelImporter;

#[derive(Parser, Debug)]
#[command(name = "gpu_perplexity", about = "CPU vs GPU full perplexity pass")]
struct Args {
    /// Model directory (never copied or committed)
    #[arg(long)]
    model_path: PathBuf,
    /// Short model label used in the report, e.g. pythia-14m
    #[arg(long)]
    label: String,
    /// Token file from scripts/prepare_eval_tokens.py
    #[arg(long)]
    tokens: PathBuf,
    /// Tokens per window
    #[arg(long, default_value_t = 1024)]
    context: usize,
    /// Paired repeats (one full CPU pass and one full GPU pass each)
    #[arg(long, default_value_t = 3)]
    runs: usize,
    /// Quiet-machine threshold, percent of total CPU
    #[arg(long, default_value_t = 15.0)]
    max_cpu: f32,
    /// Give up waiting for a quiet machine after this many minutes
    #[arg(long, default_value_t = 30)]
    wait_minutes: u64,
    /// Free-text note appended to the recorded load
    #[arg(long, default_value = "")]
    load_note: String,
    /// Output JSON path; printed to stdout when absent
    #[arg(long)]
    out: Option<PathBuf>,
}

/// Samples total CPU use `n` times, one second apart.
fn cpu_samples(sys: &mut System, n: usize) -> Vec<f32> {
    (0..n)
        .map(|_| {
            sys.refresh_cpu_usage();
            std::thread::sleep(Duration::from_millis(1000));
            sys.refresh_cpu_usage();
            sys.global_cpu_usage()
        })
        .collect()
}

/// Waits until every sample is below the threshold. Returns the samples that passed, or
/// the last samples taken when the wait ran out.
fn wait_quiet(sys: &mut System, args: &Args) -> (bool, Vec<f32>, f64) {
    let started = Instant::now();
    loop {
        let samples = cpu_samples(sys, 5);
        let quiet = samples.iter().all(|&s| s < args.max_cpu);
        println!(
            "waiting for a quiet machine ({:.0} s so far): CPU samples {samples:?}, need all below {}%",
            started.elapsed().as_secs_f64(),
            args.max_cpu
        );
        if quiet || started.elapsed() >= Duration::from_secs(args.wait_minutes * 60) {
            return (quiet, samples, started.elapsed().as_secs_f64());
        }
        std::thread::sleep(Duration::from_secs(20));
    }
}

fn run_pass(model: &dyn CausalLm, tokens: &[u32], config: &PerplexityConfig) -> Result<Value> {
    let report: PerplexityReport = evaluate_perplexity(model, tokens, config, |_, _, _| {})?;
    Ok(json!({
        "elapsed_secs": report.elapsed_secs,
        "perplexity": report.perplexity,
        "cross_entropy_nats": report.cross_entropy_nats,
        "scored_tokens": report.scored_tokens,
        "windows": report.windows,
    }))
}

fn spread(values: &[f64]) -> Value {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let n = sorted.len();
    let median = if n % 2 == 1 {
        sorted[n / 2]
    } else {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    };
    json!({ "median": median, "min": sorted[0], "max": sorted[n - 1], "samples": values })
}

fn git(args: &[&str]) -> String {
    std::process::Command::new("git")
        .args(args)
        .output()
        .ok()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_default()
}

fn main() -> Result<()> {
    let args = Args::parse();
    if args.runs == 0 {
        bail!("--runs must be at least 1");
    }
    let gpu: Arc<dyn GpuBackend> = DeviceArg::Gpu
        .resolve()
        .context("GPU adapter init failed")?;
    let adapter = gpu.adapter_json();

    let ir = ModelImporter::import(&args.model_path).context("model import failed")?;
    let config = ArchConfig::detect(&ir, &args.model_path)?;
    let cpu_model = config.load(&ir, WeightPrecision::AsStored)?;
    let gpu_model = config.load_with_gpu(&ir, WeightPrecision::AsStored, Some(gpu.clone()))?;
    let tokens = read_token_file(&args.tokens)?;
    let eval = PerplexityConfig {
        context_len: args.context,
        max_tokens: None,
    };
    // The first GPU pass pays pipeline use and the one-off weight upload: do it untimed.
    let warm = PerplexityConfig {
        context_len: args.context,
        max_tokens: Some(args.context * 2),
    };
    run_pass(&*gpu_model, &tokens, &warm)?;

    let mut sys = System::new().context("system info unavailable")?;
    let mut seed = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(1)
        | 1;
    let (mut cpu_secs, mut gpu_secs) = (Vec::new(), Vec::new());
    let mut pairs = Vec::new();
    for repeat in 0..args.runs {
        let (quiet, samples, waited) = wait_quiet(&mut sys, &args);
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let gpu_first = (seed >> 33) & 1 == 1;
        let order = if gpu_first {
            ["gpu", "cpu"]
        } else {
            ["cpu", "gpu"]
        };
        let mut results = std::collections::BTreeMap::new();
        for device in order {
            let model: &dyn CausalLm = if device == "gpu" {
                &*gpu_model
            } else {
                &*cpu_model
            };
            let result = run_pass(model, &tokens, &eval)?;
            println!("repeat {repeat} {device}: {result}");
            results.insert(device, result);
        }
        let secs = |d: &str| results[d]["elapsed_secs"].as_f64().unwrap_or(f64::NAN);
        cpu_secs.push(secs("cpu"));
        gpu_secs.push(secs("gpu"));
        let ppl = |d: &str| results[d]["perplexity"].as_f64().unwrap_or(f64::NAN);
        pairs.push(json!({
            "repeat": repeat,
            "order": order,
            "quiet_before": quiet,
            "waited_secs": waited,
            "cpu_percent_samples_before": samples,
            "cpu": results["cpu"],
            "gpu": results["gpu"],
            "perplexity_diff_gpu_minus_cpu": ppl("gpu") - ppl("cpu"),
        }));
    }
    let last = pairs.last().cloned().unwrap_or(Value::Null);
    let ratio: Vec<f64> = cpu_secs.iter().zip(&gpu_secs).map(|(c, g)| c / g).collect();
    let report = json!({
        "schema": 1,
        "label": args.label,
        "device": adapter,
        "build_profile": if cfg!(debug_assertions) { "debug" } else { "release" },
        "threads": std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1),
        "git_commit": git(&["rev-parse", "--short", "HEAD"]),
        "git_dirty": !git(&["status", "--porcelain", "--untracked-files=no"]).is_empty(),
        "corpus_tokens": tokens.len(),
        "context_len": args.context,
        "runs": args.runs,
        "quiet_threshold_cpu_percent": args.max_cpu,
        "machine_load_note": args.load_note,
        "method": "One untimed GPU warm-up (2 windows), then per repeat one full CPU pass and one full GPU pass over the whole token file in a random order, wall clock, after a quiet-machine check (5 one-second samples of total CPU, all below the threshold, else the wait limit is recorded). Perplexity is the same measured pass on both devices.",
        "cpu_secs": spread(&cpu_secs),
        "gpu_secs": spread(&gpu_secs),
        "speedup_cpu_over_gpu": spread(&ratio),
        "perplexity_cpu": last["cpu"]["perplexity"],
        "perplexity_gpu": last["gpu"]["perplexity"],
        "perplexity_diff_gpu_minus_cpu": last["perplexity_diff_gpu_minus_cpu"],
        "pairs": pairs,
    });
    let text = serde_json::to_string_pretty(&report)?;
    match args.out {
        Some(path) => std::fs::write(&path, format!("{text}\n"))?,
        None => println!("{text}"),
    }
    Ok(())
}
