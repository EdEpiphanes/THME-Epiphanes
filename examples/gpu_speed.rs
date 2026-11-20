// SPDX-License-Identifier: AGPL-3.0-only
//! GPU-only speed tools: a per-kernel profile of one window, matrix-multiply TFLOPS per shape,
//! and a perplexity pass over a token slice. No CPU pass is ever run here, so it is safe on
//! models whose CPU pass takes hours.
//!
//! ```text
//! cargo run --release --example gpu_speed -- profile --model-path models/smollm2-1.7b \
//!     --label smollm2-1.7b --tokens eval/wikitext2_test.smollm2-1.7b.u32 --out profile.json
//! cargo run --release --example gpu_speed -- gemm --label smollm2-1.7b
//! cargo run --release --example gpu_speed -- ppl --model-path models/smollm2-1.7b \
//!     --label smollm2-1.7b --tokens eval/... --max-tokens 16384 --weights f16
//! ```

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use thme::inference::gpu::{GpuBackend, GpuContext, WeightStorage};
use thme::inference::{
    evaluate_perplexity, read_token_file, ArchConfig, PerplexityConfig, WeightPrecision,
};
use thme::parser::ModelImporter;

#[derive(Parser, Debug)]
#[command(name = "gpu_speed", about = "GPU-only speed tools")]
struct Cli {
    #[command(subcommand)]
    mode: Mode,
    /// Short label used in the report
    #[arg(long, global = true, default_value = "unlabelled")]
    label: String,
    /// Free-text note on what else was loading the machine
    #[arg(long, global = true, default_value = "not recorded")]
    load_note: String,
    /// Output JSON path; printed to stdout when absent
    #[arg(long, global = true)]
    out: Option<PathBuf>,
    /// Use the old 64 x 64 matmul kernel (paired baseline)
    #[arg(long, global = true)]
    old_gemm: bool,
    /// Use the one-workgroup-per-query attention kernel (paired baseline)
    #[arg(long, global = true)]
    old_attention: bool,
    /// Resident weight budget in GiB (library default when absent)
    #[arg(long, global = true)]
    weight_budget_gb: Option<usize>,
    /// Use the cooperative-matrix (tensor-core) matmul when the adapter has it
    #[arg(long, global = true)]
    coop: bool,
    /// With --coop and f16 weights, stage the weights too instead of reading them in place
    #[arg(long, global = true)]
    no_coop_direct: bool,
    /// Weight storage on the device: f32, f16, q8_0 or q4_0
    #[arg(long, global = true, default_value = "f32")]
    weights: String,
}

#[derive(Subcommand, Debug)]
enum Mode {
    /// Per-kernel device time of one window
    Profile {
        #[arg(long)]
        model_path: PathBuf,
        #[arg(long)]
        tokens: PathBuf,
        #[arg(long, default_value_t = 1024)]
        window: usize,
    },
    /// Matrix-multiply TFLOPS for the shapes of a 1.7B-class decoder
    Gemm {
        #[arg(long, default_value_t = 1024)]
        rows: usize,
        #[arg(long, default_value_t = 5)]
        runs: usize,
    },
    /// Perplexity over a token slice, GPU only
    Ppl {
        #[arg(long)]
        model_path: PathBuf,
        #[arg(long)]
        tokens: PathBuf,
        #[arg(long, default_value_t = 1024)]
        context: usize,
        #[arg(long)]
        max_tokens: Option<usize>,
    },
}

fn git(args: &[&str]) -> String {
    std::process::Command::new("git")
        .args(args)
        .output()
        .ok()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_default()
}

fn parse_storage(s: &str) -> Result<WeightStorage> {
    Ok(match s {
        "f32" => WeightStorage::F32,
        "f16" => WeightStorage::F16,
        "q8_0" => WeightStorage::Q8_0,
        "q4_0" => WeightStorage::Q4_0,
        other => bail!("unknown --weights {other}; use f32, f16, q8_0 or q4_0"),
    })
}

fn header(cli: &Cli, gpu: &GpuContext) -> Value {
    json!({
        "label": cli.label,
        "device": gpu.adapter_json(),
        "shader_f16": gpu.shader_f16_supported(),
        "weights_requested": cli.weights,
        "coop_matrix_ready": gpu.coop_matrix_ready(),
        "coop_matrix_requested": cli.coop,
        "old_gemm_kernel": cli.old_gemm,
        "old_attention_kernel": cli.old_attention,
        "weight_budget_gb": cli.weight_budget_gb,
        "weights_used": format!("{:?}", gpu.weight_storage()),
        "build_profile": if cfg!(debug_assertions) { "debug" } else { "release" },
        "git_commit": git(&["rev-parse", "--short", "HEAD"]),
        "git_dirty": !git(&["status", "--porcelain", "--untracked-files=no"]).is_empty(),
        "machine_load_note": cli.load_note,
    })
}

fn load(gpu: &Arc<GpuContext>, model_path: &Path) -> Result<Box<dyn thme::inference::CausalLm>> {
    let ir = ModelImporter::import(model_path).context("model import failed")?;
    let config = ArchConfig::detect(&ir, model_path)?;
    let backend: Arc<dyn GpuBackend> = gpu.clone();
    config.load_with_gpu(&ir, WeightPrecision::AsStored, Some(backend))
}

/// Flops of a `linear ... RxIxO` label, or 0 for other kernels.
fn label_flops(label: &str) -> f64 {
    let shape = label.rsplit(' ').next().unwrap_or("");
    let dims: Vec<f64> = shape.split('x').filter_map(|v| v.parse().ok()).collect();
    if label.starts_with("linear") && dims.len() == 3 {
        2.0 * dims[0] * dims[1] * dims[2]
    } else {
        0.0
    }
}

fn profile(
    cli: &Cli,
    gpu: &Arc<GpuContext>,
    model_path: &Path,
    tokens: &Path,
    window: usize,
) -> Result<Value> {
    let model = load(gpu, model_path)?;
    let tokens = read_token_file(tokens)?;
    let cfg = PerplexityConfig {
        context_len: window,
        max_tokens: Some(window),
    };
    evaluate_perplexity(&*model, &tokens, &cfg, |_, _, _| {})?; // warm-up: weight upload
    gpu.set_op_profile(true);
    let before = gpu.stats();
    let t = Instant::now();
    let rep = evaluate_perplexity(&*model, &tokens, &cfg, |_, _, _| {})?;
    let wall_ms = t.elapsed().as_secs_f64() * 1e3;
    gpu.set_op_profile(false);
    let after = gpu.stats();
    let samples = gpu.take_op_profile();

    let mut by_label: BTreeMap<String, (u64, u64)> = BTreeMap::new();
    for s in &samples {
        let e = by_label.entry(s.label.clone()).or_default();
        e.0 += 1;
        e.1 += s.ns;
    }
    let total_ns: u64 = samples.iter().map(|s| s.ns).sum();
    let mut rows: Vec<Value> = by_label
        .iter()
        .map(|(label, (count, ns))| {
            let flops = label_flops(label) * *count as f64;
            json!({
                "label": label,
                "dispatches": count,
                "ms": *ns as f64 / 1e6,
                "share_of_device_time": *ns as f64 / total_ns.max(1) as f64,
                "tflops": if flops > 0.0 { flops / (*ns as f64 * 1e-9) / 1e12 } else { 0.0 },
            })
        })
        .collect();
    rows.sort_by(|a, b| {
        b["ms"]
            .as_f64()
            .partial_cmp(&a["ms"].as_f64())
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut kinds: BTreeMap<String, u64> = BTreeMap::new();
    for s in &samples {
        let kind = if s.label.starts_with("linear") {
            "linear (matmul)".to_string()
        } else {
            s.label.clone()
        };
        *kinds.entry(kind).or_default() += s.ns;
    }
    let mut value = header(cli, gpu);
    value["window_tokens"] = json!(window);
    value["wall_ms"] = json!(wall_ms);
    value["device_ms_sum"] = json!(total_ns as f64 / 1e6);
    value["device_ms_by_kind"] = json!(kinds
        .iter()
        .map(|(k, v)| (k.clone(), *v as f64 / 1e6))
        .collect::<BTreeMap<_, _>>());
    value["dispatches"] = json!(after.dispatches - before.dispatches);
    value["submits"] = json!(after.submits - before.submits);
    value["weight_uploads_during_window"] = json!(after.weight_uploads - before.weight_uploads);
    value["host_ms"] = json!({
        "weight_upload": (after.weight_upload_ns - before.weight_upload_ns) as f64 / 1e6,
        "input_write": (after.input_write_ns - before.input_write_ns) as f64 / 1e6,
        "encode": (after.encode_ns - before.encode_ns) as f64 / 1e6,
        "submit": (after.submit_ns - before.submit_ns) as f64 / 1e6,
        "wait": (after.wait_ns - before.wait_ns) as f64 / 1e6,
        "readback": (after.readback_ns - before.readback_ns) as f64 / 1e6,
    });
    value["perplexity"] = json!(rep.perplexity);
    value["kernels"] = json!(rows);
    value["method"] = json!("One untimed warm-up window (weight upload), then one window with every dispatch in its own compute pass with timestamp queries; wall_ms includes the per-dispatch pass overhead, so it exceeds an unprofiled window.");
    Ok(value)
}

fn gemm(cli: &Cli, gpu: &Arc<GpuContext>, rows: usize, runs: usize) -> Result<Value> {
    anyhow::ensure!(gpu.timestamp_query_supported(), "needs timestamp queries");
    gpu.set_kernel_timing(true);
    // (in_dim, out_dim) of SmolLM2-1.7B: q/o, k/v (MHA), gate/up, down, head.
    let shapes = [
        (2048usize, 2048usize),
        (2048, 8192),
        (8192, 2048),
        (2048, 49152),
        (576, 1536),
    ];
    let mut out = Vec::new();
    for (in_dim, out_dim) in shapes {
        let x: Vec<f32> = (0..rows * in_dim)
            .map(|i| ((i * 7 % 13) as f32 - 6.0) / 8.0)
            .collect();
        let w: Vec<f32> = (0..in_dim * out_dim)
            .map(|i| ((i * 5 % 11) as f32 - 5.0) / 64.0)
            .collect();
        gpu.compute_linear_slices(&x, &[(&w, out_dim)], in_dim)?; // upload + warm-up
        let mut ts = Vec::new();
        for _ in 0..runs {
            let before = gpu.stats().kernel_ns;
            gpu.compute_linear_slices(&x, &[(&w, out_dim)], in_dim)?;
            ts.push((gpu.stats().kernel_ns - before) as f64);
        }
        ts.sort_by(f64::total_cmp);
        let med = ts[ts.len() / 2];
        let flops = 2.0 * (rows * in_dim * out_dim) as f64;
        out.push(json!({
            "rows": rows, "in_dim": in_dim, "out_dim": out_dim,
            "median_us": med / 1e3, "min_us": ts[0] / 1e3,
            "tflops_median": flops / (med * 1e-9) / 1e12,
            "tflops_best": flops / (ts[0] * 1e-9) / 1e12,
        }));
        gpu.clear_weight_cache();
    }
    let mut value = header(cli, gpu);
    value["method"] = json!("compute_linear_slices on synthetic weights, device time of the compute pass from timestamp queries, median of the runs after one warm-up");
    value["shapes"] = json!(out);
    Ok(value)
}

fn ppl(
    cli: &Cli,
    gpu: &Arc<GpuContext>,
    model_path: &Path,
    tokens: &Path,
    context: usize,
    max_tokens: Option<usize>,
) -> Result<Value> {
    let model = load(gpu, model_path)?;
    let tokens = read_token_file(tokens)?;
    let warm = PerplexityConfig {
        context_len: context,
        max_tokens: Some(context),
    };
    evaluate_perplexity(&*model, &tokens, &warm, |_, _, _| {})?;
    let cfg = PerplexityConfig {
        context_len: context,
        max_tokens,
    };
    let before = gpu.stats();
    let mut last = 0u64;
    let began = Instant::now();
    let rep = evaluate_perplexity(&*model, &tokens, &cfg, |done, total, running| {
        let s = began.elapsed().as_secs();
        if done == total || s >= last + 30 {
            last = s;
            println!("{done}/{total} windows, {s} s, running PPL {running:.4}");
        }
    })?;
    let after = gpu.stats();
    let mut value = header(cli, gpu);
    value["context_len"] = json!(context);
    value["scored_tokens"] = json!(rep.scored_tokens);
    value["corpus_tokens_used"] = json!(rep.corpus_tokens_used);
    value["windows"] = json!(rep.windows);
    value["elapsed_secs"] = json!(rep.elapsed_secs);
    value["tokens_per_sec"] = json!(rep.tokens_per_sec);
    value["ms_per_token"] = json!(1e3 / rep.tokens_per_sec);
    value["perplexity"] = json!(rep.perplexity);
    value["cross_entropy_nats"] = json!(rep.cross_entropy_nats);
    value["weight_uploads_during_pass"] = json!(after.weight_uploads - before.weight_uploads);
    value["method"] = json!("One untimed warm-up window (weight upload and quantization), then the timed GPU-only pass over the slice, wall clock, non-overlapping windows.");
    Ok(value)
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let gpu = Arc::new(GpuContext::init().context("GPU adapter init failed")?);
    gpu.set_weight_storage(parse_storage(&cli.weights)?);
    gpu.set_register_blocked_gemm(!cli.old_gemm);
    gpu.set_coop_matrix(cli.coop);
    gpu.set_coop_direct_f16(!cli.no_coop_direct);
    gpu.set_tiled_attention(!cli.old_attention);
    if let Some(gb) = cli.weight_budget_gb {
        gpu.set_weight_budget(gb << 30);
    }
    let value = match &cli.mode {
        Mode::Profile {
            model_path,
            tokens,
            window,
        } => profile(&cli, &gpu, model_path, tokens, *window)?,
        Mode::Gemm { rows, runs } => gemm(&cli, &gpu, *rows, *runs)?,
        Mode::Ppl {
            model_path,
            tokens,
            context,
            max_tokens,
        } => ppl(&cli, &gpu, model_path, tokens, *context, *max_tokens)?,
    };
    let text = serde_json::to_string_pretty(&value)?;
    match &cli.out {
        Some(path) => std::fs::write(path, format!("{text}\n"))?,
        None => println!("{text}"),
    }
    Ok(())
}
