// SPDX-License-Identifier: AGPL-3.0-only
//! Measured decode speed: the whole-sequence path against the KV-cache path (THME-3070).
//!
//! For each generated length, the old path (`forward_logits` over the growing sequence, the
//! loop the chat tab used before THME-3070) and the new path (`forward_extend`, one token per
//! step) decode the same greedy reply from the same fixed prompt. Runs are paired and
//! interleaved: each run times the two paths back to back, and which one goes first alternates,
//! so machine noise reaches both. Warmup runs are timed and discarded.
//!
//! Tokens/s is generated tokens over the wall time of the whole reply, prompt prefill included,
//! because the old path pays for the prompt on every step. Everything here is measured on the
//! machine that runs it; the report records the machine and whether it was shared.
//!
//! ```text
//! cargo run --release --example decode_speed -- \
//!     --model-path models/SmolLM-135M --model-name smollm-135m --shared-machine \
//!     --out docs/benchmarks/smollm-135m/data/decode_speed.json
//! ```

use anyhow::{bail, Context, Result};
use clap::Parser;
use serde_json::{json, Value};
use std::path::PathBuf;
use std::process::Command;
use std::time::Instant;

use thme::inference::{ArchConfig, CausalLm, WeightPrecision};
use thme::parser::ModelImporter;
use thme::profiling::kv_cache::KvCacheEstimator;

#[derive(Parser, Debug)]
#[command(
    name = "decode_speed",
    about = "Measured decode tokens/s: whole-sequence path against the KV cache"
)]
struct Args {
    /// Model directory or file with on-disk weights
    #[arg(long)]
    model_path: PathBuf,

    /// Short model name written to the report (the path itself is never written)
    #[arg(long)]
    model_name: String,

    /// Generated-token counts to time, comma separated
    #[arg(long, value_delimiter = ',', default_value = "16,64,256")]
    lengths: Vec<usize>,

    /// Timed runs per length and path (at least 5)
    #[arg(long, default_value_t = 5)]
    runs: usize,

    /// Untimed warmup runs per length and path
    #[arg(long, default_value_t = 1)]
    warmup: usize,

    /// Prompt length in tokens (a fixed list of ids, not text)
    #[arg(long, default_value_t = 8)]
    prompt_tokens: usize,

    /// Say so when another CPU-heavy job shared the machine during the run
    #[arg(long)]
    shared_machine: bool,

    /// Where to write the JSON report
    #[arg(long)]
    out: PathBuf,
}

/// One timed reply: its tokens, its wall time, and for the cached path the cache size reached.
struct Timed {
    tokens: Vec<u32>,
    secs: f64,
    kv: Option<(usize, usize)>,
}

/// Index of the largest logit; the first one wins a tie.
fn argmax(logits: &[f32]) -> Result<u32> {
    if logits.is_empty() {
        bail!("The model returned no logits");
    }
    let mut best = 0usize;
    for (i, &v) in logits.iter().enumerate() {
        if v > logits[best] {
            best = i;
        }
    }
    Ok(u32::try_from(best)?)
}

/// The old loop: every step runs the whole growing sequence.
fn decode_whole(model: &dyn CausalLm, prompt: &[u32], generated: usize) -> Result<Timed> {
    let started = Instant::now();
    let mut ids = prompt.to_vec();
    let mut tokens = Vec::with_capacity(generated);
    for _ in 0..generated {
        let next = argmax(&model.forward_logits(&ids)?)?;
        ids.push(next);
        tokens.push(next);
    }
    Ok(Timed {
        tokens,
        secs: started.elapsed().as_secs_f64(),
        kv: None,
    })
}

/// The new loop: prefill the prompt once, then one token per step.
fn decode_cached(model: &dyn CausalLm, prompt: &[u32], generated: usize) -> Result<Timed> {
    let Some(mut cache) = model.new_cache() else {
        bail!("This architecture has no KV-cache path");
    };
    let started = Instant::now();
    let mut tokens = Vec::with_capacity(generated);
    let mut next = argmax(&model.forward_extend(&mut cache, prompt)?)?;
    for _ in 0..generated {
        tokens.push(next);
        if tokens.len() < generated {
            next = argmax(&model.forward_extend(&mut cache, &[next])?)?;
        }
    }
    let secs = started.elapsed().as_secs_f64();
    Ok(Timed {
        tokens,
        secs,
        kv: Some((cache.len(), cache.bytes())),
    })
}

/// Median, minimum and maximum of a list of values.
fn summary(values: &[f64]) -> Value {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let median = match sorted.len() {
        0 => 0.0,
        n if n % 2 == 1 => sorted[n / 2],
        n => (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0,
    };
    json!({
        "median": median,
        "min": sorted.first().copied().unwrap_or(0.0),
        "max": sorted.last().copied().unwrap_or(0.0),
    })
}

fn cpu_model() -> String {
    if let Ok(text) = std::fs::read_to_string("/proc/cpuinfo") {
        for line in text.lines() {
            if let Some((key, value)) = line.split_once(':') {
                if key.trim() == "model name" {
                    return value.trim().to_string();
                }
            }
        }
    }
    std::env::var("PROCESSOR_IDENTIFIER").unwrap_or_else(|_| "unknown".to_string())
}

/// Short commit of the code that was measured; `git_dirty` says whether tracked files differed.
fn git_state() -> (String, bool) {
    let run = |args: &[&str]| {
        Command::new("git")
            .args(args)
            .output()
            .ok()
            .filter(|out| out.status.success())
            .map(|out| String::from_utf8_lossy(&out.stdout).trim().to_string())
    };
    let commit = run(&["rev-parse", "--short", "HEAD"]).unwrap_or_else(|| "unknown".to_string());
    let dirty = run(&["status", "--porcelain", "--untracked-files=no"])
        .is_some_and(|changes| !changes.is_empty());
    (commit, dirty)
}

fn main() -> Result<()> {
    let args = Args::parse();
    if args.runs < 5 {
        bail!("--runs must be at least 5, received {}", args.runs);
    }
    if args.lengths.is_empty() || args.lengths.contains(&0) {
        bail!("--lengths must list positive token counts");
    }
    if args.prompt_tokens < 2 {
        bail!("--prompt-tokens must be at least 2");
    }

    let source = ModelImporter::import(&args.model_path)
        .with_context(|| format!("Failed to import model from {:?}", args.model_path))?;
    let arch = ArchConfig::detect(&source, &args.model_path)?;
    let model = arch.load(&source, WeightPrecision::AsStored)?;
    if model.new_cache().is_none() {
        bail!("{} has no KV-cache path to measure", arch.family());
    }
    let vocab = model.vocab_size() as u64;
    let prompt: Vec<u32> = (0..args.prompt_tokens as u64)
        .map(|i| ((i * 7919 + 13) % vocab) as u32)
        .collect();
    let model = model.as_ref();

    for _ in 0..args.warmup {
        for &generated in &args.lengths {
            decode_whole(model, &prompt, generated)?;
            decode_cached(model, &prompt, generated)?;
        }
    }

    // [length index] -> per-run samples
    let mut old_secs = vec![Vec::new(); args.lengths.len()];
    let mut new_secs = vec![Vec::new(); args.lengths.len()];
    let mut identical = vec![true; args.lengths.len()];
    let mut kv_seen = vec![None; args.lengths.len()];
    for run in 0..args.runs {
        for (index, &generated) in args.lengths.iter().enumerate() {
            let (old, new) = if run % 2 == 0 {
                let old = decode_whole(model, &prompt, generated)?;
                (old, decode_cached(model, &prompt, generated)?)
            } else {
                let new = decode_cached(model, &prompt, generated)?;
                (decode_whole(model, &prompt, generated)?, new)
            };
            identical[index] &= old.tokens == new.tokens;
            old_secs[index].push(old.secs);
            new_secs[index].push(new.secs);
            kv_seen[index] = new.kv;
            println!(
                "run {}/{} generated {generated:>4}: whole {:>8.2} tok/s  cached {:>8.2} tok/s",
                run + 1,
                args.runs,
                generated as f64 / old.secs,
                generated as f64 / new.secs
            );
        }
    }

    let mut rows = Vec::new();
    for (index, &generated) in args.lengths.iter().enumerate() {
        let tps =
            |secs: &[f64]| -> Vec<f64> { secs.iter().map(|s| generated as f64 / s).collect() };
        let old_tps = tps(&old_secs[index]);
        let new_tps = tps(&new_secs[index]);
        let ratios: Vec<f64> = old_secs[index]
            .iter()
            .zip(&new_secs[index])
            .map(|(old, new)| old / new)
            .collect();
        let kv = match kv_seen[index] {
            Some((context, measured)) => {
                // Same formula as the profiling estimate, with the model's own KV heads and head
                // dimension and f32 storage (4 bytes per element).
                let head_dim = source.hidden_size / source.num_attention_heads;
                let estimate = KvCacheEstimator::calculate_bytes(
                    source.num_layers,
                    1,
                    context,
                    source.num_key_value_heads,
                    head_dim,
                    4.0,
                );
                json!({
                    "context_tokens": context,
                    "measured_bytes": measured,
                    "estimate_bytes": estimate,
                    "matches_estimate": measured == estimate,
                })
            }
            None => Value::Null,
        };
        rows.push(json!({
            "generated_tokens": generated,
            "whole_sequence": {
                "tokens_per_sec": summary(&old_tps),
                "tokens_per_sec_runs": old_tps,
                "wall_secs_runs": old_secs[index],
            },
            "kv_cache": {
                "tokens_per_sec": summary(&new_tps),
                "tokens_per_sec_runs": new_tps,
                "wall_secs_runs": new_secs[index],
            },
            "speedup_paired": summary(&ratios),
            "speedup_paired_runs": ratios,
            "greedy_tokens_identical": identical[index],
            "kv_cache_bytes": kv,
        }));
        println!(
            "generated {generated:>4}: whole {:.2} tok/s, cached {:.2} tok/s, paired speedup median {:.2}x",
            summary(&old_tps)["median"].as_f64().unwrap_or(0.0),
            summary(&new_tps)["median"].as_f64().unwrap_or(0.0),
            summary(&ratios)["median"].as_f64().unwrap_or(0.0),
        );
    }

    let (commit, dirty) = git_state();
    let report = json!({
        "schema": "thme.decode_speed.v1",
        "label": "Measured on this machine by examples/decode_speed.rs. Not a published benchmark.",
        "machine_note": if args.shared_machine {
            "Shared machine: another CPU-heavy job ran during these measurements, so timing is noisy. Paired, interleaved runs share that noise between the two paths; the spread is reported."
        } else {
            "Machine not marked as shared."
        },
        "model": {
            "name": args.model_name,
            "family": arch.family(),
            "layers": source.num_layers,
            "hidden_size": source.hidden_size,
            "attention_heads": source.num_attention_heads,
            "kv_heads": source.num_key_value_heads,
            "vocab_size": model.vocab_size(),
            "weights": "as stored, dense f32 in memory",
        },
        "machine": {
            "cpu": cpu_model(),
            "threads": std::thread::available_parallelism().map_or(1, usize::from),
            "os": std::env::consts::OS,
            "arch": std::env::consts::ARCH,
            "build_profile": if cfg!(debug_assertions) { "debug" } else { "release" },
            "git_commit": commit,
            "git_tracked_files_dirty": dirty,
        },
        "method": {
            "decoding": "greedy (argmax); no sampling, so no seed is involved",
            "prompt_token_ids": prompt,
            "prompt_construction": "id = (i * 7919 + 13) mod vocab_size, i = 0..prompt_tokens",
            "warmup_runs": args.warmup,
            "timed_runs": args.runs,
            "order": "paired and interleaved; the first path alternates each run",
            "tokens_per_sec": "generated tokens / wall seconds of the whole reply, prompt prefill included",
            "paths": {
                "whole_sequence": "forward_logits over the growing sequence each step (the old chat loop)",
                "kv_cache": "forward_extend: prompt prefill, then one token per step",
            },
        },
        "lengths": rows,
    });
    if let Some(parent) = args.out.parent() {
        std::fs::create_dir_all(parent).with_context(|| format!("Failed to create {parent:?}"))?;
    }
    std::fs::write(&args.out, serde_json::to_string_pretty(&report)? + "\n")
        .with_context(|| format!("Failed to write {:?}", args.out))?;
    println!("Wrote {:?}", args.out);
    Ok(())
}
