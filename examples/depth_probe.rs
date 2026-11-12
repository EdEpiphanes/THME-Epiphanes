//! Measured depth probe: what the stored weights predict at each depth, and what
//! happens when layers are repeated or skipped.
//!
//! Part A (exit readout). After every layer the model's own final norm and output
//! head are applied to the intermediate state. For each depth this reports
//! perplexity, how often the top prediction already equals the full-depth top
//! prediction, and the "settling depth" of each token: the first depth from which
//! the top prediction never changes again.
//!
//! Part B (schedule changes). A block of layers is repeated or skipped, with no
//! retraining, and perplexity is measured at full readout.
//!
//! The model was not trained for early exit or looping. These numbers describe the
//! stored weights; they bound what is available without training and say nothing
//! about a model trained with exit heads or loop-aware objectives.
//!
//! ```text
//! cargo run --release --example depth_probe -- \
//!     --model-path models/SmolLM-135M \
//!     --tokens target/eval/wikitext2_test.SmolLM-135M.u32
//! ```

use anyhow::{bail, Context, Result};
use clap::Parser;
use serde_json::{json, Value};
use std::path::PathBuf;

use thme::inference::{read_token_file, ArchConfig, WeightPrecision};
use thme::parser::ModelImporter;

#[derive(Parser, Debug)]
#[command(
    name = "depth_probe",
    about = "Measured per-depth readout and layer-schedule experiments"
)]
struct Args {
    /// Model directory or file with on-disk weights
    #[arg(long)]
    model_path: PathBuf,

    /// Token file (little-endian u32 ids) from scripts/prepare_eval_tokens.py
    #[arg(long)]
    tokens: PathBuf,

    /// Tokens per forward pass
    #[arg(long, default_value_t = 1024)]
    eval_context: usize,

    /// Leading corpus tokens to evaluate
    #[arg(long, default_value_t = 16384)]
    eval_max_tokens: usize,

    /// Where to write the machine-readable results
    #[arg(long, default_value = "target/depth_probe.json")]
    report_json: PathBuf,
}

fn windows<'a>(tokens: &'a [u32], args: &Args) -> Vec<&'a [u32]> {
    let used = args.eval_max_tokens.min(tokens.len());
    tokens[..used]
        .chunks(args.eval_context)
        .filter(|w| w.len() >= 2)
        .collect()
}

fn main() -> Result<()> {
    let args = Args::parse();
    if args.eval_context < 2 {
        bail!("Context length must be at least 2");
    }
    let source = ModelImporter::import(&args.model_path)
        .with_context(|| format!("Failed to import model from {:?}", args.model_path))?;
    let tokens = read_token_file(&args.tokens)?;
    let config = ArchConfig::detect(&source, &args.model_path)?;
    let model = config.load(&source, WeightPrecision::AsStored)?;
    let depth = model.num_layers();
    let windows = windows(&tokens, &args);
    if windows.is_empty() {
        bail!("Corpus has too few tokens to score anything");
    }
    println!(
        "Depth probe on '{}': {} layers, {} windows of up to {} tokens",
        source.name,
        depth,
        windows.len(),
        args.eval_context
    );

    // Part A: read out after every layer.
    let schedule: Vec<usize> = (0..depth).collect();
    let exits: Vec<usize> = (0..=depth).collect();
    let mut nll_sum = vec![0.0f64; depth + 1];
    let mut agree_final = vec![0usize; depth + 1];
    let mut correct = vec![0usize; depth + 1];
    // settled[e] counts tokens whose top prediction is final from depth e onward.
    let mut settled = vec![0usize; depth + 1];
    let mut scored = 0usize;
    for (index, window) in windows.iter().enumerate() {
        let readouts = model.forward_scheduled(window, &schedule, &exits)?;
        let last = &readouts[depth];
        let positions = last.top1.len();
        for position in 0..positions {
            let final_top = last.top1[position];
            let target = window[position + 1];
            let mut settle = depth;
            for (e, readout) in readouts.iter().enumerate().rev() {
                if readout.top1[position] == final_top {
                    settle = e;
                } else {
                    break;
                }
            }
            settled[settle] += 1;
            for (e, readout) in readouts.iter().enumerate() {
                nll_sum[e] += f64::from(readout.nll[position]);
                agree_final[e] += usize::from(readout.top1[position] == final_top);
                correct[e] += usize::from(readout.top1[position] == target);
            }
        }
        scored += positions;
        println!("  exit readout: window {}/{}", index + 1, windows.len());
    }
    let mut cumulative = 0usize;
    let mut exit_rows: Vec<Value> = Vec::new();
    println!("\n  layers  perplexity     top1==final  top1 correct  settled by here");
    for e in 0..=depth {
        cumulative += settled[e];
        let ppl = (nll_sum[e] / scored as f64).exp();
        let agree = agree_final[e] as f64 / scored as f64;
        let accuracy = correct[e] as f64 / scored as f64;
        let settled_fraction = cumulative as f64 / scored as f64;
        println!(
            "  {e:>6}  {ppl:>12.3}  {:>10.1}%  {:>10.1}%  {:>12.1}%",
            100.0 * agree,
            100.0 * accuracy,
            100.0 * settled_fraction
        );
        exit_rows.push(json!({
            "layers_applied": e,
            "perplexity": ppl,
            "top1_agrees_with_final": agree,
            "top1_accuracy": accuracy,
            "settled_fraction": settled_fraction,
        }));
    }

    // Part B: repeat or skip layers with no retraining.
    let block_start = depth / 4;
    let block_end = depth - depth / 4;
    let middle = depth / 2;
    let plain: Vec<usize> = (0..depth).collect();
    let with_block = |times: usize| -> Vec<usize> {
        let mut s: Vec<usize> = (0..block_start).collect();
        for _ in 0..times {
            s.extend(block_start..block_end);
        }
        s.extend(block_end..depth);
        s
    };
    let schedules: Vec<(String, String, Vec<usize>)> = vec![
        (
            "plain".into(),
            "Every layer once (reference)".into(),
            plain.clone(),
        ),
        (
            "loop_middle_block_x2".into(),
            format!("Layers {block_start}..{block_end} applied twice"),
            with_block(2),
        ),
        (
            "loop_middle_block_x3".into(),
            format!("Layers {block_start}..{block_end} applied three times"),
            with_block(3),
        ),
        (
            "repeat_single_layer".into(),
            format!("Layer {middle} applied twice"),
            plain
                .iter()
                .flat_map(|&l| if l == middle { vec![l, l] } else { vec![l] })
                .collect(),
        ),
        (
            "skip_single_layer".into(),
            format!("Layer {middle} skipped"),
            plain.iter().copied().filter(|&l| l != middle).collect(),
        ),
        (
            "skip_middle_block".into(),
            format!("Layers {block_start}..{block_end} skipped"),
            with_block(0),
        ),
    ];
    println!("\n  Layer-schedule experiments (no retraining):");
    let mut schedule_rows: Vec<Value> = Vec::new();
    let mut reference: Option<f64> = None;
    for (id, description, schedule) in &schedules {
        let mut total = 0.0f64;
        let mut count = 0usize;
        for window in &windows {
            let readout = model.forward_scheduled(window, schedule, &[schedule.len()])?;
            total += readout[0].nll.iter().map(|&v| f64::from(v)).sum::<f64>();
            count += readout[0].nll.len();
        }
        let ppl = (total / count as f64).exp();
        let base = *reference.get_or_insert(ppl);
        println!(
            "    {id:<24} PPL {ppl:>12.3}  Δ {:>+12.3}  ({} layer applications)",
            ppl - base,
            schedule.len()
        );
        schedule_rows.push(json!({
            "id": id,
            "description": description,
            "layer_applications": schedule.len(),
            "perplexity": ppl,
            "delta_ppl": ppl - base,
        }));
    }

    let corpus: Value = std::fs::read_to_string(args.tokens.with_extension("json"))
        .ok()
        .and_then(|text| serde_json::from_str(&text).ok())
        .unwrap_or(Value::Null);
    let out = json!({
        "thme_version": env!("CARGO_PKG_VERSION"),
        "generated_unix_secs": std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0),
        "model_name": source.name,
        "num_layers": depth,
        "corpus": corpus,
        "context_len": args.eval_context,
        "corpus_tokens_used": args.eval_max_tokens.min(tokens.len()),
        "scored_tokens": scored,
        "notes": [
            "Measured with the THME reference CPU forward pass on stored weights; no retraining.",
            "Intermediate readouts apply the model's own final norm and output head to the state at that depth.",
            "The model was not trained for early exit or looping; these are diagnostics, not the behaviour of a model trained for either.",
            "settled_fraction: share of tokens whose top prediction equals the full-depth top prediction at this depth and every later depth.",
        ],
        "exit_readout": exit_rows,
        "schedule_experiments": schedule_rows,
    });
    if let Some(parent) = args.report_json.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(
        &args.report_json,
        serde_json::to_string_pretty(&out)? + "\n",
    )
    .with_context(|| format!("Failed to write {:?}", args.report_json))?;
    println!("\nResults written to {:?}", args.report_json);
    Ok(())
}
