//! Measured precision sweep: how much perplexity each precision assignment actually costs.
//!
//! Every variant is evaluated with a real forward pass on the same tokens. Tensors
//! not targeted by a variant keep their stored values, so each row isolates the
//! effect of the tensors it names. No retraining or calibration is applied.
//!
//! Variant groups:
//! - `uniform`: one format everywhere it applies.
//! - `role`: one tensor role ternarized across all layers (measured role sensitivity).
//! - `layer`: all linear tensors of one layer ternarized (measured depth sensitivity).
//! - `recipe`: the production surgery recipe.
//! - `random`: the recipe's tensor counts and shapes, placed at random
//!   (the matched-proportion control for the recipe).
//!
//! ```text
//! cargo run --release --example precision_sweep -- \
//!     --model-path models/SmolLM-135M \
//!     --tokens target/eval/wikitext2_test.SmolLM-135M.u32 --layer-scan
//! ```

use anyhow::{bail, Context, Result};
use clap::Parser;
use serde_json::{json, Value};
use std::path::PathBuf;

use thme::inference::{
    evaluate_perplexity, read_token_file, ArchConfig, PerplexityConfig, WeightPrecision,
};
use thme::ir::{ModelIR, TensorFormat, TensorIR, TensorRole};
use thme::parser::ModelImporter;

#[derive(Parser, Debug)]
#[command(
    name = "precision_sweep",
    about = "Measured perplexity cost of precision assignments"
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

    /// Leading corpus tokens to evaluate per variant
    #[arg(long, default_value_t = 16384)]
    eval_max_tokens: usize,

    /// Number of random matched-proportion placements
    #[arg(long, default_value_t = 3)]
    seeds: u64,

    /// Also ternarize each layer in turn (one evaluation per layer)
    #[arg(long)]
    layer_scan: bool,

    /// Where to write the machine-readable results
    #[arg(long, default_value = "target/precision_sweep.json")]
    report_json: PathBuf,
}

const LINEAR_ROLES: [(TensorRole, &str); 8] = [
    (TensorRole::AttentionQ, "attention_q"),
    (TensorRole::AttentionQkv, "attention_qkv"),
    (TensorRole::AttentionK, "attention_k"),
    (TensorRole::AttentionV, "attention_v"),
    (TensorRole::AttentionOut, "attention_out"),
    (TensorRole::MlpGate, "mlp_gate"),
    (TensorRole::MlpUp, "mlp_up"),
    (TensorRole::MlpDown, "mlp_down"),
];

struct Variant {
    id: String,
    group: &'static str,
    description: String,
    /// Format per tensor index; `None` keeps the stored values.
    formats: Vec<Option<TensorFormat>>,
}

fn is_linear(tensor: &TensorIR) -> bool {
    tensor.shape.len() == 2 && LINEAR_ROLES.iter().any(|(role, _)| *role == tensor.role)
}

fn variant(
    model: &ModelIR,
    id: impl Into<String>,
    group: &'static str,
    description: impl Into<String>,
    pick: impl Fn(&TensorIR) -> Option<TensorFormat>,
) -> Variant {
    Variant {
        id: id.into(),
        group,
        description: description.into(),
        formats: model.tensors.iter().map(pick).collect(),
    }
}

/// Deterministic Fisher-Yates shuffle (64-bit LCG), so placements are reproducible.
fn shuffled(mut items: Vec<usize>, seed: u64) -> Vec<usize> {
    let mut state = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    for i in (1..items.len()).rev() {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let j = ((state >> 33) as usize) % (i + 1);
        items.swap(i, j);
    }
    items
}

fn build_variants(model: &ModelIR, args: &Args) -> Vec<Variant> {
    let last_inner = model.num_layers.saturating_sub(2);
    let in_recipe_range = |t: &TensorIR| t.layer_index.is_some_and(|l| l >= 1 && l <= last_inner);
    let recipe_pick = |t: &TensorIR| match t.role {
        TensorRole::MlpDown if is_linear(t) && in_recipe_range(t) => Some(TensorFormat::Ternary),
        TensorRole::AttentionQ if is_linear(t) && in_recipe_range(t) => Some(TensorFormat::Q8_0),
        _ => None,
    };

    let mut variants = vec![
        variant(
            model,
            "baseline",
            "baseline",
            "Stored weights, unchanged",
            |_| None,
        ),
        variant(
            model,
            "uniform_fp16",
            "uniform",
            "Every tensor as FP16",
            |_| Some(TensorFormat::FP16),
        ),
        variant(
            model,
            "uniform_q8_linear",
            "uniform",
            "Every linear tensor as Q8_0",
            |t| is_linear(t).then_some(TensorFormat::Q8_0),
        ),
        variant(
            model,
            "uniform_q4_0_linear",
            "uniform",
            "Every linear tensor as Q4_0",
            |t| is_linear(t).then_some(TensorFormat::Q4_0),
        ),
        variant(
            model,
            "q4_0_mlp_only",
            "uniform",
            "MLP tensors as Q4_0, attention as Q8_0",
            |t| match t.role {
                TensorRole::MlpGate | TensorRole::MlpUp | TensorRole::MlpDown if is_linear(t) => {
                    Some(TensorFormat::Q4_0)
                }
                _ if is_linear(t) => Some(TensorFormat::Q8_0),
                _ => None,
            },
        ),
        variant(
            model,
            "uniform_ternary_linear",
            "uniform",
            "Every linear tensor as ternary (TQ1_0)",
            |t| is_linear(t).then_some(TensorFormat::Ternary),
        ),
    ];

    for (role, name) in LINEAR_ROLES {
        if !model.tensors.iter().any(|t| is_linear(t) && t.role == role) {
            continue;
        }
        variants.push(variant(
            model,
            format!("role_ternary_{name}"),
            "role",
            format!("Only {name} tensors ternary, all layers"),
            move |t| (is_linear(t) && t.role == role).then_some(TensorFormat::Ternary),
        ));
    }

    variants.push(variant(
        model,
        "recipe",
        "recipe",
        format!(
            "Production recipe: mlp_down ternary and attention_q Q8_0 in layers 1..={last_inner}"
        ),
        recipe_pick,
    ));

    // Random control: same number of tensors, drawn from tensors of the same shape
    // as the recipe's targets, so the parameter count per format is identical.
    let recipe_formats: Vec<Option<TensorFormat>> = model.tensors.iter().map(recipe_pick).collect();
    let pool_for = |format: &TensorFormat| -> (usize, Vec<usize>) {
        let targeted: Vec<usize> = recipe_formats
            .iter()
            .enumerate()
            .filter(|(_, f)| f.as_ref() == Some(format))
            .map(|(i, _)| i)
            .collect();
        let shape = targeted.first().map(|&i| model.tensors[i].shape.clone());
        let pool = model
            .tensors
            .iter()
            .enumerate()
            .filter(|(_, t)| is_linear(t) && Some(&t.shape) == shape.as_ref())
            .map(|(i, _)| i)
            .collect();
        (targeted.len(), pool)
    };
    let (ternary_count, ternary_pool) = pool_for(&TensorFormat::Ternary);
    let (q8_count, q8_pool) = pool_for(&TensorFormat::Q8_0);
    for seed in 1..=args.seeds {
        let mut formats: Vec<Option<TensorFormat>> = vec![None; model.tensors.len()];
        for &i in shuffled(ternary_pool.clone(), seed)
            .iter()
            .take(ternary_count)
        {
            formats[i] = Some(TensorFormat::Ternary);
        }
        for &i in shuffled(q8_pool.clone(), seed ^ 0x5eed)
            .iter()
            .take(q8_count)
        {
            formats[i] = Some(TensorFormat::Q8_0);
        }
        variants.push(Variant {
            id: format!("random_matched_seed{seed}"),
            group: "random",
            description: format!(
                "{ternary_count} ternary and {q8_count} Q8_0 tensors of the recipe's shapes, placed at random (seed {seed})"
            ),
            formats,
        });
    }

    if args.layer_scan {
        for layer in 0..model.num_layers {
            variants.push(variant(
                model,
                format!("layer_ternary_{layer:02}"),
                "layer",
                format!("All linear tensors of layer {layer} ternary"),
                move |t| {
                    (is_linear(t) && t.layer_index == Some(layer)).then_some(TensorFormat::Ternary)
                },
            ));
        }
    }
    variants
}

fn main() -> Result<()> {
    let args = Args::parse();
    let source = ModelImporter::import(&args.model_path)
        .with_context(|| format!("Failed to import model from {:?}", args.model_path))?;
    if source.tensors.iter().any(|t| t.source.is_none()) {
        bail!("Model has tensors without on-disk payloads; a sweep needs real weights");
    }
    let tokens = read_token_file(&args.tokens)?;
    let eval = PerplexityConfig {
        context_len: args.eval_context,
        max_tokens: Some(args.eval_max_tokens),
    };
    let config = ArchConfig::detect(&source, &args.model_path)?;
    let variants = build_variants(&source, &args);
    println!(
        "Precision sweep on '{}': {} variants, first {} of {} corpus tokens, context {}",
        source.name,
        variants.len(),
        args.eval_max_tokens.min(tokens.len()),
        tokens.len(),
        args.eval_context
    );

    let total_params: usize = source.tensors.iter().map(|t| t.numel).sum();
    let mut baseline_ppl: Option<f64> = None;
    let mut rows: Vec<Value> = Vec::new();
    for (index, v) in variants.iter().enumerate() {
        let mut model = source.clone();
        let mut targeted_tensors = 0usize;
        let mut targeted_params = 0usize;
        let mut nominal_bits = 0.0f64;
        for (tensor, format) in model.tensors.iter_mut().zip(&v.formats) {
            // Untargeted tensors are evaluated exactly as stored.
            tensor.selected_format = format.clone().unwrap_or(TensorFormat::FP32);
            if format.is_some() {
                targeted_tensors += 1;
                targeted_params += tensor.numel;
            }
            let stored_bits = 8.0 * tensor.byte_size as f64 / tensor.numel.max(1) as f64;
            nominal_bits += tensor.numel as f64
                * format
                    .as_ref()
                    .map_or(stored_bits, TensorFormat::bits_per_weight);
        }
        let dense = config
            .load(&model, WeightPrecision::SelectedFormat)
            .with_context(|| format!("Failed to materialize variant '{}'", v.id))?;
        let report = evaluate_perplexity(&*dense, &tokens, &eval, |_, _, _| {})?;
        let base = *baseline_ppl.get_or_insert(report.perplexity);
        let delta = report.perplexity - base;
        println!(
            "  [{:>2}/{}] {:<28} PPL {:>12.4}  Δ {:>+12.4}  ({} tensors, {:.1}% of parameters)",
            index + 1,
            variants.len(),
            v.id,
            report.perplexity,
            delta,
            targeted_tensors,
            100.0 * targeted_params as f64 / total_params as f64
        );
        rows.push(json!({
            "id": v.id,
            "group": v.group,
            "description": v.description,
            "perplexity": report.perplexity,
            "delta_ppl": delta,
            "cross_entropy_nats": report.cross_entropy_nats,
            "scored_tokens": report.scored_tokens,
            "targeted_tensors": targeted_tensors,
            "targeted_params": targeted_params,
            "targeted_param_fraction": targeted_params as f64 / total_params as f64,
            "nominal_bits_per_weight": nominal_bits / total_params as f64,
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
        "num_parameters": total_params,
        "corpus": corpus,
        "context_len": args.eval_context,
        "corpus_tokens_used": args.eval_max_tokens.min(tokens.len()),
        "notes": [
            "Measured with the THME reference CPU forward pass; no retraining or calibration.",
            "Untargeted tensors keep their stored values.",
            "nominal_bits_per_weight is arithmetic from format bit-widths, not a measured artifact size.",
            "Ternary is the GGML TQ1_0 block format with an MSE-optimal scale per 256 values.",
        ],
        "variants": rows,
    });
    if let Some(parent) = args.report_json.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(
        &args.report_json,
        serde_json::to_string_pretty(&out)? + "\n",
    )
    .with_context(|| format!("Failed to write {:?}", args.report_json))?;
    println!("Results written to {:?}", args.report_json);
    Ok(())
}
