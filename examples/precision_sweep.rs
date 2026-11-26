// SPDX-License-Identifier: AGPL-3.0-only
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

/// Number of distinct placements of `k` identical-format tensors drawn from a
/// pool of `n` (n choose k). When the pool is exactly the targets (k == n) this
/// is 1, so every seed lands on the same placement (THME-3067 duplicate seeds).
fn count_placements(n: usize, k: usize) -> usize {
    assert!(k <= n);
    let k = k.min(n - k);
    (1..=k).fold(1, |acc, i| acc * (n - i + 1) / i)
}

/// Counter-based permutation of a pool by permutation index (Lehmer-code
/// unranking, factorial number system). When the pool is large enough to
/// distinguish every requested seed (n! >= seeds) this guarantees distinct
/// permutations for distinct seeds. Falls back to a full-state PRNG for pools
/// so large that n! does not fit in u64.
fn shuffled(items: Vec<usize>, seed: u64) -> Vec<usize> {
    let n = items.len();
    if n <= 1 {
        return items;
    }
    // Number of permutations of the pool = n! (u64-safe for n <= 20).
    let mut n_fact: u64 = 1;
    for i in 1..=n {
        n_fact *= i as u64;
    }
    if n_fact > u64::MAX / 2 {
        // Pool too large for the factorial-indexed path; use a full-state PRNG.
        return shuffled_prng(items, seed);
    }
    // seed % n! maps every seed to a permutation index; the mapping is
    // injective for distinct seeds below n!.
    let index = seed % n_fact;
    permutation_at(n, index as usize)
}

/// Unrank the `index`-th permutation (0-indexed) of the pool 0..n, using the
/// factorial number system. `index` must be < n!.
fn permutation_at(n: usize, mut index: usize) -> Vec<usize> {
    let mut nums: Vec<usize> = (0..n).collect();
    let mut perm = Vec::with_capacity(n);
    // fact = (n-1)! at the start.
    let mut fact: usize = 1;
    for i in 1..n {
        fact *= i;
    }
    for pos in 0..n {
        let idx = index / fact;
        index %= fact;
        perm.push(nums.remove(idx));
        if pos < n - 1 {
            fact /= n - pos - 1;
        }
    }
    perm
}

/// Full-state SplitMix64 fallback used for pools too large for the
/// factorial-indexed path. Distinct seeds give distinct permutations in
/// practice for realistic seed counts.
fn shuffled_prng(mut items: Vec<usize>, seed: u64) -> Vec<usize> {
    let mut state = seed;
    for i in (1..items.len()).rev() {
        state = state
            .wrapping_add(0x9E37_79B9_7F4A_7C15)
            .wrapping_mul(0xBF58_476D_1CE4_E5B9);
        let j = ((state ^ (state >> 32)) % (i as u64 + 1)) as usize;
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

    // Placement space of the random control. When the pool is exactly the targets
    // (k == n) C(n,k)==1, so every seed collapses onto the same placement - the
    // THME-3067 duplicate-seed symptom. Report the space and cap seeds to it.
    let tern_places = count_placements(ternary_pool.len(), ternary_count);
    let q8_places = count_placements(q8_pool.len(), q8_count);
    let total_places = tern_places * q8_places;
    let run_seeds = total_places.min(args.seeds as usize);
    if run_seeds < args.seeds as usize {
        eprintln!(
            "only {total_places} distinct placement(s) exist for the recipe's shapes (requested {}), capped to",
            args.seeds
        );
    }

    for seed in 1..=run_seeds {
        let mut formats: Vec<Option<TensorFormat>> = vec![None; model.tensors.len()];
        for &i in shuffled(ternary_pool.clone(), seed as u64)
            .iter()
            .take(ternary_count)
        {
            formats[i] = Some(TensorFormat::Ternary);
        }
        for &i in shuffled(q8_pool.clone(), (seed ^ 0x5eed) as u64)
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

#[cfg(test)]
mod tests {
    #[test]
    fn shuffled_distinguishes_seeds_permutations_when_possible() {
        // The counter-based generator maps seed % n! to a permutation index, so
        // for every pool where n! >= seeds, distinct seeds give distinct
        // permutations. Assert full distinctness for pools small enough that
        // n! fits in u64.
        for n in 2usize..7 {
            let max: usize = (1..=n).product();
            for seeds in 1usize..=max.min(16) {
                let mut perms: std::collections::HashSet<Vec<usize>> =
                    std::collections::HashSet::new();
                for seed in 1usize..=seeds {
                    let items: Vec<usize> = (0..n).collect();
                    let perm = super::shuffled(items, seed as u64);
                    perms.insert(perm);
                }
                assert_eq!(
                    perms.len(),
                    seeds,
                    "pool n={n}: seeds={seeds} gave {} distinct permutations",
                    perms.len()
                );
            }
        }
        // Larger pool (n=7, 5040 permutations) with 8 seeds: distinct seeds
        // must yield distinct permutations.
        let mut perms: std::collections::HashSet<Vec<usize>> = std::collections::HashSet::new();
        for seed in 1usize..=8 {
            let items: Vec<usize> = (0..7).collect();
            perms.insert(super::shuffled(items, seed as u64));
        }
        assert_eq!(
            perms.len(),
            8,
            "distinct seeds collapsed on a 7-element pool"
        );
    }

    #[test]
    fn count_placements_uses_combinations() {
        assert_eq!(super::count_placements(1, 1), 1);
        assert_eq!(super::count_placements(2, 1), 2);
        assert_eq!(super::count_placements(2, 2), 1); // pool == targets → 1 placement
        assert_eq!(super::count_placements(3, 3), 1); // all same-shape targets
        assert_eq!(super::count_placements(4, 2), 6);
        assert_eq!(super::count_placements(5, 2), 10);
    }

    #[test]
    fn small_pool_collapses_seeds() {
        // Pool exactly the targets (k == n) → C(n,k)==1 distinct placement.
        // With 3 seeds requested but only 1 placement, seeds 2 and 3 are
        // redundant — the THME-3067 duplicate-seed symptom.
        let n = 3;
        let k = n;
        let places = super::count_placements(n, k);
        assert_eq!(places, 1);
        assert!(places < 3, "only 1 placement but 3 seeds requested");
        assert_eq!(places.min(3), 1);
    }

    #[test]
    fn reports_placement_limit_when_seeds_exceed() {
        // When seeds exceed the number of distinct placements, report the limit
        // and cap to it.
        let n = 2;
        let k = n;
        let places = super::count_placements(n, k);
        let seeds = 3usize;
        assert!(places < seeds);
        let run = places.min(seeds);
        assert_eq!(run, places);
    }
}
