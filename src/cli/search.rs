// SPDX-License-Identifier: AGPL-3.0-only
//! `thme search`: find a mixed-precision recipe from measured costs.
//!
//! 1. The token file is split into a calibration slice (used for every search decision) and a
//!    disjoint validation slice (used only for the final report).
//! 2. Measured cost table: for each group of linear tensors (a layer, or a single tensor) and
//!    each candidate format, the perplexity cost of putting only that group in that format.
//!    The table is cached on disk per model, corpus slice, granularity and format set.
//! 3. Greedy allocation: repeatedly lower the group whose next format saves the most bytes per
//!    unit of measured cost, until the nominal size budget is met.
//! 4. Global refinement: costs do not add up, so the whole candidate is measured; while it
//!    fails the gate, the most recent move is reverted and the candidate re-measured.
//! 5. Validation: the source, the searched candidate and random placements of the same
//!    formats over same-sized groups are measured on the validation slice.
//!
//! The result is written as a measured report (same schema as the verification pipeline) and
//! optionally as a recipe that `thme evaluate` can re-check.

use crate::cli::evaluate::{judge_reports, variant_json, MEASURED_METRICS};
use anyhow::{bail, Context, Result};
use clap::Args;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};

use crate::inference::{
    evaluate_perplexity, read_token_file, ArchConfig, PerplexityConfig, PerplexityReport,
    WeightPrecision,
};
use crate::ir::{ModelIR, TensorFormat, TensorRole};
use crate::parser::ModelImporter;

#[derive(Args, Debug, Clone)]
pub struct SearchArgs {
    /// Model directory or file with on-disk weights (LLaMA family or GPT-NeoX)
    #[arg(value_name = "MODEL_PATH")]
    pub model: PathBuf,

    /// Token file (little-endian u32) from scripts/prepare_eval_tokens.py
    #[arg(long, value_name = "TOKENS")]
    pub tokens: PathBuf,

    /// Target nominal size of the linear tensors, as a fraction of their stored size
    #[arg(long, default_value_t = 0.5)]
    pub budget: f64,

    /// Candidate formats, comma-separated: fp16, q8_0, q4_0, ternary
    #[arg(long, default_value = "q8_0,q4_0")]
    pub formats: String,

    /// Group linear tensors by `layer` or treat each `tensor` separately
    #[arg(long, default_value = "layer")]
    pub granularity: String,

    /// Leading tokens used for search decisions (calibration slice)
    #[arg(long, default_value_t = 8192)]
    pub calib_tokens: usize,

    /// Tokens after the calibration slice used for the final report (validation slice)
    #[arg(long, default_value_t = 16384)]
    pub eval_tokens: usize,

    /// Tokens per forward pass
    #[arg(long, default_value_t = 1024)]
    pub eval_context: usize,

    /// Maximum allowed perplexity increase (the default gate's absolute margin)
    #[arg(long, default_value_t = 0.45)]
    pub max_ppl_delta: f64,

    /// Quality gate file (JSON: name, version, criteria); replaces --max-ppl-delta
    #[arg(long, conflicts_with = "max_ppl_delta", value_name = "GATE_JSON")]
    pub gate: Option<PathBuf>,

    /// Random matched-placement controls measured on the validation slice
    #[arg(long, default_value_t = 5)]
    pub controls: u64,

    /// Measured cost table cache (reused when model, slice, granularity and formats match)
    #[arg(long, default_value = "target/cost_table.json")]
    pub cost_table: PathBuf,

    /// Where to write the measured report
    #[arg(long, default_value = "target/search_report.json")]
    pub report: PathBuf,

    /// Also write the searched assignment as a recipe JSON
    #[arg(long)]
    pub emit_recipe: Option<PathBuf>,

    /// Extra budgets for a measured size-vs-quality Pareto front, comma-separated (e.g. 0.9,0.7,0.3)
    #[arg(long)]
    pub pareto_budgets: Option<String>,

    /// Tokens after the validation slice held out as a test slice (0 = none); never used by the search
    #[arg(long, default_value_t = 0)]
    pub test_tokens: usize,

    /// Allocation solver on the measured cost table: `greedy` (best saving per unit cost,
    /// one step at a time) or `dp` (exact minimum predicted cost under the budget)
    #[arg(long, default_value = "greedy")]
    pub solver: String,

    /// Size grid for `--solver dp`: the stored linear size is split into this many units and
    /// each option's size is rounded up to a unit, so every plan also meets the real budget
    #[arg(long, default_value_t = 10_000)]
    pub dp_grid: u64,
}

const LINEAR: [TensorRole; 8] = [
    TensorRole::AttentionQ,
    TensorRole::AttentionK,
    TensorRole::AttentionV,
    TensorRole::AttentionOut,
    TensorRole::AttentionQkv,
    TensorRole::MlpGate,
    TensorRole::MlpUp,
    TensorRole::MlpDown,
];

/// A set of tensors that always share one format.
#[derive(Debug, Clone)]
pub struct Group {
    pub name: String,
    pub tensors: Vec<usize>,
    pub params: usize,
    pub stored_bits: f64,
}

pub fn parse_formats(list: &str) -> Result<Vec<TensorFormat>> {
    let mut formats = Vec::new();
    for item in list.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        formats.push(match item.to_ascii_lowercase().as_str() {
            "fp16" | "f16" => TensorFormat::FP16,
            "q8_0" | "q8" => TensorFormat::Q8_0,
            "q4_0" | "q4" => TensorFormat::Q4_0,
            "q4_k" => TensorFormat::Q4_K,
            "q5_k" => TensorFormat::Q5_K,
            "q6_k" => TensorFormat::Q6_K,
            "ternary" | "tq1_0" => TensorFormat::Ternary,
            other => bail!(
                "Unknown format '{other}'; supported: fp16, q8_0, q6_k, q5_k, q4_k, q4_0, ternary"
            ),
        });
    }
    if formats.is_empty() {
        bail!("No candidate formats given");
    }
    // Highest precision first, so each step down saves bytes.
    formats.sort_by(|a, b| b.bits_per_weight().total_cmp(&a.bits_per_weight()));
    formats.dedup();
    Ok(formats)
}

pub fn build_groups(model: &ModelIR, granularity: &str) -> Result<Vec<Group>> {
    let mut groups: Vec<Group> = Vec::new();
    for (index, tensor) in model.tensors.iter().enumerate() {
        if tensor.shape.len() != 2 || !LINEAR.contains(&tensor.role) {
            continue;
        }
        let key = match granularity {
            "layer" => match tensor.layer_index {
                Some(l) => format!("layer {l}"),
                None => continue,
            },
            "tensor" => tensor.name.clone(),
            other => bail!("Unknown granularity '{other}'; use layer or tensor"),
        };
        let bits = 8.0 * tensor.byte_size as f64;
        match groups.iter_mut().find(|g| g.name == key) {
            Some(group) => {
                group.tensors.push(index);
                group.params += tensor.numel;
                group.stored_bits += bits;
            }
            None => groups.push(Group {
                name: key,
                tensors: vec![index],
                params: tensor.numel,
                stored_bits: bits,
            }),
        }
    }
    if groups.is_empty() {
        bail!("Model has no linear tensors to search over");
    }
    // Stable order: by layer, then name, independent of how the model file was parsed.
    groups.sort_by(|a, b| {
        let layer = |g: &Group| {
            model.tensors[g.tensors[0]]
                .layer_index
                .unwrap_or(usize::MAX)
        };
        layer(a).cmp(&layer(b)).then_with(|| a.name.cmp(&b.name))
    });
    Ok(groups)
}

/// Nominal bits of the groups under an assignment (`None` = stored).
fn nominal_bits(groups: &[Group], assignment: &[Option<usize>], formats: &[TensorFormat]) -> f64 {
    groups
        .iter()
        .zip(assignment)
        .map(|(g, a)| match a {
            Some(f) => g.params as f64 * formats[*f].bits_per_weight(),
            None => g.stored_bits,
        })
        .sum()
}

struct Evaluator<'a> {
    arch: &'a ArchConfig,
    source: &'a ModelIR,
    groups: &'a [Group],
    formats: &'a [TensorFormat],
    context: usize,
}

impl Evaluator<'_> {
    /// The source model with an assignment applied as selected formats.
    fn candidate(&self, assignment: &[Option<usize>]) -> ModelIR {
        let mut model = self.source.clone();
        for tensor in &mut model.tensors {
            tensor.selected_format = TensorFormat::FP32; // keep as stored
        }
        for (group, slot) in self.groups.iter().zip(assignment) {
            if let Some(f) = slot {
                for &t in &group.tensors {
                    model.tensors[t].selected_format = self.formats[*f].clone();
                }
            }
        }
        model
    }

    fn measure(&self, assignment: &[Option<usize>], tokens: &[u32]) -> Result<PerplexityReport> {
        let model = self.candidate(assignment);
        let dense = self.arch.load(&model, WeightPrecision::SelectedFormat)?;
        let config = PerplexityConfig {
            context_len: self.context,
            max_tokens: None,
        };
        evaluate_perplexity(&*dense, tokens, &config, |_, _, _| {})
    }
}

/// Deterministic shuffle of assignments among groups of identical size.
fn matched_random(groups: &[Group], assignment: &[Option<usize>], seed: u64) -> Vec<Option<usize>> {
    let mut out = assignment.to_vec();
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
    let mut sizes: Vec<usize> = groups.iter().map(|g| g.params).collect();
    sizes.sort_unstable();
    sizes.dedup();
    for size in sizes {
        let members: Vec<usize> = (0..groups.len())
            .filter(|&i| groups[i].params == size)
            .collect();
        let mut values: Vec<Option<usize>> = members.iter().map(|&i| assignment[i]).collect();
        for i in (1..values.len()).rev() {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            values.swap(i, ((state >> 33) as usize) % (i + 1));
        }
        for (&m, v) in members.iter().zip(values) {
            out[m] = v;
        }
    }
    out
}

fn cost_table_key(
    model: &ModelIR,
    groups: &[Group],
    args: &SearchArgs,
    formats: &[TensorFormat],
    calib: usize,
) -> Value {
    json!({
        "model": model.name,
        "groups": groups.iter().map(|g| g.name.clone()).collect::<Vec<_>>(),
        "tensors": model.tensors.len(),
        "granularity": args.granularity,
        "formats": formats.iter().map(|f| f.display_name().to_string()).collect::<Vec<_>>(),
        "calibration_tokens": calib,
        "context": args.eval_context,
        "token_file_bytes": std::fs::metadata(&args.tokens).map(|m| m.len()).unwrap_or(0),
    })
}

fn load_or_measure_costs(
    eval: &Evaluator<'_>,
    calib: &[u32],
    key: &Value,
    path: &Path,
    base_ppl: f64,
) -> Result<Vec<Vec<f64>>> {
    if let Ok(text) = std::fs::read_to_string(path) {
        if let Ok(cached) = serde_json::from_str::<Value>(&text) {
            if &cached["key"] == key {
                if let Ok(costs) = serde_json::from_value::<Vec<Vec<f64>>>(cached["costs"].clone())
                {
                    println!("  reusing measured cost table {path:?}");
                    return Ok(costs);
                }
            }
        }
    }
    let total = eval.groups.len() * eval.formats.len();
    let mut costs = vec![vec![0.0; eval.formats.len()]; eval.groups.len()];
    let mut done = 0;
    for (g, row) in costs.iter_mut().enumerate() {
        for (f, cell) in row.iter_mut().enumerate() {
            let mut assignment = vec![None; eval.groups.len()];
            assignment[g] = Some(f);
            *cell = eval.measure(&assignment, calib)?.perplexity - base_ppl;
            done += 1;
            println!(
                "  cost [{done:>3}/{total}] {:<28} {:<16} Δ {:+.4}",
                eval.groups[g].name,
                eval.formats[f].display_name(),
                *cell
            );
        }
    }
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(
        path,
        serde_json::to_string_pretty(&json!({ "key": key, "costs": costs }))? + "\n",
    )
    .with_context(|| format!("Failed to write {path:?}"))?;
    Ok(costs)
}

/// A format choice per group (`None` = stored) and the moves that produced it, in order.
pub type Allocation = (Vec<Option<usize>>, Vec<(usize, Option<usize>)>);

/// Exact allocation on measured costs (`engine::alloc`): the assignment of minimum summed
/// cost whose nominal size fits `budget_bits`, with sizes rounded up to a grid of `grid`
/// units of the stored size. Only formats smaller than a group's stored size are options.
/// Moves are ordered from best to worst saving per unit cost, so global refinement reverts
/// the least efficient choice first.
pub fn exact(
    groups: &[Group],
    formats: &[TensorFormat],
    costs: &[Vec<f64>],
    budget_bits: f64,
    grid: u64,
) -> Result<Allocation> {
    use crate::engine::alloc::{min_cost_choice, Choice};
    if grid == 0 {
        bail!("--dp-grid must be at least 1");
    }
    let stored: f64 = groups.iter().map(|g| g.stored_bits).sum();
    if stored <= budget_bits {
        return Ok((vec![None; groups.len()], Vec::new()));
    }
    let unit = stored / grid as f64;
    let units = |bits: f64| (bits / unit - 1e-9).ceil().max(0.0) as u64;
    let mut options: Vec<Vec<(Option<usize>, Choice)>> = Vec::with_capacity(groups.len());
    for (g, group) in groups.iter().enumerate() {
        let mut row = vec![(
            None,
            Choice {
                weight: units(group.stored_bits),
                cost: 0.0,
            },
        )];
        for (f, format) in formats.iter().enumerate() {
            let bits = group.params as f64 * format.bits_per_weight();
            if bits < group.stored_bits {
                row.push((
                    Some(f),
                    Choice {
                        weight: units(bits),
                        cost: costs[g][f],
                    },
                ));
            }
        }
        options.push(row);
    }
    let items: Vec<Vec<Choice>> = options
        .iter()
        .map(|row| row.iter().map(|(_, c)| *c).collect())
        .collect();
    let capacity = (budget_bits / unit + 1e-9).floor().max(0.0) as u64;
    let plan = min_cost_choice(&items, capacity).map_err(|e| {
        anyhow::anyhow!("{e} (budget {budget_bits:.0} bits on a grid of {grid} units)")
    })?;
    let assignment: Vec<Option<usize>> = plan
        .choice
        .iter()
        .enumerate()
        .map(|(g, &o)| options[g][o].0)
        .collect();
    let mut moves: Vec<(f64, usize)> = assignment
        .iter()
        .enumerate()
        .filter_map(|(g, a)| {
            a.map(|f| {
                let saving =
                    groups[g].stored_bits - groups[g].params as f64 * formats[f].bits_per_weight();
                (costs[g][f] / saving, g)
            })
        })
        .collect();
    moves.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    Ok((
        assignment,
        moves.into_iter().map(|(_, g)| (g, None)).collect(),
    ))
}

/// Runs the solver named by `--solver`.
fn allocate(
    args: &SearchArgs,
    groups: &[Group],
    formats: &[TensorFormat],
    costs: &[Vec<f64>],
    budget_bits: f64,
) -> Result<Allocation> {
    match args.solver.as_str() {
        "greedy" => Ok(greedy(groups, formats, costs, budget_bits)),
        "dp" => exact(groups, formats, costs, budget_bits, args.dp_grid),
        other => bail!("Unknown --solver '{other}'; use greedy or dp"),
    }
}

/// Greedy allocation on measured costs. Returns the assignment and the order of moves.
pub fn greedy(
    groups: &[Group],
    formats: &[TensorFormat],
    costs: &[Vec<f64>],
    budget_bits: f64,
) -> Allocation {
    let mut assignment: Vec<Option<usize>> = vec![None; groups.len()];
    let mut moves = Vec::new();
    while nominal_bits(groups, &assignment, formats) > budget_bits {
        let mut best: Option<(f64, usize, usize)> = None;
        for (g, group) in groups.iter().enumerate() {
            let (current_bits, current_cost) = match assignment[g] {
                Some(f) => (
                    group.params as f64 * formats[f].bits_per_weight(),
                    costs[g][f],
                ),
                None => (group.stored_bits, 0.0),
            };
            let next = assignment[g].map_or(0, |f| f + 1);
            for f in next..formats.len() {
                let bits = group.params as f64 * formats[f].bits_per_weight();
                if bits >= current_bits {
                    continue;
                }
                let saving = current_bits - bits;
                let cost = (costs[g][f] - current_cost).max(1e-6);
                let score = saving / cost;
                // Scores within a relative 1e-9 tie, broken by group order, so a cost table that
                // round-trips through JSON (last-bit float changes) yields the same plan.
                if best.is_none_or(|(s, _, _)| score > s * (1.0 + 1e-9)) {
                    best = Some((score, g, f));
                }
                break; // only the next lower format is a candidate move
            }
        }
        let Some((_, g, f)) = best else { break };
        moves.push((g, assignment[g]));
        assignment[g] = Some(f);
    }
    (assignment, moves)
}

pub fn search(args: &SearchArgs) -> Result<Value> {
    if !(0.0..=1.0).contains(&args.budget) {
        bail!("--budget must be between 0 and 1");
    }
    let gate = crate::quality::gate::resolve(args.gate.as_deref(), args.max_ppl_delta)?;
    gate.require_metrics(&MEASURED_METRICS, "thme search")?;
    let formats = parse_formats(&args.formats)?;
    let source = ModelImporter::import(&args.model)
        .with_context(|| format!("Failed to import {:?}", args.model))?;
    if source.tensors.iter().any(|t| t.source.is_none()) {
        bail!("Model has tensors without on-disk payloads; search needs real weights");
    }
    let arch = ArchConfig::detect(&source, &args.model)?;
    let groups = build_groups(&source, &args.granularity)?;
    for group in &groups {
        for f in &formats {
            let block = match f {
                TensorFormat::Ternary
                | TensorFormat::Q4_K
                | TensorFormat::Q5_K
                | TensorFormat::Q6_K => 256,
                TensorFormat::Q8_0 | TensorFormat::Q4_0 => 32,
                _ => 1,
            };
            if let Some(&t) = group
                .tensors
                .iter()
                .find(|&&t| !source.tensors[t].numel.is_multiple_of(block))
            {
                bail!(
                    "Tensor '{}' cannot be stored as {} (block size {block})",
                    source.tensors[t].name,
                    f.display_name()
                );
            }
        }
    }

    let tokens = read_token_file(&args.tokens)?;
    if tokens.len() < args.calib_tokens + 2 * args.eval_context.max(2) {
        bail!(
            "Token file has {} tokens; need more than the {} calibration tokens",
            tokens.len(),
            args.calib_tokens
        );
    }
    let calib = &tokens[..args.calib_tokens];
    let validation_end = (args.calib_tokens + args.eval_tokens).min(tokens.len());
    let validation = &tokens[args.calib_tokens..validation_end];
    let eval = Evaluator {
        arch: &arch,
        source: &source,
        groups: &groups,
        formats: &formats,
        context: args.eval_context,
    };

    println!(
        "Searching {} ({}): {} groups by {}, formats {}, budget {:.0}% of linear storage",
        source.name,
        arch.family(),
        groups.len(),
        args.granularity,
        formats
            .iter()
            .map(|f| f.display_name())
            .collect::<Vec<_>>()
            .join(", "),
        100.0 * args.budget
    );
    let stored_bits: f64 = groups.iter().map(|g| g.stored_bits).sum();
    let budget_bits = stored_bits * args.budget;
    let none = vec![None; groups.len()];
    let calib_base_report = eval.measure(&none, calib)?;
    let calib_base = calib_base_report.perplexity;
    println!(
        "  calibration baseline PPL {calib_base:.4} on {} tokens",
        calib.len()
    );
    let key = cost_table_key(&source, &groups, args, &formats, calib.len());
    let costs = load_or_measure_costs(&eval, calib, &key, &args.cost_table, calib_base)?;

    let (mut assignment, mut moves) = allocate(args, &groups, &formats, &costs, budget_bits)?;
    let predicted: f64 = assignment
        .iter()
        .enumerate()
        .filter_map(|(g, a)| a.map(|f| costs[g][f]))
        .sum();
    let mut calib_report = eval.measure(&assignment, calib)?;
    let mut measured = calib_report.perplexity - calib_base;
    println!(
        "  {}: predicted Δ {predicted:+.4} (sum of group costs), measured Δ {measured:+.4}",
        args.solver
    );
    let mut reverted = 0usize;
    while !judge_reports(&gate, &calib_base_report, &calib_report).accepted {
        let Some((g, previous)) = moves.pop() else {
            break;
        };
        assignment[g] = previous;
        reverted += 1;
        calib_report = eval.measure(&assignment, calib)?;
        measured = calib_report.perplexity - calib_base;
        println!("  refine: reverted {} -> Δ {measured:+.4}", groups[g].name);
    }
    let achieved = nominal_bits(&groups, &assignment, &formats);
    let budget_met = achieved <= budget_bits + 1e-6;

    println!("  validating on {} held-out tokens", validation.len());
    let base = eval.measure(&none, validation)?;
    let searched = eval.measure(&assignment, validation)?;
    let mut variants = vec![
        variant_json(
            "baseline",
            "Source weights exactly as stored on disk",
            &base,
            None,
            &gate,
        ),
        variant_json(
            "searched",
            &format!(
                "Searched assignment: {:.0}% of linear storage",
                100.0 * achieved / stored_bits
            ),
            &searched,
            Some(&base),
            &gate,
        ),
    ];
    for seed in 1..=args.controls {
        let shuffled = matched_random(&groups, &assignment, seed);
        if shuffled == assignment {
            continue;
        }
        let report = eval.measure(&shuffled, validation)?;
        variants.push(variant_json(
            &format!("random_matched_seed{seed}"),
            &format!(
                "Same formats as the search, placed at random over same-sized groups (seed {seed})"
            ),
            &report,
            Some(&base),
            &gate,
        ));
    }
    for v in &variants {
        println!(
            "  {:<22} PPL {:>10.4}  {}",
            v["id"].as_str().unwrap_or(""),
            v["perplexity"].as_f64().unwrap_or(f64::NAN),
            v["delta_ppl"]
                .as_f64()
                .map_or_else(String::new, |d| format!("Δ {d:+.4}"))
        );
    }

    // Measured Pareto front across extra budgets, reusing the cost table.
    let control_deltas = |variants: &[Value]| -> Vec<f64> {
        variants
            .iter()
            .filter(|v| v["id"].as_str().is_some_and(|id| id.starts_with("random_")))
            .filter_map(|v| v["delta_ppl"].as_f64())
            .collect()
    };
    let mut pareto = vec![json!({
        "budget": args.budget,
        "achieved_fraction": achieved / stored_bits,
        "budget_met": budget_met,
        "validation_delta": searched.perplexity - base.perplexity,
        "control_deltas": control_deltas(&variants),
    })];
    if let Some(list) = &args.pareto_budgets {
        for item in list.split(',').map(str::trim).filter(|s| !s.is_empty()) {
            let budget: f64 = item
                .parse()
                .with_context(|| format!("Invalid budget '{item}' in --pareto-budgets"))?;
            if !(0.0..=1.0).contains(&budget) || (budget - args.budget).abs() < 1e-9 {
                continue;
            }
            let (mut a, mut mv) = allocate(args, &groups, &formats, &costs, stored_bits * budget)?;
            let mut r = eval.measure(&a, calib)?;
            while !judge_reports(&gate, &calib_base_report, &r).accepted {
                let Some((g, previous)) = mv.pop() else { break };
                a[g] = previous;
                r = eval.measure(&a, calib)?;
            }
            let got = nominal_bits(&groups, &a, &formats);
            let point = eval.measure(&a, validation)?;
            let mut controls = Vec::new();
            for seed in 1..=args.controls {
                let shuffled = matched_random(&groups, &a, seed);
                if shuffled != a {
                    controls
                        .push(eval.measure(&shuffled, validation)?.perplexity - base.perplexity);
                }
            }
            println!(
                "  pareto: budget {:.0}% -> {:.0}% of linear storage, validation Δ {:+.4}",
                100.0 * budget,
                100.0 * got / stored_bits,
                point.perplexity - base.perplexity
            );
            pareto.push(json!({
                "budget": budget,
                "achieved_fraction": got / stored_bits,
                "budget_met": got <= stored_bits * budget + 1e-6,
                "validation_delta": point.perplexity - base.perplexity,
                "control_deltas": controls,
            }));
        }
    }
    pareto.sort_by(|a, b| {
        b["achieved_fraction"]
            .as_f64()
            .unwrap_or(0.0)
            .total_cmp(&a["achieved_fraction"].as_f64().unwrap_or(0.0))
    });

    // Held-out test slice: measured once, after all search decisions.
    let test = if args.test_tokens > 1 {
        let end = (validation_end + args.test_tokens).min(tokens.len());
        let slice = &tokens[validation_end..end];
        if slice.len() < 2 {
            bail!("No tokens left for the test slice after calibration and validation");
        }
        let test_base = eval.measure(&none, slice)?;
        let test_searched = eval.measure(&assignment, slice)?;
        println!(
            "  test slice ({} tokens): Δ {:+.4}",
            slice.len(),
            test_searched.perplexity - test_base.perplexity
        );
        json!({
            "tokens": slice.len(),
            "baseline_perplexity": test_base.perplexity,
            "searched_perplexity": test_searched.perplexity,
            "delta_ppl": test_searched.perplexity - test_base.perplexity,
        })
    } else {
        Value::Null
    };

    let plan: Vec<Value> = groups
        .iter()
        .zip(&assignment)
        .map(|(g, a)| json!({
            "group": g.name,
            "format": a.map_or_else(|| "stored".to_string(), |f| formats[f].display_name().to_string()),
            "params": g.params,
        }))
        .collect();
    let corpus: Value = std::fs::read_to_string(args.tokens.with_extension("json"))
        .ok()
        .and_then(|text| serde_json::from_str(&text).ok())
        .unwrap_or(Value::Null);
    let searched_verdict = judge_reports(&gate, &base, &searched);
    let searched_passed = searched_verdict.accepted;
    let report = json!({
        "thme_version": env!("CARGO_PKG_VERSION"),
        "generated_unix_secs": std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0),
        "tool": "thme search",
        "model_name": source.name,
        "architecture": arch.family(),
        "source_payload_bytes": source.original_byte_size,
        "search": {
            "granularity": args.granularity,
            "formats": formats.iter().map(|f| f.display_name().to_string()).collect::<Vec<_>>(),
            "budget_fraction": args.budget,
            "solver": args.solver,
            "dp_grid": if args.solver == "dp" { json!(args.dp_grid) } else { Value::Null },
            "achieved_fraction": achieved / stored_bits,
            "budget_met": budget_met,
            "calibration_tokens": calib.len(),
            "validation_tokens": validation.len(),
            "predicted_delta_calibration": predicted,
            "measured_delta_calibration": measured,
            "moves_reverted": reverted,
            "pareto": pareto,
            "test": test,
            "plan": plan,
        },
        "recipe": {
            "name": format!("searched-{}", source.name),
            "format_counts": {},
            "estimated_bytes": ((source.original_byte_size as f64 * 8.0 - stored_bits + achieved) / 8.0) as u64,
            "estimated_bytes_note": "Computed from nominal bits per weight; export.artifact_bytes is the size of the written GGUF.",
        },
        "export": crate::cli::evaluate::measure_artifact(&eval.candidate(&assignment), None),
        "evaluation": {
            "measured": true,
            "corpus": corpus,
            "context_len": args.eval_context,
            "max_tokens": validation.len(),
            "max_ppl_delta": gate.max_ppl_delta(),
            "gate": gate.to_report(),
            "gate_verdict": searched_verdict,
            "gate_passed": searched_passed,
            "variants": variants,
        },
        "host": {
            "os": std::env::consts::OS,
            "arch": std::env::consts::ARCH,
            "logical_cpus": std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1),
            "backend": "THME reference CPU forward pass (f32, multithreaded)",
        },
    });
    if let Some(parent) = args.report.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&args.report, serde_json::to_string_pretty(&report)? + "\n")
        .with_context(|| format!("Failed to write {:?}", args.report))?;
    if let Some(path) = &args.emit_recipe {
        let operations: Vec<Value> = groups
            .iter()
            .zip(&assignment)
            .flat_map(|(g, a)| {
                a.map(|f| {
                    g.tensors
                        .iter()
                        .filter_map(|&t| {
                            let tensor = &source.tensors[t];
                            tensor.layer_index.map(|l| json!({
                                "type": "SetPrecision",
                                "params": {"start_layer": l, "end_layer": l, "role": tensor.role, "format": formats[f]},
                            }))
                        })
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default()
            })
            .collect();
        let recipe = json!({
            "name": format!("searched-{}", source.name),
            "author": "thme search",
            "target_model": source.name,
            "version": "1",
            "description": format!(
                "Measured-cost greedy search, budget {:.0}% of linear storage; validation Δ PPL {:+.4}",
                100.0 * args.budget,
                searched.perplexity - base.perplexity
            ),
            "operations": operations,
        });
        std::fs::write(path, serde_json::to_string_pretty(&recipe)? + "\n")
            .with_context(|| format!("Failed to write {path:?}"))?;
    }
    Ok(report)
}

pub fn execute(args: SearchArgs) -> Result<()> {
    let report = search(&args)?;
    println!("Report written to {:?}", args.report);
    if let Some(path) = &args.emit_recipe {
        println!("Recipe written to {path:?} (re-check with `thme evaluate`)");
    }
    if report["search"]["budget_met"] != Value::Bool(true) {
        println!("Note: the size budget was not met within the quality gate.");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::evaluate::tests::synthetic_checkpoint;

    fn group(name: &str, params: usize) -> Group {
        Group {
            name: name.to_string(),
            tensors: vec![],
            params,
            stored_bits: params as f64 * 32.0,
        }
    }

    #[test]
    fn greedy_lowers_the_cheapest_groups_first_and_meets_budget() {
        let groups = vec![group("a", 100), group("b", 100), group("c", 100)];
        let k = parse_formats("q4_k,q6_k,q5_k").unwrap();
        assert_eq!(
            k,
            vec![TensorFormat::Q6_K, TensorFormat::Q5_K, TensorFormat::Q4_K]
        );
        let formats = parse_formats("q4_0,q8_0").unwrap();
        assert_eq!(formats[0], TensorFormat::Q8_0);
        // b is cheap at both formats, a is expensive, c in between.
        let costs = vec![vec![5.0, 50.0], vec![0.1, 0.2], vec![1.0, 9.0]];
        let budget = 300.0 * 32.0 * 0.4;
        let (assignment, moves) = greedy(&groups, &formats, &costs, budget);
        assert!(nominal_bits(&groups, &assignment, &formats) <= budget);
        assert_eq!(moves[0].0, 1, "cheapest group moves first");
        assert_eq!(assignment[1], Some(1));
    }

    /// THME-3002 done-means: on the same cost table and budget, the exact solver's
    /// predicted cost is never above greedy's, and both meet the budget.
    #[test]
    fn exact_is_never_worse_than_greedy() {
        let formats = parse_formats("q8_0,q4_0").unwrap();
        let mut state = 99u64;
        let mut rand = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) as f64 / (1u64 << 31) as f64
        };
        let predicted = |a: &[Option<usize>], costs: &[Vec<f64>]| -> f64 {
            a.iter()
                .enumerate()
                .filter_map(|(g, f)| f.map(|f| costs[g][f]))
                .sum()
        };
        for case in 0..200 {
            let groups: Vec<Group> = (0..6)
                .map(|i| group(&format!("g{i}"), 64 * (1 + (rand() * 4.0) as usize)))
                .collect();
            let costs: Vec<Vec<f64>> = (0..6)
                .map(|_| {
                    let q8 = rand();
                    vec![q8, q8 + 4.0 * rand()]
                })
                .collect();
            let stored: f64 = groups.iter().map(|g| g.stored_bits).sum();
            let budget = stored * (0.15 + 0.8 * rand());
            let (gr, _) = greedy(&groups, &formats, &costs, budget);
            let (dp, moves) = exact(&groups, &formats, &costs, budget, 20_000).unwrap();
            assert!(
                nominal_bits(&groups, &dp, &formats) <= budget + 1e-6,
                "case {case}"
            );
            assert_eq!(moves.len(), dp.iter().filter(|a| a.is_some()).count());
            if nominal_bits(&groups, &gr, &formats) <= budget {
                assert!(
                    predicted(&dp, &costs) <= predicted(&gr, &costs) + 1e-9,
                    "case {case}: dp {} > greedy {}",
                    predicted(&dp, &costs),
                    predicted(&gr, &costs)
                );
            }
        }
        // Budget above the stored size: nothing to do.
        let groups = vec![group("a", 64)];
        assert_eq!(
            exact(&groups, &formats, &[vec![1.0, 2.0]], 1e12, 100)
                .unwrap()
                .0,
            vec![None]
        );
        // Below the smallest format: infeasible is an error, not a silent plan.
        assert!(exact(&groups, &formats, &[vec![1.0, 2.0]], 10.0, 100).is_err());
    }

    #[test]
    fn matched_random_keeps_format_counts() {
        let groups = vec![
            group("a", 10),
            group("b", 10),
            group("c", 10),
            group("d", 20),
        ];
        let assignment = vec![Some(0), None, Some(1), Some(1)];
        let shuffled = matched_random(&groups, &assignment, 3);
        assert_eq!(
            shuffled[3],
            Some(1),
            "a group of unique size keeps its format"
        );
        let mut a: Vec<_> = assignment[..3].to_vec();
        let mut b: Vec<_> = shuffled[..3].to_vec();
        a.sort();
        b.sort();
        assert_eq!(a, b);
    }

    #[test]
    fn search_runs_end_to_end_and_emits_a_recipe() {
        let dir = tempfile::tempdir().unwrap();
        let (model, tokens) = synthetic_checkpoint(dir.path());
        let recipe = dir.path().join("searched.json");
        let args = SearchArgs {
            model,
            tokens,
            budget: 0.3,
            formats: "q8_0,q4_0".into(),
            granularity: "layer".into(),
            calib_tokens: 96,
            eval_tokens: 104,
            eval_context: 32,
            max_ppl_delta: 1.0e9,
            controls: 2,
            cost_table: dir.path().join("costs.json"),
            report: dir.path().join("report.json"),
            emit_recipe: Some(recipe.clone()),
            pareto_budgets: Some("0.8,0.5".into()),
            test_tokens: 0,
            solver: "greedy".into(),
            dp_grid: 10_000,
            gate: None,
        };
        let mut dp_args = args.clone();
        let report = search(&args).unwrap();
        assert_eq!(report["search"]["budget_met"], Value::Bool(true));
        assert_eq!(report["search"]["solver"], json!("greedy"));
        // The exact solver runs through the same measured path on the cached cost table.
        dp_args.solver = "dp".into();
        dp_args.report = dir.path().join("report_dp.json");
        dp_args.emit_recipe = None;
        let dp_report = search(&dp_args).unwrap();
        assert_eq!(dp_report["search"]["budget_met"], Value::Bool(true));
        assert_eq!(dp_report["search"]["solver"], json!("dp"));
        assert!(
            dp_report["search"]["predicted_delta_calibration"]
                .as_f64()
                .unwrap()
                <= report["search"]["predicted_delta_calibration"]
                    .as_f64()
                    .unwrap()
                    + 1e-9
        );
        dp_args.solver = "simplex".into();
        assert!(search(&dp_args).is_err());
        assert_eq!(report["export"]["measured"], Value::Bool(true));
        assert!(report["export"]["artifact_bytes"].as_u64().unwrap() > 0);
        let pareto = report["search"]["pareto"].as_array().unwrap();
        assert_eq!(pareto.len(), 3);
        let fractions: Vec<f64> = pareto
            .iter()
            .map(|p| p["achieved_fraction"].as_f64().unwrap())
            .collect();
        assert!(
            fractions.windows(2).all(|w| w[0] >= w[1]),
            "sorted largest first"
        );
        assert!(report["search"]["achieved_fraction"].as_f64().unwrap() <= 0.3 + 1e-9);
        assert!(dir.path().join("costs.json").exists());
        let emitted: crate::surgery::recipe::TweakRecipe =
            serde_json::from_str(&std::fs::read_to_string(recipe).unwrap()).unwrap();
        assert!(!emitted.operations.is_empty());
        // Second run reuses the cached cost table and reaches the same plan.
        let again = search(&args).unwrap();
        assert_eq!(again["search"]["plan"], report["search"]["plan"]);
    }
}
