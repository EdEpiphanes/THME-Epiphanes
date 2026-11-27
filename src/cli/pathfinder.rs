// SPDX-License-Identifier: AGPL-3.0-only
//! `thme pathfinder`: header-only run planner for context, KV precision, and predicted memory.

use anyhow::{bail, Context, Result};
use clap::{Args, ValueEnum};
use std::path::PathBuf;

use crate::parser::ModelImporter;
use crate::profiling::run_memory::KvType;
use crate::profiling::run_plan::{plan_run, shape_from_ir, RunPlan};
use crate::profiling::shrink::{Fixed, RunSettings, ShrinkLimits};

const BYTES_PER_GIB: f64 = 1_073_741_824.0;
const DEFAULT_MAX_CONTEXT: usize = 4096;
const DEFAULT_CONTEXT_FLOOR: usize = 2048;

/// KV cache precision selectable on the CLI.
#[derive(ValueEnum, Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvArg {
    #[value(name = "f32")]
    F32,
    #[value(name = "f16")]
    F16,
    #[value(name = "int8")]
    Int8,
    #[value(name = "int4")]
    Int4,
}

impl KvArg {
    const fn into_kv_type(self) -> KvType {
        match self {
            Self::F32 => KvType::F32,
            Self::F16 => KvType::F16,
            Self::Int8 => KvType::Int8,
            Self::Int4 => KvType::Int4,
        }
    }
}

#[derive(Args, Debug, Clone)]
pub struct PathfinderArgs {
    /// Path to model directory or file (SafeTensors, GGUF, or .epiph)
    #[arg(value_name = "MODEL_PATH")]
    pub model: PathBuf,

    /// Target total memory budget in GiB (e.g. 4.0, 0.5)
    #[arg(long)]
    pub budget_gb: Option<f64>,

    /// Context length in tokens (marks context fixed when supplied)
    #[arg(long)]
    pub context: Option<usize>,

    /// KV cache precision (marks KV type fixed when supplied)
    #[arg(long, value_enum)]
    pub kv: Option<KvArg>,

    /// Print the RunPlan as JSON instead of a table
    #[arg(long)]
    pub json: bool,
}

/// Computes a [`RunPlan`] from model headers and CLI arguments without loading full payloads.
pub fn plan(args: &PathfinderArgs) -> Result<RunPlan> {
    let budget_bytes = match args.budget_gb {
        Some(gb) => {
            if !gb.is_finite() || gb < 0.0 {
                bail!("--budget-gb must be a finite non-negative number, got {gb}");
            }
            Some((gb * BYTES_PER_GIB).round() as u64)
        }
        None => None,
    };

    let ir = ModelImporter::import(&args.model)
        .with_context(|| format!("Failed to import model metadata from {:?}", args.model))?;
    let shape = shape_from_ir(&ir).ok_or_else(|| {
        anyhow::anyhow!(
            "Could not derive architectural shape from model '{}' (missing required dimensions or MLP metadata)",
            ir.name
        )
    })?;

    let default_context = if ir.max_position_embeddings > 0 {
        ir.max_position_embeddings.min(DEFAULT_MAX_CONTEXT)
    } else {
        DEFAULT_MAX_CONTEXT
    };
    let weight_bytes_per_param = if ir.num_parameters > 0 && ir.original_byte_size > 0 {
        ir.original_byte_size as f64 / ir.num_parameters as f64
    } else {
        4.0
    };

    let start = RunSettings {
        context: args.context.unwrap_or(default_context),
        kv: args.kv.map(KvArg::into_kv_type).unwrap_or(KvType::F16),
        weight_bytes_per_param,
    };
    let fixed = Fixed {
        context: args.context.is_some(),
        kv: args.kv.is_some(),
        weights: false,
    };
    let limits = ShrinkLimits {
        context_floor: DEFAULT_CONTEXT_FLOOR,
    };

    let run_plan = plan_run(&shape, ir.name, start, fixed, limits, budget_bytes)
        .map_err(|err| anyhow::anyhow!("{err}"))?;
    Ok(run_plan)
}

/// Serializes a [`RunPlan`] as pretty-printed JSON.
pub fn render_json(plan: &RunPlan) -> Result<String> {
    serde_json::to_string_pretty(plan).context("Failed to serialize RunPlan to JSON")
}

/// Formats a [`RunPlan`] as a plain table with basis labels and an explicit non-measured notice.
pub fn render_table(plan: &RunPlan) -> String {
    let basis_for = |decision: &str| -> &str {
        plan.basis
            .iter()
            .find(|b| b.decision == decision)
            .map_or("predicted", |b| b.basis.as_str())
    };
    let budget_str = plan
        .budget_bytes
        .map_or_else(|| "none".to_string(), |b| b.to_string());

    let summary_rows: [(&str, String, &str); 9] = [
        ("Model", plan.model.clone(), "-"),
        (
            "Context",
            plan.settings.context.to_string(),
            basis_for("context"),
        ),
        (
            "KV type",
            format!("{:?}", plan.settings.kv),
            basis_for("kv"),
        ),
        (
            "Weight bpp",
            format!("{}", plan.settings.weight_bytes_per_param),
            basis_for("weight_bytes_per_param"),
        ),
        (
            "Weights (bytes)",
            plan.memory.weights.to_string(),
            basis_for("memory.weights"),
        ),
        (
            "KV (bytes)",
            plan.memory.kv.to_string(),
            basis_for("memory.kv"),
        ),
        (
            "Scratch (bytes)",
            plan.memory.scratch.to_string(),
            basis_for("memory.scratch"),
        ),
        (
            "Total (bytes)",
            plan.memory.total().to_string(),
            basis_for("memory.total"),
        ),
        ("Budget (bytes)", budget_str, "-"),
    ];

    let mut lines = Vec::new();
    lines.push(format!("{:<18}  {:<16}  {}", "Field", "Value", "Basis"));
    lines.push(format!("{:-<18}  {:-<16}  {:-<10}", "", "", ""));
    for (field, value, basis) in summary_rows {
        lines.push(format!("{field:<18}  {value:<16}  {basis}"));
    }

    lines.push(String::new());
    lines.push(format!("{:<24}  {}", "Decision", "Basis"));
    lines.push(format!("{:-<24}  {:-<10}", "", ""));
    for entry in &plan.basis {
        lines.push(format!("{:<24}  {}", entry.decision, entry.basis.as_str()));
    }

    lines.push(String::new());
    lines
        .push("Nothing was measured; all numbers are predicted from the model header.".to_string());
    lines.join("\n")
}

/// Executes `thme pathfinder`.
pub fn execute(args: PathfinderArgs) -> Result<()> {
    let run_plan = plan(&args)?;
    if args.json {
        println!("{}", render_json(&run_plan)?);
    } else {
        println!("{}", render_table(&run_plan));
    }
    Ok(())
}
