// SPDX-License-Identifier: AGPL-3.0-only
use anyhow::{Context, Result};
use clap::Args;
use std::path::PathBuf;

use crate::inference::{read_token_file, ArchConfig, MeasuredEvaluator, PerplexityConfig};
use crate::ir::rebuilder::{ArchitectureBlueprint, ModelRebuilder};
use crate::parser::ModelImporter;
use crate::surgery::{AgentBridgeConfig, CoSurgeryBudget, CoSurgeryController, CoSurgeryState};

#[derive(Args, Debug)]
pub struct CoSurgeryArgs {
    /// Source GGUF or SafeTensors model path (omit to run with a synthetic scaffold)
    #[arg(value_name = "MODEL_PATH")]
    pub model: Option<PathBuf>,

    /// Maximum closed-loop optimization iterations
    #[arg(long, default_value_t = 10)]
    pub budget_iterations: usize,

    /// Maximum allowed perplexity increase (absolute PPL delta)
    #[arg(long, default_value_t = 0.15)]
    pub max_ppl_delta: f64,

    /// Target model size reduction ratio (0.0–1.0, e.g. 0.40 = 40%)
    #[arg(long, default_value_t = 0.40)]
    pub target_reduction: f64,

    /// External Teacher LLM command for subprocess bridge (e.g. "ollama run llama3")
    #[arg(long, value_name = "CMD")]
    pub agent_command: Option<String>,

    /// Token file (little-endian u32) from scripts/prepare_eval_tokens.py. With MODEL_PATH,
    /// candidates are scored by the reference forward pass instead of the analytic proxy.
    #[arg(long, value_name = "PATH")]
    pub tokens: Option<PathBuf>,

    /// Leading corpus tokens per measured evaluation
    #[arg(long, default_value_t = 16384)]
    pub eval_max_tokens: usize,

    /// Tokens per forward pass during measured evaluation
    #[arg(long, default_value_t = 1024)]
    pub eval_context: usize,

    /// Emit final report as compact JSON instead of formatted table
    #[arg(long)]
    pub json: bool,
}

pub fn execute(args: CoSurgeryArgs) -> Result<()> {
    let model = match &args.model {
        Some(path) => ModelImporter::import(path)
            .with_context(|| format!("Failed to load model from {path:?}"))?,
        // shortcut: synthetic scaffold used when no path supplied; replace with real importer when
        // the caller wants to run co-surgery on disk models without the CLI model arg.
        None => {
            eprintln!("[co-surgery] No model path provided — using synthetic 2-layer scaffold.");
            let bp = ArchitectureBlueprint::new_hybrid_llama("scaffold", 2, 128, 1);
            ModelRebuilder::build_scaffold(&bp)
        }
    };

    let budget = CoSurgeryBudget {
        max_iterations: args.budget_iterations,
        max_ppl_delta: args.max_ppl_delta,
        target_reduction_ratio: args.target_reduction,
        max_consecutive_rollbacks: 3,
    };

    // Agent bridge config — subprocess if a command was given, mock otherwise.
    let _bridge_cfg = match &args.agent_command {
        Some(cmd) => AgentBridgeConfig::Subprocess {
            command: cmd.clone(),
            args: Vec::new(),
            timeout_secs: 30,
        },
        None => AgentBridgeConfig::Mock {
            canned_recipe: Default::default(),
        },
    };

    eprintln!(
        "[co-surgery] Starting — budget: {} iters, max_ppl_delta: {:.3}, target_reduction: {:.0}%",
        budget.max_iterations,
        budget.max_ppl_delta,
        budget.target_reduction_ratio * 100.0,
    );

    let mut controller = match (&args.model, &args.tokens) {
        (Some(path), Some(tokens)) => {
            let arch = ArchConfig::detect(&model, path)?;
            let evaluator = MeasuredEvaluator::new(
                arch,
                read_token_file(tokens)?,
                PerplexityConfig {
                    context_len: args.eval_context,
                    max_tokens: Some(args.eval_max_tokens),
                },
            );
            eprintln!("[co-surgery] Quality source: measured (reference forward pass)");
            CoSurgeryController::with_evaluator(model, budget, Box::new(evaluator))
                .map_err(|error| anyhow::anyhow!("Baseline evaluation failed: {error}"))?
        }
        (_, Some(_)) => anyhow::bail!("--tokens requires MODEL_PATH with on-disk weights"),
        _ => {
            eprintln!("[co-surgery] Quality source: SIMULATED analytic proxy. Pass MODEL_PATH and --tokens for measured perplexity.");
            CoSurgeryController::new(model, budget)
        }
    };

    // Run step-by-step so we can emit per-state telemetry to stderr.
    loop {
        let state = controller.state();
        if state == CoSurgeryState::Terminated {
            break;
        }
        eprintln!(
            "[co-surgery] iter={:>3}  state={state:?}",
            controller.current_iteration()
        );
        if !controller.step() {
            break;
        }
    }

    let report = controller.report();

    if args.json {
        let json = serde_json::to_string_pretty(&report)
            .context("Failed to serialize co-surgery report to JSON")?;
        println!("{json}");
    } else {
        println!("── Co-Surgery Report ──────────────────────────────");
        println!("  Iterations      : {}", report.total_iterations);
        println!("  Commits         : {}", report.successful_commits);
        println!("  Rollbacks       : {}", report.total_rollbacks);
        println!("  Relaxations     : {}", report.total_relaxations);
        println!("  Baseline PPL    : {:.4}", report.baseline_ppl);
        println!("  Final PPL       : {:.4}", report.final_ppl);
        println!(
            "  PPL delta       : {:+.4}",
            report.final_ppl - report.baseline_ppl
        );
        println!(
            "  Size reduction  : {:.1}%",
            report.final_reduction_ratio * 100.0
        );
        println!("  Final state     : {:?}", report.final_state);
        println!("  Quality source  : {}", report.quality_source);
        println!("───────────────────────────────────────────────────");
    }

    Ok(())
}
