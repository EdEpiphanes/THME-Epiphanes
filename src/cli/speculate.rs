use anyhow::{Context, Result};
use clap::Args;
use std::path::PathBuf;

use crate::ir::ParadigmConverter;
use crate::parser::ModelImporter;
use crate::speculative::{
    DraftExtractionConfig, DraftExtractionStrategy, DraftSynthesizer, SpeculativeEngine,
    SpeculativeExecutionConfig, SpeculativeTreeTopology, SpeculativeVerificationStrategy,
};

#[derive(Args, Debug)]
pub struct SpeculateArgs {
    /// Path to target model file or directory (omit to use synthetic baseline)
    #[arg(short, long, value_name = "MODEL_PATH")]
    pub model: Option<PathBuf>,

    /// Number of prefix layers to extract for the draft model (default: 4)
    #[arg(short = 'l', long, default_value = "4")]
    pub draft_layers: usize,

    /// Target hidden dimension for downscaled draft model (optional)
    #[arg(long)]
    pub draft_hidden: Option<usize>,

    /// Apply BitNet b1.58 ternary format to draft model MLP for high-speed CPU/edge generation
    #[arg(long)]
    pub bitnet: bool,

    /// Speculative lookahead horizon gamma (number of speculative tokens drafted per step)
    #[arg(short = 'g', long, default_value = "5")]
    pub horizon: usize,

    /// Sampling temperature tau (0.0 = greedy, 0.7 = standard)
    #[arg(short = 't', long, default_value = "0.7")]
    pub temperature: f32,

    /// Number of speculative forward steps to evaluate
    #[arg(short = 's', long, default_value = "25")]
    pub steps: usize,

    /// Evaluate multi-branch tree speculative decoding (e.g. Medusa/Eagle tree attention)
    #[arg(long)]
    pub tree: bool,

    /// Save the synthesized draft model IR to a JSON file
    #[arg(long, value_name = "OUTPUT_PATH")]
    pub export_draft: Option<PathBuf>,

    /// Output report in JSON format
    #[arg(long)]
    pub json: bool,
}

pub fn execute(args: SpeculateArgs) -> Result<()> {
    let target_model = if let Some(ref path) = args.model {
        if !args.json {
            println!("Loading target model from: {:?}", path);
        }
        ModelImporter::import(path)
            .with_context(|| format!("Failed to import model from {:?}", path))?
    } else {
        if !args.json {
            println!("Using default 12-layer synthetic LLM baseline as Target Model.");
        }
        ParadigmConverter::create_synthetic_baseline()
    };

    if !args.json {
        println!(
            "Target Model: '{}' | Layers: {} | Hidden Dim: {} | Params: {}",
            target_model.name,
            target_model.num_layers,
            target_model.hidden_size,
            target_model.num_parameters
        );
    }

    // 1. Synthesize Draft Model
    let draft_cfg = DraftExtractionConfig {
        strategy: if args.bitnet {
            DraftExtractionStrategy::TernaryBitNet
        } else if args.draft_hidden.is_some() {
            DraftExtractionStrategy::ProjectedManifold
        } else {
            DraftExtractionStrategy::PrefixLayers
        },
        num_layers: args.draft_layers,
        target_hidden_size: args.draft_hidden,
        apply_bitnet: args.bitnet,
        stride_step: 4,
    };

    let (draft_model, synthesis_report) =
        DraftSynthesizer::extract_draft_model(&target_model, &draft_cfg)
            .map_err(|e| anyhow::anyhow!("Draft model synthesis failed: {}", e))?;

    if let Some(ref out_path) = args.export_draft {
        if let Some(parent) = out_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let json_str = serde_json::to_string_pretty(&draft_model)?;
        std::fs::write(out_path, json_str)
            .with_context(|| format!("Failed to export draft model to {:?}", out_path))?;
        if !args.json {
            println!("Exported draft model IR to: {:?}", out_path);
        } else {
            eprintln!("Exported draft model IR to: {:?}", out_path);
        }
    }

    // 2. Evaluate Speculative Decoding Run
    let spec_config = SpeculativeExecutionConfig {
        lookahead_horizon: args.horizon,
        temperature: args.temperature,
        strategy: SpeculativeVerificationStrategy::RejectionSampling,
        base_draft_fidelity: 0.82,
        latency_cost_ratio: synthesis_report.latency_cost_ratio,
    };

    let telemetry = SpeculativeEngine::simulate_speculative_run(&spec_config, args.steps, 12.0);

    // 3. Optional Tree Evaluation
    let tree_info = if args.tree {
        let tree = SpeculativeTreeTopology::build_canonical(&[3, 2, 1]);
        let (best_path, accepted, tree_acc) = tree.evaluate_tree_acceptance(0.82);
        Some((tree.total_nodes, best_path.len(), accepted, tree_acc))
    } else {
        None
    };

    if args.json {
        let report_obj = serde_json::json!({
            "synthesis": synthesis_report,
            "telemetry": telemetry,
            "tree_speculation": tree_info.map(|(total, path_len, acc, rate)| {
                serde_json::json!({
                    "total_tree_candidates": total,
                    "best_path_length": path_len,
                    "accepted_branch_tokens": acc,
                    "tree_acceptance_rate": rate
                })
            })
        });
        println!("{}", serde_json::to_string_pretty(&report_obj)?);
    } else {
        println!();
        println!("============================================================");
        println!("   THME SPECULATIVE DECODING & DRAFT SYNTHESIS REPORT");
        println!("============================================================");
        println!("{}", synthesis_report.format_summary());
        println!("------------------------------------------------------------");
        println!("{}", telemetry.format_summary());

        if let Some((total, path_len, acc, rate)) = tree_info {
            println!("------------------------------------------------------------");
            println!("Tree-Based Speculation (Medusa/Eagle Multi-Branch Topology):");
            println!("- Total Tree Candidate Nodes : {}", total);
            println!("- Longest Accepted Path Nodes: {}", path_len);
            println!("- Accepted Branch Tokens     : {}", acc);
            println!("- Tree Acceptance Rate       : {:.1}%", rate * 100.0);
        }

        let (opt_gamma, opt_s) = SpeculativeEngine::find_optimal_horizon(
            spec_config.base_draft_fidelity,
            synthesis_report.latency_cost_ratio,
        );
        println!("------------------------------------------------------------");
        println!(
            "Optimal Lookahead Recommendation: gamma = {} tokens (Estimated Speedup: {:.2}x)",
            opt_gamma, opt_s
        );
        println!("============================================================");
    }

    Ok(())
}
