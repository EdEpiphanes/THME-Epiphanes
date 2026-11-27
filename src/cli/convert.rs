// SPDX-License-Identifier: AGPL-3.0-only
use anyhow::{bail, Context, Result};
use clap::Args;
use std::path::PathBuf;

use crate::convert::{ConversionOp, ConversionPlan, ModelArchKind};
use crate::ir::{ParadigmConversionConfig, ParadigmConverter, SpecializedModelParadigm};
use crate::parser::ModelImporter;
use crate::surgery::{TweakOperation, TweakRecipe};

#[derive(Args, Debug)]
pub struct ConvertArgs {
    /// Path to model directory or SafeTensors, GGUF and .epiph file (omit to use synthetic baseline)
    #[arg(short, long, value_name = "MODEL_PATH")]
    pub model: Option<PathBuf>,

    /// Target model paradigm: llm, lcm, lam, moe, vlm, slm, mlm, sam
    #[arg(short, long, value_name = "PARADIGM")]
    pub to: String,

    /// Number of experts when converting to MoE (default: 4)
    #[arg(long, default_value = "4")]
    pub experts: usize,

    /// Top-K active routing experts per token when converting to MoE (default: 2)
    #[arg(long, default_value = "2")]
    pub top_k: usize,

    /// Target hidden dimension when pruning or scaling to SLM (e.g. 512, 1024, 2048)
    #[arg(long)]
    pub hidden: Option<usize>,

    /// Target layer count when pruning to SLM (e.g. 4, 8, 12)
    #[arg(long)]
    pub layers: Option<usize>,

    /// Discretize weights to BitNet b1.58 ternary format ({-1, 0, +1})
    #[arg(long)]
    pub bitnet: bool,

    /// Save declarative Teacher LLM surgery recipe JSON for this conversion
    #[arg(long, value_name = "RECIPE_FILE")]
    pub save_recipe: Option<PathBuf>,

    /// Optional §7.10 structural model-type op: mha-to-gqa, dense-to-moe, deep-to-dory, streaming-to-ssm, dialect
    #[arg(long, value_name = "ARCH_OP")]
    pub arch_op: Option<String>,

    /// Output report in JSON format
    #[arg(long)]
    pub json: bool,
}

pub fn execute(args: ConvertArgs) -> Result<()> {
    let target = match args.to.to_lowercase().as_str() {
        "llm" => SpecializedModelParadigm::LLM,
        "lcm" => SpecializedModelParadigm::LCM,
        "lam" => SpecializedModelParadigm::LAM,
        "moe" => SpecializedModelParadigm::MoE,
        "vlm" => SpecializedModelParadigm::VLM,
        "slm" => SpecializedModelParadigm::SLM,
        "mlm" => SpecializedModelParadigm::MLM,
        "sam" => SpecializedModelParadigm::SAM,
        other => bail!(
            "Unknown target paradigm '{}'. Supported paradigms: llm, lcm, lam, moe, vlm, slm, mlm, sam",
            other
        ),
    };

    let model_ir = if let Some(ref path) = args.model {
        if !args.json {
            println!("Scanning and importing source model from: {:?}", path);
        }
        ModelImporter::import(path)
            .with_context(|| format!("Failed to import model from {:?}", path))?
    } else {
        if !args.json {
            println!("Using default 12-layer synthetic LLM baseline for conversion test.");
        }
        ParadigmConverter::create_synthetic_baseline()
    };

    let source = ParadigmConverter::detect_paradigm(&model_ir);
    if !args.json {
        println!(
            "Detected Source Architecture: {} ({}) | Tensors: {} | Hidden Dim: {}",
            source.display_name(),
            source.code(),
            model_ir.tensors.len(),
            model_ir.hidden_size
        );
    }

    let mut config = ParadigmConversionConfig::new(target);
    config.num_experts = args.experts;
    config.top_k = args.top_k;
    config.target_hidden_size = args.hidden;
    config.target_layers = args.layers;
    config.apply_bitnet = args.bitnet;

    let (_converted_ir, report) = ParadigmConverter::convert(&model_ir, &config)
        .map_err(|e| anyhow::anyhow!("Paradigm conversion failed: {}", e))?;

    let arch_plan = if let Some(ref op_str) = args.arch_op {
        let (src_arch, dst_arch, op) = match op_str.to_lowercase().as_str() {
            "mha-to-gqa" => (
                ModelArchKind::DenseMha,
                ModelArchKind::DenseGqa,
                ConversionOp::MhaToGqa,
            ),
            "dense-to-moe" => (
                ModelArchKind::DenseGqa,
                ModelArchKind::SparseMoe,
                ConversionOp::DenseToMoe,
            ),
            "deep-to-dory" => (
                ModelArchKind::DenseGqa,
                ModelArchKind::DoryLooped,
                ConversionOp::DeepToDoryLoop,
            ),
            "streaming-to-ssm" => (
                ModelArchKind::DenseGqa,
                ModelArchKind::HybridSsm,
                ConversionOp::StreamingAttnToSsm,
            ),
            "dialect" => (
                ModelArchKind::DenseGqa,
                ModelArchKind::DenseGqa,
                ConversionOp::DialectPermutation,
            ),
            other => bail!("Unknown --arch-op '{other}'"),
        };
        Some(ConversionPlan::build(src_arch, dst_arch, op))
    } else {
        None
    };

    if let Some(ref recipe_path) = args.save_recipe {
        let op = TweakOperation::ConvertParadigm {
            target_paradigm: target,
            num_experts: Some(config.num_experts),
            top_k: Some(config.top_k),
            target_hidden_size: config.target_hidden_size,
            target_layers: config.target_layers,
            apply_bitnet: Some(config.apply_bitnet),
        };
        let recipe = TweakRecipe {
            name: format!("paradigm-convert-to-{}", target.code().to_lowercase()),
            author: "THME Paradigm Synthesizer".to_string(),
            target_model: model_ir.name,
            version: "0.1.5".to_string(),
            description: format!(
                "Declarative cross-paradigm surgery recipe converting {} to {}",
                source.code(),
                target.code()
            ),
            operations: vec![op],
            expected_speedup: Some(1.35),
            expected_memory_savings_pct: Some(report.storage_savings_pct as f32),
        };
        let recipe_json = serde_json::to_string_pretty(&recipe)?;
        if let Some(parent) = recipe_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(recipe_path, recipe_json)
            .with_context(|| format!("Failed to write recipe to {:?}", recipe_path))?;
        if !args.json {
            println!("Saved declarative surgery recipe to: {:?}", recipe_path);
        } else {
            eprintln!("Saved declarative surgery recipe to: {:?}", recipe_path);
        }
    }

    if args.json {
        let mut val = serde_json::to_value(&report)?;
        if let Some(ref plan) = arch_plan {
            val["arch_plan"] = serde_json::json!({
                "source_arch": format!("{:?}", plan.source_arch),
                "target_arch": format!("{:?}", plan.target_arch),
                "op": format!("{:?}", plan.op),
                "pre_quant_gate_at_f16_q8": plan.pre_quant_gate_at_f16_q8,
                "requires_bitdistill_recovery": plan.requires_bitdistill_recovery,
                "router_pinned_q8_0": plan.router_pinned_q8_0,
            });
        }
        println!("{}", serde_json::to_string_pretty(&val)?);
    } else {
        println!();
        println!("============================================================");
        println!("   THME UNIVERSAL MODEL PARADIGM CONVERSION REPORT");
        println!("============================================================");
        println!(
            "Target Paradigm  : {} ({})",
            target.display_name(),
            target.code()
        );
        println!(
            "Key Pipeline     : {}",
            target.primary_components().join(" -> ")
        );
        if let Some(ref plan) = arch_plan {
            println!(
                "§7.10 Arch Plan  : {:?} -> {:?} (pre_quant_f16_q8_gate={}, bitdistill_required={}, router_q8={})",
                plan.source_arch,
                plan.target_arch,
                plan.pre_quant_gate_at_f16_q8,
                plan.requires_bitdistill_recovery,
                plan.router_pinned_q8_0
            );
        }
        println!("------------------------------------------------------------");
        println!("{}", report.format_summary());
        println!("============================================================");
    }

    Ok(())
}
