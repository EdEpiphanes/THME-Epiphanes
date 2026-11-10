use anyhow::{bail, Context, Result};
use clap::Args;
use std::path::PathBuf;

use crate::ir::{ParadigmConversionConfig, ParadigmConverter, SpecializedModelParadigm};
use crate::parser::ModelImporter;
use crate::surgery::{TweakOperation, TweakRecipe};

#[derive(Args, Debug)]
pub struct ConvertArgs {
    /// Path to model directory or safetensors/gguf file (omit to use synthetic baseline)
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
        println!("Scanning and importing source model from: {:?}", path);
        ModelImporter::import(path)
            .with_context(|| format!("Failed to import model from {:?}", path))?
    } else {
        println!("Using default 12-layer synthetic LLM baseline for conversion test.");
        ParadigmConverter::create_synthetic_baseline()
    };

    let source = ParadigmConverter::detect_paradigm(&model_ir);
    println!(
        "Detected Source Architecture: {} ({}) | Tensors: {} | Hidden Dim: {}",
        source.display_name(),
        source.code(),
        model_ir.tensors.len(),
        model_ir.hidden_size
    );

    let mut config = ParadigmConversionConfig::new(target);
    config.num_experts = args.experts;
    config.top_k = args.top_k;
    config.target_hidden_size = args.hidden;
    config.target_layers = args.layers;
    config.apply_bitnet = args.bitnet;

    let (_converted_ir, report) = ParadigmConverter::convert(&model_ir, &config)
        .map_err(|e| anyhow::anyhow!("Paradigm conversion failed: {}", e))?;

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
            target_model: model_ir.name.clone(),
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
        std::fs::write(recipe_path, recipe_json)
            .with_context(|| format!("Failed to write recipe to {:?}", recipe_path))?;
        println!("Saved declarative surgery recipe to: {:?}", recipe_path);
    }

    if args.json {
        let json_str = serde_json::to_string_pretty(&report)?;
        println!("{json_str}");
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
        println!("------------------------------------------------------------");
        println!("{}", report.format_summary());
        println!("============================================================");
    }

    Ok(())
}
