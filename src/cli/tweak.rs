use anyhow::{Context, Result};
use clap::Args;
use std::path::PathBuf;

use crate::parser::ModelImporter;
use crate::surgery::{SurgeryEngine, TweakRecipe};

#[derive(Args, Debug)]
pub struct TweakArgs {
    /// Path to base model directory or model file (.gguf, .safetensors)
    #[arg(short, long, value_name = "MODEL_PATH")]
    pub model: PathBuf,

    /// Path to declarative JSON surgery recipe
    #[arg(short, long, value_name = "RECIPE_PATH")]
    pub recipe: PathBuf,

    /// Optional destination to save modified architecture recipe or plan
    #[arg(short, long)]
    pub output: Option<PathBuf>,
}

pub fn execute(args: TweakArgs) -> Result<()> {
    println!("Loading base model from: {:?}", args.model);
    let mut model_ir = ModelImporter::import(&args.model)
        .with_context(|| format!("Failed to load model from {:?}", args.model))?;

    println!("Loading surgery recipe from: {:?}", args.recipe);
    let recipe_str = std::fs::read_to_string(&args.recipe)
        .with_context(|| format!("Failed to read recipe file: {:?}", args.recipe))?;
    let recipe: TweakRecipe =
        serde_json::from_str(&recipe_str).with_context(|| "Failed to parse recipe JSON")?;

    let diff = SurgeryEngine::apply_recipe(&mut model_ir, &recipe);

    for log in &diff.audit_log {
        println!("{log}");
    }

    if let Some(out) = args.output {
        let serialized = serde_json::to_string_pretty(&model_ir)?;
        std::fs::write(&out, serialized)?;
        println!("Exported modified model specification to: {:?}", out);
    }

    Ok(())
}
