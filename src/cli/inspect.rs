use anyhow::Result;
use clap::Args;
use std::path::PathBuf;

use crate::parser::ModelImporter;
use crate::report::ReportPrinter;

#[derive(Args, Debug)]
pub struct InspectArgs {
    /// Path to model directory or safetensors file
    #[arg(value_name = "MODEL_PATH")]
    pub path: PathBuf,

    /// Output full json inventory
    #[arg(short, long)]
    pub json: bool,
}

pub fn execute(args: InspectArgs) -> Result<()> {
    println!("Scanning model metadata at: {:?}", args.path);
    let model_ir = ModelImporter::import(&args.path)?;

    if args.json {
        let json_str = serde_json::to_string_pretty(&model_ir)?;
        println!("{json_str}");
    } else {
        ReportPrinter::print_model_summary(&model_ir);
    }

    Ok(())
}
