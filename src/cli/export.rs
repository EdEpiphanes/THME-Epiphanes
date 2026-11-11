use anyhow::{Context, Result};
use clap::Args;
use std::path::PathBuf;

use crate::exporter::{ExportConfig, ExportFormat, ExportQuantType, ModelExporter};
use crate::parser::ModelImporter;

#[derive(Args, Debug)]
pub struct ExportArgs {
    /// Source GGUF file or SafeTensors model directory
    #[arg(value_name = "MODEL_PATH")]
    pub model: PathBuf,

    /// Destination model artifact
    #[arg(short, long, value_name = "OUTPUT")]
    pub output: PathBuf,

    /// Container format: gguf or safetensors
    #[arg(long, default_value = "gguf")]
    pub format: ExportFormat,

    /// Weight type: bitnet-b1.58, q8_0, or fp16
    #[arg(long, default_value = "fp16", value_parser = parse_quant_type)]
    pub quant: ExportQuantType,

    /// Compute SHA-256 checksum after export
    #[arg(long)]
    pub checksum: bool,
}

fn parse_quant_type(value: &str) -> std::result::Result<ExportQuantType, String> {
    match value.to_ascii_lowercase().as_str() {
        // BitNet b1.58 weights use the interoperable GGML ternary block encoding.
        "bitnet" | "bitnet-b1.58" | "bitnet_b158" | "ternary" | "tq1" | "tq1_0" => {
            Ok(ExportQuantType::TQ1_0)
        }
        "q8" | "q8_0" | "int8" => Ok(ExportQuantType::Q8_0),
        "fp16" | "f16" => Ok(ExportQuantType::FP16),
        other => Err(format!(
            "unsupported quantization '{other}'; choose bitnet-b1.58 (GGUF TQ1_0), q8_0, or fp16"
        )),
    }
}

pub fn execute(args: ExportArgs) -> Result<()> {
    let model = ModelImporter::import(&args.model)
        .with_context(|| format!("Failed to load source model from {:?}", args.model))?;
    let report = ModelExporter::export(
        &model,
        &ExportConfig {
            format: args.format,
            quant_type: args.quant,
            output_path: args.output.clone(),
            compute_checksum: args.checksum,
        },
    )?;

    println!(
        "Exported {} tensors to {:?} ({} bytes)",
        report.total_tensors, args.output, report.exported_bytes
    );
    if let Some(checksum) = report.sha256_checksum {
        println!("SHA-256: {checksum}");
    }
    Ok(())
}
