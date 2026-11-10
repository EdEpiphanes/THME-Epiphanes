use anyhow::Result;
use clap::Args;
use std::path::PathBuf;

use crate::parser::ModelImporter;
use crate::profiling::{KvCacheEstimator, SensitivityProfiler};
use crate::report::ReportPrinter;

#[derive(Args, Debug)]
pub struct ProfileArgs {
    /// Path to model directory
    #[arg(value_name = "MODEL_PATH")]
    pub path: PathBuf,

    /// Batch size for KV-cache estimation
    #[arg(short, long, default_value_t = 1)]
    pub batch_size: usize,

    /// Comma-separated list of context lengths to project
    #[arg(short, long, default_value = "1024,2048,4096,8192,16384,32768,65536,131072")]
    pub context_lengths: String,

    /// Optional output file to write JSON profile
    #[arg(short, long)]
    pub output: Option<PathBuf>,
}

pub fn execute(args: ProfileArgs) -> Result<()> {
    println!("Profiling manifold sensitivity & memory for: {:?}", args.path);
    let mut model_ir = ModelImporter::import_from_dir(&args.path)?;

    // 1. Calculate layer sensitivity
    SensitivityProfiler::profile_model(&mut model_ir);
    let layer_reports = SensitivityProfiler::summarize_layers(&model_ir);
    ReportPrinter::print_layer_sensitivity(&layer_reports);

    // 2. Parse context lengths and estimate KV cache
    let ctx_lens: Vec<usize> = args
        .context_lengths
        .split(',')
        .filter_map(|s| s.trim().parse::<usize>().ok())
        .collect();

    let kv_projections = KvCacheEstimator::project_memory_curve(
        model_ir.num_layers,
        args.batch_size,
        model_ir.num_key_value_heads,
        model_ir.head_dim,
        &ctx_lens,
    );
    ReportPrinter::print_kv_cache(&kv_projections);

    if let Some(out_path) = args.output {
        let profile_data = serde_json::json!({
            "model": model_ir.name,
            "layers": layer_reports,
            "kv_cache": kv_projections,
        });
        let file = std::fs::File::create(&out_path)?;
        serde_json::to_writer_pretty(file, &profile_data)?;
        println!("\nProfile successfully saved to: {:?}", out_path);
    }

    Ok(())
}
