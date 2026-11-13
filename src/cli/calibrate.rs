//! `thme calibrate`: record per-input-channel activation statistics of every linear tensor
//! on calibration tokens (roadmap v0.2.2.3), for activation-aware quantization.

use anyhow::{Context, Result};
use clap::Args;
use std::path::PathBuf;

use crate::inference::{collect_calibration, read_token_file, ArchConfig, WeightPrecision};
use crate::parser::ModelImporter;

#[derive(Args, Debug)]
pub struct CalibrateArgs {
    /// SafeTensors model directory or GGUF file
    #[arg(value_name = "MODEL_PATH")]
    pub model: PathBuf,

    /// Calibration token file (u32 little-endian, scripts/prepare_eval_tokens.py). Use tokens
    /// that are not part of the evaluation split.
    #[arg(long)]
    pub tokens: PathBuf,

    /// Tokens per forward pass
    #[arg(long, default_value_t = 256)]
    pub context: usize,

    /// Use at most this many leading tokens
    #[arg(long, default_value_t = 32_768)]
    pub max_tokens: usize,

    /// Output statistics file
    #[arg(long, default_value = "target/calibration.json")]
    pub out: PathBuf,
}

pub fn execute(args: CalibrateArgs) -> Result<()> {
    let ir = ModelImporter::import(&args.model)
        .with_context(|| format!("Failed to load model from {:?}", args.model))?;
    let arch = ArchConfig::detect(&ir, &args.model)?;
    let model = arch.load(&ir, WeightPrecision::AsStored)?;
    let tokens = read_token_file(&args.tokens)?;
    println!(
        "Calibrating '{}' ({}) on up to {} tokens, {} per window",
        ir.name,
        arch.family(),
        args.max_tokens.min(tokens.len()),
        args.context
    );
    let file = collect_calibration(&*model, &ir, &tokens, args.context, Some(args.max_tokens))?;
    file.save(&args.out)?;
    println!(
        "Recorded {} tensors over {} tokens -> {:?}",
        file.tensors.len(),
        file.tokens,
        args.out
    );
    Ok(())
}
