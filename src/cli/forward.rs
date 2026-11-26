// SPDX-License-Identifier: AGPL-3.0-only
//! `thme forward`: execute reference forward pass and output token logits.

use anyhow::{bail, Context, Result};
use clap::Args;
use serde_json::json;
use std::path::PathBuf;

use crate::inference::gpu::DeviceArg;
use crate::inference::load_reference_model;
use crate::inference::perplexity::read_token_file;

#[derive(Args, Debug)]
pub struct ForwardArgs {
    /// Model file or directory (GGUF, SafeTensors)
    #[arg(value_name = "MODEL_PATH")]
    pub model: PathBuf,

    /// Comma-separated token IDs (e.g. "1,450,6807,310")
    #[arg(long, value_name = "TOKENS")]
    pub tokens: Option<String>,

    /// Path to binary token file (little-endian u32)
    #[arg(long, value_name = "TOKEN_FILE")]
    pub token_file: Option<PathBuf>,

    /// Limit output to the first N vocabulary logits (0 returns all)
    #[arg(long, default_value_t = 100)]
    pub limit: usize,

    /// Execution compute device (auto, gpu, cpu)
    #[arg(long, value_enum, default_value_t = DeviceArg::Auto)]
    pub device: DeviceArg,

    /// Output results as JSON
    #[arg(long)]
    pub json: bool,
}

pub fn execute(args: ForwardArgs) -> Result<()> {
    let tokens: Vec<u32> = if let Some(ref s) = args.tokens {
        s.split(',')
            .map(|tok| {
                tok.trim()
                    .parse::<u32>()
                    .with_context(|| format!("Invalid token ID '{tok}'"))
            })
            .collect::<Result<Vec<u32>>>()?
    } else if let Some(ref path) = args.token_file {
        read_token_file(path)?
    } else {
        bail!("Must specify either --tokens <csv> or --token-file <path>");
    };

    if tokens.len() < 2 {
        bail!(
            "Forward pass requires at least 2 tokens, received {}",
            tokens.len()
        );
    }

    let (model_ir, arch_cfg, runnable) = load_reference_model(&args.model, args.device)?;

    let full_logits = runnable.forward_logits(&tokens)?;
    let vocab_size = runnable.vocab_size();

    let mut best_id = 0usize;
    let mut best_logit = f32::NEG_INFINITY;
    for (i, &l) in full_logits.iter().enumerate() {
        if l > best_logit {
            best_logit = l;
            best_id = i;
        }
    }

    let returned_logits = if args.limit > 0 && args.limit < full_logits.len() {
        full_logits[..args.limit].to_vec()
    } else {
        full_logits
    };

    if args.json {
        let output = json!({
            "model": model_ir.name,
            "architecture": arch_cfg.family(),
            "device": format!("{:?}", args.device).to_lowercase(),
            "vocab_size": vocab_size,
            "tokens": tokens,
            "top1_token_id": best_id,
            "top1_logit": best_logit,
            "limit": args.limit,
            "logits": returned_logits,
        });
        println!("{}", serde_json::to_string_pretty(&output)?);
    } else {
        println!("=== THME Reference Forward Pass ===");
        println!("Model:        {}", model_ir.name);
        println!("Architecture: {}", arch_cfg.family());
        println!("Device:       {:?}", args.device);
        println!("Tokens:       {:?}", tokens);
        println!("Vocab Size:   {}", vocab_size);
        println!("Top Token ID: {} (Logit: {:.4})", best_id, best_logit);
        println!(
            "Output Logits (first {}): {:?}",
            returned_logits.len(),
            returned_logits
        );
    }

    Ok(())
}
