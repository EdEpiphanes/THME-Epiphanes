// SPDX-License-Identifier: AGPL-3.0-only
//! Perplexity of a model with its KV cache stored as f32, int8 and int4 (THME-3082).
//!
//! Every window is fed one token at a time through a fresh cache, so each token attends to the
//! stored (possibly quantized) history. Prints one JSON object per precision. CPU only.
//!
//! ```text
//! cargo run --release --example kv_precision_ppl -- \
//!     --model-path models/SmolLM-135M --tokens target/eval/wikitext2_test.SmolLM-135M.u32 \
//!     --window 256 --max-windows 4
//! ```

use anyhow::{Context, Result};
use clap::Parser;
use serde_json::json;
use std::path::PathBuf;

use thme::inference::{read_token_file, ArchConfig, KvPrecision, WeightPrecision};
use thme::parser::ModelImporter;
use thme::quality::kv_perplexity::kv_windowed_perplexity;

#[derive(Parser, Debug)]
#[command(name = "kv_precision_ppl", about = "Perplexity per KV-cache precision")]
struct Args {
    /// Model directory or file with on-disk weights
    #[arg(long)]
    model_path: PathBuf,

    /// Token file (little-endian u32 ids) from scripts/prepare_eval_tokens.py
    #[arg(long)]
    tokens: PathBuf,

    /// Tokens per window
    #[arg(long, default_value_t = 256)]
    window: usize,

    /// Most windows to score per precision
    #[arg(long, default_value_t = 4)]
    max_windows: usize,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let source = ModelImporter::import(&args.model_path)
        .with_context(|| format!("Failed to import model from {:?}", args.model_path))?;
    let tokens = read_token_file(&args.tokens)?;
    let config = ArchConfig::detect(&source, &args.model_path)?;
    let model = config.load(&source, WeightPrecision::AsStored)?;
    for precision in [KvPrecision::F32, KvPrecision::Int8, KvPrecision::Int4] {
        let r = kv_windowed_perplexity(
            model.as_ref(),
            &tokens,
            args.window,
            args.max_windows,
            precision,
        )?;
        println!(
            "{}",
            json!({
                "model": args.model_path.file_name().map(|n| n.to_string_lossy().into_owned()),
                "kv_precision": precision.as_str(),
                "window": r.window,
                "windows": r.windows,
                "tokens_scored": r.tokens_scored,
                "perplexity": r.perplexity,
                "kv_cache_bytes": r.kv_cache_bytes,
            })
        );
    }
    Ok(())
}
