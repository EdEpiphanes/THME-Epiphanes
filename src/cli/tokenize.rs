// SPDX-License-Identifier: AGPL-3.0-only
//! `thme tokenize`: encode text with a model folder's own `tokenizer.json`, then decode it back.

use anyhow::Result;
use clap::Args;
use std::path::PathBuf;

use crate::text::Tokenizer;

#[derive(Args, Debug)]
pub struct TokenizeArgs {
    /// Model folder that holds tokenizer.json
    pub model_dir: PathBuf,

    /// Text to encode
    pub text: String,

    /// Print only the token ids, separated by spaces
    #[arg(long, conflicts_with = "json")]
    pub ids_only: bool,

    /// Print JSON with the ids and the decoded text
    #[arg(long)]
    pub json: bool,
}

pub fn execute(args: TokenizeArgs) -> Result<()> {
    let tokenizer = Tokenizer::from_model_dir(&args.model_dir)?;
    let ids = tokenizer.encode(&args.text, true)?;
    let decoded = tokenizer.decode(&ids)?;
    if args.json {
        let out = serde_json::json!({ "ids": ids, "decoded": decoded });
        println!("{}", serde_json::to_string_pretty(&out)?);
    } else if args.ids_only {
        let line: Vec<String> = ids.iter().map(u32::to_string).collect();
        println!("{}", line.join(" "));
    } else {
        println!("tokens: {}", ids.len());
        println!("ids: {ids:?}");
        println!("decoded: {decoded}");
    }
    Ok(())
}
