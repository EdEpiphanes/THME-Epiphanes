// SPDX-License-Identifier: AGPL-3.0-only
//! `thme chat-template`: render a model's own chat template for a list of messages.

use anyhow::{Context, Result};
use clap::Args;
use std::path::PathBuf;

use crate::text::{render_chat_template, ChatMessage};

#[derive(Args, Debug)]
pub struct ChatTemplateArgs {
    /// Model folder that holds tokenizer_config.json
    pub model_dir: PathBuf,

    /// JSON array of {"role": ..., "content": ...} objects
    #[arg(long, value_name = "FILE.json")]
    pub messages: PathBuf,

    /// Do not append the opening of the assistant turn (add_generation_prompt)
    #[arg(long)]
    pub no_generation_prompt: bool,
}

pub fn execute(args: ChatTemplateArgs) -> Result<()> {
    let text = std::fs::read_to_string(&args.messages)
        .with_context(|| format!("cannot read {}", args.messages.display()))?;
    let messages: Vec<ChatMessage> = serde_json::from_str(&text).with_context(|| {
        format!(
            "{} must be a JSON array of {{\"role\", \"content\"}} objects",
            args.messages.display()
        )
    })?;
    let rendered = render_chat_template(&args.model_dir, &messages, !args.no_generation_prompt)?;
    print!("{rendered}");
    Ok(())
}
