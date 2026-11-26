// SPDX-License-Identifier: AGPL-3.0-only
//! `thme chat`: greedy-decode a reply with incremental KV caching and report measured speed.

use anyhow::Result;
use clap::Args;
use serde_json::json;
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;

use crate::chat::{generate, Finish, Request};
use crate::inference::gpu::DeviceArg;
use crate::inference::load_reference_model;
use crate::text::{has_chat_template, ChatMessage, Tokenizer};

#[derive(Args, Debug)]
pub struct ChatArgs {
    /// Model folder or file (SafeTensors directory, GGUF, or .epiph)
    #[arg(value_name = "MODEL_PATH")]
    pub model: PathBuf,

    /// Folder containing tokenizer.json (defaults to MODEL_PATH or its parent folder)
    #[arg(long, value_name = "DIR")]
    pub tokenizer_dir: Option<PathBuf>,

    /// User prompt text
    #[arg(
        long,
        default_value = "Explain why KV caching speeds up token generation in one paragraph."
    )]
    pub prompt: String,

    /// Maximum number of new tokens to generate
    #[arg(long, default_value_t = 64)]
    pub max_new_tokens: usize,

    /// Join turns as plain text instead of applying tokenizer_config.json chat template
    #[arg(long)]
    pub plain_text: bool,

    /// Execution compute device (auto, gpu, cpu)
    #[arg(long, value_enum, default_value_t = DeviceArg::Auto)]
    pub device: DeviceArg,

    /// Output structured JSON with timing and token counts
    #[arg(long)]
    pub json: bool,
}

pub fn execute(args: ChatArgs) -> Result<()> {
    let tok_dir = match &args.tokenizer_dir {
        Some(dir) => dir.clone(),
        None if args.model.is_dir() => args.model.clone(),
        None => args
            .model
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or_else(|| std::path::Path::new("."))
            .to_path_buf(),
    };
    let tokenizer = Tokenizer::from_model_dir(&tok_dir)?;
    let (model_ir, arch_cfg, runnable) = load_reference_model(&args.model, args.device)?;
    let messages = vec![ChatMessage {
        role: "user".to_string(),
        content: args.prompt.clone(),
    }];
    let use_plain = args.plain_text || !has_chat_template(&tok_dir);
    let request = Request {
        dir: &tok_dir,
        messages: &messages,
        plain_text: use_plain,
        max_new_tokens: args.max_new_tokens,
    };
    let cancel = AtomicBool::new(false);
    let reply = generate(
        runnable.as_ref(),
        &tokenizer,
        &request,
        &cancel,
        &mut |_| {},
    )?;
    let finish_str = match reply.finish {
        Finish::Eos => "eos",
        Finish::MaxTokens => "max_tokens",
        Finish::Cancelled => "cancelled",
        Finish::LoopBroken => "loop_broken",
    };

    if args.json {
        let out = json!({
            "model": model_ir.name,
            "architecture": arch_cfg.family(),
            "device": format!("{:?}", args.device).to_lowercase(),
            "prompt_tokens": reply.prompt_tokens,
            "generated_tokens": reply.token_ids.len(),
            "wall_secs": reply.wall_secs,
            "prefill_secs": reply.prefill_secs,
            "tokens_per_sec": reply.tokens_per_sec,
            "decode_tokens_per_sec": reply.decode_tokens_per_sec,
            "finish": finish_str,
            "text": reply.text,
        });
        println!("{}", serde_json::to_string_pretty(&out)?);
    } else {
        println!(
            "=== THME Chat ({}, {:?}) ===",
            model_ir.name,
            arch_cfg.family()
        );
        println!(
            "Prompt: {} tok | Generated: {} tok | Wall: {:.3}s (prefill {:.3}s) | Total: {:.2} tok/s | Decode: {:.2} tok/s | Stop: {}",
            reply.prompt_tokens,
            reply.token_ids.len(),
            reply.wall_secs,
            reply.prefill_secs,
            reply.tokens_per_sec,
            reply.decode_tokens_per_sec,
            finish_str
        );
        println!("---");
        println!("{}", reply.text);
    }
    Ok(())
}
