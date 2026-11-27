// SPDX-License-Identifier: AGPL-3.0-only
//! `thme chat`: greedy-decode a reply with incremental KV caching and report measured speed.

use anyhow::Result;
use clap::{Args, ValueEnum};
use serde_json::{json, Value};
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;

use crate::chat::{generate, Finish, Reply, Request};
use crate::inference::gpu::DeviceArg;
use crate::inference::{load_reference_model, KvPrecision};
use crate::text::{has_chat_template, ChatMessage, Tokenizer};

/// KV cache storage precision selectable on the CLI.
#[derive(ValueEnum, Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum KvPrecisionArg {
    #[default]
    #[value(name = "f32")]
    F32,
    #[value(name = "int8")]
    Int8,
    #[value(name = "int4")]
    Int4,
}

impl KvPrecisionArg {
    pub const fn into_kv_precision(self) -> KvPrecision {
        match self {
            Self::F32 => KvPrecision::F32,
            Self::Int8 => KvPrecision::Int8,
            Self::Int4 => KvPrecision::Int4,
        }
    }
}

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

    /// KV cache storage precision (f32, int8, int4)
    #[arg(long, value_enum, default_value_t = KvPrecisionArg::F32)]
    pub kv_precision: KvPrecisionArg,

    /// Output structured JSON with timing and token counts
    #[arg(long)]
    pub json: bool,
}

fn reply_json(
    model_name: &str,
    arch: &str,
    device: DeviceArg,
    reply: &Reply,
    finish_str: &str,
) -> Value {
    json!({
        "model": model_name,
        "architecture": arch,
        "device": format!("{:?}", device).to_lowercase(),
        "kv_precision": reply.kv_precision.as_str(),
        "kv_cache_bytes": reply.kv_cache_bytes,
        "prompt_tokens": reply.prompt_tokens,
        "generated_tokens": reply.token_ids.len(),
        "wall_secs": reply.wall_secs,
        "prefill_secs": reply.prefill_secs,
        "tokens_per_sec": reply.tokens_per_sec,
        "decode_tokens_per_sec": reply.decode_tokens_per_sec,
        "finish": finish_str,
        "text": reply.text,
    })
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
        kv_precision: args.kv_precision.into_kv_precision(),
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
        let out = reply_json(
            &model_ir.name,
            arch_cfg.family(),
            args.device,
            &reply,
            finish_str,
        );
        println!("{}", serde_json::to_string_pretty(&out)?);
    } else {
        println!(
            "=== THME Chat ({}, {:?}) ===",
            model_ir.name,
            arch_cfg.family()
        );
        println!(
            "Prompt: {} tok | Generated: {} tok | KV: {} ({} B) | Wall: {:.3}s (prefill {:.3}s) | Total: {:.2} tok/s | Decode: {:.2} tok/s | Stop: {}",
            reply.prompt_tokens,
            reply.token_ids.len(),
            reply.kv_precision.as_str(),
            reply.kv_cache_bytes,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::{Cli, Commands};
    use clap::Parser;

    #[test]
    fn kv_precision_flag_parses_and_defaults_to_f32() {
        let cli = Cli::try_parse_from(["thme", "chat", "model.gguf"]).unwrap();
        let Commands::Chat(args) = cli.command else {
            panic!("expected chat command");
        };
        assert_eq!(args.kv_precision, KvPrecisionArg::F32);
        assert_eq!(args.kv_precision.into_kv_precision(), KvPrecision::F32);

        for (raw, expected_arg, expected_prec) in [
            ("f32", KvPrecisionArg::F32, KvPrecision::F32),
            ("int8", KvPrecisionArg::Int8, KvPrecision::Int8),
            ("int4", KvPrecisionArg::Int4, KvPrecision::Int4),
        ] {
            let argv = ["thme", "chat", "model.gguf", "--kv-precision", raw];
            let cli = Cli::try_parse_from(argv).unwrap();
            let Commands::Chat(args) = cli.command else {
                panic!("expected chat command");
            };
            assert_eq!(args.kv_precision, expected_arg);
            assert_eq!(args.kv_precision.into_kv_precision(), expected_prec);
        }

        assert!(
            Cli::try_parse_from(["thme", "chat", "model.gguf", "--kv-precision", "fp16"]).is_err()
        );
    }

    #[test]
    fn reply_json_includes_kv_precision_and_kv_cache_bytes() {
        let reply = Reply {
            prompt: "user: hi\nassistant:".to_string(),
            prompt_tokens: 4,
            token_ids: vec![7, 8],
            text: "ok".to_string(),
            wall_secs: 0.5,
            prefill_secs: 0.2,
            tokens_per_sec: 4.0,
            decode_tokens_per_sec: 3.33,
            kv_precision: KvPrecision::Int8,
            kv_cache_bytes: 240,
            finish: Finish::MaxTokens,
        };
        let out = reply_json("tiny-llama", "LLaMA", DeviceArg::Cpu, &reply, "max_tokens");
        assert_eq!(out["model"], "tiny-llama");
        assert_eq!(out["architecture"], "LLaMA");
        assert_eq!(out["device"], "cpu");
        assert_eq!(out["kv_precision"], "int8");
        assert_eq!(out["kv_cache_bytes"], 240);
        assert_eq!(out["prompt_tokens"], 4);
        assert_eq!(out["generated_tokens"], 2);
        assert_eq!(out["wall_secs"], 0.5);
        assert_eq!(out["prefill_secs"], 0.2);
        assert_eq!(out["tokens_per_sec"], 4.0);
        assert_eq!(out["decode_tokens_per_sec"], 3.33);
        assert_eq!(out["finish"], "max_tokens");
        assert_eq!(out["text"], "ok");
    }
}
