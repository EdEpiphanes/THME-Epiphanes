// SPDX-License-Identifier: AGPL-3.0-only
//! Chat with a loaded model (THME-3038): renders the model's own chat template, encodes it, and
//! greedy-decodes a reply with the reference CPU forward pass.
//!
//! Speed figures are measured on the machine that runs them. Reply text is never a quality claim,
//! and a mock model (random weights) produces meaningless text by design.

use anyhow::{bail, Context, Result};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use crate::inference::grounded_decode::NgramLoopBreaker;
use crate::inference::{CausalLm, KvCache, KvPrecision};
use crate::text::{eos_token, has_chat_template, render_chat_template, ChatMessage, Tokenizer};

/// Why a reply stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Finish {
    /// The model produced the end-of-sequence token (not included in the reply).
    Eos,
    /// The token budget was used up.
    MaxTokens,
    /// Repetitive 4-gram attractor loop detected by `NgramLoopBreaker`.
    LoopBroken,
    /// The cancel flag was set.
    Cancelled,
}

/// One reply and the facts of the run that made it.
#[derive(Debug)]
pub struct Reply {
    pub prompt: String,
    pub prompt_tokens: usize,
    pub token_ids: Vec<u32>,
    pub text: String,
    pub wall_secs: f64,
    pub prefill_secs: f64,
    pub tokens_per_sec: f64,
    pub decode_tokens_per_sec: f64,
    pub kv_precision: KvPrecision,
    pub kv_cache_bytes: usize,
    pub finish: Finish,
}

/// What to ask the model: the turns, the model folder, and how the prompt is written.
pub struct Request<'a> {
    /// Model folder with `tokenizer.json` and `tokenizer_config.json`.
    pub dir: &'a Path,
    pub messages: &'a [ChatMessage],
    /// Plain text mode: the turns are joined as text, with no chat template.
    pub plain_text: bool,
    pub max_new_tokens: usize,
    pub kv_precision: KvPrecision,
}

/// Renders the prompt, then greedy-decodes until end-of-sequence, the token budget, or `cancel`.
/// `on_token` is called with the number of generated tokens so far.
pub fn generate(
    model: &dyn CausalLm,
    tokenizer: &Tokenizer,
    request: &Request<'_>,
    cancel: &AtomicBool,
    on_token: &mut dyn FnMut(usize),
) -> Result<Reply> {
    let prompt = if request.plain_text {
        plain_prompt(request.messages)
    } else {
        render_chat_template(request.dir, request.messages, true)?
    };
    let mut ids = tokenizer.encode(&prompt, false)?;
    if ids.len() < 2 {
        bail!(
            "The prompt has {} token(s); the reference forward pass needs at least 2",
            ids.len()
        );
    }
    let prompt_tokens = ids.len();
    let eos = eos_token(request.dir).and_then(|token| tokenizer.token_id(&token));
    let mut token_ids = Vec::new();
    let mut loop_breaker = NgramLoopBreaker::new(256);
    // With a KV cache the prompt is prefilled once and each reply token is one step. Without one
    // (no cached path for this architecture) every step reruns the whole sequence; non-F32
    // precisions require a cache path supporting them.
    let mut cache = model.new_cache_with_precision(request.kv_precision);
    if cache.is_none() && request.kv_precision != KvPrecision::F32 {
        bail!(
            "This model has no KV-cache path for precision {}",
            request.kv_precision.as_str()
        );
    }
    let mut fed = 0;
    let started = Instant::now();
    let mut after_first: Option<Instant> = None;
    let finish = loop {
        if cancel.load(Ordering::Relaxed) {
            break Finish::Cancelled;
        }
        if token_ids.len() >= request.max_new_tokens {
            break Finish::MaxTokens;
        }
        let logits = match cache.as_mut() {
            Some(cache) => {
                let logits = model.forward_extend(cache, &ids[fed..])?;
                fed = ids.len();
                logits
            }
            None => model.forward_logits(&ids)?,
        };
        let next = u32::try_from(argmax(&logits).context("The model returned no logits")?)?;
        if Some(next) == eos {
            break Finish::Eos;
        }
        ids.push(next);
        token_ids.push(next);
        if after_first.is_none() {
            after_first = Some(Instant::now());
        }
        on_token(token_ids.len());
        if loop_breaker.push_and_check_loop(next) {
            break Finish::LoopBroken;
        }
    };
    let kv_cache_bytes = cache.as_ref().map_or(0, KvCache::bytes);
    let wall_secs = started.elapsed().as_secs_f64();
    let prefill_secs = after_first.map_or(wall_secs, |t| t.duration_since(started).as_secs_f64());
    let decode_secs = (wall_secs - prefill_secs).max(0.0);
    let tokens_per_sec = if wall_secs > 0.0 {
        token_ids.len() as f64 / wall_secs
    } else {
        0.0
    };
    let decode_tokens_per_sec = if token_ids.len() > 1 && decode_secs > 0.0 {
        (token_ids.len() - 1) as f64 / decode_secs
    } else {
        tokens_per_sec
    };
    let text = tokenizer.decode(&token_ids)?;
    Ok(Reply {
        prompt,
        prompt_tokens,
        token_ids,
        text,
        wall_secs,
        prefill_secs,
        tokens_per_sec,
        decode_tokens_per_sec,
        kv_precision: request.kv_precision,
        kv_cache_bytes,
        finish,
    })
}

/// Plain text mode: one line per turn, then the cue for the model's own turn.
pub fn plain_prompt(messages: &[ChatMessage]) -> String {
    let mut out: String = messages
        .iter()
        .map(|m| format!("{}: {}\n", m.role, m.content))
        .collect();
    out.push_str("assistant:");
    out
}

/// What a model folder can do for chat.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Folder {
    /// The folder's own chat template renders the prompt.
    Template,
    /// No chat template; plain text mode is the only way to talk. Holds the reason.
    PlainTextOnly(String),
}

/// Checks a model folder before chat. A missing tokenizer is an error: no text can be encoded.
pub fn check_folder(dir: &Path) -> Result<Folder, String> {
    if !dir.join("tokenizer.json").is_file() {
        return Err(format!(
            "tokenizer.json is missing from {}. Chat needs the model's own tokenizer.json to turn text into tokens.",
            dir.display()
        ));
    }
    if has_chat_template(dir) {
        return Ok(Folder::Template);
    }
    let reason = if dir.join("tokenizer_config.json").is_file() {
        "tokenizer_config.json has no chat_template".to_string()
    } else {
        "tokenizer_config.json is missing".to_string()
    };
    Ok(Folder::PlainTextOnly(reason))
}

/// The index of the largest logit; the first one wins a tie, NaN never wins.
fn argmax(logits: &[f32]) -> Option<usize> {
    let mut best: Option<(usize, f32)> = None;
    for (i, &v) in logits.iter().enumerate() {
        if best.is_none_or(|(_, b)| v > b) {
            best = Some((i, v));
        }
    }
    best.map(|(i, _)| i)
}

#[cfg(test)]
mod tests {
    use crate::cli::evaluate::tests::synthetic_checkpoint;
    use crate::inference::{load_reference_model, CausalLm, DeviceArg};
    use crate::text::{render_chat_template, ChatMessage, Tokenizer};
    use std::path::Path;
    use std::sync::atomic::{AtomicBool, Ordering};

    use super::*;

    const TEMPLATE: &str = "{% for m in messages %}{{ m['role'] + ': ' + m['content'] + '\\n' }}{% endfor %}{% if add_generation_prompt %}{{ 'assistant:' }}{% endif %}";

    fn msg(role: &str, content: &str) -> ChatMessage {
        ChatMessage {
            role: role.to_string(),
            content: content.to_string(),
        }
    }

    /// Vocabulary word for an id: ids 0..=2 are the special and unknown tokens, the rest `w<id>`.
    fn word(id: u32) -> String {
        match id {
            0 => "<s>".to_string(),
            1 => "</s>".to_string(),
            2 => "<unk>".to_string(),
            _ => format!("w{id}"),
        }
    }

    /// A 64-word WordLevel tokenizer, the same size as the mock model's vocabulary.
    fn write_tokenizer(dir: &Path) {
        let vocab: serde_json::Map<String, serde_json::Value> = (0..64u32)
            .map(|id| (word(id), serde_json::json!(id)))
            .collect();
        let json = serde_json::json!({
            "version": "1.0", "truncation": null, "padding": null,
            "added_tokens": [
                {"id": 0, "content": "<s>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
                {"id": 1, "content": "</s>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}
            ],
            "normalizer": null, "pre_tokenizer": {"type": "Whitespace"},
            "post_processor": null, "decoder": null,
            "model": {"type": "WordLevel", "vocab": vocab, "unk_token": "<unk>"}
        });
        std::fs::write(dir.join("tokenizer.json"), json.to_string()).unwrap();
    }

    /// Writes the template and, when given, the end-of-sequence word.
    fn write_config(dir: &Path, eos: Option<&str>) {
        let mut config = serde_json::json!({ "chat_template": TEMPLATE });
        if let Some(eos) = eos {
            config["eos_token"] = serde_json::json!(eos);
        }
        std::fs::write(dir.join("tokenizer_config.json"), config.to_string()).unwrap();
    }

    /// A mock model folder with random weights, a tokenizer and a template.
    fn mock_folder(eos: Option<&str>) -> (tempfile::TempDir, std::path::PathBuf) {
        let root = tempfile::tempdir().unwrap();
        let (dir, _tokens) = synthetic_checkpoint(root.path());
        write_tokenizer(&dir);
        write_config(&dir, eos);
        (root, dir)
    }

    fn load(dir: &Path) -> (Box<dyn CausalLm>, Tokenizer) {
        let (_, _, model) = load_reference_model(dir, DeviceArg::Cpu).unwrap();
        (model, Tokenizer::from_model_dir(dir).unwrap())
    }

    fn request<'a>(dir: &'a Path, messages: &'a [ChatMessage], max: usize) -> Request<'a> {
        Request {
            dir,
            messages,
            plain_text: false,
            max_new_tokens: max,
            kv_precision: KvPrecision::F32,
        }
    }

    fn run(
        dir: &Path,
        messages: &[ChatMessage],
        max: usize,
        cancel: &AtomicBool,
    ) -> anyhow::Result<Reply> {
        let (model, tok) = load(dir);
        generate(
            model.as_ref(),
            &tok,
            &request(dir, messages, max),
            cancel,
            &mut |_| {},
        )
    }

    #[test]
    fn respects_max_new_tokens_when_no_eos_is_set() {
        let (_root, dir) = mock_folder(None);
        let messages = [msg("user", "hello world")];
        let reply = run(&dir, &messages, 3, &AtomicBool::new(false)).unwrap();
        assert_eq!(reply.token_ids.len(), 3);
        assert_eq!(reply.finish, Finish::MaxTokens);
    }

    #[test]
    fn stops_at_the_end_of_sequence_token() {
        let (_root, dir) = mock_folder(None);
        let messages = [msg("user", "hello world")];
        let free = run(&dir, &messages, 6, &AtomicBool::new(false)).unwrap();
        // The third free-running token becomes the end-of-sequence word; the run must stop
        // before its first occurrence, whatever the random weights chose.
        let eos_id = free.token_ids[2];
        write_config(&dir, Some(&word(eos_id)));
        let stopped = run(&dir, &messages, 6, &AtomicBool::new(false)).unwrap();
        let first = free.token_ids.iter().position(|&t| t == eos_id).unwrap();
        assert_eq!(stopped.token_ids, free.token_ids[..first]);
        assert_eq!(stopped.finish, Finish::Eos);
    }

    #[test]
    fn cancel_flag_set_before_the_run_generates_nothing() {
        let (_root, dir) = mock_folder(None);
        let messages = [msg("user", "hello")];
        let reply = run(&dir, &messages, 8, &AtomicBool::new(true)).unwrap();
        assert!(reply.token_ids.is_empty());
        assert_eq!(reply.finish, Finish::Cancelled);
    }

    #[test]
    fn cancel_flag_set_during_the_run_stops_it_and_progress_counts_tokens() {
        let (_root, dir) = mock_folder(None);
        let messages = [msg("user", "hello")];
        let (model, tok) = load(&dir);
        let cancel = AtomicBool::new(false);
        let mut seen = Vec::new();
        let reply = generate(
            model.as_ref(),
            &tok,
            &request(&dir, &messages, 8),
            &cancel,
            &mut |n| {
                seen.push(n);
                if n == 2 {
                    cancel.store(true, Ordering::Relaxed);
                }
            },
        )
        .unwrap();
        assert_eq!(seen, vec![1, 2]);
        assert_eq!(reply.token_ids.len(), 2);
        assert_eq!(reply.finish, Finish::Cancelled);
    }

    #[test]
    fn run_facts_are_filled_and_the_prompt_is_the_rendered_template() {
        let (_root, dir) = mock_folder(None);
        let messages = [msg("user", "hello world"), msg("assistant", "hi")];
        let reply = run(&dir, &messages, 4, &AtomicBool::new(false)).unwrap();
        let rendered = render_chat_template(&dir, &messages, true).unwrap();
        assert_eq!(reply.prompt, rendered);
        let tok = Tokenizer::from_model_dir(&dir).unwrap();
        assert_eq!(
            reply.prompt_tokens,
            tok.encode(&rendered, false).unwrap().len()
        );
        assert!(reply.prompt_tokens >= 2);
        assert!(reply.wall_secs >= 0.0);
        assert!(reply.tokens_per_sec.is_finite());
        if reply.wall_secs > 0.0 {
            let expected = reply.token_ids.len() as f64 / reply.wall_secs;
            assert!((reply.tokens_per_sec - expected).abs() < 1e-6);
        }
        assert_eq!(reply.text, tok.decode(&reply.token_ids).unwrap());
    }

    #[test]
    fn plain_text_mode_joins_turns_and_cues_the_reply() {
        let text = plain_prompt(&[msg("user", "hi"), msg("assistant", "hey")]);
        assert_eq!(text, "user: hi\nassistant: hey\nassistant:");
    }

    #[test]
    fn missing_tokenizer_says_what_is_missing() {
        let dir = tempfile::tempdir().unwrap();
        let err = check_folder(dir.path()).unwrap_err();
        assert!(err.contains("tokenizer.json is missing"), "{err}");
        assert!(err.contains("own tokenizer.json"), "{err}");
    }

    #[test]
    fn missing_template_offers_plain_text_only() {
        let (_root, dir) = mock_folder(None);
        std::fs::write(dir.join("tokenizer_config.json"), "{}").unwrap();
        match check_folder(&dir).unwrap() {
            Folder::PlainTextOnly(reason) => assert!(reason.contains("chat_template"), "{reason}"),
            Folder::Template => panic!("no template must not count as template"),
        }
        let (_root, full) = mock_folder(None);
        assert_eq!(check_folder(&full).unwrap(), Folder::Template);
    }

    #[test]
    fn missing_config_is_named_in_the_reason() {
        let (_root, dir) = mock_folder(None);
        std::fs::remove_file(dir.join("tokenizer_config.json")).unwrap();
        match check_folder(&dir).unwrap() {
            Folder::PlainTextOnly(reason) => {
                assert!(
                    reason.contains("tokenizer_config.json is missing"),
                    "{reason}"
                )
            }
            Folder::Template => panic!("missing config must not count as template"),
        }
    }

    #[test]
    fn cached_reply_is_the_whole_sequence_greedy_decode() {
        // The reply loop now prefills once and steps the KV cache; the old whole-sequence
        // greedy loop is the reference it must reproduce token for token.
        let (_root, dir) = mock_folder(None);
        let messages = [msg("user", "hello world")];
        let (model, tok) = load(&dir);
        let reply = generate(
            model.as_ref(),
            &tok,
            &request(&dir, &messages, 8),
            &AtomicBool::new(false),
            &mut |_| {},
        )
        .unwrap();
        let mut ids = tok.encode(&reply.prompt, false).unwrap();
        let mut expected = Vec::new();
        for _ in 0..8 {
            let logits = model.forward_logits(&ids).unwrap();
            let next = u32::try_from(argmax(&logits).unwrap()).unwrap();
            ids.push(next);
            expected.push(next);
        }
        // The loop breaker may cut a degenerate repeat short, never change the tokens before the cut.
        assert!(!reply.token_ids.is_empty());
        assert_eq!(reply.token_ids, expected[..reply.token_ids.len()]);
    }

    #[test]
    fn generate_with_int8_and_int4_records_smaller_cache_bytes() {
        let (_root, dir) = mock_folder(None);
        let messages = [msg("user", "hello world")];
        let (model, tok) = load(&dir);
        let run_prec = |kv_precision: KvPrecision| {
            let mut req = request(&dir, &messages, 3);
            req.kv_precision = kv_precision;
            generate(
                model.as_ref(),
                &tok,
                &req,
                &AtomicBool::new(false),
                &mut |_| {},
            )
            .unwrap()
        };

        let f32_reply = run_prec(KvPrecision::F32);
        let int8_reply = run_prec(KvPrecision::Int8);
        let int4_reply = run_prec(KvPrecision::Int4);

        assert_eq!(f32_reply.kv_precision, KvPrecision::F32);
        assert_eq!(int8_reply.kv_precision, KvPrecision::Int8);
        assert_eq!(int4_reply.kv_precision, KvPrecision::Int4);
        assert!(int4_reply.kv_cache_bytes > 0);
        assert!(int4_reply.kv_cache_bytes < int8_reply.kv_cache_bytes);
        assert!(int8_reply.kv_cache_bytes < f32_reply.kv_cache_bytes);
    }

    #[test]
    fn generate_errors_when_model_lacks_requested_kv_precision() {
        use crate::inference::llama::DepthExit;
        use crate::inference::probe::ForwardTrace;

        struct NoCacheModel;
        impl CausalLm for NoCacheModel {
            fn vocab_size(&self) -> usize {
                64
            }
            fn num_layers(&self) -> usize {
                1
            }
            fn forward_scheduled(
                &self,
                _tokens: &[u32],
                _schedule: &[usize],
                _exits: &[usize],
            ) -> anyhow::Result<Vec<DepthExit>> {
                Ok(Vec::new())
            }
            fn forward_traced(
                &self,
                _tokens: &[u32],
                _schedule: &[usize],
            ) -> anyhow::Result<(DepthExit, ForwardTrace)> {
                anyhow::bail!("unused in test")
            }
            fn forward_logits(&self, _tokens: &[u32]) -> anyhow::Result<Vec<f32>> {
                Ok(vec![0.0; 64])
            }
        }

        let (_root, dir) = mock_folder(None);
        let tok = Tokenizer::from_model_dir(&dir).unwrap();
        let messages = [msg("user", "hello world")];
        let model = NoCacheModel;

        for (precision, expected_name) in [(KvPrecision::Int8, "int8"), (KvPrecision::Int4, "int4")]
        {
            let mut req = request(&dir, &messages, 2);
            req.kv_precision = precision;
            let err = generate(&model, &tok, &req, &AtomicBool::new(false), &mut |_| {})
                .unwrap_err()
                .to_string();
            assert!(
                err.contains(expected_name),
                "expected error naming '{expected_name}', got: {err}"
            );
        }
    }
}
