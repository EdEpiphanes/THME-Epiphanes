// SPDX-License-Identifier: AGPL-3.0-only
//! Fixtures are written into temporary folders by each test: a tiny WordLevel `tokenizer.json`
//! and small `tokenizer_config.json` files. No model is opened or downloaded.

use super::*;
use serde_json::json;

/// A WordLevel tokenizer: `<s>` is added on encode, `hello` and `world` are words, anything else
/// maps to `<unk>` (id 2, not a special token, so it shows in decoded text).
const TOKENIZER_JSON: &str = r#"{
  "version": "1.0",
  "truncation": null,
  "padding": null,
  "added_tokens": [
    {"id": 0, "content": "<s>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
    {"id": 1, "content": "</s>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}
  ],
  "normalizer": null,
  "pre_tokenizer": {"type": "Whitespace"},
  "post_processor": {
    "type": "TemplateProcessing",
    "single": [
      {"SpecialToken": {"id": "<s>", "type_id": 0}},
      {"Sequence": {"id": "A", "type_id": 0}}
    ],
    "pair": [
      {"SpecialToken": {"id": "<s>", "type_id": 0}},
      {"Sequence": {"id": "A", "type_id": 0}},
      {"Sequence": {"id": "B", "type_id": 1}}
    ],
    "special_tokens": {"<s>": {"id": "<s>", "ids": [0], "tokens": ["<s>"]}}
  },
  "decoder": null,
  "model": {
    "type": "WordLevel",
    "vocab": {"<s>": 0, "</s>": 1, "<unk>": 2, "hello": 3, "world": 4},
    "unk_token": "<unk>"
  }
}"#;

const CHATML: &str = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}";

const LLAMA: &str = "{% for message in messages %}{% if message['role'] == 'user' %}{{ bos_token + '[INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ ' ' + message['content'] + ' ' + eos_token }}{% endif %}{% endfor %}";

const ALTERNATION: &str = "{% for m in messages %}{% if m['role'] != ('user' if loop.index0 % 2 == 0 else 'assistant') %}{{ raise_exception('Conversation roles must alternate user and assistant') }}{% endif %}{{ m['content'] }}{% endfor %}";

fn msg(role: &str, content: &str) -> ChatMessage {
    ChatMessage {
        role: role.to_string(),
        content: content.to_string(),
    }
}

fn write_tokenizer(dir: &Path) {
    std::fs::write(dir.join("tokenizer.json"), TOKENIZER_JSON).unwrap();
}

fn write_config(dir: &Path, config: serde_json::Value) {
    std::fs::write(dir.join("tokenizer_config.json"), config.to_string()).unwrap();
}

#[test]
fn encode_then_decode_round_trips_words() {
    let dir = tempfile::tempdir().unwrap();
    write_tokenizer(dir.path());
    let tok = Tokenizer::from_model_dir(dir.path()).unwrap();
    let ids = tok.encode("hello world", false).unwrap();
    assert_eq!(ids, vec![3, 4]);
    assert_eq!(tok.decode(&ids).unwrap(), "hello world");
}

#[test]
fn add_special_puts_the_bos_token_first_and_decode_hides_it() {
    let dir = tempfile::tempdir().unwrap();
    write_tokenizer(dir.path());
    let tok = Tokenizer::from_model_dir(dir.path()).unwrap();
    assert_eq!(tok.encode("hello", true).unwrap(), vec![0, 3]);
    assert_eq!(tok.encode("hello", false).unwrap(), vec![3]);
    assert_eq!(tok.decode(&[0, 3]).unwrap(), "hello");
}

#[test]
fn unknown_word_maps_to_unk_and_decodes_as_text() {
    let dir = tempfile::tempdir().unwrap();
    write_tokenizer(dir.path());
    let tok = Tokenizer::from_model_dir(dir.path()).unwrap();
    let ids = tok.encode("hello zebra", false).unwrap();
    assert_eq!(ids, vec![3, 2]);
    assert_eq!(tok.decode(&ids).unwrap(), "hello <unk>");
}

#[test]
fn missing_tokenizer_json_names_the_file() {
    let dir = tempfile::tempdir().unwrap();
    let err = Tokenizer::from_model_dir(dir.path()).unwrap_err();
    assert!(err.to_string().contains("tokenizer.json"), "{err:#}");
}

#[test]
fn broken_tokenizer_json_is_an_error_not_a_panic() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("tokenizer.json"), "{ not json").unwrap();
    let err = Tokenizer::from_model_dir(dir.path()).unwrap_err();
    assert!(err.to_string().contains("tokenizer.json"), "{err:#}");
}

#[test]
fn chatml_template_renders_with_generation_prompt() {
    let dir = tempfile::tempdir().unwrap();
    write_config(dir.path(), json!({ "chat_template": CHATML }));
    let messages = [msg("system", "You are helpful."), msg("user", "Hi")];
    let out = render_chat_template(dir.path(), &messages, true).unwrap();
    assert_eq!(
        out,
        "<|im_start|>system\nYou are helpful.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n"
    );
    let no_prompt = render_chat_template(dir.path(), &messages, false).unwrap();
    assert!(
        !no_prompt.ends_with("<|im_start|>assistant\n"),
        "{no_prompt}"
    );
}

#[test]
fn llama_template_uses_bos_object_and_eos_string() {
    let dir = tempfile::tempdir().unwrap();
    write_config(
        dir.path(),
        json!({
            "chat_template": LLAMA,
            "bos_token": {"content": "<s>", "lstrip": false},
            "eos_token": "</s>"
        }),
    );
    let messages = [
        msg("user", "Hi"),
        msg("assistant", "Hello"),
        msg("user", "How are you?"),
    ];
    let out = render_chat_template(dir.path(), &messages, true).unwrap();
    assert_eq!(
        out,
        "<s>[INST] Hi [/INST] Hello </s><s>[INST] How are you? [/INST]"
    );
}

#[test]
fn list_form_template_uses_the_default_entry() {
    let dir = tempfile::tempdir().unwrap();
    write_config(
        dir.path(),
        json!({
            "chat_template": [
                {"name": "tool_use", "template": "TOOLS"},
                {"name": "default", "template": CHATML}
            ]
        }),
    );
    let out = render_chat_template(dir.path(), &[msg("user", "Hi")], false).unwrap();
    assert_eq!(out, "<|im_start|>user\nHi<|im_end|>\n");
}

#[test]
fn list_form_without_default_is_an_error() {
    let dir = tempfile::tempdir().unwrap();
    write_config(
        dir.path(),
        json!({"chat_template": [{"name": "tool_use", "template": "TOOLS"}]}),
    );
    let err = render_chat_template(dir.path(), &[], true).unwrap_err();
    assert!(err.to_string().contains("default"), "{err:#}");
}

#[test]
fn raise_exception_in_template_becomes_an_error() {
    let dir = tempfile::tempdir().unwrap();
    write_config(dir.path(), json!({ "chat_template": ALTERNATION }));
    let ok = [msg("user", "a"), msg("assistant", "b")];
    assert_eq!(render_chat_template(dir.path(), &ok, false).unwrap(), "ab");
    let bad = [msg("assistant", "b")];
    let err = render_chat_template(dir.path(), &bad, false).unwrap_err();
    assert!(err.to_string().contains("must alternate"), "{err:#}");
}

#[test]
fn python_string_methods_work_in_templates() {
    let dir = tempfile::tempdir().unwrap();
    write_config(
        dir.path(),
        json!({ "chat_template": "{{ messages[0]['content'].strip().upper() }}" }),
    );
    let out = render_chat_template(dir.path(), &[msg("user", "  hi  ")], false).unwrap();
    assert_eq!(out, "HI");
}

#[test]
fn strftime_now_format_is_checked_on_a_fixed_instant() {
    // 2026-11-19 00:00:00 UTC was a Thursday.
    assert_eq!(format_date(1_795_046_400, "%d %b %Y"), "19 Nov 2026");
    assert_eq!(
        format_date(1_795_046_400, "%A %B %e %%"),
        "Thursday November 19 %"
    );
    assert_eq!(format_date(0, "%Y-%m-%d %H:%M:%S"), "1970-01-01 00:00:00");
}

#[test]
fn missing_tokenizer_config_names_the_file() {
    let dir = tempfile::tempdir().unwrap();
    let err = render_chat_template(dir.path(), &[], true).unwrap_err();
    assert!(err.to_string().contains("tokenizer_config.json"), "{err:#}");
}

#[test]
fn config_without_chat_template_says_so() {
    let dir = tempfile::tempdir().unwrap();
    write_config(dir.path(), json!({ "bos_token": "<s>" }));
    let err = render_chat_template(dir.path(), &[], true).unwrap_err();
    assert!(err.to_string().contains("chat_template"), "{err:#}");
}

#[test]
fn scanner_still_flags_the_template_text_the_renderer_reads() {
    let dir = tempfile::tempdir().unwrap();
    let injected = "{{ ''.__class__.__mro__[1].__subclasses__() }}";
    write_config(dir.path(), json!({ "chat_template": injected }));
    let config = read_config(dir.path()).unwrap();
    assert_eq!(template_of(&config).unwrap(), injected);
    let report = crate::scan::scan(dir.path()).unwrap();
    let hit = report
        .findings
        .iter()
        .find(|f| f.rule == "template-injection")
        .unwrap();
    assert_eq!(hit.file, "tokenizer_config.json");
    assert!(hit.detail.contains("chat_template"), "{}", hit.detail);
}
