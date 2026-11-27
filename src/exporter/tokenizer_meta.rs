// SPDX-License-Identifier: AGPL-3.0-only
//! Clean-room Hugging Face `tokenizer.json` to GGUF tokenizer metadata converter (THME-3091).

use anyhow::{bail, Context, Result};
use serde_json::Value;
use std::collections::{BTreeMap, HashSet};

use crate::parser::gguf::GgufValue;

const SPECIAL_ID_KEYS: [(&str, &str); 4] = [
    ("bos_token_id", "tokenizer.ggml.bos_token_id"),
    ("eos_token_id", "tokenizer.ggml.eos_token_id"),
    ("unk_token_id", "tokenizer.ggml.unknown_token_id"),
    ("pad_token_id", "tokenizer.ggml.padding_token_id"),
];

/// Converts Hugging Face `tokenizer.json` (and optional `config.json`) into GGUF tokenizer
/// metadata entries (`tokenizer.ggml.*`).
pub fn tokenizer_metadata(
    tokenizer_json: &str,
    config_json: Option<&str>,
) -> Result<Vec<(String, GgufValue)>> {
    let root: Value =
        serde_json::from_str(tokenizer_json).context("Failed to parse tokenizer.json")?;
    let model = root
        .get("model")
        .and_then(Value::as_object)
        .context("tokenizer.json missing 'model' object")?;
    let vocab = model
        .get("vocab")
        .and_then(Value::as_object)
        .context("tokenizer.json missing 'model.vocab' object")?;

    let mut id_to_token: BTreeMap<usize, String> = BTreeMap::new();
    for (token, id_val) in vocab {
        let id_u64 = id_val
            .as_u64()
            .with_context(|| format!("Token '{token}' has non-integer id {id_val}"))?;
        let id = usize::try_from(id_u64)
            .with_context(|| format!("Token '{token}' id {id_u64} exceeds usize"))?;
        if id_to_token.insert(id, token.clone()).is_some() {
            bail!("Duplicate token id {id} in tokenizer.json");
        }
    }

    let mut special_ids: HashSet<usize> = HashSet::new();
    if let Some(added_val) = root.get("added_tokens").filter(|v| !v.is_null()) {
        let added = added_val
            .as_array()
            .context("tokenizer.json 'added_tokens' must be an array")?;
        for (idx, entry) in added.iter().enumerate() {
            let obj = entry
                .as_object()
                .with_context(|| format!("tokenizer.json added_tokens[{idx}] must be an object"))?;
            let id_u64 = obj.get("id").and_then(Value::as_u64).with_context(|| {
                format!("tokenizer.json added_tokens[{idx}] missing integer 'id'")
            })?;
            let id = usize::try_from(id_u64).with_context(|| {
                format!("tokenizer.json added_tokens[{idx}] id {id_u64} exceeds usize")
            })?;
            if obj.get("special").and_then(Value::as_bool) == Some(true) {
                special_ids.insert(id);
            }
            if let Some(content) = obj.get("content").and_then(Value::as_str) {
                id_to_token.entry(id).or_insert_with(|| content.to_string());
            }
        }
    }

    let Some(&max_id) = id_to_token.keys().next_back() else {
        bail!("tokenizer.json has an empty vocabulary");
    };
    let vocab_len = max_id
        .checked_add(1)
        .context("Tokenizer vocabulary size overflows usize")?;
    let mut tokens = Vec::with_capacity(vocab_len);
    let mut scores = Vec::with_capacity(vocab_len);
    let mut token_types = Vec::with_capacity(vocab_len);
    for id in 0..=max_id {
        let token = id_to_token
            .get(&id)
            .ok_or_else(|| anyhow::anyhow!("Gap in tokenizer.json token ids: missing id {id}"))?;
        tokens.push(GgufValue::String(token.clone()));
        scores.push(GgufValue::Float32(0.0));
        let token_type = if special_ids.contains(&id) { 3 } else { 1 };
        token_types.push(GgufValue::Int32(token_type));
    }

    let mut merges = Vec::new();
    if let Some(merges_val) = model.get("merges").filter(|v| !v.is_null()) {
        let arr = merges_val
            .as_array()
            .context("tokenizer.json 'model.merges' must be an array")?;
        merges.reserve(arr.len());
        for (idx, item) in arr.iter().enumerate() {
            let merge = match item {
                Value::String(s) => s.clone(),
                Value::Array(pair) => match pair.as_slice() {
                    [Value::String(a), Value::String(b)] => format!("{a} {b}"),
                    _ => bail!("tokenizer.json merge entry {idx} must be a [a, b] string pair"),
                },
                _ => bail!("tokenizer.json merge entry {idx} must be a string or [a, b] pair"),
            };
            merges.push(GgufValue::String(merge));
        }
    }

    let is_byte_level = root.get("pre_tokenizer").is_some_and(has_byte_level)
        || root.get("decoder").is_some_and(has_byte_level);
    let model_kind = if is_byte_level { "gpt2" } else { "llama" };

    let mut metadata = vec![
        (
            "tokenizer.ggml.model".to_string(),
            GgufValue::String(model_kind.to_string()),
        ),
        (
            "tokenizer.ggml.tokens".to_string(),
            GgufValue::Array(tokens),
        ),
        (
            "tokenizer.ggml.scores".to_string(),
            GgufValue::Array(scores),
        ),
        (
            "tokenizer.ggml.token_type".to_string(),
            GgufValue::Array(token_types),
        ),
    ];
    if !merges.is_empty() {
        metadata.push((
            "tokenizer.ggml.merges".to_string(),
            GgufValue::Array(merges),
        ));
    }

    if let Some(cfg_raw) = config_json {
        let cfg: Value = serde_json::from_str(cfg_raw)
            .context("Failed to parse config.json for tokenizer special token ids")?;
        for (cfg_key, gguf_key) in SPECIAL_ID_KEYS {
            if let Some(val) = cfg.get(cfg_key).filter(|v| !v.is_null()) {
                let id = parse_token_id(val, cfg_key)?;
                metadata.push((gguf_key.to_string(), GgufValue::Uint32(id)));
            }
        }
    }

    Ok(metadata)
}

fn has_byte_level(node: &Value) -> bool {
    let Some(obj) = node.as_object() else {
        return false;
    };
    if obj.get("type").and_then(Value::as_str) == Some("ByteLevel") {
        return true;
    }
    for key in ["pretokenizers", "decoders"] {
        if let Some(items) = obj.get(key).and_then(Value::as_array) {
            if items.iter().any(has_byte_level) {
                return true;
            }
        }
    }
    false
}

fn parse_token_id(val: &Value, key: &str) -> Result<u32> {
    let raw = match val {
        Value::Number(n) => n
            .as_u64()
            .with_context(|| format!("config.json '{key}' must be a non-negative integer"))?,
        Value::Array(items) => items.first().and_then(Value::as_u64).with_context(|| {
            format!("config.json '{key}' array must start with a non-negative integer")
        })?,
        _ => bail!("config.json '{key}' must be a non-negative integer"),
    };
    u32::try_from(raw).with_context(|| format!("config.json '{key}' value {raw} exceeds u32"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::model::ModelMetadataValue;
    use std::collections::HashMap;

    fn as_map(entries: Vec<(String, GgufValue)>) -> HashMap<String, ModelMetadataValue> {
        entries
            .into_iter()
            .map(|(k, v)| (k, ModelMetadataValue::from(v)))
            .collect()
    }

    #[test]
    fn five_token_byte_level_bpe_metadata_matches_literal_expectations() {
        let tokenizer_json = r#"{
            "added_tokens": [
                {"id": 0, "content": "<s>", "special": true},
                {"id": 1, "content": "</s>", "special": true}
            ],
            "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false},
            "decoder": {"type": "ByteLevel"},
            "model": {
                "type": "BPE",
                "vocab": {"ab": 4, "a": 2, "<s>": 0, "b": 3, "</s>": 1},
                "merges": ["a b"]
            }
        }"#;
        let config_json = r#"{"bos_token_id": 0, "eos_token_id": 1, "pad_token_id": null}"#;

        let map = as_map(tokenizer_metadata(tokenizer_json, Some(config_json)).unwrap());
        assert_eq!(
            map.get("tokenizer.ggml.model"),
            Some(&ModelMetadataValue::String("gpt2".to_string()))
        );
        assert_eq!(
            map.get("tokenizer.ggml.tokens"),
            Some(&ModelMetadataValue::Array(vec![
                ModelMetadataValue::String("<s>".to_string()),
                ModelMetadataValue::String("</s>".to_string()),
                ModelMetadataValue::String("a".to_string()),
                ModelMetadataValue::String("b".to_string()),
                ModelMetadataValue::String("ab".to_string()),
            ]))
        );
        assert_eq!(
            map.get("tokenizer.ggml.scores"),
            Some(&ModelMetadataValue::Array(vec![
                ModelMetadataValue::Float32(0.0),
                ModelMetadataValue::Float32(0.0),
                ModelMetadataValue::Float32(0.0),
                ModelMetadataValue::Float32(0.0),
                ModelMetadataValue::Float32(0.0),
            ]))
        );
        assert_eq!(
            map.get("tokenizer.ggml.token_type"),
            Some(&ModelMetadataValue::Array(vec![
                ModelMetadataValue::Int32(3),
                ModelMetadataValue::Int32(3),
                ModelMetadataValue::Int32(1),
                ModelMetadataValue::Int32(1),
                ModelMetadataValue::Int32(1),
            ]))
        );
        assert_eq!(
            map.get("tokenizer.ggml.merges"),
            Some(&ModelMetadataValue::Array(vec![
                ModelMetadataValue::String("a b".to_string())
            ]))
        );
        assert_eq!(
            map.get("tokenizer.ggml.bos_token_id"),
            Some(&ModelMetadataValue::Uint32(0))
        );
        assert_eq!(
            map.get("tokenizer.ggml.eos_token_id"),
            Some(&ModelMetadataValue::Uint32(1))
        );
        assert!(!map.contains_key("tokenizer.ggml.padding_token_id"));
        assert!(!map.contains_key("tokenizer.ggml.unknown_token_id"));
    }

    #[test]
    fn model_kind_detects_byte_level_sequence_and_llama_fallback() {
        let llama_tok = r#"{
            "added_tokens": [],
            "pre_tokenizer": {"type": "Metaspace"},
            "decoder": null,
            "model": {"type": "BPE", "vocab": {"<s>": 0, "a": 1}, "merges": []}
        }"#;
        let llama_map = as_map(tokenizer_metadata(llama_tok, None).unwrap());
        assert_eq!(
            llama_map.get("tokenizer.ggml.model"),
            Some(&ModelMetadataValue::String("llama".to_string()))
        );

        let seq_tok = r#"{
            "added_tokens": [],
            "pre_tokenizer": {
                "type": "Sequence",
                "pretokenizers": [
                    {"type": "Split"},
                    {"type": "ByteLevel"}
                ]
            },
            "model": {"type": "BPE", "vocab": {"<s>": 0, "a": 1}, "merges": []}
        }"#;
        let seq_map = as_map(tokenizer_metadata(seq_tok, None).unwrap());
        assert_eq!(
            seq_map.get("tokenizer.ggml.model"),
            Some(&ModelMetadataValue::String("gpt2".to_string()))
        );
    }

    #[test]
    fn id_gap_returns_error_naming_missing_id() {
        let gap_tok = r#"{
            "added_tokens": [],
            "model": {"type": "BPE", "vocab": {"<s>": 0, "</s>": 1, "b": 3}, "merges": []}
        }"#;
        let err = tokenizer_metadata(gap_tok, None).unwrap_err();
        assert!(err.to_string().contains('2'), "{err:#}");

        let missing_zero = r#"{
            "added_tokens": [],
            "model": {"type": "BPE", "vocab": {"a": 1, "b": 2}, "merges": []}
        }"#;
        let err_zero = tokenizer_metadata(missing_zero, None).unwrap_err();
        assert!(err_zero.to_string().contains('0'), "{err_zero:#}");
    }

    #[test]
    fn both_merge_encodings_parse_and_config_maps_unk_and_pad() {
        let pair_tok = r#"{
            "added_tokens": [
                {"id": 0, "content": "<s>", "special": true},
                {"id": 1, "content": "</s>", "special": true}
            ],
            "decoder": {"type": "ByteLevel"},
            "model": {
                "type": "BPE",
                "vocab": {"<s>": 0, "</s>": 1, "a": 2, "b": 3, "ab": 4},
                "merges": [["a", "b"]]
            }
        }"#;
        let cfg =
            r#"{"bos_token_id": 0, "eos_token_id": [1, 0], "unk_token_id": 2, "pad_token_id": 3}"#;

        let map = as_map(tokenizer_metadata(pair_tok, Some(cfg)).unwrap());
        assert_eq!(
            map.get("tokenizer.ggml.merges"),
            Some(&ModelMetadataValue::Array(vec![
                ModelMetadataValue::String("a b".to_string())
            ]))
        );
        assert_eq!(
            map.get("tokenizer.ggml.unknown_token_id"),
            Some(&ModelMetadataValue::Uint32(2))
        );
        assert_eq!(
            map.get("tokenizer.ggml.padding_token_id"),
            Some(&ModelMetadataValue::Uint32(3))
        );
        assert_eq!(
            map.get("tokenizer.ggml.eos_token_id"),
            Some(&ModelMetadataValue::Uint32(1))
        );
    }
}
