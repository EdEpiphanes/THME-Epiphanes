// SPDX-License-Identifier: AGPL-3.0-only
//! Tests for `thme download`. The fake fetcher stands in for the network, so nothing here
//! talks to the Hugging Face hub.

use super::*;
use anyhow::{anyhow, Result};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::cell::RefCell;
use std::collections::HashMap;
use std::io::Write;
use std::path::Path;

const TOKEN: &str = "hf_test_secret_token_123";
const API: &str = "https://huggingface.co/api/models/org/name/revision/main?blobs=true";
const FILE_PREFIX: &str = "https://huggingface.co/org/name/resolve/main/";

#[derive(Default)]
struct Fake {
    texts: HashMap<String, String>,
    files: HashMap<String, Vec<u8>>,
    calls: RefCell<Vec<String>>,
    tokens_seen: RefCell<Vec<Option<String>>>,
}

impl Fetch for Fake {
    fn text(&self, url: &str, token: Option<&str>) -> Result<String> {
        self.calls.borrow_mut().push(url.to_string());
        self.tokens_seen
            .borrow_mut()
            .push(token.map(str::to_string));
        self.texts
            .get(url)
            .cloned()
            .ok_or_else(|| anyhow!("fake: no text for {url}"))
    }

    fn file(&self, url: &str, token: Option<&str>, sink: &mut dyn Write) -> Result<u64> {
        self.calls.borrow_mut().push(url.to_string());
        self.tokens_seen
            .borrow_mut()
            .push(token.map(str::to_string));
        let bytes = self
            .files
            .get(url)
            .ok_or_else(|| anyhow!("fake: no file for {url}"))?;
        sink.write_all(bytes)?;
        Ok(u64::try_from(bytes.len())?)
    }
}

fn hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// One API sibling. LFS entries carry the sha256 the hub reports; plain files carry only size.
fn sib(name: &str, bytes: &[u8], lfs: bool) -> Value {
    let mut s = json!({"rfilename": name, "size": bytes.len()});
    if lfs {
        s["lfs"] = json!({"sha256": hex(bytes), "size": bytes.len()});
    }
    s
}

fn api_json(license: Option<&str>, siblings: Vec<Value>) -> String {
    let card = match license {
        Some(l) => json!({"license": l}),
        None => json!({}),
    };
    json!({"id": "org/name", "siblings": siblings, "cardData": card}).to_string()
}

fn fake_with(json: String) -> Fake {
    let mut f = Fake::default();
    f.texts.insert(API.to_string(), json);
    f
}

fn request<'a>(files: &'a [String], out: &'a Path, dry_run: bool) -> Request<'a> {
    Request {
        repo: "org/name",
        revision: "main",
        files,
        out,
        dry_run,
    }
}

fn file_calls(f: &Fake) -> Vec<String> {
    f.calls
        .borrow()
        .iter()
        .filter(|u| u.starts_with(FILE_PREFIX))
        .cloned()
        .collect()
}

fn names(v: &[String]) -> Vec<&str> {
    v.iter().map(String::as_str).collect()
}

#[test]
fn default_selection_for_safetensors_repo_skips_pickle() {
    let dir = tempfile::tempdir().unwrap();
    let configs: [(&str, &[u8]); 5] = [
        ("config.json", b"{}"),
        ("generation_config.json", b"{}"),
        ("tokenizer.json", b"{}"),
        ("tokenizer_config.json", b"{}"),
        ("special_tokens_map.json", b"{}"),
    ];
    let mut sibs: Vec<Value> = configs.iter().map(|(n, b)| sib(n, b, false)).collect();
    sibs.push(sib("model.safetensors", b"weights", true));
    sibs.push(sib("pytorch_model.bin", b"pickle", true));
    sibs.push(sib("README.md", b"# card", false));
    let mut f = fake_with(api_json(Some("apache-2.0"), sibs));
    for (n, b) in configs {
        f.files.insert(format!("{FILE_PREFIX}{n}"), b.to_vec());
    }
    f.files.insert(
        format!("{FILE_PREFIX}model.safetensors"),
        b"weights".to_vec(),
    );
    f.files.insert(
        format!("{FILE_PREFIX}pytorch_model.bin"),
        b"pickle".to_vec(),
    );

    let mut log: Vec<u8> = Vec::new();
    let out = run(&f, &request(&[], dir.path(), false), None, &mut log).unwrap();

    assert_eq!(
        names(&out.downloaded),
        [
            "config.json",
            "generation_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "model.safetensors"
        ]
    );
    assert!(file_calls(&f).iter().all(|u| !u.ends_with(".bin")));
    assert!(dir.path().join("model.safetensors").is_file());
}

#[test]
fn sharded_repo_fetches_index_then_each_shard_once() {
    let dir = tempfile::tempdir().unwrap();
    let index = br#"{"metadata":{},"weight_map":{"a":"model-00001-of-00002.safetensors","b":"model-00002-of-00002.safetensors","c":"model-00001-of-00002.safetensors"}}"#;
    let sibs = vec![
        sib("config.json", b"{}", false),
        sib("model.safetensors.index.json", index, false),
        sib("model-00001-of-00002.safetensors", b"shard1", true),
        sib("model-00002-of-00002.safetensors", b"shard2", true),
    ];
    let mut f = fake_with(api_json(Some("mit"), sibs));
    f.files
        .insert(format!("{FILE_PREFIX}config.json"), b"{}".to_vec());
    f.files.insert(
        format!("{FILE_PREFIX}model.safetensors.index.json"),
        index.to_vec(),
    );
    f.files.insert(
        format!("{FILE_PREFIX}model-00001-of-00002.safetensors"),
        b"shard1".to_vec(),
    );
    f.files.insert(
        format!("{FILE_PREFIX}model-00002-of-00002.safetensors"),
        b"shard2".to_vec(),
    );

    let mut log: Vec<u8> = Vec::new();
    let out = run(&f, &request(&[], dir.path(), false), None, &mut log).unwrap();

    assert_eq!(
        names(&out.downloaded),
        [
            "config.json",
            "model.safetensors.index.json",
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors"
        ]
    );
    assert!(!dir.path().join("model.safetensors").exists());
}

#[test]
fn gguf_only_repo_selects_the_single_gguf() {
    let dir = tempfile::tempdir().unwrap();
    let sibs = vec![
        sib("README.md", b"# card", false),
        sib("model-Q4_K_M.gguf", b"gguf", true),
    ];
    let mut f = fake_with(api_json(Some("apache-2.0"), sibs));
    f.files
        .insert(format!("{FILE_PREFIX}model-Q4_K_M.gguf"), b"gguf".to_vec());

    let mut log: Vec<u8> = Vec::new();
    let out = run(&f, &request(&[], dir.path(), false), None, &mut log).unwrap();

    assert_eq!(names(&out.downloaded), ["model-Q4_K_M.gguf"]);
}

#[test]
fn gguf_repo_with_several_weights_asks_for_file_names() {
    let dir = tempfile::tempdir().unwrap();
    let sibs = vec![sib("a.gguf", b"a", true), sib("b.gguf", b"b", true)];
    let f = fake_with(api_json(Some("mit"), sibs));

    let mut log: Vec<u8> = Vec::new();
    let err = run(&f, &request(&[], dir.path(), false), None, &mut log).unwrap_err();

    assert!(err.to_string().contains("--file"));
    assert!(file_calls(&f).is_empty());
}

#[test]
fn pickle_only_repo_downloads_nothing_by_default() {
    let dir = tempfile::tempdir().unwrap();
    let sibs = vec![
        sib("config.json", b"{}", false),
        sib("pytorch_model.bin", b"pickle", true),
    ];
    let f = fake_with(api_json(Some("mit"), sibs));

    let mut log: Vec<u8> = Vec::new();
    let err = run(&f, &request(&[], dir.path(), false), None, &mut log).unwrap_err();

    assert!(err.to_string().contains("--file"));
    assert!(file_calls(&f).is_empty());
}

#[test]
fn named_pickle_file_is_downloaded_with_warning() {
    let dir = tempfile::tempdir().unwrap();
    let sibs = vec![sib("pytorch_model.bin", b"pickle", true)];
    let mut f = fake_with(api_json(Some("mit"), sibs));
    f.files.insert(
        format!("{FILE_PREFIX}pytorch_model.bin"),
        b"pickle".to_vec(),
    );

    let mut log: Vec<u8> = Vec::new();
    let files = vec!["pytorch_model.bin".to_string()];
    let out = run(&f, &request(&files, dir.path(), false), None, &mut log).unwrap();

    assert_eq!(names(&out.downloaded), ["pytorch_model.bin"]);
    let text = String::from_utf8_lossy(&log);
    assert!(text.contains("warning"));
    assert!(text.contains("can execute code"));
}

#[test]
fn checksum_mismatch_leaves_no_final_file() {
    let dir = tempfile::tempdir().unwrap();
    let mut bad = sib("model.safetensors", b"weights", false);
    bad["lfs"] = json!({"sha256": hex(b"something else"), "size": 7});
    let mut f = fake_with(api_json(Some("mit"), vec![bad]));
    f.files.insert(
        format!("{FILE_PREFIX}model.safetensors"),
        b"weights".to_vec(),
    );

    let files = vec!["model.safetensors".to_string()];
    let mut log: Vec<u8> = Vec::new();
    let err = run(&f, &request(&files, dir.path(), false), None, &mut log).unwrap_err();

    assert!(format!("{err:#}").contains("checksum mismatch"));
    assert!(!dir.path().join("model.safetensors").exists());
    assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 0);
}

#[test]
fn existing_matching_file_is_not_downloaded_again() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("config.json"), b"{}").unwrap();
    let mut f = fake_with(api_json(Some("mit"), vec![sib("config.json", b"{}", true)]));
    f.files
        .insert(format!("{FILE_PREFIX}config.json"), b"{}".to_vec());

    let files = vec!["config.json".to_string()];
    let mut log: Vec<u8> = Vec::new();
    let out = run(&f, &request(&files, dir.path(), false), None, &mut log).unwrap();

    assert!(out.downloaded.is_empty());
    assert_eq!(names(&out.kept), ["config.json"]);
    assert!(file_calls(&f).is_empty());
}

#[test]
fn token_is_sent_but_never_printed() {
    let dir = tempfile::tempdir().unwrap();
    let sibs = vec![sib("model.safetensors", b"weights", true)];
    let mut f = fake_with(api_json(Some("mit"), sibs));
    f.files.insert(
        format!("{FILE_PREFIX}model.safetensors"),
        b"weights".to_vec(),
    );

    let mut log: Vec<u8> = Vec::new();
    run(&f, &request(&[], dir.path(), false), Some(TOKEN), &mut log).unwrap();

    assert!(f
        .tokens_seen
        .borrow()
        .iter()
        .all(|t| t.as_deref() == Some(TOKEN)));
    assert!(!String::from_utf8_lossy(&log).contains(TOKEN));

    // A failing run must not echo the token in its error either.
    let empty = Fake::default();
    let err = run(
        &empty,
        &request(&[], dir.path(), false),
        Some(TOKEN),
        &mut log,
    )
    .unwrap_err();
    assert!(!format!("{err:#}").contains(TOKEN));
}

#[test]
fn dry_run_prints_plan_and_downloads_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("models").join("name");
    let sibs = vec![
        sib("config.json", b"{}", false),
        sib("model.safetensors", b"weights", true),
    ];
    let f = fake_with(api_json(Some("mit"), sibs));

    let mut log: Vec<u8> = Vec::new();
    let res = run(&f, &request(&[], &out, true), None, &mut log).unwrap();

    assert!(res.downloaded.is_empty());
    assert!(file_calls(&f).is_empty());
    assert!(!out.exists());
    let text = String::from_utf8_lossy(&log);
    assert!(text.contains("dry run"));
    assert!(text.contains("model.safetensors"));
}

#[test]
fn bad_repo_id_is_rejected_before_any_request() {
    let dir = tempfile::tempdir().unwrap();
    for bad in ["../x", "org/na me", "org", "a/b/c", "", "org/../x", "/x"] {
        let f = fake_with(api_json(Some("mit"), vec![]));
        let mut req = request(&[], dir.path(), false);
        req.repo = bad;
        let mut log: Vec<u8> = Vec::new();
        assert!(run(&f, &req, None, &mut log).is_err(), "accepted {bad:?}");
        assert!(f.calls.borrow().is_empty(), "request made for {bad:?}");
    }
}

#[test]
fn repo_id_parses_only_org_and_name() {
    assert_eq!(RepoId::parse("org/name").unwrap().to_string(), "org/name");
    assert_eq!(
        RepoId::parse("Qwen/Qwen2.5-0.5B_x").unwrap().name(),
        "Qwen2.5-0.5B_x"
    );
    for bad in [
        "../x",
        ".hidden/x",
        "org/na me",
        "org",
        "a/b/c",
        "org/",
        "/x",
    ] {
        assert!(RepoId::parse(bad).is_err(), "accepted {bad:?}");
    }
}

#[test]
fn resolve_url_encodes_revision_and_keeps_file_path() {
    let repo = RepoId::parse("org/name").unwrap();
    assert_eq!(
        resolve_url(&repo, "refs/pr/1", "model.safetensors").unwrap(),
        "https://huggingface.co/org/name/resolve/refs%2Fpr%2F1/model.safetensors"
    );
    assert_eq!(
        resolve_url(&repo, "main", "sub/tok.json").unwrap(),
        "https://huggingface.co/org/name/resolve/main/sub/tok.json"
    );
    assert!(validate_revision("main/../x").is_err());
    assert!(validate_revision("").is_err());
}

#[test]
fn file_names_must_stay_inside_the_output_folder() {
    assert!(safe_relpath("../x").is_err());
    assert!(safe_relpath("/abs").is_err());
    assert!(safe_relpath("a\\b").is_err());
    assert!(safe_relpath("./x").is_err());
    assert!(safe_relpath("").is_err());
    assert_eq!(
        safe_relpath("sub/model.safetensors").unwrap(),
        Path::new("sub").join("model.safetensors")
    );
}

#[test]
fn licence_note_says_when_missing_or_not_open() {
    assert!(licence_note(None).contains("not stated"));
    assert!(licence_note(Some("apache-2.0")).contains("open"));
    assert!(!licence_note(Some("apache-2.0")).contains("not on the open"));
    assert!(licence_note(Some("llama3")).contains("not on the open"));
}

#[test]
fn index_lists_each_shard_once_in_order() {
    let index =
        r#"{"weight_map":{"a":"s2.safetensors","b":"s1.safetensors","c":"s2.safetensors"}}"#;
    assert_eq!(
        shards_from_index(index).unwrap(),
        ["s1.safetensors", "s2.safetensors"]
    );
    assert!(shards_from_index("{}").is_err());
}

#[test]
fn pickle_extensions_are_recognised() {
    for p in ["a.bin", "a.PT", "a.pth", "a.pkl", "a.ckpt"] {
        assert!(is_pickle(p), "{p}");
    }
    for p in ["a.safetensors", "a.gguf", "tokenizer.json"] {
        assert!(!is_pickle(p), "{p}");
    }
}
