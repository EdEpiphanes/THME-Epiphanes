// SPDX-License-Identifier: AGPL-3.0-only
//! `thme download`: fetch a model's files from the Hugging Face hub into a local folder.
//!
//! Network access sits behind [`Fetch`], so file selection, URLs, the licence note, checksums,
//! the atomic rename and the pickle rule are tested without a connection. Nothing here loads
//! a model; the caller runs the scan on the output folder afterwards.

mod http;
#[cfg(test)]
mod tests;

pub use http::HttpFetch;

use anyhow::{bail, Context, Result};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::fmt;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Component, Path, PathBuf};

const HUB: &str = "https://huggingface.co";
const PART_SUFFIX: &str = ".thme-part";
const CONFIG_FILES: [&str; 5] = [
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
];
const SAFETENSORS: &str = "model.safetensors";
const INDEX_FILE: &str = "model.safetensors.index.json";
const PICKLE_EXTS: [&str; 5] = [".bin", ".pt", ".pth", ".pkl", ".ckpt"];
// shortcut: a name-only list of open licences, not a legal review; add names when the owner asks
const OPEN_LICENCES: [&str; 12] = [
    "apache-2.0",
    "mit",
    "bsd-2-clause",
    "bsd-3-clause",
    "isc",
    "mpl-2.0",
    "cc0-1.0",
    "cc-by-4.0",
    "cc-by-sa-4.0",
    "gpl-3.0",
    "lgpl-3.0",
    "agpl-3.0",
];

/// A Hugging Face model id, `ORG/NAME`, with both parts limited to safe characters.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RepoId {
    org: String,
    name: String,
}

impl RepoId {
    pub fn parse(s: &str) -> Result<Self> {
        let mut parts = s.split('/');
        match (parts.next(), parts.next(), parts.next()) {
            (Some(org), Some(name), None) if valid_segment(org) && valid_segment(name) => {
                Ok(Self {
                    org: org.to_string(),
                    name: name.to_string(),
                })
            }
            _ => bail!("repo id must be ORG/NAME (letters, digits, '.', '-', '_'), not {s:?}"),
        }
    }

    /// The model name without the organisation; the default output folder is `models/NAME`.
    pub fn name(&self) -> &str {
        &self.name
    }
}

impl fmt::Display for RepoId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}/{}", self.org, self.name)
    }
}

fn valid_segment(s: &str) -> bool {
    !s.is_empty()
        && s.len() <= 96
        && !s.starts_with('.')
        && s.chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
}

/// A branch, tag or commit. Slashes are allowed (`refs/pr/1`), `..` is not.
pub fn validate_revision(rev: &str) -> Result<&str> {
    let ok = !rev.is_empty()
        && rev.len() <= 200
        && !rev.starts_with('/')
        && !rev.ends_with('/')
        && !rev.contains("..")
        && rev
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.' | '/'));
    if ok {
        Ok(rev)
    } else {
        bail!("revision {rev:?} is not a branch, tag or commit name")
    }
}

/// A file name from the hub or the command line, which must stay inside the output folder.
pub fn safe_relpath(name: &str) -> Result<PathBuf> {
    let plain = !name.is_empty()
        && !name.contains(['\\', '\0'])
        && Path::new(name)
            .components()
            .all(|c| matches!(c, Component::Normal(_)));
    if plain {
        Ok(PathBuf::from(name))
    } else {
        bail!("file name {name:?} is not a plain relative path")
    }
}

/// Percent-encode everything except the RFC 3986 unreserved characters.
fn pct(s: &str) -> String {
    let mut out = String::new();
    for b in s.bytes() {
        if b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.' | b'~') {
            out.push(char::from(b));
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out
}

pub fn api_url(repo: &RepoId, rev: &str) -> String {
    format!("{HUB}/api/models/{repo}/revision/{}?blobs=true", pct(rev))
}

pub fn resolve_url(repo: &RepoId, rev: &str, file: &str) -> Result<String> {
    safe_relpath(file)?;
    let path: Vec<String> = file.split('/').map(pct).collect();
    Ok(format!(
        "{HUB}/{repo}/resolve/{}/{}",
        pct(rev),
        path.join("/")
    ))
}

/// Pickle files (`.bin`, `.pt`, `.pth`, `.pkl`, `.ckpt`) can run code when loaded.
pub fn is_pickle(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    PICKLE_EXTS.iter().any(|ext| lower.ends_with(ext))
}

/// One line for the terminal: the licence name, or a clear note when it is missing or not open.
pub fn licence_note(licence: Option<&str>) -> String {
    match licence {
        None => "not stated in the model card; check the repo before reuse".to_string(),
        Some(l) if OPEN_LICENCES.contains(&l.to_ascii_lowercase().as_str()) => {
            format!("{l} (open)")
        }
        Some(l) => format!("{l} is not on the open-licence list; read its terms before use"),
    }
}

/// Shard names from a `model.safetensors.index.json`, each once, in sorted order.
pub fn shards_from_index(json: &str) -> Result<Vec<String>> {
    let v: Value = serde_json::from_str(json).context("safetensors index is not JSON")?;
    let map = v
        .get("weight_map")
        .and_then(Value::as_object)
        .context("safetensors index has no weight_map")?;
    let shards: BTreeSet<String> = map
        .values()
        .filter_map(Value::as_str)
        .map(str::to_string)
        .collect();
    Ok(shards.into_iter().collect())
}

/// A file as the hub describes it. `sha256` is present for LFS files only.
#[derive(Debug, Clone)]
pub struct Sibling {
    pub name: String,
    pub size: Option<u64>,
    pub sha256: Option<String>,
}

#[derive(Debug)]
pub struct ModelInfo {
    pub siblings: Vec<Sibling>,
    pub licence: Option<String>,
}

impl ModelInfo {
    /// Read the `api/models/.../revision/...` reply (with `blobs=true`).
    pub fn parse(json: &str) -> Result<Self> {
        let v: Value = serde_json::from_str(json).context("hub API reply is not JSON")?;
        let list = v
            .get("siblings")
            .and_then(Value::as_array)
            .context("hub API reply has no siblings list")?;
        let siblings = list
            .iter()
            .filter_map(|s| {
                Some(Sibling {
                    name: s.get("rfilename")?.as_str()?.to_string(),
                    size: s
                        .get("size")
                        .or_else(|| s.pointer("/lfs/size"))
                        .and_then(Value::as_u64),
                    sha256: s
                        .pointer("/lfs/sha256")
                        .and_then(Value::as_str)
                        .map(str::to_ascii_lowercase),
                })
            })
            .collect();
        let licence = v
            .pointer("/cardData/license")
            .and_then(Value::as_str)
            .filter(|l| !l.is_empty())
            .map(str::to_string);
        Ok(Self { siblings, licence })
    }

    fn find(&self, name: &str) -> Option<&Sibling> {
        self.siblings.iter().find(|s| s.name == name)
    }
}

/// Network seam. The real one is [`HttpFetch`]; tests use a fake.
pub trait Fetch {
    /// GET a small text body (the hub API reply).
    fn text(&self, url: &str, token: Option<&str>) -> Result<String>;
    /// GET a file body, streamed into `sink`. Returns the number of bytes received.
    fn file(&self, url: &str, token: Option<&str>, sink: &mut dyn Write) -> Result<u64>;
}

pub struct Request<'a> {
    pub repo: &'a str,
    pub revision: &'a str,
    /// Names given with `--file`. Empty means the default selection.
    pub files: &'a [String],
    pub out: &'a Path,
    pub dry_run: bool,
}

#[derive(Debug, Default)]
pub struct Outcome {
    pub downloaded: Vec<String>,
    pub kept: Vec<String>,
}

/// What the default or named selection asks for. `index` is fetched after `files`, and its
/// shards are added once it is on disk.
struct Selection {
    files: Vec<Sibling>,
    index: Option<Sibling>,
}

fn select(info: &ModelInfo, requested: &[String], log: &mut dyn Write) -> Result<Selection> {
    let mut files = Vec::new();
    if !requested.is_empty() {
        for name in requested {
            safe_relpath(name)?;
            if files.iter().all(|f: &Sibling| f.name != *name) {
                let sib = info
                    .find(name)
                    .cloned()
                    .with_context(|| format!("{name} is not in this repo"))?;
                files.push(sib);
            }
        }
        return Ok(Selection { files, index: None });
    }
    for name in CONFIG_FILES {
        match info.find(name) {
            Some(sib) => files.push(sib.clone()),
            None => writeln!(log, "  {name} is not in this repo")?,
        }
    }
    if let Some(sib) = info.find(SAFETENSORS) {
        files.push(sib.clone());
        return Ok(Selection { files, index: None });
    }
    if let Some(index) = info.find(INDEX_FILE) {
        return Ok(Selection {
            files,
            index: Some(index.clone()),
        });
    }
    let ggufs: Vec<&Sibling> = info
        .siblings
        .iter()
        .filter(|s| s.name.to_ascii_lowercase().ends_with(".gguf"))
        .collect();
    match ggufs.as_slice() {
        [one] => {
            files.push((*one).clone());
            Ok(Selection { files, index: None })
        }
        [] => bail!(
            "no model.safetensors, index or .gguf in this repo; name the weight files with --file (pickle files need --file too)"
        ),
        many => {
            let names: Vec<&str> = many.iter().map(|s| s.name.as_str()).collect();
            bail!(
                "{} GGUF files in this repo; name one with --file: {}",
                many.len(),
                names.join(", ")
            )
        }
    }
}

/// Pickle files are never fetched unless they were named with `--file`; then they are warned.
fn pickle_gate(name: &str, requested: &[String], log: &mut dyn Write) -> Result<()> {
    if !is_pickle(name) {
        return Ok(());
    }
    if requested.iter().any(|r| r == name) {
        writeln!(
            log,
            "warning: {name} is a pickle file. Loading it can execute code on your machine; scan it before use."
        )?;
        return Ok(());
    }
    bail!("{name} is a pickle file, which can execute code; it is not downloaded by default. Name it with --file to allow it")
}

fn size_text(s: &Sibling) -> String {
    s.size
        .map_or_else(|| "size unknown".to_string(), |n| format!("{n} bytes"))
}

/// Download one file into `out`, or keep it when a file with the hub's sha256 is already there.
struct Ctx<'a> {
    fetch: &'a dyn Fetch,
    repo: &'a RepoId,
    rev: &'a str,
    token: Option<&'a str>,
    out: &'a Path,
}

impl Ctx<'_> {
    fn fetch_one(&self, sib: &Sibling, outcome: &mut Outcome, log: &mut dyn Write) -> Result<()> {
        let dest = self.out.join(safe_relpath(&sib.name)?);
        if let Some(want) = &sib.sha256 {
            if dest.is_file() && sha256_file(&dest)? == *want {
                writeln!(log, "keep {} (sha256 matches)", sib.name)?;
                outcome.kept.push(sib.name.clone());
                return Ok(());
            }
        }
        let url = resolve_url(self.repo, self.rev, &sib.name)?;
        writeln!(log, "fetch {}", sib.name)?;
        if let Some(parent) = dest.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("create folder for {}", sib.name))?;
        }
        let part = part_path(&dest);
        // Any failure removes the temporary file, so no final file appears for a bad download.
        let result = download_to(self.fetch, &url, self.token, sib, &part)
            .and_then(|()| std::fs::rename(&part, &dest).map_err(anyhow::Error::from));
        if let Err(e) = result {
            let _ = std::fs::remove_file(&part);
            return Err(e.context(format!("download {}", sib.name)));
        }
        outcome.downloaded.push(sib.name.clone());
        Ok(())
    }
}

fn part_path(dest: &Path) -> PathBuf {
    let name = dest
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("download");
    dest.with_file_name(format!("{name}{PART_SUFFIX}"))
}

fn download_to(
    fetch: &dyn Fetch,
    url: &str,
    token: Option<&str>,
    sib: &Sibling,
    part: &Path,
) -> Result<()> {
    let file = File::create(part).context("create temporary file")?;
    let mut sink = HashingWriter::new(BufWriter::new(file));
    let received = fetch.file(url, token, &mut sink)?;
    sink.flush()?;
    let (_, got) = sink.finish();
    if let Some(size) = sib.size {
        if received != size {
            bail!(
                "size mismatch for {}: hub says {size} bytes, got {received}",
                sib.name
            );
        }
    }
    if let Some(want) = &sib.sha256 {
        if got != *want {
            bail!(
                "checksum mismatch for {}: hub sha256 {want}, downloaded {got}",
                sib.name
            );
        }
    }
    Ok(())
}

struct HashingWriter<W: Write> {
    inner: W,
    hasher: Sha256,
}

impl<W: Write> HashingWriter<W> {
    fn new(inner: W) -> Self {
        Self {
            inner,
            hasher: Sha256::new(),
        }
    }

    fn finish(self) -> (W, String) {
        (self.inner, hex(self.hasher.finalize().as_slice()))
    }
}

impl<W: Write> Write for HashingWriter<W> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let n = self.inner.write(buf)?;
        self.hasher.update(&buf[..n]);
        Ok(n)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
    }
}

fn sha256_file(path: &Path) -> Result<String> {
    let mut file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut hasher = Sha256::new();
    std::io::copy(&mut file, &mut hasher)?;
    Ok(hex(hasher.finalize().as_slice()))
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// Run a download. Progress and the plan go to `log`; the token is sent but never written.
pub fn run(
    fetch: &dyn Fetch,
    req: &Request,
    token: Option<&str>,
    log: &mut dyn Write,
) -> Result<Outcome> {
    let repo = RepoId::parse(req.repo)?;
    let rev = validate_revision(req.revision)?;
    let info = ModelInfo::parse(&fetch.text(&api_url(&repo, rev), token)?)?;
    writeln!(log, "model {repo} at {rev}")?;
    writeln!(log, "licence: {}", licence_note(info.licence.as_deref()))?;

    let sel = select(&info, req.files, log)?;
    for f in &sel.files {
        pickle_gate(&f.name, req.files, log)?;
    }
    writeln!(log, "plan:")?;
    for f in &sel.files {
        writeln!(log, "  {} ({})", f.name, size_text(f))?;
    }
    if let Some(index) = &sel.index {
        writeln!(
            log,
            "  {} ({}), then the shards it lists",
            index.name,
            size_text(index)
        )?;
    }
    if req.dry_run {
        writeln!(log, "dry run: nothing downloaded")?;
        return Ok(Outcome::default());
    }

    let ctx = Ctx {
        fetch,
        repo: &repo,
        rev,
        token,
        out: req.out,
    };
    let mut outcome = Outcome::default();
    for f in &sel.files {
        ctx.fetch_one(f, &mut outcome, log)?;
    }
    if let Some(index) = &sel.index {
        ctx.fetch_one(index, &mut outcome, log)?;
        let path = req.out.join(safe_relpath(&index.name)?);
        let text = std::fs::read_to_string(&path).context("read safetensors index")?;
        for shard in shards_from_index(&text)? {
            let sib = info
                .find(&shard)
                .cloned()
                .with_context(|| format!("index names {shard}, which is not in this repo"))?;
            pickle_gate(&sib.name, req.files, log)?;
            ctx.fetch_one(&sib, &mut outcome, log)?;
        }
    }
    writeln!(
        log,
        "done: {} downloaded, {} kept in {}",
        outcome.downloaded.len(),
        outcome.kept.len(),
        req.out.display()
    )?;
    Ok(outcome)
}
