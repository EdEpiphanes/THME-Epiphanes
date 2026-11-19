// SPDX-License-Identifier: AGPL-3.0-only
//! `thme discover`: list the models already on this machine, from the tools users came from.
//! Read-only: no network, nothing downloaded, nothing written. Only the first few bytes of a file
//! are read, to tell the formats apart.

use serde_json::{json, Value};
use std::collections::BTreeSet;
use std::io::Read;
use std::path::{Path, PathBuf};

/// Folders deeper than this under a root are not searched; guards against symlink loops.
const MAX_DEPTH: usize = 8;
/// Largest Ollama manifest read; a bigger file is reported as broken, not allocated.
const MAX_MANIFEST_BYTES: u64 = 1 << 20;
const OLLAMA_MODEL_MEDIA_TYPE: &str = "application/vnd.ollama.image.model";

/// The tool a model was found under.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Source {
    HuggingFace,
    Ollama,
    LmStudio,
    LlamaCpp,
    /// A folder given with `--path`.
    Folder,
}

impl Source {
    pub fn label(self) -> &'static str {
        match self {
            Source::HuggingFace => "Hugging Face",
            Source::Ollama => "Ollama",
            Source::LmStudio => "LM Studio",
            Source::LlamaCpp => "llama.cpp",
            Source::Folder => "folder",
        }
    }
}

/// The file formats `discover` can tell apart by their first bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    Gguf,
    SafeTensors,
    Epiph,
    /// Recognised by extension only: loading it runs code, so it is never opened here.
    Pickle,
}

impl Format {
    pub fn label(self) -> &'static str {
        match self {
            Format::Gguf => "GGUF",
            Format::SafeTensors => "SafeTensors",
            Format::Epiph => "epiph",
            Format::Pickle => "pickle (scan before loading)",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiscoveredModel {
    pub name: String,
    pub source: Source,
    pub format: Format,
    pub size_bytes: u64,
    pub path: PathBuf,
}

#[derive(Debug, Default)]
pub struct Discovery {
    pub models: Vec<DiscoveredModel>,
    /// Problems that did not stop the search: unreadable folders, broken manifests.
    pub warnings: Vec<String>,
}

/// The environment variables and home folder `discover` reads, captured as data so the search
/// itself never touches the process environment.
#[derive(Debug, Default, Clone)]
pub struct EnvSnapshot {
    pub home: Option<PathBuf>,
    pub hf_hub_cache: Option<PathBuf>,
    pub hf_home: Option<PathBuf>,
    pub ollama_models: Option<PathBuf>,
    pub llama_cache: Option<PathBuf>,
}

impl EnvSnapshot {
    /// Reads the real environment. An empty variable counts as unset.
    pub fn from_process() -> Self {
        let var = |name: &str| {
            std::env::var_os(name)
                .filter(|v| !v.is_empty())
                .map(PathBuf::from)
        };
        Self {
            home: var("USERPROFILE").or_else(|| var("HOME")),
            hf_hub_cache: var("HF_HUB_CACHE"),
            hf_home: var("HF_HOME"),
            ollama_models: var("OLLAMA_MODELS"),
            llama_cache: var("LLAMA_CACHE"),
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum Layout {
    /// `models--ORG--NAME/snapshots/<rev>/...`
    HubCache,
    /// `manifests/<registry>/<ns>/<model>/<tag>` plus `blobs/sha256-<hex>`
    Ollama,
    /// Any folder tree; names are relative paths.
    Folder,
}

struct Root {
    source: Source,
    layout: Layout,
    path: PathBuf,
    /// Set by an env variable or `--path`. A missing explicit root is a warning; a missing
    /// default location is simply not there.
    explicit: bool,
}

/// Searches the known locations, then `paths`, and lists every recognised model file. Read-only.
pub fn discover(paths: &[PathBuf], env: &EnvSnapshot) -> Discovery {
    let mut out = Discovery::default();
    for root in roots(paths, env) {
        if !root.path.is_dir() {
            if root.explicit {
                out.warnings
                    .push(format!("{} is not a folder, skipped", root.path.display()));
            }
            continue;
        }
        match root.layout {
            Layout::HubCache => hub_models(&root, &mut out),
            Layout::Ollama => ollama_models(&root, &mut out),
            Layout::Folder => folder_models(&root, &mut out),
        }
    }
    // Two roots can reach the same file (a --path inside the hub cache); the first root wins.
    let mut seen = BTreeSet::new();
    out.models
        .retain(|m| seen.insert(std::fs::canonicalize(&m.path).unwrap_or_else(|_| m.path.clone())));
    out.models.sort_by(|a, b| {
        a.source
            .label()
            .cmp(b.source.label())
            .then_with(|| a.name.cmp(&b.name))
            .then_with(|| a.path.cmp(&b.path))
    });
    out
}

fn roots(paths: &[PathBuf], env: &EnvSnapshot) -> Vec<Root> {
    let mut roots = Vec::new();
    let hub = if let Some(p) = &env.hf_hub_cache {
        Some((p.clone(), true))
    } else if let Some(p) = &env.hf_home {
        Some((p.join("hub"), true))
    } else {
        env.home
            .as_ref()
            .map(|h| (h.join(".cache/huggingface/hub"), false))
    };
    roots.extend(hub.map(|(path, explicit)| Root {
        source: Source::HuggingFace,
        layout: Layout::HubCache,
        path,
        explicit,
    }));
    let ollama = match &env.ollama_models {
        Some(p) => Some((p.clone(), true)),
        None => env.home.as_ref().map(|h| (h.join(".ollama/models"), false)),
    };
    roots.extend(ollama.map(|(path, explicit)| Root {
        source: Source::Ollama,
        layout: Layout::Ollama,
        path,
        explicit,
    }));
    if let Some(home) = &env.home {
        for rel in [".lmstudio/models", ".cache/lm-studio/models"] {
            roots.push(Root {
                source: Source::LmStudio,
                layout: Layout::Folder,
                path: home.join(rel),
                explicit: false,
            });
        }
    }
    roots.extend(env.llama_cache.clone().map(|path| Root {
        source: Source::LlamaCpp,
        layout: Layout::Folder,
        path,
        explicit: true,
    }));
    roots.extend(paths.iter().map(|path| Root {
        source: Source::Folder,
        layout: Layout::Folder,
        path: path.clone(),
        explicit: true,
    }));
    roots
}

/// Child paths of `dir`; a folder or entry that cannot be read becomes a warning.
fn children(dir: &Path, warnings: &mut Vec<String>) -> Vec<PathBuf> {
    match std::fs::read_dir(dir) {
        Ok(entries) => entries
            .filter_map(|entry| match entry {
                Ok(entry) => Some(entry.path()),
                Err(err) => {
                    warnings.push(format!("cannot read an entry in {}: {err}", dir.display()));
                    None
                }
            })
            .collect(),
        Err(err) => {
            warnings.push(format!("cannot read {}: {err}", dir.display()));
            Vec::new()
        }
    }
}

fn list_files(dir: &Path, depth: usize, files: &mut Vec<PathBuf>, warnings: &mut Vec<String>) {
    for path in children(dir, warnings) {
        if path.is_dir() {
            if depth < MAX_DEPTH {
                list_files(&path, depth + 1, files, warnings);
            }
        } else if path.is_file() {
            files.push(path);
        }
    }
}

/// Format and size of one file, or `None` when it is not a recognised model.
fn classify(path: &Path, warnings: &mut Vec<String>) -> Option<(Format, u64)> {
    let len = match std::fs::metadata(path) {
        Ok(meta) => meta.len(),
        Err(err) => {
            warnings.push(format!("cannot read {}: {err}", path.display()));
            return None;
        }
    };
    let head = crate::scan::head(path, 9);
    let format = if head.starts_with(&crate::epiph::MAGIC) {
        Format::Epiph
    } else if head.starts_with(b"GGUF") {
        Format::Gguf
    } else if is_safetensors(&head, len) {
        Format::SafeTensors
    } else if matches!(crate::scan::extension(path).as_str(), "bin" | "pt" | "pth") {
        Format::Pickle
    } else {
        return None;
    };
    Some((format, len))
}

/// SafeTensors: an 8-byte little-endian header length, then a JSON object that fits the file.
fn is_safetensors(head: &[u8], len: u64) -> bool {
    let Some(prefix) = head.get(..8).and_then(|b| <[u8; 8]>::try_from(b).ok()) else {
        return false;
    };
    let header_len = u64::from_le_bytes(prefix);
    head.get(8) == Some(&b'{') && header_len > 0 && header_len <= len.saturating_sub(8)
}

fn relative_name(root: &Path, file: &Path) -> String {
    file.strip_prefix(root)
        .unwrap_or(file)
        .components()
        .map(|c| c.as_os_str().to_string_lossy().into_owned())
        .collect::<Vec<_>>()
        .join("/")
}

fn folder_models(root: &Root, out: &mut Discovery) {
    let mut files = Vec::new();
    list_files(&root.path, 0, &mut files, &mut out.warnings);
    for file in files {
        if let Some((format, size_bytes)) = classify(&file, &mut out.warnings) {
            out.models.push(DiscoveredModel {
                name: relative_name(&root.path, &file),
                source: root.source,
                format,
                size_bytes,
                path: file,
            });
        }
    }
}

fn hub_models(root: &Root, out: &mut Discovery) {
    for dir in children(&root.path, &mut out.warnings) {
        let Some(repo) = dir
            .file_name()
            .and_then(|n| n.to_str())
            .and_then(|n| n.strip_prefix("models--"))
            .map(|r| r.replacen("--", "/", 1))
        else {
            continue;
        };
        let snapshots = dir.join("snapshots");
        if !snapshots.is_dir() {
            continue;
        }
        let mut files = Vec::new();
        list_files(&snapshots, 0, &mut files, &mut out.warnings);
        for file in files {
            if let Some((format, size_bytes)) = classify(&file, &mut out.warnings) {
                out.models.push(DiscoveredModel {
                    name: repo.clone(),
                    source: Source::HuggingFace,
                    format,
                    size_bytes,
                    path: file,
                });
            }
        }
    }
}

/// `manifests/<host>/<ns...>/<model>/<tag>` becomes `<ns...>/<model>:<tag>`; the host is dropped.
fn ollama_name(file: &Path, manifests: &Path) -> Option<String> {
    let parts: Vec<String> = file
        .strip_prefix(manifests)
        .ok()?
        .components()
        .map(|c| c.as_os_str().to_string_lossy().into_owned())
        .collect();
    let (tag, rest) = parts.split_last()?;
    if rest.len() < 2 {
        return None;
    }
    Some(format!("{}:{tag}", rest[1..].join("/")))
}

/// The sha256 hex of the manifest's model layer, or why the manifest cannot be used.
fn model_digest(file: &Path) -> Result<String, String> {
    let mut text = String::new();
    std::fs::File::open(file)
        .map_err(|e| e.to_string())?
        .take(MAX_MANIFEST_BYTES)
        .read_to_string(&mut text)
        .map_err(|e| e.to_string())?;
    let json: Value = serde_json::from_str(&text).map_err(|e| format!("not JSON: {e}"))?;
    let layers = json
        .get("layers")
        .and_then(Value::as_array)
        .ok_or("no layers array")?;
    let layer = layers
        .iter()
        .find(|l| l.get("mediaType").and_then(Value::as_str) == Some(OLLAMA_MODEL_MEDIA_TYPE))
        .ok_or("no layer with the Ollama model media type")?;
    let digest = layer
        .get("digest")
        .and_then(Value::as_str)
        .ok_or("model layer has no digest")?;
    digest
        .strip_prefix("sha256:")
        .filter(|hex| !hex.is_empty() && hex.bytes().all(|b| b.is_ascii_hexdigit()))
        .map(str::to_string)
        .ok_or_else(|| format!("digest {digest:?} is not sha256:HEX"))
}

fn ollama_models(root: &Root, out: &mut Discovery) {
    let manifests = root.path.join("manifests");
    if !manifests.is_dir() {
        return;
    }
    let mut files = Vec::new();
    list_files(&manifests, 0, &mut files, &mut out.warnings);
    for file in files {
        let Some(name) = ollama_name(&file, &manifests) else {
            out.warnings.push(format!(
                "{}: not a registry/model/tag manifest path, skipped",
                file.display()
            ));
            continue;
        };
        let digest = match model_digest(&file) {
            Ok(digest) => digest,
            Err(err) => {
                out.warnings.push(format!(
                    "{}: broken Ollama manifest ({err})",
                    file.display()
                ));
                continue;
            }
        };
        let blob = root.path.join("blobs").join(format!("sha256-{digest}"));
        if !blob.is_file() {
            out.warnings.push(format!(
                "{}: model blob {} is missing",
                file.display(),
                blob.display()
            ));
            continue;
        }
        if let Some((format, size_bytes)) = classify(&blob, &mut out.warnings) {
            out.models.push(DiscoveredModel {
                name,
                source: Source::Ollama,
                format,
                size_bytes,
                path: blob,
            });
        }
    }
}

/// One JSON object per model plus the warnings, in the same order as the table.
pub fn render_json(found: &Discovery) -> Value {
    let models: Vec<Value> = found
        .models
        .iter()
        .map(|m| {
            json!({
                "name": m.name,
                "source": m.source.label(),
                "format": m.format.label(),
                "size_bytes": m.size_bytes,
                "path": m.path.display().to_string(),
            })
        })
        .collect();
    json!({ "models": models, "warnings": found.warnings })
}

/// Aligned plain-text table; the path is last so it never needs padding.
pub fn render_table(models: &[DiscoveredModel]) -> String {
    if models.is_empty() {
        return "No models found.".to_string();
    }
    let header = ["SOURCE", "NAME", "FORMAT", "SIZE", "PATH"].map(str::to_string);
    let rows: Vec<[String; 5]> = models
        .iter()
        .map(|m| {
            [
                m.source.label().to_string(),
                m.name.clone(),
                m.format.label().to_string(),
                m.size_bytes.to_string(),
                m.path.display().to_string(),
            ]
        })
        .collect();
    let mut widths = [0usize; 4];
    for row in std::iter::once(&header).chain(rows.iter()) {
        for (w, cell) in widths.iter_mut().zip(row.iter()) {
            *w = (*w).max(cell.chars().count());
        }
    }
    let [w0, w1, w2, w3] = widths;
    let line = |row: &[String; 5]| {
        format!(
            "{:<w0$}  {:<w1$}  {:<w2$}  {:>w3$}  {}",
            row[0], row[1], row[2], row[3], row[4]
        )
    };
    let mut lines = vec![line(&header)];
    lines.extend(rows.iter().map(line));
    lines.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::Path;

    const HEX: &str = "ab";

    fn hex64(byte: &str) -> String {
        byte.repeat(32)
    }

    fn gguf() -> Vec<u8> {
        let mut v = b"GGUF".to_vec();
        v.extend_from_slice(&3u32.to_le_bytes());
        v.extend_from_slice(&[0u8; 16]);
        v
    }

    fn safetensors() -> Vec<u8> {
        let header = br#"{"__metadata__":{}}"#;
        let mut v = (header.len() as u64).to_le_bytes().to_vec();
        v.extend_from_slice(header);
        v
    }

    fn pickle() -> Vec<u8> {
        b"\x80\x02}q\x00.".to_vec()
    }

    fn epiph() -> Vec<u8> {
        let mut v = crate::epiph::MAGIC.to_vec();
        v.extend_from_slice(&[0u8; 24]);
        v
    }

    fn write(path: &Path, bytes: &[u8]) {
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, bytes).unwrap();
    }

    fn home_only(home: &Path) -> EnvSnapshot {
        EnvSnapshot {
            home: Some(home.to_path_buf()),
            ..Default::default()
        }
    }

    /// Writes `<root>/manifests/<rel>` pointing its model layer at `blobs/sha256-<HEX>`.
    fn ollama_manifest(root: &Path, rel: &str, blob: &[u8]) {
        let manifest = serde_json::json!({
            "schemaVersion": 2,
            "layers": [
                {"mediaType": "application/vnd.ollama.image.template", "digest": format!("sha256:{}", hex64("cd")), "size": 3},
                {"mediaType": "application/vnd.ollama.image.model", "digest": format!("sha256:{}", hex64(HEX)), "size": blob.len()}
            ]
        });
        write(
            &root.join("manifests").join(rel),
            manifest.to_string().as_bytes(),
        );
        write(
            &root.join("blobs").join(format!("sha256-{}", hex64(HEX))),
            blob,
        );
    }

    fn names(found: &Discovery) -> Vec<String> {
        found.models.iter().map(|m| m.name.clone()).collect()
    }

    #[test]
    fn finds_each_source_and_format() {
        let tmp = tempfile::tempdir().unwrap();
        let home = tmp.path().join("home");
        let hf = home.join(".cache/huggingface/hub");
        write(
            &hf.join("models--org--name/snapshots/rev1/model.safetensors"),
            &safetensors(),
        );
        write(
            &hf.join("models--org--name/snapshots/rev1/config.json"),
            b"{}",
        );
        ollama_manifest(
            &home.join(".ollama/models"),
            "registry.ollama.ai/library/tiny/latest",
            &gguf(),
        );
        write(&home.join(".lmstudio/models/pub/tiny/tiny.gguf"), &gguf());
        write(&home.join(".cache/lm-studio/models/pub2/m/w.pt"), &pickle());
        let llama = tmp.path().join("llama");
        write(&llama.join("q.gguf"), &gguf());
        let plain = tmp.path().join("plain");
        write(&plain.join("sub/x.epiph"), &epiph());

        let env = EnvSnapshot {
            llama_cache: Some(llama),
            ..home_only(&home)
        };
        let found = discover(&[plain], &env);

        assert!(found.warnings.is_empty(), "{:?}", found.warnings);
        assert_eq!(found.models.len(), 6, "{:?}", found.models);
        let has = |source: Source, format: Format| {
            found
                .models
                .iter()
                .any(|m| m.source == source && m.format == format)
        };
        assert!(has(Source::HuggingFace, Format::SafeTensors));
        assert!(has(Source::Ollama, Format::Gguf));
        assert!(has(Source::LmStudio, Format::Gguf));
        assert!(has(Source::LmStudio, Format::Pickle));
        assert!(has(Source::LlamaCpp, Format::Gguf));
        assert!(has(Source::Folder, Format::Epiph));
        let hf_model = found
            .models
            .iter()
            .find(|m| m.source == Source::HuggingFace)
            .unwrap();
        assert_eq!(hf_model.name, "org/name");
        assert_eq!(hf_model.size_bytes, safetensors().len() as u64);
    }

    #[test]
    fn ollama_digest_resolves_to_model_blob() {
        let tmp = tempfile::tempdir().unwrap();
        let models = tmp.path().join("ollama");
        let model = gguf();
        ollama_manifest(&models, "registry.ollama.ai/library/tiny/latest", &model);
        // The template layer's blob is a decoy that must not be picked.
        write(
            &models.join("blobs").join(format!("sha256-{}", hex64("cd"))),
            b"not a model",
        );
        let env = EnvSnapshot {
            ollama_models: Some(models.clone()),
            ..Default::default()
        };

        let found = discover(&[], &env);

        assert!(found.warnings.is_empty(), "{:?}", found.warnings);
        assert_eq!(found.models.len(), 1, "{:?}", found.models);
        let m = &found.models[0];
        assert_eq!(
            m.path,
            models.join("blobs").join(format!("sha256-{}", hex64(HEX)))
        );
        assert_eq!(m.name, "library/tiny:latest");
        assert_eq!(m.format, Format::Gguf);
        assert_eq!(m.size_bytes, model.len() as u64);
    }

    #[test]
    fn broken_ollama_manifests_warn_and_other_models_still_found() {
        let tmp = tempfile::tempdir().unwrap();
        let models = tmp.path().join("ollama");
        ollama_manifest(&models, "registry.ollama.ai/library/good/latest", &gguf());
        write(
            &models.join("manifests/registry.ollama.ai/library/bad/latest"),
            b"{not json",
        );
        write(
            &models.join("manifests/registry.ollama.ai/library/nolayer/latest"),
            br#"{"layers":[{"mediaType":"application/vnd.ollama.image.template","digest":"sha256:ff"}]}"#,
        );
        write(
            &models.join("manifests/registry.ollama.ai/library/gone/latest"),
            format!(
                r#"{{"layers":[{{"mediaType":"application/vnd.ollama.image.model","digest":"sha256:{}"}}]}}"#,
                hex64("ef")
            )
            .as_bytes(),
        );
        let env = EnvSnapshot {
            ollama_models: Some(models),
            ..Default::default()
        };

        let found = discover(&[], &env);

        assert_eq!(names(&found), vec!["library/good:latest".to_string()]);
        assert_eq!(found.warnings.len(), 3, "{:?}", found.warnings);
        for tag in ["bad", "nolayer", "gone"] {
            assert!(
                found.warnings.iter().any(|w| w.contains(tag)),
                "no warning for {tag}: {:?}",
                found.warnings
            );
        }
    }

    #[test]
    fn unknown_files_are_skipped_without_warning() {
        let tmp = tempfile::tempdir().unwrap();
        let plain = tmp.path().join("plain");
        write(&plain.join("notes.txt"), b"hello");
        write(&plain.join("data.dat"), &[1, 2, 3, 4, 5, 6, 7, 8, 9]);
        // A safetensors-looking file whose header length points past the end of the file.
        write(&plain.join("bad.safetensors"), &[0xff; 16]);
        // A .gguf name without the GGUF magic.
        write(&plain.join("fake.gguf"), b"NOPE0000000000000000");

        let found = discover(&[plain], &EnvSnapshot::default());

        assert!(found.models.is_empty(), "{:?}", found.models);
        assert!(found.warnings.is_empty(), "{:?}", found.warnings);
    }

    #[test]
    fn output_is_sorted_by_source_then_name_whatever_the_root_order() {
        let tmp = tempfile::tempdir().unwrap();
        let home = tmp.path().join("home");
        write(
            &home.join(".cache/huggingface/hub/models--org--zeta/snapshots/r/m.safetensors"),
            &safetensors(),
        );
        write(&home.join(".lmstudio/models/pub/alpha/a.gguf"), &gguf());
        let plain_one = tmp.path().join("one");
        let plain_two = tmp.path().join("two");
        write(&plain_one.join("b.gguf"), &gguf());
        write(&plain_two.join("a.gguf"), &gguf());
        let env = home_only(&home);

        let forward = discover(&[plain_one.clone(), plain_two.clone()], &env);
        let backward = discover(&[plain_two, plain_one], &env);

        assert_eq!(forward.models, backward.models);
        let keys: Vec<(String, String)> = forward
            .models
            .iter()
            .map(|m| (m.source.label().to_string(), m.name.clone()))
            .collect();
        let mut sorted = keys.clone();
        sorted.sort();
        assert_eq!(keys, sorted);
        assert_eq!(forward.models.len(), 4);
    }

    #[test]
    fn json_output_parses() {
        let tmp = tempfile::tempdir().unwrap();
        let models = tmp.path().join("ollama");
        let model = gguf();
        ollama_manifest(&models, "registry.ollama.ai/library/tiny/latest", &model);
        write(
            &models.join("manifests/registry.ollama.ai/library/bad/latest"),
            b"{",
        );
        let env = EnvSnapshot {
            ollama_models: Some(models),
            ..Default::default()
        };
        let found = discover(&[], &env);

        let text = serde_json::to_string_pretty(&render_json(&found)).unwrap();
        let value: serde_json::Value = serde_json::from_str(&text).unwrap();

        let list = value["models"].as_array().unwrap();
        assert_eq!(list.len(), 1);
        assert_eq!(list[0]["name"], "library/tiny:latest");
        assert_eq!(list[0]["source"], "Ollama");
        assert_eq!(list[0]["format"], "GGUF");
        assert_eq!(list[0]["size_bytes"], model.len() as u64);
        assert!(list[0]["path"]
            .as_str()
            .unwrap()
            .ends_with(&format!("sha256-{}", hex64(HEX))));
        let warnings = value["warnings"].as_array().unwrap();
        assert_eq!(warnings.len(), 1);
        assert!(warnings[0].as_str().unwrap().contains("bad"));
    }

    #[test]
    fn table_lists_each_model() {
        let tmp = tempfile::tempdir().unwrap();
        let plain = tmp.path().join("plain");
        write(&plain.join("tiny.gguf"), &gguf());
        let found = discover(&[plain], &EnvSnapshot::default());

        let table = render_table(&found.models);

        assert!(table.contains("SOURCE"));
        assert!(table.contains("tiny.gguf"));
        assert!(table.contains("GGUF"));
    }

    #[test]
    fn no_home_skips_default_locations_cleanly() {
        let tmp = tempfile::tempdir().unwrap();
        let home = tmp.path().join("home");
        ollama_manifest(
            &home.join(".ollama/models"),
            "registry.ollama.ai/library/tiny/latest",
            &gguf(),
        );
        write(&home.join(".lmstudio/models/pub/tiny/tiny.gguf"), &gguf());

        let found = discover(&[], &EnvSnapshot::default());

        assert!(found.models.is_empty(), "{:?}", found.models);
        assert!(found.warnings.is_empty(), "{:?}", found.warnings);
    }

    #[test]
    fn ollama_models_env_works_without_home() {
        let tmp = tempfile::tempdir().unwrap();
        let models = tmp.path().join("ollama");
        ollama_manifest(&models, "registry.ollama.ai/library/tiny/latest", &gguf());
        let env = EnvSnapshot {
            ollama_models: Some(models),
            ..Default::default()
        };

        let found = discover(&[], &env);

        assert_eq!(names(&found), vec!["library/tiny:latest".to_string()]);
    }

    #[test]
    fn hf_hub_cache_env_overrides_home_and_hf_home_is_used_next() {
        let tmp = tempfile::tempdir().unwrap();
        let home = tmp.path().join("home");
        write(
            &home.join(".cache/huggingface/hub/models--a--home/snapshots/r/m.gguf"),
            &gguf(),
        );
        let custom = tmp.path().join("custom-hub");
        write(
            &custom.join("models--b--custom/snapshots/r/m.gguf"),
            &gguf(),
        );
        let hf_home = tmp.path().join("hf-home");
        write(
            &hf_home.join("hub/models--c--viahome/snapshots/r/m.gguf"),
            &gguf(),
        );

        let overridden = EnvSnapshot {
            hf_hub_cache: Some(custom),
            hf_home: Some(hf_home.clone()),
            ..home_only(&home)
        };
        assert_eq!(
            names(&discover(&[], &overridden)),
            vec!["b/custom".to_string()]
        );

        let via_home = EnvSnapshot {
            hf_home: Some(hf_home),
            ..home_only(&home)
        };
        assert_eq!(
            names(&discover(&[], &via_home)),
            vec!["c/viahome".to_string()]
        );
    }

    #[test]
    fn explicit_path_to_hub_cache_does_not_double_count() {
        let tmp = tempfile::tempdir().unwrap();
        let home = tmp.path().join("home");
        let hub = home.join(".cache/huggingface/hub");
        write(&hub.join("models--org--name/snapshots/r/m.gguf"), &gguf());

        let found = discover(&[hub], &home_only(&home));

        assert_eq!(found.models.len(), 1, "{:?}", found.models);
        assert_eq!(found.models[0].source, Source::HuggingFace);
    }

    #[test]
    fn missing_explicit_path_warns_and_does_not_panic() {
        let tmp = tempfile::tempdir().unwrap();
        let missing = tmp.path().join("nope");

        let found = discover(&[missing], &EnvSnapshot::default());

        assert!(found.models.is_empty());
        assert_eq!(found.warnings.len(), 1, "{:?}", found.warnings);
        assert!(found.warnings[0].contains("nope"));
    }

    #[test]
    fn format_labels_match_the_spec() {
        assert_eq!(Format::Gguf.label(), "GGUF");
        assert_eq!(Format::SafeTensors.label(), "SafeTensors");
        assert_eq!(Format::Epiph.label(), "epiph");
        assert_eq!(Format::Pickle.label(), "pickle (scan before loading)");
    }
}
