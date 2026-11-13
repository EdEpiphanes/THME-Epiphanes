// SPDX-License-Identifier: AGPL-3.0-only
//! Benchmark registry (roadmap v0.2.8.2, dossier WP31).
//!
//! A benchmark is a declarative **task file**: dataset path and hash, split, licence, prompt
//! template, answer-extraction rule and metric. A run produces a **result record** with every
//! field of dossier Table R2, including per-item results, so intervals and paired comparisons
//! can be computed later without re-running. Both are strict JSON: unknown fields, missing
//! fields and inconsistent values are refused, never guessed.
//!
//! The registry only describes and checks. Running tasks is the benchmark runner (v0.2.8.3);
//! `code` tasks need the sandbox (docs/sandbox.md), which is reserved for the commercial version.

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Component, Path, PathBuf};

use crate::exporter::compute_sha256;

pub mod metrics;

/// `schema` value of a task file.
pub const TASK_SCHEMA: &str = "thme-bench-task/v1";
/// `schema` value of a result record.
pub const RECORD_SCHEMA: &str = "thme-bench-record/v1";

/// What kind of benchmark a task is (dossier Table R1).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Family {
    /// Pick one of several answers (MMLU, ARC style).
    MultipleChoice,
    /// A generated answer that can be checked (GSM8K style).
    CheckableAnswer,
    /// Generated code scored by unit tests; needs the sandbox (commercial version).
    Code,
}

/// Where a task's items live. `path` is relative to the task file and stays inside its folder.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Dataset {
    pub path: String,
    pub sha256: String,
    pub split: String,
    pub licence: String,
}

/// How the answer is taken out of a model's output.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case", deny_unknown_fields)]
pub enum Extract {
    /// The whole output, trimmed.
    Exact,
    /// The first non-empty line, trimmed.
    FirstLine,
    /// The last number in the output.
    LastNumber,
    /// The first standalone choice letter (A, B, ...).
    ChoiceLetter,
    /// The text after the last occurrence of `marker` (for example `####`).
    AfterMarker { marker: String },
    /// Nothing is extracted: choices are scored by likelihood, or code by its tests.
    None,
}

/// How one item is scored. Every metric gives each item a score in [0, 1].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case", deny_unknown_fields)]
pub enum Metric {
    ExactMatch,
    /// Case, surrounding whitespace and punctuation ignored.
    NormalizedMatch,
    /// Correct when within `abs` or within `rel` of the target (either suffices).
    NumericTolerance {
        #[serde(default)]
        abs: Option<f64>,
        #[serde(default)]
        rel: Option<f64>,
    },
    /// The choice with the highest log-likelihood is the answer.
    ChoiceByLikelihood,
    /// Unbiased pass@k from n samples per item.
    PassAtK {
        k: u32,
        samples: u32,
    },
}

/// A declarative benchmark task.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TaskSpec {
    pub schema: String,
    pub id: String,
    pub version: u32,
    pub family: Family,
    pub dataset: Dataset,
    /// Must contain `{input}`; multiple-choice templates may also use `{choices}`.
    pub prompt_template: String,
    pub extract: Extract,
    pub metric: Metric,
}

impl TaskSpec {
    /// Reads a task file and checks it.
    pub fn load(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("Failed to read task {path:?}"))?;
        let task: Self = serde_json::from_str(&text)
            .with_context(|| format!("{path:?} is not a valid benchmark task"))?;
        task.validate()
            .with_context(|| format!("{path:?} is not a valid benchmark task"))?;
        Ok(task)
    }

    /// Refuses tasks that are incomplete, unportable or that combine rules that cannot work.
    pub fn validate(&self) -> Result<()> {
        if self.schema != TASK_SCHEMA {
            bail!("schema is '{}', expected '{TASK_SCHEMA}'", self.schema);
        }
        check_slug("id", &self.id)?;
        if self.version == 0 {
            bail!("version starts at 1");
        }
        check_sha256("dataset.sha256", &self.dataset.sha256)?;
        check_relative_path(&self.dataset.path)?;
        for (field, value) in [
            ("dataset.split", &self.dataset.split),
            ("dataset.licence", &self.dataset.licence),
        ] {
            if value.trim().is_empty() {
                bail!("{field} is empty");
            }
        }
        if !self.prompt_template.contains("{input}") {
            bail!("prompt_template has no {{input}} placeholder");
        }
        if self.prompt_template.contains("{choices}") && self.family != Family::MultipleChoice {
            bail!("{{choices}} is only for multiple-choice tasks");
        }
        if let Extract::AfterMarker { marker } = &self.extract {
            if marker.is_empty() {
                bail!("extract after-marker needs a non-empty marker");
            }
        }
        match &self.metric {
            Metric::NumericTolerance { abs, rel } => {
                if abs.is_none() && rel.is_none() {
                    bail!("numeric-tolerance needs abs, rel or both");
                }
                for t in [abs, rel].into_iter().flatten() {
                    if !t.is_finite() || *t < 0.0 {
                        bail!("numeric-tolerance has an invalid tolerance {t}");
                    }
                }
            }
            Metric::PassAtK { k, samples } if *k == 0 || samples < k => {
                bail!("pass-at-k needs 1 <= k <= samples (k = {k}, samples = {samples})");
            }
            _ => {}
        }
        let fits = match (&self.family, &self.metric, &self.extract) {
            (Family::Code, Metric::PassAtK { .. }, Extract::None) => true,
            (Family::Code, _, _) | (_, Metric::PassAtK { .. }, _) => false,
            (Family::MultipleChoice, Metric::ChoiceByLikelihood, Extract::None) => true,
            (_, Metric::ChoiceByLikelihood, _) => false,
            (_, _, Extract::None) => false,
            (Family::MultipleChoice, _, _) | (Family::CheckableAnswer, _, _) => true,
        };
        if !fits {
            bail!(
                "family {:?} cannot use metric {:?} with extract {:?}",
                self.family,
                self.metric,
                self.extract
            );
        }
        Ok(())
    }

    /// `id@vN`.
    pub fn key(&self) -> String {
        format!("{}@v{}", self.id, self.version)
    }

    /// SHA-256 of the task's canonical JSON; changes with any field.
    pub fn hash(&self) -> String {
        compute_sha256(&serde_json::to_vec(self).unwrap_or_default())
    }

    /// SHA-256 of the prompt template, recorded in every result.
    pub fn prompt_template_hash(&self) -> String {
        compute_sha256(self.prompt_template.as_bytes())
    }

    /// Checks the dataset file against its recorded hash. `task_dir` is the task file's folder.
    pub fn verify_dataset(&self, task_dir: &Path) -> Result<PathBuf> {
        let path = task_dir.join(&self.dataset.path);
        let bytes =
            std::fs::read(&path).with_context(|| format!("Dataset {path:?} cannot be read"))?;
        let actual = compute_sha256(&bytes);
        if actual != self.dataset.sha256 {
            bail!(
                "Dataset {path:?} has SHA-256 {actual}, task {} expects {}",
                self.key(),
                self.dataset.sha256
            );
        }
        Ok(path)
    }
}

/// How the numbers in a record were obtained (dossier R.4).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum RunMode {
    /// THME's own forward pass.
    Native,
    /// An existing evaluation tool run as an external process.
    Bridged,
    /// Numbers supplied by the user.
    Imported,
}

/// Evidence status of a record, as in CONTEXT.md.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Evidence {
    Measured,
    Imported,
    Estimate,
    Simulated,
}

/// How the interval was computed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum IntervalMethod {
    Wilson,
    Bootstrap,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BenchmarkRef {
    pub id: String,
    pub version: u32,
    pub task_sha256: String,
    pub dataset_sha256: String,
    pub split: String,
    pub licence: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelRef {
    pub sha256: String,
    /// Recipe or conversion that produced the model; `None` for an unmodified source model.
    pub transformation: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Harness {
    pub version: String,
    pub run_mode: RunMode,
    pub prompt_template_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Decoding {
    /// 0 means greedy.
    pub temperature: f64,
    /// Required whenever `temperature > 0`.
    pub seed: Option<u64>,
    pub max_new_tokens: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ItemResult {
    pub id: String,
    /// The model's output; `None` for likelihood scoring or when not kept.
    pub output: Option<String>,
    /// In [0, 1]: 0 or 1 for correct/incorrect, a fraction for pass@k.
    pub score: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Interval {
    pub low: f64,
    pub high: f64,
    pub method: IntervalMethod,
    /// Confidence level, for example 0.95.
    pub level: f64,
}

/// One run of one task on one model: every field of dossier Table R2.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResultRecord {
    pub schema: String,
    pub benchmark: BenchmarkRef,
    pub model: ModelRef,
    pub harness: Harness,
    pub decoding: Decoding,
    pub items: Vec<ItemResult>,
    /// Mean of the item scores.
    pub score: f64,
    pub interval: Interval,
    pub evidence: Evidence,
    pub hardware: String,
    /// `YYYY-MM-DD`.
    pub run_date: String,
    pub duration_s: f64,
}

impl ResultRecord {
    /// Reads a record and checks it on its own.
    pub fn load(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("Failed to read record {path:?}"))?;
        let record: Self = serde_json::from_str(&text)
            .with_context(|| format!("{path:?} is not a valid result record"))?;
        record
            .validate()
            .with_context(|| format!("{path:?} is not a valid result record"))?;
        Ok(record)
    }

    /// Refuses records whose fields contradict each other.
    pub fn validate(&self) -> Result<()> {
        if self.schema != RECORD_SCHEMA {
            bail!("schema is '{}', expected '{RECORD_SCHEMA}'", self.schema);
        }
        check_slug("benchmark.id", &self.benchmark.id)?;
        if self.benchmark.version == 0 {
            bail!("benchmark.version starts at 1");
        }
        for (field, value) in [
            ("benchmark.task_sha256", &self.benchmark.task_sha256),
            ("benchmark.dataset_sha256", &self.benchmark.dataset_sha256),
            ("model.sha256", &self.model.sha256),
            (
                "harness.prompt_template_sha256",
                &self.harness.prompt_template_sha256,
            ),
        ] {
            check_sha256(field, value)?;
        }
        for (field, value) in [
            ("benchmark.split", &self.benchmark.split),
            ("benchmark.licence", &self.benchmark.licence),
            ("harness.version", &self.harness.version),
            ("hardware", &self.hardware),
        ] {
            if value.trim().is_empty() {
                bail!("{field} is empty");
            }
        }
        match (self.harness.run_mode, self.evidence) {
            (RunMode::Imported, Evidence::Imported) => {}
            (RunMode::Imported, e) => bail!("imported numbers are labelled imported, not {e:?}"),
            (_, Evidence::Imported) => bail!("evidence 'imported' needs run_mode 'imported'"),
            _ => {}
        }
        let d = &self.decoding;
        if !d.temperature.is_finite() || d.temperature < 0.0 {
            bail!("decoding.temperature {} is invalid", d.temperature);
        }
        if d.temperature > 0.0 && d.seed.is_none() {
            bail!("sampling at temperature {} needs a seed", d.temperature);
        }
        if self.items.is_empty() {
            bail!("a record needs its per-item results");
        }
        let mut ids = BTreeSet::new();
        let mut sum = 0.0;
        for item in &self.items {
            if !ids.insert(item.id.as_str()) {
                bail!("item '{}' appears twice", item.id);
            }
            if !(0.0..=1.0).contains(&item.score) {
                bail!("item '{}' has score {} outside [0, 1]", item.id, item.score);
            }
            sum += item.score;
        }
        let mean = sum / self.items.len() as f64;
        if !self.score.is_finite() || (self.score - mean).abs() > 1e-9 {
            bail!(
                "score {} is not the mean of the {} item scores ({mean})",
                self.score,
                self.items.len()
            );
        }
        let i = &self.interval;
        if !(0.0..=1.0).contains(&i.low) || !(0.0..=1.0).contains(&i.high) {
            bail!("interval [{}, {}] is outside [0, 1]", i.low, i.high);
        }
        if !(i.low <= self.score && self.score <= i.high) {
            bail!(
                "interval [{}, {}] does not contain the score {}",
                i.low,
                i.high,
                self.score
            );
        }
        if !(i.level > 0.0 && i.level < 1.0) {
            bail!("interval.level {} must be between 0 and 1", i.level);
        }
        if !self.duration_s.is_finite() || self.duration_s < 0.0 {
            bail!("duration_s {} is invalid", self.duration_s);
        }
        check_date(&self.run_date)?;
        Ok(())
    }

    /// Checks that the record was produced by exactly this task.
    pub fn check_against(&self, task: &TaskSpec) -> Result<()> {
        let b = &self.benchmark;
        let pairs = [
            ("benchmark id", b.id.clone(), task.id.clone()),
            (
                "benchmark version",
                b.version.to_string(),
                task.version.to_string(),
            ),
            ("task hash", b.task_sha256.clone(), task.hash()),
            (
                "dataset hash",
                b.dataset_sha256.clone(),
                task.dataset.sha256.clone(),
            ),
            ("split", b.split.clone(), task.dataset.split.clone()),
            ("licence", b.licence.clone(), task.dataset.licence.clone()),
            (
                "prompt template hash",
                self.harness.prompt_template_sha256.clone(),
                task.prompt_template_hash(),
            ),
        ];
        for (what, record, expected) in pairs {
            if record != expected {
                bail!("{what} is '{record}', task {} has '{expected}'", task.key());
            }
        }
        Ok(())
    }
}

/// Every task file in a folder, keyed by `id@vN`.
#[derive(Debug, Default)]
pub struct Registry {
    tasks: BTreeMap<String, (PathBuf, TaskSpec)>,
}

impl Registry {
    /// Loads every `*.json` file in `dir` (not recursive). One malformed file or one duplicate
    /// `id@vN` refuses the whole folder, so a registry is never partly loaded.
    pub fn load(dir: &Path) -> Result<Self> {
        let mut paths: Vec<PathBuf> = std::fs::read_dir(dir)
            .with_context(|| format!("Failed to read task folder {dir:?}"))?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.is_file() && p.extension().is_some_and(|x| x == "json"))
            .collect();
        paths.sort();
        let mut tasks = BTreeMap::new();
        for path in paths {
            let task = TaskSpec::load(&path)?;
            if let Some((first, _)) = tasks.get(&task.key()) {
                bail!("{} is defined twice: {first:?} and {path:?}", task.key());
            }
            tasks.insert(task.key(), (path, task));
        }
        Ok(Self { tasks })
    }

    pub fn get(&self, id: &str, version: u32) -> Option<&TaskSpec> {
        self.tasks.get(&format!("{id}@v{version}")).map(|(_, t)| t)
    }

    /// Tasks with the file each came from, sorted by `id@vN`.
    pub fn iter(&self) -> impl Iterator<Item = (&Path, &TaskSpec)> {
        self.tasks.values().map(|(p, t)| (p.as_path(), t))
    }

    pub fn len(&self) -> usize {
        self.tasks.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tasks.is_empty()
    }
}

fn check_slug(field: &str, value: &str) -> Result<()> {
    let ok = !value.is_empty()
        && value.len() <= 64
        && value
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || "-_.".contains(c))
        && value.starts_with(|c: char| c.is_ascii_alphanumeric());
    if !ok {
        bail!("{field} '{value}' must be 1-64 characters of a-z, 0-9, '-', '_', '.', starting with a letter or digit");
    }
    Ok(())
}

fn check_sha256(field: &str, value: &str) -> Result<()> {
    if value.len() != 64 || !value.chars().all(|c| matches!(c, '0'..='9' | 'a'..='f')) {
        bail!("{field} must be 64 lowercase hex characters");
    }
    Ok(())
}

/// Dataset paths are relative and stay inside the task's folder, so task files are portable
/// and never carry a developer's absolute path.
fn check_relative_path(path: &str) -> Result<()> {
    let p = Path::new(path);
    let inside = !path.is_empty()
        && !path.contains('\\')
        && p.components().all(|c| matches!(c, Component::Normal(_)));
    if !inside {
        bail!("dataset.path '{path}' must be a relative path with '/' separators, inside the task folder");
    }
    Ok(())
}

fn check_date(value: &str) -> Result<()> {
    let parts: Vec<&str> = value.split('-').collect();
    let ok = matches!(parts.as_slice(), [y, m, d]
        if y.len() == 4 && m.len() == 2 && d.len() == 2
            && [y, m, d].iter().all(|s| s.chars().all(|c| c.is_ascii_digit()))
            && (1..=12).contains(&m.parse::<u32>().unwrap_or(0))
            && (1..=31).contains(&d.parse::<u32>().unwrap_or(0)));
    if !ok {
        bail!("run_date '{value}' must be YYYY-MM-DD");
    }
    Ok(())
}

#[cfg(test)]
mod tests;
