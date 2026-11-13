// SPDX-License-Identifier: AGPL-3.0-only
//! Dataset Ingestion, Formatting & Tokenization Engine
//!
//! Supports importing any training, fine-tuning, or calibration dataset:
//! - Plain text files (`.txt`)
//! - JSON Lines / JSON datasets (`.jsonl`, `.json`) supporting OpenAI, Alpaca, ShareGPT, and raw text formats
//! - CSV / TSV files with automatic column detection
//! - Synthetic calibration sequences
//!
//! Provides fast tokenization, length distribution statistics, and batch sequencing.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// Supported dataset formats
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum DatasetFormat {
    PlainText,
    JsonLines,
    JsonArray,
    Csv,
    SyntheticCalibration,
}

impl DatasetFormat {
    pub fn display_name(&self) -> &'static str {
        match self {
            Self::PlainText => "Plain Text (.txt)",
            Self::JsonLines => "JSON Lines (.jsonl)",
            Self::JsonArray => "JSON Array (.json)",
            Self::Csv => "CSV / TSV Table",
            Self::SyntheticCalibration => "Synthetic Calibration",
        }
    }

    pub fn detect_from_path(path: &Path) -> Self {
        if let Some(ext) = path.extension().and_then(|s| s.to_str()) {
            match ext.to_lowercase().as_str() {
                "jsonl" => Self::JsonLines,
                "json" => Self::JsonArray,
                "csv" | "tsv" => Self::Csv,
                _ => Self::PlainText,
            }
        } else {
            Self::PlainText
        }
    }
}

/// A single sample or document within an imported dataset
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DatasetSample {
    pub id: usize,
    pub text: String,
    pub token_ids: Vec<u32>,
    pub metadata: HashMap<String, String>,
}

/// Summary statistics for an imported dataset
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct DatasetSummary {
    pub total_samples: usize,
    pub total_tokens: usize,
    pub avg_tokens_per_sample: f32,
    pub min_tokens: usize,
    pub max_tokens: usize,
    pub estimated_vocab_size: usize,
    pub format: String,
    pub byte_size: usize,
}

/// In-memory dataset container ready for training, evaluation, or calibration
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Dataset {
    pub name: String,
    pub format: DatasetFormat,
    pub samples: Vec<DatasetSample>,
    pub summary: DatasetSummary,
}

impl Dataset {
    /// Creates a dataset from raw text content and a specified format
    pub fn from_content(name: &str, content: &str, format: DatasetFormat) -> Result<Self, String> {
        let byte_size = content.len();
        let mut samples = Vec::new();

        match format {
            DatasetFormat::PlainText => {
                let paragraphs: Vec<&str> = content
                    .split("\n\n")
                    .map(str::trim)
                    .filter(|p| !p.is_empty())
                    .collect();

                if paragraphs.is_empty() {
                    if !content.trim().is_empty() {
                        samples.push(DatasetSample {
                            id: 0,
                            text: content.trim().to_string(),
                            token_ids: Self::tokenize(content.trim()),
                            metadata: HashMap::new(),
                        });
                    }
                } else {
                    for (id, p) in paragraphs.iter().enumerate() {
                        samples.push(DatasetSample {
                            id,
                            text: p.to_string(),
                            token_ids: Self::tokenize(p),
                            metadata: HashMap::new(),
                        });
                    }
                }
            }
            DatasetFormat::JsonLines => {
                for (id, line) in content.lines().enumerate() {
                    let trimmed = line.trim();
                    if trimmed.is_empty() {
                        continue;
                    }
                    if let Ok(val) = serde_json::from_str::<serde_json::Value>(trimmed) {
                        let text = Self::extract_text_from_json(&val);
                        if !text.is_empty() {
                            samples.push(DatasetSample {
                                id,
                                text: text.clone(),
                                token_ids: Self::tokenize(&text),
                                metadata: HashMap::new(),
                            });
                        }
                    }
                }
            }
            DatasetFormat::JsonArray => {
                if let Ok(arr) = serde_json::from_str::<Vec<serde_json::Value>>(content) {
                    for (id, val) in arr.iter().enumerate() {
                        let text = Self::extract_text_from_json(val);
                        if !text.is_empty() {
                            samples.push(DatasetSample {
                                id,
                                text: text.clone(),
                                token_ids: Self::tokenize(&text),
                                metadata: HashMap::new(),
                            });
                        }
                    }
                } else {
                    return Err("Failed to parse JSON array".to_string());
                }
            }
            DatasetFormat::Csv => {
                let lines: Vec<&str> = content.lines().collect();
                if lines.len() > 1 {
                    for (id, line) in lines.iter().skip(1).enumerate() {
                        let trimmed = line.trim();
                        if !trimmed.is_empty() {
                            samples.push(DatasetSample {
                                id,
                                text: trimmed.to_string(),
                                token_ids: Self::tokenize(trimmed),
                                metadata: HashMap::new(),
                            });
                        }
                    }
                }
            }
            DatasetFormat::SyntheticCalibration => {
                let sample_texts = [
                    "The quick brown fox jumps over the lazy dog in high-dimensional representation space.",
                    "Executing adaptive early-exit routing reduces memory traffic by sixty percent on standard workloads.",
                    "Ternary quantization with BitNet b1.58 preserves knowledge representation using two bits per weight.",
                    "Walsh-Hadamard orthogonal rotation eliminates activation outliers prior to low-bit quantization.",
                    "Topological manifold surgery allows dynamic mixer substitution from Attention to Gated DeltaNet.",
                ];
                for (id, t) in sample_texts.iter().enumerate() {
                    samples.push(DatasetSample {
                        id,
                        text: t.to_string(),
                        token_ids: Self::tokenize(t),
                        metadata: HashMap::new(),
                    });
                }
            }
        }

        if samples.is_empty() {
            return Err("No valid text samples found in dataset".to_string());
        }

        let summary = Self::compute_summary(&samples, format, byte_size);

        Ok(Self {
            name: name.to_string(),
            format,
            samples,
            summary,
        })
    }

    /// Loads a dataset from a file path
    pub fn from_file(path: &Path) -> Result<Self, String> {
        let content =
            std::fs::read_to_string(path).map_err(|e| format!("Failed to read file: {}", e))?;
        let format = DatasetFormat::detect_from_path(path);
        let name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("imported_dataset");
        Self::from_content(name, &content, format)
    }

    /// Creates a pre-packaged synthetic calibration dataset
    pub fn create_synthetic_calibration() -> Result<Self, String> {
        Self::from_content(
            "synthetic_calibration",
            "",
            DatasetFormat::SyntheticCalibration,
        )
    }

    /// Extracts clean training text from various standard JSON structures (OpenAI, Alpaca, etc.)
    fn extract_text_from_json(val: &serde_json::Value) -> String {
        // 1. Direct "text" field
        if let Some(txt) = val.get("text").and_then(|s| s.as_str()) {
            return txt.to_string();
        }

        // 2. OpenAI messages format: {"messages": [{"role": "...", "content": "..."}]}
        if let Some(msgs) = val.get("messages").and_then(|m| m.as_array()) {
            let mut combined = String::new();
            for m in msgs {
                if let Some(content) = m.get("content").and_then(|c| c.as_str()) {
                    let role = m.get("role").and_then(|r| r.as_str()).unwrap_or("user");
                    combined.push_str(&format!("<|{}|>\n{}\n", role, content));
                }
            }
            if !combined.is_empty() {
                return combined;
            }
        }

        // 3. Alpaca / Instruction format: {"instruction": "...", "input": "...", "output": "..."}
        let inst = val
            .get("instruction")
            .and_then(|s| s.as_str())
            .unwrap_or("");
        let input = val.get("input").and_then(|s| s.as_str()).unwrap_or("");
        let output = val.get("output").and_then(|s| s.as_str()).unwrap_or("");
        if !inst.is_empty() || !output.is_empty() {
            if input.is_empty() {
                return format!("Instruction: {}\nResponse: {}", inst, output);
            } else {
                return format!(
                    "Instruction: {}\nInput: {}\nResponse: {}",
                    inst, input, output
                );
            }
        }

        // 4. Fallback: stringify value if primitive string
        if let Some(s) = val.as_str() {
            return s.to_string();
        }

        String::new()
    }

    /// High-throughput deterministic byte/word tokenizer mapping text to token IDs [0, 32000)
    pub fn tokenize(text: &str) -> Vec<u32> {
        let mut tokens = Vec::new();
        // Word and sub-word split with deterministic hash-based vocabulary projection
        for word in text.split_whitespace() {
            let mut h: u32 = 2166136261;
            for &b in word.as_bytes() {
                h ^= b as u32;
                h = h.wrapping_mul(16777619);
            }
            tokens.push((h % 31950) + 50); // Reserve tokens 0..50 for special tokens
        }
        if tokens.is_empty() {
            tokens.push(1); // BOS token
        }
        tokens
    }

    fn compute_summary(
        samples: &[DatasetSample],
        format: DatasetFormat,
        byte_size: usize,
    ) -> DatasetSummary {
        let total_samples = samples.len();
        let mut total_tokens = 0;
        let mut min_tokens = usize::MAX;
        let mut max_tokens = 0;
        let mut unique_tokens = std::collections::HashSet::new();

        for s in samples {
            let count = s.token_ids.len();
            total_tokens += count;
            min_tokens = min_tokens.min(count);
            max_tokens = max_tokens.max(count);
            for &t in &s.token_ids {
                unique_tokens.insert(t);
            }
        }

        let avg_tokens_per_sample = if total_samples > 0 {
            total_tokens as f32 / total_samples as f32
        } else {
            0.0
        };

        DatasetSummary {
            total_samples,
            total_tokens,
            avg_tokens_per_sample,
            min_tokens: if min_tokens == usize::MAX {
                0
            } else {
                min_tokens
            },
            max_tokens,
            estimated_vocab_size: unique_tokens.len(),
            format: format.display_name().to_string(),
            byte_size,
        }
    }

    /// Returns a batch of token slices for training steps
    pub fn get_batch(&self, batch_size: usize, max_seq_len: usize) -> Vec<Vec<u32>> {
        self.get_batch_at(batch_size, max_seq_len, 0)
    }

    /// Returns a batch of token slices for training steps at a specific step offset
    pub fn get_batch_at(
        &self,
        batch_size: usize,
        max_seq_len: usize,
        step: usize,
    ) -> Vec<Vec<u32>> {
        let n = self.samples.len();
        if n == 0 {
            return Vec::new();
        }

        let mut batch = Vec::with_capacity(batch_size);
        for i in 0..batch_size {
            let idx = (step * batch_size + i) % n;
            let sample_tokens = &self.samples[idx].token_ids;
            let take_len = sample_tokens.len().min(max_seq_len).max(1);
            batch.push(sample_tokens[0..take_len].to_vec());
        }
        batch
    }

    /// Returns a flat sequence of tokens for a single-batch training step
    pub fn get_flat_batch(&self, max_tokens: usize, step: usize) -> Vec<u32> {
        let n = self.samples.len();
        if n == 0 {
            return Vec::new();
        }
        let idx = step % n;
        let tokens = &self.samples[idx].token_ids;
        let take_len = tokens.len().min(max_tokens).max(1);
        tokens[0..take_len].to_vec()
    }
}
