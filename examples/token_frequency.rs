// SPDX-License-Identifier: AGPL-3.0-only
//! Token-frequency count (THME-3079): how often each vocabulary row occurs in the WikiText-2
//! train split, and how well train frequencies predict the test split. Weights-only: the model
//! is never run, only its tokenizer output is counted.
//!
//! ```text
//! cargo run --release --example token_frequency -- \
//!     --train-tokens target/eval/wikitext2_train.pythia-14m.u32 \
//!     --test-tokens target/eval/wikitext2_test.pythia-14m.u32 \
//!     --vocab-size 50304
//! ```
//!
//! Each `.u32` file needs its provenance `.json` sidecar from `scripts/prepare_eval_tokens.py`;
//! the report pins both token files and the tokenizer by SHA-256.

use anyhow::{bail, Context, Result};
use clap::Parser;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

use thme::inference::read_token_file;

/// Row counts whose coverage is reported.
const TOP_K: [usize; 3] = [1024, 4096, 8192];

#[derive(Parser, Debug)]
#[command(
    name = "token_frequency",
    about = "Token frequency table and top-K row coverage"
)]
struct Args {
    /// Train-split token file (little-endian u32)
    #[arg(long)]
    train_tokens: PathBuf,

    /// Test-split token file (little-endian u32)
    #[arg(long)]
    test_tokens: PathBuf,

    /// Rows of the model's embedding matrix (config vocab_size, not the tokenizer's)
    #[arg(long, default_value_t = 50304)]
    vocab_size: usize,

    /// Where to write the report
    #[arg(
        long,
        default_value = "docs/benchmarks/pythia-14m/data/token_frequency.json"
    )]
    report_json: PathBuf,
}

/// Occurrences of each id; an id outside the vocabulary is an error, not a silent drop.
fn count_tokens(tokens: &[u32], vocab_size: usize) -> Result<Vec<u64>> {
    let mut counts = vec![0u64; vocab_size];
    for &t in tokens {
        match counts.get_mut(t as usize) {
            Some(c) => *c += 1,
            None => bail!("token id {t} is outside the vocabulary of {vocab_size} rows"),
        }
    }
    Ok(counts)
}

/// Row ids by descending count, ties by ascending id so the order is reproducible.
fn rank(counts: &[u64]) -> Vec<usize> {
    let mut ids: Vec<usize> = (0..counts.len()).collect();
    ids.sort_by(|&a, &b| counts[b].cmp(&counts[a]).then(a.cmp(&b)));
    ids
}

/// Share of the tokens in `counts` that fall in the first `k` rows of `ranking`.
fn coverage(counts: &[u64], ranking: &[usize], k: usize) -> f64 {
    let total: u64 = counts.iter().sum();
    let covered: u64 = ranking.iter().take(k).map(|&r| counts[r]).sum();
    covered as f64 / total.max(1) as f64
}

fn sha256_file(path: &Path) -> Result<String> {
    let bytes = std::fs::read(path).with_context(|| format!("Failed to read {path:?}"))?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}

fn sidecar(tokens: &Path) -> Result<Value> {
    let path = tokens.with_extension("json");
    let text = std::fs::read_to_string(&path)
        .with_context(|| format!("Failed to read sidecar {path:?}"))?;
    serde_json::from_str(&text).with_context(|| format!("Failed to parse sidecar {path:?}"))
}

/// One split's identity: the file hash is recomputed, the rest comes from the sidecar.
fn split_json(tokens_path: &Path, tokens: &[u32]) -> Result<Value> {
    let meta = sidecar(tokens_path)?;
    let hash = sha256_file(tokens_path)?;
    if meta["token_file_sha256"].as_str() != Some(hash.as_str()) {
        bail!("{tokens_path:?} does not match the hash in its sidecar");
    }
    Ok(json!({
        "split": meta["split"],
        "corpus_sha256": meta["corpus_sha256"],
        "token_file_sha256": hash,
        "token_count": tokens.len(),
    }))
}

fn main() -> Result<()> {
    let args = Args::parse();
    let train = read_token_file(&args.train_tokens)?;
    let test = read_token_file(&args.test_tokens)?;
    let train_meta = sidecar(&args.train_tokens)?;
    let test_meta = sidecar(&args.test_tokens)?;
    if train_meta["tokenizer_sha256"] != test_meta["tokenizer_sha256"] {
        bail!("train and test were tokenised with different tokenizers");
    }
    let train_counts = count_tokens(&train, args.vocab_size)?;
    let test_counts = count_tokens(&test, args.vocab_size)?;
    let ranking = rank(&train_counts);
    let oracle = rank(&test_counts);
    let unseen = |counts: &[u64]| counts.iter().filter(|&&c| c == 0).count();
    let unseen_test_tokens: u64 = (0..args.vocab_size)
        .filter(|&r| train_counts[r] == 0)
        .map(|r| test_counts[r])
        .sum();

    let top: Vec<Value> = TOP_K
        .iter()
        .map(|&k| {
            json!({
                "rows": k,
                "train_coverage": coverage(&train_counts, &ranking, k),
                "test_coverage_with_train_rows": coverage(&test_counts, &ranking, k),
                "test_coverage_with_test_rows": coverage(&test_counts, &oracle, k),
            })
        })
        .collect();
    let report = json!({
        "tool": "examples/token_frequency.rs (THME-3079)",
        "model_run": false,
        "vocab_size": args.vocab_size,
        "tokenizer_sha256": train_meta["tokenizer_sha256"],
        "tokenizer_repo": train_meta["tokenizer_repo"],
        "train": split_json(&args.train_tokens, &train)?,
        "test": split_json(&args.test_tokens, &test)?,
        "ranking": "rows by descending train count, ties by ascending id",
        "rows_unseen_in_train": unseen(&train_counts),
        "rows_unseen_in_test": unseen(&test_counts),
        "test_tokens_in_rows_unseen_in_train": unseen_test_tokens,
        "coverage": top,
        "train_counts": train_counts,
    });
    if let Some(dir) = args.report_json.parent() {
        std::fs::create_dir_all(dir).with_context(|| format!("Failed to create {dir:?}"))?;
    }
    std::fs::write(&args.report_json, serde_json::to_string(&report)? + "\n")
        .with_context(|| format!("Failed to write {:?}", args.report_json))?;
    for row in report["coverage"].as_array().into_iter().flatten() {
        println!("{row}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counts_rank_and_coverage_on_a_tiny_array() {
        let counts = count_tokens(&[3, 1, 3, 3, 2, 1, 0], 5).unwrap();
        assert_eq!(counts, vec![1, 2, 1, 3, 0]);
        let ranking = rank(&counts);
        assert_eq!(ranking, vec![3, 1, 0, 2, 4]); // tie 0 vs 2 resolved by id
        assert_eq!(coverage(&counts, &ranking, 1), 3.0 / 7.0);
        assert_eq!(coverage(&counts, &ranking, 2), 5.0 / 7.0);
        assert_eq!(coverage(&counts, &ranking, 5), 1.0);
        // a second split scored with the first split's ranking
        let other = count_tokens(&[4, 4, 3], 5).unwrap();
        assert_eq!(coverage(&other, &ranking, 1), 1.0 / 3.0);
    }

    #[test]
    fn out_of_range_id_is_an_error() {
        assert!(count_tokens(&[0, 5], 5).is_err());
    }
}
