// SPDX-License-Identifier: AGPL-3.0-only
//! Contamination check (roadmap v0.2.8.7, dossier WP36 and Q.6): does any benchmark item appear
//! in training or teacher-written data?
//!
//! Items and corpora are compared as normalized word n-grams (lower case, letters and digits
//! only), so changes of case, punctuation or line breaks do not hide a copy. An item shorter
//! than `n` words is matched as a whole. Corpora are streamed line by line with a carry-over
//! window, so they never have to fit in memory and a copy split across lines is still found.

use anyhow::{bail, Context, Result};
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::io::BufRead;
use std::path::Path;

/// Default n-gram length in words.
pub const DEFAULT_N: usize = 13;

/// Lower-case words made of letters and digits; everything else separates words.
pub fn words(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .map(str::to_lowercase)
        .collect()
}

/// One benchmark item found in a corpus.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ItemOverlap {
    pub id: String,
    /// Distinct n-grams of the item found in the corpora.
    pub shared_ngrams: usize,
    /// Distinct n-grams of the item.
    pub item_ngrams: usize,
    /// `shared_ngrams / item_ngrams`.
    pub fraction: f64,
    /// Corpus sources that contain at least one of the item's n-grams.
    pub sources: Vec<String>,
}

/// What a check found.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ContaminationReport {
    pub ngram_words: usize,
    pub items_checked: usize,
    pub corpus_words: u64,
    pub sources: Vec<String>,
    /// Items with at least one shared n-gram, most copied first.
    pub contaminated: Vec<ItemOverlap>,
}

/// A contamination check over a fixed set of benchmark items.
#[derive(Debug)]
pub struct ContaminationCheck {
    n: usize,
    ids: Vec<String>,
    /// n-gram (words joined by spaces) -> items containing it.
    grams: HashMap<String, Vec<usize>>,
    /// Gram lengths in use (n, and shorter for short items).
    lengths: BTreeSet<usize>,
    item_grams: Vec<usize>,
    /// item -> (shared grams, sources)
    hits: BTreeMap<usize, (BTreeSet<String>, BTreeSet<String>)>,
    corpus_words: u64,
    sources: Vec<String>,
}

impl ContaminationCheck {
    /// Prepares a check of `items` as `(id, text)` with `n`-word grams.
    pub fn new(items: &[(String, String)], n: usize) -> Result<Self> {
        if n == 0 {
            bail!("n-gram length must be at least 1");
        }
        let mut grams: HashMap<String, Vec<usize>> = HashMap::new();
        let mut lengths = BTreeSet::new();
        let mut item_grams = Vec::with_capacity(items.len());
        for (index, (_, text)) in items.iter().enumerate() {
            let w = words(text);
            let len = n.min(w.len());
            let mut own = BTreeSet::new();
            if len > 0 {
                lengths.insert(len);
                for window in w.windows(len) {
                    own.insert(window.join(" "));
                }
            }
            item_grams.push(own.len());
            for g in own {
                grams.entry(g).or_default().push(index);
            }
        }
        Ok(Self {
            n,
            ids: items.iter().map(|(id, _)| id.clone()).collect(),
            grams,
            lengths,
            item_grams,
            hits: BTreeMap::new(),
            corpus_words: 0,
            sources: Vec::new(),
        })
    }

    /// Scans one corpus, named `source` in the report.
    pub fn scan(&mut self, reader: impl BufRead, source: &str) -> Result<()> {
        self.sources.push(source.to_string());
        let longest = self.lengths.iter().next_back().copied().unwrap_or(0);
        if longest == 0 {
            return Ok(());
        }
        // The last `longest - 1` words of the previous line, so grams can span lines.
        let mut carry: Vec<String> = Vec::new();
        for line in reader.lines() {
            let line = line.with_context(|| format!("reading corpus {source}"))?;
            let fresh = words(&line);
            if fresh.is_empty() {
                continue;
            }
            self.corpus_words += fresh.len() as u64;
            let start = carry.len();
            carry.extend(fresh);
            // Every window that ends inside the fresh words.
            for end in start + 1..=carry.len() {
                for &len in &self.lengths {
                    if end < len {
                        continue;
                    }
                    let key = carry[end - len..end].join(" ");
                    if let Some(items) = self.grams.get(&key) {
                        for &i in items {
                            let entry = self.hits.entry(i).or_default();
                            entry.0.insert(key.clone());
                            entry.1.insert(source.to_string());
                        }
                    }
                }
            }
            let keep = carry.len().saturating_sub(longest - 1);
            carry.drain(..keep);
        }
        Ok(())
    }

    /// Scans a corpus file.
    pub fn scan_file(&mut self, path: &Path) -> Result<()> {
        let file = std::fs::File::open(path).with_context(|| format!("opening corpus {path:?}"))?;
        let name = path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        self.scan(std::io::BufReader::new(file), &name)
    }

    pub fn report(&self) -> ContaminationReport {
        let mut contaminated: Vec<ItemOverlap> = self
            .hits
            .iter()
            .map(|(&i, (shared, sources))| ItemOverlap {
                id: self.ids[i].clone(),
                shared_ngrams: shared.len(),
                item_ngrams: self.item_grams[i],
                fraction: shared.len() as f64 / self.item_grams[i].max(1) as f64,
                sources: sources.iter().cloned().collect(),
            })
            .collect();
        contaminated.sort_by(|a, b| b.fraction.total_cmp(&a.fraction).then(a.id.cmp(&b.id)));
        ContaminationReport {
            ngram_words: self.n,
            items_checked: self.ids.len(),
            corpus_words: self.corpus_words,
            sources: self.sources.clone(),
            contaminated,
        }
    }
}

/// Benchmark items of a JSONL dataset as `(id, text)`, the text being the input and the target
/// (see [`super::score::read_dataset`] for the format).
pub fn read_items(path: &Path) -> Result<Vec<(String, String)>> {
    Ok(super::score::read_dataset(path)?
        .into_iter()
        .map(|i| {
            let text = match i.target {
                Some(t) => format!("{} {t}", i.input),
                None => i.input,
            };
            (i.id, text)
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn items() -> Vec<(String, String)> {
        vec![
            (
                "long".into(),
                "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether?".into(),
            ),
            ("short".into(), "Capital of France?".into()),
            (
                "clean".into(),
                "A completely different question about prime numbers and their gaps in large intervals of integers".into(),
            ),
        ]
    }

    fn check(corpus: &str, n: usize) -> ContaminationReport {
        let mut c = ContaminationCheck::new(&items(), n).unwrap();
        c.scan(std::io::Cursor::new(corpus), "corpus.txt").unwrap();
        c.report()
    }

    #[test]
    fn planted_copies_are_found_despite_formatting() {
        // Planted with different case, punctuation and a line break in the middle.
        let corpus = "unrelated text here\nNATALIA sold clips to 48 of her friends -- in April; and then she\nsold half as many clips in May!!! how many clips did natalia sell altogether\nmore text";
        let r = check(corpus, 13);
        assert_eq!(r.contaminated.len(), 1, "{r:?}");
        let hit = &r.contaminated[0];
        assert_eq!(hit.id, "long");
        assert_eq!(
            hit.shared_ngrams, hit.item_ngrams,
            "the whole item was planted"
        );
        assert_eq!(hit.fraction, 1.0);
        assert_eq!(hit.sources, ["corpus.txt"]);
    }

    #[test]
    fn partial_copies_report_their_fraction() {
        // The first 14 words of the 27-word item: 2 of its 15 13-grams.
        let corpus = "natalia sold clips to 48 of her friends in april and then she sold";
        let r = check(corpus, 13);
        let hit = &r.contaminated[0];
        assert_eq!((hit.shared_ngrams, hit.item_ngrams), (2, 15), "{hit:?}");
        assert!((hit.fraction - 2.0 / 15.0).abs() < 1e-12);
    }

    #[test]
    fn short_items_match_whole_and_clean_items_do_not() {
        let r = check("quiz: capital of france? paris", 13);
        assert_eq!(r.contaminated.len(), 1);
        assert_eq!(r.contaminated[0].id, "short");
        let r = check(
            "nothing to see in this corpus at all, only filler words",
            13,
        );
        assert!(r.contaminated.is_empty());
        assert_eq!(r.items_checked, 3);
        assert_eq!(r.corpus_words, 11);
    }

    #[test]
    fn several_sources_and_many_lines() {
        let mut c = ContaminationCheck::new(&items(), 8).unwrap();
        // One word per line: every gram spans lines.
        let one_per_line: String =
            words("prime numbers and their gaps in large intervals of integers").join("\n");
        c.scan(std::io::Cursor::new("filler"), "a.txt").unwrap();
        c.scan(std::io::Cursor::new(one_per_line), "b.txt").unwrap();
        c.scan(std::io::Cursor::new("Capital of France"), "c.txt")
            .unwrap();
        let r = c.report();
        let ids: Vec<&str> = r.contaminated.iter().map(|o| o.id.as_str()).collect();
        assert_eq!(ids, ["short", "clean"], "{r:?}");
        assert_eq!(r.contaminated[1].sources, ["b.txt"]);
        assert_eq!(r.sources, ["a.txt", "b.txt", "c.txt"]);
    }

    #[test]
    fn dataset_items_are_read_strictly() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("d.jsonl");
        std::fs::write(
            &p,
            "{\"id\":\"a\",\"input\":\"2+2\",\"target\":\"4\"}\n\n{\"id\":\"b\",\"input\":\"x\"}\n",
        )
        .unwrap();
        assert_eq!(
            read_items(&p).unwrap(),
            [
                ("a".to_string(), "2+2 4".to_string()),
                ("b".to_string(), "x".to_string())
            ]
        );
        std::fs::write(
            &p,
            "{\"id\":\"a\",\"input\":\"x\"}\n{\"id\":\"a\",\"input\":\"y\"}\n",
        )
        .unwrap();
        assert!(read_items(&p).unwrap_err().to_string().contains("twice"));
        std::fs::write(&p, "{\"input\":\"x\"}\n").unwrap();
        assert!(read_items(&p).is_err());
        std::fs::write(&p, "").unwrap();
        assert!(read_items(&p).is_err());
        assert!(ContaminationCheck::new(&[], 0).is_err());
    }
}
