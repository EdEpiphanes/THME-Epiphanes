// SPDX-License-Identifier: AGPL-3.0-only
use thme::dataset::{Dataset, DatasetFormat};

#[test]
fn test_dataset_plain_text_import_and_tokenize() {
    let text = "Hello world! This is a test of the topological manifold dataset ingestion.\n\nSecond paragraph with more tokens.";
    let ds = Dataset::from_content("test_plain", text, DatasetFormat::PlainText).unwrap();

    assert_eq!(ds.samples.len(), 2);
    assert_eq!(ds.summary.total_samples, 2);
    assert!(ds.summary.total_tokens > 5);
    assert!(ds.summary.estimated_vocab_size > 3);
    assert_eq!(ds.format, DatasetFormat::PlainText);

    // Verify token batch retrieval
    let batch = ds.get_batch(2, 10);
    assert!(!batch.is_empty());
    assert_eq!(batch.len(), 2);

    let flat = ds.get_flat_batch(10, 0);
    assert!(!flat.is_empty());
}

#[test]
fn test_dataset_json_lines_import() {
    let jsonl = r#"{"text": "Sample number one for LLM training"}
{"text": "Sample number two with instruction context"}
{"messages": [{"role": "user", "content": "Tell me about quantum physics"}, {"role": "assistant", "content": "Quantum mechanics governs subatomic particles"}]}
{"instruction": "Summarize this article", "input": "Long context here", "output": "Summary here"}"#;

    let ds = Dataset::from_content("test_jsonl", jsonl, DatasetFormat::JsonLines).unwrap();
    assert_eq!(ds.samples.len(), 4);
    assert_eq!(ds.summary.total_samples, 4);
    assert!(ds.summary.total_tokens >= 15);
}

#[test]
fn test_dataset_json_array_import() {
    let json_arr = r#"[
        {"instruction": "Explain gravity", "output": "Gravity is spacetime curvature"},
        {"instruction": "Code in Rust", "output": "fn main() { println!(\"Hello\"); }"}
    ]"#;

    let ds = Dataset::from_content("test_json_array", json_arr, DatasetFormat::JsonArray).unwrap();
    assert_eq!(ds.samples.len(), 2);
    assert_eq!(ds.summary.total_samples, 2);
}

#[test]
fn test_dataset_csv_import() {
    let csv = "id,prompt,response\n1,Translate hello,Hola\n2,Compute 2+2,4\n3,What is rust,A systems language";
    let ds = Dataset::from_content("test_csv", csv, DatasetFormat::Csv).unwrap();
    assert_eq!(ds.samples.len(), 3);
    assert_eq!(ds.summary.total_samples, 3);
}

#[test]
fn test_dataset_synthetic_calibration() {
    let ds = Dataset::create_synthetic_calibration().unwrap();
    assert_eq!(ds.samples.len(), 5);
    assert_eq!(ds.summary.total_samples, 5);
    assert!(ds.summary.total_tokens > 50);
}
