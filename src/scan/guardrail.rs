// SPDX-License-Identifier: AGPL-3.0-only
//! Guardrail detection (v0.2.7.7, THME-3021). Read-only: the scan only reports safety components
//! and refusal instructions it finds as `Info` findings. It never removes, edits or tests them,
//! and a clean result does not mean a model has no guardrails (a keyword match is a hint, not a
//! measurement of refusal behaviour).

use super::{ScanReport, Severity};

/// Phrases (lower case) that instruct a model to refuse or decline.
const REFUSAL_PHRASES: [&str; 10] = [
    "must refuse",
    "should refuse",
    "you refuse",
    "refuse to answer",
    "must decline",
    "should decline",
    "decline to answer",
    "i cannot assist",
    "i can't assist",
    "i'm sorry, but i can",
];

/// Name fragments (lower case) of known safety classifiers and filters.
const COMPONENT_MARKERS: [&str; 9] = [
    "llama-guard",
    "llamaguard",
    "llama_guard",
    "shieldgemma",
    "prompt-guard",
    "granite-guardian",
    "wildguard",
    "nemoguard",
    "safetychecker",
];

/// Reports refusal instructions found in a chat template or system prompt.
pub(super) fn check_refusal(text: &str, rel: &str, key: &str, report: &mut ScanReport) {
    let lower = text.to_ascii_lowercase();
    let hits: Vec<&str> = REFUSAL_PHRASES
        .iter()
        .copied()
        .filter(|p| lower.contains(p))
        .collect();
    if !hits.is_empty() {
        report.add(
            Severity::Info,
            "guardrail-refusal-instruction",
            rel,
            format!("'{key}' contains refusal wording: {}", hits.join(", ")),
        );
    }
}

/// Reports a model name or identifier that names a known safety classifier.
pub(super) fn check_name(name: &str, rel: &str, key: &str, report: &mut ScanReport) {
    let lower = name.to_ascii_lowercase().replace([' ', '.'], "-");
    if let Some(m) = COMPONENT_MARKERS.iter().find(|m| lower.contains(*m)) {
        report.add(
            Severity::Info,
            "guardrail-component",
            rel,
            format!("'{key}' = '{name}' looks like a safety classifier ({m})"),
        );
    }
}

/// `config.json`: architecture and source names.
pub(super) fn check_config(json: &serde_json::Value, rel: &str, report: &mut ScanReport) {
    if let Some(name) = json.get("_name_or_path").and_then(|v| v.as_str()) {
        check_name(name, rel, "_name_or_path", report);
    }
    if let Some(list) = json.get("architectures").and_then(|v| v.as_array()) {
        for arch in list.iter().filter_map(|v| v.as_str()) {
            check_name(arch, rel, "architectures", report);
        }
    }
}

/// Diffusers `model_index.json`: a non-null `safety_checker` entry is a safety component.
pub(super) fn check_model_index(json: &serde_json::Value, rel: &str, report: &mut ScanReport) {
    let present = match json.get("safety_checker") {
        None | Some(serde_json::Value::Null) => false,
        Some(serde_json::Value::Array(a)) => a.iter().any(|v| !v.is_null()),
        Some(_) => true,
    };
    if present {
        report.add(
            Severity::Info,
            "guardrail-component",
            rel,
            "pipeline declares a safety_checker component".to_string(),
        );
    }
}
