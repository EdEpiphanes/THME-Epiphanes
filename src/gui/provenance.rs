// SPDX-License-Identifier: AGPL-3.0-only
//! Provenance labels for numbers shown in the Studio.
//!
//! A figure built from a formula carries `[estimate]`; a figure from a fixed script carries
//! `[simulated]`. Each listed value has a `// provenance: <id>` marker above the statement that
//! builds its text, and the statement must call [`labelled`] within 12 lines of the marker.

/// How a displayed figure was obtained (see CONTEXT.md).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Evidence {
    /// A figure from a stated formula, never from a run.
    Estimate,
    /// A figure from a toy or synthetic simulation.
    Simulated,
}

impl Evidence {
    /// The tag appended to a displayed value.
    pub fn tag(self) -> &'static str {
        match self {
            Evidence::Estimate => "[estimate]",
            Evidence::Simulated => "[simulated]",
        }
    }
}

/// Append the evidence tag to a displayed value, e.g. `"12.0 tok/s [estimate]"`.
pub fn labelled(text: impl Into<String>, e: Evidence) -> String {
    format!("{} {}", text.into(), e.tag())
}

/// One displayed value covered by a provenance marker.
pub struct Listed {
    pub id: &'static str,
    pub file: &'static str,
    pub formula: &'static str,
    pub evidence: Evidence,
}

/// Every marked value in the Studio. A marker without a row, or a row without a marker, fails.
pub const LISTED: &[Listed] = &[
    Listed {
        id: "baseline-profile",
        file: "src/gui/app.rs",
        formula: "tok/s = tokens / (TTFT + tokens * ms_per_tok); TTFT and KV cache from prompt length and layer count",
        evidence: Evidence::Estimate,
    },
    Listed {
        id: "baseline-throughput",
        file: "src/gui/app.rs",
        formula: "tok/s = tokens / (TTFT + tokens * ms_per_tok), ms_per_tok = params * 0.5 / (bandwidth from a timer)",
        evidence: Evidence::Estimate,
    },
    Listed {
        id: "tuner-candidate",
        file: "src/gui/app.rs",
        formula: "latency = 3.2 * (bytes_per_weight / 0.5) * mixer factor; distortion = 0.04a + 0.012b + 0.018g",
        evidence: Evidence::Simulated,
    },
    Listed {
        id: "tuner-winner",
        file: "src/gui/app.rs",
        formula: "latency = 1.65 + 0.4 * depth; distortion = 0.008 + 0.004 * depth; storage from channel width and bit shares",
        evidence: Evidence::Simulated,
    },
    Listed {
        id: "scorecard",
        file: "src/gui/app.rs",
        formula: "speedup = 1 / (f_bar * 0.80 + 0.15); storage = stored MB from the tuned layers",
        evidence: Evidence::Estimate,
    },
    Listed {
        id: "scorecard-log",
        file: "src/gui/app.rs",
        formula: "speedup = 1 / (f_bar * 0.80 + 0.15)",
        evidence: Evidence::Estimate,
    },
    Listed {
        id: "scorecard-checkpoint",
        file: "src/gui/app.rs",
        formula: "speedup = 1 / (f_bar * 0.80 + 0.15); storage = stored MB from the tuned layers",
        evidence: Evidence::Estimate,
    },
    Listed {
        id: "ribbon-vram",
        file: "src/gui/app.rs",
        formula: "stored MB = channel_width^2 * bytes_per_weight * 3 / 2^20, summed over layers",
        evidence: Evidence::Estimate,
    },
    Listed {
        id: "ribbon-traffic",
        file: "src/gui/app.rs",
        formula: "traffic MB = reach_probability * stored MB per layer, summed over layers",
        evidence: Evidence::Estimate,
    },
    Listed {
        id: "ribbon-depth",
        file: "src/gui/app.rs",
        formula: "f_bar = mean reach_probability over layers",
        evidence: Evidence::Estimate,
    },
    Listed {
        id: "ribbon-energy",
        file: "src/gui/app.rs",
        formula: "mJ per token = traffic MB / 1024 * 8 * 15 * 1000",
        evidence: Evidence::Estimate,
    },
    Listed {
        id: "ribbon-throughput",
        file: "src/gui/app.rs",
        formula: "tok/s = 1000 / (f_bar * 38 + 10)",
        evidence: Evidence::Estimate,
    },
];

const MARKER: &str = "// provenance: ";
const WINDOW: usize = 12;

/// Ok when `source` has exactly one `// provenance: <id>` comment and `labelled(` or `.tag()`
/// appears within the 12 lines after it.
pub fn check_marker(source: &str, id: &str) -> Result<(), String> {
    let lines: Vec<&str> = source.lines().collect();
    let hits: Vec<usize> = lines
        .iter()
        .enumerate()
        .filter(|(_, line)| marker_id(line) == Some(id))
        .map(|(i, _)| i)
        .collect();
    let at = match hits.as_slice() {
        [] => return Err(format!("no `{MARKER}{id}` marker")),
        [one] => *one,
        _ => return Err(format!("`{id}` marker appears {} times", hits.len())),
    };
    let end = (at + 1 + WINDOW).min(lines.len());
    // A label that belongs to the next marker's statement does not count for this one.
    let labelled = lines[at + 1..end]
        .iter()
        .take_while(|line| marker_id(line).is_none())
        .any(|line| line.contains("labelled(") || line.contains(".tag()"));
    if labelled {
        Ok(())
    } else {
        Err(format!(
            "no label within {WINDOW} lines after `{id}` marker"
        ))
    }
}

fn marker_id(line: &str) -> Option<&str> {
    line.trim_start().strip_prefix(MARKER).map(str::trim)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn labelled_appends_estimate_tag() {
        assert_eq!(
            labelled("12.0 tok/s", Evidence::Estimate),
            "12.0 tok/s [estimate]"
        );
    }

    #[test]
    fn labelled_appends_simulated_tag() {
        assert_eq!(
            labelled(String::from("3.20 ms"), Evidence::Simulated),
            "3.20 ms [simulated]"
        );
    }

    #[test]
    fn check_marker_passes_with_label_three_lines_later() {
        let src = "// provenance: demo\nlet a = 1;\nlet b = 2;\nlet s = labelled(x, Evidence::Estimate);\n";
        assert!(check_marker(src, "demo").is_ok());
    }

    #[test]
    fn check_marker_accepts_tag_call() {
        let src = "// provenance: demo\nlet s = format!(\"x\") + e.tag();\n";
        assert!(check_marker(src, "demo").is_ok());
    }

    #[test]
    fn check_marker_fails_without_label() {
        let src = "// provenance: demo\nlet s = format!(\"x\");\n";
        assert!(check_marker(src, "demo").is_err());
    }

    #[test]
    fn check_marker_fails_without_marker() {
        let src = "let s = labelled(x, Evidence::Estimate);\n";
        assert!(check_marker(src, "demo").is_err());
    }

    #[test]
    fn check_marker_fails_on_duplicate_id() {
        let src = "// provenance: demo\nlet s = labelled(x, Evidence::Estimate);\n// provenance: demo\nlet t = labelled(y, Evidence::Estimate);\n";
        assert!(check_marker(src, "demo").is_err());
    }

    #[test]
    fn check_marker_ignores_label_owned_by_next_marker() {
        let src = "// provenance: a\nlet x = 1;\n// provenance: b\nlet s = labelled(x, Evidence::Estimate);\n";
        assert!(check_marker(src, "a").is_err());
        assert!(check_marker(src, "b").is_ok());
    }

    #[test]
    fn check_marker_fails_when_label_is_thirteen_lines_away() {
        let mut src = String::from("// provenance: demo\n");
        for _ in 0..12 {
            src.push_str("let filler = 0;\n");
        }
        src.push_str("let s = labelled(x, Evidence::Estimate);\n");
        assert!(check_marker(&src, "demo").is_err());
    }

    #[test]
    fn listed_values_are_labelled_in_source() {
        for row in LISTED {
            let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(row.file);
            let source = std::fs::read_to_string(&path)
                .unwrap_or_else(|e| panic!("cannot read {}: {e}", row.file));
            if let Err(msg) = check_marker(&source, row.id) {
                panic!("{} ({}): {msg}", row.id, row.file);
            }
        }
    }

    #[test]
    fn every_marker_is_listed() {
        let dirs = ["src/gui", "src/gui/app"];
        for dir in dirs {
            let base = Path::new(env!("CARGO_MANIFEST_DIR")).join(dir);
            for entry in std::fs::read_dir(&base).expect("read gui dir") {
                let path = entry.expect("dir entry").path();
                if path.extension().and_then(|e| e.to_str()) != Some("rs")
                    || path.file_name().and_then(|n| n.to_str()) == Some("provenance.rs")
                {
                    continue;
                }
                let source = std::fs::read_to_string(&path).expect("read source");
                for line in source.lines() {
                    if let Some(pos) = line.find(MARKER) {
                        let id = line[pos + MARKER.len()..].trim();
                        assert!(
                            LISTED.iter().any(|row| row.id == id),
                            "marker '{id}' in {} has no LISTED row",
                            path.display()
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn ids_are_unique() {
        for (i, row) in LISTED.iter().enumerate() {
            assert!(
                LISTED[i + 1..].iter().all(|other| other.id != row.id),
                "duplicate id {}",
                row.id
            );
        }
    }
}
