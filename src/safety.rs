// SPDX-License-Identifier: AGPL-3.0-only
//! Optional safety layers THME adds on top of a model (v0.2.7.8). Design: docs/safety-layers.md.
//!
//! A layer wraps text going into or coming out of a model and can block it. Layers are off
//! unless a deployment's config switches them on, and the layers belong to the deployment, not
//! to the model: THME never edits model weights or removes a model's own safety behaviour.
//! This is the tested core only; no serving path calls it yet.

use anyhow::{bail, Result};
use serde::Deserialize;

/// Deployment config, for example `{"deny_phrases": {"enabled": true, "phrases": ["x"]}}`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SafetyConfig {
    #[serde(default)]
    pub max_input_chars: MaxInputChars,
    #[serde(default)]
    pub deny_phrases: DenyPhrases,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MaxInputChars {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub limit: usize,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DenyPhrases {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub phrases: Vec<String>,
}

/// Which way the text travels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    Input,
    Output,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Decision {
    Allow,
    /// Blocked by the named layer, with a reason that is safe to show.
    Block {
        layer: &'static str,
        reason: String,
    },
}

impl SafetyConfig {
    /// Parses a config; unknown layers or fields are errors so a typo cannot silently leave a
    /// layer off, and an enabled layer must be usable.
    pub fn from_json(text: &str) -> Result<Self> {
        let cfg: Self = serde_json::from_str(text)?;
        if cfg.max_input_chars.enabled && cfg.max_input_chars.limit == 0 {
            bail!("max_input_chars is enabled with limit 0; set a positive limit");
        }
        if cfg.deny_phrases.enabled && cfg.deny_phrases.phrases.iter().all(|p| p.trim().is_empty())
        {
            bail!("deny_phrases is enabled with no phrases");
        }
        Ok(cfg)
    }

    /// Runs the enabled layers in a fixed order; the first block wins.
    pub fn check(&self, direction: Direction, text: &str) -> Decision {
        let limit = &self.max_input_chars;
        if limit.enabled && direction == Direction::Input && text.chars().count() > limit.limit {
            return Decision::Block {
                layer: "max_input_chars",
                reason: format!("input is longer than {} characters", limit.limit),
            };
        }
        if self.deny_phrases.enabled {
            let lower = text.to_lowercase();
            let hit = self
                .deny_phrases
                .phrases
                .iter()
                .map(|p| p.trim().to_lowercase())
                .any(|p| !p.is_empty() && lower.contains(&p));
            if hit {
                return Decision::Block {
                    layer: "deny_phrases",
                    reason: "text contains a phrase this deployment blocks".to_string(),
                };
            }
        }
        Decision::Allow
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn everything_is_off_by_default() {
        let cfg = SafetyConfig::from_json("{}").unwrap();
        assert_eq!(
            cfg.check(Direction::Input, &"x".repeat(10_000)),
            Decision::Allow
        );
        assert_eq!(cfg.check(Direction::Output, "anything"), Decision::Allow);
    }

    #[test]
    fn switching_a_layer_off_again_restores_allow() {
        let on = r#"{"deny_phrases":{"enabled":true,"phrases":["Secret Plan"]}}"#;
        let off = r#"{"deny_phrases":{"enabled":false,"phrases":["Secret Plan"]}}"#;
        let text = "the secret plan is here";
        let blocked = SafetyConfig::from_json(on)
            .unwrap()
            .check(Direction::Output, text);
        assert!(
            matches!(
                blocked,
                Decision::Block {
                    layer: "deny_phrases",
                    ..
                }
            ),
            "{blocked:?}"
        );
        assert_eq!(
            SafetyConfig::from_json(off)
                .unwrap()
                .check(Direction::Output, text),
            Decision::Allow
        );
    }

    #[test]
    fn length_limit_applies_to_input_only_and_counts_characters() {
        let cfg =
            SafetyConfig::from_json(r#"{"max_input_chars":{"enabled":true,"limit":3}}"#).unwrap();
        assert_eq!(cfg.check(Direction::Input, "äöü"), Decision::Allow);
        assert!(matches!(
            cfg.check(Direction::Input, "äöüx"),
            Decision::Block { .. }
        ));
        assert_eq!(cfg.check(Direction::Output, "äöüxxxx"), Decision::Allow);
    }

    #[test]
    fn bad_configs_are_errors() {
        for body in [
            "",
            "[",
            r#"{"typo_layer":{}}"#,
            r#"{"deny_phrases":{"enabeld":true}}"#,
            r#"{"max_input_chars":{"enabled":true}}"#,
            r#"{"deny_phrases":{"enabled":true,"phrases":[" "]}}"#,
            r#"{"deny_phrases":{"enabled":"yes"}}"#,
        ] {
            assert!(SafetyConfig::from_json(body).is_err(), "{body}");
        }
    }
}
