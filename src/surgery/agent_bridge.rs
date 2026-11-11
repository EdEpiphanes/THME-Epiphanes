use serde::{Deserialize, Serialize};
use std::io::Write;
use std::process::{Command, Stdio};

use crate::surgery::recipe::TweakRecipe;

/// Driver configuration for connecting to an external Teacher agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AgentBridgeConfig {
    /// Mock agent for deterministic testing and headless CI.
    Mock { canned_recipe: TweakRecipe },
    /// Subprocess CLI agent invocation (e.g. `claude`, `gemini`, `python script.py`).
    Subprocess {
        command: String,
        args: Vec<String>,
        timeout_secs: u64,
    },
}

/// Agent communication bridge invoking external Teacher models to synthesize surgery recipes.
pub struct AgentBridge {
    config: AgentBridgeConfig,
}

impl AgentBridge {
    pub fn new(config: AgentBridgeConfig) -> Self {
        Self { config }
    }

    /// Invokes the configured Teacher agent with prompt context and parses the returned TweakRecipe JSON.
    pub fn request_recipe(&self, prompt_text: &str) -> anyhow::Result<TweakRecipe> {
        match &self.config {
            AgentBridgeConfig::Mock { canned_recipe } => Ok(canned_recipe.clone()),
            AgentBridgeConfig::Subprocess { command, args, .. } => {
                let mut child = Command::new(command)
                    .args(args)
                    .stdin(Stdio::piped())
                    .stdout(Stdio::piped())
                    .stderr(Stdio::piped())
                    .spawn()
                    .map_err(|e| {
                        anyhow::anyhow!("Failed to spawn agent command '{}': {}", command, e)
                    })?;

                if let Some(mut stdin) = child.stdin.take() {
                    stdin.write_all(prompt_text.as_bytes())?;
                }

                let output = child.wait_with_output()?;
                if !output.status.success() {
                    let stderr = String::from_utf8_lossy(&output.stderr);
                    return Err(anyhow::anyhow!(
                        "Agent process exited with code {:?}: {}",
                        output.status.code(),
                        stderr
                    ));
                }

                let raw_str = String::from_utf8_lossy(&output.stdout);
                Self::extract_and_parse_json(&raw_str)
            }
        }
    }

    /// Robustly extracts JSON block even if model surrounds output with markdown backticks.
    pub fn extract_and_parse_json(text: &str) -> anyhow::Result<TweakRecipe> {
        let trimmed = text.trim();
        let json_slice = if let Some(start) = trimmed.find("```json") {
            let after = &trimmed[start + 7..];
            if let Some(end) = after.find("```") {
                &after[..end]
            } else {
                after
            }
        } else if let Some(start) = trimmed.find('{') {
            if let Some(end) = trimmed.rfind('}') {
                &trimmed[start..=end]
            } else {
                trimmed
            }
        } else {
            trimmed
        };

        let recipe: TweakRecipe = serde_json::from_str(json_slice.trim()).map_err(|e| {
            anyhow::anyhow!(
                "Failed to deserialize TweakRecipe from JSON: {}. Payload: {}",
                e,
                json_slice
            )
        })?;
        Ok(recipe)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{TensorFormat, TensorRole};
    use crate::surgery::recipe::TweakOperation;

    #[test]
    fn test_mock_agent_bridge_and_json_extraction() {
        let sample_recipe = TweakRecipe {
            name: "Agent Test".to_string(),
            version: "1.0.0".to_string(),
            author: "Mock Agent".to_string(),
            description: "Agent suggested test recipe".to_string(),
            target_model: "llama".to_string(),
            expected_memory_savings_pct: Some(25.0),
            expected_speedup: Some(1.15),
            operations: vec![TweakOperation::SetPrecision {
                start_layer: 0,
                end_layer: 1,
                role: Some(TensorRole::MlpDown),
                format: TensorFormat::Ternary,
            }],
        };

        let bridge = AgentBridge::new(AgentBridgeConfig::Mock {
            canned_recipe: sample_recipe.clone(),
        });

        let recipe = bridge.request_recipe("test prompt").expect("mock recipe");
        assert_eq!(recipe.version, "1.0.0");
        assert_eq!(recipe.operations.len(), 1);

        // Test markdown extraction parser
        let wrapped_json = format!(
            "Here is your recipe:\n```json\n{}\n```\nHope it helps!",
            serde_json::to_string(&sample_recipe).unwrap()
        );
        let parsed =
            AgentBridge::extract_and_parse_json(&wrapped_json).expect("parse markdown json");
        assert_eq!(parsed.description, "Agent suggested test recipe");
    }
}
