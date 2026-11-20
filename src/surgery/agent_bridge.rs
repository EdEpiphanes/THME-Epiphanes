// SPDX-License-Identifier: AGPL-3.0-only
use serde::{Deserialize, Serialize};
use std::io::{Read, Write};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

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
            AgentBridgeConfig::Subprocess {
                command,
                args,
                timeout_secs,
            } => {
                let output = run_with_deadline(
                    Command::new(command).args(args),
                    prompt_text.as_bytes(),
                    Duration::from_secs(*timeout_secs),
                )
                .map_err(|e| anyhow::anyhow!("Agent command '{}': {}", command, e))?;
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

/// Runs `cmd`, feeds it `input` on stdin and collects its output, killing it once `timeout`
/// passes. Pipes are serviced on threads so a child that never reads or never exits cannot block
/// past the deadline.
pub(crate) fn run_with_deadline(
    cmd: &mut Command,
    input: &[u8],
    timeout: Duration,
) -> anyhow::Result<std::process::Output> {
    let mut child = cmd
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| anyhow::anyhow!("failed to spawn: {}", e))?;

    let mut stdin = child.stdin.take();
    let input = input.to_vec();
    std::thread::spawn(move || {
        if let Some(pipe) = stdin.as_mut() {
            let _ = pipe.write_all(&input);
        }
    });
    let mut stdout = child.stdout.take();
    let out = std::thread::spawn(move || {
        let mut buf = Vec::new();
        if let Some(pipe) = stdout.as_mut() {
            let _ = pipe.read_to_end(&mut buf);
        }
        buf
    });
    let mut stderr = child.stderr.take();
    let err = std::thread::spawn(move || {
        let mut buf = Vec::new();
        if let Some(pipe) = stderr.as_mut() {
            let _ = pipe.read_to_end(&mut buf);
        }
        buf
    });

    let deadline = Instant::now() + timeout;
    let status = loop {
        if let Some(status) = child.try_wait()? {
            break status;
        }
        if Instant::now() >= deadline {
            // Reader threads are left to end on their own: a grandchild may still hold the pipes.
            let _ = child.kill();
            let _ = child.wait();
            return Err(anyhow::anyhow!(
                "timed out after {} s and was killed",
                timeout.as_secs()
            ));
        }
        std::thread::sleep(Duration::from_millis(10));
    };
    Ok(std::process::Output {
        status,
        stdout: out.join().unwrap_or_default(),
        stderr: err.join().unwrap_or_default(),
    })
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

    /// Sleeper for the timeout test: ignored in normal runs, started as a child of the test
    /// binary itself (works the same on Windows, Linux and macOS).
    #[test]
    #[ignore = "child process of subprocess_is_killed_at_the_timeout"]
    fn sleeper_child() {
        std::thread::sleep(Duration::from_secs(120));
    }

    #[test]
    fn subprocess_is_killed_at_the_timeout() {
        let exe = std::env::current_exe().expect("test binary path");
        let bridge = AgentBridge::new(AgentBridgeConfig::Subprocess {
            command: exe.to_string_lossy().into_owned(),
            args: vec![
                "surgery::agent_bridge::tests::sleeper_child".into(),
                "--exact".into(),
                "--ignored".into(),
            ],
            timeout_secs: 1,
        });
        let start = Instant::now();
        let result = bridge.request_recipe("prompt");
        let err = result
            .expect_err("sleeping agent must time out")
            .to_string();
        assert!(err.contains("timed out"), "{err}");
        assert!(start.elapsed() < Duration::from_secs(60));
    }
}
