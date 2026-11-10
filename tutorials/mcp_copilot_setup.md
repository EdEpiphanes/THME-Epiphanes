# Model Context Protocol (MCP) and VS Code Copilot Integration Guide: Autonomous AI Architecture Surgery

## Executive Summary

The Model Context Protocol (MCP) is an open specification that standardizes how AI applications (such as GitHub Copilot, Claude Desktop, and Cursor) discover and invoke local engineering tools. 

THME exposes an integrated, headless MCP server (`thme mcp`) conforming to the JSON-RPC 2.0 specification (protocol version `2024-11-05`). It provides 10 specialized tools spanning zero-copy model inspection, automated Pareto precision planning, real-time Teacher LLM surgery, empirical perplexity quality validation, token flow simulation, speculative decoding evaluation, and hardware Roofline profiling. 

In addition to headless JSON-RPC operation, THME provides a native Visual Studio Code extension (`thme-vscode`) that registers four Activity Bar explorer views and connects directly to the GitHub Copilot Language Model Tools API.

---

## Protocol Architecture

The MCP interface communicates over standard input and standard output streams (`stdin` / `stdout`), allowing AI orchestrators to run THME as an isolated local background daemon:

```text
┌────────────────────────────────────────────────────────┐
│               AI Clients & IDE Interfaces              │
│  GitHub Copilot (VS Code) │ Claude Desktop │ Cursor    │
└───────────────────────────┬────────────────────────────┘
                            │ JSON-RPC 2.0 over stdin/stdout
                            ▼
┌────────────────────────────────────────────────────────┐
│              THME MCP Daemon (thme mcp)                │
│             Protocol Version: 2024-11-05               │
├────────────────────────────────────────────────────────┤
│  Tool Registry (10 Tools):                             │
│   1. thme_inspect_model                                │
│   2. thme_auto_decide                                  │
│   3. thme_apply_surgery                                │
│   4. thme_validate_quality                             │
│   5. thme_simulate_token_flow                          │
│   6. thme_run_experiment                               │
│   7. thme_micro_train                                  │
│   8. thme_convert_paradigm                             │
│   9. thme_simulate_speculative_decoding                │
│  10. thme_profile_roofline                             │
└───────────────────────────┬────────────────────────────┘
                            │ Zero-Copy In-Memory Manipulation
                            ▼
┌────────────────────────────────────────────────────────┐
│             THME Pure Rust Engine Core                 │
│      ModelIR │ SurgeryEngine │ QualityGate │ SVD       │
└────────────────────────────────────────────────────────┘
```

---

## Prerequisites and Environment Setup

- Visual Studio Code: Version 1.90.0 or later.
- GitHub Copilot Chat Extension: Installed and authenticated in VS Code.
- Node.js & npm (optional): Required only if compiling the VS Code extension from source (`npm install && npm run compile`).
- Compiled THME Binary: Available at `./target/release/thme`.

---

## Step 1: Launching and Testing the Headless MCP Daemon

To verify that the MCP server compiles and responds cleanly over JSON-RPC 2.0:

```bash
# Send an initialize request over standard input
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test-client","version":"1.0"}}}' | ./target/release/thme mcp
```

Expected output:

```json
{"jsonrpc":"2.0","id":1,"result":{"capabilities":{"tools":{"listChanged":false}},"protocolVersion":"2024-11-05","serverInfo":{"author":"Epiphanes","name":"thme-mcp","repository":"https://github.com/EdEpiphanes/THME-Epiphanes","version":"0.1.4"}}}
```

Query the full tool list:

```bash
echo '{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}' | ./target/release/thme mcp
```

The daemon returns JSON schemas for all 10 exposed tools, including parameter names, types, descriptions, and required arguments.

---

## Step 2: Configuring Workspace Auto-Discovery via .vscode/mcp.json

To enable VS Code, GitHub Copilot, and compatible coding agents to launch the THME MCP daemon automatically upon opening the repository workspace, create `.vscode/mcp.json`:

```json
{
  "servers": {
    "thme-mcp": {
      "command": "${workspaceFolder}/target/release/thme",
      "args": ["mcp"],
      "env": {
        "RUST_BACKTRACE": "1"
      }
    }
  }
}
```

If you are evaluating a specific model during your development session, pass `--model` to bind the model to the MCP session context:

```json
{
  "servers": {
    "thme-mcp": {
      "command": "${workspaceFolder}/target/release/thme",
      "args": ["mcp", "--model", "${workspaceFolder}/examples/mock_llama"]
    }
  }
}
```

---

## Step 3: Installing the VS Code Extension Package

THME includes a packaged VSIX extension in `editors/vscode/`:

```bash
# Install the packaged extension into VS Code
code --install-extension editors/vscode/thme-vscode-0.1.5.vsix
```

Alternatively, to build the extension from source:

```bash
cd editors/vscode
npm install
npm run compile
npx @vscode/vsce package --no-dependencies
code --install-extension thme-vscode-0.1.5.vsix
```

Once installed, the **THME Engine** activity bar icon appears on the left navigation panel of Visual Studio Code.

---

## Step 4: Exploring the 4 Activity Bar Views

The extension contributes four real-time explorer trees in the Activity Bar:

1. **Active Model & Layer Anatomy (`thme-models`)**:
   - Displays the currently loaded model architecture, parameter count, raw byte size, and tensor tree grouped by layer and functional role.
2. **Pareto Precision Plan (`thme-decisions`)**:
   - Visualizes recommended precisions per layer (`FP16`, `INT8`, `Ternary 1.58b`) generated by the Pareto allocator.
3. **REA-LLM & EWHA Telemetry (`thme-telemetry`)**:
   - Displays real-time estimated energy dissipation (Joules per token) and memory traffic metrics.
4. **GitHub Copilot Live Tools (`thme-copilot`)**:
   - Status indicators showing connection health with the local `thme mcp` daemon.

---

## Step 5: Interactive GitHub Copilot Chat Surgery

The VS Code extension registers language model tools under the VS Code LM Tools API. Within GitHub Copilot Chat, you can invoke THME operations directly using natural language prompts.

### 1. Zero-Copy Model Inspection
Prompt Copilot:
> "@copilot Inspect the model located at examples/mock_llama and tell me how many attention heads and layers it has."

Copilot internally executes `thme_copilot_inspect` and responds:
> "Model `mock_llama` has 4 layers, hidden size 256, 8 Query attention heads, 4 Key-Value heads, and 30 tensors totaling 12.12M parameters."

### 2. Autonomous Surgery Recipe Execution
Prompt Copilot:
> "@copilot Construct a THME surgery recipe that replaces the attention mixer in layer 2 with gated_deltanet and quantizes layer 1 to BitNet b1.58 ternary representation, then apply it."

Copilot synthesizes the JSON recipe and calls `thme_copilot_surgery`:

```json
{
  "recipe_version": "1.0",
  "author": "GitHub Copilot",
  "target_model": "mock_llama",
  "operations": [
    {
      "type": "ReplaceMixer",
      "layer": 2,
      "mixer_type": "gated_deltanet"
    },
    {
      "type": "SetPrecision",
      "start_layer": 1,
      "end_layer": 1,
      "role": "MlpGate",
      "format": "Ternary"
    }
  ]
}
```

THME applies the transformation AST in memory and returns an audit diff confirming that the sequence mixer and precision representations were updated.

### 3. Perplexity Quality Verification Gate
Prompt Copilot:
> "@copilot Validate if a candidate perplexity of 5.92 on WikiText-2 is acceptable given a baseline of 5.84 and a tolerance of 0.10."

Copilot invokes `thme_copilot_validate` and reports:
> "Quality Gate Verdict: **Accept**. $\Delta \text{PPL} = +0.080$ (+1.37% degradation), which is within the allowed limit ($\le 0.100$)."

---

## Step 6: Integrating with Claude Desktop and Cursor

### Claude Desktop Configuration
To connect THME tools to Claude Desktop, add the server definition to `claude_desktop_config.json`:
- **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
- **Linux**: `~/.config/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "thme": {
      "command": "/path/to/THME-Epiphanes/target/release/thme",
      "args": ["mcp"]
    }
  }
}
```

Restart Claude Desktop. The hammer icon appears in the chat box displaying all 10 THME tools.

### Cursor / Windsurf Configuration
Add the daemon command under **Cursor Settings > Features > MCP**:
- Name: `thme`
- Type: `command`
- Command: `/path/to/THME-Epiphanes/target/release/thme mcp`

---

## Comprehensive MCP Tool Registry

| Tool Name | Key Parameters | Functionality |
|:---|:---|:---|
| `thme_inspect_model` | `model_path` (string, optional) | Inspects model architecture, layer count, tensor shapes, and memory footprint zero-copy |
| `thme_auto_decide` | `model_path`, `quality_tolerance` (number) | Generates Pareto precision decisions (BitNet b1.58, WHT, SmoothQuant, 2:4 Sparsity) |
| `thme_apply_surgery` | `recipe_json` (string, required) | Applies declarative surgery recipe in real time: replaces mixers, scales parameters, prunes layers |
| `thme_validate_quality` | `baseline_ppl`, `candidate_ppl`, `corpus`, `max_delta_ppl` | Quality gate: evaluates perplexity degradation against WikiText-2, C4, or PTB |
| `thme_simulate_token_flow` | `prompt`, `num_tokens`, `early_exit_ratio` | Simulates forward token pass, calculating per-token memory traffic (bytes/token) and energy (mJ) |
| `thme_run_experiment` | `technique` (enum), `parameter` (number) | Executes experimental transformations (spectral reshaping, phase modulation, noise injection) |
| `thme_micro_train` | `learning_rate`, `loss_type`, `optimizer`, `steps` | Runs in-engine parameter adaptation steps using Cross-Entropy, Distillation, or EWHA loss |
| `thme_convert_paradigm` | `target_paradigm`, `num_experts`, `top_k`, `apply_bitnet` | Cross-converts topology across 8 specialized paradigms (LLM, LCM, LAM, MoE, VLM, SLM, MLM, SAM) |
| `thme_simulate_speculative_decoding` | `draft_layers`, `lookahead_horizon`, `apply_bitnet` | Simulates speculative decoding, returning acceptance rate $\alpha$, tokens/pass, and speedup $S$ |
| `thme_profile_roofline` | `hardware`, `batch_size`, `seq_len`, `thermal_throttling` | Profiles layer arithmetic intensity, cache residency, and compute vs memory bounds |

---

## Troubleshooting and Common Issues

### Issue 1: MCP Server Hangs or Timeouts
- **Symptom**: AI client reports `Connection timed out waiting for MCP server initialize`.
- **Cause**: THME was launched without `mcp` subcommand, or an interactive terminal command wrote non-JSON debug logs to `stdout`.
- **Remedy**: Verify that the command invoked is exactly `thme mcp`. THME strictly routes logging output to `stderr` during MCP execution, preserving clean JSON-RPC 2.0 streaming on `stdout`.

### Issue 2: VS Code LM Tools Not Appearing in Copilot
- **Symptom**: Copilot Chat does not recognize `/thme_copilot_inspect`.
- **Cause**: VS Code settings may have disabled external Language Model Tools, or the extension has not activated.
- **Remedy**: Run `THME: Start Live MCP Server for GitHub Copilot` from the VS Code Command Palette (`Ctrl+Shift+P` / `Cmd+Shift+P`).

---

## Next Steps and Related Guides

- [Universal 8 AI Model Paradigm Converter](paradigm_converter.md): Transform architectures across MoE, SLM, and multimodal models via MCP.
- [Speculative Decoding Tutorial](speculative_decoding.md): Simulate draft models and verify Leviathan speedup curves.
- [Roofline Profiling Tutorial](roofline_profiling.md): Analyze hardware ceilings and cache residency.
- [README Overview](../README.md): Central architectural documentation and production roadmap.
