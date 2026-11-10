# THME for Visual Studio Code & GitHub Copilot

> **Topological Heterogeneous Manifold Engine (THME)** extension by **Epiphanes**  
> Direct IDE integration for live LLM memory profiling, layer sensitivity analysis, Pareto precision optimization, and real-time Teacher LLM surgery.

---

## 🌟 Key Capabilities

1. **Native GitHub Copilot Integration (`vscode.lm.registerTool`)**:
   - Equips GitHub Copilot Chat and Copilot Edits with direct tool calling into the THME engine.
   - Copilot can inspect model layers zero-copy, propose Pareto precision strategies, submit surgery recipes, and verify perplexity before/after modifications in real time.

2. **Universal Model Context Protocol (MCP) Server**:
   - Connects any AI assistant (VS Code Copilot, Claude Desktop, Cursor, Windsurf, Antigravity) to THME via standard JSON-RPC (`thme mcp`).

3. **Interactive Sidebar Views**:
   - **Active Model & Layer Anatomy**: Inspect layers, sub-blocks (RMSNorm, QKV, MLP, Residuals), and early-exit checkpoints.
   - **Pareto Precision Plan**: Live algorithm assignments (BitNet b1.58, Hadamard WHT, SmoothQuant, 2:4 Sparsity).
   - **REA-LLM & EWHA Telemetry**: Depth fraction $\bar{f}$, memory traffic, and energy consumption projections.
   - **Copilot Live Tools**: Real-time status of registered AI tools.

4. **1-Click Native Studio GUI**:
   - Status bar item `$(circuit-board) THME: 60 FPS` launches the native GPU-accelerated desktop cockpit directly from your IDE.

---

## 🛠️ GitHub Copilot Tools

When installed, GitHub Copilot Chat has access to the following live tools:

- `thme_copilot_inspect`: Inspects model architecture, layer count, parameter shapes, and memory footprint.
  - *Example Copilot Prompt*: `"Inspect the current model and tell me how many attention heads and layers it has."`
- `thme_copilot_surgery`: Submits a declarative surgery recipe to restructure layers or apply custom algorithms.
  - *Example Copilot Prompt*: `"Convert Layer 1 to Gated DeltaNet and apply BitNet b1.58 ternary quantization."`
- `thme_copilot_validate`: Tests empirical perplexity degradation against WikiText-2/C4 and checks quality gate ($\Delta PPL \le \epsilon$).
  - *Example Copilot Prompt*: `"Validate the candidate model against baseline PPL 6.50 with a tolerance of 0.25."`

---

## 🚀 Packaging & Installation

### Option A: Install via VSIX Package
To package the extension into an installable `.vsix` file:
```bash
cd editors/vscode
npx @vscode/vsce package --no-dependencies
```
Then in VS Code:
1. Open the Extensions view (`Ctrl+Shift+X` / `Cmd+Shift+X`).
2. Click the `...` (Views and More Actions) menu at the top right.
3. Select **Install from VSIX...** and pick the generated `.vsix` file.

### Option B: Universal MCP Server for Any IDE
THME runs as a standalone MCP server conforming to the 2024-11-05 standard.

Add to your `.vscode/mcp.json` or Cursor/Claude MCP configuration:
```json
{
  "servers": {
    "thme-mcp": {
      "command": "cargo",
      "args": ["run", "--bin", "thme", "--", "mcp"],
      "env": {}
    }
  }
}
```
Or with pre-compiled binary:
```json
{
  "servers": {
    "thme-mcp": {
      "command": "thme",
      "args": ["mcp"],
      "env": {}
    }
  }
}
```

---

## ⚙️ Configuration Settings

- `thme.executablePath`: Custom path to the `thme` or `llm-tweaker` binary. Defaults to automatic workspace target resolution.
- `thme.qualityTolerance`: Default quality tolerance epsilon ($\epsilon_Q$) for Pareto precision allocation (default: `0.025`).
- `thme.autoStartMcp`: Automatically starts the MCP server for Copilot upon workspace activation (default: `true`).

---

## 📄 License

Licensed under the Apache License, Version 2.0. Developed by Epiphanes.
