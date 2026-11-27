# THME for Visual Studio Code & GitHub Copilot

> **Topological Heterogeneous Manifold Engine (THME)** extension by **Epiphanes**  
> Direct IDE integration for live LLM memory profiling, layer sensitivity analysis, Pareto precision optimization, and real-time Teacher LLM surgery.

---

## Key Capabilities

1. **Native GitHub Copilot Integration (`vscode.lm.registerTool`)**:
   - Equips GitHub Copilot Chat and Copilot Edits with direct tool calling into the THME engine.
   - Copilot can inspect model layers zero-copy, propose Pareto precision strategies, submit surgery recipes, and verify perplexity before/after modifications in real time.

2. **Universal Model Context Protocol (MCP) Server**:
   - Connects any AI assistant (VS Code Copilot, Claude Desktop, Cursor, Windsurf, Antigravity) to THME via standard JSON-RPC (`thme mcp`).

3. **Interactive Sidebar Views** (every value is read from the CLI's own output for the model you pick with **THME: Select Model**; with no model selected each tree says so, and the tool list is labelled as an example — no made-up numbers):
   - **Active Model & Layer Anatomy**: `thme inspect --json` — name, architecture, parameters, layers, hidden size, heads, vocabulary, context, tensor count, raw size.
   - **Pareto Precision Plan**: `thme plan` — original and planned size, compression ratio, quality retention, per-layer formats.
   - **REA-LLM & EWHA Telemetry**: `thme profile` — layer sensitivity and the KV-cache projection.
   - **Copilot Live Tools**: the extension's own registered tools, labelled as an example.

4. **1-Click Native Studio GUI**:
   - Status bar item `$(circuit-board) THME: 60 FPS` launches the native GPU-accelerated desktop cockpit directly from your IDE.

---

## GitHub Copilot Tools

When installed, GitHub Copilot Chat has access to the following live tools:

- `thme_copilot_inspect`: Inspects model architecture, layer count, parameter shapes, and memory footprint.
  - *Example Copilot Prompt*: `"Inspect the current model and tell me how many attention heads and layers it has."`
- `thme_copilot_surgery`: Submits a declarative surgery recipe to restructure layers or apply custom algorithms.
  - *Example Copilot Prompt*: `"Convert Layer 1 to Gated DeltaNet and apply BitNet b1.58 ternary quantization."`
- `thme_copilot_validate`: Tests empirical perplexity degradation against WikiText-2/C4 and checks quality gate ($\Delta PPL \le \epsilon$).
  - *Example Copilot Prompt*: `"Validate the candidate model against baseline PPL 6.50 with a tolerance of 0.25."`

---

## Packaging & Installation

### Option A: Install via VSIX Package
To package the extension into an installable `.vsix` file (needs Node.js 22 or newer):
```bash
cd editors/vscode
npm ci
npm run package
```
Then in VS Code:
1. Open the Extensions view (`Ctrl+Shift+X` / `Cmd+Shift+X`).
2. Click the `...` (Views and More Actions) menu at the top right.
3. Select **Install from VSIX...** and pick the generated `.vsix` file.

### Option B: Universal MCP Server for Any IDE
THME runs as a standalone MCP server conforming to the 2024-11-05 standard.

Add to your `.vscode/mcp.json` or Cursor/Claude MCP configuration:
Build once with `cargo build --release --bin thme`, then point the config at the binary (do not start the server through `cargo run`: it runs build scripts every time):

```json
{
  "servers": {
    "thme-mcp": {
      "command": "${workspaceFolder}/target/release/thme",
      "args": ["mcp", "--allow-root", "${workspaceFolder}"],
      "env": {}
    }
  }
}
```
Or with an installed binary:
```json
{
  "servers": {
    "thme-mcp": {
      "command": "thme",
      "args": ["mcp", "--allow-root", "/path/to/your/models"],
      "env": {}
    }
  }
}
```

---

## Configuration Settings

- `thme.executablePath`: Custom path to the `thme` or `llm-tweaker` binary. Defaults to automatic workspace target resolution.
- `thme.qualityTolerance`: Default quality tolerance epsilon ($\epsilon_Q$) for Pareto precision allocation (default: `0.025`).
- `thme.autoStartMcp`: Automatically starts the MCP server for Copilot upon workspace activation (default: `true`).

---

## License

Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0-only), with commercial
licenses available from Epiphanes. See [LICENSING.md](../../LICENSING.md).
