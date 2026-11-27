// SPDX-License-Identifier: AGPL-3.0-only
import * as vscode from 'vscode';
import * as path from 'path';
import * as fs from 'fs';
import { spawn } from 'child_process';
import { inspectArgs, planArgs, tweakArgs, validateArgs, mcpArgs, run } from './thme';
import {
    LoadState,
    ModelData,
    copilotRows,
    decisionRows,
    modelRows,
    parseInspectJson,
    parsePlanJson,
    parseProfileJson,
    telemetryRows,
    TreeRow,
} from './treeData';

let outputChannel: vscode.OutputChannel;
let statusBarItem: vscode.StatusBarItem;

// The sidebar shows no made-up numbers: every value comes from the CLI's own output for
// the model the user selected, or the tree says "no model selected".
let selectedModel: string | undefined;
let modelData: ModelData = { state: "none" };
let dataDir = "";
let modelProvider: ThmeTreeProvider;
let decisionProvider: ThmeTreeProvider;
let telemetryProvider: ThmeTreeProvider;

export function activate(context: vscode.ExtensionContext) {
    outputChannel = vscode.window.createOutputChannel("THME Engine");
    outputChannel.appendLine("[THME] Extension activated. Initializing AI & Copilot bridge...");
    dataDir = context.globalStorageUri.fsPath;
    fs.mkdirSync(dataDir, { recursive: true });

    // 1. Status Bar Item
    statusBarItem = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Right, 100);
    statusBarItem.command = "thme.launchStudio";
    statusBarItem.text = "$(circuit-board) THME: 60 FPS";
    statusBarItem.tooltip = "Topological Heterogeneous Manifold Engine (Click to open Studio GUI)";
    statusBarItem.show();
    context.subscriptions.push(statusBarItem);

    // 2. Sidebar Tree Views
    modelProvider = new ThmeTreeProvider(() => modelRows(modelData));
    decisionProvider = new ThmeTreeProvider(() => decisionRows(modelData));
    telemetryProvider = new ThmeTreeProvider(() => telemetryRows(modelData));
    const copilotProvider = new ThmeTreeProvider(() => copilotRows());

    vscode.window.registerTreeDataProvider("thme-models", modelProvider);
    vscode.window.registerTreeDataProvider("thme-decisions", decisionProvider);
    vscode.window.registerTreeDataProvider("thme-telemetry", telemetryProvider);
    vscode.window.registerTreeDataProvider("thme-copilot", copilotProvider);
    context.subscriptions.push(modelProvider, decisionProvider, telemetryProvider, copilotProvider);

    // 3. Register Commands
    context.subscriptions.push(
        vscode.commands.registerCommand("thme.selectModel", () => selectModel()),
        vscode.commands.registerCommand("thme.refreshSidebar", () => refreshModelData()),
        vscode.commands.registerCommand("thme.launchStudio", () => launchStudio(context)),
        vscode.commands.registerCommand("thme.inspectModel", () => inspectModel(context)),
        vscode.commands.registerCommand("thme.autoDecide", () => autoDecide(context)),
        vscode.commands.registerCommand("thme.validateQuality", () => validateQuality(context)),
        vscode.commands.registerCommand("thme.applySurgery", () => applySurgery(context)),
        vscode.commands.registerCommand("thme.startMcpServer", () => startMcpServer(context))
    );

    // 4. Register GitHub Copilot Language Model Tools (VS Code 1.146+ API)
    registerCopilotTools(context);

    outputChannel.appendLine("[THME] System ready. GitHub Copilot tools registered.");
}

export function deactivate() {
    if (statusBarItem) {
        statusBarItem.dispose();
    }
}

/**
 * Resolves the path to the THME binary
 */
function resolveBinaryPath(preferredBinary: string = "thme"): string {
    const config = vscode.workspace.getConfiguration("thme");
    const configuredPath = config.get<string>("executablePath");
    if (configuredPath && fs.existsSync(configuredPath)) {
        return configuredPath;
    }

    const isWindows = process.platform === "win32";
    const exeName = isWindows ? `${preferredBinary}.exe` : preferredBinary;

    // Check workspace target folders
    const workspaceFolders = vscode.workspace.workspaceFolders;
    if (workspaceFolders) {
        for (const folder of workspaceFolders) {
            const releasePath = path.join(folder.uri.fsPath, "target", "release", exeName);
            if (fs.existsSync(releasePath)) {
                return releasePath;
            }
            const debugPath = path.join(folder.uri.fsPath, "target", "debug", exeName);
            if (fs.existsSync(debugPath)) {
                return debugPath;
            }
        }
    }

    // Fallback to binary in PATH
    return preferredBinary;
}

/**
 * Executes a CLI command (no shell, raw argument array) and streams output to channel
 */
function runThmeCommand(args: string[], opts: { reveal?: boolean } = {}): Promise<string> {
    const bin = resolveBinaryPath();
    outputChannel.appendLine(`[THME CMD] ${bin} ${JSON.stringify(args)}`);
    if (opts.reveal !== false) {
        outputChannel.show(true);
    }
    return run(bin, args, (s) => outputChannel.append(s));
}

/** Runs a CLI command without stealing focus; null when it fails. */
async function tryThmeCommand(args: string[]): Promise<string | null> {
    try {
        return await runThmeCommand(args, { reveal: false });
    } catch (err: any) {
        outputChannel.appendLine(`[THME] ${args[0]} failed: ${err.message}`);
        return null;
    }
}

function readTextFile(p: string): string | null {
    try {
        return fs.readFileSync(p, "utf8");
    } catch {
        return null;
    }
}

/**
 * Reads the selected model through the CLI. Every sidebar number comes from these three
 * commands; nothing here is a constant.
 */
async function loadModelData(model: string): Promise<ModelData> {
    const planPath = path.join(dataDir, "optimization_plan.json");
    const profilePath = path.join(dataDir, "profile.json");

    const inspectText = await tryThmeCommand(["inspect", model, "--json"]);
    const planRan = await tryThmeCommand(["plan", model, "--output", planPath]);
    const profileRan = await tryThmeCommand(["profile", model, "--output", profilePath]);

    const inspect = inspectText === null ? null : parseInspectJson(inspectText);
    const plan = planRan === null ? null : parsePlanJson(readTextFile(planPath) ?? "");
    const profile = profileRan === null ? null : parseProfileJson(readTextFile(profilePath) ?? "");

    const state: LoadState = inspect !== null || plan !== null || profile !== null ? "ok" : "failed";
    return { state, inspect: inspect ?? undefined, plan: plan ?? undefined, profile: profile ?? undefined };
}

async function refreshModelData(): Promise<void> {
    modelData = selectedModel === undefined ? { state: "none" } : await loadModelData(selectedModel);
    modelProvider.refresh();
    decisionProvider.refresh();
    telemetryProvider.refresh();
}

async function selectModel(): Promise<void> {
    const picked = await pickFile("Select Model", ["safetensors", "gguf"]);
    if (!picked) {
        return;
    }
    selectedModel = picked;
    await refreshModelData();
}

async function pickFile(label: string, ext: string[]): Promise<string | undefined> {
    const u = await vscode.window.showOpenDialog({ canSelectMany: false, openLabel: label, filters: { Files: ext } });
    return u?.[0]?.fsPath;
}

function launchStudio(context: vscode.ExtensionContext) {
    const guiBin = resolveBinaryPath("thme-gui");
    outputChannel.appendLine(`[THME] Launching Native Desktop Studio GUI: ${guiBin}`);
    vscode.window.showInformationMessage("Launching THME Native Studio GUI (60 FPS)...");
    spawn(guiBin, [], { detached: true, stdio: "ignore", shell: false }).unref();
}

async function inspectModel(context: vscode.ExtensionContext) {
    const model = await pickFile("Inspect Model", ["safetensors", "gguf"]);
    if (!model) { return; }
    selectedModel = model;
    await refreshModelData();
    outputChannel.show(true);
    vscode.window.showInformationMessage("thme inspect, plan and profile ran for the selected model. See THME Engine output and the sidebar.");
}

async function autoDecide(context: vscode.ExtensionContext) {
    const model = await pickFile("Plan Model", ["safetensors", "gguf"]);
    if (!model) { return; }
    try {
        await runThmeCommand(planArgs(model, 2.0));
        vscode.window.showInformationMessage("thme plan exited 0. See THME Engine output and optimization_plan.json.");
    } catch (err: any) {
        vscode.window.showErrorMessage(`Plan generation failed: ${err.message}`);
    }
}

async function askNumber(prompt: string): Promise<number | undefined> {
    const v = await vscode.window.showInputBox({ prompt, validateInput: (t) => (t.trim() !== "" && Number.isFinite(Number(t)) && Number(t) > 0 ? undefined : "Enter a positive number") });
    return v === undefined ? undefined : Number(v);
}

async function validateQuality(context: vscode.ExtensionContext) {
    // Perplexities must come from a real measurement; the CLI's own exit status is the verdict.
    const baseline = await askNumber("Measured baseline perplexity");
    if (baseline === undefined) { return; }
    const candidate = await askNumber("Measured candidate perplexity");
    if (candidate === undefined) { return; }
    try {
        await runThmeCommand(validateArgs(baseline, candidate));
        vscode.window.showInformationMessage("thme validate exited 0: the CLI reports the quality gate as passed for the values you entered.");
    } catch (err: any) {
        vscode.window.showErrorMessage(`thme validate reported a failure: ${err.message}`);
    }
}

async function applySurgery(context: vscode.ExtensionContext) {
    const model = await pickFile("Select Model", ["safetensors", "gguf"]);
    if (!model) { return; }
    const recipe = await pickFile("Select Surgery Recipe", ["json"]);
    if (!recipe) { return; }
    try {
        await runThmeCommand(tweakArgs(model, recipe));
        vscode.window.showInformationMessage("thme tweak exited 0. See THME Engine output for what it did.");
    } catch (err: any) {
        vscode.window.showErrorMessage(`Surgery failed: ${err.message}`);
    }
}

function startMcpServer(context: vscode.ExtensionContext) {
    const bin = resolveBinaryPath();
    const roots = (vscode.workspace.workspaceFolders ?? []).map((f) => f.uri.fsPath);
    const terminal = vscode.window.createTerminal({ name: "THME MCP Daemon", shellPath: bin, shellArgs: mcpArgs(roots) });
    terminal.show();
    vscode.window.showInformationMessage("Started `thme mcp` in a terminal (stdin/stdout JSON-RPC).");
}

/**
 * Registers GitHub Copilot Language Model Tools API (`vscode.lm.registerTool`)
 */
function registerCopilotTools(context: vscode.ExtensionContext) {
    const lm = (vscode as any).lm;
    if (!lm || typeof lm.registerTool !== "function") {
        outputChannel.appendLine("[THME] Note: vscode.lm.registerTool not available in this VS Code version. Using standard MCP bridge.");
        return;
    }

    // Tool 1: Inspect
    try {
        context.subscriptions.push(
            lm.registerTool("thme_copilot_inspect", {
                invoke: async (options: any) => {
                    const result = await runThmeCommand(inspectArgs(options.input?.modelPath));
                    return {
                        content: [{ type: "text", value: result }]
                    };
                }
            })
        );

        // Tool 2: Surgery
        context.subscriptions.push(
            lm.registerTool("thme_copilot_surgery", {
                invoke: async (options: any) => {
                    const recipeJson = options.input?.recipeJson;
                    if (!recipeJson) {
                        return { content: [{ type: "text", value: "Error: recipeJson required" }] };
                    }
                    const tmpFile = path.join(context.globalStorageUri.fsPath, "copilot_surgery.json");
                    fs.mkdirSync(context.globalStorageUri.fsPath, { recursive: true });
                    fs.writeFileSync(tmpFile, recipeJson, "utf8");
                    const modelPath = options.input?.modelPath;
                    const result = await runThmeCommand(tweakArgs(modelPath, tmpFile));
                    return {
                        content: [{ type: "text", value: result }]
                    };
                }
            })
        );

        // Tool 3: Validate
        context.subscriptions.push(
            lm.registerTool("thme_copilot_validate", {
                invoke: async (options: any) => {
                    const { baselinePpl, candidatePpl, maxDeltaPpl = 0.25 } = options.input || {};
                    const result = await runThmeCommand([...validateArgs(baselinePpl, candidatePpl, maxDeltaPpl), "--json"]);
                    return {
                        content: [{ type: "text", value: result }]
                    };
                }
            })
        );

        outputChannel.appendLine("[THME] GitHub Copilot Language Model Tools registered successfully.");
    } catch (e: any) {
        outputChannel.appendLine(`[THME] Warning registering LM tools: ${e.message}`);
    }
}

// ========================
// Sidebar Tree Data Providers
// ========================

function toItems(rows: TreeRow[]): vscode.TreeItem[] {
    return rows.map((r) => new ThmeItem(r.label, r.description, vscode.TreeItemCollapsibleState.None));
}

/** One provider per view; the rows come from treeData, which reads the CLI's own output. */
class ThmeTreeProvider implements vscode.TreeDataProvider<vscode.TreeItem> {
    private readonly emitter = new vscode.EventEmitter<void>();
    readonly onDidChangeTreeData: vscode.Event<void> = this.emitter.event;

    constructor(private readonly rows: () => TreeRow[]) { }

    refresh(): void {
        this.emitter.fire();
    }

    dispose(): void {
        this.emitter.dispose();
    }

    getTreeItem(element: vscode.TreeItem): vscode.TreeItem {
        return element;
    }

    getChildren(): Thenable<vscode.TreeItem[]> {
        return Promise.resolve(toItems(this.rows()));
    }
}

class ThmeItem extends vscode.TreeItem {
    constructor(
        public readonly label: string,
        private readonly desc: string,
        public readonly collapsibleState: vscode.TreeItemCollapsibleState
    ) {
        super(label, collapsibleState);
        this.description = this.desc;
    }
}
