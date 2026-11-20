// SPDX-License-Identifier: AGPL-3.0-only
// Pure helpers (no vscode import) so they run under `npm test`.
import * as fs from 'fs';
import { spawn } from 'child_process';

/** Existing regular file, else throws. Arguments are never quoted: spawn gets a raw array. */
export function requireFile(p: unknown, what: string): string {
    if (typeof p !== 'string' || p.length === 0 || p.includes('\0')) {
        throw new Error(`${what}: a file path is required`);
    }
    if (!fs.existsSync(p) || !fs.statSync(p).isFile()) {
        throw new Error(`${what}: not an existing file: ${p}`);
    }
    return p;
}

/** Finite number, optionally with bounds; returned as the string passed to the CLI. */
export function requireNumber(v: unknown, what: string, min = -Infinity, max = Infinity): string {
    if (typeof v !== 'number' || !Number.isFinite(v) || v < min || v > max) {
        throw new Error(`${what}: expected a finite number in [${min}, ${max}], got ${String(v)}`);
    }
    return String(v);
}

export const inspectArgs = (modelPath: unknown): string[] =>
    ["inspect", requireFile(modelPath, "modelPath")];

export const planArgs = (modelPath: unknown, maxDegradation: unknown): string[] =>
    ["plan", requireFile(modelPath, "modelPath"), "--max-degradation", requireNumber(maxDegradation, "maxDegradation", 0)];

export const tweakArgs = (modelPath: unknown, recipePath: unknown): string[] =>
    ["tweak", "--model", requireFile(modelPath, "modelPath"), "--recipe", requireFile(recipePath, "recipePath")];

export const validateArgs = (baseline: unknown, candidate: unknown, maxDelta: unknown = 0.25): string[] => [
    "validate",
    "--baseline-ppl", requireNumber(baseline, "baselinePpl", Number.MIN_VALUE),
    "--candidate-ppl", requireNumber(candidate, "candidatePpl", Number.MIN_VALUE),
    "--max-ppl-delta", requireNumber(maxDelta, "maxDeltaPpl", 0),
];

/** Runs bin with a raw argument array and no shell. Rejects on non-zero exit with stderr. */
export function run(bin: string, args: string[], onData?: (s: string) => void): Promise<string> {
    return new Promise((resolve, reject) => {
        const child = spawn(bin, args, { shell: false });
        let stdout = "";
        let stderr = "";
        child.stdout.on("data", (d) => { const s = d.toString(); stdout += s; onData?.(s); });
        child.stderr.on("data", (d) => { const s = d.toString(); stderr += s; onData?.(s); });
        child.on("error", reject);
        child.on("close", (code) =>
            code === 0 ? resolve(stdout) : reject(new Error(stderr.trim() || stdout.trim() || `Process exited with code ${code}`)));
    });
}
