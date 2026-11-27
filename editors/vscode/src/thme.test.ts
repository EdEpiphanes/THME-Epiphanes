// SPDX-License-Identifier: AGPL-3.0-only
import { test } from 'node:test';
import * as assert from 'node:assert/strict';
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import { inspectArgs, planArgs, tweakArgs, validateArgs, mcpArgs, requireNumber, run } from './thme';

const dir = fs.mkdtempSync(path.join(os.tmpdir(), "thme-ext-"));
// Quotes and an ampersand are legal in file names on Linux and macOS; on Windows only the ampersand is.
const name = process.platform === "win32" ? "a & b $(x) `y`.gguf" : `a "q" 'r' & b $(x) \`y\`;.gguf`;
const model = path.join(dir, name);
fs.writeFileSync(model, "x");
test.after(() => fs.rmSync(dir, { recursive: true, force: true }));

test("arguments stay raw: no quoting added", () => {
    assert.deepEqual(inspectArgs(model), ["inspect", model]);
    assert.deepEqual(tweakArgs(model, model), ["tweak", "--model", model, "--recipe", model]);
    assert.deepEqual(planArgs(model, 2), ["plan", model, "--max-degradation", "2"]);
});

test("a hostile path reaches the child intact and runs nothing", async () => {
    const out = await run(process.execPath, ["-e", "console.log(JSON.stringify(process.argv[1]))", model]);
    assert.equal(JSON.parse(out.trim()), model);
});

test("modelPath must be an existing file", () => {
    assert.throws(() => inspectArgs(path.join(dir, "missing.gguf")), /not an existing file/);
    assert.throws(() => inspectArgs(dir), /not an existing file/);
    assert.throws(() => inspectArgs("x\0y"), /file path is required/);
    assert.throws(() => inspectArgs(undefined), /file path is required/);
});

test("numeric inputs are validated", () => {
    assert.deepEqual(validateArgs(6.5, 6.65).slice(0, 2), ["validate", "--baseline-ppl"]);
    for (const bad of ["1; calc", NaN, Infinity, -1, undefined]) {
        assert.throws(() => validateArgs(bad, 6.65), /expected a finite number/);
    }
    assert.throws(() => validateArgs(6.5, 6.65, -0.1), /maxDeltaPpl/);
    assert.throws(() => requireNumber(5, "x", 0, 1), /expected/);
});

test("run rejects with stderr on non-zero exit", async () => {
    await assert.rejects(run(process.execPath, ["-e", "console.error('boom');process.exit(3)"]), /boom/);
});

test("the MCP server is started with one --allow-root per workspace folder", () => {
    assert.deepEqual(mcpArgs([]), ["mcp"]);
    assert.deepEqual(mcpArgs(["/w/one", "/w/two"]), ["mcp", "--allow-root", "/w/one", "--allow-root", "/w/two"]);
});
