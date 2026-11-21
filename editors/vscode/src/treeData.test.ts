// SPDX-License-Identifier: AGPL-3.0-only
import { test } from 'node:test';
import * as assert from 'node:assert/strict';
import * as fs from 'fs';
import * as path from 'path';
import {
    allTreeRows,
    copilotRows,
    decisionRows,
    modelRows,
    parseInspectJson,
    parsePlanJson,
    parseProfileJson,
    telemetryRows,
    valueNumbers,
} from './treeData';

// The fixtures are real CLI output: `thme inspect --json`, `thme plan` and `thme profile`
// on examples/mock_llama (inspect trimmed to two tensors). See the report for the command.
const fixturesDir = path.join(__dirname, '..', 'src', 'fixtures');

function readFixture(name: string): string {
    return fs.readFileSync(path.join(fixturesDir, name), 'utf8');
}

function loadFixtures() {
    return {
        inspect: readFixture('inspect.json'),
        plan: readFixture('plan.json'),
        profile: readFixture('profile.json'),
    };
}

test('the committed fixtures parse into the fields the sidebar shows', () => {
    const f = loadFixtures();
    const inspect = parseInspectJson(f.inspect);
    assert.ok(inspect);
    assert.equal(inspect.name, 'mock_llama');
    assert.equal(inspect.architecture, 'Llama');
    assert.equal(inspect.numLayers, 4);
    assert.equal(inspect.hiddenSize, 256);
    assert.equal(inspect.numParameters, 12124416);
    assert.equal(inspect.tensorCount, 2);

    const plan = parsePlanJson(f.plan);
    assert.ok(plan);
    assert.equal(plan.modelName, 'mock_llama');
    assert.equal(plan.layers.length, 4);
    assert.equal(plan.layers[0].format, 'Q4_0');

    const profile = parseProfileJson(f.profile);
    assert.ok(profile);
    assert.equal(profile.model, 'mock_llama');
    assert.equal(profile.layers.length, 4);
    assert.equal(profile.kvCache.length, 8);
});

test('unreadable or incomplete CLI output is rejected, not partly shown', () => {
    assert.equal(parseInspectJson(''), null);
    assert.equal(parseInspectJson('not json at all'), null);
    assert.equal(parseInspectJson('{"name":"x"}'), null);
    assert.equal(parseInspectJson('{"name":"x","architecture":"Llama"}'), null);
    assert.equal(parsePlanJson('[]'), null);
    assert.equal(parsePlanJson('{"model_name":"x"}'), null);
    assert.equal(parseProfileJson('{"layers":[]}'), null);
    // Surrounding noise is tolerated, the outermost JSON object is read.
    assert.ok(parseInspectJson(`noise before ${loadFixtures().inspect} noise after`));
});

test('every string the tree can show is number-free or a number the CLI output contains', () => {
    const fixtures = loadFixtures();
    const rows = allTreeRows(fixtures);
    assert.ok(rows.length >= 20, `expected a full enumeration, got ${rows.length}`);

    const cliNumbers = valueNumbers(`${fixtures.inspect}\n${fixtures.plan}\n${fixtures.profile}`);
    assert.ok(cliNumbers.length > 0);

    const problems: string[] = [];
    for (const { row, provenance } of rows) {
        const text = `${row.label} ${row.description}`;
        for (const shown of valueNumbers(text)) {
            if (provenance === 'static') {
                problems.push(`static row shows a number: "${text}" (${shown.value})`);
                continue;
            }
            // The sidebar rounds to three decimals; allow half a unit in the last place.
            const tolerance = 0.5 * Math.pow(10, -shown.decimals) + 1e-9;
            const found = cliNumbers.some((n) => Math.abs(n.value - shown.value) <= tolerance);
            if (!found) {
                problems.push(`"${text}" shows ${shown.value}, which the CLI output does not contain`);
            }
        }
    }
    assert.deepEqual(problems, []);
});

test('the made-up example strings are gone', () => {
    const texts = allTreeRows(loadFixtures()).map(({ row }) => `${row.label} ${row.description}`);
    for (const madeUp of ['thme-hybrid-12l', '620 tok/sec', '603.9M', '1.61 ms/token', '80% Exit Fraction']) {
        assert.equal(
            texts.some((t) => t.includes(madeUp)),
            false,
            `the tree still shows the made-up string "${madeUp}"`
        );
    }
});

test('a tree with no model says so instead of showing a placeholder', () => {
    const none = { state: 'none' as const };
    for (const rows of [modelRows(none), decisionRows(none), telemetryRows(none)]) {
        assert.equal(rows.length, 1);
        assert.match(rows[0].label, /no model selected/);
        assert.deepEqual(valueNumbers(`${rows[0].label} ${rows[0].description}`), []);
    }
});

test('the Copilot tree is labelled as an example and carries no numbers', () => {
    const rows = copilotRows();
    assert.ok(rows.length > 0);
    assert.match(rows[0].label, /example/i);
    for (const r of rows) {
        assert.deepEqual(valueNumbers(`${r.label} ${r.description}`), [], `number in static row: ${r.label}`);
    }
});
