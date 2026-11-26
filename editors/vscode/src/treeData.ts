// SPDX-License-Identifier: AGPL-3.0-only
// Pure helpers for the sidebar trees (no vscode import) so they run under `npm test`.
// Rule (BACKLOG THME-3076): no sidebar value is a made-up number. Every number a row
// shows is parsed out of the CLI's own JSON output (`thme inspect --json`, `thme plan`,
// `thme profile`). Rows that are not from the CLI carry no numbers at all, or say so.

export interface TreeRow {
    label: string;
    description: string;
}

/** Where a row's numbers come from. `static` rows must contain no numbers at all. */
export type Provenance = "cli-inspect" | "cli-plan" | "cli-profile" | "static";

export interface SourcedRow {
    row: TreeRow;
    provenance: Provenance;
}

export type LoadState = "none" | "ok" | "failed";

export interface ModelData {
    state: LoadState;
    inspect?: InspectSummary;
    plan?: PlanSummary;
    profile?: ProfileSummary;
}

export interface InspectSummary {
    name: string;
    architecture: string;
    numParameters: number;
    originalByteSize: number;
    numLayers: number;
    hiddenSize: number;
    numAttentionHeads: number;
    numKeyValueHeads: number;
    headDim: number;
    vocabSize: number;
    maxPositionEmbeddings: number;
    tensorCount: number;
}

export interface PlanLayer {
    layerIndex: number;
    format: string;
    originalMb: number;
    optimizedMb: number;
}

export interface PlanSummary {
    modelName: string;
    originalSizeGb: number;
    plannedSizeGb: number;
    compressionRatio: number;
    qualityRetentionPct: number;
    layers: PlanLayer[];
}

export interface ProfileLayer {
    layerIndex: number;
    averageSensitivity: number;
    recommendedFormat: string;
}

export interface KvPoint {
    contextLength: number;
    bytesFp16: number;
    bytesQ4: number;
}

export interface ProfileSummary {
    model: string;
    layers: ProfileLayer[];
    kvCache: KvPoint[];
}

export interface Fixtures {
    inspect: string;
    plan: string;
    profile: string;
}

function row(label: string, description: string): TreeRow {
    return { label, description };
}

/**
 * Numbers that are values, not digits inside a name: `Q4_0`, `INT8`, `SmolLM-135M`,
 * `b1.58` and `thme-hybrid-12l` contribute nothing, while `Layer 0`, `97.295%` and
 * `620 tok/sec` do. Used by the test that checks no sidebar string shows a made-up number.
 */
export interface ValueNumber {
    value: number;
    decimals: number;
}

const IDENT_CHAR = /[A-Za-z0-9_]/;

export function valueNumbers(text: string): ValueNumber[] {
    const out: ValueNumber[] = [];
    const re = /\d+(?:\.\d+)?(?:[eE][+-]?\d+)?/g;
    let m: RegExpExecArray | null;
    while ((m = re.exec(text)) !== null) {
        const before = m.index > 0 ? text[m.index - 1] : undefined;
        const after = text[m.index + m[0].length];
        if (before !== undefined && IDENT_CHAR.test(before)) {
            continue;
        }
        if (after !== undefined && IDENT_CHAR.test(after)) {
            continue;
        }
        const dot = m[0].indexOf(".");
        out.push({ value: Number(m[0]), decimals: dot < 0 ? 0 : m[0].length - dot - 1 });
    }
    return out;
}

/** The CLI's number, rounded for display. Integers stay exact. */
export function formatNumber(n: number): string {
    if (!Number.isFinite(n)) {
        return String(n);
    }
    if (Number.isInteger(n)) {
        return String(n);
    }
    return String(Number(n.toFixed(3)));
}

function asObject(v: unknown): Record<string, unknown> | null {
    return typeof v === "object" && v !== null && !Array.isArray(v) ? (v as Record<string, unknown>) : null;
}

function asString(v: unknown): string | null {
    return typeof v === "string" && v.length > 0 ? v : null;
}

function asFiniteNumber(v: unknown): number | null {
    return typeof v === "number" && Number.isFinite(v) ? v : null;
}

/** serde renders the architecture enum as "Llama" or {"Unknown": "..."}. */
function asArchitecture(v: unknown): string | null {
    if (typeof v === "string" && v.length > 0) {
        return v;
    }
    const o = asObject(v);
    if (!o) {
        return null;
    }
    const keys = Object.keys(o);
    if (keys.length === 1) {
        return asString(o[keys[0]]);
    }
    return null;
}

/** The CLI prints pure JSON for these commands; fall back to the outermost braces. */
function parseJson(text: string): unknown | null {
    try {
        return JSON.parse(text);
    } catch {
        /* not pure JSON */
    }
    const start = text.indexOf("{");
    const end = text.lastIndexOf("}");
    if (start < 0 || end <= start) {
        return null;
    }
    try {
        return JSON.parse(text.slice(start, end + 1));
    } catch {
        return null;
    }
}

export function parseInspectJson(text: string): InspectSummary | null {
    const o = asObject(parseJson(text));
    if (!o) {
        return null;
    }
    const name = asString(o.name);
    const architecture = asArchitecture(o.architecture);
    const numParameters = asFiniteNumber(o.num_parameters);
    const originalByteSize = asFiniteNumber(o.original_byte_size);
    const numLayers = asFiniteNumber(o.num_layers);
    const hiddenSize = asFiniteNumber(o.hidden_size);
    const numAttentionHeads = asFiniteNumber(o.num_attention_heads);
    const numKeyValueHeads = asFiniteNumber(o.num_key_value_heads);
    const headDim = asFiniteNumber(o.head_dim);
    const vocabSize = asFiniteNumber(o.vocab_size);
    const maxPositionEmbeddings = asFiniteNumber(o.max_position_embeddings);
    const tensorCount = Array.isArray(o.tensors) ? o.tensors.length : null;
    const required = [
        name,
        architecture,
        numParameters,
        originalByteSize,
        numLayers,
        hiddenSize,
        numAttentionHeads,
        numKeyValueHeads,
        headDim,
        vocabSize,
        maxPositionEmbeddings,
        tensorCount,
    ];
    if (required.some((v) => v === null)) {
        return null;
    }
    return {
        name: name as string,
        architecture: architecture as string,
        numParameters: numParameters as number,
        originalByteSize: originalByteSize as number,
        numLayers: numLayers as number,
        hiddenSize: hiddenSize as number,
        numAttentionHeads: numAttentionHeads as number,
        numKeyValueHeads: numKeyValueHeads as number,
        headDim: headDim as number,
        vocabSize: vocabSize as number,
        maxPositionEmbeddings: maxPositionEmbeddings as number,
        tensorCount: tensorCount as number,
    };
}

export function parsePlanJson(text: string): PlanSummary | null {
    const o = asObject(parseJson(text));
    if (!o) {
        return null;
    }
    const modelName = asString(o.model_name);
    const originalSizeGb = asFiniteNumber(o.original_size_gb);
    const plannedSizeGb = asFiniteNumber(o.planned_size_gb);
    const compressionRatio = asFiniteNumber(o.compression_ratio);
    const qualityRetentionPct = asFiniteNumber(o.estimated_quality_retention_pct);
    if ([modelName, originalSizeGb, plannedSizeGb, compressionRatio, qualityRetentionPct].some((v) => v === null)) {
        return null;
    }
    const layers: PlanLayer[] = [];
    if (Array.isArray(o.layer_allocations)) {
        for (const item of o.layer_allocations) {
            const layer = asObject(item);
            if (!layer) {
                continue;
            }
            const layerIndex = asFiniteNumber(layer.layer_index);
            const format = asString(layer.assigned_format);
            const originalMb = asFiniteNumber(layer.original_mb);
            const optimizedMb = asFiniteNumber(layer.optimized_mb);
            if ([layerIndex, format, originalMb, optimizedMb].every((v) => v !== null)) {
                layers.push({
                    layerIndex: layerIndex as number,
                    format: format as string,
                    originalMb: originalMb as number,
                    optimizedMb: optimizedMb as number,
                });
            }
        }
    }
    return {
        modelName: modelName as string,
        originalSizeGb: originalSizeGb as number,
        plannedSizeGb: plannedSizeGb as number,
        compressionRatio: compressionRatio as number,
        qualityRetentionPct: qualityRetentionPct as number,
        layers,
    };
}

export function parseProfileJson(text: string): ProfileSummary | null {
    const o = asObject(parseJson(text));
    if (!o) {
        return null;
    }
    const model = asString(o.model);
    const layers: ProfileLayer[] = [];
    if (Array.isArray(o.layers)) {
        for (const item of o.layers) {
            const layer = asObject(item);
            if (!layer) {
                continue;
            }
            const layerIndex = asFiniteNumber(layer.layer_index);
            const averageSensitivity = asFiniteNumber(layer.average_sensitivity);
            const recommendedFormat = asString(layer.recommended_format);
            if ([layerIndex, averageSensitivity, recommendedFormat].every((v) => v !== null)) {
                layers.push({
                    layerIndex: layerIndex as number,
                    averageSensitivity: averageSensitivity as number,
                    recommendedFormat: recommendedFormat as string,
                });
            }
        }
    }
    const kvCache: KvPoint[] = [];
    if (Array.isArray(o.kv_cache)) {
        for (const item of o.kv_cache) {
            const point = asObject(item);
            if (!point) {
                continue;
            }
            const contextLength = asFiniteNumber(point.context_length);
            const bytesFp16 = asFiniteNumber(point.bytes_fp16);
            const bytesQ4 = asFiniteNumber(point.bytes_q4);
            if ([contextLength, bytesFp16, bytesQ4].every((v) => v !== null)) {
                kvCache.push({
                    contextLength: contextLength as number,
                    bytesFp16: bytesFp16 as number,
                    bytesQ4: bytesQ4 as number,
                });
            }
        }
    }
    if (model === null) {
        return null;
    }
    return { model, layers, kvCache };
}

const NO_MODEL_ROW: TreeRow = {
    label: "no model selected",
    description: "run THME: Select Model",
};

const INSPECT_FAILED_ROW: TreeRow = {
    label: "model data unavailable",
    description: "thme inspect --json returned nothing readable; see the THME Engine output",
};

const PLAN_FAILED_ROW: TreeRow = {
    label: "no precision plan yet",
    description: "run THME: Run Pareto Precision AutoDecider for the selected model",
};

const PROFILE_FAILED_ROW: TreeRow = {
    label: "no profile yet",
    description: "thme profile returned nothing readable; see the THME Engine output",
};

export function modelRows(data: ModelData): TreeRow[] {
    if (data.state === "none") {
        return [NO_MODEL_ROW];
    }
    const s = data.inspect;
    if (!s) {
        return [INSPECT_FAILED_ROW];
    }
    return [
        row(`Model: ${s.name}`, "thme inspect --json: name"),
        row(`Architecture: ${s.architecture}`, "thme inspect --json: architecture"),
        row(`Parameters: ${formatNumber(s.numParameters)}`, "num_parameters"),
        row(`Layers: ${formatNumber(s.numLayers)}`, "num_layers"),
        row(`Hidden size: ${formatNumber(s.hiddenSize)}`, "hidden_size"),
        row(`Attention heads: ${formatNumber(s.numAttentionHeads)}`, "num_attention_heads"),
        row(`KV heads: ${formatNumber(s.numKeyValueHeads)}`, "num_key_value_heads"),
        row(`Head dim: ${formatNumber(s.headDim)}`, "head_dim"),
        row(`Vocab: ${formatNumber(s.vocabSize)}`, "vocab_size"),
        row(`Max context: ${formatNumber(s.maxPositionEmbeddings)}`, "max_position_embeddings"),
        row(`Tensors: ${formatNumber(s.tensorCount)}`, "tensors"),
        row(`Raw size: ${formatNumber(s.originalByteSize)} bytes`, "original_byte_size"),
    ];
}

export function decisionRows(data: ModelData): TreeRow[] {
    if (data.state === "none") {
        return [NO_MODEL_ROW];
    }
    const p = data.plan;
    if (!p) {
        return [PLAN_FAILED_ROW];
    }
    const rows = [
        row(`Model: ${p.modelName}`, "thme plan: model_name"),
        row(`Original size: ${formatNumber(p.originalSizeGb)} GB`, "original_size_gb"),
        row(`Planned size: ${formatNumber(p.plannedSizeGb)} GB`, "planned_size_gb"),
        row(`Compression: ${formatNumber(p.compressionRatio)}`, "compression_ratio"),
        row(`Quality retention: ${formatNumber(p.qualityRetentionPct)}%`, "estimated_quality_retention_pct"),
    ];
    for (const layer of p.layers.slice(0, 3)) {
        rows.push(
            row(
                `Layer ${formatNumber(layer.layerIndex)}: ${layer.format}`,
                `${formatNumber(layer.originalMb)} -> ${formatNumber(layer.optimizedMb)} MB`
            )
        );
    }
    return rows;
}

export function telemetryRows(data: ModelData): TreeRow[] {
    if (data.state === "none") {
        return [NO_MODEL_ROW];
    }
    const p = data.profile;
    if (!p) {
        return [PROFILE_FAILED_ROW];
    }
    const rows = [
        row(`Model: ${p.model}`, "thme profile: model"),
        row(`Layers: ${formatNumber(p.layers.length)}`, "layers"),
    ];
    const kv = p.kvCache;
    if (kv.length > 0) {
        const first = kv[0];
        rows.push(row(`KV cache @ ${formatNumber(first.contextLength)} (fp16): ${formatNumber(first.bytesFp16)} bytes`, "kv_cache bytes_fp16"));
        const last = kv[kv.length - 1];
        if (last !== first) {
            rows.push(row(`KV cache @ ${formatNumber(last.contextLength)} (q4): ${formatNumber(last.bytesQ4)} bytes`, "kv_cache bytes_q4"));
        }
    }
    const mostSensitive = p.layers.slice().sort((a, b) => b.averageSensitivity - a.averageSensitivity)[0];
    if (mostSensitive) {
        rows.push(
            row(
                `Most sensitive layer: ${formatNumber(mostSensitive.layerIndex)}`,
                `average_sensitivity ${formatNumber(mostSensitive.averageSensitivity)}`
            )
        );
    }
    return rows;
}

/**
 * The Copilot tree is the extension's own tool list, not a measurement. It is labelled
 * as an example and carries no numbers.
 */
export function copilotRows(): TreeRow[] {
    return [
        row("Example: GitHub Copilot tools", "registered by this extension, not a measurement"),
        row("thme_copilot_inspect", "Inspect model tensors zero-copy"),
        row("thme_copilot_surgery", "Dispatch a surgery recipe"),
        row("thme_copilot_validate", "Check the perplexity gate"),
        row("MCP server", "Started by the THME: Start Live MCP Server command"),
    ];
}

/**
 * Every string the sidebar can show, in every state: no model selected, a model whose
 * CLI output parsed, a model whose CLI output did not parse, and the static tool list.
 * The test walks this list and checks that no row shows a number the CLI did not output.
 */
export function allTreeRows(fixtures: Fixtures): SourcedRow[] {
    const out: SourcedRow[] = [];
    const push = (rows: TreeRow[], provenance: Provenance) => {
        for (const r of rows) {
            out.push({ row: r, provenance });
        }
    };

    const none: ModelData = { state: "none" };
    push(modelRows(none), "static");
    push(decisionRows(none), "static");
    push(telemetryRows(none), "static");

    const ok: ModelData = {
        state: "ok",
        inspect: parseInspectJson(fixtures.inspect) ?? undefined,
        plan: parsePlanJson(fixtures.plan) ?? undefined,
        profile: parseProfileJson(fixtures.profile) ?? undefined,
    };
    push(modelRows(ok), "cli-inspect");
    push(decisionRows(ok), "cli-plan");
    push(telemetryRows(ok), "cli-profile");

    const failed: ModelData = { state: "failed" };
    push(modelRows(failed), "static");
    push(decisionRows(failed), "static");
    push(telemetryRows(failed), "static");

    push(copilotRows(), "static");
    return out;
}
