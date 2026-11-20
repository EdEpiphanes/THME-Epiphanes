// SPDX-License-Identifier: AGPL-3.0-only
// Elementwise and reduction kernels for the device-resident LLaMA-style decoder.
// Every entry point shares one bind group: params (8 u32), three read-only float inputs and
// one read-write float output. Unused slots are bound to a dummy buffer by the host.

@group(0) @binding(0) var<storage, read> p: array<u32, 8>;
@group(0) @binding(1) var<storage, read> in_a: array<f32>;
@group(0) @binding(2) var<storage, read> in_b: array<f32>;
@group(0) @binding(3) var<storage, read> in_c: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;

var<workgroup> red: array<f32, 64>;
var<workgroup> scores: array<f32, 2048>;

fn reduce_sum(tid: u32, v: f32) -> f32 {
    red[tid] = v;
    workgroupBarrier();
    for (var s = 32u; s > 0u; s >>= 1u) {
        if (tid < s) { red[tid] += red[tid + s]; }
        workgroupBarrier();
    }
    let r = red[0];
    workgroupBarrier();
    return r;
}

fn reduce_max(tid: u32, v: f32) -> f32 {
    red[tid] = v;
    workgroupBarrier();
    for (var s = 32u; s > 0u; s >>= 1u) {
        if (tid < s) { red[tid] = max(red[tid], red[tid + s]); }
        workgroupBarrier();
    }
    let r = red[0];
    workgroupBarrier();
    return r;
}

// params: rows, d, eps (f32 bits), input offset in floats. in_a = x, in_b = weight.
@compute @workgroup_size(64)
fn rms_norm(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let d = p[1];
    let row = wid.x;
    let off = p[3] + row * d;
    let tid = lid.x;
    var ss = 0.0;
    for (var i = tid; i < d; i += 64u) {
        let v = in_a[off + i];
        ss += v * v;
    }
    let total = reduce_sum(tid, ss);
    let scale = 1.0 / sqrt(total / f32(d) + bitcast<f32>(p[2]));
    for (var i = tid; i < d; i += 64u) {
        out[row * d + i] = in_a[off + i] * scale * in_b[i];
    }
}

// params: rows, row_len, head_dim, half. in_a = cos, in_b = sin, out = rows updated in place.
@compute @workgroup_size(64)
fn rope(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let idx = (wid.y * 65535u + wid.x) * 64u + lid.x;
    let rows = p[0];
    let row_len = p[1];
    let head_dim = p[2];
    let half = p[3];
    let heads = row_len / head_dim;
    let per_row = heads * half;
    if (idx >= rows * per_row) { return; }
    let row = idx / per_row;
    let r = idx % per_row;
    let head = r / half;
    let j = r % half;
    let base = row * row_len + head * head_dim;
    let a = out[base + j];
    let b = out[base + half + j];
    let c = in_a[row * half + j];
    let s = in_b[row * half + j];
    out[base + j] = a * c - b * s;
    out[base + half + j] = b * c + a * s;
}

// Causal grouped-query attention. params: t_len, heads, kv_heads, head_dim, scale (f32 bits).
// in_a = q, in_b = k, in_c = v; one workgroup per (head, position).
@compute @workgroup_size(64)
fn attention(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let heads = p[1];
    let kv_heads = p[2];
    let hd = p[3];
    let scale = bitcast<f32>(p[4]);
    let h = wid.x;
    let t = wid.y;
    let tid = lid.x;
    let d = heads * hd;
    let kv_dim = kv_heads * hd;
    let kv_off = (h / (heads / kv_heads)) * hd;
    let q_base = t * d + h * hd;

    var local_max = -3.0e38;
    for (var s = tid; s <= t; s += 64u) {
        var dotp = 0.0;
        let k_base = s * kv_dim + kv_off;
        for (var i = 0u; i < hd; i++) { dotp += in_a[q_base + i] * in_b[k_base + i]; }
        let sc = dotp * scale;
        scores[s] = sc;
        local_max = max(local_max, sc);
    }
    let m = reduce_max(tid, local_max);
    var local_sum = 0.0;
    for (var s = tid; s <= t; s += 64u) {
        let e = exp(scores[s] - m);
        scores[s] = e;
        local_sum += e;
    }
    let inv = 1.0 / reduce_sum(tid, local_sum);
    for (var i = tid; i < hd; i += 64u) {
        var acc = 0.0;
        for (var s = 0u; s <= t; s++) {
            acc += scores[s] * in_c[s * kv_dim + kv_off + i];
        }
        out[q_base + i] = acc * inv;
    }
}

// params: n. out = in_a + in_b.
@compute @workgroup_size(64)
fn add(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let idx = (wid.y * 65535u + wid.x) * 64u + lid.x;
    if (idx >= p[0]) { return; }
    out[idx] = in_a[idx] + in_b[idx];
}

// params: n. out = silu(in_a) * in_b.
@compute @workgroup_size(64)
fn swiglu(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let idx = (wid.y * 65535u + wid.x) * 64u + lid.x;
    if (idx >= p[0]) { return; }
    let g = in_a[idx];
    out[idx] = g / (1.0 + exp(-g)) * in_b[idx];
}

// Negative log-likelihood of the next token. params: rows, vocab, first row index.
// in_a = logits [rows, vocab], in_b = target ids (u32 bits) indexed by global row,
// out[global row] = logsumexp(logits) - logits[target].
@compute @workgroup_size(64)
fn nll(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let vocab = p[1];
    let row = wid.x;
    let base = row * vocab;
    let tid = lid.x;
    var lm = -3.0e38;
    for (var i = tid; i < vocab; i += 64u) { lm = max(lm, in_a[base + i]); }
    let m = reduce_max(tid, lm);
    var ls = 0.0;
    for (var i = tid; i < vocab; i += 64u) { ls += exp(in_a[base + i] - m); }
    let total = reduce_sum(tid, ls);
    if (tid == 0u) {
        let g = p[2] + row;
        let target_id = bitcast<u32>(in_b[g]);
        out[g] = m + log(total) - in_a[base + target_id];
    }
}
