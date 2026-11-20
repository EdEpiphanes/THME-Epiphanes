// SPDX-License-Identifier: AGPL-3.0-only
// Quantizes the f32 activations of a matrix product to int8 per block of 32 values: one f32
// scale (largest magnitude / 127) per block, then the codes, four signed bytes per word, eight
// words per block. Output layout (u32 words): all scales first (row-major, in_dim / 32 per row),
// then the codes in the same block order. One thread per block. Uses the four bindings of the
// matrix kernels (params at 0, f32 activations at 1, output at 3).

struct GemvParams {
    in_dim: u32,
    out_dim: u32,
    num_blocks_per_row: u32,
    rows: u32,
    out_offset: u32,
};

@group(0) @binding(0) var<storage, read> params: GemvParams;
@group(0) @binding(1) var<storage, read> input_x: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> xq: array<u32>;

fn code(v: f32, inv: f32) -> u32 {
    return u32(i32(clamp(round(v * inv), -127.0, 127.0))) & 255u;
}

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let i = (wid.y * 65535u + wid.x) * 256u + lid;
    let nbw = params.in_dim / 32u;
    let blocks = params.rows * nbw;
    if (i >= blocks) { return; }
    let in4 = params.in_dim / 4u;
    let row = i / nbw;
    let b = i % nbw;
    let base = row * in4 + b * 8u;
    var amax = 0.0;
    for (var c = 0u; c < 8u; c++) {
        let v = abs(input_x[base + c]);
        amax = max(amax, max(max(v.x, v.y), max(v.z, v.w)));
    }
    let d = amax / 127.0;
    var inv = 0.0;
    if (d > 0.0) { inv = 1.0 / d; }
    xq[i] = bitcast<u32>(d);
    for (var c = 0u; c < 8u; c++) {
        let v = input_x[base + c];
        xq[blocks + i * 8u + c] = code(v.x, inv) | (code(v.y, inv) << 8u)
            | (code(v.z, inv) << 16u) | (code(v.w, inv) << 24u);
    }
}
