// SPDX-License-Identifier: AGPL-3.0-only
// Tiled matrix-matrix product for dense F32 weights: Y[t, o] = sum_k X[t, k] * W[o, k].
// One dispatch covers every prompt row. A workgroup computes a 64 x 64 tile of Y with
// 16 x 16 threads (4 x 4 outputs each), staging 16-deep slices of X and W in shared memory.
// Accumulators are four named vec4 values, not an indexed array, so they stay in registers.

struct GemvParams {
    in_dim: u32,
    out_dim: u32,
    num_blocks_per_row: u32,
    rows: u32,
    out_offset: u32,
};

@group(0) @binding(0) var<storage, read> params: GemvParams;
@group(0) @binding(1) var<storage, read> input_x: array<f32>;
@group(0) @binding(2) var<storage, read> weights: array<f32>;
@group(0) @binding(3) var<storage, read_write> output_y: array<f32>;

const TILE: u32 = 64u;
const KSTEP: u32 = 16u;

// Shared slices are stored [k][row / 4] as vec4 so one read feeds four outputs.
var<workgroup> xs: array<vec4<f32>, 256>;
var<workgroup> ws: array<vec4<f32>, 256>;

@compute @workgroup_size(16, 16, 1)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let o0 = wid.x * TILE;
    let t0 = wid.y * TILE;
    let tid = lid.y * 16u + lid.x;
    var acc0 = vec4<f32>(0.0);
    var acc1 = vec4<f32>(0.0);
    var acc2 = vec4<f32>(0.0);
    var acc3 = vec4<f32>(0.0);

    for (var k0 = 0u; k0 < params.in_dim; k0 += KSTEP) {
        for (var i = 0u; i < 4u; i++) {
            let idx = tid + i * 256u;
            let r = idx / KSTEP;
            let c = idx % KSTEP;
            let k = k0 + c;
            var xv = 0.0;
            var wv = 0.0;
            if (k < params.in_dim) {
                if (t0 + r < params.rows) { xv = input_x[(t0 + r) * params.in_dim + k]; }
                if (o0 + r < params.out_dim) { wv = weights[(o0 + r) * params.in_dim + k]; }
            }
            xs[c * 16u + (r >> 2u)][r & 3u] = xv;
            ws[c * 16u + (r >> 2u)][r & 3u] = wv;
        }
        workgroupBarrier();

        for (var kk = 0u; kk < KSTEP; kk++) {
            let a = xs[kk * 16u + lid.y];
            let b = ws[kk * 16u + lid.x];
            acc0 += a.x * b;
            acc1 += a.y * b;
            acc2 += a.z * b;
            acc3 += a.w * b;
        }
        workgroupBarrier();
    }

    let t = t0 + lid.y * 4u;
    let o = o0 + lid.x * 4u;
    let rows_ok = vec4<bool>(t < params.rows, t + 1u < params.rows, t + 2u < params.rows, t + 3u < params.rows);
    for (var j = 0u; j < 4u; j++) {
        if (o + j >= params.out_dim) { continue; }
        if (rows_ok.x) { output_y[params.out_offset + t * params.out_dim + o + j] = acc0[j]; }
        if (rows_ok.y) { output_y[params.out_offset + (t + 1u) * params.out_dim + o + j] = acc1[j]; }
        if (rows_ok.z) { output_y[params.out_offset + (t + 2u) * params.out_dim + o + j] = acc2[j]; }
        if (rows_ok.w) { output_y[params.out_offset + (t + 3u) * params.out_dim + o + j] = acc3[j]; }
    }
}
