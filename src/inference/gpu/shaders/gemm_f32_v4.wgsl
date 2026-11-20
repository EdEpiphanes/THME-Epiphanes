// SPDX-License-Identifier: AGPL-3.0-only
// Register-blocked matrix-matrix product for dense F32 weights: Y[t, o] = sum_k X[t, k] * W[o, k].
// A workgroup computes a 128 x 128 tile of Y with 16 x 16 threads, each owning an 8 x 8 block
// (sixteen vec4 accumulators). Slices of X and W, 16 deep, are loaded as vec4 along k and kept in
// shared memory k-major with a padded stride, so each k step reads four vec4 per thread and does
// 64 multiply-adds. The host uses this kernel only when in_dim, out_dim and the output offset are
// multiples of 4; every other shape goes through gemm_f32.wgsl.

struct GemvParams {
    in_dim: u32,
    out_dim: u32,
    num_blocks_per_row: u32,
    rows: u32,
    out_offset: u32,
};

@group(0) @binding(0) var<storage, read> params: GemvParams;
@group(0) @binding(1) var<storage, read> input_x: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> weights: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> output_y: array<vec4<f32>>;

const BK: u32 = 16u;
// 128 values per k row = 32 vec4, plus one vec4 of padding against bank conflicts.
const STRIDE: u32 = 33u;

var<workgroup> xs: array<vec4<f32>, 528>;
var<workgroup> ws: array<vec4<f32>, 528>;

@compute @workgroup_size(16, 16, 1)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let o0 = wid.x * 128u;
    let t0 = wid.y * 128u;
    let tid = lid.y * 16u + lid.x;
    let in4 = params.in_dim / 4u;
    var acc: array<vec4<f32>, 16>;
    for (var i = 0u; i < 16u; i++) { acc[i] = vec4<f32>(0.0); }

    for (var k0 = 0u; k0 < in4; k0 += 4u) {
        for (var j = 0u; j < 2u; j++) {
            let idx = tid + j * 256u;
            let r = idx >> 2u;
            let kq = idx & 3u;
            let k4 = k0 + kq;
            var xv = vec4<f32>(0.0);
            var wv = vec4<f32>(0.0);
            if (k4 < in4) {
                if (t0 + r < params.rows) { xv = input_x[(t0 + r) * in4 + k4]; }
                if (o0 + r < params.out_dim) { wv = weights[(o0 + r) * in4 + k4]; }
            }
            let at = kq * 4u * STRIDE + (r >> 2u);
            let lane = r & 3u;
            xs[at][lane] = xv.x;
            xs[at + STRIDE][lane] = xv.y;
            xs[at + 2u * STRIDE][lane] = xv.z;
            xs[at + 3u * STRIDE][lane] = xv.w;
            ws[at][lane] = wv.x;
            ws[at + STRIDE][lane] = wv.y;
            ws[at + 2u * STRIDE][lane] = wv.z;
            ws[at + 3u * STRIDE][lane] = wv.w;
        }
        workgroupBarrier();

        for (var kk = 0u; kk < BK; kk++) {
            let base = kk * STRIDE;
            let a0 = xs[base + lid.y * 2u];
            let a1 = xs[base + lid.y * 2u + 1u];
            let b0 = ws[base + lid.x * 2u];
            let b1 = ws[base + lid.x * 2u + 1u];
            acc[0] += a0.x * b0;  acc[1] += a0.x * b1;
            acc[2] += a0.y * b0;  acc[3] += a0.y * b1;
            acc[4] += a0.z * b0;  acc[5] += a0.z * b1;
            acc[6] += a0.w * b0;  acc[7] += a0.w * b1;
            acc[8] += a1.x * b0;  acc[9] += a1.x * b1;
            acc[10] += a1.y * b0; acc[11] += a1.y * b1;
            acc[12] += a1.z * b0; acc[13] += a1.z * b1;
            acc[14] += a1.w * b0; acc[15] += a1.w * b1;
        }
        workgroupBarrier();
    }

    let out4 = params.out_dim / 4u;
    let o = o0 + lid.x * 8u;
    for (var r = 0u; r < 8u; r++) {
        let t = t0 + lid.y * 8u + r;
        if (t >= params.rows) { continue; }
        let at = params.out_offset / 4u + t * out4 + o / 4u;
        if (o < params.out_dim) { output_y[at] = acc[r * 2u]; }
        if (o + 4u < params.out_dim) { output_y[at + 1u] = acc[r * 2u + 1u]; }
    }
}
