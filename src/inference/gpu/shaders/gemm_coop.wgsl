// SPDX-License-Identifier: AGPL-3.0-only
// Cooperative-matrix product: Y[t, o] = sum_k X[t, k] * W[o, k]. Tiles of X and W are converted
// to f16 while they are staged in workgroup memory and multiplied by 16 x 16 cooperative
// matrices (f16 in, f32 accumulate). The weights binding and `load_w` come from dense.rs at
// //WEIGHTS_PRELUDE, so f32, f16, Q8_0 and Q4_0 weights all use this one kernel.
// A workgroup is eight subgroups of 32 lanes (2 x 4 grid) and computes a 128 x 128 tile of Y;
// each subgroup owns 64 x 32, eight accumulators. The host uses it only on adapters whose
// subgroup size is 32, and when in_dim, out_dim and the output offset are multiples of 4.
enable f16;
enable wgpu_cooperative_matrix;

struct GemvParams {
    in_dim: u32,
    out_dim: u32,
    num_blocks_per_row: u32,
    rows: u32,
    out_offset: u32,
};

@group(0) @binding(0) var<storage, read> params: GemvParams;
@group(0) @binding(1) var<storage, read> input_x: array<vec4<f32>>;
//WEIGHTS_PRELUDE
@group(0) @binding(3) var<storage, read_write> output_y: array<vec4<f32>>;

const BK: u32 = 32u;
// 32 values per row plus 8 of padding: 80 bytes, a multiple of 16, against bank conflicts.
const LD: u32 = 40u;

var<workgroup> xs: array<f16, 5120>;
var<workgroup> ws: array<f16, 5120>;
var<workgroup> scratch: array<f32, 2048>;

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_index) tid: u32
) {
    let o0 = wid.x * 128u;
    let t0 = wid.y * 128u;
    let sg = tid >> 5u;
    let lane = tid & 31u;
    let wm = (sg >> 2u) * 64u;
    let wn = (sg & 3u) * 32u;
    let in4 = params.in_dim / 4u;

    var acc00 = coop_mat16x16<f32,C>();
    var acc01 = coop_mat16x16<f32,C>();
    var acc10 = coop_mat16x16<f32,C>();
    var acc11 = coop_mat16x16<f32,C>();
    var acc20 = coop_mat16x16<f32,C>();
    var acc21 = coop_mat16x16<f32,C>();
    var acc30 = coop_mat16x16<f32,C>();
    var acc31 = coop_mat16x16<f32,C>();

    for (var k0 = 0u; k0 < in4; k0 += 8u) {
        // 128 rows x 8 vec4 = 1024 vec4 per operand, four per thread.
        for (var j = 0u; j < 4u; j++) {
            let idx = tid + j * 256u;
            let r = idx >> 3u;
            let kq = idx & 7u;
            let k4 = k0 + kq;
            var xv = vec4<f32>(0.0);
            var wv = vec4<f32>(0.0);
            if (k4 < in4) {
                if (t0 + r < params.rows) { xv = input_x[(t0 + r) * in4 + k4]; }
                if (o0 + r < params.out_dim) { wv = load_w((o0 + r) * in4 + k4); }
            }
            let at = r * LD + kq * 4u;
            xs[at] = f16(xv.x);
            xs[at + 1u] = f16(xv.y);
            xs[at + 2u] = f16(xv.z);
            xs[at + 3u] = f16(xv.w);
            ws[at] = f16(wv.x);
            ws[at + 1u] = f16(wv.y);
            ws[at + 2u] = f16(wv.z);
            ws[at + 3u] = f16(wv.w);
        }
        workgroupBarrier();

        for (var ks = 0u; ks < BK; ks += 16u) {
            let a0 = coopLoadT<coop_mat16x16<f16,A>>(&xs[(wm) * LD + ks], LD);
            let a1 = coopLoadT<coop_mat16x16<f16,A>>(&xs[(wm + 16u) * LD + ks], LD);
            let a2 = coopLoadT<coop_mat16x16<f16,A>>(&xs[(wm + 32u) * LD + ks], LD);
            let a3 = coopLoadT<coop_mat16x16<f16,A>>(&xs[(wm + 48u) * LD + ks], LD);
            let b0 = coopLoad<coop_mat16x16<f16,B>>(&ws[(wn) * LD + ks], LD);
            let b1 = coopLoad<coop_mat16x16<f16,B>>(&ws[(wn + 16u) * LD + ks], LD);
            acc00 = coopMultiplyAdd(a0, b0, acc00);
            acc01 = coopMultiplyAdd(a0, b1, acc01);
            acc10 = coopMultiplyAdd(a1, b0, acc10);
            acc11 = coopMultiplyAdd(a1, b1, acc11);
            acc20 = coopMultiplyAdd(a2, b0, acc20);
            acc21 = coopMultiplyAdd(a2, b1, acc21);
            acc30 = coopMultiplyAdd(a3, b0, acc30);
            acc31 = coopMultiplyAdd(a3, b1, acc31);
        }
        workgroupBarrier();
    }

    // Each subgroup writes its sixteen fragments one at a time through its own 16 x 16 scratch.
    let out4 = params.out_dim / 4u;
    let base = sg * 256u;
    for (var f = 0u; f < 8u; f++) {
        switch (f) {
            case 0u: { coopStoreT(acc00, &scratch[base], 16u); }
            case 1u: { coopStoreT(acc01, &scratch[base], 16u); }
            case 2u: { coopStoreT(acc10, &scratch[base], 16u); }
            case 3u: { coopStoreT(acc11, &scratch[base], 16u); }
            case 4u: { coopStoreT(acc20, &scratch[base], 16u); }
            case 5u: { coopStoreT(acc21, &scratch[base], 16u); }
            case 6u: { coopStoreT(acc30, &scratch[base], 16u); }
            default: { coopStoreT(acc31, &scratch[base], 16u); }
        }
        workgroupBarrier();
        // 16 rows x 4 vec4: two vec4 per lane.
        let fm = wm + (f >> 1u) * 16u;
        let fn_ = wn + (f & 1u) * 16u;
        for (var j = 0u; j < 2u; j++) {
            let item = lane + j * 32u;
            let r = item >> 2u;
            let c4 = item & 3u;
            let t = t0 + fm + r;
            let o = o0 + fn_ + c4 * 4u;
            if (t < params.rows && o < params.out_dim) {
                let s = base + r * 16u + c4 * 4u;
                output_y[params.out_offset / 4u + t * out4 + o / 4u] =
                    vec4<f32>(scratch[s], scratch[s + 1u], scratch[s + 2u], scratch[s + 3u]);
            }
        }
        workgroupBarrier();
    }
}
