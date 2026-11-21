// SPDX-License-Identifier: AGPL-3.0-only
// Cooperative-matrix product with f16 weights and f16 activations read in place:
// Y[t, o] = sum_k X[t, k] * W[o, k]. x_to_f16.wgsl has already converted the activations (rows
// padded to a multiple of 16), the weight storage is f16 already, so every 16 x 16 fragment is
// loaded straight from device memory (f16 in, f32 accumulate).
// A workgroup is four subgroups of 32 lanes (2 x 2 grid) and computes a 128 x 128 tile of Y;
// each subgroup owns 64 x 64, sixteen accumulators, so four activation and four weight
// fragments feed sixteen multiplies. The host uses it only on adapters whose subgroup size is
// 32, with in_dim and out_dim multiples of 16, rows above 8 and the output offset a multiple
// of 4.
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
@group(0) @binding(1) var<storage, read> input_x: array<f16>;
@group(0) @binding(2) var<storage, read> weights: array<f16>;
@group(0) @binding(3) var<storage, read_write> output_y: array<vec4<f32>>;

var<workgroup> scratch: array<f32, 1024>;

@compute @workgroup_size(128, 1, 1)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_index) tid: u32
) {
    // Row tiles vary fastest, so the workgroups running together share each weight tile in L2.
    let lin = wid.y * nwg.x + wid.x;
    let row_tiles = nwg.y;
    let o0 = (lin / row_tiles) * 128u;
    let t0 = (lin % row_tiles) * 128u;
    let sg = tid >> 5u;
    let lane = tid & 31u;
    let wm = (sg >> 1u) * 64u;
    let wn = (sg & 1u) * 64u;
    let in_dim = params.in_dim;
    // Fragments past the last real row or output re-read the last real ones; their results are
    // dropped in the epilogue.
    let last_w = params.out_dim - 16u;
    let last_x = (params.rows + 15u) / 16u * 16u - 16u;
    let x0 = min(t0 + wm + 0u, last_x) * in_dim;
    let w0 = min(o0 + wn + 0u, last_w) * in_dim;
    let x1 = min(t0 + wm + 16u, last_x) * in_dim;
    let w1 = min(o0 + wn + 16u, last_w) * in_dim;
    let x2 = min(t0 + wm + 32u, last_x) * in_dim;
    let w2 = min(o0 + wn + 32u, last_w) * in_dim;
    let x3 = min(t0 + wm + 48u, last_x) * in_dim;
    let w3 = min(o0 + wn + 48u, last_w) * in_dim;

    var acc00 = coop_mat16x16<f32,C>();
    var acc01 = coop_mat16x16<f32,C>();
    var acc02 = coop_mat16x16<f32,C>();
    var acc03 = coop_mat16x16<f32,C>();
    var acc10 = coop_mat16x16<f32,C>();
    var acc11 = coop_mat16x16<f32,C>();
    var acc12 = coop_mat16x16<f32,C>();
    var acc13 = coop_mat16x16<f32,C>();
    var acc20 = coop_mat16x16<f32,C>();
    var acc21 = coop_mat16x16<f32,C>();
    var acc22 = coop_mat16x16<f32,C>();
    var acc23 = coop_mat16x16<f32,C>();
    var acc30 = coop_mat16x16<f32,C>();
    var acc31 = coop_mat16x16<f32,C>();
    var acc32 = coop_mat16x16<f32,C>();
    var acc33 = coop_mat16x16<f32,C>();

    for (var k = 0u; k < in_dim; k += 16u) {
        let a0 = coopLoadT<coop_mat16x16<f16,A>>(&input_x[x0 + k], in_dim);
        let a1 = coopLoadT<coop_mat16x16<f16,A>>(&input_x[x1 + k], in_dim);
        let a2 = coopLoadT<coop_mat16x16<f16,A>>(&input_x[x2 + k], in_dim);
        let a3 = coopLoadT<coop_mat16x16<f16,A>>(&input_x[x3 + k], in_dim);
        let b0 = coopLoad<coop_mat16x16<f16,B>>(&weights[w0 + k], in_dim);
        let b1 = coopLoad<coop_mat16x16<f16,B>>(&weights[w1 + k], in_dim);
        let b2 = coopLoad<coop_mat16x16<f16,B>>(&weights[w2 + k], in_dim);
        let b3 = coopLoad<coop_mat16x16<f16,B>>(&weights[w3 + k], in_dim);
        acc00 = coopMultiplyAdd(a0, b0, acc00);
        acc01 = coopMultiplyAdd(a0, b1, acc01);
        acc02 = coopMultiplyAdd(a0, b2, acc02);
        acc03 = coopMultiplyAdd(a0, b3, acc03);
        acc10 = coopMultiplyAdd(a1, b0, acc10);
        acc11 = coopMultiplyAdd(a1, b1, acc11);
        acc12 = coopMultiplyAdd(a1, b2, acc12);
        acc13 = coopMultiplyAdd(a1, b3, acc13);
        acc20 = coopMultiplyAdd(a2, b0, acc20);
        acc21 = coopMultiplyAdd(a2, b1, acc21);
        acc22 = coopMultiplyAdd(a2, b2, acc22);
        acc23 = coopMultiplyAdd(a2, b3, acc23);
        acc30 = coopMultiplyAdd(a3, b0, acc30);
        acc31 = coopMultiplyAdd(a3, b1, acc31);
        acc32 = coopMultiplyAdd(a3, b2, acc32);
        acc33 = coopMultiplyAdd(a3, b3, acc33);
    }

    // Each subgroup writes its sixteen fragments one at a time through its own 16 x 16 scratch.
    let out4 = params.out_dim / 4u;
    let base = sg * 256u;
    for (var f = 0u; f < 16u; f++) {
        switch (f) {
            case 0u: { coopStoreT(acc00, &scratch[base], 16u); }
            case 1u: { coopStoreT(acc01, &scratch[base], 16u); }
            case 2u: { coopStoreT(acc02, &scratch[base], 16u); }
            case 3u: { coopStoreT(acc03, &scratch[base], 16u); }
            case 4u: { coopStoreT(acc10, &scratch[base], 16u); }
            case 5u: { coopStoreT(acc11, &scratch[base], 16u); }
            case 6u: { coopStoreT(acc12, &scratch[base], 16u); }
            case 7u: { coopStoreT(acc13, &scratch[base], 16u); }
            case 8u: { coopStoreT(acc20, &scratch[base], 16u); }
            case 9u: { coopStoreT(acc21, &scratch[base], 16u); }
            case 10u: { coopStoreT(acc22, &scratch[base], 16u); }
            case 11u: { coopStoreT(acc23, &scratch[base], 16u); }
            case 12u: { coopStoreT(acc30, &scratch[base], 16u); }
            case 13u: { coopStoreT(acc31, &scratch[base], 16u); }
            case 14u: { coopStoreT(acc32, &scratch[base], 16u); }
            default: { coopStoreT(acc33, &scratch[base], 16u); }
        }
        workgroupBarrier();
        // 16 rows x 4 vec4: two vec4 per lane.
        let fm = wm + (f >> 2u) * 16u;
        let fn_ = wn + (f & 3u) * 16u;
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
