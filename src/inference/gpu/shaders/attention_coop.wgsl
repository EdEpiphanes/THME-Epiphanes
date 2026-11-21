// SPDX-License-Identifier: AGPL-3.0-only
// Causal grouped-query attention on cooperative matrices, head size 64. One workgroup of four
// subgroups (32 lanes each) handles 64 consecutive queries of one head, a subgroup 16 of them.
// Keys and values go through workgroup memory in tiles of 32, converted to f16; scores and
// outputs are 16 x 16 f32 cooperative matrices (f16 in, f32 accumulate).
// Two passes over the key tiles replace the running rescale of an online softmax: pass one
// finds the maximum score of each query, pass two recomputes the scores, forms exp(score - max)
// in f32, sums them, and multiplies the f16 weights with the values. Lanes work on whole score
// rows through a per-subgroup scratch: the lane pair (2r, 2r + 1) owns query row r.
// Bindings match decoder_ops.wgsl: params (t_len, heads, kv_heads, head_dim, scale bits), q, k,
// v, and the output. Heavy (late) query blocks are scheduled first.
enable f16;
enable wgpu_cooperative_matrix;

@group(0) @binding(0) var<storage, read> p: array<u32, 8>;
@group(0) @binding(1) var<storage, read> q_in: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> k_in: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> v_in: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> out: array<vec4<f32>>;

const HD4: u32 = 16u;
// 64 values per row plus 8 of padding: 144 bytes, a multiple of 16, against bank conflicts.
const LD: u32 = 72u;
const KT: u32 = 32u;
// Probability tile row: 32 keys plus 8 of padding.
const PLD: u32 = 40u;

var<workgroup> qs: array<f16, 4608>;
var<workgroup> ks: array<f16, 2304>;
var<workgroup> vs: array<f16, 2304>;
var<workgroup> ss: array<f32, 2048>;
var<workgroup> ps: array<f16, 2560>;
var<workgroup> red: array<f32, 128>;

fn stage_k(tid: u32, k0: u32, t_len: u32, kv4: u32, kvh: u32) {
    // 32 keys x 16 vec4 = 512 vec4, four per thread.
    for (var j = 0u; j < 4u; j++) {
        let idx = tid + j * 128u;
        let key = idx >> 4u;
        let c4 = idx & 15u;
        let kk = k0 + key;
        var x = vec4<f32>(0.0);
        if (kk < t_len) { x = k_in[kk * kv4 + kvh * HD4 + c4]; }
        let at = key * LD + c4 * 4u;
        ks[at] = f16(x.x);
        ks[at + 1u] = f16(x.y);
        ks[at + 2u] = f16(x.z);
        ks[at + 3u] = f16(x.w);
    }
}

fn stage_v(tid: u32, k0: u32, t_len: u32, kv4: u32, kvh: u32) {
    for (var j = 0u; j < 4u; j++) {
        let idx = tid + j * 128u;
        let key = idx >> 4u;
        let c4 = idx & 15u;
        let kk = k0 + key;
        var x = vec4<f32>(0.0);
        if (kk < t_len) { x = v_in[kk * kv4 + kvh * HD4 + c4]; }
        let at = key * LD + c4 * 4u;
        vs[at] = f16(x.x);
        vs[at + 1u] = f16(x.y);
        vs[at + 2u] = f16(x.z);
        vs[at + 3u] = f16(x.w);
    }
}

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) tid: u32) {
    let t_len = p[0];
    let heads = p[1];
    let kv_heads = p[2];
    let scale = bitcast<f32>(p[4]);
    let blocks = (t_len + 63u) / 64u;
    let qb = blocks - 1u - wid.x;
    let h = wid.y;
    let q4 = heads * HD4;
    let kv4 = kv_heads * HD4;
    let kvh = h / (heads / kv_heads);
    let sg = tid >> 5u;
    let lane = tid & 31u;
    let r = lane >> 1u;
    let half = lane & 1u;
    let qrow = qb * 64u + sg * 16u + r;
    let live = qrow < t_len;
    let kend = min(qb * 64u + 64u, t_len);

    // Queries of the block as f16 (zero past the end).
    for (var j = 0u; j < 8u; j++) {
        let idx = tid + j * 128u;
        let row = idx >> 4u;
        let c4 = idx & 15u;
        let qi = qb * 64u + row;
        var x = vec4<f32>(0.0);
        if (qi < t_len) { x = q_in[qi * q4 + h * HD4 + c4]; }
        let at = row * LD + c4 * 4u;
        qs[at] = f16(x.x);
        qs[at + 1u] = f16(x.y);
        qs[at + 2u] = f16(x.z);
        qs[at + 3u] = f16(x.w);
    }
    workgroupBarrier();
    let qa0 = coopLoadT<coop_mat16x16<f16,A>>(&qs[sg * 16u * LD], LD);
    let qa1 = coopLoadT<coop_mat16x16<f16,A>>(&qs[sg * 16u * LD + 16u], LD);
    let qa2 = coopLoadT<coop_mat16x16<f16,A>>(&qs[sg * 16u * LD + 32u], LD);
    let qa3 = coopLoadT<coop_mat16x16<f16,A>>(&qs[sg * 16u * LD + 48u], LD);
    let sbase = sg * 512u;

    // Pass one: the largest scaled score of each query row.
    var m = -3.0e38;
    for (var k0 = 0u; k0 < kend; k0 += KT) {
        stage_k(tid, k0, t_len, kv4, kvh);
        workgroupBarrier();
        var s0 = coop_mat16x16<f32,C>();
        var s1 = coop_mat16x16<f32,C>();
        s0 = coopMultiplyAdd(qa0, coopLoad<coop_mat16x16<f16,B>>(&ks[0u], LD), s0);
        s1 = coopMultiplyAdd(qa0, coopLoad<coop_mat16x16<f16,B>>(&ks[16u * LD], LD), s1);
        s0 = coopMultiplyAdd(qa1, coopLoad<coop_mat16x16<f16,B>>(&ks[16u], LD), s0);
        s1 = coopMultiplyAdd(qa1, coopLoad<coop_mat16x16<f16,B>>(&ks[16u * LD + 16u], LD), s1);
        s0 = coopMultiplyAdd(qa2, coopLoad<coop_mat16x16<f16,B>>(&ks[32u], LD), s0);
        s1 = coopMultiplyAdd(qa2, coopLoad<coop_mat16x16<f16,B>>(&ks[16u * LD + 32u], LD), s1);
        s0 = coopMultiplyAdd(qa3, coopLoad<coop_mat16x16<f16,B>>(&ks[48u], LD), s0);
        s1 = coopMultiplyAdd(qa3, coopLoad<coop_mat16x16<f16,B>>(&ks[16u * LD + 48u], LD), s1);
        coopStoreT(s0, &ss[sbase], 32u);
        coopStoreT(s1, &ss[sbase + 16u], 32u);
        workgroupBarrier();
        for (var jj = 0u; jj < 16u; jj++) {
            let col = half * 16u + jj;
            let key = k0 + col;
            if (live && key <= qrow && key < t_len) {
                m = max(m, ss[sbase + r * 32u + col] * scale);
            }
        }
        workgroupBarrier();
    }
    red[tid] = m;
    workgroupBarrier();
    m = max(m, red[tid ^ 1u]);
    workgroupBarrier();

    // Pass two: weights, their sum, and the weighted values.
    var l = 0.0;
    var o0 = coop_mat16x16<f32,C>();
    var o1 = coop_mat16x16<f32,C>();
    var o2 = coop_mat16x16<f32,C>();
    var o3 = coop_mat16x16<f32,C>();
    let pbase = sg * 16u * PLD;
    for (var k0 = 0u; k0 < kend; k0 += KT) {
        stage_k(tid, k0, t_len, kv4, kvh);
        stage_v(tid, k0, t_len, kv4, kvh);
        workgroupBarrier();
        var s0 = coop_mat16x16<f32,C>();
        var s1 = coop_mat16x16<f32,C>();
        s0 = coopMultiplyAdd(qa0, coopLoad<coop_mat16x16<f16,B>>(&ks[0u], LD), s0);
        s1 = coopMultiplyAdd(qa0, coopLoad<coop_mat16x16<f16,B>>(&ks[16u * LD], LD), s1);
        s0 = coopMultiplyAdd(qa1, coopLoad<coop_mat16x16<f16,B>>(&ks[16u], LD), s0);
        s1 = coopMultiplyAdd(qa1, coopLoad<coop_mat16x16<f16,B>>(&ks[16u * LD + 16u], LD), s1);
        s0 = coopMultiplyAdd(qa2, coopLoad<coop_mat16x16<f16,B>>(&ks[32u], LD), s0);
        s1 = coopMultiplyAdd(qa2, coopLoad<coop_mat16x16<f16,B>>(&ks[16u * LD + 32u], LD), s1);
        s0 = coopMultiplyAdd(qa3, coopLoad<coop_mat16x16<f16,B>>(&ks[48u], LD), s0);
        s1 = coopMultiplyAdd(qa3, coopLoad<coop_mat16x16<f16,B>>(&ks[16u * LD + 48u], LD), s1);
        coopStoreT(s0, &ss[sbase], 32u);
        coopStoreT(s1, &ss[sbase + 16u], 32u);
        workgroupBarrier();
        for (var jj = 0u; jj < 16u; jj++) {
            let col = half * 16u + jj;
            let key = k0 + col;
            var w = 0.0;
            if (live && key <= qrow && key < t_len) {
                w = exp(ss[sbase + r * 32u + col] * scale - m);
            }
            l += w;
            ps[pbase + r * PLD + col] = f16(w);
        }
        workgroupBarrier();
        let pa0 = coopLoadT<coop_mat16x16<f16,A>>(&ps[pbase], PLD);
        let pa1 = coopLoadT<coop_mat16x16<f16,A>>(&ps[pbase + 16u], PLD);
        o0 = coopMultiplyAdd(pa0, coopLoadT<coop_mat16x16<f16,B>>(&vs[0u], LD), o0);
        o1 = coopMultiplyAdd(pa0, coopLoadT<coop_mat16x16<f16,B>>(&vs[16u], LD), o1);
        o2 = coopMultiplyAdd(pa0, coopLoadT<coop_mat16x16<f16,B>>(&vs[32u], LD), o2);
        o3 = coopMultiplyAdd(pa0, coopLoadT<coop_mat16x16<f16,B>>(&vs[48u], LD), o3);
        o0 = coopMultiplyAdd(pa1, coopLoadT<coop_mat16x16<f16,B>>(&vs[16u * LD], LD), o0);
        o1 = coopMultiplyAdd(pa1, coopLoadT<coop_mat16x16<f16,B>>(&vs[16u * LD + 16u], LD), o1);
        o2 = coopMultiplyAdd(pa1, coopLoadT<coop_mat16x16<f16,B>>(&vs[16u * LD + 32u], LD), o2);
        o3 = coopMultiplyAdd(pa1, coopLoadT<coop_mat16x16<f16,B>>(&vs[16u * LD + 48u], LD), o3);
        workgroupBarrier();
    }
    red[tid] = l;
    workgroupBarrier();
    let inv = 1.0 / (l + red[tid ^ 1u]);
    workgroupBarrier();

    // Four 16 x 16 output fragments through the scratch of the subgroup.
    for (var c = 0u; c < 4u; c++) {
        switch (c) {
            case 0u: { coopStoreT(o0, &ss[sbase], 16u); }
            case 1u: { coopStoreT(o1, &ss[sbase], 16u); }
            case 2u: { coopStoreT(o2, &ss[sbase], 16u); }
            default: { coopStoreT(o3, &ss[sbase], 16u); }
        }
        workgroupBarrier();
        if (live) {
            let s = sbase + r * 16u + half * 8u;
            let at = qrow * q4 + h * HD4 + c * 4u + half * 2u;
            out[at] = vec4<f32>(ss[s], ss[s + 1u], ss[s + 2u], ss[s + 3u]) * inv;
            out[at + 1u] = vec4<f32>(ss[s + 4u], ss[s + 5u], ss[s + 6u], ss[s + 7u]) * inv;
        }
        workgroupBarrier();
    }
}
