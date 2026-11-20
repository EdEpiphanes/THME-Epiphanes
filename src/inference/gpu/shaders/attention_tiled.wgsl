// SPDX-License-Identifier: AGPL-3.0-only
// Tiled causal grouped-query attention with an online softmax (no score row in memory).
// One workgroup of 64 threads handles 64 consecutive queries of one head; each thread owns one
// query: it keeps the scaled query and the output accumulator in registers and walks the keys in
// tiles of 16 that the whole workgroup loads once into shared memory. Per tile a thread scores
// the 16 keys, folds them into a running maximum and sum, rescales its accumulator once and adds
// the weighted values. Heavy (late) query blocks are scheduled first.
//
// The host replaces HEAD_DIM_VEC4 with head_dim / 4 (head_dim a multiple of 4, at most 128) and
// compiles one pipeline per head size. Bindings match decoder_ops.wgsl: params (t_len, heads,
// kv_heads, head_dim, scale bits), q, k, v, and the output.

@group(0) @binding(0) var<storage, read> p: array<u32, 8>;
@group(0) @binding(1) var<storage, read> q_in: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> k_in: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> v_in: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> out: array<vec4<f32>>;

const HD4: u32 = HEAD_DIM_VEC4u;
const KT: u32 = 16u;
const BQ: u32 = 64u;

var<workgroup> ks: array<vec4<f32>, KT * HD4>;
var<workgroup> vs: array<vec4<f32>, KT * HD4>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let t_len = p[0];
    let heads = p[1];
    let kv_heads = p[2];
    let scale = bitcast<f32>(p[4]);
    let blocks = (t_len + BQ - 1u) / BQ;
    let qb = blocks - 1u - wid.x;
    let h = wid.y;
    let tid = lid.x;
    let qi = qb * BQ + tid;
    let live = qi < t_len;
    let q4 = heads * HD4;
    let kv4 = kv_heads * HD4;
    let kvh = h / (heads / kv_heads);

    var q: array<vec4<f32>, HD4>;
    var o: array<vec4<f32>, HD4>;
    for (var c = 0u; c < HD4; c++) {
        o[c] = vec4<f32>(0.0);
        if (live) { q[c] = q_in[qi * q4 + h * HD4 + c] * scale; } else { q[c] = vec4<f32>(0.0); }
    }
    var m = -3.0e38;
    var l = 0.0;
    let kend = min(qb * BQ + BQ, t_len);

    for (var k0 = 0u; k0 < kend; k0 += KT) {
        for (var i = tid; i < KT * HD4; i += BQ) {
            let key = k0 + i / HD4;
            let c = i % HD4;
            var kk = vec4<f32>(0.0);
            var vv = vec4<f32>(0.0);
            if (key < t_len) {
                kk = k_in[key * kv4 + kvh * HD4 + c];
                vv = v_in[key * kv4 + kvh * HD4 + c];
            }
            ks[i] = kk;
            vs[i] = vv;
        }
        workgroupBarrier();

        var s: array<f32, KT>;
        var tile_max = -3.0e38;
        for (var j = 0u; j < KT; j += 4u) {
            var a0 = vec4<f32>(0.0);
            var a1 = vec4<f32>(0.0);
            var a2 = vec4<f32>(0.0);
            var a3 = vec4<f32>(0.0);
            for (var c = 0u; c < HD4; c++) {
                let qc = q[c];
                a0 += qc * ks[j * HD4 + c];
                a1 += qc * ks[(j + 1u) * HD4 + c];
                a2 += qc * ks[(j + 2u) * HD4 + c];
                a3 += qc * ks[(j + 3u) * HD4 + c];
            }
            s[j] = select(-3.0e38, a0.x + a0.y + a0.z + a0.w, k0 + j <= qi);
            s[j + 1u] = select(-3.0e38, a1.x + a1.y + a1.z + a1.w, k0 + j + 1u <= qi);
            s[j + 2u] = select(-3.0e38, a2.x + a2.y + a2.z + a2.w, k0 + j + 2u <= qi);
            s[j + 3u] = select(-3.0e38, a3.x + a3.y + a3.z + a3.w, k0 + j + 3u <= qi);
            tile_max = max(tile_max, max(max(s[j], s[j + 1u]), max(s[j + 2u], s[j + 3u])));
        }
        let m_new = max(m, tile_max);
        let alpha = exp(m - m_new);
        l *= alpha;
        for (var c = 0u; c < HD4; c++) { o[c] *= alpha; }
        for (var j = 0u; j < KT; j += 4u) {
            let w0 = select(0.0, exp(s[j] - m_new), k0 + j <= qi);
            let w1 = select(0.0, exp(s[j + 1u] - m_new), k0 + j + 1u <= qi);
            let w2 = select(0.0, exp(s[j + 2u] - m_new), k0 + j + 2u <= qi);
            let w3 = select(0.0, exp(s[j + 3u] - m_new), k0 + j + 3u <= qi);
            l += (w0 + w1) + (w2 + w3);
            for (var c = 0u; c < HD4; c++) {
                o[c] += (w0 * vs[j * HD4 + c] + w1 * vs[(j + 1u) * HD4 + c])
                    + (w2 * vs[(j + 2u) * HD4 + c] + w3 * vs[(j + 3u) * HD4 + c]);
            }
        }
        m = m_new;
        workgroupBarrier();
    }

    if (live) {
        let inv = 1.0 / l;
        for (var c = 0u; c < HD4; c++) { out[qi * q4 + h * HD4 + c] = o[c] * inv; }
    }
}
