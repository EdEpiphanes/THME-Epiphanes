// SPDX-License-Identifier: AGPL-3.0-only
// Integer matrix product on packed 4 x 8-bit dot products: Y[t, o] = sum over 32-value blocks b
// of sx[t, b] * sw[o, b] * (sum over the block of int8 x * int8 w). x_to_q8.wgsl has already
// quantized the activations (binding 1: scales, then codes). The weights binding, and
// `word(block, k)` (word k of a weight block as four signed bytes: Q8_0 codes as stored, Q4_0
// nibbles minus 8), come from dense.rs at //WEIGHTS_PRELUDE and //WORD_LOADER; the weight scales
// are the first `out_dim * in_dim / 32` words of the weights binding.
// A workgroup of 16 x 16 threads computes a 128 x 128 tile; thread (lx, ly) owns the outputs
// at rows ly + 16 i and columns lx + 16 j (i, j in 0..8), so neighbouring threads read
// neighbouring rows of the odd-strided workgroup tiles without bank conflicts and write
// neighbouring columns. The host uses it for Q8_0 and Q4_0 storage with more than 8 rows.

struct GemvParams {
    in_dim: u32,
    out_dim: u32,
    num_blocks_per_row: u32,
    rows: u32,
    out_offset: u32,
};

@group(0) @binding(0) var<storage, read> params: GemvParams;
@group(0) @binding(1) var<storage, read> xq: array<u32>;
//WEIGHTS_PRELUDE
//WORD_LOADER
@group(0) @binding(3) var<storage, read_write> output_y: array<vec4<f32>>;

// Eight words per block row plus one of padding (odd stride).
const STR: u32 = 9u;

var<workgroup> xs: array<u32, 1152>;
var<workgroup> ws: array<u32, 1152>;
var<workgroup> xsc: array<f32, 128>;
var<workgroup> wsc: array<f32, 128>;

fn put(t: u32, o: u32, v: f32) {
    if (o < params.out_dim) {
        let at = params.out_offset + t * params.out_dim + o;
        output_y[at >> 2u][at & 3u] = v;
    }
}

@compute @workgroup_size(16, 16, 1)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    // Row tiles vary fastest, so the workgroups running together share each weight tile in L2.
    let lin = wid.y * nwg.x + wid.x;
    let o0 = (lin / nwg.y) * 128u;
    let t0 = (lin % nwg.y) * 128u;
    let tid = lid.y * 16u + lid.x;
    let nbw = params.in_dim / 32u;
    let xblocks = params.rows * nbw;
    let wblocks = params.out_dim * nbw;
    var a00 = 0.0;
    var a01 = 0.0;
    var a02 = 0.0;
    var a03 = 0.0;
    var a04 = 0.0;
    var a05 = 0.0;
    var a06 = 0.0;
    var a07 = 0.0;
    var a10 = 0.0;
    var a11 = 0.0;
    var a12 = 0.0;
    var a13 = 0.0;
    var a14 = 0.0;
    var a15 = 0.0;
    var a16 = 0.0;
    var a17 = 0.0;
    var a20 = 0.0;
    var a21 = 0.0;
    var a22 = 0.0;
    var a23 = 0.0;
    var a24 = 0.0;
    var a25 = 0.0;
    var a26 = 0.0;
    var a27 = 0.0;
    var a30 = 0.0;
    var a31 = 0.0;
    var a32 = 0.0;
    var a33 = 0.0;
    var a34 = 0.0;
    var a35 = 0.0;
    var a36 = 0.0;
    var a37 = 0.0;
    var a40 = 0.0;
    var a41 = 0.0;
    var a42 = 0.0;
    var a43 = 0.0;
    var a44 = 0.0;
    var a45 = 0.0;
    var a46 = 0.0;
    var a47 = 0.0;
    var a50 = 0.0;
    var a51 = 0.0;
    var a52 = 0.0;
    var a53 = 0.0;
    var a54 = 0.0;
    var a55 = 0.0;
    var a56 = 0.0;
    var a57 = 0.0;
    var a60 = 0.0;
    var a61 = 0.0;
    var a62 = 0.0;
    var a63 = 0.0;
    var a64 = 0.0;
    var a65 = 0.0;
    var a66 = 0.0;
    var a67 = 0.0;
    var a70 = 0.0;
    var a71 = 0.0;
    var a72 = 0.0;
    var a73 = 0.0;
    var a74 = 0.0;
    var a75 = 0.0;
    var a76 = 0.0;
    var a77 = 0.0;

    for (var b = 0u; b < nbw; b++) {
        // Thread tid stages half (four words) of block row tid / 2 of each operand.
        let r = tid >> 1u;
        let h = tid & 1u;
        let xr = t0 + r;
        let wr = o0 + r;
        for (var q = 0u; q < 4u; q++) {
            let kw = h * 4u + q;
            var xv = 0u;
            var wv = 0u;
            if (xr < params.rows) { xv = xq[xblocks + (xr * nbw + b) * 8u + kw]; }
            if (wr < params.out_dim) { wv = word(wr * nbw + b, kw); }
            xs[r * STR + kw] = xv;
            ws[r * STR + kw] = wv;
        }
        if (tid < 128u) {
            var s = 0.0;
            if (t0 + tid < params.rows) { s = bitcast<f32>(xq[(t0 + tid) * nbw + b]); }
            xsc[tid] = s;
        } else {
            var s = 0.0;
            let o = o0 + tid - 128u;
            if (o < params.out_dim) { s = bitcast<f32>(weights[o * nbw + b]); }
            wsc[tid - 128u] = s;
        }
        workgroupBarrier();

        var s00 = 0;
        var s01 = 0;
        var s02 = 0;
        var s03 = 0;
        var s04 = 0;
        var s05 = 0;
        var s06 = 0;
        var s07 = 0;
        var s10 = 0;
        var s11 = 0;
        var s12 = 0;
        var s13 = 0;
        var s14 = 0;
        var s15 = 0;
        var s16 = 0;
        var s17 = 0;
        var s20 = 0;
        var s21 = 0;
        var s22 = 0;
        var s23 = 0;
        var s24 = 0;
        var s25 = 0;
        var s26 = 0;
        var s27 = 0;
        var s30 = 0;
        var s31 = 0;
        var s32 = 0;
        var s33 = 0;
        var s34 = 0;
        var s35 = 0;
        var s36 = 0;
        var s37 = 0;
        var s40 = 0;
        var s41 = 0;
        var s42 = 0;
        var s43 = 0;
        var s44 = 0;
        var s45 = 0;
        var s46 = 0;
        var s47 = 0;
        var s50 = 0;
        var s51 = 0;
        var s52 = 0;
        var s53 = 0;
        var s54 = 0;
        var s55 = 0;
        var s56 = 0;
        var s57 = 0;
        var s60 = 0;
        var s61 = 0;
        var s62 = 0;
        var s63 = 0;
        var s64 = 0;
        var s65 = 0;
        var s66 = 0;
        var s67 = 0;
        var s70 = 0;
        var s71 = 0;
        var s72 = 0;
        var s73 = 0;
        var s74 = 0;
        var s75 = 0;
        var s76 = 0;
        var s77 = 0;
        for (var kw = 0u; kw < 8u; kw++) {
            let x0 = xs[(lid.y + 0u) * STR + kw];
            let x1 = xs[(lid.y + 16u) * STR + kw];
            let x2 = xs[(lid.y + 32u) * STR + kw];
            let x3 = xs[(lid.y + 48u) * STR + kw];
            let x4 = xs[(lid.y + 64u) * STR + kw];
            let x5 = xs[(lid.y + 80u) * STR + kw];
            let x6 = xs[(lid.y + 96u) * STR + kw];
            let x7 = xs[(lid.y + 112u) * STR + kw];
            let w0 = ws[(lid.x + 0u) * STR + kw];
            let w1 = ws[(lid.x + 16u) * STR + kw];
            let w2 = ws[(lid.x + 32u) * STR + kw];
            let w3 = ws[(lid.x + 48u) * STR + kw];
            let w4 = ws[(lid.x + 64u) * STR + kw];
            let w5 = ws[(lid.x + 80u) * STR + kw];
            let w6 = ws[(lid.x + 96u) * STR + kw];
            let w7 = ws[(lid.x + 112u) * STR + kw];
            s00 += dot4I8Packed(x0, w0);
            s01 += dot4I8Packed(x0, w1);
            s02 += dot4I8Packed(x0, w2);
            s03 += dot4I8Packed(x0, w3);
            s04 += dot4I8Packed(x0, w4);
            s05 += dot4I8Packed(x0, w5);
            s06 += dot4I8Packed(x0, w6);
            s07 += dot4I8Packed(x0, w7);
            s10 += dot4I8Packed(x1, w0);
            s11 += dot4I8Packed(x1, w1);
            s12 += dot4I8Packed(x1, w2);
            s13 += dot4I8Packed(x1, w3);
            s14 += dot4I8Packed(x1, w4);
            s15 += dot4I8Packed(x1, w5);
            s16 += dot4I8Packed(x1, w6);
            s17 += dot4I8Packed(x1, w7);
            s20 += dot4I8Packed(x2, w0);
            s21 += dot4I8Packed(x2, w1);
            s22 += dot4I8Packed(x2, w2);
            s23 += dot4I8Packed(x2, w3);
            s24 += dot4I8Packed(x2, w4);
            s25 += dot4I8Packed(x2, w5);
            s26 += dot4I8Packed(x2, w6);
            s27 += dot4I8Packed(x2, w7);
            s30 += dot4I8Packed(x3, w0);
            s31 += dot4I8Packed(x3, w1);
            s32 += dot4I8Packed(x3, w2);
            s33 += dot4I8Packed(x3, w3);
            s34 += dot4I8Packed(x3, w4);
            s35 += dot4I8Packed(x3, w5);
            s36 += dot4I8Packed(x3, w6);
            s37 += dot4I8Packed(x3, w7);
            s40 += dot4I8Packed(x4, w0);
            s41 += dot4I8Packed(x4, w1);
            s42 += dot4I8Packed(x4, w2);
            s43 += dot4I8Packed(x4, w3);
            s44 += dot4I8Packed(x4, w4);
            s45 += dot4I8Packed(x4, w5);
            s46 += dot4I8Packed(x4, w6);
            s47 += dot4I8Packed(x4, w7);
            s50 += dot4I8Packed(x5, w0);
            s51 += dot4I8Packed(x5, w1);
            s52 += dot4I8Packed(x5, w2);
            s53 += dot4I8Packed(x5, w3);
            s54 += dot4I8Packed(x5, w4);
            s55 += dot4I8Packed(x5, w5);
            s56 += dot4I8Packed(x5, w6);
            s57 += dot4I8Packed(x5, w7);
            s60 += dot4I8Packed(x6, w0);
            s61 += dot4I8Packed(x6, w1);
            s62 += dot4I8Packed(x6, w2);
            s63 += dot4I8Packed(x6, w3);
            s64 += dot4I8Packed(x6, w4);
            s65 += dot4I8Packed(x6, w5);
            s66 += dot4I8Packed(x6, w6);
            s67 += dot4I8Packed(x6, w7);
            s70 += dot4I8Packed(x7, w0);
            s71 += dot4I8Packed(x7, w1);
            s72 += dot4I8Packed(x7, w2);
            s73 += dot4I8Packed(x7, w3);
            s74 += dot4I8Packed(x7, w4);
            s75 += dot4I8Packed(x7, w5);
            s76 += dot4I8Packed(x7, w6);
            s77 += dot4I8Packed(x7, w7);
        }
        let sw0 = wsc[lid.x + 0u];
        let sw1 = wsc[lid.x + 16u];
        let sw2 = wsc[lid.x + 32u];
        let sw3 = wsc[lid.x + 48u];
        let sw4 = wsc[lid.x + 64u];
        let sw5 = wsc[lid.x + 80u];
        let sw6 = wsc[lid.x + 96u];
        let sw7 = wsc[lid.x + 112u];
        let sx0 = xsc[lid.y + 0u];
        a00 += f32(s00) * sx0 * sw0;
        a01 += f32(s01) * sx0 * sw1;
        a02 += f32(s02) * sx0 * sw2;
        a03 += f32(s03) * sx0 * sw3;
        a04 += f32(s04) * sx0 * sw4;
        a05 += f32(s05) * sx0 * sw5;
        a06 += f32(s06) * sx0 * sw6;
        a07 += f32(s07) * sx0 * sw7;
        let sx1 = xsc[lid.y + 16u];
        a10 += f32(s10) * sx1 * sw0;
        a11 += f32(s11) * sx1 * sw1;
        a12 += f32(s12) * sx1 * sw2;
        a13 += f32(s13) * sx1 * sw3;
        a14 += f32(s14) * sx1 * sw4;
        a15 += f32(s15) * sx1 * sw5;
        a16 += f32(s16) * sx1 * sw6;
        a17 += f32(s17) * sx1 * sw7;
        let sx2 = xsc[lid.y + 32u];
        a20 += f32(s20) * sx2 * sw0;
        a21 += f32(s21) * sx2 * sw1;
        a22 += f32(s22) * sx2 * sw2;
        a23 += f32(s23) * sx2 * sw3;
        a24 += f32(s24) * sx2 * sw4;
        a25 += f32(s25) * sx2 * sw5;
        a26 += f32(s26) * sx2 * sw6;
        a27 += f32(s27) * sx2 * sw7;
        let sx3 = xsc[lid.y + 48u];
        a30 += f32(s30) * sx3 * sw0;
        a31 += f32(s31) * sx3 * sw1;
        a32 += f32(s32) * sx3 * sw2;
        a33 += f32(s33) * sx3 * sw3;
        a34 += f32(s34) * sx3 * sw4;
        a35 += f32(s35) * sx3 * sw5;
        a36 += f32(s36) * sx3 * sw6;
        a37 += f32(s37) * sx3 * sw7;
        let sx4 = xsc[lid.y + 64u];
        a40 += f32(s40) * sx4 * sw0;
        a41 += f32(s41) * sx4 * sw1;
        a42 += f32(s42) * sx4 * sw2;
        a43 += f32(s43) * sx4 * sw3;
        a44 += f32(s44) * sx4 * sw4;
        a45 += f32(s45) * sx4 * sw5;
        a46 += f32(s46) * sx4 * sw6;
        a47 += f32(s47) * sx4 * sw7;
        let sx5 = xsc[lid.y + 80u];
        a50 += f32(s50) * sx5 * sw0;
        a51 += f32(s51) * sx5 * sw1;
        a52 += f32(s52) * sx5 * sw2;
        a53 += f32(s53) * sx5 * sw3;
        a54 += f32(s54) * sx5 * sw4;
        a55 += f32(s55) * sx5 * sw5;
        a56 += f32(s56) * sx5 * sw6;
        a57 += f32(s57) * sx5 * sw7;
        let sx6 = xsc[lid.y + 96u];
        a60 += f32(s60) * sx6 * sw0;
        a61 += f32(s61) * sx6 * sw1;
        a62 += f32(s62) * sx6 * sw2;
        a63 += f32(s63) * sx6 * sw3;
        a64 += f32(s64) * sx6 * sw4;
        a65 += f32(s65) * sx6 * sw5;
        a66 += f32(s66) * sx6 * sw6;
        a67 += f32(s67) * sx6 * sw7;
        let sx7 = xsc[lid.y + 112u];
        a70 += f32(s70) * sx7 * sw0;
        a71 += f32(s71) * sx7 * sw1;
        a72 += f32(s72) * sx7 * sw2;
        a73 += f32(s73) * sx7 * sw3;
        a74 += f32(s74) * sx7 * sw4;
        a75 += f32(s75) * sx7 * sw5;
        a76 += f32(s76) * sx7 * sw6;
        a77 += f32(s77) * sx7 * sw7;
        workgroupBarrier();
    }

    let tr0 = t0 + lid.y + 0u;
    if (tr0 < params.rows) {
        put(tr0, o0 + lid.x + 0u, a00);
        put(tr0, o0 + lid.x + 16u, a01);
        put(tr0, o0 + lid.x + 32u, a02);
        put(tr0, o0 + lid.x + 48u, a03);
        put(tr0, o0 + lid.x + 64u, a04);
        put(tr0, o0 + lid.x + 80u, a05);
        put(tr0, o0 + lid.x + 96u, a06);
        put(tr0, o0 + lid.x + 112u, a07);
    }
    let t1 = t0 + lid.y + 16u;
    if (t1 < params.rows) {
        put(t1, o0 + lid.x + 0u, a10);
        put(t1, o0 + lid.x + 16u, a11);
        put(t1, o0 + lid.x + 32u, a12);
        put(t1, o0 + lid.x + 48u, a13);
        put(t1, o0 + lid.x + 64u, a14);
        put(t1, o0 + lid.x + 80u, a15);
        put(t1, o0 + lid.x + 96u, a16);
        put(t1, o0 + lid.x + 112u, a17);
    }
    let t2 = t0 + lid.y + 32u;
    if (t2 < params.rows) {
        put(t2, o0 + lid.x + 0u, a20);
        put(t2, o0 + lid.x + 16u, a21);
        put(t2, o0 + lid.x + 32u, a22);
        put(t2, o0 + lid.x + 48u, a23);
        put(t2, o0 + lid.x + 64u, a24);
        put(t2, o0 + lid.x + 80u, a25);
        put(t2, o0 + lid.x + 96u, a26);
        put(t2, o0 + lid.x + 112u, a27);
    }
    let t3 = t0 + lid.y + 48u;
    if (t3 < params.rows) {
        put(t3, o0 + lid.x + 0u, a30);
        put(t3, o0 + lid.x + 16u, a31);
        put(t3, o0 + lid.x + 32u, a32);
        put(t3, o0 + lid.x + 48u, a33);
        put(t3, o0 + lid.x + 64u, a34);
        put(t3, o0 + lid.x + 80u, a35);
        put(t3, o0 + lid.x + 96u, a36);
        put(t3, o0 + lid.x + 112u, a37);
    }
    let t4 = t0 + lid.y + 64u;
    if (t4 < params.rows) {
        put(t4, o0 + lid.x + 0u, a40);
        put(t4, o0 + lid.x + 16u, a41);
        put(t4, o0 + lid.x + 32u, a42);
        put(t4, o0 + lid.x + 48u, a43);
        put(t4, o0 + lid.x + 64u, a44);
        put(t4, o0 + lid.x + 80u, a45);
        put(t4, o0 + lid.x + 96u, a46);
        put(t4, o0 + lid.x + 112u, a47);
    }
    let t5 = t0 + lid.y + 80u;
    if (t5 < params.rows) {
        put(t5, o0 + lid.x + 0u, a50);
        put(t5, o0 + lid.x + 16u, a51);
        put(t5, o0 + lid.x + 32u, a52);
        put(t5, o0 + lid.x + 48u, a53);
        put(t5, o0 + lid.x + 64u, a54);
        put(t5, o0 + lid.x + 80u, a55);
        put(t5, o0 + lid.x + 96u, a56);
        put(t5, o0 + lid.x + 112u, a57);
    }
    let t6 = t0 + lid.y + 96u;
    if (t6 < params.rows) {
        put(t6, o0 + lid.x + 0u, a60);
        put(t6, o0 + lid.x + 16u, a61);
        put(t6, o0 + lid.x + 32u, a62);
        put(t6, o0 + lid.x + 48u, a63);
        put(t6, o0 + lid.x + 64u, a64);
        put(t6, o0 + lid.x + 80u, a65);
        put(t6, o0 + lid.x + 96u, a66);
        put(t6, o0 + lid.x + 112u, a67);
    }
    let t7 = t0 + lid.y + 112u;
    if (t7 < params.rows) {
        put(t7, o0 + lid.x + 0u, a70);
        put(t7, o0 + lid.x + 16u, a71);
        put(t7, o0 + lid.x + 32u, a72);
        put(t7, o0 + lid.x + 48u, a73);
        put(t7, o0 + lid.x + 64u, a74);
        put(t7, o0 + lid.x + 80u, a75);
        put(t7, o0 + lid.x + 96u, a76);
        put(t7, o0 + lid.x + 112u, a77);
    }
}
