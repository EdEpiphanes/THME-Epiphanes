// SPDX-License-Identifier: AGPL-3.0-only
// Fused Dequantization + GEMV (y = W * x) for GGML IQ4_XS (136 bytes per 256 weights)

struct GemvParams {
    in_dim: u32,
    out_dim: u32,
    num_blocks_per_row: u32,
    stride_bytes: u32,
};

@group(0) @binding(0) var<storage, read> params: GemvParams;
@group(0) @binding(1) var<storage, read> input_x: array<f32>;
@group(0) @binding(2) var<storage, read> weights: array<u32>;
@group(0) @binding(3) var<storage, read_write> output_y: array<f32>;

var<workgroup> sdata: array<f32, 32>;

const KVALUES_IQ4NL: array<f32, 16> = array<f32, 16>(
    -127.0, -104.0, -83.0, -65.0, -49.0, -35.0, -22.0, -10.0,
    1.0, 13.0, 25.0, 38.0, 53.0, 69.0, 89.0, 113.0
);

@compute @workgroup_size(32, 1, 1)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let row = wid.x;
    let batch_row = wid.y;
    if (row >= params.out_dim) {
        return;
    }
    let tid = lid.x;
    var acc: f32 = 0.0;

    // Each IQ4_XS super-block is 136 bytes = 34 u32 words
    let row_word_base = row * params.num_blocks_per_row * 34u;
    let x_batch_base = batch_row * params.in_dim;

    for (var b = 0u; b < params.num_blocks_per_row; b++) {
        let b_base = row_word_base + b * 34u;
        let x_base = x_batch_base + b * 256u;

        // Word 0: d (f16 in bits 0..15), scales_h (u16 in bits 16..31)
        let w0 = weights[b_base];
        let d = unpack2x16float(w0).x;
        let scales_h = (w0 >> 16u) & 0xFFFFu;

        // Word 1: scales_l (4 bytes for 8 sub-blocks)
        let w1 = weights[b_base + 1u];

        // 8 sub-blocks: processed in 4 passes of pairs (2 sub-blocks per pass)
        for (var p = 0u; p < 4u; p++) {
            let is_sub1 = tid >= 16u;
            let ib = 2u * p + select(0u, 1u, is_sub1);
            let j = tid % 16u; // 0..15

            let scale_l_byte = (w1 >> ((ib / 2u) * 8u)) & 0xFFu;
            let low_nibble = (scale_l_byte >> (4u * (ib % 2u))) & 0x0Fu;
            let high_bits = (scales_h >> (2u * ib)) & 3u;
            let ls = i32(low_nibble | (high_bits << 4u));
            let dl = d * f32(ls - 32);

            // qs starts at word 2. Each sub-block has 16 bytes (4 u32 words).
            let qs_word = weights[b_base + 2u + ib * 4u + (j / 4u)];
            let byte = (qs_word >> ((j % 4u) * 8u)) & 0xFFu;
            let q_lo = byte & 0x0Fu;
            let q_hi = byte >> 4u;

            let w_lo = dl * KVALUES_IQ4NL[q_lo];
            let w_hi = dl * KVALUES_IQ4NL[q_hi];

            let x_lo = input_x[x_base + ib * 32u + j];
            let x_hi = input_x[x_base + ib * 32u + j + 16u];

            acc += w_lo * x_lo + w_hi * x_hi;
        }
    }

    sdata[tid] = acc;
    workgroupBarrier();

    if (tid < 16u) { sdata[tid] += sdata[tid + 16u]; }
    workgroupBarrier();
    if (tid < 8u)  { sdata[tid] += sdata[tid + 8u]; }
    workgroupBarrier();
    if (tid < 4u)  { sdata[tid] += sdata[tid + 4u]; }
    workgroupBarrier();
    if (tid < 2u)  { sdata[tid] += sdata[tid + 2u]; }
    workgroupBarrier();
    if (tid < 1u)  { sdata[tid] += sdata[tid + 1u]; }
    workgroupBarrier();

    if (tid == 0u) {
        output_y[batch_row * params.out_dim + row] = sdata[0];
    }
}
