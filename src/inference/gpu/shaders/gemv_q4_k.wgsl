// SPDX-License-Identifier: AGPL-3.0-only
// Fused Dequantization + GEMV (y = W * x) for GGML Q4_K (144 bytes per 256 weights)

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

fn get_scale_byte(w1: u32, w2: u32, w3: u32, idx: u32) -> u32 {
    if (idx < 4u) {
        return (w1 >> (idx * 8u)) & 0xFFu;
    } else if (idx < 8u) {
        return (w2 >> ((idx - 4u) * 8u)) & 0xFFu;
    } else {
        return (w3 >> ((idx - 8u) * 8u)) & 0xFFu;
    }
}

fn scale_min_k4(j: u32, w1: u32, w2: u32, w3: u32) -> vec2<f32> {
    if (j < 4u) {
        let sc = f32(get_scale_byte(w1, w2, w3, j) & 63u);
        let m  = f32(get_scale_byte(w1, w2, w3, j + 4u) & 63u);
        return vec2<f32>(sc, m);
    } else {
        let q_j_plus_4 = get_scale_byte(w1, w2, w3, j + 4u);
        let q_j_minus_4 = get_scale_byte(w1, w2, w3, j - 4u);
        let q_j = get_scale_byte(w1, w2, w3, j);
        let sc = f32((q_j_plus_4 & 0x0Fu) | ((q_j_minus_4 >> 6u) << 4u));
        let m  = f32((q_j_plus_4 >> 4u) | ((q_j >> 6u) << 4u));
        return vec2<f32>(sc, m);
    }
}

@compute @workgroup_size(32, 1, 1)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let row = wid.x;
    if (row >= params.out_dim) {
        return;
    }
    let tid = lid.x;
    var acc: f32 = 0.0;

    // Each Q4_K super-block is 144 bytes = 36 u32 words
    let row_word_base = row * params.num_blocks_per_row * 36u;

    for (var b = 0u; b < params.num_blocks_per_row; b++) {
        let b_base = row_word_base + b * 36u;
        let x_base = b * 256u;

        // Word 0 holds f16 d and dmin
        let d_dmin = unpack2x16float(weights[b_base]);
        let d = d_dmin.x;
        let dmin = d_dmin.y;

        // Words 1..3 hold the 12 packed scale bytes
        let w1 = weights[b_base + 1u];
        let w2 = weights[b_base + 2u];
        let w3 = weights[b_base + 3u];

        // Process 4 chunks (each covering 64 weights)
        for (var c = 0u; c < 4u; c++) {
            let sm1 = scale_min_k4(2u * c, w1, w2, w3);
            let d1 = d * sm1.x;
            let min1 = dmin * sm1.y;

            let sm2 = scale_min_k4(2u * c + 1u, w1, w2, w3);
            let d2 = d * sm2.x;
            let min2 = dmin * sm2.y;

            // qs starts at word 4. Each chunk has 32 bytes (8 u32 words).
            let q_word = weights[b_base + 4u + c * 8u + (tid / 4u)];
            let q_byte = (q_word >> ((tid % 4u) * 8u)) & 0xFFu;

            let qa = f32(q_byte & 0x0Fu);
            let qb = f32(q_byte >> 4u);

            let w1_val = d1 * qa - min1;
            let w2_val = d2 * qb - min2;

            let x1 = input_x[x_base + 64u * c + tid];
            let x2 = input_x[x_base + 64u * c + 32u + tid];

            acc += w1_val * x1 + w2_val * x2;
        }
    }

    // Shared memory reduction across 32 threads in the warp
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
        output_y[row] = sdata[0];
    }
}
