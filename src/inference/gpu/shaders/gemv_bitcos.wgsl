// SPDX-License-Identifier: AGPL-3.0-only
// GEMV (y = W * x) for BITCOS packed ternary weights (2-bit code per weight, 16 weights/u32)

struct GemvParams {
    in_dim: u32,
    out_dim: u32,
    num_blocks_per_row: u32,
    superblock_nz_cap: u32,
};

@group(0) @binding(0) var<storage, read> params: GemvParams;
@group(0) @binding(1) var<storage, read> input_x: array<f32>;
@group(0) @binding(2) var<storage, read> weights_packed: array<u32>;
@group(0) @binding(3) var<storage, read_write> output_y: array<f32>;

var<workgroup> sdata: array<f32, 32>;

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
    let words_per_row = (params.in_dim + 15u) / 16u;
    let row_word_base = row * words_per_row;
    let x_base = batch_row * params.in_dim;
    let max_idx = max(params.superblock_nz_cap, 1u) - 1u;

    var acc: f32 = 0.0;
    for (var w = tid; w < words_per_row; w += 32u) {
        let word = weights_packed[row_word_base + w];
        let base_k = w * 16u;
        for (var j = 0u; j < 16u; j += 1u) {
            let k = base_k + j;
            if (k < params.in_dim) {
                let code = (word >> (j * 2u)) & 0x3u;
                let clamped_k = min(k, max_idx);
                if (code == 1u) {
                    acc += input_x[x_base + clamped_k];
                } else if (code == 2u) {
                    acc -= input_x[x_base + clamped_k];
                }
            }
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
    if (tid == 0u) {
        output_y[batch_row * params.out_dim + row] = sdata[0] + sdata[1];
    }
}
