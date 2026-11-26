// SPDX-License-Identifier: AGPL-3.0-only
// GEMV (y = W * x) for dense F32 matrices

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

var<workgroup> sdata: array<f32, 32>;

@compute @workgroup_size(32, 1, 1)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let row = wid.y * 65535u + wid.x;
    let tok = wid.z;
    if (row >= params.out_dim) {
        return;
    }
    let tid = lid.x;
    var acc: f32 = 0.0;
    let row_base = row * params.in_dim;

    for (var k = tid; k < params.in_dim; k += 32u) {
        acc += weights[row_base + k] * input_x[tok * params.in_dim + k];
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
        output_y[params.out_offset + tok * params.out_dim + row] = sdata[0];
    }
}
