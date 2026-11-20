// SPDX-License-Identifier: AGPL-3.0-only
// Matrix-vector product for few rows: Y[t, o] = sum_k X[t, k] * W[o, k]. One workgroup of 64
// threads per (output row, token row); every thread strides over the row four values at a time,
// so a decode step reads each weight once. The weights binding and `load_w` come from dense.rs
// at //WEIGHTS_PRELUDE (f32, f16, Q8_0 or Q4_0). in_dim must be a multiple of 4.

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
@group(0) @binding(3) var<storage, read_write> output_y: array<f32>;

var<workgroup> sums: array<f32, 64>;

@compute @workgroup_size(64, 1, 1)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let row = wid.y * 65535u + wid.x;
    let tok = wid.z;
    if (row >= params.out_dim) {
        return;
    }
    let in4 = params.in_dim / 4u;
    let tid = lid.x;
    var acc = vec4<f32>(0.0);
    for (var i = tid; i < in4; i += 64u) {
        acc += load_w(row * in4 + i) * input_x[tok * in4 + i];
    }
    sums[tid] = acc.x + acc.y + acc.z + acc.w;
    workgroupBarrier();
    for (var s = 32u; s > 0u; s >>= 1u) {
        if (tid < s) { sums[tid] += sums[tid + s]; }
        workgroupBarrier();
    }
    if (tid == 0u) {
        output_y[params.out_offset + tok * params.out_dim + row] = sums[0];
    }
}
