// SPDX-License-Identifier: AGPL-3.0-only
// Converts the f32 activations of a matrix product to f16 (two values per word) and zero-fills
// the rows up to the next multiple of 16, so the cooperative-matrix kernel can load whole
// 16 x 16 fragments straight from device memory. Uses the same four bindings as the matrix
// kernels (params at 0, f32 activations at 1, f16 words at 3).

struct GemvParams {
    in_dim: u32,
    out_dim: u32,
    num_blocks_per_row: u32,
    rows: u32,
    out_offset: u32,
};

@group(0) @binding(0) var<storage, read> params: GemvParams;
@group(0) @binding(1) var<storage, read> input_x: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> xh: array<vec2<u32>>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let i = (wid.y * 65535u + wid.x) * 256u + lid;
    let in4 = params.in_dim / 4u;
    let rows16 = (params.rows + 15u) / 16u * 16u;
    if (i >= rows16 * in4) { return; }
    var v = vec4<f32>(0.0);
    if (i < params.rows * in4) { v = input_x[i]; }
    xh[i] = vec2<u32>(pack2x16float(v.xy), pack2x16float(v.zw));
}
