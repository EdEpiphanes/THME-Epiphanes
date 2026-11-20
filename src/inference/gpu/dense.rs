// SPDX-License-Identifier: AGPL-3.0-only
//! Device storage formats for dense weights (f16, Q8_0, Q4_0) and the kernel sources that read
//! them. Activations and accumulation stay f32 on every path.
//!
//! Device layouts (all little-endian `u32` words, rows contiguous, no padding between rows):
//! - F16: two half values per word, element `2i` in the low half.
//! - Q8_0 and Q4_0: one f32 scale per 32-element block (`nblocks` words), then the block codes:
//!   8 words of four signed bytes (Q8_0) or 4 words that keep the GGML nibble order of Q4_0
//!   (byte `j` holds element `j` in its low nibble and element `j + 16` in its high nibble).
//!   The scale is the GGML half-precision scale widened to f32, so decoding matches the CPU.

use super::backend::WeightStorage;
use super::error::ThmeError;
use crate::exporter::conversion::{encode_q4_0_block, encode_q8_block, f32_to_f16};

/// Elements per Q8_0 / Q4_0 block.
const BLOCK: usize = 32;

impl WeightStorage {
    /// Bytes one 32-element block takes in GGML block form (Q8_0, Q4_0) or per element (f16/f32).
    fn block_bytes(self) -> usize {
        match self {
            Self::Q8_0 => 34,
            Self::Q4_0 => 18,
            Self::F16 => 2 * BLOCK,
            Self::F32 => 4 * BLOCK,
        }
    }

    /// True when a `[out_dim, in_dim]` matrix can be stored this way: the matrix kernels read
    /// four values at a time, and block formats need whole blocks per row.
    pub fn fits(self, in_dim: usize, out_dim: usize) -> bool {
        let rows_ok = out_dim.is_multiple_of(4);
        match self {
            Self::F32 => true,
            Self::F16 => rows_ok && in_dim.is_multiple_of(4),
            Self::Q8_0 | Self::Q4_0 => rows_ok && in_dim.is_multiple_of(BLOCK),
        }
    }

    /// Device bytes for `elements` weights.
    pub fn device_bytes(self, elements: usize) -> usize {
        match self {
            Self::F32 => elements * 4,
            Self::F16 => elements * 2,
            Self::Q8_0 | Self::Q4_0 => {
                let blocks = elements / BLOCK;
                blocks * 4 + blocks * (self.block_bytes() - 2)
            }
        }
    }

    /// Stable tag for cache keys.
    pub(super) fn tag(self) -> u8 {
        self as u8
    }
}

/// Quantizes `w` (a multiple of 32 values) into GGML block bytes: 34 bytes per block for Q8_0,
/// 18 for Q4_0. Fails for the other storages.
pub fn quantize_blocks(storage: WeightStorage, w: &[f32]) -> Result<Vec<u8>, ThmeError> {
    let encode: fn(&[f32], &mut Vec<u8>) -> anyhow::Result<()> = match storage {
        WeightStorage::Q8_0 => encode_q8_block,
        WeightStorage::Q4_0 => encode_q4_0_block,
        other => {
            return Err(ThmeError::UnsupportedFormat {
                dtype: format!("{other:?}"),
                backend: "WGPU block quantizer",
            })
        }
    };
    if !w.len().is_multiple_of(BLOCK) {
        return Err(ThmeError::DimensionMismatch {
            expected: format!("a multiple of {BLOCK} values"),
            found: w.len().to_string(),
        });
    }
    let mut out = Vec::with_capacity(w.len() / BLOCK * storage.block_bytes());
    for block in w.chunks_exact(BLOCK) {
        encode(block, &mut out).map_err(|e| ThmeError::ModelLoadError(e.to_string()))?;
    }
    Ok(out)
}

/// Splits GGML block bytes into the device layout (scales first, then the codes).
pub fn repack_blocks(storage: WeightStorage, blocks: &[u8]) -> Result<Vec<u8>, ThmeError> {
    let size = storage.block_bytes();
    if !matches!(storage, WeightStorage::Q8_0 | WeightStorage::Q4_0)
        || !blocks.len().is_multiple_of(size)
    {
        return Err(ThmeError::DimensionMismatch {
            expected: format!("whole {storage:?} blocks of {size} bytes"),
            found: blocks.len().to_string(),
        });
    }
    let n = blocks.len() / size;
    let mut out = vec![0u8; n * 4 + n * (size - 2)];
    let (scales, codes) = out.split_at_mut(n * 4);
    for (i, block) in blocks.chunks_exact(size).enumerate() {
        let d = crate::profiling::f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        scales[i * 4..i * 4 + 4].copy_from_slice(&d.to_le_bytes());
        codes[i * (size - 2)..(i + 1) * (size - 2)].copy_from_slice(&block[2..]);
    }
    Ok(out)
}

/// Packed bytes of one chunk: scales (or the whole payload) and codes.
type Packed = Result<(Vec<u8>, Vec<u8>), ThmeError>;

/// Converts `w` to the device layout of `storage`, splitting the work over the host threads.
pub(super) fn pack_for_device(storage: WeightStorage, w: &[f32]) -> Result<Vec<u8>, ThmeError> {
    let one = |part: &[f32]| -> Packed {
        match storage {
            WeightStorage::F32 => Ok((bytemuck::cast_slice(part).to_vec(), Vec::new())),
            WeightStorage::F16 => Ok((
                part.iter()
                    .flat_map(|&v| f32_to_f16(v).to_le_bytes())
                    .collect(),
                Vec::new(),
            )),
            _ => {
                let packed = repack_blocks(storage, &quantize_blocks(storage, part)?)?;
                let scale_bytes = part.len() / BLOCK * 4;
                Ok((
                    packed[..scale_bytes].to_vec(),
                    packed[scale_bytes..].to_vec(),
                ))
            }
        }
    };
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    let step = (w.len() / threads).next_multiple_of(BLOCK).max(BLOCK);
    let parts: Vec<Packed> = std::thread::scope(|scope| {
        let handles: Vec<_> = w
            .chunks(step)
            .map(|part| scope.spawn(move || one(part)))
            .collect();
        handles
            .into_iter()
            .map(|h| {
                h.join().unwrap_or_else(|_| {
                    Err(ThmeError::ExecutionFailed(
                        "weight packing thread failed".into(),
                    ))
                })
            })
            .collect()
    });
    let mut first = Vec::with_capacity(storage.device_bytes(w.len()));
    let mut second = Vec::new();
    for part in parts {
        let (a, b) = part?;
        first.extend_from_slice(&a);
        second.extend_from_slice(&b);
    }
    first.extend_from_slice(&second);
    Ok(first)
}

/// Which kernel a source is generated for.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(super) enum DenseKernel {
    Gemm,
    Gemv,
    /// Cooperative matrices, any weight storage, weights converted in workgroup memory.
    Coop,
    /// Cooperative matrices, f16 weights read in place.
    CoopF16,
    /// f32 activations to padded f16 for `CoopF16`.
    ConvertX,
}

/// WGSL declaration of the weights binding and `load_w(i)`: the four consecutive weights
/// `4i .. 4i + 4` of the matrix as a vec4.
fn weights_prelude(storage: WeightStorage) -> &'static str {
    match storage {
        WeightStorage::F32 => {
            "@group(0) @binding(2) var<storage, read> weights: array<vec4<f32>>;\n\
             fn load_w(i: u32) -> vec4<f32> { return weights[i]; }\n"
        }
        WeightStorage::F16 => {
            "@group(0) @binding(2) var<storage, read> weights: array<vec2<u32>>;\n\
             fn load_w(i: u32) -> vec4<f32> {\n\
                 let w = weights[i];\n\
                 return vec4<f32>(unpack2x16float(w.x), unpack2x16float(w.y));\n\
             }\n"
        }
        WeightStorage::Q8_0 => {
            "@group(0) @binding(2) var<storage, read> weights: array<u32>;\n\
             fn load_w(i: u32) -> vec4<f32> {\n\
                 let blocks = params.out_dim * (params.in_dim >> 5u);\n\
                 let d = bitcast<f32>(weights[i >> 3u]);\n\
                 let w = weights[blocks + i];\n\
                 let b = vec4<i32>(i32(w << 24u) >> 24u, i32(w << 16u) >> 24u,\n\
                                   i32(w << 8u) >> 24u, i32(w) >> 24u);\n\
                 return vec4<f32>(b) * d;\n\
             }\n"
        }
        WeightStorage::Q4_0 => {
            "@group(0) @binding(2) var<storage, read> weights: array<u32>;\n\
             fn load_w(i: u32) -> vec4<f32> {\n\
                 let blocks = params.out_dim * (params.in_dim >> 5u);\n\
                 let block = i >> 3u;\n\
                 let pos = i & 7u;\n\
                 let d = bitcast<f32>(weights[block]);\n\
                 let w = weights[blocks + block * 4u + (pos & 3u)];\n\
                 let n = select(w & 0x0F0F0F0Fu, (w >> 4u) & 0x0F0F0F0Fu, pos >= 4u);\n\
                 let b = vec4<f32>(f32(n & 255u), f32((n >> 8u) & 255u),\n\
                                   f32((n >> 16u) & 255u), f32(n >> 24u));\n\
                 return (b - vec4<f32>(8.0)) * d;\n\
             }\n"
        }
    }
}

/// Complete WGSL source of the matrix kernel for `storage`.
pub(super) fn kernel_source(kernel: DenseKernel, storage: WeightStorage) -> String {
    let template = match kernel {
        DenseKernel::Gemm => include_str!("shaders/gemm_v4.wgsl"),
        DenseKernel::Gemv => include_str!("shaders/gemv_v4.wgsl"),
        DenseKernel::Coop => include_str!("shaders/gemm_coop.wgsl"),
        DenseKernel::CoopF16 => include_str!("shaders/gemm_coop_f16w.wgsl"),
        DenseKernel::ConvertX => include_str!("shaders/x_to_f16.wgsl"),
    };
    // Shader files may carry CRLF line endings on Windows checkouts.
    template
        .replace("\r\n", "\n")
        .replace("//WEIGHTS_PRELUDE\n", weights_prelude(storage))
}
