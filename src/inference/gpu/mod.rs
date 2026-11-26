// SPDX-License-Identifier: AGPL-3.0-only
//! GPU acceleration pipeline for THME reference inference.

pub mod backend;
pub mod decoder;
pub mod dense;
pub mod error;
pub mod features;
pub mod ptx_kernels;
pub mod tensor;

pub use backend::{
    dispatch_linear, dispatch_linear_array, dispatch_linear_slice, dispatch_linear_slices,
    dispatch_score_head, dispatch_score_head_slice, score_head_from_logits, DeviceArg, GemvParams,
    GpuBackend, GpuContext, GpuStats, OpSample, ShaderKind, WeightStorage,
};
pub use decoder::{DecoderPlan, LayerWeights, MAX_DEVICE_CONTEXT};
pub use dense::{quantize_blocks, repack_blocks};
pub use error::ThmeError;
pub use tensor::{QuantDtype, QuantizedTensor};
