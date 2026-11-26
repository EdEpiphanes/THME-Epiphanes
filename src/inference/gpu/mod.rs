// SPDX-License-Identifier: AGPL-3.0-only
//! GPU acceleration pipeline for THME reference inference.

pub mod backend;
pub mod error;
pub mod ptx_kernels;
pub mod tensor;

pub use backend::{
    dispatch_linear, dispatch_linear_multi_slice, dispatch_linear_slice, dispatch_score_head,
    dispatch_score_head_slice, score_head_from_logits, DeviceArg, GemvParams, GpuBackend,
    GpuContext, GpuDispatchMode, ShaderKind,
};
pub use error::ThmeError;
pub use tensor::{QuantDtype, QuantizedTensor};
