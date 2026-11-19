// SPDX-License-Identifier: AGPL-3.0-only
//! Structured errors for GPU compute pipeline operations.

use std::fmt;

#[derive(Debug)]
pub enum ThmeError {
    GpuInitFailed(String),
    ShaderCompilationFailed {
        shader_name: String,
        error: String,
    },
    BufferAllocationFailed {
        name: String,
        bytes: usize,
        reason: String,
    },
    DataTransferFailed(String),
    ExecutionFailed(String),
    DimensionMismatch {
        expected: String,
        found: String,
    },
    UnsupportedFormat {
        dtype: String,
        backend: &'static str,
    },
    ModelLoadError(String),
    NonFiniteOutput {
        what: String,
        index: usize,
    },
    Other(String),
}

impl fmt::Display for ThmeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::GpuInitFailed(msg) => write!(f, "GPU initialization failed: {msg}"),
            Self::ShaderCompilationFailed { shader_name, error } => {
                write!(f, "Shader compilation failed for '{shader_name}': {error}")
            }
            Self::BufferAllocationFailed {
                name,
                bytes,
                reason,
            } => {
                write!(
                    f,
                    "Failed to allocate GPU buffer '{name}' ({bytes} bytes): {reason}"
                )
            }
            Self::DataTransferFailed(msg) => write!(f, "GPU data transfer failed: {msg}"),
            Self::ExecutionFailed(msg) => write!(f, "GPU execution failed: {msg}"),
            Self::DimensionMismatch { expected, found } => {
                write!(f, "Dimension mismatch: expected {expected}, found {found}")
            }
            Self::UnsupportedFormat { dtype, backend } => {
                write!(
                    f,
                    "Format '{dtype}' is not supported on backend '{backend}'"
                )
            }
            Self::ModelLoadError(msg) => write!(f, "Model load error: {msg}"),
            Self::NonFiniteOutput { what, index } => {
                write!(f, "Non-finite output in {what} at index {index}")
            }
            Self::Other(msg) => write!(f, "{msg}"),
        }
    }
}

impl std::error::Error for ThmeError {}
