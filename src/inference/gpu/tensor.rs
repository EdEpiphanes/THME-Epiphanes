// SPDX-License-Identifier: AGPL-3.0-only
//! Quantized tensor representation storing packed weights with lazy CPU dequantization.

use std::sync::{Arc, OnceLock};
use wgpu::util::DeviceExt;

use super::error::ThmeError;
use crate::inference::weights::{decode_payload, read_tensor_bytes};
use crate::inference::WeightPrecision;
use crate::ir::TensorIR;

#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum QuantDtype {
    F32,
    F16,
    BF16,
    Q8_0,
    Q4_0,
    Q4_K,
    Q5_K,
    Q6_K,
    IQ4_XS,
}

impl QuantDtype {
    pub fn from_str_name(s: &str) -> Option<Self> {
        match s {
            "F32" => Some(Self::F32),
            "F16" => Some(Self::F16),
            "BF16" => Some(Self::BF16),
            "Q8_0" => Some(Self::Q8_0),
            "Q4_0" => Some(Self::Q4_0),
            "Q4_K" => Some(Self::Q4_K),
            "Q5_K" => Some(Self::Q5_K),
            "Q6_K" => Some(Self::Q6_K),
            "IQ4_XS" => Some(Self::IQ4_XS),
            _ => None,
        }
    }

    pub fn block_size(&self) -> usize {
        match self {
            Self::F32 | Self::F16 | Self::BF16 => 1,
            Self::Q8_0 | Self::Q4_0 => 32,
            Self::Q4_K | Self::Q5_K | Self::Q6_K | Self::IQ4_XS => 256,
        }
    }

    pub fn block_bytes(&self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::BF16 => 2,
            Self::Q8_0 => 34,
            Self::Q4_0 => 18,
            Self::Q4_K => 144,
            Self::Q5_K => 176,
            Self::Q6_K => 210,
            Self::IQ4_XS => 136,
        }
    }
}

pub struct QuantizedTensor {
    name: String,
    shape: Vec<usize>,
    in_dim: usize,
    out_dim: usize,
    dtype: QuantDtype,
    raw_bytes: Arc<[u8]>,
    cpu_f32: OnceLock<Vec<f32>>,
    gpu_buffer: OnceLock<wgpu::Buffer>,
}

impl QuantizedTensor {
    pub fn new_quantized(
        name: impl Into<String>,
        shape: Vec<usize>,
        dtype: QuantDtype,
        raw_bytes: Arc<[u8]>,
    ) -> Result<Self, ThmeError> {
        let name = name.into();
        if shape.len() < 2 {
            return Err(ThmeError::DimensionMismatch {
                expected: ">= 2 dimensions".into(),
                found: format!("{shape:?}"),
            });
        }
        let out_dim = shape[0];
        let in_dim = shape[1];
        Ok(Self {
            name,
            shape,
            in_dim,
            out_dim,
            dtype,
            raw_bytes,
            cpu_f32: OnceLock::new(),
            gpu_buffer: OnceLock::new(),
        })
    }

    pub fn new_f32(
        name: impl Into<String>,
        shape: Vec<usize>,
        values: Vec<f32>,
    ) -> Result<Self, ThmeError> {
        let name = name.into();
        let (out_dim, in_dim) = if shape.len() >= 2 {
            (shape[0], shape[1])
        } else {
            (shape.first().copied().unwrap_or(0), 1)
        };
        let cell = OnceLock::new();
        let _ = cell.set(values);
        Ok(Self {
            name,
            shape,
            in_dim,
            out_dim,
            dtype: QuantDtype::F32,
            raw_bytes: Arc::new([]),
            cpu_f32: cell,
            gpu_buffer: OnceLock::new(),
        })
    }

    pub fn from_f32_slice(
        name: impl Into<String>,
        shape: &[usize],
        values: &[f32],
    ) -> Result<Self, ThmeError> {
        Self::new_f32(name, shape.to_vec(), values.to_vec())
    }

    pub fn from_tensor_ir(
        tensor: &TensorIR,
        precision: WeightPrecision,
    ) -> Result<Self, ThmeError> {
        let (out_dim, in_dim) = if tensor.shape.len() >= 2 {
            (tensor.shape[0], tensor.shape[1])
        } else {
            (tensor.numel, 1)
        };

        if let Some(dtype) = QuantDtype::from_str_name(&tensor.original_dtype) {
            if precision == WeightPrecision::AsStored {
                if let Ok(bytes) = read_tensor_bytes(tensor) {
                    return Ok(Self {
                        name: tensor.name.clone(),
                        shape: tensor.shape.clone(),
                        in_dim,
                        out_dim,
                        dtype,
                        raw_bytes: Arc::from(bytes.into_boxed_slice()),
                        cpu_f32: OnceLock::new(),
                        gpu_buffer: OnceLock::new(),
                    });
                }
            }
        }

        let values = crate::inference::llama::load_tensor(tensor, tensor.numel, precision)
            .map_err(|e| {
                ThmeError::ModelLoadError(format!("Failed to load tensor '{}': {e}", tensor.name))
            })?;
        let cell = OnceLock::new();
        let _ = cell.set(values);
        Ok(Self {
            name: tensor.name.clone(),
            shape: tensor.shape.clone(),
            in_dim,
            out_dim,
            dtype: QuantDtype::F32,
            raw_bytes: Arc::new([]),
            cpu_f32: cell,
            gpu_buffer: OnceLock::new(),
        })
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn in_dim(&self) -> usize {
        self.in_dim
    }

    pub fn out_dim(&self) -> usize {
        self.out_dim
    }

    pub fn dtype(&self) -> QuantDtype {
        self.dtype
    }

    pub fn raw_bytes(&self) -> &[u8] {
        &self.raw_bytes
    }

    pub fn as_f32_slice(&self) -> Result<&[f32], ThmeError> {
        if let Some(slice) = self.cpu_f32.get() {
            return Ok(slice.as_slice());
        }
        let numel: usize = self.shape.iter().product();
        let decoded = decode_payload(&self.raw_bytes, &format!("{:?}", self.dtype), numel)
            .map_err(|e| {
                ThmeError::ModelLoadError(format!(
                    "Failed to dequantize tensor '{}' for CPU: {e}",
                    self.name
                ))
            })?;
        let _ = self.cpu_f32.set(decoded);
        match self.cpu_f32.get() {
            Some(s) => Ok(s.as_slice()),
            None => Err(ThmeError::ModelLoadError(format!(
                "Failed to initialize CPU storage for '{}'",
                self.name
            ))),
        }
    }

    /// Returns a reference to the uploaded GPU buffer, uploading on first access.
    /// True once the weights live on a device buffer.
    pub fn has_gpu_buffer(&self) -> bool {
        self.gpu_buffer.get().is_some()
    }

    pub fn get_or_create_gpu_buffer(
        &self,
        device: &wgpu::Device,
    ) -> Result<&wgpu::Buffer, ThmeError> {
        if let Some(buf) = self.gpu_buffer.get() {
            return Ok(buf);
        }

        let max_buf_size = device.limits().max_buffer_size;
        let max_binding_size = device.limits().max_storage_buffer_binding_size;
        let max_limit = max_buf_size.min(max_binding_size);

        let buffer = if !self.raw_bytes.is_empty() {
            if self.raw_bytes.len() as u64 > max_limit {
                return Err(ThmeError::BufferAllocationFailed {
                    name: self.name.clone(),
                    bytes: self.raw_bytes.len(),
                    reason: format!(
                        "Tensor size {} exceeds GPU limit {}",
                        self.raw_bytes.len(),
                        max_limit
                    ),
                });
            }
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("gpu_weight_{}", self.name)),
                contents: &self.raw_bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            })
        } else {
            let f32_data = self.as_f32_slice()?;
            let byte_count = f32_data.len() * 4;
            if byte_count as u64 > max_limit {
                return Err(ThmeError::BufferAllocationFailed {
                    name: self.name.clone(),
                    bytes: byte_count,
                    reason: format!("Tensor size {} exceeds GPU limit {}", byte_count, max_limit),
                });
            }
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("gpu_weight_{}", self.name)),
                contents: bytemuck::cast_slice(f32_data),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            })
        };

        let _ = self.gpu_buffer.set(buffer);
        match self.gpu_buffer.get() {
            Some(b) => Ok(b),
            None => Err(ThmeError::BufferAllocationFailed {
                name: self.name.clone(),
                bytes: self.raw_bytes.len(),
                reason: "GPU buffer initialization failed".into(),
            }),
        }
    }
}
