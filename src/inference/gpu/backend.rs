// SPDX-License-Identifier: AGPL-3.0-only
//! WGPU Compute Acceleration Backend for THME Tensor Forward Passes.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use super::error::ThmeError;
use super::tensor::{QuantDtype, QuantizedTensor};
use crate::inference::llama::{linear, score_head, DepthExit};

#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    clap::ValueEnum,
    Default,
    serde::Serialize,
    serde::Deserialize,
)]
pub enum DeviceArg {
    #[default]
    Auto,
    Gpu,
    Cpu,
}

impl DeviceArg {
    /// Resolves the device argument into an optional GPU backend.
    /// Returns `None` for CPU or if GPU initialization fails in Auto mode.
    pub fn resolve(self) -> Option<Arc<dyn GpuBackend>> {
        match self {
            Self::Cpu => None,
            Self::Gpu => match GpuContext::init() {
                Ok(ctx) => Some(Arc::new(ctx)),
                Err(err) => {
                    eprintln!("Warning: GPU requested but initialization failed ({err}); falling back to CPU");
                    None
                }
            },
            Self::Auto => GpuContext::init()
                .ok()
                .map(|ctx| Arc::new(ctx) as Arc<dyn GpuBackend>),
        }
    }
}

#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ShaderKind {
    Q4_K,
    Q5_K,
    IQ4_XS,
    F32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GemvParams {
    pub in_dim: u32,
    pub out_dim: u32,
    pub num_blocks_per_row: u32,
    pub stride_bytes: u32,
}

struct BufferPool {
    input_buffer: wgpu::Buffer,
    input_capacity: usize,
    output_buffer: wgpu::Buffer,
    output_capacity: usize,
    staging_buffer: wgpu::Buffer,
    staging_capacity: usize,
    params_buffer: wgpu::Buffer,
}

pub struct GpuContext {
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
    pub pipelines: HashMap<ShaderKind, wgpu::ComputePipeline>,
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub adapter_info: wgpu::AdapterInfo,
    buffer_pool: Mutex<BufferPool>,
}

impl GpuContext {
    /// Initializes WGPU on the Vulkan adapter.
    pub fn init() -> Result<Self, ThmeError> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            flags: wgpu::InstanceFlags::empty(),
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
            ..wgpu::RequestAdapterOptions::default()
        }))
        .map_err(|e| ThmeError::GpuInitFailed(format!("No suitable Vulkan GPU adapter: {e}")))?;

        let adapter_info = adapter.get_info();

        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("thme_gpu_device"),
            required_features: wgpu::Features::empty(),
            required_limits: adapter.limits(),
            memory_hints: wgpu::MemoryHints::Performance,
            ..wgpu::DeviceDescriptor::default()
        }))
        .map_err(|e| ThmeError::GpuInitFailed(format!("Failed to request WGPU device: {e}")))?;

        let device = Arc::new(device);
        let queue = Arc::new(queue);

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("gemv_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("gemv_pipeline_layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let mut pipelines = HashMap::new();
        let shader_sources = [
            (ShaderKind::Q4_K, include_str!("shaders/gemv_q4_k.wgsl")),
            (ShaderKind::Q5_K, include_str!("shaders/gemv_q5_k.wgsl")),
            (ShaderKind::IQ4_XS, include_str!("shaders/gemv_iq4_xs.wgsl")),
            (ShaderKind::F32, include_str!("shaders/gemv_f32.wgsl")),
        ];

        for (kind, source) in shader_sources {
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(&format!("{kind:?}_shader")),
                source: wgpu::ShaderSource::Wgsl(source.into()),
            });
            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(&format!("{kind:?}_pipeline")),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });
            pipelines.insert(kind, pipeline);
        }

        let default_x_cap = 64 * 1024 * 4; // 256 KB
        let default_y_cap = 262_208 * 4; // ~1.05 MB (Vocab size)

        let input_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pool_input_buf"),
            size: default_x_cap as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pool_output_buf"),
            size: default_y_cap as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pool_staging_buf"),
            size: default_y_cap as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pool_params_buf"),
            size: std::mem::size_of::<GemvParams>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let buffer_pool = Mutex::new(BufferPool {
            input_buffer,
            input_capacity: default_x_cap,
            output_buffer,
            output_capacity: default_y_cap,
            staging_buffer,
            staging_capacity: default_y_cap,
            params_buffer,
        });

        Ok(Self {
            device,
            queue,
            pipelines,
            bind_group_layout,
            adapter_info,
            buffer_pool,
        })
    }

    /// Dispatches a single vector GEMV (y = W * x) for length `in_dim` -> `out_dim`.
    pub fn dispatch_gemv_single(
        &self,
        kind: ShaderKind,
        x: &[f32],
        weights_buf: &wgpu::Buffer,
        in_dim: usize,
        out_dim: usize,
    ) -> Result<Vec<f32>, ThmeError> {
        let pipeline = self.pipelines.get(&kind).ok_or_else(|| {
            ThmeError::ExecutionFailed(format!("Compute pipeline for {kind:?} not found"))
        })?;

        let x_bytes = x.len() * 4;
        let y_bytes = out_dim * 4;

        let num_blocks = match kind {
            ShaderKind::Q4_K | ShaderKind::Q5_K | ShaderKind::IQ4_XS => in_dim / 256,
            ShaderKind::F32 => 0,
        };

        let params = GemvParams {
            in_dim: in_dim as u32,
            out_dim: out_dim as u32,
            num_blocks_per_row: num_blocks as u32,
            stride_bytes: 0,
        };

        let mut pool = self
            .buffer_pool
            .lock()
            .map_err(|_| ThmeError::ExecutionFailed("GPU buffer pool mutex poisoned".into()))?;

        // Resize pool buffers dynamically if required
        if x_bytes > pool.input_capacity {
            let new_cap = x_bytes.next_power_of_two();
            pool.input_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("pool_input_buf_resized"),
                size: new_cap as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            pool.input_capacity = new_cap;
        }

        if y_bytes > pool.output_capacity {
            let new_cap = y_bytes.next_power_of_two();
            pool.output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("pool_output_buf_resized"),
                size: new_cap as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            pool.output_capacity = new_cap;
        }

        if y_bytes > pool.staging_capacity {
            let new_cap = y_bytes.next_power_of_two();
            pool.staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("pool_staging_buf_resized"),
                size: new_cap as u64,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            pool.staging_capacity = new_cap;
        }

        self.queue
            .write_buffer(&pool.params_buffer, 0, bytemuck::bytes_of(&params));
        self.queue
            .write_buffer(&pool.input_buffer, 0, bytemuck::cast_slice(x));

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gemv_bind_group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: pool.params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: pool.input_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: weights_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: pool.output_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("gemv_encoder"),
            });

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("gemv_compute_pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch_workgroups(out_dim as u32, 1, 1);
        }

        encoder.copy_buffer_to_buffer(
            &pool.output_buffer,
            0,
            &pool.staging_buffer,
            0,
            y_bytes as u64,
        );

        self.queue.submit(Some(encoder.finish()));

        let slice = pool.staging_buffer.slice(..y_bytes as u64);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .map_err(|e| ThmeError::DataTransferFailed(format!("GPU poll failed: {e}")))?;

        rx.recv()
            .map_err(|e| ThmeError::DataTransferFailed(format!("GPU channel recv failed: {e}")))?
            .map_err(|e| ThmeError::DataTransferFailed(format!("GPU buffer map failed: {e}")))?;

        let data = slice
            .get_mapped_range()
            .map_err(|e| ThmeError::DataTransferFailed(format!("GPU map range failed: {e}")))?;
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        pool.staging_buffer.unmap();

        Ok(result)
    }
}

pub trait GpuBackend: Send + Sync {
    /// Executes matrix-vector multiplication y = W * x for linear projections.
    fn compute_linear(
        &self,
        x: &[f32],
        weights: &QuantizedTensor,
        in_dim: usize,
        out_dim: usize,
    ) -> Result<Vec<f32>, ThmeError>;

    /// Executes readout head projection y = head * x returning logits [rows * vocab_size].
    fn compute_score_head(
        &self,
        x: &[f32],
        weights: &QuantizedTensor,
        in_dim: usize,
        vocab_size: usize,
    ) -> Result<Vec<f32>, ThmeError> {
        self.compute_linear(x, weights, in_dim, vocab_size)
    }

    fn is_format_supported(&self, dtype: QuantDtype) -> bool;
    fn device_name(&self) -> &str;
}

impl GpuBackend for GpuContext {
    fn compute_linear(
        &self,
        x: &[f32],
        weights: &QuantizedTensor,
        in_dim: usize,
        out_dim: usize,
    ) -> Result<Vec<f32>, ThmeError> {
        if !x.is_empty() && !x.len().is_multiple_of(in_dim) {
            return Err(ThmeError::DimensionMismatch {
                expected: format!("Multiple of in_dim {in_dim}"),
                found: format!("{}", x.len()),
            });
        }

        let kind = match weights.dtype() {
            QuantDtype::Q4_K => ShaderKind::Q4_K,
            QuantDtype::Q5_K => ShaderKind::Q5_K,
            QuantDtype::IQ4_XS => ShaderKind::IQ4_XS,
            QuantDtype::F32 => ShaderKind::F32,
            other => {
                return Err(ThmeError::UnsupportedFormat {
                    dtype: format!("{other:?}"),
                    backend: "WGPU/Vulkan",
                });
            }
        };

        let weights_buf = weights.get_or_create_gpu_buffer(&self.device)?;
        let rows = x.len() / in_dim;

        if rows == 1 {
            let res = self.dispatch_gemv_single(kind, x, weights_buf, in_dim, out_dim)?;
            if let Some(pos) = res.iter().position(|v| !v.is_finite()) {
                return Err(ThmeError::NonFiniteOutput {
                    what: format!("linear projection '{}'", weights.name()),
                    index: pos,
                });
            }
            return Ok(res);
        }

        let mut output = Vec::with_capacity(rows * out_dim);
        for row_idx in 0..rows {
            let slice = &x[row_idx * in_dim..(row_idx + 1) * in_dim];
            let row_res = self.dispatch_gemv_single(kind, slice, weights_buf, in_dim, out_dim)?;
            if let Some(pos) = row_res.iter().position(|v| !v.is_finite()) {
                return Err(ThmeError::NonFiniteOutput {
                    what: format!("linear projection '{}' at row {row_idx}", weights.name()),
                    index: pos,
                });
            }
            output.extend_from_slice(&row_res);
        }

        Ok(output)
    }

    fn is_format_supported(&self, dtype: QuantDtype) -> bool {
        matches!(
            dtype,
            QuantDtype::Q4_K | QuantDtype::Q5_K | QuantDtype::IQ4_XS | QuantDtype::F32
        )
    }

    fn device_name(&self) -> &str {
        &self.adapter_info.name
    }
}

pub fn dispatch_linear(
    backend: Option<&dyn GpuBackend>,
    x: &[f32],
    weights: &QuantizedTensor,
    in_dim: usize,
    out_dim: usize,
) -> Result<Vec<f32>, ThmeError> {
    if let Some(gpu) = backend {
        if gpu.is_format_supported(weights.dtype()) {
            match gpu.compute_linear(x, weights, in_dim, out_dim) {
                Ok(y) => return Ok(y),
                Err(err) => {
                    eprintln!("Notice: GPU compute_linear fallback to CPU SIMD: {err}");
                }
            }
        }
    }
    let w_f32 = weights.as_f32_slice()?;
    Ok(linear(x, w_f32, in_dim, out_dim))
}

pub fn dispatch_linear_slice(
    backend: Option<&dyn GpuBackend>,
    x: &[f32],
    w: &[f32],
    in_dim: usize,
    out_dim: usize,
) -> Vec<f32> {
    if let Some(gpu) = backend {
        if let Ok(tensor) = QuantizedTensor::from_f32_slice("slice_weight", &[out_dim, in_dim], w) {
            if let Ok(y) = gpu.compute_linear(x, &tensor, in_dim, out_dim) {
                return y;
            }
        }
    }
    linear(x, w, in_dim, out_dim)
}

pub fn dispatch_score_head(
    backend: Option<&dyn GpuBackend>,
    head: &QuantizedTensor,
    d: usize,
    vocab: usize,
    normed: &[f32],
    tokens: &[u32],
    layers_applied: usize,
) -> Result<DepthExit, ThmeError> {
    if let Some(gpu) = backend {
        if gpu.is_format_supported(head.dtype()) {
            match gpu.compute_score_head(normed, head, d, vocab) {
                Ok(logits) => {
                    return Ok(score_head_from_logits(
                        &logits,
                        vocab,
                        tokens,
                        layers_applied,
                    ));
                }
                Err(err) => {
                    eprintln!("Notice: GPU compute_score_head fallback to CPU SIMD: {err}");
                }
            }
        }
    }
    let head_f32 = head.as_f32_slice()?;
    Ok(score_head(
        head_f32,
        d,
        vocab,
        normed,
        tokens,
        layers_applied,
    ))
}

pub fn dispatch_score_head_slice(
    backend: Option<&dyn GpuBackend>,
    head: &[f32],
    d: usize,
    vocab: usize,
    normed: &[f32],
    tokens: &[u32],
    layers_applied: usize,
) -> DepthExit {
    if let Some(gpu) = backend {
        if let Ok(tensor) = QuantizedTensor::from_f32_slice("head_weight", &[vocab, d], head) {
            if let Ok(exit) =
                dispatch_score_head(Some(gpu), &tensor, d, vocab, normed, tokens, layers_applied)
            {
                return exit;
            }
        }
    }
    score_head(head, d, vocab, normed, tokens, layers_applied)
}

pub fn score_head_from_logits(
    logits: &[f32],
    vocab: usize,
    tokens: &[u32],
    layers_applied: usize,
) -> DepthExit {
    let rows = tokens.len() - 1;
    let mut nll = Vec::with_capacity(rows);
    let mut top1 = Vec::with_capacity(rows);

    for r in 0..rows {
        let row = &logits[r * vocab..(r + 1) * vocab];
        let target = tokens[r + 1] as usize;
        let mut max = f32::NEG_INFINITY;
        let mut best = 0usize;
        for (i, &l) in row.iter().enumerate() {
            if l > max {
                max = l;
                best = i;
            }
        }
        let sum: f64 = row.iter().map(|&l| f64::from((l - max).exp())).sum();
        let target_val = row.get(target).copied().unwrap_or(0.0);
        nll.push((f64::from(max - target_val) + sum.ln()) as f32);
        top1.push(best as u32);
    }

    DepthExit {
        layers_applied,
        nll,
        top1,
    }
}
