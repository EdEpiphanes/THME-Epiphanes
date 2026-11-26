// SPDX-License-Identifier: AGPL-3.0-only
//! WGPU Compute Acceleration Backend for THME Tensor Forward Passes.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
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
pub enum GpuDispatchMode {
    /// Dispatch all prompt tokens in one batched GPU submission (`wid.y`).
    #[default]
    Batched,
    /// Dispatch each token row in its own single-row GPU submission.
    PerRow,
}

static GLOBAL_PER_ROW_DISPATCH: AtomicBool = AtomicBool::new(false);

impl GpuDispatchMode {
    /// Applies this dispatch mode globally for newly initialized `GpuContext`s.
    pub fn apply(self) {
        GLOBAL_PER_ROW_DISPATCH.store(matches!(self, Self::PerRow), Ordering::Relaxed);
    }
}

#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ShaderKind {
    Q4_K,
    Q5_K,
    IQ4_XS,
    F32,
    Bitcos,
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
    per_row_dispatch: AtomicBool,
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
            (ShaderKind::Bitcos, include_str!("shaders/gemv_bitcos.wgsl")),
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

        let env_per_row = std::env::var("THME_GPU_PER_ROW")
            .map(|v| matches!(v.as_str(), "1" | "true" | "TRUE"))
            .unwrap_or(false);
        let per_row = env_per_row || GLOBAL_PER_ROW_DISPATCH.load(Ordering::Relaxed);

        Ok(Self {
            device,
            queue,
            pipelines,
            bind_group_layout,
            adapter_info,
            buffer_pool,
            per_row_dispatch: AtomicBool::new(per_row),
        })
    }

    /// Toggles between single-submission batched dispatch (`false`) and per-row dispatch (`true`).
    pub fn set_per_row_dispatch(&self, per_row: bool) {
        self.per_row_dispatch.store(per_row, Ordering::Relaxed);
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
        self.dispatch_gemv_batched(kind, x, weights_buf, in_dim, out_dim, 1)
    }

    /// Dispatches a batched GEMV/GEMM (`rows` tokens in one GPU command submission & readback).
    pub fn dispatch_gemv_batched(
        &self,
        kind: ShaderKind,
        x: &[f32],
        weights_buf: &wgpu::Buffer,
        in_dim: usize,
        out_dim: usize,
        rows: usize,
    ) -> Result<Vec<f32>, ThmeError> {
        let pipeline = self.pipelines.get(&kind).ok_or_else(|| {
            ThmeError::ExecutionFailed(format!("Compute pipeline for {kind:?} not found"))
        })?;

        let x_bytes = x.len() * 4;
        let y_bytes = rows * out_dim * 4;

        let (num_blocks, stride_bytes) = match kind {
            ShaderKind::Q4_K | ShaderKind::Q5_K | ShaderKind::IQ4_XS => (in_dim / 256, 0),
            ShaderKind::Bitcos => (in_dim.div_ceil(1024), in_dim as u32),
            ShaderKind::F32 => (0, 0),
        };

        let params = GemvParams {
            in_dim: in_dim as u32,
            out_dim: out_dim as u32,
            num_blocks_per_row: num_blocks as u32,
            stride_bytes,
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
            cpass.dispatch_workgroups(out_dim as u32, rows as u32, 1);
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

    /// Executes multiple projections sharing the same `x` input in a single GPU submission.
    fn compute_linear_multi(
        &self,
        x: &[f32],
        specs: &[(&QuantizedTensor, usize)],
        in_dim: usize,
    ) -> Result<Vec<Vec<f32>>, ThmeError> {
        specs
            .iter()
            .map(|(w, out_dim)| self.compute_linear(x, w, in_dim, *out_dim))
            .collect()
    }

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
            QuantDtype::Bitcos => ShaderKind::Bitcos,
            other => {
                return Err(ThmeError::UnsupportedFormat {
                    dtype: format!("{other:?}"),
                    backend: "WGPU/Vulkan",
                });
            }
        };

        let weights_buf = weights.get_or_create_gpu_buffer(&self.device)?;
        let rows = x.len() / in_dim;

        let use_batched = !self.per_row_dispatch.load(Ordering::Relaxed);
        if rows == 1 || use_batched {
            let res = self.dispatch_gemv_batched(kind, x, weights_buf, in_dim, out_dim, rows)?;
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

    fn compute_linear_multi(
        &self,
        x: &[f32],
        specs: &[(&QuantizedTensor, usize)],
        in_dim: usize,
    ) -> Result<Vec<Vec<f32>>, ThmeError> {
        if specs.is_empty() || x.is_empty() || !x.len().is_multiple_of(in_dim) {
            return specs
                .iter()
                .map(|(w, out_dim)| self.compute_linear(x, w, in_dim, *out_dim))
                .collect();
        }
        if self.per_row_dispatch.load(Ordering::Relaxed) {
            return specs
                .iter()
                .map(|(w, out_dim)| self.compute_linear(x, w, in_dim, *out_dim))
                .collect();
        }

        let rows = x.len() / in_dim;
        let x_bytes = x.len() * 4;
        let mut total_y_bytes = 0usize;
        let mut pass_meta = Vec::with_capacity(specs.len());

        for &(weights, out_dim) in specs {
            let kind = match weights.dtype() {
                QuantDtype::Q4_K => ShaderKind::Q4_K,
                QuantDtype::Q5_K => ShaderKind::Q5_K,
                QuantDtype::IQ4_XS => ShaderKind::IQ4_XS,
                QuantDtype::F32 => ShaderKind::F32,
                QuantDtype::Bitcos => ShaderKind::Bitcos,
                other => {
                    return Err(ThmeError::UnsupportedFormat {
                        dtype: format!("{other:?}"),
                        backend: "WGPU/Vulkan",
                    });
                }
            };
            let pipeline = self.pipelines.get(&kind).ok_or_else(|| {
                ThmeError::ExecutionFailed(format!("Compute pipeline for {kind:?} not found"))
            })?;
            let weights_buf = weights.get_or_create_gpu_buffer(&self.device)?;
            let (num_blocks, stride_bytes) = match kind {
                ShaderKind::Q4_K | ShaderKind::Q5_K | ShaderKind::IQ4_XS => (in_dim / 256, 0),
                ShaderKind::Bitcos => (in_dim.div_ceil(1024), in_dim as u32),
                ShaderKind::F32 => (0, 0),
            };
            let params = GemvParams {
                in_dim: in_dim as u32,
                out_dim: out_dim as u32,
                num_blocks_per_row: num_blocks as u32,
                stride_bytes,
            };
            let y_bytes = rows * out_dim * 4;
            let offset = total_y_bytes;
            total_y_bytes += y_bytes;
            pass_meta.push((pipeline, weights_buf, params, out_dim, y_bytes, offset));
        }

        let mut pool = self
            .buffer_pool
            .lock()
            .map_err(|_| ThmeError::ExecutionFailed("GPU buffer pool mutex poisoned".into()))?;

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
        if total_y_bytes > pool.staging_capacity {
            let new_cap = total_y_bytes.next_power_of_two();
            pool.staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("pool_staging_buf_resized"),
                size: new_cap as u64,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            pool.staging_capacity = new_cap;
        }

        // Upload x ONCE for all projections in this group
        self.queue
            .write_buffer(&pool.input_buffer, 0, bytemuck::cast_slice(x));

        use wgpu::util::DeviceExt;
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("gemv_multi_encoder"),
            });

        for &(pipeline, weights_buf, params, out_dim, y_bytes, offset) in &pass_meta {
            let p_buf = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("gemv_multi_params"),
                    contents: bytemuck::bytes_of(&params),
                    usage: wgpu::BufferUsages::STORAGE,
                });
            let out_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("gemv_multi_out"),
                size: y_bytes as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("gemv_multi_bg"),
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: p_buf.as_entire_binding(),
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
                        resource: out_buf.as_entire_binding(),
                    },
                ],
            });
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("gemv_multi_pass"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(pipeline);
                cpass.set_bind_group(0, &bg, &[]);
                cpass.dispatch_workgroups(out_dim as u32, rows as u32, 1);
            }
            encoder.copy_buffer_to_buffer(
                &out_buf,
                0,
                &pool.staging_buffer,
                offset as u64,
                y_bytes as u64,
            );
        }

        self.queue.submit(Some(encoder.finish()));

        let slice = pool.staging_buffer.slice(..total_y_bytes as u64);
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
        let flat: &[f32] = bytemuck::cast_slice(&data);
        let mut results = Vec::with_capacity(pass_meta.len());
        let mut elem_cursor = 0usize;
        for &(_, _, _, out_dim, _, _) in &pass_meta {
            let elems = rows * out_dim;
            results.push(flat[elem_cursor..elem_cursor + elems].to_vec());
            elem_cursor += elems;
        }
        drop(data);
        pool.staging_buffer.unmap();
        Ok(results)
    }

    fn is_format_supported(&self, dtype: QuantDtype) -> bool {
        matches!(
            dtype,
            QuantDtype::Q4_K
                | QuantDtype::Q5_K
                | QuantDtype::IQ4_XS
                | QuantDtype::F32
                | QuantDtype::Bitcos
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

/// Dispatches multiple linear projections sharing `x` in one GPU submission, falling back to CPU.
pub fn dispatch_linear_multi_slice(
    backend: Option<&dyn GpuBackend>,
    x: &[f32],
    slices: &[(&[f32], usize)],
    in_dim: usize,
) -> Vec<Vec<f32>> {
    if let Some(gpu) = backend {
        let mut tensors = Vec::with_capacity(slices.len());
        let mut ok = true;
        for &(w, out_dim) in slices {
            match QuantizedTensor::from_f32_slice("slice_weight", &[out_dim, in_dim], w) {
                Ok(t) => tensors.push((t, out_dim)),
                Err(_) => {
                    ok = false;
                    break;
                }
            }
        }
        if ok {
            let refs: Vec<(&QuantizedTensor, usize)> =
                tensors.iter().map(|(t, od)| (t, *od)).collect();
            if let Ok(ys) = gpu.compute_linear_multi(x, &refs, in_dim) {
                return ys;
            }
        }
    }
    slices
        .iter()
        .map(|&(w, out_dim)| linear(x, w, in_dim, out_dim))
        .collect()
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bitcos_dtype_and_shader_kind_registered() {
        assert_eq!(
            QuantDtype::from_str_name("BITCOS"),
            Some(QuantDtype::Bitcos)
        );
        assert_eq!(QuantDtype::Bitcos.block_size(), 1024);
        assert_eq!(QuantDtype::Bitcos.block_bytes(), 256);
        assert_ne!(ShaderKind::Bitcos, ShaderKind::F32);
    }

    #[test]
    fn vulkan_bitcos_shader_executes_on_adapter_if_present() {
        let Ok(ctx) = GpuContext::init() else {
            return;
        };
        let in_dim = 16usize;
        let out_dim = 1usize;
        let mut packed_word = 0u32;
        let mut x = vec![0.0f32; in_dim];
        let mut expected = 0.0f32;
        for (j, x_j) in x.iter_mut().enumerate() {
            *x_j = (j as f32) + 1.0;
            let code = match j % 3 {
                0 => {
                    expected += *x_j;
                    1u32
                }
                1 => {
                    expected -= *x_j;
                    2u32
                }
                _ => 0u32,
            };
            packed_word |= code << (j * 2);
        }
        let raw_bytes: Arc<[u8]> = Arc::from(packed_word.to_le_bytes().to_vec());
        let tensor = QuantizedTensor::new_quantized(
            "test_bitcos",
            vec![out_dim, in_dim],
            QuantDtype::Bitcos,
            raw_bytes,
        )
        .unwrap();
        let y = ctx.compute_linear(&x, &tensor, in_dim, out_dim).unwrap();
        assert_eq!(y.len(), 1);
        assert!(
            (y[0] - expected).abs() < 1e-4,
            "gpu={} expected={}",
            y[0],
            expected
        );

        // Batched 2-row dispatch in a single GPU submission
        let mut x2 = x.clone();
        x2.extend(x.iter().map(|v| v * 2.0));
        let y2 = ctx.compute_linear(&x2, &tensor, in_dim, out_dim).unwrap();
        assert_eq!(y2.len(), 2);
        assert!((y2[0] - expected).abs() < 1e-4);
        assert!((y2[1] - expected * 2.0).abs() < 1e-4);

        // Toggle per-row dispatch and verify identical output
        ctx.set_per_row_dispatch(true);
        let y2_per_row = ctx.compute_linear(&x2, &tensor, in_dim, out_dim).unwrap();
        assert_eq!(y2_per_row.len(), 2);
        assert!((y2_per_row[0] - expected).abs() < 1e-4);
        assert!((y2_per_row[1] - expected * 2.0).abs() < 1e-4);
    }
}
