// SPDX-License-Identifier: AGPL-3.0-only
//! WGPU Compute Acceleration Backend for THME Tensor Forward Passes.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use wgpu::util::DeviceExt;

use super::decoder::{DecoderKernels, DecoderPlan};

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
    /// Input rows (tokens) processed by the dispatch.
    pub rows: u32,
    /// Offset, in floats, of this op's output block in the shared output buffer.
    pub out_offset: u32,
    pub _pad: [u32; 3],
}

/// Cumulative host-side wall-clock counters for the GPU path, read with [`GpuContext::stats`].
/// Times are nanoseconds on the host; they say where the host waited, not kernel time.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize)]
pub struct GpuStats {
    /// Compute dispatches encoded.
    pub dispatches: u64,
    /// Command buffer submissions (each one is followed by one wait and read-back).
    pub submits: u64,
    /// Weight buffers created on the device.
    pub weight_uploads: u64,
    /// Time creating and filling weight buffers.
    pub weight_upload_ns: u64,
    /// Time writing inputs and parameters to the device.
    pub input_write_ns: u64,
    /// Time creating bind groups and encoding commands.
    pub encode_ns: u64,
    /// Time inside `queue.submit`.
    pub submit_ns: u64,
    /// Time waiting for the device to finish and the map callback.
    pub wait_ns: u64,
    /// Time copying mapped results into host vectors.
    pub readback_ns: u64,
    /// Device time of the compute passes from timestamp queries; 0 unless
    /// [`GpuContext::set_kernel_timing`] is on and the adapter supports them.
    pub kernel_ns: u64,
}

#[derive(Default)]
pub(super) struct StatCounters {
    pub(super) dispatches: AtomicU64,
    pub(super) submits: AtomicU64,
    pub(super) weight_uploads: AtomicU64,
    pub(super) weight_upload_ns: AtomicU64,
    pub(super) input_write_ns: AtomicU64,
    pub(super) encode_ns: AtomicU64,
    pub(super) submit_ns: AtomicU64,
    pub(super) wait_ns: AtomicU64,
    pub(super) readback_ns: AtomicU64,
    pub(super) kernel_ns: AtomicU64,
}

pub(super) fn bump(counter: &AtomicU64, since: Instant) {
    counter.fetch_add(since.elapsed().as_nanos() as u64, Ordering::Relaxed);
}

/// Scratch device buffers, grown on demand and reused across calls.
struct Scratch {
    input: wgpu::Buffer,
    input_capacity: usize,
    output: wgpu::Buffer,
    output_capacity: usize,
    staging: wgpu::Buffer,
    staging_capacity: usize,
    params: wgpu::Buffer,
    params_capacity: usize,
}

/// Stride between per-dispatch parameter blocks (storage binding offsets must be aligned).
pub(super) const PARAM_STRIDE: usize = 256;
/// Largest workgroup count along one dispatch dimension that every adapter must accept.
pub(super) const MAX_GROUPS: u32 = 65_535;
/// Output rows and columns handled by one tiled matrix-matrix workgroup.
pub(super) const GEMM_TILE: u32 = 64;
/// Row counts at or below this use the matrix-vector kernel per row: a 64-row tile mostly
/// computes padding, and small grids leave the device idle.
pub(super) const GEMV_MAX_ROWS: usize = 8;
/// Weight bytes kept resident for slice-backed linears before the cache is flushed.
/// shortcut: flush-all, not LRU; switch to LRU if models above this size must stay fast.
const WEIGHT_CACHE_BUDGET: usize = 4 << 30;

pub(super) fn storage_buffer(
    device: &wgpu::Device,
    label: &str,
    bytes: usize,
    usage: wgpu::BufferUsages,
) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: bytes as u64,
        usage,
        mapped_at_creation: false,
    })
}

impl Scratch {
    fn new(device: &wgpu::Device) -> Self {
        let (x, y) = (256 << 10, 1 << 20);
        let in_use = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        Self {
            input: storage_buffer(device, "scratch_input", x, in_use),
            input_capacity: x,
            output: storage_buffer(
                device,
                "scratch_output",
                y,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            ),
            output_capacity: y,
            staging: storage_buffer(
                device,
                "scratch_staging",
                y,
                wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            ),
            staging_capacity: y,
            params: storage_buffer(device, "scratch_params", 16 * PARAM_STRIDE, in_use),
            params_capacity: 16 * PARAM_STRIDE,
        }
    }

    fn ensure(&mut self, device: &wgpu::Device, x_bytes: usize, y_bytes: usize, p_bytes: usize) {
        if x_bytes > self.input_capacity {
            self.input_capacity = x_bytes.next_power_of_two();
            self.input = storage_buffer(
                device,
                "scratch_input",
                self.input_capacity,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            );
        }
        if y_bytes > self.output_capacity {
            self.output_capacity = y_bytes.next_power_of_two();
            self.output = storage_buffer(
                device,
                "scratch_output",
                self.output_capacity,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            );
        }
        if y_bytes > self.staging_capacity {
            self.staging_capacity = y_bytes.next_power_of_two();
            self.staging = storage_buffer(
                device,
                "scratch_staging",
                self.staging_capacity,
                wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            );
        }
        if p_bytes > self.params_capacity {
            self.params_capacity = p_bytes.next_power_of_two();
            self.params = storage_buffer(
                device,
                "scratch_params",
                self.params_capacity,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            );
        }
    }
}

/// Two-slot timestamp query around a compute pass, with buffers to read it back.
pub(super) struct TimestampProbe {
    pub(super) queries: wgpu::QuerySet,
    pub(super) resolve: wgpu::Buffer,
    pub(super) staging: wgpu::Buffer,
    period_ns: f32,
}

impl TimestampProbe {
    fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        Self {
            queries: device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("kernel_time"),
                ty: wgpu::QueryType::Timestamp,
                count: 2,
            }),
            resolve: storage_buffer(
                device,
                "kernel_time_resolve",
                16,
                wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            ),
            staging: storage_buffer(
                device,
                "kernel_time_staging",
                16,
                wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            ),
            period_ns: queue.get_timestamp_period(),
        }
    }
}

struct CachedWeight {
    fingerprint: u64,
    buffer: Arc<wgpu::Buffer>,
}

#[derive(Default)]
struct WeightCache {
    entries: HashMap<(usize, usize), CachedWeight>,
    bytes: usize,
}

/// Cheap identity check for a weight slice: length plus 64 samples spread over it.
/// Guards against a freed model's address being reused by different weights.
fn fingerprint(w: &[f32]) -> u64 {
    let step = (w.len() / 64).max(1);
    let mut h: u64 = 0xcbf2_9ce4_8422_2325 ^ w.len() as u64;
    for v in w.iter().step_by(step) {
        h = (h ^ u64::from(v.to_bits())).wrapping_mul(0x0100_0000_01b3);
    }
    h
}

/// One linear in a batched submission: `Y = X * W^T` for the shared input rows.
struct LinearOp<'a> {
    kind: ShaderKind,
    weights: &'a wgpu::Buffer,
    out_dim: usize,
}

pub struct GpuContext {
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
    pub pipelines: HashMap<ShaderKind, wgpu::ComputePipeline>,
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub adapter_info: wgpu::AdapterInfo,
    pub(super) gemm_f32: wgpu::ComputePipeline,
    pub(super) decoder: DecoderKernels,
    pub(super) timestamps: Option<TimestampProbe>,
    pub(super) kernel_timing: AtomicBool,
    scratch: Mutex<Scratch>,
    weight_cache: Mutex<WeightCache>,
    pub(super) counters: StatCounters,
}

fn adapter_rank(info: &wgpu::AdapterInfo) -> (u8, u8) {
    let kind = match info.device_type {
        wgpu::DeviceType::DiscreteGpu => 0,
        wgpu::DeviceType::IntegratedGpu => 1,
        wgpu::DeviceType::VirtualGpu => 2,
        _ => 3,
    };
    let api = match info.backend {
        wgpu::Backend::Vulkan => 0,
        wgpu::Backend::Metal => 1,
        wgpu::Backend::Dx12 => 2,
        _ => 3,
    };
    (kind, api)
}

/// Picks a real GPU explicitly: discrete before integrated, Vulkan before Metal before DX12.
/// `THME_GPU_BACKEND=vulkan|metal|dx12` restricts the choice to one API.
fn select_adapter(instance: &wgpu::Instance) -> Result<wgpu::Adapter, ThmeError> {
    let only = std::env::var("THME_GPU_BACKEND")
        .ok()
        .map(|v| v.to_ascii_lowercase());
    pollster::block_on(instance.enumerate_adapters(wgpu::Backends::PRIMARY))
        .into_iter()
        .filter(|a| {
            let info = a.get_info();
            let api = format!("{:?}", info.backend).to_ascii_lowercase();
            info.device_type != wgpu::DeviceType::Cpu && only.as_ref().is_none_or(|o| *o == api)
        })
        .min_by_key(|a| adapter_rank(&a.get_info()))
        .ok_or_else(|| {
            ThmeError::GpuInitFailed("No suitable GPU adapter (Vulkan, DX12 or Metal)".into())
        })
}

pub(super) fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

impl GpuContext {
    /// Initializes WGPU on the best real adapter (Vulkan, Metal or DX12).
    pub fn init() -> Result<Self, ThmeError> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            flags: wgpu::InstanceFlags::empty(),
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let adapter = select_adapter(&instance)?;
        let adapter_info = adapter.get_info();

        let timestamp_support = adapter.features().contains(wgpu::Features::TIMESTAMP_QUERY);
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("thme_gpu_device"),
            required_features: if timestamp_support {
                wgpu::Features::TIMESTAMP_QUERY
            } else {
                wgpu::Features::empty()
            },
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
                storage_entry(0, true),
                storage_entry(1, true),
                storage_entry(2, true),
                storage_entry(3, false),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("gemv_pipeline_layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });
        let build = |label: &str, source: &str| {
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(source.into()),
            });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            })
        };

        let pipelines = HashMap::from([
            (
                ShaderKind::Q4_K,
                build("Q4_K", include_str!("shaders/gemv_q4_k.wgsl")),
            ),
            (
                ShaderKind::Q5_K,
                build("Q5_K", include_str!("shaders/gemv_q5_k.wgsl")),
            ),
            (
                ShaderKind::IQ4_XS,
                build("IQ4_XS", include_str!("shaders/gemv_iq4_xs.wgsl")),
            ),
            (
                ShaderKind::F32,
                build("F32", include_str!("shaders/gemv_f32.wgsl")),
            ),
        ]);
        let gemm_f32 = build("F32_gemm", include_str!("shaders/gemm_f32.wgsl"));
        let scratch = Mutex::new(Scratch::new(&device));
        let decoder = DecoderKernels::new(&device);
        let timestamps = timestamp_support.then(|| TimestampProbe::new(&device, &queue));

        Ok(Self {
            device,
            queue,
            pipelines,
            bind_group_layout,
            adapter_info,
            gemm_f32,
            decoder,
            timestamps,
            kernel_timing: AtomicBool::new(false),
            scratch,
            weight_cache: Mutex::new(WeightCache::default()),
            counters: StatCounters::default(),
        })
    }

    /// Snapshot of the cumulative host-side counters.
    pub fn stats(&self) -> GpuStats {
        let c = &self.counters;
        let get = |a: &AtomicU64| a.load(Ordering::Relaxed);
        GpuStats {
            dispatches: get(&c.dispatches),
            submits: get(&c.submits),
            weight_uploads: get(&c.weight_uploads),
            weight_upload_ns: get(&c.weight_upload_ns),
            input_write_ns: get(&c.input_write_ns),
            encode_ns: get(&c.encode_ns),
            submit_ns: get(&c.submit_ns),
            wait_ns: get(&c.wait_ns),
            readback_ns: get(&c.readback_ns),
            kernel_ns: get(&c.kernel_ns),
        }
    }

    /// True when the adapter offers timestamp queries, so kernel time can be measured.
    pub fn timestamp_query_supported(&self) -> bool {
        self.timestamps.is_some()
    }

    /// Turns per-submit kernel timing on or off (off by default; adds one tiny read-back per submit).
    pub fn set_kernel_timing(&self, on: bool) {
        self.kernel_timing.store(on, Ordering::Relaxed);
    }

    /// Drops every weight buffer cached for slice-backed linears (frees device memory).
    pub fn clear_weight_cache(&self) {
        if let Ok(mut cache) = self.weight_cache.lock() {
            *cache = WeightCache::default();
        }
    }

    /// Device buffer holding `w`, uploaded once per (address, length) and then reused.
    pub(super) fn resident_slice(&self, w: &[f32]) -> Result<Arc<wgpu::Buffer>, ThmeError> {
        let key = (w.as_ptr() as usize, w.len());
        let print = fingerprint(w);
        let mut cache = self
            .weight_cache
            .lock()
            .map_err(|_| ThmeError::ExecutionFailed("GPU weight cache mutex poisoned".into()))?;
        if let Some(hit) = cache.entries.get(&key) {
            if hit.fingerprint == print {
                return Ok(Arc::clone(&hit.buffer));
            }
        }
        let bytes = w.len() * 4;
        let limits = self.device.limits();
        let cap = limits
            .max_buffer_size
            .min(limits.max_storage_buffer_binding_size);
        if bytes as u64 > cap {
            return Err(ThmeError::BufferAllocationFailed {
                name: "slice_weight".into(),
                bytes,
                reason: format!("Tensor size {bytes} exceeds GPU limit {cap}"),
            });
        }
        if cache.bytes + bytes > WEIGHT_CACHE_BUDGET {
            *cache = WeightCache::default();
        }
        let t_up = Instant::now();
        let buffer = Arc::new(
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("gpu_weight_slice"),
                    contents: bytemuck::cast_slice(w),
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                }),
        );
        self.counters.weight_uploads.fetch_add(1, Ordering::Relaxed);
        bump(&self.counters.weight_upload_ns, t_up);
        let entry = CachedWeight {
            fingerprint: print,
            buffer: Arc::clone(&buffer),
        };
        if cache.entries.insert(key, entry).is_none() {
            cache.bytes += bytes;
        }
        Ok(buffer)
    }

    /// Reads the two timestamps of the submission that just finished and adds the difference.
    pub(super) fn read_kernel_time(&self, p: &TimestampProbe) -> Result<(), ThmeError> {
        let slice = p.staging.slice(..);
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
        let ticks = {
            let data = slice
                .get_mapped_range()
                .map_err(|e| ThmeError::DataTransferFailed(format!("GPU map range failed: {e}")))?;
            let t: &[u64] = bytemuck::cast_slice(&data);
            t[1].saturating_sub(t[0])
        };
        p.staging.unmap();
        self.counters.kernel_ns.fetch_add(
            (ticks as f64 * f64::from(p.period_ns)) as u64,
            Ordering::Relaxed,
        );
        Ok(())
    }

    /// Runs every op over the same `rows` input rows in ONE command buffer: one submit,
    /// one wait and one read-back, whatever the number of ops or rows.
    fn run_ops(
        &self,
        x: &[f32],
        rows: usize,
        in_dim: usize,
        ops: &[LinearOp<'_>],
    ) -> Result<Vec<Vec<f32>>, ThmeError> {
        if rows == 0 || ops.is_empty() {
            return Ok(ops.iter().map(|_| Vec::new()).collect());
        }
        let total_out: usize = ops.iter().map(|o| rows * o.out_dim).sum();
        let out_bytes = total_out * 4;
        let mut scratch = self
            .scratch
            .lock()
            .map_err(|_| ThmeError::ExecutionFailed("GPU scratch mutex poisoned".into()))?;
        scratch.ensure(
            &self.device,
            x.len() * 4,
            out_bytes,
            ops.len() * PARAM_STRIDE,
        );

        let mut params_bytes = vec![0u8; ops.len() * PARAM_STRIDE];
        let mut out_offset = 0usize;
        for (i, op) in ops.iter().enumerate() {
            let params = GemvParams {
                in_dim: in_dim as u32,
                out_dim: op.out_dim as u32,
                num_blocks_per_row: (in_dim / 256) as u32,
                rows: rows as u32,
                out_offset: out_offset as u32,
                _pad: [0; 3],
            };
            let at = i * PARAM_STRIDE;
            params_bytes[at..at + std::mem::size_of::<GemvParams>()]
                .copy_from_slice(bytemuck::bytes_of(&params));
            out_offset += rows * op.out_dim;
        }

        let t_in = Instant::now();
        self.queue
            .write_buffer(&scratch.input, 0, bytemuck::cast_slice(x));
        self.queue.write_buffer(&scratch.params, 0, &params_bytes);
        bump(&self.counters.input_write_ns, t_in);

        let t_enc = Instant::now();
        let probe = self
            .timestamps
            .as_ref()
            .filter(|_| self.kernel_timing.load(Ordering::Relaxed));
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("linear_batch_encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("linear_batch_pass"),
                timestamp_writes: probe.map(|p| wgpu::ComputePassTimestampWrites {
                    query_set: &p.queries,
                    beginning_of_pass_write_index: Some(0),
                    end_of_pass_write_index: Some(1),
                }),
            });
            for (i, op) in ops.iter().enumerate() {
                let use_gemm = op.kind == ShaderKind::F32 && rows > GEMV_MAX_ROWS;
                let pipeline = if use_gemm {
                    &self.gemm_f32
                } else {
                    self.pipelines.get(&op.kind).ok_or_else(|| {
                        ThmeError::ExecutionFailed(format!(
                            "Compute pipeline for {:?} not found",
                            op.kind
                        ))
                    })?
                };
                let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("linear_bind_group"),
                    layout: &self.bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                buffer: &scratch.params,
                                offset: (i * PARAM_STRIDE) as u64,
                                size: wgpu::BufferSize::new(
                                    std::mem::size_of::<GemvParams>() as u64
                                ),
                            }),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: scratch.input.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: op.weights.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: scratch.output.as_entire_binding(),
                        },
                    ],
                });
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                let out = op.out_dim as u32;
                if use_gemm {
                    pass.dispatch_workgroups(
                        out.div_ceil(GEMM_TILE),
                        (rows as u32).div_ceil(GEMM_TILE),
                        1,
                    );
                } else {
                    // Output rows spread over x then y; token rows ride on z.
                    pass.dispatch_workgroups(
                        out.min(MAX_GROUPS),
                        out.div_ceil(MAX_GROUPS),
                        rows as u32,
                    );
                }
            }
        }
        encoder.copy_buffer_to_buffer(&scratch.output, 0, &scratch.staging, 0, out_bytes as u64);
        if let Some(p) = probe {
            encoder.resolve_query_set(&p.queries, 0..2, &p.resolve, 0);
            encoder.copy_buffer_to_buffer(&p.resolve, 0, &p.staging, 0, 16);
        }
        let finished = encoder.finish();
        bump(&self.counters.encode_ns, t_enc);
        self.counters
            .dispatches
            .fetch_add(ops.len() as u64, Ordering::Relaxed);
        self.counters.submits.fetch_add(1, Ordering::Relaxed);

        let t_sub = Instant::now();
        self.queue.submit(Some(finished));
        bump(&self.counters.submit_ns, t_sub);

        let t_wait = Instant::now();
        let slice = scratch.staging.slice(..out_bytes as u64);
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
        bump(&self.counters.wait_ns, t_wait);
        if let Some(p) = probe {
            self.read_kernel_time(p)?;
        }

        let t_read = Instant::now();
        let data = slice
            .get_mapped_range()
            .map_err(|e| ThmeError::DataTransferFailed(format!("GPU map range failed: {e}")))?;
        let all: &[f32] = bytemuck::cast_slice(&data);
        let mut results = Vec::with_capacity(ops.len());
        let mut at = 0usize;
        for op in ops {
            let len = rows * op.out_dim;
            results.push(all[at..at + len].to_vec());
            at += len;
        }
        drop(data);
        scratch.staging.unmap();
        bump(&self.counters.readback_ns, t_read);
        Ok(results)
    }
}

fn shader_kind(dtype: QuantDtype) -> Result<ShaderKind, ThmeError> {
    match dtype {
        QuantDtype::Q4_K => Ok(ShaderKind::Q4_K),
        QuantDtype::Q5_K => Ok(ShaderKind::Q5_K),
        QuantDtype::IQ4_XS => Ok(ShaderKind::IQ4_XS),
        QuantDtype::F32 => Ok(ShaderKind::F32),
        other => Err(ThmeError::UnsupportedFormat {
            dtype: format!("{other:?}"),
            backend: "WGPU",
        }),
    }
}

pub(super) fn check_finite(y: &[f32], what: &str) -> Result<(), ThmeError> {
    match y.iter().position(|v| !v.is_finite()) {
        Some(index) => Err(ThmeError::NonFiniteOutput {
            what: what.to_string(),
            index,
        }),
        None => Ok(()),
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

    /// Applies several dense F32 linears (`(weights, out_dim)` each, row-major `[out_dim, in_dim]`)
    /// to the same input rows. Backends that can should run them in one submission.
    fn compute_linear_slices(
        &self,
        x: &[f32],
        ops: &[(&[f32], usize)],
        in_dim: usize,
    ) -> Result<Vec<Vec<f32>>, ThmeError> {
        ops.iter()
            .map(|(w, out_dim)| {
                let tensor =
                    QuantizedTensor::from_f32_slice("slice_weight", &[*out_dim, in_dim], w)?;
                self.compute_linear(x, &tensor, in_dim, *out_dim)
            })
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

    /// Runs a whole LLaMA-style decoder stack on the device and returns the last position's
    /// logits. `Ok(None)` means this backend or shape is unsupported; use the per-op path.
    fn forward_decoder(
        &self,
        _plan: &DecoderPlan<'_>,
        _embedded: &[f32],
    ) -> Result<Option<Vec<f32>>, ThmeError> {
        Ok(None)
    }

    /// Output head plus negative log-likelihood on the device for already normalized rows:
    /// returns one score per row but the last (`targets[r]` is row `r`'s next token) and reads
    /// back only those scores. `Ok(None)` means unsupported; use the per-op path.
    fn score_head_nll(
        &self,
        _normed: &[f32],
        _head: &[f32],
        _d: usize,
        _vocab: usize,
        _targets: &[u32],
    ) -> Result<Option<Vec<f32>>, ThmeError> {
        Ok(None)
    }

    fn is_format_supported(&self, dtype: QuantDtype) -> bool;
    fn device_name(&self) -> &str;

    /// Adapter details for measured reports (name, graphics API, driver).
    fn adapter_json(&self) -> serde_json::Value {
        serde_json::json!({ "name": self.device_name() })
    }
}

impl GpuBackend for GpuContext {
    fn compute_linear(
        &self,
        x: &[f32],
        weights: &QuantizedTensor,
        in_dim: usize,
        out_dim: usize,
    ) -> Result<Vec<f32>, ThmeError> {
        if in_dim == 0 || !x.len().is_multiple_of(in_dim) {
            return Err(ThmeError::DimensionMismatch {
                expected: format!("Multiple of in_dim {in_dim}"),
                found: format!("{}", x.len()),
            });
        }
        let kind = shader_kind(weights.dtype())?;
        let t_up = Instant::now();
        let fresh = !weights.has_gpu_buffer();
        let buffer = weights.get_or_create_gpu_buffer(&self.device)?;
        if fresh {
            self.counters.weight_uploads.fetch_add(1, Ordering::Relaxed);
            bump(&self.counters.weight_upload_ns, t_up);
        }
        let rows = x.len() / in_dim;
        let op = LinearOp {
            kind,
            weights: buffer,
            out_dim,
        };
        let mut out = self.run_ops(x, rows, in_dim, &[op])?;
        let y = out.pop().unwrap_or_default();
        check_finite(&y, &format!("linear projection '{}'", weights.name()))?;
        Ok(y)
    }

    fn compute_linear_slices(
        &self,
        x: &[f32],
        ops: &[(&[f32], usize)],
        in_dim: usize,
    ) -> Result<Vec<Vec<f32>>, ThmeError> {
        if in_dim == 0 || !x.len().is_multiple_of(in_dim) {
            return Err(ThmeError::DimensionMismatch {
                expected: format!("Multiple of in_dim {in_dim}"),
                found: format!("{}", x.len()),
            });
        }
        if let Some((w, out_dim)) = ops.iter().find(|(w, o)| w.len() != o * in_dim) {
            return Err(ThmeError::DimensionMismatch {
                expected: format!(
                    "{} weights for out_dim {out_dim} x in_dim {in_dim}",
                    out_dim * in_dim
                ),
                found: format!("{}", w.len()),
            });
        }
        let buffers = ops
            .iter()
            .map(|(w, _)| self.resident_slice(w))
            .collect::<Result<Vec<_>, _>>()?;
        let linear_ops: Vec<LinearOp<'_>> = buffers
            .iter()
            .zip(ops)
            .map(|(buffer, (_, out_dim))| LinearOp {
                kind: ShaderKind::F32,
                weights: buffer,
                out_dim: *out_dim,
            })
            .collect();
        let results = self.run_ops(x, x.len() / in_dim, in_dim, &linear_ops)?;
        for y in &results {
            check_finite(y, "linear projection (slice weights)")?;
        }
        Ok(results)
    }

    fn forward_decoder(
        &self,
        plan: &DecoderPlan<'_>,
        embedded: &[f32],
    ) -> Result<Option<Vec<f32>>, ThmeError> {
        GpuContext::forward_decoder(self, plan, embedded)
    }

    fn score_head_nll(
        &self,
        normed: &[f32],
        head: &[f32],
        d: usize,
        vocab: usize,
        targets: &[u32],
    ) -> Result<Option<Vec<f32>>, ThmeError> {
        GpuContext::score_head_nll(self, normed, head, d, vocab, targets)
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

    fn adapter_json(&self) -> serde_json::Value {
        let info = &self.adapter_info;
        serde_json::json!({
            "name": info.name,
            "backend": format!("{:?}", info.backend),
            "device_type": format!("{:?}", info.device_type),
            "driver": info.driver,
            "driver_info": info.driver_info,
        })
    }
}

/// Applies several linears that share one input. On the GPU they go in a single submission;
/// any GPU failure falls back to the CPU path for the whole group.
pub fn dispatch_linear_slices(
    backend: Option<&dyn GpuBackend>,
    x: &[f32],
    ops: &[(&[f32], usize)],
    in_dim: usize,
) -> Vec<Vec<f32>> {
    if let Some(gpu) = backend {
        match gpu.compute_linear_slices(x, ops, in_dim) {
            Ok(ys) => return ys,
            Err(err) => eprintln!("Notice: GPU linear fallback to CPU SIMD: {err}"),
        }
    }
    ops.iter()
        .map(|(w, out_dim)| linear(x, w, in_dim, *out_dim))
        .collect()
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

/// Fixed-size form of [`dispatch_linear_slices`] for call sites that destructure the results.
pub fn dispatch_linear_array<const N: usize>(
    backend: Option<&dyn GpuBackend>,
    x: &[f32],
    ops: [(&[f32], usize); N],
    in_dim: usize,
) -> [Vec<f32>; N] {
    let mut results = dispatch_linear_slices(backend, x, &ops, in_dim).into_iter();
    std::array::from_fn(|_| results.next().unwrap_or_default())
}

pub fn dispatch_linear_slice(
    backend: Option<&dyn GpuBackend>,
    x: &[f32],
    w: &[f32],
    in_dim: usize,
    out_dim: usize,
) -> Vec<f32> {
    dispatch_linear_slices(backend, x, &[(w, out_dim)], in_dim)
        .pop()
        .unwrap_or_default()
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
        match gpu.compute_linear_slices(normed, &[(head, vocab)], d) {
            Ok(mut logits) => {
                if let Some(logits) = logits.pop() {
                    return score_head_from_logits(&logits, vocab, tokens, layers_applied);
                }
            }
            Err(err) => eprintln!("Notice: GPU score head fallback to CPU SIMD: {err}"),
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
