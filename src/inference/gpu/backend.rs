// SPDX-License-Identifier: AGPL-3.0-only
//! WGPU Compute Acceleration Backend for THME Tensor Forward Passes.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use wgpu::util::DeviceExt;

use super::decoder::{DecoderKernels, DecoderPlan};
use super::dense::DenseKernel;

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

/// How dense weights are stored on the device. Activations and accumulation stay f32.
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum WeightStorage {
    #[default]
    F32,
    F16,
    Q8_0,
    Q4_0,
}

/// Device time of one dispatch from a profiled decoder pass.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct OpSample {
    /// Kernel name; linears carry their kernel and `rows x in x out` shape.
    pub label: String,
    pub ns: u64,
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
/// Output rows and columns handled by one register-blocked (vec4) workgroup.
pub(super) const GEMM_V4_TILE: u32 = 128;
/// Shared memory the register-blocked kernel needs: two slices of 16 x 33 vec4.
const GEMM_V4_SHARED_BYTES: u32 = 2 * 528 * 16;
/// Row counts at or below this use the matrix-vector kernel per row: a 64-row tile mostly
/// computes padding, and small grids leave the device idle.
/// Kernel name reported for the cooperative product that reads f16 weights and activations in
/// place; callers convert the activations first with `coop_convert_x`.
pub(super) const COOP_DIRECT_KERNEL: &str = "gemm_coop_f16";
/// Kernel name of the integer dot product on Q8_0 and Q4_0 weights; callers quantize the
/// activations first with `convert_activations`.
pub(super) const Q8DOT_KERNEL: &str = "gemm_q8dot";

/// Which activation copy a kernel reads at binding 1.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum ActivationCopy {
    F16,
    Int8,
}

impl ActivationCopy {
    /// The copy the named kernel needs, if any.
    pub(super) fn for_kernel(name: &str) -> Option<Self> {
        match name {
            COOP_DIRECT_KERNEL => Some(Self::F16),
            Q8DOT_KERNEL => Some(Self::Int8),
            _ => None,
        }
    }
}

pub(super) const GEMV_MAX_ROWS: usize = 8;
/// Default weight bytes kept resident for slice-backed linears before LRU eviction;
/// raise it with [`GpuContext::set_weight_budget`] when the card has the memory.
const WEIGHT_CACHE_BUDGET: usize = 10 << 30;

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
    bytes: usize,
    last_used: u64,
}

#[derive(Default)]
struct WeightCache {
    entries: HashMap<(usize, usize, u8), CachedWeight>,
    bytes: usize,
    tick: u64,
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

/// A weight matrix on the device with the storage format its kernel must decode.
#[derive(Clone)]
pub(super) struct DeviceWeight {
    pub(super) buffer: Arc<wgpu::Buffer>,
    pub(super) storage: WeightStorage,
}

/// One linear in a batched submission: `Y = X * W^T` for the shared input rows.
struct LinearOp<'a> {
    kind: ShaderKind,
    storage: WeightStorage,
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
    /// Register-blocked kernel; `None` when the device's workgroup memory is too small.
    pub(super) gemm_v4: Option<wgpu::ComputePipeline>,
    /// Kernels for f16 / Q8_0 / Q4_0 weights, compiled on first use.
    dense: Mutex<HashMap<(DenseKernel, WeightStorage), wgpu::ComputePipeline>>,
    dense_layout: wgpu::PipelineLayout,
    pub(super) decoder: DecoderKernels,
    pub(super) timestamps: Option<TimestampProbe>,
    pub(super) kernel_timing: AtomicBool,
    shader_f16: bool,
    /// The device was created with cooperative matrices and the kernel passed its check.
    coop_ready: AtomicBool,
    /// Cooperative matrices for every weight storage (opt-in: f16 inputs round the result).
    use_coop: AtomicBool,
    /// Cooperative matrices for f16 weight storage, whose values are f16 already (default).
    coop_f16_default: AtomicBool,
    use_coop_direct: AtomicBool,
    use_int8_dot: AtomicBool,
    use_coop_attention: AtomicBool,
    /// f16 copy of the activations for the in-place cooperative kernel, grown on demand.
    x16: Mutex<Option<wgpu::Buffer>>,
    use_gemm_v4: AtomicBool,
    use_tiled_attention: AtomicBool,
    weight_budget: std::sync::atomic::AtomicUsize,
    pub(super) op_profile: AtomicBool,
    weight_storage: std::sync::atomic::AtomicU8,
    op_samples: Mutex<Vec<OpSample>>,
    wait_timeout_ms: std::sync::atomic::AtomicU64,
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

/// Cooperative matrices with f16 inputs, f32 accumulate and 16 x 16 x 16 tiles on a 32-lane
/// subgroup, which is what `gemm_coop.wgsl` is written for. `THME_GPU_COOP=0` turns it off.
fn coop_matrix_supported(adapter: &wgpu::Adapter) -> bool {
    use wgpu::CooperativeScalarType::{F16, F32};
    let info = adapter.get_info();
    std::env::var("THME_GPU_COOP").ok().as_deref() != Some("0")
        && adapter
            .features()
            .contains(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX | wgpu::Features::SHADER_F16)
        && info.subgroup_min_size == 32
        && info.subgroup_max_size == 32
        && adapter.cooperative_matrix_properties().iter().any(|c| {
            (c.m_size, c.n_size, c.k_size) == (16, 16, 16) && c.ab_type == F16 && c.cr_type == F32
        })
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
        let coop_wanted = coop_matrix_supported(&adapter);
        let mut features = if timestamp_support {
            wgpu::Features::TIMESTAMP_QUERY
        } else {
            wgpu::Features::empty()
        };
        if coop_wanted {
            features |=
                wgpu::Features::SHADER_F16 | wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX;
        }
        // SAFETY: the experimental token is only given when cooperative matrices are requested;
        // the kernel is checked against a CPU result before it is ever used (`coop_probe`).
        let experimental = if coop_wanted {
            unsafe { wgpu::ExperimentalFeatures::enabled() }
        } else {
            wgpu::ExperimentalFeatures::disabled()
        };
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("thme_gpu_device"),
            required_features: features,
            experimental_features: experimental,
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
        let dense_layout = pipeline_layout.clone();
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
        let gemm_v4 = (device.limits().max_compute_workgroup_storage_size >= GEMM_V4_SHARED_BYTES)
            .then(|| {
                build(
                    "F32_gemm_v4",
                    &super::dense::kernel_source(DenseKernel::Gemm, WeightStorage::F32),
                )
            });
        let scratch = Mutex::new(Scratch::new(&device));
        let decoder = DecoderKernels::new(&device);
        let timestamps = timestamp_support.then(|| TimestampProbe::new(&device, &queue));

        let ctx = Self {
            device,
            queue,
            pipelines,
            bind_group_layout,
            adapter_info,
            gemm_f32,
            gemm_v4,
            dense: Mutex::new(HashMap::new()),
            dense_layout,
            decoder,
            timestamps,
            kernel_timing: AtomicBool::new(false),
            shader_f16: coop_wanted,
            coop_ready: AtomicBool::new(false),
            use_coop: AtomicBool::new(false),
            coop_f16_default: AtomicBool::new(true),
            use_coop_direct: AtomicBool::new(true),
            use_int8_dot: AtomicBool::new(false),
            use_coop_attention: AtomicBool::new(true),
            x16: Mutex::new(None),
            use_gemm_v4: AtomicBool::new(true),
            use_tiled_attention: AtomicBool::new(true),
            weight_budget: std::sync::atomic::AtomicUsize::new(WEIGHT_CACHE_BUDGET),
            op_profile: AtomicBool::new(false),
            weight_storage: std::sync::atomic::AtomicU8::new(0),
            op_samples: Mutex::new(Vec::new()),
            wait_timeout_ms: std::sync::atomic::AtomicU64::new(120_000),
            scratch,
            weight_cache: Mutex::new(WeightCache::default()),
            counters: StatCounters::default(),
        };
        ctx.self_test()?;
        if coop_wanted {
            ctx.coop_probe();
        }
        match std::env::var("THME_GPU_STORAGE").ok().as_deref() {
            Some("f16") => ctx.set_weight_storage(WeightStorage::F16),
            Some("f32") => ctx.set_weight_storage(WeightStorage::F32),
            _ => {}
        }
        if std::env::var("THME_GPU_COOP").ok().as_deref() == Some("1") && ctx.coop_matrix_ready() {
            ctx.set_coop_matrix(true);
        }
        Ok(ctx)
    }

    /// Compiles the cooperative-matrix kernel and checks it on tile-edge shapes against a CPU
    /// product. On any failure the kernel stays off and the f32 kernels keep running.
    fn coop_probe(&self) {
        let (rows, in_dim, out_dim) = (150usize, 100usize, 132usize);
        let x: Vec<f32> = (0..rows * in_dim)
            .map(|i| ((i * 7 % 13) as f32 - 6.0) / 8.0)
            .collect();
        let w: Vec<f32> = (0..out_dim * in_dim)
            .map(|i| ((i * 5 % 11) as f32 - 5.0) / 8.0)
            .collect();
        let Ok(weights) = QuantizedTensor::new_f32("coop_probe", vec![out_dim, in_dim], w.clone())
        else {
            return;
        };
        self.coop_ready.store(true, Ordering::Relaxed);
        self.use_coop.store(true, Ordering::Relaxed);
        let ok = self
            .compute_linear(&x, &weights, in_dim, out_dim)
            .is_ok_and(|got| {
                (0..rows).all(|r| {
                    (0..out_dim).all(|o| {
                        let want: f32 = (0..in_dim)
                            .map(|j| x[r * in_dim + j] * w[o * in_dim + j])
                            .sum();
                        let have = got.get(r * out_dim + o).copied().unwrap_or(f32::NAN);
                        (have - want).abs() <= 2e-2 * want.abs().max(1.0)
                    })
                })
            });
        self.use_coop.store(false, Ordering::Relaxed);
        self.coop_ready.store(ok, Ordering::Relaxed);
        if !ok {
            return;
        }
        // The in-place f16 kernel and the attention kernel compile and run here too, or stay off.
        let stored = self.weight_storage();
        self.set_weight_storage(WeightStorage::F16);
        self.use_coop.store(true, Ordering::Relaxed);
        let (rows, in_dim, out_dim) = (150usize, 96usize, 144usize);
        let x: Vec<f32> = (0..rows * in_dim)
            .map(|i| ((i * 7 % 13) as f32 - 6.0) / 8.0)
            .collect();
        let w: Vec<f32> = (0..out_dim * in_dim)
            .map(|i| ((i * 5 % 11) as f32 - 5.0) / 8.0)
            .collect();
        let direct_ok =
            QuantizedTensor::new_f32("coop_probe_f16", vec![out_dim, in_dim], w.clone())
                .and_then(|t| self.compute_linear(&x, &t, in_dim, out_dim))
                .is_ok_and(|got| {
                    (0..rows).all(|r| {
                        (0..out_dim).all(|o| {
                            let want: f32 = (0..in_dim)
                                .map(|j| x[r * in_dim + j] * w[o * in_dim + j])
                                .sum();
                            let have = got.get(r * out_dim + o).copied().unwrap_or(f32::NAN);
                            (have - want).abs() <= 2e-2 * want.abs().max(1.0)
                        })
                    })
                });
        self.use_coop.store(false, Ordering::Relaxed);
        self.set_weight_storage(stored);
        self.use_coop_direct.store(direct_ok, Ordering::Relaxed);
        let scope = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let attention = self.decoder.coop_attention_pipeline(&self.device);
        let compiled = pollster::block_on(scope.pop()).is_none() && attention.is_ok();
        self.use_coop_attention.store(compiled, Ordering::Relaxed);
    }

    /// Whether cooperative matrices run for tensors stored as `storage`.
    fn coop_active(&self, storage: WeightStorage) -> bool {
        self.coop_matrix_ready()
            && (self.use_coop.load(Ordering::Relaxed)
                || (storage == WeightStorage::F16 && self.coop_f16_default.load(Ordering::Relaxed)))
    }

    /// Cooperative-matrix kernels are usable on this adapter.
    pub fn coop_matrix_ready(&self) -> bool {
        self.coop_ready.load(Ordering::Relaxed)
    }

    /// Records the conversion of the f32 activations `x` (rows x in_dim) into the shared
    /// scratch buffer (f16 with rows padded to 16, or int8 blocks) and returns that buffer for
    /// binding 1 of the matching product. `params` is any GemvParams binding with the same rows and
    /// in_dim; `spare` fills the layout's unused weights slot.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn convert_activations(
        &self,
        copy: ActivationCopy,
        pass: &mut wgpu::ComputePass<'_>,
        params: wgpu::BindingResource<'_>,
        x: wgpu::BindingResource<'_>,
        spare: wgpu::BindingResource<'_>,
        rows: usize,
        in_dim: usize,
    ) -> Result<wgpu::Buffer, ThmeError> {
        // f16: one thread per vec4 of the padded rows; int8: one thread per block of 32.
        let (threads, bytes, kernel) = match copy {
            ActivationCopy::F16 => {
                let vec4s = rows.next_multiple_of(16) * in_dim / 4;
                (vec4s, (vec4s * 8) as u64, DenseKernel::ConvertX)
            }
            ActivationCopy::Int8 => {
                let blocks = rows * (in_dim / 32);
                (blocks, (blocks * 36) as u64, DenseKernel::ConvertQ8)
            }
        };
        let groups = threads.div_ceil(256);
        let (gx, gy) = (groups.min(65535) as u32, groups.div_ceil(65535) as u32);
        let buffer = {
            let mut slot = self
                .x16
                .lock()
                .map_err(|_| ThmeError::ExecutionFailed("f16 scratch poisoned".into()))?;
            match slot.as_ref().filter(|b| b.size() >= bytes) {
                Some(b) => b.clone(),
                None => {
                    let b = self.device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("x_f16_scratch"),
                        size: bytes.next_multiple_of(1 << 20),
                        usage: wgpu::BufferUsages::STORAGE,
                        mapped_at_creation: false,
                    });
                    *slot = Some(b.clone());
                    b
                }
            }
        };
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("x_to_f16"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params,
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: x,
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: spare,
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: buffer.as_entire_binding(),
                },
            ],
        });
        pass.set_pipeline(&self.dense_pipeline(kernel, WeightStorage::F32)?);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(gx, gy, 1);
        Ok(buffer)
    }

    /// Runs Q8_0 and Q4_0 weights on packed 4 x 8-bit integer dot products, quantizing the
    /// activations per block of 32 to int8 (off by default).
    pub fn set_int8_dot(&self, on: bool) {
        self.use_int8_dot.store(on, Ordering::Relaxed);
    }

    /// Lets head size 64 use the cooperative-matrix attention when the matmul does (on by default).
    pub fn set_coop_attention(&self, on: bool) {
        self.use_coop_attention.store(on, Ordering::Relaxed);
    }

    /// Lets f16 weights use the kernel that reads them in place (on by default).
    pub fn set_coop_direct_f16(&self, on: bool) {
        self.use_coop_direct.store(on, Ordering::Relaxed);
    }

    /// Turns the cooperative-matrix matmul on for every weight storage, or off for all of them
    /// including f16 (ignored when not ready). Without a call, f16 weights use it and the other
    /// storages keep the exact f32 kernels.
    pub fn set_coop_matrix(&self, on: bool) {
        self.use_coop
            .store(on && self.coop_matrix_ready(), Ordering::Relaxed);
        self.coop_f16_default.store(on, Ordering::Relaxed);
    }

    /// Known-answer check on tile-edge shapes. Some adapters (for example the virtualised GPU of
    /// a CI runner) run the kernels but return wrong numbers; such an adapter must not be used.
    fn self_test(&self) -> Result<(), ThmeError> {
        for (rows, in_dim, out_dim) in [(1usize, 70usize, 100usize), (17, 70, 100)] {
            let x: Vec<f32> = (0..rows * in_dim)
                .map(|i| ((i * 7 % 13) as f32 - 6.0) / 8.0)
                .collect();
            let w: Vec<f32> = (0..out_dim * in_dim)
                .map(|i| ((i * 5 % 11) as f32 - 5.0) / 8.0)
                .collect();
            let weights =
                QuantizedTensor::new_f32("gpu_self_test", vec![out_dim, in_dim], w.clone())?;
            let got = self.compute_linear(&x, &weights, in_dim, out_dim)?;
            for r in 0..rows {
                for o in 0..out_dim {
                    let want: f32 = (0..in_dim)
                        .map(|j| x[r * in_dim + j] * w[o * in_dim + j])
                        .sum();
                    let have = got.get(r * out_dim + o).copied().unwrap_or(f32::NAN);
                    if have.is_nan() || (have - want).abs() > 1e-3 * want.abs().max(1.0) {
                        return Err(ThmeError::GpuInitFailed(format!(
                            "adapter '{}' ({:?}) failed the known-answer self-test (rows={rows}: {have} instead of {want}); not using it",
                            self.adapter_info.name, self.adapter_info.backend
                        )));
                    }
                }
            }
        }
        Ok(())
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
    /// Longest the host waits for the device before a pass fails with an error (default 120 s).
    pub fn set_wait_timeout(&self, timeout: std::time::Duration) {
        self.wait_timeout_ms
            .store(timeout.as_millis() as u64, Ordering::Relaxed);
    }

    /// Blocks until submitted work completes, or fails with a message after the wait timeout
    /// so a lost device or stuck queue is an error, never a silent freeze.
    pub(super) fn wait_for_device(&self) -> Result<(), ThmeError> {
        let ms = self.wait_timeout_ms.load(Ordering::Relaxed);
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(std::time::Duration::from_millis(ms)),
            })
            .map(|_| ())
            .map_err(|e| {
                ThmeError::DataTransferFailed(format!(
                    "GPU wait failed after at most {ms} ms (timed out or device lost): {e}"
                ))
            })
    }

    /// Which dense f32 kernel runs a linear of this shape: `gemv` (few rows), `gemm128`
    /// (register-blocked, needs in_dim, out_dim and the output offset in multiples of 4) or
    /// `gemm64` (any shape).
    pub fn gemm_kernel_name(
        &self,
        rows: usize,
        in_dim: usize,
        out_dim: usize,
        out_offset: usize,
    ) -> &'static str {
        if rows <= GEMV_MAX_ROWS {
            "gemv"
        } else if self.gemm_v4.is_some()
            && self.use_gemm_v4.load(Ordering::Relaxed)
            && in_dim.is_multiple_of(4)
            && out_dim.is_multiple_of(4)
            && out_offset.is_multiple_of(4)
        {
            "gemm128"
        } else {
            "gemm64"
        }
    }

    /// Pipeline, workgroup grid and kernel name for a dense linear stored as `storage`.
    pub(super) fn dense_dispatch(
        &self,
        storage: WeightStorage,
        rows: usize,
        in_dim: usize,
        out_dim: usize,
        out_offset: usize,
    ) -> Result<(wgpu::ComputePipeline, [u32; 3], &'static str), ThmeError> {
        let (r, o) = (rows as u32, out_dim as u32);
        if rows > GEMV_MAX_ROWS
            && self.use_int8_dot.load(Ordering::Relaxed)
            && matches!(storage, WeightStorage::Q8_0 | WeightStorage::Q4_0)
            && storage.fits(in_dim, out_dim)
            && out_offset.is_multiple_of(4)
        {
            return Ok((
                self.dense_pipeline(DenseKernel::Q8Dot, storage)?,
                [o.div_ceil(GEMM_V4_TILE), r.div_ceil(GEMM_V4_TILE), 1],
                Q8DOT_KERNEL,
            ));
        }
        if rows > GEMV_MAX_ROWS
            && self.coop_active(storage)
            && storage.fits(in_dim, out_dim)
            && in_dim.is_multiple_of(4)
            && out_dim.is_multiple_of(4)
            && out_offset.is_multiple_of(4)
        {
            let direct = storage == WeightStorage::F16
                && in_dim.is_multiple_of(16)
                && out_dim.is_multiple_of(16);
            let kernel = if direct && self.use_coop_direct.load(Ordering::Relaxed) {
                DenseKernel::CoopF16
            } else {
                DenseKernel::Coop
            };
            return Ok((
                self.dense_pipeline(kernel, storage)?,
                [o.div_ceil(GEMM_V4_TILE), r.div_ceil(GEMM_V4_TILE), 1],
                if kernel == DenseKernel::CoopF16 {
                    COOP_DIRECT_KERNEL
                } else {
                    "gemm_coop"
                },
            ));
        }
        if storage != WeightStorage::F32 {
            let few = rows <= GEMV_MAX_ROWS;
            let kernel = if few {
                DenseKernel::Gemv
            } else {
                DenseKernel::Gemm
            };
            if !storage.fits(in_dim, out_dim) || !out_offset.is_multiple_of(4) {
                return Err(ThmeError::UnsupportedFormat {
                    dtype: format!("{storage:?} {out_dim}x{in_dim} at offset {out_offset}"),
                    backend: "WGPU dense kernels",
                });
            }
            let pipeline = self.dense_pipeline(kernel, storage)?;
            return Ok(if few {
                (
                    pipeline,
                    [o.min(MAX_GROUPS), o.div_ceil(MAX_GROUPS), r],
                    "gemv",
                )
            } else {
                (
                    pipeline,
                    [o.div_ceil(GEMM_V4_TILE), r.div_ceil(GEMM_V4_TILE), 1],
                    "gemm128",
                )
            });
        }
        match (
            self.gemm_kernel_name(rows, in_dim, out_dim, out_offset),
            &self.gemm_v4,
        ) {
            ("gemm128", Some(v4)) => Ok((
                v4.clone(),
                [o.div_ceil(GEMM_V4_TILE), r.div_ceil(GEMM_V4_TILE), 1],
                "gemm128",
            )),
            ("gemm64", _) => Ok((
                self.gemm_f32.clone(),
                [o.div_ceil(GEMM_TILE), r.div_ceil(GEMM_TILE), 1],
                "gemm64",
            )),
            _ => {
                let gemv = self.pipelines.get(&ShaderKind::F32).ok_or_else(|| {
                    ThmeError::ExecutionFailed("Compute pipeline for F32 not found".into())
                })?;
                Ok((
                    gemv.clone(),
                    [o.min(MAX_GROUPS), o.div_ceil(MAX_GROUPS), r],
                    "gemv",
                ))
            }
        }
    }

    /// The f16 / Q8_0 / Q4_0 kernel, compiled on first use.
    fn dense_pipeline(
        &self,
        kernel: DenseKernel,
        storage: WeightStorage,
    ) -> Result<wgpu::ComputePipeline, ThmeError> {
        let mut cache = self
            .dense
            .lock()
            .map_err(|_| ThmeError::ExecutionFailed("dense pipeline cache poisoned".into()))?;
        if let Some(hit) = cache.get(&(kernel, storage)) {
            return Ok(hit.clone());
        }
        let label = format!("{kernel:?}_{storage:?}");
        let module = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(&label),
                source: wgpu::ShaderSource::Wgsl(
                    super::dense::kernel_source(kernel, storage).into(),
                ),
            });
        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(&label),
                layout: Some(&self.dense_layout),
                module: &module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });
        cache.insert((kernel, storage), pipeline.clone());
        Ok(pipeline)
    }

    /// Turns the register-blocked kernel off (or back on) so a paired before/after run can use
    /// the 64 x 64 kernel in the same binary.
    pub fn set_register_blocked_gemm(&self, on: bool) {
        self.use_gemm_v4.store(on, Ordering::Relaxed);
    }

    /// Turns the tiled attention kernel off (or back on) for paired before/after runs.
    pub fn set_tiled_attention(&self, on: bool) {
        self.use_tiled_attention.store(on, Ordering::Relaxed);
    }

    /// Which attention kernel runs for this head size: `tiled` (online softmax, head size a
    /// multiple of 4 and at most 128), `coop` (cooperative matrices, head size 64) or `reference` (one workgroup per query and head).
    pub fn attention_kernel_name(&self, head_dim: usize) -> &'static str {
        if head_dim == 64
            && self.coop_active(self.weight_storage())
            && self.use_coop_attention.load(Ordering::Relaxed)
        {
            "coop"
        } else if self.use_tiled_attention.load(Ordering::Relaxed)
            && super::decoder::tiled_attention_supports(head_dim)
        {
            "tiled"
        } else {
            "reference"
        }
    }

    /// Bytes of weights kept on the device before the cache is flushed (default 4 GiB). Size it
    /// to the card: a model larger than the budget re-uploads its weights on every pass.
    pub fn set_weight_budget(&self, bytes: usize) {
        self.weight_budget.store(bytes, Ordering::Relaxed);
    }

    /// Chooses how weights are stored from now on (buffers already resident keep their format
    /// until the cache is cleared). A format the adapter cannot run falls back to f32.
    pub fn set_weight_storage(&self, storage: WeightStorage) {
        self.clear_weight_cache();
        self.weight_storage.store(storage as u8, Ordering::Relaxed);
    }

    /// The weight storage in effect.
    pub fn weight_storage(&self) -> WeightStorage {
        match self.weight_storage.load(Ordering::Relaxed) {
            1 => WeightStorage::F16,
            2 => WeightStorage::Q8_0,
            3 => WeightStorage::Q4_0,
            _ => WeightStorage::F32,
        }
    }

    /// True when the adapter reports the `shader-f16` feature.
    pub fn shader_f16_supported(&self) -> bool {
        self.shader_f16
    }

    /// Times every dispatch of the device-resident decoder on its own (needs timestamp queries;
    /// adds one read-back per submit and ends the single-pass fusion, so total time grows).
    pub fn set_op_profile(&self, on: bool) {
        self.kernel_timing.store(on, Ordering::Relaxed);
        self.op_profile.store(on, Ordering::Relaxed);
    }

    /// Drains the per-dispatch samples recorded since the last call.
    pub fn take_op_profile(&self) -> Vec<OpSample> {
        self.op_samples
            .lock()
            .map(|mut v| std::mem::take(&mut *v))
            .unwrap_or_default()
    }

    /// Reads the per-dispatch timestamp pairs of the submission that just finished.
    pub(super) fn read_op_profile(
        &self,
        p: &TimestampProbe,
        prof: &super::decoder::OpProfile,
        staging: &wgpu::Buffer,
    ) -> Result<(), ThmeError> {
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        self.wait_for_device()?;
        rx.recv()
            .map_err(|e| ThmeError::DataTransferFailed(format!("GPU channel recv failed: {e}")))?
            .map_err(|e| ThmeError::DataTransferFailed(format!("GPU buffer map failed: {e}")))?;
        let samples: Vec<OpSample> = {
            let data = slice
                .get_mapped_range()
                .map_err(|e| ThmeError::DataTransferFailed(format!("GPU map range failed: {e}")))?;
            let t: &[u64] = bytemuck::cast_slice(&data);
            prof.labels
                .iter()
                .zip(t.chunks_exact(2))
                .map(|(label, pair)| OpSample {
                    label: label.clone(),
                    ns: (pair[1].saturating_sub(pair[0]) as f64 * f64::from(p.period_ns)) as u64,
                })
                .collect()
        };
        staging.unmap();
        if let Ok(mut all) = self.op_samples.lock() {
            all.extend(samples);
        }
        Ok(())
    }

    pub fn set_kernel_timing(&self, on: bool) {
        self.kernel_timing.store(on, Ordering::Relaxed);
    }

    /// Drops every weight buffer cached for slice-backed linears (frees device memory).
    pub fn clear_weight_cache(&self) {
        if let Ok(mut cache) = self.weight_cache.lock() {
            *cache = WeightCache::default();
        }
    }

    /// Device buffer holding `w` as f32, uploaded once per (address, length) and then reused.
    /// Used for norm weights and anything read as plain floats.
    pub(super) fn resident_slice(&self, w: &[f32]) -> Result<Arc<wgpu::Buffer>, ThmeError> {
        Ok(self.resident_as(w, WeightStorage::F32)?.buffer)
    }

    /// Device buffer for a `[out_dim, in_dim]` linear in the storage chosen with
    /// [`GpuContext::set_weight_storage`]; falls back to f32 when the shape does not fit it.
    pub(super) fn resident_linear(
        &self,
        w: &[f32],
        in_dim: usize,
        out_dim: usize,
    ) -> Result<DeviceWeight, ThmeError> {
        let wanted = self.weight_storage();
        let storage = if wanted.fits(in_dim, out_dim)
            && (wanted == WeightStorage::F32 || self.gemm_v4.is_some())
        {
            wanted
        } else {
            WeightStorage::F32
        };
        self.resident_as(w, storage)
    }

    fn resident_as(&self, w: &[f32], storage: WeightStorage) -> Result<DeviceWeight, ThmeError> {
        let key = (w.as_ptr() as usize, w.len(), storage.tag());
        let print = fingerprint(w);
        let mut cache = self
            .weight_cache
            .lock()
            .map_err(|_| ThmeError::ExecutionFailed("GPU weight cache mutex poisoned".into()))?;
        cache.tick = cache.tick.wrapping_add(1);
        let tick = cache.tick;
        if let Some(hit) = cache.entries.get_mut(&key) {
            if hit.fingerprint == print {
                hit.last_used = tick;
                return Ok(DeviceWeight {
                    buffer: Arc::clone(&hit.buffer),
                    storage,
                });
            }
        }
        let bytes = storage.device_bytes(w.len());
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
        let budget = self.weight_budget.load(Ordering::Relaxed);
        while cache.bytes + bytes > budget {
            let Some((&oldest_key, _)) = cache.entries.iter().min_by_key(|(_, v)| v.last_used)
            else {
                break;
            };
            if let Some(removed) = cache.entries.remove(&oldest_key) {
                cache.bytes = cache.bytes.saturating_sub(removed.bytes);
            }
        }
        let t_up = Instant::now();
        let packed;
        let contents: &[u8] = if storage == WeightStorage::F32 {
            bytemuck::cast_slice(w)
        } else {
            packed = super::dense::pack_for_device(storage, w)?;
            &packed
        };
        let buffer = Arc::new(
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("gpu_weight_slice"),
                    contents,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                }),
        );
        self.counters.weight_uploads.fetch_add(1, Ordering::Relaxed);
        bump(&self.counters.weight_upload_ns, t_up);
        let entry = CachedWeight {
            fingerprint: print,
            buffer: Arc::clone(&buffer),
            bytes,
            last_used: tick,
        };
        if cache.entries.insert(key, entry).is_none() {
            cache.bytes += bytes;
        }
        Ok(DeviceWeight { buffer, storage })
    }

    /// Reads the two timestamps of the submission that just finished and adds the difference.
    pub(super) fn read_kernel_time(&self, p: &TimestampProbe) -> Result<(), ThmeError> {
        let slice = p.staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        self.wait_for_device()?;
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
        // Each op's block starts on a multiple of 4 floats so vec4 kernels can write it.
        let offsets: Vec<usize> = ops
            .iter()
            .scan(0usize, |next, op| {
                let at = *next;
                *next = (at + rows * op.out_dim).next_multiple_of(4);
                Some(at)
            })
            .collect();
        let total_out = ops
            .last()
            .zip(offsets.last())
            .map_or(0, |(op, at)| at + rows * op.out_dim);
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
        for (i, op) in ops.iter().enumerate() {
            let params = GemvParams {
                in_dim: in_dim as u32,
                out_dim: op.out_dim as u32,
                num_blocks_per_row: (in_dim / 256) as u32,
                rows: rows as u32,
                out_offset: offsets[i] as u32,
                _pad: [0; 3],
            };
            let at = i * PARAM_STRIDE;
            params_bytes[at..at + std::mem::size_of::<GemvParams>()]
                .copy_from_slice(bytemuck::bytes_of(&params));
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
            let mut converted: Option<(ActivationCopy, wgpu::Buffer)> = None;
            for (i, op) in ops.iter().enumerate() {
                let params_binding = wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &scratch.params,
                    offset: (i * PARAM_STRIDE) as u64,
                    size: wgpu::BufferSize::new(std::mem::size_of::<GemvParams>() as u64),
                });
                let mut copy = None;
                let (pipeline, groups) = if op.kind == ShaderKind::F32 {
                    let (p, g, name) =
                        self.dense_dispatch(op.storage, rows, in_dim, op.out_dim, offsets[i])?;
                    copy = ActivationCopy::for_kernel(name);
                    (p, g)
                } else {
                    let p = self
                        .pipelines
                        .get(&op.kind)
                        .ok_or_else(|| {
                            ThmeError::ExecutionFailed(format!(
                                "Compute pipeline for {:?} not found",
                                op.kind
                            ))
                        })?
                        .clone();
                    let out = op.out_dim as u32;
                    // Output rows spread over x then y; token rows ride on z.
                    (
                        p,
                        [out.min(MAX_GROUPS), out.div_ceil(MAX_GROUPS), rows as u32],
                    )
                };
                if let Some(kind) = copy.filter(|k| converted.as_ref().is_none_or(|c| c.0 != *k)) {
                    converted = Some((
                        kind,
                        self.convert_activations(
                            kind,
                            &mut pass,
                            params_binding.clone(),
                            scratch.input.as_entire_binding(),
                            op.weights.as_entire_binding(),
                            rows,
                            in_dim,
                        )?,
                    ));
                }
                let x_resource = match (&converted, copy) {
                    (Some((k, b)), Some(want)) if *k == want => b.as_entire_binding(),
                    _ => scratch.input.as_entire_binding(),
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
                            resource: x_resource,
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
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(groups[0], groups[1], groups[2]);
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
        self.wait_for_device()?;
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
        for (op, &at) in ops.iter().zip(&offsets) {
            results.push(all[at..at + rows * op.out_dim].to_vec());
        }
        drop(data);
        scratch.staging.unmap();
        bump(&self.counters.readback_ns, t_read);
        Ok(results)
    }
}

/// Kernel family and weight storage a tensor of this dtype runs with.
fn shader_kind(dtype: QuantDtype) -> Result<(ShaderKind, WeightStorage), ThmeError> {
    match dtype {
        QuantDtype::Q4_K => Ok((ShaderKind::Q4_K, WeightStorage::F32)),
        QuantDtype::Q5_K => Ok((ShaderKind::Q5_K, WeightStorage::F32)),
        QuantDtype::IQ4_XS => Ok((ShaderKind::IQ4_XS, WeightStorage::F32)),
        QuantDtype::F32 => Ok((ShaderKind::F32, WeightStorage::F32)),
        QuantDtype::F16 => Ok((ShaderKind::F32, WeightStorage::F16)),
        QuantDtype::Q8_0 => Ok((ShaderKind::F32, WeightStorage::Q8_0)),
        QuantDtype::Q4_0 => Ok((ShaderKind::F32, WeightStorage::Q4_0)),
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

    /// Drops any cached device weight buffers to free GPU memory between models.
    fn clear_cache(&self) {}

    /// Enables cooperative-matrix Tensor Core kernels when supported and not disabled via `THME_GPU_COOP=0`.
    fn enable_coop_if_ready(&self) {}
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
        let (kind, storage) = shader_kind(weights.dtype())?;
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
            storage,
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
            .map(|(w, out_dim)| self.resident_linear(w, in_dim, *out_dim))
            .collect::<Result<Vec<_>, _>>()?;
        let linear_ops: Vec<LinearOp<'_>> = buffers
            .iter()
            .zip(ops)
            .map(|(dw, (_, out_dim))| LinearOp {
                kind: ShaderKind::F32,
                storage: dw.storage,
                weights: &dw.buffer,
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

    fn clear_cache(&self) {
        self.clear_weight_cache();
    }

    fn enable_coop_if_ready(&self) {
        if self.coop_matrix_ready() && std::env::var("THME_GPU_COOP").ok().as_deref() != Some("0") {
            self.set_coop_matrix(true);
        }
    }

    fn is_format_supported(&self, dtype: QuantDtype) -> bool {
        matches!(
            dtype,
            QuantDtype::Q4_K
                | QuantDtype::Q5_K
                | QuantDtype::IQ4_XS
                | QuantDtype::F32
                | QuantDtype::F16
                | QuantDtype::Q8_0
                | QuantDtype::Q4_0
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
