// SPDX-License-Identifier: AGPL-3.0-only
//! NVIDIA CUDA + cuBLAS Tensor Core acceleration backend for THME.

use std::collections::HashMap;
use std::ffi::c_void;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex};

use cudarc::cublas::sys as cublas_sys;
use cudarc::cublas::CudaBlas;
use cudarc::driver::{
    CudaContext as DriverContext, CudaFunction, CudaSlice, CudaStream, CudaView, DevicePtr,
    DevicePtrMut, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::Ptx;

use super::backend::{check_finite, GpuBackend};
use super::decoder::DecoderPlan;
use super::error::ThmeError;
use super::tensor::{QuantDtype, QuantizedTensor};

/// Default weight bytes kept resident on CUDA before LRU eviction (10 GiB).
const WEIGHT_CACHE_BUDGET: usize = 10 << 30;
/// Maximum logits elements buffered per NLL chunk (128 MiB in FP16).
const NLL_CHUNK_MAX_ELEMS: usize = 64 * 1024 * 1024;
const BLOCK_256: u32 = 256;
/// Persistent cuBLAS workspace size (32 MiB) to avoid per-call allocations and unlock Ada Tensor Core algos.
const CUBLAS_WORKSPACE_BYTES: usize = 32 << 20;

#[derive(Clone)]
enum CachedBuffer {
    F16(Arc<CudaSlice<u16>>),
    F32(Arc<CudaSlice<f32>>),
}

struct CachedEntry {
    fingerprint: u64,
    buffer: CachedBuffer,
    bytes: usize,
    last_used: u64,
}

#[derive(Default)]
struct WeightCache {
    entries: HashMap<(usize, usize, u8), CachedEntry>,
    bytes: usize,
    tick: u64,
}

impl WeightCache {
    fn evict_if_needed(&mut self, incoming: usize) {
        while self.bytes + incoming > WEIGHT_CACHE_BUDGET && !self.entries.is_empty() {
            let Some(oldest_key) = self
                .entries
                .iter()
                .min_by_key(|(_, entry)| entry.last_used)
                .map(|(k, _)| *k)
            else {
                break;
            };
            if let Some(removed) = self.entries.remove(&oldest_key) {
                self.bytes = self.bytes.saturating_sub(removed.bytes);
            }
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct CudaWorkspaceKey {
    t: usize,
    d: usize,
    kv_dim: usize,
    inter: usize,
    vocab: usize,
    nll_mode: bool,
    head_dim: usize,
}

struct CudaWorkspace {
    x_a: CudaSlice<f32>,
    x_b: CudaSlice<f32>,
    x16: CudaSlice<u16>,
    q16: CudaSlice<u16>,
    k16: CudaSlice<u16>,
    v16: CudaSlice<u16>,
    attn_scores: CudaSlice<f32>,
    attn_probs16: CudaSlice<u16>,
    q: CudaSlice<f32>,
    k: CudaSlice<f32>,
    v: CudaSlice<f32>,
    mixed: CudaSlice<f32>,
    tmp: CudaSlice<f32>,
    tmp16: CudaSlice<u16>,
    gate: CudaSlice<f32>,
    gate16: CudaSlice<u16>,
    up: CudaSlice<f32>,
    up16: CudaSlice<u16>,
    last: CudaSlice<f32>,
    logits: CudaSlice<f32>,
    logits16: CudaSlice<u16>,
    scores: CudaSlice<f32>,
    targets: CudaSlice<u32>,
    cos: CudaSlice<f32>,
    sin: CudaSlice<f32>,
    cos_hash: u64,
    sin_hash: u64,
}

/// CUDA + cuBLAS compute backend implementing [`GpuBackend`].
pub struct CudaContext {
    _ctx: Arc<DriverContext>,
    stream: Arc<CudaStream>,
    blas: Option<Mutex<CudaBlas>>,
    _blas_workspace: Option<CudaSlice<u8>>,
    fn_f32_to_f16: CudaFunction,
    fn_rms_norm: CudaFunction,
    fn_rms_norm_f16: CudaFunction,
    fn_rope: CudaFunction,
    fn_rope_qk_f16: CudaFunction,
    fn_softmax: CudaFunction,
    fn_swiglu_f16: CudaFunction,
    fn_swiglu_f16_in: CudaFunction,
    fn_add: CudaFunction,
    fn_add_rms_norm_f16: CudaFunction,
    fn_nll: CudaFunction,
    fn_nll_f16: CudaFunction,
    fn_gemm_f16_tiled: CudaFunction,
    fn_attn_f32: CudaFunction,
    fn_flash_attn_f16: CudaFunction,
    fn_kv_append_f16: CudaFunction,
    fn_kv_attention_f16: CudaFunction,
    device_name: String,
    weight_cache: Mutex<WeightCache>,
    workspace: Mutex<Option<(CudaWorkspaceKey, CudaWorkspace)>>,
    scratch_x16: Mutex<Option<CudaSlice<u16>>>,
}

fn cuda_err(ctx: &str, err: impl std::fmt::Debug) -> ThmeError {
    ThmeError::ExecutionFailed(format!("{ctx}: {err:?}"))
}

fn fingerprint(w: &[f32]) -> u64 {
    let step = (w.len() / 64).max(1);
    let mut h: u64 = 0xcbf2_9ce4_8422_2325 ^ w.len() as u64;
    for v in w.iter().step_by(step) {
        h = (h ^ u64::from(v.to_bits())).wrapping_mul(0x0100_0000_01b3);
    }
    h
}

fn slice_hash(data: &[f32]) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    let bytes: &[u8] = bytemuck::cast_slice(data);
    bytes.hash(&mut h);
    h.finish()
}

/// Converts IEEE 754 `f32` to `f16` bit pattern (round-to-nearest-even).
fn f32_to_f16_bits(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let frac = bits & 0x007f_ffff;

    if exp == 255 {
        if frac == 0 {
            return sign | 0x7c00;
        }
        return sign | 0x7e00 | ((frac >> 13) as u16);
    }

    let unbiased = exp - 127 + 15;
    if unbiased >= 31 {
        return sign | 0x7c00;
    }
    if unbiased <= 0 {
        if unbiased < -10 {
            return sign;
        }
        let mant = (frac | 0x0080_0000) >> (1 - unbiased);
        let round_bit = (mant >> 12) & 1;
        let sticky = (mant & 0x0fff) != 0;
        let half_mant = (mant >> 13) as u16;
        let lsb = half_mant & 1;
        return sign | (half_mant + (round_bit & (sticky as u32 | lsb as u32)) as u16);
    }

    let round_bit = (frac >> 12) & 1;
    let sticky = (frac & 0x0fff) != 0;
    let half_mant = (frac >> 13) as u16;
    let lsb = half_mant & 1;
    let rounded = ((unbiased as u16) << 10) | half_mant;
    sign | (rounded + (round_bit & (sticky as u32 | lsb as u32)) as u16)
}

fn convert_f32_slice_to_f16(w: &[f32]) -> Vec<u16> {
    const PARALLEL_THRESHOLD: usize = 65_536;
    if w.len() < PARALLEL_THRESHOLD {
        return w.iter().copied().map(f32_to_f16_bits).collect();
    }
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    if threads <= 1 {
        return w.iter().copied().map(f32_to_f16_bits).collect();
    }
    let chunk_size = w.len().div_ceil(threads).max(1);
    let mut out = vec![0u16; w.len()];
    std::thread::scope(|scope| {
        for (src, dst) in w.chunks(chunk_size).zip(out.chunks_mut(chunk_size)) {
            scope.spawn(move || {
                for (s, d) in src.iter().zip(dst.iter_mut()) {
                    *d = f32_to_f16_bits(*s);
                }
            });
        }
    });
    out
}

fn grid_1d(n: usize) -> LaunchConfig {
    let blocks = n.div_ceil(BLOCK_256 as usize).max(1) as u32;
    LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (BLOCK_256, 1, 1),
        shared_mem_bytes: 0,
    }
}

impl CudaContext {
    /// Initializes CUDA on device 0, optionally loads cuBLAS, and loads the embedded decoder PTX.
    pub fn init() -> Result<Self, ThmeError> {
        if !unsafe { cudarc::driver::sys::is_culib_present() } {
            return Err(ThmeError::GpuInitFailed(
                "CUDA driver library not found".into(),
            ));
        }
        let ctx = DriverContext::new(0)
            .map_err(|e| ThmeError::GpuInitFailed(format!("CUDA device 0 init failed: {e:?}")))?;
        let stream = ctx.default_stream();
        let (blas, _blas_workspace) = if unsafe { cublas_sys::is_culib_present() } {
            match CudaBlas::new(stream.clone()) {
                Ok(blas) => {
                    let ws = match stream.alloc_zeros::<u8>(CUBLAS_WORKSPACE_BYTES) {
                        Ok(mut buf) => {
                            let status = {
                                let (ptr, _guard) = buf.device_ptr_mut(&stream);
                                unsafe {
                                    cublas_sys::cublasSetWorkspace_v2(
                                        *blas.handle(),
                                        ptr as *mut c_void,
                                        CUBLAS_WORKSPACE_BYTES,
                                    )
                                }
                            };
                            if status == cublas_sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
                                Some(buf)
                            } else {
                                None
                            }
                        }
                        Err(_) => None,
                    };
                    (Some(Mutex::new(blas)), ws)
                }
                Err(_) => (None, None),
            }
        } else {
            (None, None)
        };
        let ptx = Ptx::from_src(include_str!("kernels/decoder.ptx"));
        let module = ctx
            .load_module(ptx)
            .map_err(|e| ThmeError::ShaderCompilationFailed {
                shader_name: "decoder.ptx".into(),
                error: format!("{e:?}"),
            })?;

        let load = |name: &str| {
            module
                .load_function(name)
                .map_err(|e| ThmeError::ShaderCompilationFailed {
                    shader_name: name.into(),
                    error: format!("{e:?}"),
                })
        };

        let device_name = if blas.is_some() {
            "NVIDIA CUDA (cuBLAS Tensor Cores, Embedded PTX)".to_string()
        } else {
            "NVIDIA CUDA (Pure Embedded PTX, Zero External DLLs)".to_string()
        };

        Ok(Self {
            _ctx: ctx,
            stream,
            blas,
            _blas_workspace,
            fn_f32_to_f16: load("f32_to_f16")?,
            fn_rms_norm: load("rms_norm")?,
            fn_rms_norm_f16: load("rms_norm_f16")?,
            fn_rope: load("rope")?,
            fn_rope_qk_f16: load("rope_qk_f16")?,
            fn_softmax: load("causal_softmax_f16")?,
            fn_swiglu_f16: load("swiglu_f16")?,
            fn_swiglu_f16_in: load("swiglu_f16_in")?,
            fn_add: load("add")?,
            fn_add_rms_norm_f16: load("add_rms_norm_f16")?,
            fn_nll: load("nll")?,
            fn_nll_f16: load("nll_f16")?,
            fn_gemm_f16_tiled: load("gemm_f16_tiled")?,
            fn_attn_f32: load("causal_attention_f32")?,
            fn_flash_attn_f16: load("flash_attention_f16")?,
            fn_kv_append_f16: load("kv_append_f16")?,
            fn_kv_attention_f16: load("kv_attention_f16")?,
            device_name,
            weight_cache: Mutex::new(WeightCache::default()),
            workspace: Mutex::new(None),
            scratch_x16: Mutex::new(None),
        })
    }

    fn resident_f16(&self, w: &[f32]) -> Result<Arc<CudaSlice<u16>>, ThmeError> {
        let key = (w.as_ptr() as usize, w.len(), 16u8);
        let fp = fingerprint(w);
        {
            let mut cache = self
                .weight_cache
                .lock()
                .map_err(|_| ThmeError::ExecutionFailed("CUDA weight cache poisoned".into()))?;
            cache.tick += 1;
            let tick = cache.tick;
            if let Some(entry) = cache.entries.get_mut(&key) {
                if entry.fingerprint == fp {
                    if let CachedBuffer::F16(buf) = &entry.buffer {
                        entry.last_used = tick;
                        return Ok(Arc::clone(buf));
                    }
                }
            }
        }

        let f16_data = convert_f32_slice_to_f16(w);
        let buf = Arc::new(self.stream.clone_htod(&f16_data).map_err(|e| {
            ThmeError::DataTransferFailed(format!("CUDA f16 weight upload: {e:?}"))
        })?);
        let bytes = w.len() * std::mem::size_of::<u16>();

        let mut cache = self
            .weight_cache
            .lock()
            .map_err(|_| ThmeError::ExecutionFailed("CUDA weight cache poisoned".into()))?;
        cache.evict_if_needed(bytes);
        cache.tick += 1;
        let tick = cache.tick;
        if let Some(prev) = cache.entries.insert(
            key,
            CachedEntry {
                fingerprint: fp,
                buffer: CachedBuffer::F16(Arc::clone(&buf)),
                bytes,
                last_used: tick,
            },
        ) {
            cache.bytes = cache.bytes.saturating_sub(prev.bytes);
        }
        cache.bytes += bytes;
        Ok(buf)
    }

    fn resident_f32(&self, w: &[f32]) -> Result<Arc<CudaSlice<f32>>, ThmeError> {
        let key = (w.as_ptr() as usize, w.len(), 32u8);
        let fp = fingerprint(w);
        {
            let mut cache = self
                .weight_cache
                .lock()
                .map_err(|_| ThmeError::ExecutionFailed("CUDA weight cache poisoned".into()))?;
            cache.tick += 1;
            let tick = cache.tick;
            if let Some(entry) = cache.entries.get_mut(&key) {
                if entry.fingerprint == fp {
                    if let CachedBuffer::F32(buf) = &entry.buffer {
                        entry.last_used = tick;
                        return Ok(Arc::clone(buf));
                    }
                }
            }
        }

        let buf = Arc::new(self.stream.clone_htod(w).map_err(|e| {
            ThmeError::DataTransferFailed(format!("CUDA f32 weight upload: {e:?}"))
        })?);
        let bytes = std::mem::size_of_val(w);

        let mut cache = self
            .weight_cache
            .lock()
            .map_err(|_| ThmeError::ExecutionFailed("CUDA weight cache poisoned".into()))?;
        cache.evict_if_needed(bytes);
        cache.tick += 1;
        let tick = cache.tick;
        if let Some(prev) = cache.entries.insert(
            key,
            CachedEntry {
                fingerprint: fp,
                buffer: CachedBuffer::F32(Arc::clone(&buf)),
                bytes,
                last_used: tick,
            },
        ) {
            cache.bytes = cache.bytes.saturating_sub(prev.bytes);
        }
        cache.bytes += bytes;
        Ok(buf)
    }

    fn convert_x_to_f16(
        &self,
        x_view: CudaView<'_, f32>,
        x16: &mut CudaSlice<u16>,
        n: usize,
    ) -> Result<(), ThmeError> {
        let n_u32 = n as u32;
        unsafe {
            self.stream
                .launch_builder(&self.fn_f32_to_f16)
                .arg(&x_view)
                .arg(x16)
                .arg(&n_u32)
                .launch(grid_1d(n))
        }
        .map_err(|e| cuda_err("f32_to_f16 launch", e))?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn gemm_f16_preconverted(
        &self,
        x16: &CudaSlice<u16>,
        w16: &CudaSlice<u16>,
        out: &mut CudaSlice<f32>,
        rows: usize,
        in_dim: usize,
        out_dim: usize,
    ) -> Result<(), ThmeError> {
        if let Some(blas_mutex) = &self.blas {
            let blas = blas_mutex
                .lock()
                .map_err(|_| ThmeError::ExecutionFailed("cuBLAS lock poisoned".into()))?;
            let (w_ptr, _w_guard) = w16.device_ptr(&self.stream);
            let (x_ptr, _x_guard) = x16.device_ptr(&self.stream);
            let (y_ptr, _y_guard) = out.device_ptr_mut(&self.stream);

            let alpha: f32 = 1.0;
            let beta: f32 = 0.0;
            let status = unsafe {
                cublas_sys::cublasGemmEx(
                    *blas.handle(),
                    cublas_sys::cublasOperation_t::CUBLAS_OP_T,
                    cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                    out_dim as i32,
                    rows as i32,
                    in_dim as i32,
                    (&alpha as *const f32).cast::<c_void>(),
                    w_ptr as *const c_void,
                    cublas_sys::cudaDataType_t::CUDA_R_16F,
                    in_dim as i32,
                    x_ptr as *const c_void,
                    cublas_sys::cudaDataType_t::CUDA_R_16F,
                    in_dim as i32,
                    (&beta as *const f32).cast::<c_void>(),
                    y_ptr as *mut c_void,
                    cublas_sys::cudaDataType_t::CUDA_R_32F,
                    out_dim as i32,
                    cublas_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                    cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
                )
            };
            if status != cublas_sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
                return Err(ThmeError::ExecutionFailed(format!(
                    "cublasGemmEx failed with status {status:?}"
                )));
            }
            return Ok(());
        }

        let rows_u32 = rows as u32;
        let in_u32 = in_dim as u32;
        let out_u32 = out_dim as u32;
        let zero = 0u32;
        let dummy_f16 = w16.slice(0..1);
        let cfg = LaunchConfig {
            grid_dim: (out_u32.div_ceil(16), rows_u32.div_ceil(16), 1),
            block_dim: (16, 16, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream
                .launch_builder(&self.fn_gemm_f16_tiled)
                .arg(x16)
                .arg(w16)
                .arg(out)
                .arg(&dummy_f16)
                .arg(&rows_u32)
                .arg(&in_u32)
                .arg(&out_u32)
                .arg(&zero)
                .arg(&zero)
                .launch(cfg)
        }
        .map_err(|e| cuda_err("gemm_f16_tiled launch", e))?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn linear_into(
        &self,
        x: &CudaSlice<f32>,
        x_row_offset: usize,
        x16: &mut CudaSlice<u16>,
        w16: &CudaSlice<u16>,
        out: &mut CudaSlice<f32>,
        rows: usize,
        in_dim: usize,
        out_dim: usize,
    ) -> Result<(), ThmeError> {
        let start = x_row_offset * in_dim;
        let n = rows * in_dim;
        let view = x.slice(start..start + n);
        self.convert_x_to_f16(view, x16, n)?;
        self.gemm_f16_preconverted(x16, w16, out, rows, in_dim, out_dim)
    }

    fn alloc_workspace(
        &self,
        key: CudaWorkspaceKey,
        chunk_rows: usize,
    ) -> Result<CudaWorkspace, ThmeError> {
        let alloc_f32 = |len: usize, name: &str| {
            self.stream.alloc_zeros::<f32>(len.max(1)).map_err(|e| {
                ThmeError::BufferAllocationFailed {
                    name: name.into(),
                    bytes: len.max(1) * 4,
                    reason: format!("{e:?}"),
                }
            })
        };
        let (t, d, kv_dim, inter, vocab, hd) =
            (key.t, key.d, key.kv_dim, key.inter, key.vocab, key.head_dim);
        let half = hd / 2;
        let max_x16 = (t * d).max(t * inter).max(1);
        let x16 = self.stream.alloc_zeros::<u16>(max_x16).map_err(|e| {
            ThmeError::BufferAllocationFailed {
                name: "x16".into(),
                bytes: max_x16 * 2,
                reason: format!("{e:?}"),
            }
        })?;
        let targets = self.stream.alloc_zeros::<u32>(t.max(1)).map_err(|e| {
            ThmeError::BufferAllocationFailed {
                name: "targets".into(),
                bytes: t.max(1) * 4,
                reason: format!("{e:?}"),
            }
        })?;

        let logits_len = if key.nll_mode {
            chunk_rows * vocab
        } else {
            vocab
        };

        let alloc_u16 = |len: usize, name: &str| {
            self.stream.alloc_zeros::<u16>(len.max(1)).map_err(|e| {
                ThmeError::BufferAllocationFailed {
                    name: name.into(),
                    bytes: len.max(1) * 2,
                    reason: format!("{e:?}"),
                }
            })
        };
        let heads = d / hd.max(1);
        let q16 = alloc_u16(t * d, "q16")?;
        let k16 = alloc_u16(t * kv_dim, "k16")?;
        let v16 = alloc_u16(t * kv_dim, "v16")?;
        let attn_scores = alloc_f32(heads * t * t, "attn_scores")?;
        let attn_probs16 = alloc_u16(heads * t * t, "attn_probs16")?;

        let logits16_len = if key.nll_mode { chunk_rows * vocab } else { 1 };
        let logits16 = alloc_u16(logits16_len, "logits16")?;

        Ok(CudaWorkspace {
            x_a: alloc_f32(t * d, "x_a")?,
            x_b: alloc_f32(t * d, "x_b")?,
            x16,
            q16,
            k16,
            v16,
            attn_scores,
            attn_probs16,
            q: alloc_f32(t * d, "q")?,
            k: alloc_f32(t * kv_dim, "k")?,
            v: alloc_f32(t * kv_dim, "v")?,
            mixed: alloc_f32(t * d, "mixed")?,
            tmp: alloc_f32(t * d, "tmp")?,
            tmp16: alloc_u16(t * d, "tmp16")?,
            gate: alloc_f32(t * inter, "gate")?,
            gate16: alloc_u16(t * inter, "gate16")?,
            up: alloc_f32(t * inter, "up")?,
            up16: alloc_u16(t * inter, "up16")?,
            last: alloc_f32(d, "last")?,
            logits: alloc_f32(logits_len, "logits")?,
            logits16,
            scores: alloc_f32(t, "scores")?,
            targets,
            cos: alloc_f32(t * half, "cos")?,
            sin: alloc_f32(t * half, "sin")?,
            cos_hash: 0,
            sin_hash: 0,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_rms_norm_f16(
        &self,
        x: &CudaSlice<f32>,
        w: &CudaSlice<f32>,
        out16: &mut CudaSlice<u16>,
        rows: usize,
        d: usize,
        eps: f32,
        x_offset: usize,
    ) -> Result<(), ThmeError> {
        let d_u32 = d as u32;
        let off_u32 = x_offset as u32;
        let cfg = LaunchConfig {
            grid_dim: (rows as u32, 1, 1),
            block_dim: (BLOCK_256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream
                .launch_builder(&self.fn_rms_norm_f16)
                .arg(x)
                .arg(w)
                .arg(out16)
                .arg(&d_u32)
                .arg(&eps)
                .arg(&off_u32)
                .launch(cfg)
        }
        .map_err(|e| cuda_err("rms_norm_f16 launch", e))?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_rms_norm(
        &self,
        x: &CudaSlice<f32>,
        w: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        rows: usize,
        d: usize,
        eps: f32,
        x_offset: usize,
    ) -> Result<(), ThmeError> {
        let d_u32 = d as u32;
        let off_u32 = x_offset as u32;
        let cfg = LaunchConfig {
            grid_dim: (rows as u32, 1, 1),
            block_dim: (BLOCK_256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream
                .launch_builder(&self.fn_rms_norm)
                .arg(x)
                .arg(w)
                .arg(out)
                .arg(&d_u32)
                .arg(&eps)
                .arg(&off_u32)
                .launch(cfg)
        }
        .map_err(|e| cuda_err("rms_norm launch", e))?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_rope(
        &self,
        cos: &CudaSlice<f32>,
        sin: &CudaSlice<f32>,
        qk: &mut CudaSlice<f32>,
        rows: usize,
        row_len: usize,
        head_dim: usize,
        half: usize,
    ) -> Result<(), ThmeError> {
        let rows_u32 = rows as u32;
        let row_len_u32 = row_len as u32;
        let hd_u32 = head_dim as u32;
        let half_u32 = half as u32;
        let total = rows * (row_len / head_dim) * half;
        unsafe {
            self.stream
                .launch_builder(&self.fn_rope)
                .arg(cos)
                .arg(sin)
                .arg(qk)
                .arg(&rows_u32)
                .arg(&row_len_u32)
                .arg(&hd_u32)
                .arg(&half_u32)
                .launch(grid_1d(total))
        }
        .map_err(|e| cuda_err("rope launch", e))?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_rope_qk_f16(
        &self,
        cos: &CudaSlice<f32>,
        sin: &CudaSlice<f32>,
        q16: &mut CudaSlice<u16>,
        k16: &mut CudaSlice<u16>,
        rows: usize,
        q_dim: usize,
        kv_dim: usize,
        head_dim: usize,
        half: usize,
    ) -> Result<(), ThmeError> {
        let rows_u32 = rows as u32;
        let q_dim_u32 = q_dim as u32;
        let kv_dim_u32 = kv_dim as u32;
        let hd_u32 = head_dim as u32;
        let half_u32 = half as u32;
        let total = rows * ((q_dim + kv_dim) / head_dim) * half;
        unsafe {
            self.stream
                .launch_builder(&self.fn_rope_qk_f16)
                .arg(cos)
                .arg(sin)
                .arg(q16)
                .arg(k16)
                .arg(&rows_u32)
                .arg(&q_dim_u32)
                .arg(&kv_dim_u32)
                .arg(&hd_u32)
                .arg(&half_u32)
                .launch(grid_1d(total))
        }
        .map_err(|e| cuda_err("rope_qk_f16 launch", e))?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_add_rms_norm_f16(
        &self,
        x_in: &CudaSlice<f32>,
        res16: &CudaSlice<u16>,
        w: &CudaSlice<f32>,
        x_out: &mut CudaSlice<f32>,
        out16: &mut CudaSlice<u16>,
        rows: usize,
        d: usize,
        eps: f32,
    ) -> Result<(), ThmeError> {
        let d_u32 = d as u32;
        let cfg = LaunchConfig {
            grid_dim: (rows as u32, 1, 1),
            block_dim: (BLOCK_256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream
                .launch_builder(&self.fn_add_rms_norm_f16)
                .arg(x_in)
                .arg(res16)
                .arg(w)
                .arg(x_out)
                .arg(out16)
                .arg(&d_u32)
                .arg(&eps)
                .launch(cfg)
        }
        .map_err(|e| cuda_err("add_rms_norm_f16 launch", e))?;
        Ok(())
    }

    fn launch_swiglu_f16_in(
        &self,
        gate16: &CudaSlice<u16>,
        up16: &CudaSlice<u16>,
        out16: &mut CudaSlice<u16>,
        n: usize,
    ) -> Result<(), ThmeError> {
        let n_u32 = n as u32;
        unsafe {
            self.stream
                .launch_builder(&self.fn_swiglu_f16_in)
                .arg(gate16)
                .arg(up16)
                .arg(out16)
                .arg(&n_u32)
                .launch(grid_1d(n))
        }
        .map_err(|e| cuda_err("swiglu_f16_in launch", e))?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_flash_attention_f16(
        &self,
        ws: &mut CudaWorkspace,
        t_len: usize,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
        scale: f32,
    ) -> Result<(), ThmeError> {
        let t_u32 = t_len as u32;
        let heads_u32 = heads as u32;
        let kv_heads_u32 = kv_heads as u32;
        let hd_u32 = head_dim as u32;
        let smem_bytes = (32 * head_dim * 2 * 2) as u32;
        let cfg = LaunchConfig {
            grid_dim: (heads_u32, t_u32.div_ceil(8), 1),
            block_dim: (32, 8, 1),
            shared_mem_bytes: smem_bytes,
        };
        unsafe {
            self.stream
                .launch_builder(&self.fn_flash_attn_f16)
                .arg(&ws.q16)
                .arg(&ws.k16)
                .arg(&ws.v16)
                .arg(&mut ws.x16)
                .arg(&t_u32)
                .arg(&heads_u32)
                .arg(&kv_heads_u32)
                .arg(&hd_u32)
                .arg(&scale)
                .launch(cfg)
        }
        .map_err(|e| cuda_err("flash_attention_f16 launch", e))?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_kv_append_f16(
        &self,
        k_src: &CudaSlice<u16>,
        v_src: &CudaSlice<u16>,
        k_dst: &mut CudaSlice<u16>,
        v_dst: &mut CudaSlice<u16>,
        offset: usize,
        t_len: usize,
        kv_dim: usize,
    ) -> Result<(), ThmeError> {
        let offset_u32 = offset as u32;
        let t_u32 = t_len as u32;
        let kv_dim_u32 = kv_dim as u32;
        unsafe {
            self.stream
                .launch_builder(&self.fn_kv_append_f16)
                .arg(k_src)
                .arg(v_src)
                .arg(k_dst)
                .arg(v_dst)
                .arg(&offset_u32)
                .arg(&t_u32)
                .arg(&kv_dim_u32)
                .launch(grid_1d(t_len * kv_dim))
        }
        .map_err(|e| cuda_err("kv_append_f16 launch", e))?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_kv_attention_f16(
        &self,
        q16: &CudaSlice<u16>,
        k_cache16: &CudaSlice<u16>,
        v_cache16: &CudaSlice<u16>,
        out16: &mut CudaSlice<u16>,
        offset: usize,
        t_len: usize,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
        scale: f32,
    ) -> Result<(), ThmeError> {
        let offset_u32 = offset as u32;
        let t_u32 = t_len as u32;
        let heads_u32 = heads as u32;
        let kv_heads_u32 = kv_heads as u32;
        let hd_u32 = head_dim as u32;
        let smem_bytes = (32 * head_dim * 2 * 2) as u32;
        let cfg = LaunchConfig {
            grid_dim: (heads_u32, t_u32.div_ceil(8), 1),
            block_dim: (32, 8, 1),
            shared_mem_bytes: smem_bytes,
        };
        unsafe {
            self.stream
                .launch_builder(&self.fn_kv_attention_f16)
                .arg(q16)
                .arg(k_cache16)
                .arg(v_cache16)
                .arg(out16)
                .arg(&offset_u32)
                .arg(&t_u32)
                .arg(&heads_u32)
                .arg(&kv_heads_u32)
                .arg(&hd_u32)
                .arg(&scale)
                .launch(cfg)
        }
        .map_err(|e| cuda_err("kv_attention_f16 launch", e))?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_attention(
        &self,
        ws: &mut CudaWorkspace,
        t_len: usize,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
        scale: f32,
    ) -> Result<(), ThmeError> {
        let Some(_blas_mutex) = &self.blas else {
            let t_u32 = t_len as u32;
            let heads_u32 = heads as u32;
            let kv_heads_u32 = kv_heads as u32;
            let hd_u32 = head_dim as u32;
            let cfg = LaunchConfig {
                grid_dim: (heads_u32, t_u32.div_ceil(8), 1),
                block_dim: (32, 8, 1),
                shared_mem_bytes: 0,
            };
            unsafe {
                self.stream
                    .launch_builder(&self.fn_attn_f32)
                    .arg(&ws.q)
                    .arg(&ws.k)
                    .arg(&ws.v)
                    .arg(&mut ws.mixed)
                    .arg(&t_u32)
                    .arg(&heads_u32)
                    .arg(&kv_heads_u32)
                    .arg(&hd_u32)
                    .arg(&scale)
                    .launch(cfg)
            }
            .map_err(|e| cuda_err("causal_attention_f32 launch", e))?;
            return Ok(());
        };

        let d = heads * head_dim;
        let kv_dim = kv_heads * head_dim;
        self.convert_x_to_f16(ws.q.slice(..t_len * d), &mut ws.q16, t_len * d)?;
        self.convert_x_to_f16(ws.k.slice(..t_len * kv_dim), &mut ws.k16, t_len * kv_dim)?;
        self.launch_attention_qkv16(ws, t_len, heads, kv_heads, head_dim, scale)
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_attention_qkv16(
        &self,
        ws: &mut CudaWorkspace,
        t_len: usize,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
        scale: f32,
    ) -> Result<(), ThmeError> {
        let Some(blas_mutex) = &self.blas else {
            return Err(ThmeError::ExecutionFailed("cuBLAS unavailable".into()));
        };
        let d = heads * head_dim;
        let kv_dim = kv_heads * head_dim;

        let group = heads / kv_heads;
        let alpha: f32 = 1.0;
        let beta: f32 = 0.0;
        let t_i32 = t_len as i32;
        let hd_i32 = head_dim as i32;
        let d_i32 = d as i32;
        let kv_dim_i32 = kv_dim as i32;
        let tt = (t_len * t_len) as i64;

        {
            let blas = blas_mutex
                .lock()
                .map_err(|_| ThmeError::ExecutionFailed("cuBLAS lock poisoned".into()))?;
            let (q_ptr, _qg) = ws.q16.device_ptr(&self.stream);
            let (k_ptr, _kg) = ws.k16.device_ptr(&self.stream);
            let (s_ptr, _sg) = ws.attn_scores.device_ptr_mut(&self.stream);

            for g in 0..group {
                let q_off = (g * head_dim) as u64 * 2;
                let s_off = (g * t_len * t_len) as u64 * 4;
                let status = unsafe {
                    cublas_sys::cublasGemmStridedBatchedEx(
                        *blas.handle(),
                        cublas_sys::cublasOperation_t::CUBLAS_OP_T,
                        cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                        t_i32,
                        t_i32,
                        hd_i32,
                        (&alpha as *const f32).cast::<c_void>(),
                        k_ptr as *const c_void,
                        cublas_sys::cudaDataType_t::CUDA_R_16F,
                        kv_dim_i32,
                        head_dim as i64,
                        (q_ptr + q_off) as *const c_void,
                        cublas_sys::cudaDataType_t::CUDA_R_16F,
                        d_i32,
                        (group * head_dim) as i64,
                        (&beta as *const f32).cast::<c_void>(),
                        (s_ptr + s_off) as *mut c_void,
                        cublas_sys::cudaDataType_t::CUDA_R_32F,
                        t_i32,
                        (group as i64) * tt,
                        kv_heads as i32,
                        cublas_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                        cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
                    )
                };
                if status != cublas_sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
                    return Err(ThmeError::ExecutionFailed(format!(
                        "cublasGemmStridedBatchedEx QK^T failed: {status:?}"
                    )));
                }
            }
        }

        let t_u32 = t_len as u32;
        let heads_u32 = heads as u32;
        let cfg = LaunchConfig {
            grid_dim: (heads_u32, t_u32, 1),
            block_dim: (BLOCK_256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream
                .launch_builder(&self.fn_softmax)
                .arg(&ws.attn_scores)
                .arg(&mut ws.attn_probs16)
                .arg(&t_u32)
                .arg(&heads_u32)
                .arg(&scale)
                .launch(cfg)
        }
        .map_err(|e| cuda_err("causal_softmax_f16 launch", e))?;

        {
            let blas = blas_mutex
                .lock()
                .map_err(|_| ThmeError::ExecutionFailed("cuBLAS lock poisoned".into()))?;
            let (p_ptr, _pg) = ws.attn_probs16.device_ptr(&self.stream);
            let (v_ptr, _vg) = ws.v16.device_ptr(&self.stream);
            let (o_ptr, _og) = ws.x16.device_ptr_mut(&self.stream);

            for g in 0..group {
                let p_off = (g * t_len * t_len) as u64 * 2;
                let o_off = (g * head_dim) as u64 * 2;
                let status = unsafe {
                    cublas_sys::cublasGemmStridedBatchedEx(
                        *blas.handle(),
                        cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                        cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                        hd_i32,
                        t_i32,
                        t_i32,
                        (&alpha as *const f32).cast::<c_void>(),
                        v_ptr as *const c_void,
                        cublas_sys::cudaDataType_t::CUDA_R_16F,
                        kv_dim_i32,
                        head_dim as i64,
                        (p_ptr + p_off) as *const c_void,
                        cublas_sys::cudaDataType_t::CUDA_R_16F,
                        t_i32,
                        (group as i64) * tt,
                        (&beta as *const f32).cast::<c_void>(),
                        (o_ptr + o_off) as *mut c_void,
                        cublas_sys::cudaDataType_t::CUDA_R_16F,
                        d_i32,
                        (group * head_dim) as i64,
                        kv_heads as i32,
                        cublas_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                        cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
                    )
                };
                if status != cublas_sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
                    return Err(ThmeError::ExecutionFailed(format!(
                        "cublasGemmStridedBatchedEx PV failed: {status:?}"
                    )));
                }
            }
        }
        Ok(())
    }

    fn launch_swiglu_f16(
        &self,
        gate: &CudaSlice<f32>,
        up: &CudaSlice<f32>,
        out16: &mut CudaSlice<u16>,
        n: usize,
    ) -> Result<(), ThmeError> {
        let n_u32 = n as u32;
        unsafe {
            self.stream
                .launch_builder(&self.fn_swiglu_f16)
                .arg(gate)
                .arg(up)
                .arg(out16)
                .arg(&n_u32)
                .launch(grid_1d(n))
        }
        .map_err(|e| cuda_err("swiglu_f16 launch", e))?;
        Ok(())
    }

    fn launch_nll_f16(
        &self,
        logits16: &CudaSlice<u16>,
        targets: &CudaSlice<u32>,
        out: &mut CudaSlice<f32>,
        rows: usize,
        vocab: usize,
        first_row: usize,
    ) -> Result<(), ThmeError> {
        let vocab_u32 = vocab as u32;
        let first_u32 = first_row as u32;
        let cfg = LaunchConfig {
            grid_dim: (rows as u32, 1, 1),
            block_dim: (BLOCK_256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream
                .launch_builder(&self.fn_nll_f16)
                .arg(logits16)
                .arg(targets)
                .arg(out)
                .arg(&vocab_u32)
                .arg(&first_u32)
                .launch(cfg)
        }
        .map_err(|e| cuda_err("nll_f16 launch", e))?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn gemm_f16_to_f16(
        &self,
        x16: &CudaSlice<u16>,
        x_row_offset: usize,
        w16: &CudaSlice<u16>,
        out16: &mut CudaSlice<u16>,
        rows: usize,
        in_dim: usize,
        out_dim: usize,
    ) -> Result<(), ThmeError> {
        if let Some(blas_mutex) = &self.blas {
            let blas = blas_mutex
                .lock()
                .map_err(|_| ThmeError::ExecutionFailed("cuBLAS lock poisoned".into()))?;
            let (w_ptr, _w_guard) = w16.device_ptr(&self.stream);
            let (x_ptr, _x_guard) = x16.device_ptr(&self.stream);
            let (y_ptr, _y_guard) = out16.device_ptr_mut(&self.stream);
            let x_off_bytes = (x_row_offset * in_dim * 2) as u64;
            let f16_acc = std::env::var("THME_CUDA_F16_ACC").as_deref() == Ok("1");
            let alpha_f32: f32 = 1.0;
            let beta_f32: f32 = 0.0;
            let alpha_f16: u16 = 0x3c00;
            let beta_f16: u16 = 0x0000;
            let (alpha_ptr, beta_ptr, compute) = if f16_acc {
                (
                    (&alpha_f16 as *const u16).cast::<c_void>(),
                    (&beta_f16 as *const u16).cast::<c_void>(),
                    cublas_sys::cublasComputeType_t::CUBLAS_COMPUTE_16F,
                )
            } else {
                (
                    (&alpha_f32 as *const f32).cast::<c_void>(),
                    (&beta_f32 as *const f32).cast::<c_void>(),
                    cublas_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                )
            };
            let status = unsafe {
                cublas_sys::cublasGemmEx(
                    *blas.handle(),
                    cublas_sys::cublasOperation_t::CUBLAS_OP_T,
                    cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                    out_dim as i32,
                    rows as i32,
                    in_dim as i32,
                    alpha_ptr,
                    w_ptr as *const c_void,
                    cublas_sys::cudaDataType_t::CUDA_R_16F,
                    in_dim as i32,
                    (x_ptr + x_off_bytes) as *const c_void,
                    cublas_sys::cudaDataType_t::CUDA_R_16F,
                    in_dim as i32,
                    beta_ptr,
                    y_ptr as *mut c_void,
                    cublas_sys::cudaDataType_t::CUDA_R_16F,
                    out_dim as i32,
                    compute,
                    cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
                )
            };
            if status != cublas_sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
                return Err(ThmeError::ExecutionFailed(format!(
                    "cublasGemmEx f16 out failed: {status:?}"
                )));
            }
            return Ok(());
        }

        let rows_u32 = rows as u32;
        let in_u32 = in_dim as u32;
        let out_u32 = out_dim as u32;
        let off_u32 = x_row_offset as u32;
        let one = 1u32;
        let mut dummy_f32 =
            self.stream
                .alloc_zeros::<f32>(1)
                .map_err(|e| ThmeError::BufferAllocationFailed {
                    name: "dummy_f32".into(),
                    bytes: 4,
                    reason: format!("{e:?}"),
                })?;
        let cfg = LaunchConfig {
            grid_dim: (out_u32.div_ceil(16), rows_u32.div_ceil(16), 1),
            block_dim: (16, 16, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream
                .launch_builder(&self.fn_gemm_f16_tiled)
                .arg(x16)
                .arg(w16)
                .arg(&mut dummy_f32)
                .arg(out16)
                .arg(&rows_u32)
                .arg(&in_u32)
                .arg(&out_u32)
                .arg(&off_u32)
                .arg(&one)
                .launch(cfg)
        }
        .map_err(|e| cuda_err("gemm_f16_tiled f16 out launch", e))?;
        Ok(())
    }

    fn launch_add(
        &self,
        a: &CudaSlice<f32>,
        b: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        n: usize,
    ) -> Result<(), ThmeError> {
        let n_u32 = n as u32;
        unsafe {
            self.stream
                .launch_builder(&self.fn_add)
                .arg(a)
                .arg(b)
                .arg(out)
                .arg(&n_u32)
                .launch(grid_1d(n))
        }
        .map_err(|e| cuda_err("add launch", e))?;
        Ok(())
    }

    fn launch_nll(
        &self,
        logits: &CudaSlice<f32>,
        targets: &CudaSlice<u32>,
        out: &mut CudaSlice<f32>,
        rows: usize,
        vocab: usize,
        first_row: usize,
    ) -> Result<(), ThmeError> {
        let vocab_u32 = vocab as u32;
        let first_u32 = first_row as u32;
        let cfg = LaunchConfig {
            grid_dim: (rows as u32, 1, 1),
            block_dim: (BLOCK_256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream
                .launch_builder(&self.fn_nll)
                .arg(logits)
                .arg(targets)
                .arg(out)
                .arg(&vocab_u32)
                .arg(&first_u32)
                .launch(cfg)
        }
        .map_err(|e| cuda_err("nll launch", e))?;
        Ok(())
    }

    fn ensure_scratch_x16(&self, n: usize) -> Result<CudaSlice<u16>, ThmeError> {
        let mut guard = self
            .scratch_x16
            .lock()
            .map_err(|_| ThmeError::ExecutionFailed("CUDA x16 scratch poisoned".into()))?;
        if let Some(buf) = guard.take() {
            if buf.len() >= n {
                return Ok(buf);
            }
        }
        self.stream
            .alloc_zeros::<u16>(n.next_power_of_two())
            .map_err(|e| ThmeError::BufferAllocationFailed {
                name: "scratch_x16".into(),
                bytes: n.next_power_of_two() * 2,
                reason: format!("{e:?}"),
            })
    }

    fn return_scratch_x16(&self, buf: CudaSlice<u16>) {
        if let Ok(mut guard) = self.scratch_x16.lock() {
            *guard = Some(buf);
        }
    }
}

impl GpuBackend for CudaContext {
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
        let w_f32 = weights.as_f32_slice()?;
        let rows = x.len() / in_dim;
        let w16 = self.resident_f16(w_f32)?;
        let x_dev = self
            .stream
            .clone_htod(x)
            .map_err(|e| ThmeError::DataTransferFailed(format!("CUDA x upload: {e:?}")))?;
        let mut x16 = self.ensure_scratch_x16(x.len())?;
        let mut y_dev = self
            .stream
            .alloc_zeros::<f32>(rows * out_dim)
            .map_err(|e| ThmeError::BufferAllocationFailed {
                name: "linear_y".into(),
                bytes: rows * out_dim * 4,
                reason: format!("{e:?}"),
            })?;
        self.linear_into(&x_dev, 0, &mut x16, &w16, &mut y_dev, rows, in_dim, out_dim)?;
        self.return_scratch_x16(x16);

        let out = self
            .stream
            .clone_dtoh(&y_dev)
            .map_err(|e| ThmeError::DataTransferFailed(format!("CUDA y readback: {e:?}")))?;
        check_finite(&out, &format!("linear projection '{}'", weights.name()))?;
        Ok(out)
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

        let rows = x.len() / in_dim;
        let x_dev = self
            .stream
            .clone_htod(x)
            .map_err(|e| ThmeError::DataTransferFailed(format!("CUDA x upload: {e:?}")))?;
        let mut x16 = self.ensure_scratch_x16(x.len())?;
        self.convert_x_to_f16(x_dev.slice(..), &mut x16, x.len())?;

        let mut outs = Vec::with_capacity(ops.len());
        for (w, out_dim) in ops {
            let w16 = self.resident_f16(w)?;
            let mut y_dev = self
                .stream
                .alloc_zeros::<f32>(rows * *out_dim)
                .map_err(|e| ThmeError::BufferAllocationFailed {
                    name: "linear_slice_y".into(),
                    bytes: rows * *out_dim * 4,
                    reason: format!("{e:?}"),
                })?;
            self.gemm_f16_preconverted(&x16, &w16, &mut y_dev, rows, in_dim, *out_dim)?;
            outs.push(y_dev);
        }
        self.return_scratch_x16(x16);

        let mut results = Vec::with_capacity(outs.len());
        for y_dev in outs {
            let y = self
                .stream
                .clone_dtoh(&y_dev)
                .map_err(|e| ThmeError::DataTransferFailed(format!("CUDA y readback: {e:?}")))?;
            check_finite(&y, "linear projection (slice weights)")?;
            results.push(y);
        }
        Ok(results)
    }

    fn forward_decoder(
        &self,
        plan: &DecoderPlan<'_>,
        x0: &[f32],
    ) -> Result<Option<Vec<f32>>, ThmeError> {
        let (d, heads, kv_heads, hd) = (plan.hidden, plan.heads, plan.kv_heads, plan.head_dim);
        if d == 0 || x0.is_empty() || !x0.len().is_multiple_of(d) {
            return Ok(None);
        }
        let t = x0.len() / d;
        let kv_dim = kv_heads * hd;
        let supported = hd > 0
            && hd <= BLOCK_256 as usize
            && heads * hd == d
            && kv_heads > 0
            && heads.is_multiple_of(kv_heads)
            && hd.is_multiple_of(2)
            && plan.rope_cos.len() >= t * (hd / 2)
            && plan.rope_sin.len() >= t * (hd / 2)
            && !plan.layers.is_empty();
        if !supported {
            return Ok(None);
        }

        let scored = t - 1;
        let chunk_rows = match plan.targets {
            Some(tg) if tg.len() == scored => {
                if scored == 0 || plan.vocab == 0 {
                    return Ok(None);
                }
                (NLL_CHUNK_MAX_ELEMS / plan.vocab).clamp(1, scored)
            }
            Some(_) => return Ok(None),
            None => 0,
        };
        let nll_mode = plan.targets.is_some();

        let inter = plan.inter;
        let mut resident = Vec::with_capacity(plan.layers.len());
        for l in &plan.layers {
            resident.push((
                self.resident_f32(l.attn_norm)?,
                self.resident_f32(l.mlp_norm)?,
                [
                    self.resident_f16(l.q)?,
                    self.resident_f16(l.k)?,
                    self.resident_f16(l.v)?,
                    self.resident_f16(l.o)?,
                    self.resident_f16(l.gate)?,
                    self.resident_f16(l.up)?,
                    self.resident_f16(l.down)?,
                ],
            ));
        }
        let final_norm = self.resident_f32(plan.final_norm)?;
        let head = self.resident_f16(plan.head)?;

        let half = hd / 2;
        let key = CudaWorkspaceKey {
            t,
            d,
            kv_dim,
            inter,
            vocab: plan.vocab,
            nll_mode,
            head_dim: hd,
        };
        let mut ws = {
            let mut guard = self
                .workspace
                .lock()
                .map_err(|_| ThmeError::ExecutionFailed("CUDA workspace lock poisoned".into()))?;
            match guard.take() {
                Some((k, w))
                    if k == key && (!nll_mode || w.logits16.len() >= chunk_rows * plan.vocab) =>
                {
                    w
                }
                _ => self.alloc_workspace(key, chunk_rows)?,
            }
        };

        self.stream
            .memcpy_htod(x0, &mut ws.x_a)
            .map_err(|e| ThmeError::DataTransferFailed(format!("CUDA x0 upload: {e:?}")))?;
        if let Some(tg) = plan.targets {
            let mut ids = tg.to_vec();
            ids.resize(t.max(1), 0);
            self.stream
                .memcpy_htod(&ids, &mut ws.targets)
                .map_err(|e| {
                    ThmeError::DataTransferFailed(format!("CUDA targets upload: {e:?}"))
                })?;
        }
        let cos_slice = &plan.rope_cos[..t * half];
        let sin_slice = &plan.rope_sin[..t * half];
        let cos_hash = slice_hash(cos_slice);
        if cos_hash != ws.cos_hash {
            self.stream
                .memcpy_htod(cos_slice, &mut ws.cos)
                .map_err(|e| ThmeError::DataTransferFailed(format!("CUDA cos upload: {e:?}")))?;
            ws.cos_hash = cos_hash;
        }
        let sin_hash = slice_hash(sin_slice);
        if sin_hash != ws.sin_hash {
            self.stream
                .memcpy_htod(sin_slice, &mut ws.sin)
                .map_err(|e| ThmeError::DataTransferFailed(format!("CUDA sin upload: {e:?}")))?;
            ws.sin_hash = sin_hash;
        }

        let td = t * d;
        let tint = t * inter;
        let scale = 1.0 / (hd as f32).sqrt();
        let use_flash = std::env::var("THME_CUDA_FLASH_ATTN").as_deref() == Ok("1");

        if self.blas.is_some() {
            self.launch_rms_norm_f16(&ws.x_a, &resident[0].0, &mut ws.x16, t, d, plan.eps, 0)?;
            for (idx, (_attn_norm, mlp_norm, [wq, wk, wv, wo, wg, wu, wd])) in
                resident.iter().enumerate()
            {
                self.gemm_f16_to_f16(&ws.x16, 0, wq, &mut ws.q16, t, d, d)?;
                self.gemm_f16_to_f16(&ws.x16, 0, wk, &mut ws.k16, t, d, kv_dim)?;
                self.gemm_f16_to_f16(&ws.x16, 0, wv, &mut ws.v16, t, d, kv_dim)?;

                self.launch_rope_qk_f16(
                    &ws.cos,
                    &ws.sin,
                    &mut ws.q16,
                    &mut ws.k16,
                    t,
                    d,
                    kv_dim,
                    hd,
                    half,
                )?;
                if use_flash {
                    self.launch_flash_attention_f16(&mut ws, t, heads, kv_heads, hd, scale)?;
                } else {
                    self.launch_attention_qkv16(&mut ws, t, heads, kv_heads, hd, scale)?;
                }

                self.gemm_f16_to_f16(&ws.x16, 0, wo, &mut ws.tmp16, t, d, d)?;
                self.launch_add_rms_norm_f16(
                    &ws.x_a,
                    &ws.tmp16,
                    mlp_norm,
                    &mut ws.x_b,
                    &mut ws.x16,
                    t,
                    d,
                    plan.eps,
                )?;

                self.gemm_f16_to_f16(&ws.x16, 0, wg, &mut ws.gate16, t, d, inter)?;
                self.gemm_f16_to_f16(&ws.x16, 0, wu, &mut ws.up16, t, d, inter)?;
                self.launch_swiglu_f16_in(&ws.gate16, &ws.up16, &mut ws.x16, tint)?;
                self.gemm_f16_to_f16(&ws.x16, 0, wd, &mut ws.tmp16, t, inter, d)?;

                let next_norm = if idx + 1 < resident.len() {
                    &resident[idx + 1].0
                } else {
                    &final_norm
                };
                self.launch_add_rms_norm_f16(
                    &ws.x_b,
                    &ws.tmp16,
                    next_norm,
                    &mut ws.x_a,
                    &mut ws.x16,
                    t,
                    d,
                    plan.eps,
                )?;
            }
        } else {
            for (attn_norm, mlp_norm, [wq, wk, wv, wo, wg, wu, wd]) in &resident {
                self.launch_rms_norm_f16(&ws.x_a, attn_norm, &mut ws.x16, t, d, plan.eps, 0)?;
                self.gemm_f16_preconverted(&ws.x16, wq, &mut ws.q, t, d, d)?;
                self.gemm_f16_preconverted(&ws.x16, wk, &mut ws.k, t, d, kv_dim)?;
                self.gemm_f16_preconverted(&ws.x16, wv, &mut ws.v, t, d, kv_dim)?;

                self.launch_rope(&ws.cos, &ws.sin, &mut ws.q, t, d, hd, half)?;
                self.launch_rope(&ws.cos, &ws.sin, &mut ws.k, t, kv_dim, hd, half)?;
                self.launch_attention(&mut ws, t, heads, kv_heads, hd, scale)?;

                self.linear_into(&ws.mixed, 0, &mut ws.x16, wo, &mut ws.tmp, t, d, d)?;
                self.launch_add(&ws.x_a, &ws.tmp, &mut ws.x_b, td)?;

                self.launch_rms_norm_f16(&ws.x_b, mlp_norm, &mut ws.x16, t, d, plan.eps, 0)?;
                self.gemm_f16_preconverted(&ws.x16, wg, &mut ws.gate, t, d, inter)?;
                self.gemm_f16_preconverted(&ws.x16, wu, &mut ws.up, t, d, inter)?;
                self.launch_swiglu_f16(&ws.gate, &ws.up, &mut ws.x16, tint)?;
                self.gemm_f16_preconverted(&ws.x16, wd, &mut ws.tmp, t, inter, d)?;
                self.launch_add(&ws.x_b, &ws.tmp, &mut ws.x_a, td)?;
            }
        }

        let mut out = if nll_mode {
            if self.blas.is_none() {
                self.launch_rms_norm_f16(
                    &ws.x_a,
                    &final_norm,
                    &mut ws.x16,
                    scored,
                    d,
                    plan.eps,
                    0,
                )?;
            }
            let mut first = 0;
            while first < scored {
                let rows = chunk_rows.min(scored - first);
                self.gemm_f16_to_f16(&ws.x16, first, &head, &mut ws.logits16, rows, d, plan.vocab)?;
                self.launch_nll_f16(
                    &ws.logits16,
                    &ws.targets,
                    &mut ws.scores,
                    rows,
                    plan.vocab,
                    first,
                )?;
                first += rows;
            }
            let mut scores = self
                .stream
                .clone_dtoh(&ws.scores)
                .map_err(|e| ThmeError::DataTransferFailed(format!("CUDA nll readback: {e:?}")))?;
            scores.truncate(scored);
            scores
        } else {
            self.launch_rms_norm(
                &ws.x_a,
                &final_norm,
                &mut ws.last,
                1,
                d,
                plan.eps,
                (t - 1) * d,
            )?;
            self.linear_into(
                &ws.last,
                0,
                &mut ws.x16,
                &head,
                &mut ws.logits,
                1,
                d,
                plan.vocab,
            )?;
            self.stream.clone_dtoh(&ws.logits).map_err(|e| {
                ThmeError::DataTransferFailed(format!("CUDA logits readback: {e:?}"))
            })?
        };

        if let Ok(mut guard) = self.workspace.lock() {
            *guard = Some((key, ws));
        }
        drop(resident);
        check_finite(&out, "decoder logits")?;
        if nll_mode {
            out.truncate(scored);
        }
        Ok(Some(out))
    }

    fn forward_decoder_extend(
        &self,
        plan: &DecoderPlan<'_>,
        x0: &[f32],
        cache: &mut crate::inference::KvCache,
    ) -> Result<Option<Vec<f32>>, ThmeError> {
        if self.blas.is_none() {
            return Ok(None);
        }
        let (d, heads, kv_heads, hd) = (plan.hidden, plan.heads, plan.kv_heads, plan.head_dim);
        if d == 0 || x0.is_empty() || !x0.len().is_multiple_of(d) {
            return Ok(None);
        }
        let t = x0.len() / d;
        let kv_dim = kv_heads * hd;
        let supported = hd > 0
            && hd <= BLOCK_256 as usize
            && heads * hd == d
            && kv_heads > 0
            && heads.is_multiple_of(kv_heads)
            && hd.is_multiple_of(2)
            && plan.rope_cos.len() >= t * (hd / 2)
            && plan.rope_sin.len() >= t * (hd / 2)
            && !plan.layers.is_empty();
        if !supported {
            return Ok(None);
        }
        if !cache.has_device_state() && !cache.is_empty() {
            return Ok(None);
        }

        struct CudaKvState {
            capacity: usize,
            layers: Vec<(CudaSlice<u16>, CudaSlice<u16>)>,
        }

        let offset = cache.len();
        let needed = offset + t;
        if cache.device_state_mut::<CudaKvState>().is_none() {
            let cap = needed.max(2048).next_power_of_two();
            let mut layers = Vec::with_capacity(plan.layers.len());
            for _ in 0..plan.layers.len() {
                let k = self.stream.alloc_zeros::<u16>(cap * kv_dim).map_err(|e| {
                    ThmeError::BufferAllocationFailed {
                        name: "kv_k".into(),
                        bytes: cap * kv_dim * 2,
                        reason: format!("{e:?}"),
                    }
                })?;
                let v = self.stream.alloc_zeros::<u16>(cap * kv_dim).map_err(|e| {
                    ThmeError::BufferAllocationFailed {
                        name: "kv_v".into(),
                        bytes: cap * kv_dim * 2,
                        reason: format!("{e:?}"),
                    }
                })?;
                layers.push((k, v));
            }
            cache.set_device_state(
                CudaKvState {
                    capacity: cap,
                    layers,
                },
                0,
                0,
            );
        }

        let Some((kv_state, dev_len, dev_bytes)) = cache.device_state_mut::<CudaKvState>() else {
            return Ok(None);
        };
        if needed > kv_state.capacity {
            let new_cap = needed.next_power_of_two();
            for (k_buf, v_buf) in &mut kv_state.layers {
                let mut new_k = self
                    .stream
                    .alloc_zeros::<u16>(new_cap * kv_dim)
                    .map_err(|e| ThmeError::BufferAllocationFailed {
                        name: "kv_k_grow".into(),
                        bytes: new_cap * kv_dim * 2,
                        reason: format!("{e:?}"),
                    })?;
                let mut new_v = self
                    .stream
                    .alloc_zeros::<u16>(new_cap * kv_dim)
                    .map_err(|e| ThmeError::BufferAllocationFailed {
                        name: "kv_v_grow".into(),
                        bytes: new_cap * kv_dim * 2,
                        reason: format!("{e:?}"),
                    })?;
                if offset > 0 {
                    let used = offset * kv_dim;
                    self.stream
                        .memcpy_dtod(&k_buf.slice(..used), &mut new_k.slice_mut(..used))
                        .map_err(|e| {
                            ThmeError::DataTransferFailed(format!("CUDA kv_k grow copy: {e:?}"))
                        })?;
                    self.stream
                        .memcpy_dtod(&v_buf.slice(..used), &mut new_v.slice_mut(..used))
                        .map_err(|e| {
                            ThmeError::DataTransferFailed(format!("CUDA kv_v grow copy: {e:?}"))
                        })?;
                }
                *k_buf = new_k;
                *v_buf = new_v;
            }
            kv_state.capacity = new_cap;
        }

        let inter = plan.inter;
        let mut resident = Vec::with_capacity(plan.layers.len());
        for l in &plan.layers {
            resident.push((
                self.resident_f32(l.attn_norm)?,
                self.resident_f32(l.mlp_norm)?,
                [
                    self.resident_f16(l.q)?,
                    self.resident_f16(l.k)?,
                    self.resident_f16(l.v)?,
                    self.resident_f16(l.o)?,
                    self.resident_f16(l.gate)?,
                    self.resident_f16(l.up)?,
                    self.resident_f16(l.down)?,
                ],
            ));
        }
        let final_norm = self.resident_f32(plan.final_norm)?;
        let head = self.resident_f16(plan.head)?;

        let half = hd / 2;
        let key = CudaWorkspaceKey {
            t,
            d,
            kv_dim,
            inter,
            vocab: plan.vocab,
            nll_mode: false,
            head_dim: hd,
        };
        let mut ws = {
            let mut guard = self
                .workspace
                .lock()
                .map_err(|_| ThmeError::ExecutionFailed("CUDA workspace lock poisoned".into()))?;
            match guard.take() {
                Some((k, w)) if k == key => w,
                _ => self.alloc_workspace(key, 0)?,
            }
        };

        self.stream
            .memcpy_htod(x0, &mut ws.x_a)
            .map_err(|e| ThmeError::DataTransferFailed(format!("CUDA x0 upload: {e:?}")))?;
        let cos_slice = &plan.rope_cos[..t * half];
        let sin_slice = &plan.rope_sin[..t * half];
        let cos_hash = slice_hash(cos_slice);
        if cos_hash != ws.cos_hash {
            self.stream
                .memcpy_htod(cos_slice, &mut ws.cos)
                .map_err(|e| ThmeError::DataTransferFailed(format!("CUDA cos upload: {e:?}")))?;
            ws.cos_hash = cos_hash;
        }
        let sin_hash = slice_hash(sin_slice);
        if sin_hash != ws.sin_hash {
            self.stream
                .memcpy_htod(sin_slice, &mut ws.sin)
                .map_err(|e| ThmeError::DataTransferFailed(format!("CUDA sin upload: {e:?}")))?;
            ws.sin_hash = sin_hash;
        }

        let tint = t * inter;
        let scale = 1.0 / (hd as f32).sqrt();

        self.launch_rms_norm_f16(&ws.x_a, &resident[0].0, &mut ws.x16, t, d, plan.eps, 0)?;
        for (idx, (_attn_norm, mlp_norm, [wq, wk, wv, wo, wg, wu, wd])) in
            resident.iter().enumerate()
        {
            self.gemm_f16_to_f16(&ws.x16, 0, wq, &mut ws.q16, t, d, d)?;
            self.gemm_f16_to_f16(&ws.x16, 0, wk, &mut ws.k16, t, d, kv_dim)?;
            self.gemm_f16_to_f16(&ws.x16, 0, wv, &mut ws.v16, t, d, kv_dim)?;

            self.launch_rope_qk_f16(
                &ws.cos,
                &ws.sin,
                &mut ws.q16,
                &mut ws.k16,
                t,
                d,
                kv_dim,
                hd,
                half,
            )?;

            let (k_cache, v_cache) = &mut kv_state.layers[idx];
            self.launch_kv_append_f16(&ws.k16, &ws.v16, k_cache, v_cache, offset, t, kv_dim)?;
            self.launch_kv_attention_f16(
                &ws.q16,
                k_cache,
                v_cache,
                &mut ws.x16,
                offset,
                t,
                heads,
                kv_heads,
                hd,
                scale,
            )?;

            self.gemm_f16_to_f16(&ws.x16, 0, wo, &mut ws.tmp16, t, d, d)?;
            self.launch_add_rms_norm_f16(
                &ws.x_a,
                &ws.tmp16,
                mlp_norm,
                &mut ws.x_b,
                &mut ws.x16,
                t,
                d,
                plan.eps,
            )?;

            self.gemm_f16_to_f16(&ws.x16, 0, wg, &mut ws.gate16, t, d, inter)?;
            self.gemm_f16_to_f16(&ws.x16, 0, wu, &mut ws.up16, t, d, inter)?;
            self.launch_swiglu_f16_in(&ws.gate16, &ws.up16, &mut ws.x16, tint)?;
            self.gemm_f16_to_f16(&ws.x16, 0, wd, &mut ws.tmp16, t, inter, d)?;

            let next_norm = if idx + 1 < resident.len() {
                &resident[idx + 1].0
            } else {
                &final_norm
            };
            self.launch_add_rms_norm_f16(
                &ws.x_b,
                &ws.tmp16,
                next_norm,
                &mut ws.x_a,
                &mut ws.x16,
                t,
                d,
                plan.eps,
            )?;
        }

        if t == 1 {
            self.gemm_f16_preconverted(&ws.x16, &head, &mut ws.logits, 1, d, plan.vocab)?;
        } else {
            self.launch_rms_norm(
                &ws.x_a,
                &final_norm,
                &mut ws.last,
                1,
                d,
                plan.eps,
                (t - 1) * d,
            )?;
            self.linear_into(
                &ws.last,
                0,
                &mut ws.x16,
                &head,
                &mut ws.logits,
                1,
                d,
                plan.vocab,
            )?;
        }
        let out = self.stream.clone_dtoh(&ws.logits).map_err(|e| {
            ThmeError::DataTransferFailed(format!("CUDA extend logits readback: {e:?}"))
        })?;

        *dev_len = needed;
        *dev_bytes = 2 * kv_state.layers.len() * needed * kv_dim * std::mem::size_of::<u16>();

        if let Ok(mut guard) = self.workspace.lock() {
            *guard = Some((key, ws));
        }
        drop(resident);
        check_finite(&out, "decoder logits")?;
        Ok(Some(out))
    }

    fn score_head_nll(
        &self,
        normed: &[f32],
        head: &[f32],
        d: usize,
        vocab: usize,
        targets: &[u32],
    ) -> Result<Option<Vec<f32>>, ThmeError> {
        if d == 0 || vocab == 0 || normed.is_empty() || !normed.len().is_multiple_of(d) {
            return Ok(None);
        }
        let t = normed.len() / d;
        let scored = t.saturating_sub(1);
        if scored == 0 || targets.len() != scored || head.len() != vocab * d {
            return Ok(None);
        }
        let chunk_rows = (NLL_CHUNK_MAX_ELEMS / vocab).clamp(1, scored);
        let head16 = self.resident_f16(head)?;
        let x_dev = self
            .stream
            .clone_htod(normed)
            .map_err(|e| ThmeError::DataTransferFailed(format!("CUDA head x upload: {e:?}")))?;
        let mut x16 = self.ensure_scratch_x16(chunk_rows * d)?;
        let mut logits = self
            .stream
            .alloc_zeros::<f32>(chunk_rows * vocab)
            .map_err(|e| ThmeError::BufferAllocationFailed {
                name: "head_logits".into(),
                bytes: chunk_rows * vocab * 4,
                reason: format!("{e:?}"),
            })?;
        let mut scores =
            self.stream
                .alloc_zeros::<f32>(t)
                .map_err(|e| ThmeError::BufferAllocationFailed {
                    name: "head_nll".into(),
                    bytes: t * 4,
                    reason: format!("{e:?}"),
                })?;
        let mut ids = targets.to_vec();
        ids.resize(t.max(1), 0);
        let target_dev = self.stream.clone_htod(&ids).map_err(|e| {
            ThmeError::DataTransferFailed(format!("CUDA head targets upload: {e:?}"))
        })?;

        let mut first = 0;
        while first < scored {
            let rows = chunk_rows.min(scored - first);
            self.linear_into(
                &x_dev,
                first,
                &mut x16,
                &head16,
                &mut logits,
                rows,
                d,
                vocab,
            )?;
            self.launch_nll(&logits, &target_dev, &mut scores, rows, vocab, first)?;
            first += rows;
        }
        self.return_scratch_x16(x16);

        let mut out = self
            .stream
            .clone_dtoh(&scores)
            .map_err(|e| ThmeError::DataTransferFailed(format!("CUDA head nll readback: {e:?}")))?;
        out.truncate(scored);
        check_finite(&out, "head nll")?;
        Ok(Some(out))
    }

    fn is_format_supported(&self, dtype: QuantDtype) -> bool {
        matches!(dtype, QuantDtype::F32)
    }

    fn device_name(&self) -> &str {
        &self.device_name
    }

    fn backend_description(&self) -> String {
        if self.blas.is_some() {
            "THME reference forward pass on the GPU (CUDA + cuBLAS Tensor Cores, f16 weights)"
                .to_string()
        } else {
            "THME reference forward pass on the GPU (CUDA pure PTX, f16 weights)".to_string()
        }
    }

    fn clear_cache(&self) {
        if let Ok(mut cache) = self.weight_cache.lock() {
            cache.entries.clear();
            cache.bytes = 0;
            cache.tick = 0;
        }
        if let Ok(mut ws) = self.workspace.lock() {
            *ws = None;
        }
        if let Ok(mut scratch) = self.scratch_x16.lock() {
            *scratch = None;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parallel_f16_conversion_matches_scalar_for_small_and_large_slices() {
        let small: Vec<f32> = (0..129)
            .map(|i| ((i as f32) - 64.0) * 0.0625)
            .chain([0.0, -0.0, f32::INFINITY, f32::NEG_INFINITY, 1e-8, 65536.0])
            .collect();
        let expected_small: Vec<u16> = small.iter().copied().map(f32_to_f16_bits).collect();
        assert_eq!(convert_f32_slice_to_f16(&small), expected_small);

        let large: Vec<f32> = (0..70_000)
            .map(|i| (((i * 17) % 1021) as f32 - 510.0) * 0.015625)
            .collect();
        let expected_large: Vec<u16> = large.iter().copied().map(f32_to_f16_bits).collect();
        assert_eq!(convert_f32_slice_to_f16(&large), expected_large);
    }
}
