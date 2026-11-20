// SPDX-License-Identifier: AGPL-3.0-only
//! Device-resident forward pass for LLaMA-style decoders.
//!
//! The whole layer stack (norms, projections, rotary embedding, attention, SwiGLU and
//! residual adds) is encoded into ONE command buffer with a single read-back of the last
//! position's logits. This keeps the GPU busy (idle gaps between small submissions let
//! the driver drop clocks) and removes every per-layer host round trip.

use std::sync::atomic::Ordering;
use std::time::Instant;

use super::backend::{
    bump, check_finite, storage_buffer, storage_entry, DeviceWeight, GemvParams, GpuContext,
    MAX_GROUPS, PARAM_STRIDE,
};
use super::error::ThmeError;
use wgpu::util::DeviceExt;

/// Longest context the attention kernel's shared score row holds.
pub const MAX_DEVICE_CONTEXT: usize = 2048;

/// Borrowed weights of one decoder layer, each row-major `[out, in]` (norms are `[d]`).
pub struct LayerWeights<'a> {
    pub attn_norm: &'a [f32],
    pub q: &'a [f32],
    pub k: &'a [f32],
    pub v: &'a [f32],
    pub o: &'a [f32],
    pub mlp_norm: &'a [f32],
    pub gate: &'a [f32],
    pub up: &'a [f32],
    pub down: &'a [f32],
}

/// Everything the device needs to run the stack on already embedded rows.
pub struct DecoderPlan<'a> {
    pub layers: Vec<LayerWeights<'a>>,
    pub final_norm: &'a [f32],
    /// Output head, `[vocab, d]` (the embedding matrix when tied).
    pub head: &'a [f32],
    pub hidden: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub inter: usize,
    pub vocab: usize,
    pub eps: f32,
    /// Rotary tables `[positions, head_dim / 2]`, built by the caller so parity is exact.
    pub rope_cos: &'a [f32],
    pub rope_sin: &'a [f32],
    /// Next-token ids, one per scored position (`t - 1`). When set, the device returns the
    /// negative log-likelihood of each target instead of the last position's logits, so the
    /// only read-back is `t` floats.
    pub targets: Option<&'a [u32]>,
}

pub(super) struct DecoderKernels {
    pub(super) layout: wgpu::BindGroupLayout,
    rms_norm: wgpu::ComputePipeline,
    rope: wgpu::ComputePipeline,
    attention: wgpu::ComputePipeline,
    add: wgpu::ComputePipeline,
    swiglu: wgpu::ComputePipeline,
    pub(super) nll: wgpu::ComputePipeline,
    pipeline_layout: wgpu::PipelineLayout,
    /// Tiled attention pipelines by head size (compiled on first use).
    tiled: std::sync::Mutex<TiledPipelines>,
}

type TiledPipelines = std::collections::HashMap<usize, std::sync::Arc<wgpu::ComputePipeline>>;

/// Head sizes the tiled attention kernel supports.
pub(super) fn tiled_attention_supports(head_dim: usize) -> bool {
    head_dim > 0 && head_dim <= 128 && head_dim.is_multiple_of(4)
}

impl DecoderKernels {
    pub(super) fn new(device: &wgpu::Device) -> Self {
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("decoder_ops_bgl"),
            entries: &[
                storage_entry(0, true),
                storage_entry(1, true),
                storage_entry(2, true),
                storage_entry(3, true),
                storage_entry(4, false),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("decoder_ops_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("decoder_ops"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/decoder_ops.wgsl").into()),
        });
        let build = |entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        Self {
            rms_norm: build("rms_norm"),
            rope: build("rope"),
            attention: build("attention"),
            add: build("add"),
            swiglu: build("swiglu"),
            nll: build("nll"),
            layout,
            pipeline_layout,
            tiled: std::sync::Mutex::new(std::collections::HashMap::new()),
        }
    }
}

impl DecoderKernels {
    /// The cooperative-matrix attention pipeline (head size 64), compiled once; cached under the
    /// head size 0 of the tiled map.
    fn coop_attention(
        &self,
        device: &wgpu::Device,
    ) -> Result<std::sync::Arc<wgpu::ComputePipeline>, ThmeError> {
        let mut cache = self
            .tiled
            .lock()
            .map_err(|_| ThmeError::ExecutionFailed("attention pipeline cache poisoned".into()))?;
        if let Some(hit) = cache.get(&0) {
            return Ok(std::sync::Arc::clone(hit));
        }
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("attention_coop"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/attention_coop.wgsl").into()),
        });
        let pipeline = std::sync::Arc::new(device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some("attention_coop"),
                layout: Some(&self.pipeline_layout),
                module: &module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            },
        ));
        cache.insert(0, std::sync::Arc::clone(&pipeline));
        Ok(pipeline)
    }

    /// Compiles the cooperative-matrix attention pipeline ahead of use (the capability probe).
    pub(super) fn coop_attention_pipeline(&self, device: &wgpu::Device) -> Result<(), ThmeError> {
        self.coop_attention(device).map(|_| ())
    }

    /// The tiled attention pipeline for `head_dim`, compiled once per head size.
    fn tiled_attention(
        &self,
        device: &wgpu::Device,
        head_dim: usize,
    ) -> Result<std::sync::Arc<wgpu::ComputePipeline>, ThmeError> {
        let mut cache = self
            .tiled
            .lock()
            .map_err(|_| ThmeError::ExecutionFailed("attention pipeline cache poisoned".into()))?;
        if let Some(hit) = cache.get(&head_dim) {
            return Ok(std::sync::Arc::clone(hit));
        }
        let source = include_str!("shaders/attention_tiled.wgsl")
            .replace("HEAD_DIM_VEC4", &(head_dim / 4).to_string());
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("attention_tiled"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let pipeline = std::sync::Arc::new(device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some("attention_tiled"),
                layout: Some(&self.pipeline_layout),
                module: &module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            },
        ));
        cache.insert(head_dim, std::sync::Arc::clone(&pipeline));
        Ok(pipeline)
    }
}

/// Records dispatches into one pass, giving each its own 256-byte parameter block.
struct Recorder<'a> {
    ctx: &'a GpuContext,
    encoder: &'a mut wgpu::CommandEncoder,
    /// The open compute pass; in profile mode every dispatch gets its own, timed pass.
    pass: Option<wgpu::ComputePass<'static>>,
    /// Single-pass begin/end timestamps (whole-stack timing), unused in profile mode.
    whole_pass: Option<&'a super::backend::TimestampProbe>,
    profile: Option<OpProfile>,
    params: &'a wgpu::Buffer,
    param_bytes: Vec<u8>,
    dispatches: u64,
}

/// Per-dispatch timestamp slots and the label of each.
pub(super) struct OpProfile {
    pub(super) queries: wgpu::QuerySet,
    pub(super) labels: Vec<String>,
}

/// Most dispatches one profiled submission can time (two timestamps each, 4096 per set).
const MAX_PROFILED_DISPATCHES: usize = 2048;

impl<'a> Recorder<'a> {
    /// The pass the next dispatch goes into, labelled for the profile.
    fn pass_for(&mut self, label: String) -> Result<&mut wgpu::ComputePass<'static>, ThmeError> {
        let timed = match self.profile.as_mut() {
            Some(prof) => {
                let slot = prof.labels.len();
                if slot >= MAX_PROFILED_DISPATCHES {
                    return Err(ThmeError::ExecutionFailed(
                        "too many dispatches for one profiled submission".into(),
                    ));
                }
                prof.labels.push(label);
                Some(slot as u32)
            }
            None => None,
        };
        if timed.is_some() || self.pass.is_none() {
            self.pass = None; // end the open pass before the encoder begins another
            let writes = match (timed, self.profile.as_ref(), self.whole_pass) {
                (Some(slot), Some(prof), _) => Some(wgpu::ComputePassTimestampWrites {
                    query_set: &prof.queries,
                    beginning_of_pass_write_index: Some(2 * slot),
                    end_of_pass_write_index: Some(2 * slot + 1),
                }),
                (None, _, Some(p)) => Some(wgpu::ComputePassTimestampWrites {
                    query_set: &p.queries,
                    beginning_of_pass_write_index: Some(0),
                    end_of_pass_write_index: Some(1),
                }),
                _ => None,
            };
            let pass = self
                .encoder
                .begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("decoder_pass"),
                    timestamp_writes: writes,
                })
                .forget_lifetime();
            return Ok(self.pass.insert(pass));
        }
        self.pass
            .as_mut()
            .ok_or_else(|| ThmeError::ExecutionFailed("no open compute pass".into()))
    }

    /// Ends the open pass and returns the parameter bytes, dispatch count and profile.
    fn finish(mut self) -> (Vec<u8>, u64, Option<OpProfile>) {
        self.pass = None;
        (self.param_bytes, self.dispatches, self.profile)
    }

    fn params_binding(&mut self, words: &[u8]) -> wgpu::BindingResource<'a> {
        let offset = self.param_bytes.len();
        self.param_bytes.extend_from_slice(words);
        self.param_bytes.resize(offset + PARAM_STRIDE, 0);
        wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer: self.params,
            offset: offset as u64,
            size: wgpu::BufferSize::new(32),
        })
    }

    /// Y[rows, out_dim] = X[rows, in_dim] * W^T, written at `out` from float 0.
    #[allow(clippy::too_many_arguments)]
    fn linear(
        &mut self,
        x: &wgpu::Buffer,
        x_row0: usize,
        w: &DeviceWeight,
        out: &wgpu::Buffer,
        rows: usize,
        in_dim: usize,
        out_dim: usize,
    ) -> Result<(), ThmeError> {
        let params = GemvParams {
            in_dim: in_dim as u32,
            out_dim: out_dim as u32,
            num_blocks_per_row: 0,
            rows: rows as u32,
            out_offset: 0,
            _pad: [0; 3],
        };
        let ctx = self.ctx;
        let (pipeline, groups, kernel) = ctx.dense_dispatch(w.storage, rows, in_dim, out_dim, 0)?;
        let binding = self.params_binding(bytemuck::bytes_of(&params));
        let x_binding = wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer: x,
            offset: (x_row0 * in_dim * 4) as u64,
            size: None,
        });
        let xh = if kernel == super::backend::COOP_DIRECT_KERNEL {
            let pass = self.pass_for(format!("convert_x_f16 {rows}x{in_dim}"))?;
            let xh = ctx.coop_convert_x(
                pass,
                binding.clone(),
                x_binding.clone(),
                w.buffer.as_entire_binding(),
                rows,
                in_dim,
            )?;
            self.dispatches += 1;
            Some(xh)
        } else {
            None
        };
        let x_resource = match &xh {
            Some(b) => b.as_entire_binding(),
            None => x_binding,
        };
        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("decoder_linear"),
            layout: &ctx.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: binding,
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: x_resource,
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: w.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: out.as_entire_binding(),
                },
            ],
        });
        let label = format!("linear {kernel} {rows}x{in_dim}x{out_dim}");
        let pass = self.pass_for(label)?;
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(groups[0], groups[1], groups[2]);
        self.dispatches += 1;
        Ok(())
    }

    /// One elementwise or reduction kernel from `decoder_ops.wgsl`.
    fn op(
        &mut self,
        name: &str,
        pipeline: &wgpu::ComputePipeline,
        words: [u32; 8],
        bufs: [&wgpu::Buffer; 4],
        groups: (u32, u32, u32),
        layout: &wgpu::BindGroupLayout,
    ) -> Result<(), ThmeError> {
        let ctx = self.ctx;
        let binding = self.params_binding(bytemuck::bytes_of(&words));
        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("decoder_op"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: binding,
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: bufs[0].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: bufs[1].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: bufs[2].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: bufs[3].as_entire_binding(),
                },
            ],
        });
        let pass = self.pass_for(name.to_string())?;
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(groups.0, groups.1, groups.2);
        self.dispatches += 1;
        Ok(())
    }
}

/// Workgroup grid for `n` elements at 64 per workgroup, spread over x then y.
fn grid(n: usize) -> (u32, u32, u32) {
    let groups = (n as u32).div_ceil(64);
    (groups.min(MAX_GROUPS), groups.div_ceil(MAX_GROUPS), 1)
}

fn words(a: &[u32]) -> [u32; 8] {
    let mut w = [0u32; 8];
    w[..a.len()].copy_from_slice(a);
    w
}

impl GpuContext {
    /// Rows of logits scored per head dispatch, so a chunk stays under ~64 MB and starts at an
    /// aligned buffer offset. `None` when the device cannot hold even 64 rows.
    fn nll_chunk_rows(&self, d: usize, vocab: usize, scored: usize) -> Option<usize> {
        let limits = self.device.limits();
        let cap = limits
            .max_buffer_size
            .min(limits.max_storage_buffer_binding_size)
            .min(64 << 20) as usize;
        let rows = ((cap / (vocab * 4)) / 64 * 64).min(scored.div_ceil(64) * 64);
        let aligned = rows >= scored
            || (rows * d * 4).is_multiple_of(limits.min_storage_buffer_offset_alignment as usize);
        (rows > 0 && aligned).then_some(rows)
    }

    /// Output head and next-token negative log-likelihood for normalized rows `[t, d]`;
    /// returns `t - 1` scores and reads back nothing else.
    pub fn score_head_nll(
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
        let scored = t - 1;
        if scored == 0 || targets.len() != scored {
            return Ok(None);
        }
        let Some(chunk_rows) = self.nll_chunk_rows(d, vocab, scored) else {
            return Ok(None);
        };
        let head_buf = self.resident_linear(head, d, vocab)?;
        let dev = &self.device;
        let io = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let src = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC;
        let f32s = |n: usize| (n * 4).max(16);
        let x = storage_buffer(dev, "head_x", f32s(t * d), io);
        let logits = storage_buffer(dev, "head_logits", f32s(chunk_rows * vocab), src);
        let scores = storage_buffer(dev, "head_nll", f32s(t), src);
        let dummy = storage_buffer(dev, "head_dummy", 16, io);
        let staging = storage_buffer(
            dev,
            "head_staging",
            f32s(t),
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );
        let mut ids = targets.to_vec();
        ids.resize(t.max(4), 0);
        let target_buf = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("head_targets"),
            contents: bytemuck::cast_slice(&ids),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let slots = 2 * (t.div_ceil(64) + 1);
        let params = storage_buffer(dev, "head_params", slots * PARAM_STRIDE, io);
        self.queue.write_buffer(&x, 0, bytemuck::cast_slice(normed));

        let mut encoder = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("head_nll_encoder"),
        });
        let (param_bytes, dispatches) = {
            let mut rec = Recorder {
                ctx: self,
                encoder: &mut encoder,
                pass: None,
                whole_pass: None,
                profile: None,
                params: &params,
                param_bytes: Vec::with_capacity(slots * PARAM_STRIDE),
                dispatches: 0,
            };
            let mut first = 0;
            while first < scored {
                let rows = chunk_rows.min(scored - first);
                rec.linear(&x, first, &head_buf, &logits, rows, d, vocab)?;
                rec.op(
                    "nll",
                    &self.decoder.nll,
                    words(&[rows as u32, vocab as u32, first as u32, 0]),
                    [&logits, &target_buf, &dummy, &scores],
                    (rows as u32, 1, 1),
                    &self.decoder.layout,
                )?;
                first += rows;
            }
            let (bytes, dispatches, _) = rec.finish();
            (bytes, dispatches)
        };
        let bytes = (t * 4) as u64;
        encoder.copy_buffer_to_buffer(&scores, 0, &staging, 0, bytes);
        self.counters
            .dispatches
            .fetch_add(dispatches, Ordering::Relaxed);
        self.counters.submits.fetch_add(1, Ordering::Relaxed);
        self.queue.write_buffer(&params, 0, &param_bytes);
        self.queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..bytes);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        self.wait_for_device()?;
        rx.recv()
            .map_err(|e| ThmeError::DataTransferFailed(format!("GPU channel recv failed: {e}")))?
            .map_err(|e| ThmeError::DataTransferFailed(format!("GPU buffer map failed: {e}")))?;
        let data = slice
            .get_mapped_range()
            .map_err(|e| ThmeError::DataTransferFailed(format!("GPU map range failed: {e}")))?;
        let mut out: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging.unmap();
        out.truncate(scored);
        check_finite(&out, "head nll")?;
        Ok(Some(out))
    }
}

impl GpuContext {
    /// Runs the full decoder stack on the device for `x0` (embedded rows `[t, hidden]`) and
    /// returns the last position's logits `[vocab]`.
    ///
    /// Returns `Ok(None)` when the shape is outside what the kernels support (the caller then
    /// uses the per-operation path).
    pub fn forward_decoder(
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
        let supported = t <= MAX_DEVICE_CONTEXT
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
            Some(tg) if tg.len() == scored => match self.nll_chunk_rows(d, plan.vocab, scored) {
                Some(rows) => rows,
                None => return Ok(None),
            },
            Some(_) => return Ok(None),
            None => 0,
        };
        let nll_mode = plan.targets.is_some();
        let out_len = if nll_mode { t } else { plan.vocab };

        // Weights stay resident; keep the handles alive until the work has finished.
        let norm_buf = |w: &[f32]| self.resident_slice(w);
        let inter = plan.inter;
        let lin =
            |w: &[f32], in_dim: usize, out_dim: usize| self.resident_linear(w, in_dim, out_dim);
        let mut resident = Vec::with_capacity(plan.layers.len());
        for l in &plan.layers {
            resident.push((
                norm_buf(l.attn_norm)?,
                norm_buf(l.mlp_norm)?,
                [
                    lin(l.q, d, d)?,
                    lin(l.k, d, kv_dim)?,
                    lin(l.v, d, kv_dim)?,
                    lin(l.o, d, d)?,
                    lin(l.gate, d, inter)?,
                    lin(l.up, d, inter)?,
                    lin(l.down, inter, d)?,
                ],
            ));
        }
        let final_norm = norm_buf(plan.final_norm)?;
        let head = lin(plan.head, d, plan.vocab)?;

        let dev = &self.device;
        let io = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let f32s = |n: usize| (n * 4).max(16);
        let act = |label: &str, n: usize| storage_buffer(dev, label, f32s(n), io);
        let x_a = storage_buffer(dev, "x_a", f32s(t * d), io);
        let x_b = act("x_b", t * d);
        let normed = act("normed", t * d);
        let q = act("q", t * d);
        let k = act("k", t * kv_dim);
        let v = act("v", t * kv_dim);
        let mixed = act("mixed", t * d);
        let tmp = act("tmp", t * d);
        let gate = act("gate", t * plan.inter);
        let up = act("up", t * plan.inter);
        let hidden = act("hidden", t * plan.inter);
        let last = act("last_normed", d);
        let src = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC;
        let logits = storage_buffer(
            dev,
            "logits",
            f32s(if nll_mode {
                chunk_rows * plan.vocab
            } else {
                plan.vocab
            }),
            src,
        );
        let scores = storage_buffer(dev, "nll", f32s(t), src);
        let mut target_ids = plan.targets.unwrap_or(&[]).to_vec();
        target_ids.resize(t.max(4), 0);
        let target_buf = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("targets"),
            contents: bytemuck::cast_slice(&target_ids),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let dummy = act("dummy", 4);
        let logits_staging = storage_buffer(
            dev,
            "logits_staging",
            f32s(out_len),
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );
        let half = hd / 2;
        let cos = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("rope_cos"),
            contents: bytemuck::cast_slice(&plan.rope_cos[..t * half]),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let sin = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("rope_sin"),
            contents: bytemuck::cast_slice(&plan.rope_sin[..t * half]),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let slots = 16 * plan.layers.len() + 4 + 2 * (t.div_ceil(64) + 1);
        let params = storage_buffer(dev, "decoder_params", slots * PARAM_STRIDE, io);

        let t_in = Instant::now();
        self.queue.write_buffer(&x_a, 0, bytemuck::cast_slice(x0));
        bump(&self.counters.input_write_ns, t_in);

        let t_enc = Instant::now();
        let probe = self
            .timestamps
            .as_ref()
            .filter(|_| self.kernel_timing.load(Ordering::Relaxed));
        let op_profile = probe.is_some() && self.op_profile.load(Ordering::Relaxed);
        let mut encoder = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("decoder_encoder"),
        });
        let (param_bytes, dispatches, profile) = {
            let mut rec = Recorder {
                ctx: self,
                encoder: &mut encoder,
                pass: None,
                whole_pass: if op_profile { None } else { probe },
                profile: op_profile.then(|| OpProfile {
                    queries: dev.create_query_set(&wgpu::QuerySetDescriptor {
                        label: Some("op_profile"),
                        ty: wgpu::QueryType::Timestamp,
                        count: 2 * MAX_PROFILED_DISPATCHES as u32,
                    }),
                    labels: Vec::new(),
                }),
                params: &params,
                param_bytes: Vec::with_capacity(slots * PARAM_STRIDE),
                dispatches: 0,
            };
            let kx = &self.decoder;
            let lay = &kx.layout;
            let (td, tint) = (t * d, t * plan.inter);
            let coop_attention = self.attention_kernel_name(hd) == "coop";
            let tiled_attention = match self.attention_kernel_name(hd) {
                "tiled" => Some(kx.tiled_attention(dev, hd)?),
                "coop" => Some(kx.coop_attention(dev)?),
                _ => None,
            };
            let eps = plan.eps.to_bits();
            let scale = (1.0 / (hd as f32).sqrt()).to_bits();
            for res in &resident {
                let (attn_norm, mlp_norm, [wq, wk, wv, wo, wg, wu, wd]) = res;
                rec.op(
                    "rms_norm",
                    &kx.rms_norm,
                    words(&[t as u32, d as u32, eps, 0]),
                    [&x_a, attn_norm, &dummy, &normed],
                    (t as u32, 1, 1),
                    lay,
                )?;
                rec.linear(&normed, 0, wq, &q, t, d, d)?;
                rec.linear(&normed, 0, wk, &k, t, d, kv_dim)?;
                rec.linear(&normed, 0, wv, &v, t, d, kv_dim)?;
                rec.op(
                    "rope",
                    &kx.rope,
                    words(&[t as u32, d as u32, hd as u32, half as u32]),
                    [&cos, &sin, &dummy, &q],
                    grid(t * (d / hd) * half),
                    lay,
                )?;
                rec.op(
                    "rope",
                    &kx.rope,
                    words(&[t as u32, kv_dim as u32, hd as u32, half as u32]),
                    [&cos, &sin, &dummy, &k],
                    grid(t * kv_heads * half),
                    lay,
                )?;
                match &tiled_attention {
                    Some(tiled) => rec.op(
                        if coop_attention {
                            "attention coop"
                        } else {
                            "attention tiled"
                        },
                        tiled,
                        words(&[t as u32, heads as u32, kv_heads as u32, hd as u32, scale]),
                        [&q, &k, &v, &mixed],
                        (t.div_ceil(64) as u32, heads as u32, 1),
                        lay,
                    )?,
                    None => rec.op(
                        "attention",
                        &kx.attention,
                        words(&[t as u32, heads as u32, kv_heads as u32, hd as u32, scale]),
                        [&q, &k, &v, &mixed],
                        (heads as u32, t as u32, 1),
                        lay,
                    )?,
                }
                rec.linear(&mixed, 0, wo, &tmp, t, d, d)?;
                rec.op(
                    "add",
                    &kx.add,
                    words(&[td as u32]),
                    [&x_a, &tmp, &dummy, &x_b],
                    grid(td),
                    lay,
                )?;
                rec.op(
                    "rms_norm",
                    &kx.rms_norm,
                    words(&[t as u32, d as u32, eps, 0]),
                    [&x_b, mlp_norm, &dummy, &normed],
                    (t as u32, 1, 1),
                    lay,
                )?;
                rec.linear(&normed, 0, wg, &gate, t, d, plan.inter)?;
                rec.linear(&normed, 0, wu, &up, t, d, plan.inter)?;
                rec.op(
                    "swiglu",
                    &kx.swiglu,
                    words(&[tint as u32]),
                    [&gate, &up, &dummy, &hidden],
                    grid(tint),
                    lay,
                )?;
                rec.linear(&hidden, 0, wd, &tmp, t, plan.inter, d)?;
                rec.op(
                    "add",
                    &kx.add,
                    words(&[td as u32]),
                    [&x_b, &tmp, &dummy, &x_a],
                    grid(td),
                    lay,
                )?;
            }
            if nll_mode {
                rec.op(
                    "rms_norm",
                    &kx.rms_norm,
                    words(&[t as u32, d as u32, eps, 0]),
                    [&x_a, &final_norm, &dummy, &normed],
                    (t as u32, 1, 1),
                    lay,
                )?;
                let mut first = 0;
                while first < scored {
                    let rows = chunk_rows.min(scored - first);
                    rec.linear(&normed, first, &head, &logits, rows, d, plan.vocab)?;
                    rec.op(
                        "nll",
                        &kx.nll,
                        words(&[rows as u32, plan.vocab as u32, first as u32, 0]),
                        [&logits, &target_buf, &dummy, &scores],
                        (rows as u32, 1, 1),
                        lay,
                    )?;
                    first += rows;
                }
            } else {
                // Final norm of the last position only, then the output head.
                rec.op(
                    "rms_norm",
                    &kx.rms_norm,
                    words(&[1, d as u32, eps, ((t - 1) * d) as u32]),
                    [&x_a, &final_norm, &dummy, &last],
                    (1, 1, 1),
                    lay,
                )?;
                rec.linear(&last, 0, &head, &logits, 1, d, plan.vocab)?;
            }
            rec.finish()
        };
        let logit_bytes = (out_len * 4) as u64;
        let result = if nll_mode { &scores } else { &logits };
        encoder.copy_buffer_to_buffer(result, 0, &logits_staging, 0, logit_bytes);
        let op_readback = profile.as_ref().map(|prof| {
            let n = prof.labels.len() as u32 * 2;
            let bytes = u64::from(n) * 8;
            let resolve = storage_buffer(
                dev,
                "op_profile_resolve",
                bytes as usize,
                wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            );
            let staging = storage_buffer(
                dev,
                "op_profile_staging",
                bytes as usize,
                wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            );
            encoder.resolve_query_set(&prof.queries, 0..n, &resolve, 0);
            encoder.copy_buffer_to_buffer(&resolve, 0, &staging, 0, bytes);
            staging
        });
        if let (Some(p), false) = (probe, op_profile) {
            encoder.resolve_query_set(&p.queries, 0..2, &p.resolve, 0);
            encoder.copy_buffer_to_buffer(&p.resolve, 0, &p.staging, 0, 16);
        }
        let finished = encoder.finish();
        bump(&self.counters.encode_ns, t_enc);
        self.counters
            .dispatches
            .fetch_add(dispatches, Ordering::Relaxed);
        self.counters.submits.fetch_add(1, Ordering::Relaxed);

        let t_in = Instant::now();
        self.queue.write_buffer(&params, 0, &param_bytes);
        bump(&self.counters.input_write_ns, t_in);

        let t_sub = Instant::now();
        self.queue.submit(Some(finished));
        bump(&self.counters.submit_ns, t_sub);

        let t_wait = Instant::now();
        let slice = logits_staging.slice(..logit_bytes);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        self.wait_for_device()?;
        rx.recv()
            .map_err(|e| ThmeError::DataTransferFailed(format!("GPU channel recv failed: {e}")))?
            .map_err(|e| ThmeError::DataTransferFailed(format!("GPU buffer map failed: {e}")))?;
        bump(&self.counters.wait_ns, t_wait);
        if let (Some(p), false) = (probe, op_profile) {
            self.read_kernel_time(p)?;
        }
        if let (Some(prof), Some(staging), Some(p)) = (&profile, &op_readback, probe) {
            self.read_op_profile(p, prof, staging)?;
        }

        let t_read = Instant::now();
        let data = slice
            .get_mapped_range()
            .map_err(|e| ThmeError::DataTransferFailed(format!("GPU map range failed: {e}")))?;
        let mut out: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        if nll_mode {
            out.truncate(scored);
        }
        drop(data);
        logits_staging.unmap();
        bump(&self.counters.readback_ns, t_read);
        drop(resident);
        check_finite(&out, "decoder logits")?;
        Ok(Some(out))
    }
}
