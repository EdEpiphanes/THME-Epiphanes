// SPDX-License-Identifier: AGPL-3.0-only
//! Feature audit (THME-3086): which optional GPU features the adapter exposes through wgpu,
//! and a tiny kernel per feature whose output is checked against a CPU result. A feature that
//! is listed but whose kernel fails or does not compile is reported as unusable.

use serde_json::{json, Value};
use std::sync::Arc;
use wgpu::util::DeviceExt;

/// One tiny kernel run: `src` with entry `main`, `inputs` bound read-only at 1.., an output of
/// `out_len` f32 at binding 0. Returns the output or the validation error text.
pub fn run_kernel(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    src: &str,
    inputs: &[&[f32]],
    out_len: usize,
) -> Result<Vec<f32>, String> {
    let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("feature_probe"),
        source: wgpu::ShaderSource::Wgsl(src.into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("feature_probe"),
        layout: None,
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let bytes = out_len * 4;
    let out = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: bytes as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let read = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: bytes as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let ins: Vec<wgpu::Buffer> = inputs
        .iter()
        .map(|d| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: &d.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>(),
                usage: wgpu::BufferUsages::STORAGE,
            })
        })
        .collect();
    let mut entries = vec![wgpu::BindGroupEntry {
        binding: 0,
        resource: out.as_entire_binding(),
    }];
    for (i, b) in ins.iter().enumerate() {
        entries.push(wgpu::BindGroupEntry {
            binding: i as u32 + 1,
            resource: b.as_entire_binding(),
        });
    }
    let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &entries,
    });
    let mut enc = device.create_command_encoder(&Default::default());
    {
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    enc.copy_buffer_to_buffer(&out, 0, &read, 0, bytes as u64);
    queue.submit([enc.finish()]);
    if let Some(e) = pollster::block_on(scope.pop()) {
        return Err(e.to_string());
    }
    let (tx, rx) = std::sync::mpsc::channel();
    read.slice(..).map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .map_err(|e| e.to_string())?;
    rx.recv()
        .map_err(|e| e.to_string())?
        .map_err(|e| e.to_string())?;
    let data = read.slice(..).get_mapped_range().map_err(|e| e.to_string())?;
    let v = data
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    drop(data);
    read.unmap();
    Ok(v)
}

const F16_KERNEL: &str = "enable f16;
@group(0) @binding(0) var<storage, read_write> o: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    let x = f16(g.x);
    o[g.x] = f32(x * f16(2.0) + f16(0.5));
}";

const SUBGROUP_KERNEL: &str = "
@group(0) @binding(0) var<storage, read_write> o: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>, @builtin(subgroup_size) sg: u32) {
    o[g.x * 2u] = f32(subgroupAdd(1u));
    o[g.x * 2u + 1u] = f32(sg);
}";

const DOT4_KERNEL: &str = "@group(0) @binding(0) var<storage, read_write> o: array<f32>;
@compute @workgroup_size(32)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    let i = i32(g.x);
    // bytes (i-2, -3, 4, 5-i) against (3, 2, -1, i)
    let a = u32(i - 2 & 255) | (u32(-3 & 255) << 8u) | (u32(4) << 16u) | (u32(5 - i & 255) << 24u);
    let b = u32(3) | (u32(2) << 8u) | (u32(-1 & 255) << 16u) | (u32(i) << 24u);
    o[g.x] = f32(dot4I8Packed(a, b));
}";

fn scalar_name(t: wgpu::CooperativeScalarType) -> &'static str {
    match t {
        wgpu::CooperativeScalarType::F32 => "f32",
        wgpu::CooperativeScalarType::F16 => "f16",
        wgpu::CooperativeScalarType::I32 => "i32",
        wgpu::CooperativeScalarType::U32 => "u32",
    }
}

/// One 16 x 16 (or 8 x 8) multiply-add `C = A * B + 0` in workgroup memory.
pub fn coop_kernel(ab: &str, cr: &str, n: u32, workgroup: u32) -> String {
    let f16 = if ab == "f16" || cr == "f16" {
        "enable f16;\n"
    } else {
        ""
    };
    let nn = n * n;
    format!(
        "{f16}enable wgpu_cooperative_matrix;
@group(0) @binding(0) var<storage, read_write> o: array<f32>;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
var<workgroup> wa: array<{ab}, {nn}>;
var<workgroup> wb: array<{ab}, {nn}>;
var<workgroup> wc: array<{cr}, {nn}>;
@compute @workgroup_size({workgroup})
fn main(@builtin(local_invocation_index) li: u32) {{
    for (var i = li; i < {nn}u; i += {workgroup}u) {{
        wa[i] = {ab}(a[i]);
        wb[i] = {ab}(b[i]);
        wc[i] = {cr}(0.0);
    }}
    workgroupBarrier();
    let ma = coopLoadT<coop_mat{n}x{n}<{ab},A>>(&wa[0], {n}u);
    let mb = coopLoadT<coop_mat{n}x{n}<{ab},B>>(&wb[0], {n}u);
    let mc = coopLoadT<coop_mat{n}x{n}<{cr},C>>(&wc[0], {n}u);
    let r = coopMultiplyAdd(ma, mb, mc);
    coopStoreT(r, &wc[0], {n}u);
    workgroupBarrier();
    for (var i = li; i < {nn}u; i += {workgroup}u) {{
        o[i] = f32(wc[i]);
    }}
}}"
    )
}

fn small_matrix(n: usize, seed: usize) -> Vec<f32> {
    (0..n * n)
        .map(|i| ((i * seed + i / n) % 7) as f32 - 3.0)
        .collect()
}

fn coop_check(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    ab: &str,
    cr: &str,
    n: usize,
    workgroup: u32,
) -> Result<(), String> {
    let a = small_matrix(n, 3);
    let b = small_matrix(n, 5);
    let got = run_kernel(
        device,
        queue,
        &coop_kernel(ab, cr, n as u32, workgroup),
        &[&a, &b],
        n * n,
    )?;
    for r in 0..n {
        for c in 0..n {
            let want: f32 = (0..n).map(|k| a[r * n + k] * b[k * n + c]).sum();
            let have = got.get(r * n + c).copied().unwrap_or(f32::NAN);
            if have != want {
                return Err(format!("element ({r},{c}): got {have}, want {want}"));
            }
        }
    }
    Ok(())
}

fn verdict(r: Result<(), String>) -> Value {
    match r {
        Ok(()) => json!({ "kernel_check_passed": true }),
        Err(e) => json!({ "kernel_check_passed": false, "error": e }),
    }
}

/// Audits adapter `adapter`; the report is the committed JSON.
pub fn audit(adapter: &wgpu::Adapter) -> Value {
    let info = adapter.get_info();
    let have = adapter.features();
    let limits = adapter.limits();
    let wanted = wgpu::Features::SHADER_F16
        | wgpu::Features::SUBGROUP
        | wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX;
    let request = have & wanted;
    let mut report = json!({
        "device": info.name,
        "driver": format!("{} {}", info.driver, info.driver_info).trim().to_string(),
        "backend": format!("{:?}", info.backend),
        "device_type": format!("{:?}", info.device_type),
        "build_profile": if cfg!(debug_assertions) { "debug" } else { "release" },
        "wgpu": "30.0",
        "limits": {
            "max_compute_workgroup_storage_size": limits.max_compute_workgroup_storage_size,
            "max_compute_invocations_per_workgroup": limits.max_compute_invocations_per_workgroup,
            "max_buffer_size": limits.max_buffer_size,
            "subgroup_min_size": info.subgroup_min_size,
            "subgroup_max_size": info.subgroup_max_size,
        },
        "features": {
            "SHADER_F16": { "present": have.contains(wgpu::Features::SHADER_F16) },
            "SUBGROUP": { "present": have.contains(wgpu::Features::SUBGROUP) },
            "EXPERIMENTAL_COOPERATIVE_MATRIX": {
                "present": have.contains(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX)
            },
            "packed_4x8_integer_dot_product": { "present": null },
        },
    });
    let configs = adapter.cooperative_matrix_properties();
    report["cooperative_matrix_shapes"] = Value::Array(
        configs
            .iter()
            .map(|c| {
                json!({
                    "m": c.m_size, "n": c.n_size, "k": c.k_size,
                    "ab": scalar_name(c.ab_type), "cr": scalar_name(c.cr_type),
                    "saturating": c.saturating_accumulation,
                })
            })
            .collect(),
    );
    // SAFETY: the audit only runs tiny kernels with known inputs and checks every result.
    let experimental = unsafe { wgpu::ExperimentalFeatures::enabled() };
    let dev = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("thme_feature_audit"),
        required_features: request,
        required_limits: limits.clone(),
        experimental_features: experimental,
        ..wgpu::DeviceDescriptor::default()
    }));
    let (device, queue) = match dev {
        Ok((d, q)) => (Arc::new(d), Arc::new(q)),
        Err(e) => {
            report["device_request_error"] = json!(e.to_string());
            return report;
        }
    };
    let features = &mut report["features"];
    if request.contains(wgpu::Features::SHADER_F16) {
        let want: Vec<f32> = (0..64).map(|i| 2.0 * i as f32 + 0.5).collect();
        let r = run_kernel(&device, &queue, F16_KERNEL, &[], 64)
            .and_then(|g| (g == want).then_some(()).ok_or("wrong values".into()));
        features["SHADER_F16"]["check"] = verdict(r);
    }
    if request.contains(wgpu::Features::SUBGROUP) {
        let r = run_kernel(&device, &queue, SUBGROUP_KERNEL, &[], 128).and_then(|g| {
            let size = g.get(1).copied().unwrap_or(0.0);
            let ok = g.chunks(2).all(|p| p[0] == size && p[1] == size) && size >= 1.0;
            ok.then_some(size)
                .ok_or_else(|| "subgroupAdd did not equal subgroup_size".to_string())
        });
        features["SUBGROUP"]["check"] = match r {
            Ok(s) => json!({ "kernel_check_passed": true, "observed_subgroup_size": s }),
            Err(e) => json!({ "kernel_check_passed": false, "error": e }),
        };
    }
    // dot4I8Packed needs no device feature in wgpu 30; the kernel decides.
    let want: Vec<f32> = (0..32)
        .map(|i: i32| (3 * (i - 2) - 6 - 4 + (5 - i) * i) as f32)
        .collect();
    let r = run_kernel(&device, &queue, DOT4_KERNEL, &[], 32)
        .and_then(|g| (g == want).then_some(()).ok_or("wrong values".into()));
    features["packed_4x8_integer_dot_product"] = {
        let mut v = verdict(r);
        v["present"] = json!(v["kernel_check_passed"] == true);
        v
    };
    if request.contains(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX) {
        let workgroup = info.subgroup_min_size.max(32);
        let mut checks = Vec::new();
        for c in configs.iter().filter(|c| c.m_size == c.n_size) {
            if c.m_size != c.k_size {
                continue;
            }
            let (ab, cr) = (scalar_name(c.ab_type), scalar_name(c.cr_type));
            if !matches!(ab, "f16" | "f32") || !matches!(cr, "f16" | "f32") {
                continue;
            }
            let mut v = verdict(coop_check(
                &device,
                &queue,
                ab,
                cr,
                c.m_size as usize,
                workgroup,
            ));
            v["m"] = json!(c.m_size);
            v["ab"] = json!(ab);
            v["cr"] = json!(cr);
            checks.push(v);
        }
        features["EXPERIMENTAL_COOPERATIVE_MATRIX"]["checks"] = Value::Array(checks);
    }
    report
}

/// True when the report shows an f16-in, f32-accumulate square shape whose kernel passed.
pub fn coop_f16_usable(report: &Value) -> bool {
    report["features"]["EXPERIMENTAL_COOPERATIVE_MATRIX"]["checks"]
        .as_array()
        .is_some_and(|c| {
            c.iter().any(|v| {
                v["ab"] == "f16" && v["cr"] == "f32" && v["kernel_check_passed"] == true
            })
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn adapter() -> Option<wgpu::Adapter> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            flags: wgpu::InstanceFlags::empty(),
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        pollster::block_on(instance.enumerate_adapters(wgpu::Backends::PRIMARY))
            .into_iter()
            .find(|a| a.get_info().device_type == wgpu::DeviceType::DiscreteGpu)
    }

    #[test]
    fn coop_kernel_text_names_the_shape_and_types() {
        let s = coop_kernel("f16", "f32", 16, 32);
        assert!(s.contains("coop_mat16x16<f16,A>"));
        assert!(s.contains("coop_mat16x16<f32,C>"));
        assert!(s.contains("workgroup_size(32)"));
    }

    #[test]
    fn a_listed_feature_never_hides_a_failed_kernel() {
        let Some(a) = adapter() else { return };
        let r = audit(&a);
        // Every check that ran is a verdict, and the dot-product flag follows its own check.
        let dot = &r["features"]["packed_4x8_integer_dot_product"];
        assert_eq!(dot["present"], dot["kernel_check_passed"]);
        for f in ["SHADER_F16", "SUBGROUP"] {
            let c = &r["features"][f]["check"];
            if r["features"][f]["present"] == true {
                assert_eq!(c["kernel_check_passed"], true, "{f}: {r}");
            }
        }
    }
}
