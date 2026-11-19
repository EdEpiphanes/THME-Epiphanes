// SPDX-License-Identifier: AGPL-3.0-only
//! `thme doctor`: say what works on this machine and what to try next.
//! Read-only: nothing is downloaded, loaded or written. Every fact says whether it was checked
//! on this machine or only assumed; a fact that could not be read says unknown.

use serde_json::{json, Value};

use crate::discover::{self, DiscoveredModel, EnvSnapshot};

/// Models listed in full; the rest are only counted.
pub const MAX_SHOWN_MODELS: usize = 10;

/// One host fact and how it was obtained.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Fact<T> {
    /// Read from this machine by a query that succeeded.
    Checked(T),
    /// Taken from the build (for example the target OS), not read from this machine.
    Assumed(T),
    /// The query failed, so the value is not known.
    Unknown,
}

/// Every fact `doctor` reports. `gpu_adapter` is `None` when wgpu finds no Vulkan adapter.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HostFacts {
    pub os: Fact<String>,
    pub arch: Fact<String>,
    pub memory_total_bytes: Fact<u64>,
    pub memory_available_bytes: Fact<u64>,
    pub disk_free_bytes: Fact<u64>,
    pub gpu_adapter: Option<String>,
    pub cpu_threads: Fact<usize>,
    pub models: Vec<DiscoveredModel>,
}

impl<T> Fact<T> {
    fn basis(&self) -> &'static str {
        match self {
            Fact::Checked(_) => "checked",
            Fact::Assumed(_) => "assumed",
            Fact::Unknown => "unknown",
        }
    }

    fn value(&self) -> Option<&T> {
        match self {
            Fact::Checked(v) | Fact::Assumed(v) => Some(v),
            Fact::Unknown => None,
        }
    }
}

/// The next commands to try, chosen from what was found. Pure: no I/O.
pub fn next_steps(facts: &HostFacts) -> Vec<String> {
    let mut steps = Vec::new();
    match facts.models.first() {
        None => steps.push(
            "No models found: run `thme discover --path DIR` on a folder that holds one, or download a small model first (thme doctor downloads nothing)."
                .to_string(),
        ),
        Some(model) => {
            let path = model.path.display();
            steps.push(format!("Inspect the first model found: `thme inspect {path}`"));
            steps.push(format!("Draw it: `thme figure {path} --out figures`"));
        }
    }
    if facts.gpu_adapter.is_some() {
        steps.push(
            "A GPU adapter was found. GPU kernels exist for F32, Q4_K, Q5_K and IQ4_XS only; this check does not measure speed, so no speed is claimed."
                .to_string(),
        );
    }
    steps
}

/// The host-fact lines of the text report, each with its basis. Shared with the Studio Welcome panel.
pub fn fact_lines(facts: &HostFacts) -> Vec<String> {
    let gpu = match &facts.gpu_adapter {
        Some(name) => format!("{name} (checked)"),
        None => "no GPU adapter found (checked)".to_string(),
    };
    vec![
        format!("Operating system: {}", show(&facts.os, |s| s.to_string())),
        format!("Architecture: {}", show(&facts.arch, |s| s.to_string())),
        format!("Memory total: {}", show(&facts.memory_total_bytes, gib)),
        format!(
            "Memory available: {}",
            show(&facts.memory_available_bytes, gib)
        ),
        format!(
            "Disk free (current folder): {}",
            show(&facts.disk_free_bytes, gib)
        ),
        format!("GPU adapter (Vulkan): {gpu}"),
        format!(
            "CPU threads: {}",
            show(&facts.cpu_threads, |n| n.to_string())
        ),
    ]
}

/// Plain-text report: facts with their basis, the models found, then the next steps.
pub fn render_text(facts: &HostFacts) -> String {
    let mut lines = vec!["THME doctor (read-only)".to_string()];
    lines.extend(fact_lines(facts));
    lines.push(format!("Models found: {} (checked)", facts.models.len()));
    let shown_len = facts.models.len().min(MAX_SHOWN_MODELS);
    let shown = facts.models.get(..shown_len).unwrap_or_default();
    if !shown.is_empty() {
        lines.push(discover::render_table(shown));
    }
    if facts.models.len() > shown_len {
        lines.push(format!("... and {} more", facts.models.len() - shown_len));
    }
    lines.push("Next:".to_string());
    lines.extend(next_steps(facts).into_iter().map(|s| format!("  - {s}")));
    lines.join("\n")
}

/// JSON report with one `{value, basis}` object per fact.
pub fn render_json(facts: &HostFacts) -> Value {
    let shown: Vec<Value> = facts
        .models
        .iter()
        .take(MAX_SHOWN_MODELS)
        .map(|m| {
            json!({
                "name": m.name,
                "source": m.source.label(),
                "format": m.format.label(),
                "size_bytes": m.size_bytes,
                "path": m.path.display().to_string(),
            })
        })
        .collect();
    json!({
        "os": fact_json(&facts.os),
        "arch": fact_json(&facts.arch),
        "memory_total_bytes": fact_json(&facts.memory_total_bytes),
        "memory_available_bytes": fact_json(&facts.memory_available_bytes),
        "disk_free_bytes": fact_json(&facts.disk_free_bytes),
        "gpu_adapter": { "value": facts.gpu_adapter, "basis": "checked" },
        "cpu_threads": fact_json(&facts.cpu_threads),
        "models": { "total": facts.models.len(), "shown": shown },
        "next": next_steps(facts),
    })
}

fn fact_json<T: serde::Serialize>(fact: &Fact<T>) -> Value {
    json!({ "value": fact.value(), "basis": fact.basis() })
}

/// A fact as text: the value with its basis, or `unknown`.
fn show<T>(fact: &Fact<T>, text: impl Fn(&T) -> String) -> String {
    match fact {
        Fact::Checked(v) => format!("{} (checked)", text(v)),
        Fact::Assumed(v) => format!("{} (assumed)", text(v)),
        Fact::Unknown => "unknown".to_string(),
    }
}

fn gib(bytes: &u64) -> String {
    format!("{:.1} GiB", *bytes as f64 / (1u64 << 30) as f64)
}

/// Reads the host. Each query that fails becomes `Unknown`, or no GPU adapter; nothing panics.
pub fn collect_host_facts() -> HostFacts {
    let memory = sysinfo::System::new_with_specifics(
        sysinfo::RefreshKind::nothing().with_memory(sysinfo::MemoryRefreshKind::everything()),
    )
    .ok();
    let (total, available) = match memory {
        Some(sys) => (
            checked_if_positive(sys.total_memory()),
            checked_if_positive(sys.available_memory()),
        ),
        None => (Fact::Unknown, Fact::Unknown),
    };
    HostFacts {
        os: Fact::Assumed(std::env::consts::OS.to_string()),
        arch: Fact::Assumed(std::env::consts::ARCH.to_string()),
        memory_total_bytes: total,
        memory_available_bytes: available,
        disk_free_bytes: disk_free_here(),
        gpu_adapter: gpu_adapter_name(),
        cpu_threads: std::thread::available_parallelism()
            .map_or(Fact::Unknown, |n| Fact::Checked(n.get())),
        models: discover::discover(&[], &EnvSnapshot::from_process()).models,
    }
}

/// Zero from sysinfo means it did not report the figure, so it is unknown rather than zero.
fn checked_if_positive(bytes: u64) -> Fact<u64> {
    if bytes == 0 {
        Fact::Unknown
    } else {
        Fact::Checked(bytes)
    }
}

/// Free space on the disk whose mount point is the deepest one holding the current folder.
fn disk_free_here() -> Fact<u64> {
    let (Ok(cwd), Ok(disks)) = (
        std::env::current_dir(),
        sysinfo::Disks::new_with_refreshed_list(),
    ) else {
        return Fact::Unknown;
    };
    let mut best: Option<(usize, u64)> = None;
    for disk in &disks {
        let mount = disk.mount_point();
        let depth = mount.components().count();
        if cwd.starts_with(mount) && best.is_none_or(|(d, _)| depth > d) {
            best = Some((depth, disk.available_space()));
        }
    }
    best.map_or(Fact::Unknown, |(_, free)| Fact::Checked(free))
}

/// Name of the first Vulkan adapter, asked the same way `GpuContext::init` asks.
/// Requests no device, so nothing is allocated on the GPU.
fn gpu_adapter_name() -> Option<String> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN,
        flags: wgpu::InstanceFlags::empty(),
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        ..wgpu::RequestAdapterOptions::default()
    }))
    .ok()?;
    Some(adapter.get_info().name)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discover::{Format, Source};
    use std::path::PathBuf;

    fn model(name: &str) -> DiscoveredModel {
        DiscoveredModel {
            name: name.to_string(),
            source: Source::Folder,
            format: Format::Gguf,
            size_bytes: 1024,
            path: PathBuf::from("models").join(format!("{name}.gguf")),
        }
    }

    fn facts(models: Vec<DiscoveredModel>, gpu: Option<&str>) -> HostFacts {
        HostFacts {
            os: Fact::Assumed("windows".to_string()),
            arch: Fact::Assumed("x86_64".to_string()),
            memory_total_bytes: Fact::Checked(32 << 30),
            memory_available_bytes: Fact::Checked(16 << 30),
            disk_free_bytes: Fact::Checked(100 << 30),
            gpu_adapter: gpu.map(str::to_string),
            cpu_threads: Fact::Checked(12),
            models,
        }
    }

    #[test]
    fn next_with_no_models_points_at_discover_and_downloads_nothing() {
        let steps = next_steps(&facts(vec![], None));
        assert_eq!(steps.len(), 1);
        assert!(steps[0].contains("thme discover --path DIR"));
        assert!(steps[0].contains("downloads nothing"));
    }

    #[test]
    fn next_with_one_model_suggests_inspect_and_figure() {
        let m = model("tiny");
        let path = m.path.display().to_string();
        let steps = next_steps(&facts(vec![m], None));
        assert!(steps
            .iter()
            .any(|s| s.contains(&format!("thme inspect {path}"))));
        assert!(steps
            .iter()
            .any(|s| s.contains(&format!("thme figure {path}"))));
        assert!(steps.iter().all(|s| !s.contains("discover --path")));
    }

    #[test]
    fn next_with_gpu_names_kernels_and_claims_no_speed() {
        let steps = next_steps(&facts(vec![model("tiny")], Some("Test GPU")));
        let gpu = steps
            .iter()
            .find(|s| s.contains("GPU kernels"))
            .expect("a GPU step");
        assert!(gpu.contains("F32, Q4_K, Q5_K and IQ4_XS"));
        assert!(gpu.contains("no speed is claimed"));
    }

    #[test]
    fn next_without_gpu_makes_no_kernel_claim() {
        let steps = next_steps(&facts(vec![model("tiny")], None));
        assert!(steps.iter().all(|s| !s.contains("GPU")));
    }

    #[test]
    fn text_says_when_no_gpu_adapter_is_found() {
        let text = render_text(&facts(vec![], None));
        assert!(text.contains("no GPU adapter found (checked)"));
        assert!(text.contains("Operating system: windows (assumed)"));
        assert!(text.contains("Next:"));
    }

    #[test]
    fn json_has_every_field_and_parses() {
        let text = serde_json::to_string_pretty(&render_json(&facts(
            vec![model("tiny")],
            Some("Test GPU"),
        )))
        .expect("json");
        let v: Value = serde_json::from_str(&text).expect("parses");
        for key in [
            "os",
            "arch",
            "memory_total_bytes",
            "memory_available_bytes",
            "disk_free_bytes",
            "cpu_threads",
        ] {
            assert!(v[key]["basis"].is_string(), "{key} has a basis");
            assert!(v[key].get("value").is_some(), "{key} has a value");
        }
        assert_eq!(v["os"]["basis"], "assumed");
        assert_eq!(v["memory_total_bytes"]["basis"], "checked");
        assert_eq!(v["gpu_adapter"]["value"], "Test GPU");
        assert_eq!(v["models"]["total"], 1);
        assert_eq!(v["models"]["shown"].as_array().map(Vec::len), Some(1));
        assert!(v["next"].is_array());
    }

    #[test]
    fn fact_lines_are_the_lines_the_text_report_prints() {
        let f = facts(vec![model("tiny")], Some("Test GPU"));
        let lines = fact_lines(&f);
        assert_eq!(lines.len(), 7);
        assert!(lines
            .iter()
            .any(|l| l == "GPU adapter (Vulkan): Test GPU (checked)"));
        let text = render_text(&f);
        for line in &lines {
            assert!(text.lines().any(|l| l == line.as_str()), "{line}");
        }
    }

    #[test]
    fn unknown_facts_print_as_unknown() {
        let mut f = facts(vec![], None);
        f.disk_free_bytes = Fact::Unknown;
        let text = render_text(&f);
        assert!(text.contains("Disk free (current folder): unknown"));
        let v = render_json(&f);
        assert_eq!(v["disk_free_bytes"]["basis"], "unknown");
        assert!(v["disk_free_bytes"]["value"].is_null());
    }

    #[test]
    fn text_lists_at_most_ten_models_and_counts_the_rest() {
        let models: Vec<DiscoveredModel> = (0..12).map(|i| model(&format!("m{i}"))).collect();
        let text = render_text(&facts(models.clone(), None));
        assert!(text.contains("... and 2 more"));
        assert!(text.contains("m9.gguf"));
        assert!(!text.contains("m10.gguf"));
        let v = render_json(&facts(models, None));
        assert_eq!(v["models"]["total"], 12);
        assert_eq!(v["models"]["shown"].as_array().map(Vec::len), Some(10));
    }

    #[test]
    fn collect_host_facts_returns_without_panicking() {
        let facts = collect_host_facts();
        assert_eq!(facts.os, Fact::Assumed(std::env::consts::OS.to_string()));
    }
}
