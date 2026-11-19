// SPDX-License-Identifier: AGPL-3.0-only
use anyhow::Result;
use clap::Args;
use std::thread;
use std::time::Duration;

use crate::profiling::telemetry::{GpuTelemetry, SystemTelemetry, SystemTelemetrySnapshot};

#[derive(Args, Debug, Clone)]
pub struct TelemetryArgs {
    /// Output telemetry snapshot in machine-readable JSON format
    #[arg(long)]
    pub json: bool,

    /// Stream live telemetry updates continuously (1 Hz refresh)
    #[arg(short, long)]
    pub watch: bool,

    /// Watch refresh interval in milliseconds (default: 1000 ms)
    #[arg(long, default_value_t = 1000)]
    pub interval_ms: u64,
}

pub fn execute(args: TelemetryArgs) -> Result<()> {
    let mut collector = SystemTelemetry::new()?;

    if args.watch {
        // Continuous watch mode: stream telemetry updates every interval
        loop {
            // Sleep for the configured interval before collecting delta
            thread::sleep(Duration::from_millis(args.interval_ms));
            let snapshot = collector.collect()?;

            if args.json {
                let json_str = serde_json::to_string(&snapshot)?;
                println!("{json_str}");
            } else {
                print_telemetry_dashboard(&snapshot);
            }
        }
    } else {
        // One-shot mode: pause briefly for accurate CPU delta
        thread::sleep(Duration::from_millis(200));
        let snapshot = collector.collect()?;

        if args.json {
            let json_str = serde_json::to_string_pretty(&snapshot)?;
            println!("{json_str}");
        } else {
            print_telemetry_dashboard(&snapshot);
        }
    }

    Ok(())
}

fn format_bytes(bytes: u64) -> String {
    const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
    const MIB: f64 = 1024.0 * 1024.0;

    if bytes >= 1024 * 1024 * 1024 {
        format!("{:.1} GiB", bytes as f64 / GIB)
    } else {
        format!("{:.1} MiB", bytes as f64 / MIB)
    }
}

fn print_telemetry_dashboard(snapshot: &SystemTelemetrySnapshot) {
    println!("================================================================================");
    println!("THME HARDWARE TELEMETRY DASHBOARD");
    println!("================================================================================");

    let cpu = &snapshot.cpu;
    println!("HOST CPU & SYSTEM MEMORY:");
    println!("  Model / Brand      : {}", cpu.brand);
    println!(
        "  Topology           : {} physical cores, {} logical threads",
        cpu.physical_cores, cpu.logical_cpus
    );
    println!("  CPU Utilization    : {:.1}%", cpu.global_cpu_usage_pct);

    let avg_freq: u64 = if !cpu.core_frequencies_mhz.is_empty() {
        cpu.core_frequencies_mhz.iter().sum::<u64>() / cpu.core_frequencies_mhz.len() as u64
    } else {
        0
    };
    if avg_freq > 0 {
        println!("  Avg Clock Frequency: {avg_freq} MHz");
    }

    let ram_used_pct = if cpu.ram_total_bytes > 0 {
        (cpu.ram_used_bytes as f64 / cpu.ram_total_bytes as f64) * 100.0
    } else {
        0.0
    };
    println!(
        "  Physical RAM       : {} used / {} total ({:.1}% used, {} free)",
        format_bytes(cpu.ram_used_bytes),
        format_bytes(cpu.ram_total_bytes),
        ram_used_pct,
        format_bytes(cpu.ram_free_bytes)
    );

    if cpu.swap_total_bytes > 0 {
        let swap_used_pct = (cpu.swap_used_bytes as f64 / cpu.swap_total_bytes as f64) * 100.0;
        println!(
            "  Swap Memory        : {} used / {} total ({:.1}% used)",
            format_bytes(cpu.swap_used_bytes),
            format_bytes(cpu.swap_total_bytes),
            swap_used_pct
        );
    }

    println!("--------------------------------------------------------------------------------");
    if snapshot.gpus.is_empty() {
        println!("ACCELERATORS:");
        println!("  No discrete GPU detected or hardware telemetry query unavailable.");
    } else {
        for (idx, gpu) in snapshot.gpus.iter().enumerate() {
            print_gpu_details(idx, gpu);
        }
    }
    println!("================================================================================");
}

fn print_gpu_details(idx: usize, gpu: &GpuTelemetry) {
    let pci_info = gpu
        .pci_bus_id
        .as_deref()
        .map(|bus| format!(" (PCI {bus})"))
        .unwrap_or_default();
    println!("DISCRETE ACCELERATOR #{idx}:");
    println!("  Device             : {}{pci_info}", gpu.name);

    if let (Some(vid), Some(did)) = (gpu.vendor_id, gpu.device_id) {
        let arch_label = if did == 0x2803 || did == 0x2805 {
            " (Ada Lovelace AD106 silicon)"
        } else if did == 0x2684 {
            " (Ada Lovelace AD102 silicon)"
        } else if did == 0x2204 {
            " (Ampere GA102 silicon)"
        } else {
            ""
        };
        println!("  Hardware ID        : 0x{vid:04X}:0x{did:04X}{arch_label}");
    }

    let vram_pct = if gpu.vram_total_bytes > 0 {
        (gpu.vram_used_bytes as f64 / gpu.vram_total_bytes as f64) * 100.0
    } else {
        0.0
    };
    println!(
        "  VRAM Allocation    : {} used / {} total ({:.1}% used, {} free)",
        format_bytes(gpu.vram_used_bytes),
        format_bytes(gpu.vram_total_bytes),
        vram_pct,
        format_bytes(gpu.vram_free_bytes)
    );

    let max_gfx = gpu
        .max_clock_graphics_mhz
        .map(|m| format!(" (Max: {m} MHz)"))
        .unwrap_or_default();
    let max_mem = gpu
        .max_clock_memory_mhz
        .map(|m| format!(" (Max: {m} MHz)"))
        .unwrap_or_default();
    println!(
        "  Engine Clocks      : Graphics: {} MHz{max_gfx}, Memory: {} MHz{max_mem}",
        gpu.clock_graphics_mhz, gpu.clock_memory_mhz
    );

    let power_pct = if gpu.power_limit_watts > 0.0 {
        (gpu.power_draw_watts / gpu.power_limit_watts) * 100.0
    } else {
        0.0
    };
    println!(
        "  Power Telemetry    : {:.1} W / {:.1} W TDP ({:.1}% capacity)",
        gpu.power_draw_watts, gpu.power_limit_watts, power_pct
    );

    let throttle_status = if gpu.temp_celsius >= 83.0 {
        format!("{:.1} °C (Thermal Throttling Active)", gpu.temp_celsius)
    } else {
        format!("{:.1} °C (Nominal, unthrottled)", gpu.temp_celsius)
    };
    println!("  Die Temperature    : {throttle_status}");

    if let (Some(gpu_util), Some(mem_util)) = (gpu.gpu_utilization_pct, gpu.memory_utilization_pct)
    {
        println!("  Activity Engine    : GPU: {gpu_util:.1}%, Memory Controller: {mem_util:.1}%");
    }
}
