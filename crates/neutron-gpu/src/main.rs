// SPDX-License-Identifier: AGPL-3.0-only
use clap::{Parser, Subcommand};
use neutron_gpu::pci::PciScanner;
use neutron_gpu::recipes::all_recipes;

#[derive(Parser, Debug)]
#[command(
    name = "neutron-gpu",
    version,
    about = "Read-only PCI accelerator discovery and hardware recipe lookup",
    long_about = "Read-only PCI accelerator discovery and display metadata: PCI device and accelerator inspection, and the hardware recipe table."
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Scan the PCI bus for NVIDIA GPU devices and inspect accelerator tiers
    Scan {
        /// Output scan results as formatted JSON
        #[arg(long)]
        json: bool,
    },

    /// List all supported hardware architectures and GPU recipes
    Recipes {
        /// Filter by architecture family (Turing, Ampere, Ada)
        #[arg(short, long)]
        family: Option<String>,

        /// Output recipes as formatted JSON
        #[arg(long)]
        json: bool,
    },

    /// Profile accelerator topology, unified memory, and precision support (APUs, NPUs, TPUs, Apple Silicon)
    Profile {
        /// Filter by accelerator target DBDF or keyword (e.g. apple, npu, tpu, amd, nvidia)
        target: Option<String>,

        /// Output profile as formatted JSON
        #[arg(long)]
        json: bool,
    },
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Scan { json } => {
            let devices = PciScanner::scan();
            if json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&devices)
                        .unwrap_or_else(|e| format!("{{\"error\": \"{e}\"}}"))
                );
            } else {
                println!("=== Neutron-GPU: Hardware Accelerator & Topology Discovery ===");
                println!(
                    "{:<18} {:<10} {:<8} {:<10} {:<42} {:<24} {:<10}",
                    "DBDF / Address",
                    "Tier",
                    "Vendor",
                    "Device ID",
                    "Model / Accelerator",
                    "Memory Architecture",
                    "Driver"
                );
                println!("{}", "-".repeat(126));
                for dev in &devices {
                    let driver = dev.driver.as_deref().unwrap_or("none");

                    println!(
                        "{:<18} {:<10} {:<8} 0x{:04x}     {:<42} {:<24} {:<10}",
                        dev.dbdf,
                        dev.tier.label(),
                        dev.vendor_name,
                        dev.device_id,
                        dev.model_name,
                        dev.memory_type,
                        driver
                    );
                }
                println!(
                    "\nDiscovered {} accelerator(s) across dGPU, APU, NPU, TPU, and Apple Silicon tiers.",
                    devices.len()
                );
            }
        }

        Commands::Recipes { family, json } => {
            let recipes = all_recipes();
            let filtered: Vec<_> = if let Some(fam) = family {
                recipes
                    .iter()
                    .filter(|r| r.family.name().to_lowercase().contains(&fam.to_lowercase()))
                    .collect()
            } else {
                recipes.iter().collect()
            };

            if json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&filtered)
                        .unwrap_or_else(|e| format!("{{\"error\": \"{e}\"}}"))
                );
            } else {
                println!("=== Neutron-GPU: Hardware Recipes ===");
                println!(
                    "{:<10} {:<14} {:<36} {:<8}",
                    "Device ID", "Family", "Model", "Chip ID"
                );
                println!("{}", "-".repeat(72));
                for r in &filtered {
                    println!(
                        "0x{:04x}     {:<14} {:<36} 0x{:04x}",
                        r.device_id,
                        r.family.name(),
                        r.name,
                        r.chip_id
                    );
                }
                println!("\nTotal registered hardware recipes: {}", filtered.len());
            }
        }

        Commands::Profile { target, json } => {
            let devices = PciScanner::scan();
            let filtered: Vec<_> = if let Some(t) = target {
                let term = t.to_lowercase();
                devices
                    .into_iter()
                    .filter(|d| {
                        d.dbdf.to_lowercase().contains(&term)
                            || d.vendor_name.to_lowercase().contains(&term)
                            || d.model_name.to_lowercase().contains(&term)
                            || d.tier.label().to_lowercase().contains(&term)
                    })
                    .collect()
            } else {
                devices
            };

            if json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&filtered)
                        .unwrap_or_else(|e| format!("{{\"error\": \"{e}\"}}"))
                );
            } else {
                println!("=== Neutron-GPU: Deep Accelerator & Precision Architecture Profile ===");
                for dev in &filtered {
                    println!("\n[{}] {}", dev.tier.label(), dev.model_name);
                    println!("  Address / DBDF       : {}", dev.dbdf);
                    println!("  Vendor               : {}", dev.vendor_name);
                    println!("  Memory Architecture  : {}", dev.memory_type);
                    println!("  Power & State        : {}", dev.power_state);

                    match dev.tier {
                        neutron_gpu::pci::DeviceTier::ArmSilicon => {
                            println!("  Execution Backend    : Apple Metal MSL & Apple Neural Engine (ANE)");
                            println!("  Zero-Copy Unified RAM: YES (Shared Host-Device)");
                            println!("  Precision Matrix     : BitNet b1.58 [100% Native], INT4, INT8, FP16, BF16");
                            println!("  Recommendation       : Ideal for low-latency zero-copy LLM surgery & 60 FPS studio");
                        }
                        neutron_gpu::pci::DeviceTier::DiscreteGpu => {
                            println!("  Execution Backend    : WebGPU / Vulkan / CUDA / VFIO");
                            println!("  Zero-Copy Unified RAM: NO (PCIe DMA Staged)");
                            println!("  Precision Matrix     : Tensor Core MMA (FP16, BF16, TF32), BitNet b1.58 (WGSL)");
                            println!("  Recommendation       : High-throughput batch inference & server vGPU virtualization");
                        }
                        neutron_gpu::pci::DeviceTier::IntegratedApu => {
                            println!("  Execution Backend    : WebGPU / Vulkan / ROCm");
                            println!("  Zero-Copy Unified RAM: YES (Unified System Memory)");
                            println!("  Precision Matrix     : BitNet b1.58, INT8, FP16, FP32");
                            println!("  Recommendation       : Efficient on-device model tweaking without discrete power draw");
                        }
                        neutron_gpu::pci::DeviceTier::NeuralNpu => {
                            println!("  Execution Backend    : Tiled Spatial NPU Coprocessor");
                            println!("  Zero-Copy Unified RAM: PARTIAL (Tile-DMA / On-Chip SRAM)");
                            println!(
                                "  Precision Matrix     : INT8 [Peak TOPS], INT4, BitNet b1.58"
                            );
                            println!("  Recommendation       : Ultra low-power continuous background token decoding");
                        }
                        neutron_gpu::pci::DeviceTier::TensorTpu => {
                            println!("  Execution Backend    : Matrix Multiply Unit (MXU)");
                            println!("  Zero-Copy Unified RAM: NO (On-Chip Scratchpad)");
                            println!("  Precision Matrix     : INT8, BF16, BitNet b1.58");
                            println!(
                                "  Recommendation       : Fixed systolic array matrix compute"
                            );
                        }
                    }
                }
            }
        }
    }
}
