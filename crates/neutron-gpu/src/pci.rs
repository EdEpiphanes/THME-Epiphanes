// SPDX-License-Identifier: AGPL-3.0-only
use crate::recipes::{find_recipe, find_recipe_with_vram, CardRecipe};
use serde::Serialize;
use std::fs;
use std::path::Path;

pub const PCI_VENDOR_NVIDIA: u16 = 0x10DE;
pub const PCI_VENDOR_AMD: u16 = 0x1002;
pub const PCI_VENDOR_AMD_CPU: u16 = 0x1022;
pub const PCI_VENDOR_INTEL: u16 = 0x8086;
pub const PCI_VENDOR_APPLE: u16 = 0x106B;
pub const PCI_VENDOR_CORAL: u16 = 0x1AC1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum DeviceTier {
    DiscreteGpu,
    IntegratedApu,
    NeuralNpu,
    TensorTpu,
    ArmSilicon,
}

impl DeviceTier {
    pub const fn label(&self) -> &'static str {
        match self {
            Self::DiscreteGpu => "dGPU",
            Self::IntegratedApu => "APU",
            Self::NeuralNpu => "NPU",
            Self::TensorTpu => "TPU",
            Self::ArmSilicon => "Apple/ARM",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct PciDevice {
    pub dbdf: String,
    pub vendor_id: u16,
    pub device_id: u16,
    pub tier: DeviceTier,
    pub vendor_name: &'static str,
    pub model_name: String,
    pub driver: Option<String>,
    pub power_state: String,
    pub memory_type: &'static str,
    pub matched_recipe: Option<CardRecipe>,
}

// shortcut: sysfs /sys/bus/pci/devices reader with deterministic fallback, upgrade to libpci / ioctl bindings when hotplug events are needed
pub struct PciScanner;

impl PciScanner {
    /// Scans the system PCI bus. Reads from `/sys/bus/pci/devices` on Linux,
    /// queries live hardware utilities (e.g. nvidia-smi) when available,
    /// or falls back to an embedded multi-accelerator simulation table.
    pub fn scan() -> Vec<PciDevice> {
        let sysfs_path = Path::new("/sys/bus/pci/devices");
        if sysfs_path.is_dir() {
            Self::scan_sysfs(sysfs_path)
        } else {
            let live = Self::scan_live();
            if !live.is_empty() {
                live
            } else {
                Self::scan_fallback()
            }
        }
    }

    /// Attempts live hardware discovery on systems where `/sys/bus/pci/devices` is unavailable (e.g. Windows).
    /// Queries utilities such as `nvidia-smi` to discover physical discrete GPUs and operational parameters.
    pub fn scan_live() -> Vec<PciDevice> {
        let mut devices = Vec::new();

        if let Ok(output) = std::process::Command::new("nvidia-smi")
            .args([
                "--query-gpu=name,pci.bus_id,pci.device_id,pci.sub_device_id,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ])
            .output()
        {
            if output.status.success() {
                let stdout = String::from_utf8_lossy(&output.stdout);
                for line in stdout.lines() {
                    let parts: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
                    if parts.len() >= 6 {
                        let raw_name = parts[0];
                        let bus_id = parts[1];
                        let pci_dev_hex = parts[2].trim_start_matches("0x");
                        let _sub_dev = parts[3];
                        let driver_ver = parts[4];
                        let mem_mib = parts[5].parse::<u64>().unwrap_or(0);
                        let vram_bytes = mem_mib * 1024 * 1024;

                        let (vid, did) = if pci_dev_hex.len() >= 8 {
                            if let Ok(val) = u32::from_str_radix(pci_dev_hex, 16) {
                                ((val & 0xFFFF) as u16, ((val >> 16) & 0xFFFF) as u16)
                            } else {
                                (PCI_VENDOR_NVIDIA, 0)
                            }
                        } else if let Ok(val) = u16::from_str_radix(pci_dev_hex, 16) {
                            (PCI_VENDOR_NVIDIA, val)
                        } else {
                            (PCI_VENDOR_NVIDIA, 0)
                        };

                        let matched_recipe = find_recipe_with_vram(did, Some(vram_bytes));
                        let model_name = matched_recipe
                            .as_ref()
                            .map(|r| r.name.to_string())
                            .unwrap_or_else(|| {
                                if !raw_name.is_empty() {
                                    raw_name.to_string()
                                } else {
                                    format!("NVIDIA Device (0x{:04x})", did)
                                }
                            });

                        let memory_type = if did == 0x2803 || did == 0x2805 {
                            "Dedicated GDDR6"
                        } else if (0x2200..=0x2700).contains(&did) {
                            "Dedicated GDDR6X"
                        } else {
                            "Dedicated VRAM"
                        };

                        devices.push(PciDevice {
                            dbdf: bus_id.to_string(),
                            vendor_id: vid,
                            device_id: did,
                            tier: DeviceTier::DiscreteGpu,
                            vendor_name: "NVIDIA",
                            model_name,
                            driver: Some(format!("NVIDIA {}", driver_ver)),
                            power_state: "D0".to_string(),
                            memory_type,
                            matched_recipe,
                        });
                    }
                }
            }
        }

        devices
    }

    fn scan_sysfs(pci_dir: &Path) -> Vec<PciDevice> {
        let mut devices = Vec::new();

        if let Ok(entries) = fs::read_dir(pci_dir) {
            for entry in entries.flatten() {
                let dbdf = entry.file_name().to_string_lossy().to_string();
                let dev_path = entry.path();

                let vendor_id = Self::read_sysfs_hex(&dev_path.join("vendor"));
                let device_id = Self::read_sysfs_hex(&dev_path.join("device"));

                if let (Some(vid), Some(did)) = (vendor_id, device_id) {
                    if let Some((tier, vendor_name, model_name, mem_type)) =
                        Self::classify(vid, did)
                    {
                        let driver = fs::read_link(dev_path.join("driver"))
                            .ok()
                            .and_then(|p| p.file_name().map(|n| n.to_string_lossy().to_string()));

                        let power_state = fs::read_to_string(dev_path.join("power_state"))
                            .unwrap_or_else(|_| "D0".to_string())
                            .trim()
                            .to_string();

                        let matched_recipe = if vid == PCI_VENDOR_NVIDIA {
                            find_recipe(did).cloned()
                        } else {
                            None
                        };

                        devices.push(PciDevice {
                            dbdf,
                            vendor_id: vid,
                            device_id: did,
                            tier,
                            vendor_name,
                            model_name,
                            driver,
                            power_state,
                            memory_type: mem_type,
                            matched_recipe,
                        });
                    }
                }
            }
        }

        devices
    }

    fn classify(
        vendor: u16,
        device: u16,
    ) -> Option<(DeviceTier, &'static str, String, &'static str)> {
        match vendor {
            PCI_VENDOR_NVIDIA => {
                let name = find_recipe(device)
                    .map(|r| r.name.to_string())
                    .unwrap_or_else(|| format!("NVIDIA Device (0x{:04x})", device));
                Some((DeviceTier::DiscreteGpu, "NVIDIA", name, "Dedicated VRAM"))
            }
            PCI_VENDOR_AMD => Some((
                DeviceTier::IntegratedApu,
                "AMD",
                format!("AMD Radeon Graphics / APU (0x{:04x})", device),
                "Unified System RAM",
            )),
            PCI_VENDOR_AMD_CPU => {
                if device == 0x1502 {
                    Some((
                        DeviceTier::NeuralNpu,
                        "AMD",
                        "AMD Ryzen AI XDNA NPU".to_string(),
                        "Tiled SRAM / Host Shared",
                    ))
                } else {
                    None
                }
            }
            PCI_VENDOR_INTEL => {
                if device == 0x7d1d {
                    Some((
                        DeviceTier::NeuralNpu,
                        "Intel",
                        "Intel Core Ultra NPU".to_string(),
                        "On-Chip SRAM",
                    ))
                } else {
                    Some((
                        DeviceTier::IntegratedApu,
                        "Intel",
                        format!("Intel Arc / Iris Xe APU (0x{:04x})", device),
                        "Unified System RAM",
                    ))
                }
            }
            PCI_VENDOR_CORAL => Some((
                DeviceTier::TensorTpu,
                "Google",
                "Coral Edge TPU Accelerator".to_string(),
                "Local Scratchpad SRAM",
            )),
            _ => None,
        }
    }

    fn read_sysfs_hex(path: &Path) -> Option<u16> {
        let content = fs::read_to_string(path).ok()?;
        let trimmed = content.trim().trim_start_matches("0x");
        u16::from_str_radix(trimmed, 16).ok()
    }

    /// Provides comprehensive multi-accelerator simulation covering dGPUs, APUs, NPUs, TPUs, and Apple Silicon.
    pub fn scan_fallback() -> Vec<PciDevice> {
        vec![
            PciDevice {
                dbdf: "0000:01:00.0".to_string(),
                vendor_id: PCI_VENDOR_NVIDIA,
                device_id: 0x2204, // RTX 3090
                tier: DeviceTier::DiscreteGpu,
                vendor_name: "NVIDIA",
                model_name: "GeForce RTX 3090 (24GB)".to_string(),
                driver: None,
                power_state: "D0".to_string(),
                memory_type: "Dedicated GDDR6X",
                matched_recipe: find_recipe(0x2204).cloned(),
            },
            PciDevice {
                dbdf: "0000:00:02.0".to_string(),
                vendor_id: PCI_VENDOR_AMD,
                device_id: 0x164e, // Ryzen 7000/8000 APU
                tier: DeviceTier::IntegratedApu,
                vendor_name: "AMD",
                model_name: "AMD Ryzen AI 9 Phoenix/Strix APU (RDNA3)".to_string(),
                driver: Some("amdgpu".to_string()),
                power_state: "D0".to_string(),
                memory_type: "Unified System LPDDR5X",
                matched_recipe: None,
            },
            PciDevice {
                dbdf: "0000:00:0b.0".to_string(),
                vendor_id: PCI_VENDOR_AMD_CPU,
                device_id: 0x1502, // Ryzen AI XDNA NPU
                tier: DeviceTier::NeuralNpu,
                vendor_name: "AMD",
                model_name: "AMD XDNA Neural Processing Unit (50 TOPS)".to_string(),
                driver: Some("amdxdna".to_string()),
                power_state: "D0".to_string(),
                memory_type: "Tiled On-Chip SRAM (Zero-Copy Host)",
                matched_recipe: None,
            },
            PciDevice {
                dbdf: "0000:00:0e.0".to_string(),
                vendor_id: PCI_VENDOR_INTEL,
                device_id: 0x7d1d, // Intel Lunar Lake NPU
                tier: DeviceTier::NeuralNpu,
                vendor_name: "Intel",
                model_name: "Intel Core Ultra NPU 4.0 (48 TOPS)".to_string(),
                driver: Some("intel_vpu".to_string()),
                power_state: "D0".to_string(),
                memory_type: "Embedded Cache SRAM",
                matched_recipe: None,
            },
            PciDevice {
                dbdf: "soc:apple-m4-max".to_string(),
                vendor_id: PCI_VENDOR_APPLE,
                device_id: 0x0001,
                tier: DeviceTier::ArmSilicon,
                vendor_name: "Apple",
                model_name: "Apple M4 Max SoC (40-core GPU + 16-core ANE + AMX)".to_string(),
                driver: Some("AppleMetal / AppleNeuralEngine".to_string()),
                power_state: "Active (D0)".to_string(),
                memory_type: "Unified Memory 128GB (546 GB/s Zero-Copy)",
                matched_recipe: None,
            },
            PciDevice {
                dbdf: "0000:04:00.0".to_string(),
                vendor_id: PCI_VENDOR_CORAL,
                device_id: 0x089a, // Coral Edge TPU
                tier: DeviceTier::TensorTpu,
                vendor_name: "Google",
                model_name: "Coral Edge TPU (4 TOPS INT8 Matrix Unit)".to_string(),
                driver: Some("apex".to_string()),
                power_state: "D0".to_string(),
                memory_type: "Internal Scratchpad 8MB",
                matched_recipe: None,
            },
        ]
    }
}
