use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MdevProfile {
    pub type_id: String,
    pub name: String,
    pub available_instances: u32,
    pub description: String,
    pub device_api: String,
}

pub struct MdevManager;

impl MdevManager {
    /// Discovers available mediated device profiles for a given PCI DBDF.
    pub fn list_profiles(dbdf: &str) -> Vec<MdevProfile> {
        let types_path = format!("/sys/bus/pci/devices/{}/mdev_supported_types", dbdf);
        let path = Path::new(&types_path);

        if path.is_dir() {
            Self::scan_sysfs_types(path)
        } else {
            Self::mock_profiles(dbdf)
        }
    }

    fn scan_sysfs_types(types_dir: &Path) -> Vec<MdevProfile> {
        let mut profiles = Vec::new();

        if let Ok(entries) = fs::read_dir(types_dir) {
            for entry in entries.flatten() {
                let type_id = entry.file_name().to_string_lossy().to_string();
                let type_path = entry.path();

                let name = fs::read_to_string(type_path.join("name"))
                    .unwrap_or_else(|_| type_id.clone())
                    .trim()
                    .to_string();

                let available_instances = fs::read_to_string(type_path.join("available_instances"))
                    .ok()
                    .and_then(|s| s.trim().parse::<u32>().ok())
                    .unwrap_or(0);

                let description = fs::read_to_string(type_path.join("description"))
                    .unwrap_or_else(|_| "NVIDIA vGPU Profile".to_string())
                    .trim()
                    .to_string();

                let device_api = fs::read_to_string(type_path.join("device_api"))
                    .unwrap_or_else(|_| "vfio-pci".to_string())
                    .trim()
                    .to_string();

                profiles.push(MdevProfile {
                    type_id,
                    name,
                    available_instances,
                    description,
                    device_api,
                });
            }
        }

        profiles
    }

    fn mock_profiles(dbdf: &str) -> Vec<MdevProfile> {
        vec![
            MdevProfile {
                type_id: format!("nvidia-{}-24q", dbdf.replace([':', '.'], "_")),
                name: "GRID RTX-24Q (24GB Framebuffer)".to_string(),
                available_instances: 1,
                description: "Full-die compute vGPU profile for virtual machines".to_string(),
                device_api: "vfio-pci".to_string(),
            },
            MdevProfile {
                type_id: format!("nvidia-{}-12q", dbdf.replace([':', '.'], "_")),
                name: "GRID RTX-12Q (12GB Framebuffer)".to_string(),
                available_instances: 2,
                description: "Half-die partitioned vGPU profile for dual VM host".to_string(),
                device_api: "vfio-pci".to_string(),
            },
            MdevProfile {
                type_id: format!("nvidia-{}-8q", dbdf.replace([':', '.'], "_")),
                name: "GRID RTX-8Q (8GB Framebuffer)".to_string(),
                available_instances: 3,
                description: "Triple-slice partitioned vGPU profile".to_string(),
                device_api: "vfio-pci".to_string(),
            },
        ]
    }

    /// Generates a QEMU CLI execution string for launching a guest VM with mediated vGPU.
    pub fn generate_qemu_cmd(uuid: &str, memory: &str, cpus: u32, disk: Option<&str>) -> String {
        let disk_arg = disk
            .map(|d| format!(" -drive file={},format=qcow2,if=virtio", d))
            .unwrap_or_else(|| " -drive file=thme-vm.qcow2,format=qcow2,if=virtio".to_string());

        format!(
            "qemu-system-x86_64 -enable-kvm -m {} -smp {}{}\\\n  -device vfio-pci,sysfsdev=/sys/bus/mdev/devices/{},display=off\\\n  -vga none -nographic",
            memory, cpus, disk_arg, uuid
        )
    }

    /// Generates a libvirt hostdev XML snippet for mediated device attachment.
    pub fn generate_libvirt_xml(uuid: &str) -> String {
        format!(
            "<hostdev mode='subsystem' type='mdev' managed='no' model='vfio-pci'>\n  <source>\n    <address uuid='{}'/>\n  </source>\n</hostdev>",
            uuid
        )
    }
}
