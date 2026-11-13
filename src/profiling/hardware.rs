// SPDX-License-Identifier: AGPL-3.0-only
use serde::{Deserialize, Serialize};
use std::fmt;
use std::str::FromStr;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PrecisionType {
    FP32,
    FP16,
    BF16,
    INT8,
    INT4,
    Ternary,
}

impl PrecisionType {
    pub fn bytes_per_element(&self) -> f64 {
        match self {
            Self::FP32 => 4.0,
            Self::FP16 | Self::BF16 => 2.0,
            Self::INT8 => 1.0,
            Self::INT4 => 0.5,
            Self::Ternary => 0.2,
        }
    }

    pub fn display_name(&self) -> &'static str {
        match self {
            Self::FP32 => "FP32",
            Self::FP16 => "FP16",
            Self::BF16 => "BF16",
            Self::INT8 => "INT8",
            Self::INT4 => "INT4 (Q4)",
            Self::Ternary => "Ternary (b1.58)",
        }
    }
}

impl fmt::Display for PrecisionType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.display_name())
    }
}

impl FromStr for PrecisionType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let normalized = s.trim().to_lowercase();
        match normalized.as_str() {
            "fp32" | "float32" | "f32" => Ok(Self::FP32),
            "fp16" | "float16" | "f16" => Ok(Self::FP16),
            "bf16" | "bfloat16" => Ok(Self::BF16),
            "int8" | "i8" | "q8" | "q8_0" => Ok(Self::INT8),
            "int4" | "i4" | "q4" | "q4_0" | "q4_k" => Ok(Self::INT4),
            "ternary" | "b1.58" | "bitnet" | "1.58" => Ok(Self::Ternary),
            _ => Err(format!("Unknown precision type: {s}")),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum HardwareTarget {
    H100,
    RTX4090,
    AppleM3Max,
    MI300X,
    IntelEmeraldRapids,
}

impl fmt::Display for HardwareTarget {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::H100 => write!(f, "NVIDIA H100 SXM5"),
            Self::RTX4090 => write!(f, "NVIDIA GeForce RTX 4090"),
            Self::AppleM3Max => write!(f, "Apple M3 Max"),
            Self::MI300X => write!(f, "AMD Instinct MI300X"),
            Self::IntelEmeraldRapids => write!(f, "Intel Xeon Platinum 8592+"),
        }
    }
}

impl FromStr for HardwareTarget {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let norm = s.trim().to_lowercase();
        if norm.contains("h100") || norm.contains("hopper") {
            Ok(Self::H100)
        } else if norm.contains("4090") || norm.contains("ada") {
            Ok(Self::RTX4090)
        } else if norm.contains("m3") || norm.contains("apple") {
            Ok(Self::AppleM3Max)
        } else if norm.contains("mi300") || norm.contains("cdna") {
            Ok(Self::MI300X)
        } else if norm.contains("emerald") || norm.contains("8592") || norm.contains("xeon") {
            Ok(Self::IntelEmeraldRapids)
        } else {
            Err(format!("Unknown hardware target: {s}"))
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheTierSpec {
    pub name: String,
    pub capacity_bytes: usize,
    pub bandwidth_gbps: f64,
    pub latency_ns: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareDevice {
    pub name: String,
    pub vendor: String,
    pub architecture: String,
    pub peak_fp32_tflops: f64,
    pub peak_fp16_tflops: f64,
    pub peak_int8_tops: f64,
    pub dram_bandwidth_gbps: f64,
    pub tdp_watts: f64,
    pub idle_power_watts: f64,
    pub caches: Vec<CacheTierSpec>,
}

impl HardwareDevice {
    pub fn peak_throughput_tflops(&self, precision: PrecisionType) -> f64 {
        match precision {
            PrecisionType::FP32 => self.peak_fp32_tflops,
            PrecisionType::FP16 | PrecisionType::BF16 => self.peak_fp16_tflops,
            PrecisionType::INT8 => self.peak_int8_tops,
            PrecisionType::INT4 => self.peak_int8_tops * 2.0,
            PrecisionType::Ternary => self.peak_int8_tops * 4.0,
        }
    }

    pub fn machine_balance(&self, precision: PrecisionType) -> f64 {
        if self.dram_bandwidth_gbps <= 0.0 {
            return 0.0;
        }
        let peak_flops = self.peak_throughput_tflops(precision) * 1e12;
        let bandwidth_bytes = self.dram_bandwidth_gbps * 1e9;
        peak_flops / bandwidth_bytes
    }

    pub fn machine_balance_knee(&self, precision: PrecisionType) -> f64 {
        self.machine_balance(precision)
    }

    pub fn memory_bandwidth_tbps(&self) -> f64 {
        self.dram_bandwidth_gbps / 1000.0
    }

    pub fn peak_tflops_fp16(&self) -> f64 {
        self.peak_fp16_tflops
    }

    pub fn from_target(target: HardwareTarget) -> Self {
        match target {
            HardwareTarget::H100 => Self::predefined_h100(),
            HardwareTarget::RTX4090 => Self::predefined_rtx4090(),
            HardwareTarget::AppleM3Max => Self::predefined_apple_m3_max(),
            HardwareTarget::MI300X => Self::predefined_mi300x(),
            HardwareTarget::IntelEmeraldRapids => Self::predefined_intel_emerald_rapids(),
        }
    }

    pub fn predefined_h100() -> Self {
        Self {
            name: "NVIDIA H100 SXM5".to_string(),
            vendor: "NVIDIA".to_string(),
            architecture: "Hopper GH100".to_string(),
            peak_fp32_tflops: 67.0,
            peak_fp16_tflops: 989.4,
            peak_int8_tops: 1978.8,
            dram_bandwidth_gbps: 3350.0,
            tdp_watts: 700.0,
            idle_power_watts: 110.0,
            caches: vec![
                CacheTierSpec {
                    name: "L1 SRAM".to_string(),
                    capacity_bytes: 33_800_000,
                    bandwidth_gbps: 22_000.0,
                    latency_ns: 1.5,
                },
                CacheTierSpec {
                    name: "L2 Cache".to_string(),
                    capacity_bytes: 50_000_000,
                    bandwidth_gbps: 6_500.0,
                    latency_ns: 8.0,
                },
                CacheTierSpec {
                    name: "HBM3 DRAM".to_string(),
                    capacity_bytes: 80_000_000_000,
                    bandwidth_gbps: 3350.0,
                    latency_ns: 120.0,
                },
            ],
        }
    }

    pub fn predefined_rtx4090() -> Self {
        Self {
            name: "NVIDIA GeForce RTX 4090".to_string(),
            vendor: "NVIDIA".to_string(),
            architecture: "Ada Lovelace AD102".to_string(),
            peak_fp32_tflops: 82.6,
            peak_fp16_tflops: 165.2,
            peak_int8_tops: 330.4,
            dram_bandwidth_gbps: 1008.0,
            tdp_watts: 450.0,
            idle_power_watts: 35.0,
            caches: vec![
                CacheTierSpec {
                    name: "L1 SRAM".to_string(),
                    capacity_bytes: 16_384_000,
                    bandwidth_gbps: 9_000.0,
                    latency_ns: 1.8,
                },
                CacheTierSpec {
                    name: "L2 Cache".to_string(),
                    capacity_bytes: 72_000_000,
                    bandwidth_gbps: 3_200.0,
                    latency_ns: 12.0,
                },
                CacheTierSpec {
                    name: "GDDR6X DRAM".to_string(),
                    capacity_bytes: 24_000_000_000,
                    bandwidth_gbps: 1008.0,
                    latency_ns: 180.0,
                },
            ],
        }
    }

    pub fn predefined_apple_m3_max() -> Self {
        Self {
            name: "Apple M3 Max".to_string(),
            vendor: "Apple".to_string(),
            architecture: "M3 GPU 40-core".to_string(),
            peak_fp32_tflops: 16.2,
            peak_fp16_tflops: 32.4,
            peak_int8_tops: 64.8,
            dram_bandwidth_gbps: 400.0,
            tdp_watts: 75.0,
            idle_power_watts: 8.0,
            caches: vec![
                CacheTierSpec {
                    name: "L1 Cache".to_string(),
                    capacity_bytes: 5_120_000,
                    bandwidth_gbps: 2_400.0,
                    latency_ns: 2.0,
                },
                CacheTierSpec {
                    name: "L2 Cache".to_string(),
                    capacity_bytes: 16_000_000,
                    bandwidth_gbps: 1_200.0,
                    latency_ns: 10.0,
                },
                CacheTierSpec {
                    name: "SLC (L3)".to_string(),
                    capacity_bytes: 48_000_000,
                    bandwidth_gbps: 800.0,
                    latency_ns: 25.0,
                },
                CacheTierSpec {
                    name: "Unified LPDDR5".to_string(),
                    capacity_bytes: 128_000_000_000,
                    bandwidth_gbps: 400.0,
                    latency_ns: 140.0,
                },
            ],
        }
    }

    pub fn predefined_mi300x() -> Self {
        Self {
            name: "AMD Instinct MI300X".to_string(),
            vendor: "AMD".to_string(),
            architecture: "CDNA 3".to_string(),
            peak_fp32_tflops: 163.4,
            peak_fp16_tflops: 1307.4,
            peak_int8_tops: 2614.8,
            dram_bandwidth_gbps: 5300.0,
            tdp_watts: 750.0,
            idle_power_watts: 120.0,
            caches: vec![
                CacheTierSpec {
                    name: "L1 Cache".to_string(),
                    capacity_bytes: 9_728_000,
                    bandwidth_gbps: 25_000.0,
                    latency_ns: 1.5,
                },
                CacheTierSpec {
                    name: "L2 Cache".to_string(),
                    capacity_bytes: 32_000_000,
                    bandwidth_gbps: 8_000.0,
                    latency_ns: 9.0,
                },
                CacheTierSpec {
                    name: "Mall (L3)".to_string(),
                    capacity_bytes: 256_000_000,
                    bandwidth_gbps: 4_200.0,
                    latency_ns: 30.0,
                },
                CacheTierSpec {
                    name: "HBM3 DRAM".to_string(),
                    capacity_bytes: 192_000_000_000,
                    bandwidth_gbps: 5300.0,
                    latency_ns: 115.0,
                },
            ],
        }
    }

    pub fn predefined_intel_emerald_rapids() -> Self {
        Self {
            name: "Intel Xeon Platinum 8592+".to_string(),
            vendor: "Intel".to_string(),
            architecture: "Emerald Rapids AMX".to_string(),
            peak_fp32_tflops: 7.2,
            peak_fp16_tflops: 179.2,
            peak_int8_tops: 358.4,
            dram_bandwidth_gbps: 560.0,
            tdp_watts: 350.0,
            idle_power_watts: 65.0,
            caches: vec![
                CacheTierSpec {
                    name: "L1 Data".to_string(),
                    capacity_bytes: 3_072_000,
                    bandwidth_gbps: 3_800.0,
                    latency_ns: 1.2,
                },
                CacheTierSpec {
                    name: "L2 Cache".to_string(),
                    capacity_bytes: 128_000_000,
                    bandwidth_gbps: 1_900.0,
                    latency_ns: 8.0,
                },
                CacheTierSpec {
                    name: "L3 LLC".to_string(),
                    capacity_bytes: 320_000_000,
                    bandwidth_gbps: 1_100.0,
                    latency_ns: 35.0,
                },
                CacheTierSpec {
                    name: "DDR5 DRAM".to_string(),
                    capacity_bytes: 512_000_000_000,
                    bandwidth_gbps: 560.0,
                    latency_ns: 90.0,
                },
            ],
        }
    }

    pub fn all_predefined() -> Vec<Self> {
        vec![
            Self::predefined_h100(),
            Self::predefined_rtx4090(),
            Self::predefined_apple_m3_max(),
            Self::predefined_mi300x(),
            Self::predefined_intel_emerald_rapids(),
        ]
    }

    pub fn find_device(query: &str) -> Option<Self> {
        let q = query.to_lowercase();
        Self::all_predefined().into_iter().find(|d| {
            d.name.to_lowercase().contains(&q) || d.architecture.to_lowercase().contains(&q)
        })
    }
}

pub type HardwareSpec = HardwareDevice;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hardware_database_predefined_specs() {
        let devices = HardwareDevice::all_predefined();
        assert_eq!(devices.len(), 5);

        for dev in &devices {
            assert!(dev.peak_fp32_tflops > 0.0);
            assert!(dev.peak_fp16_tflops > 0.0);
            assert!(dev.dram_bandwidth_gbps > 0.0);
            assert!(dev.tdp_watts > 0.0);
            assert!(!dev.caches.is_empty());
            for c in &dev.caches {
                assert!(c.capacity_bytes > 0);
                assert!(c.bandwidth_gbps > 0.0);
                assert!(c.latency_ns > 0.0);
            }
        }
    }

    #[test]
    fn test_machine_balance_knee_calculation() {
        let h100 = HardwareDevice::predefined_h100();
        let fp16_knee = h100.machine_balance(PrecisionType::FP16);
        let expected = (989.4 * 1000.0) / 3350.0;
        assert!((fp16_knee - expected).abs() < 1e-4);
        assert!((fp16_knee - 295.343).abs() < 0.1);

        let fp32_knee = h100.machine_balance(PrecisionType::FP32);
        let expected_fp32 = (67.0 * 1000.0) / 3350.0;
        assert!((fp32_knee - expected_fp32).abs() < 1e-4);
    }

    #[test]
    fn test_find_device_by_substring() {
        assert!(HardwareDevice::find_device("h100").is_some());
        assert!(HardwareDevice::find_device("4090").is_some());
        assert!(HardwareDevice::find_device("m3").is_some());
        assert!(HardwareDevice::find_device("mi300x").is_some());
        assert!(HardwareDevice::find_device("emerald").is_some());
        assert!(HardwareDevice::find_device("nonexistent_device").is_none());
    }

    #[test]
    fn test_precision_type_parsing() {
        assert_eq!(
            "fp16".parse::<PrecisionType>().unwrap(),
            PrecisionType::FP16
        );
        assert_eq!(
            "bf16".parse::<PrecisionType>().unwrap(),
            PrecisionType::BF16
        );
        assert_eq!(
            "int8".parse::<PrecisionType>().unwrap(),
            PrecisionType::INT8
        );
        assert_eq!("q4".parse::<PrecisionType>().unwrap(), PrecisionType::INT4);
        assert_eq!(
            "b1.58".parse::<PrecisionType>().unwrap(),
            PrecisionType::Ternary
        );
        assert!("invalid_prec".parse::<PrecisionType>().is_err());
    }
}
