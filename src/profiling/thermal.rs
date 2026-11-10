use crate::profiling::hardware::HardwareDevice;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThermalConfig {
    pub ambient_temp_c: f64,
    pub thermal_resistance_k_per_w: f64,
    pub thermal_capacitance_j_per_k: f64,
    pub throttle_temp_c: f64,
    pub critical_shutdown_temp_c: f64,
    pub leakage_coefficient: f64,
}

impl Default for ThermalConfig {
    fn default() -> Self {
        Self {
            ambient_temp_c: 25.0,
            thermal_resistance_k_per_w: 0.09,
            thermal_capacitance_j_per_k: 120.0,
            throttle_temp_c: 83.0,
            critical_shutdown_temp_c: 95.0,
            leakage_coefficient: 0.005,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThermalStepReport {
    pub timestamp_sec: f64,
    pub die_temperature_c: f64,
    pub total_power_watts: f64,
    pub dynamic_power_watts: f64,
    pub static_power_watts: f64,
    pub throttle_factor: f64,
    pub is_throttling: bool,
    pub effective_tflops: f64,
}

pub struct DynamicThermalModel {
    device: HardwareDevice,
    config: ThermalConfig,
    current_temp_c: f64,
}

impl DynamicThermalModel {
    pub fn new(device: HardwareDevice, config: ThermalConfig) -> Self {
        let initial_temp = config.ambient_temp_c;
        Self {
            device,
            config,
            current_temp_c: initial_temp,
        }
    }

    pub fn current_temperature(&self) -> f64 {
        self.current_temp_c
    }

    pub fn set_temperature(&mut self, temp_c: f64) {
        self.current_temp_c = temp_c;
    }

    pub fn steady_state_temp(&self, compute_util: f64, mem_util: f64) -> f64 {
        let max_dyn = self.device.tdp_watts - self.device.idle_power_watts;
        let dyn_power = (0.75 * compute_util + 0.25 * mem_util) * max_dyn;
        let mut t = self.config.ambient_temp_c;
        for _ in 0..10 {
            let static_power = self.device.idle_power_watts
                * (1.0 + self.config.leakage_coefficient * (t - self.config.ambient_temp_c));
            let total_power = dyn_power + static_power;
            t = self.config.ambient_temp_c + total_power * self.config.thermal_resistance_k_per_w;
        }
        t
    }

    pub fn calculate_throttling_factor(temp_c: f32, _tdp_watts: f64) -> f32 {
        if temp_c >= 83.0 {
            let delta = temp_c - 83.0;
            (1.0 - 0.05 * delta).clamp(0.40, 1.0)
        } else {
            1.0
        }
    }

    pub fn step_simulation(
        &mut self,
        dt_sec: f64,
        compute_util: f64,
        mem_util: f64,
    ) -> ThermalStepReport {
        let max_dyn = self.device.tdp_watts - self.device.idle_power_watts;
        let mut dyn_power = (0.75 * compute_util + 0.25 * mem_util) * max_dyn;
        let static_power = self.device.idle_power_watts
            * (1.0
                + self.config.leakage_coefficient
                    * (self.current_temp_c - self.config.ambient_temp_c));

        let (throttle_factor, is_throttling) = if self.current_temp_c >= self.config.throttle_temp_c
        {
            let delta = self.current_temp_c - self.config.throttle_temp_c;
            let factor = (1.0 - 0.05 * delta).clamp(0.40, 1.0);
            (factor, true)
        } else {
            (1.0, false)
        };

        dyn_power *= throttle_factor * throttle_factor;
        let total_power = dyn_power + static_power;

        let p_dissipated = (self.current_temp_c - self.config.ambient_temp_c)
            / self.config.thermal_resistance_k_per_w;
        let dt_temp =
            ((total_power - p_dissipated) / self.config.thermal_capacitance_j_per_k) * dt_sec;
        self.current_temp_c += dt_temp;

        let effective_tflops = self.device.peak_fp16_tflops * throttle_factor;

        ThermalStepReport {
            timestamp_sec: dt_sec,
            die_temperature_c: self.current_temp_c,
            total_power_watts: total_power,
            dynamic_power_watts: dyn_power,
            static_power_watts: static_power,
            throttle_factor,
            is_throttling,
            effective_tflops,
        }
    }
}

pub type ThermalModel = DynamicThermalModel;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dynamic_thermal_and_throttling() {
        let h100 = HardwareDevice::predefined_h100();
        let config = ThermalConfig::default();
        let mut model = DynamicThermalModel::new(h100, config);

        assert_eq!(model.current_temperature(), 25.0);

        let initial_step = model.step_simulation(1.0, 1.0, 1.0);
        assert!(!initial_step.is_throttling);
        assert_eq!(initial_step.throttle_factor, 1.0);
        assert!(model.current_temperature() > 25.0);

        model.set_temperature(88.0);
        let throttled_step = model.step_simulation(1.0, 1.0, 1.0);
        assert!(throttled_step.is_throttling);
        assert!(throttled_step.throttle_factor < 1.0);

        let factor_f32 = DynamicThermalModel::calculate_throttling_factor(88.0, 700.0);
        assert!(factor_f32 < 1.0);
        let normal_factor = DynamicThermalModel::calculate_throttling_factor(70.0, 700.0);
        assert_eq!(normal_factor, 1.0);
    }
}
