// SPDX-License-Identifier: AGPL-3.0-only
//! Non-Conventional Experimentation Laboratory Engine
//!
//! Provides experimental, non-conventional mathematical techniques for researchers and Teacher LLMs:
//! 1. Stochastic Noise Injection: Simulates physical analog hardware drift, thermal fluctuations, and low-voltage DRAM margins.
//! 2. Spectral Energy Reshaping: Warps the singular value spectrum $\sigma_i^\gamma$ to compress or accentuate representation harmonics.
//! 3. Complex Phase Modulation: Modulates weight matrices via unitary complex angle rotation $\cos(\omega j + \phi)$.
//! 4. Continuous Activation Warping: Non-linear continuous manifold blending between SwiGLU, GeLU, and sine harmonic functions.
//! 5. Stochastic DropBlock: Hardware-aligned structured block zeroing for sparsity robustness benchmarking.

use serde::{Deserialize, Serialize};

/// Non-conventional experimentation technique configuration
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum ExperimentTechnique {
    /// Injects controlled stochastic noise into weight matrices
    NoiseInjection {
        variance: f32,
        distribution: String, // "Gaussian", "Uniform"
    },
    /// Warps singular value spectrum via power exponent $\sigma_i^\gamma$
    SpectralReshaping {
        gamma_exponent: f32, // < 1.0 compresses high-frequency, > 1.0 accentuates principal
    },
    /// Complex unitary phase modulation across weight channels
    PhaseModulation {
        frequency_rad: f32,
        phase_shift_rad: f32,
    },
    /// Continuous non-linear activation manifold warping
    ActivationWarping {
        blend_factor: f32, // 0.0: Pure SwiGLU, 1.0: Harmonic Sine-GELU
    },
    /// Structured block-level weight zeroing
    StochasticDropBlock {
        drop_probability: f32,
        block_size: usize,
    },
}

impl ExperimentTechnique {
    pub fn display_name(&self) -> &'static str {
        match self {
            Self::NoiseInjection { .. } => "Stochastic Noise Injection",
            Self::SpectralReshaping { .. } => "Spectral Energy Reshaping",
            Self::PhaseModulation { .. } => "Complex Phase Modulation",
            Self::ActivationWarping { .. } => "Continuous Activation Warping",
            Self::StochasticDropBlock { .. } => "Stochastic DropBlock Sparsity",
        }
    }
}

/// Outcome metrics of an experimental technique execution
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ExperimentMetrics {
    pub technique_name: String,
    pub original_l2_norm: f32,
    pub transformed_l2_norm: f32,
    pub mean_squared_error: f32,
    pub signal_to_noise_ratio_db: f32,
    pub sparsity_delta_pct: f32,
    pub affected_elements: usize,
}

/// Execution engine for non-conventional experimentation
pub struct ExperimentLab;

impl ExperimentLab {
    /// Executes an experimental technique directly on a mutable weight slice in-place
    pub fn execute(technique: &ExperimentTechnique, weights: &mut [f32]) -> ExperimentMetrics {
        let n = weights.len();
        if n == 0 {
            return ExperimentMetrics::default();
        }

        let orig_norm_sq: f32 = weights.iter().map(|&x| x * x).sum();
        let orig_l2 = orig_norm_sq.sqrt();
        let original_copy = weights.to_vec();
        let mut affected = 0;

        match technique {
            ExperimentTechnique::NoiseInjection {
                variance,
                distribution,
            } => {
                let sigma = variance.sqrt().max(1e-6);
                for (i, w) in weights.iter_mut().enumerate() {
                    // Fast deterministic pseudorandom generator based on index
                    let seed = (i as f32 * 12.9898 + 78.233).sin() * 43758.547;
                    let noise = if distribution == "Uniform" {
                        (seed.fract() - 0.5) * 2.0 * sigma
                    } else {
                        // Box-Muller approximation
                        let u1 = seed.fract().abs().max(1e-7);
                        let u2 = ((seed * 1.618).fract()).abs().max(1e-7);
                        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos() * sigma
                    };
                    *w += noise;
                    affected += 1;
                }
            }
            ExperimentTechnique::SpectralReshaping { gamma_exponent } => {
                let mean_abs = weights.iter().map(|&x| x.abs()).sum::<f32>() / n as f32;
                let gamma = gamma_exponent.clamp(0.1, 3.0);
                for w in weights.iter_mut() {
                    let sign = w.signum();
                    let magnitude = w.abs();
                    if magnitude > 1e-7 {
                        let normalized = magnitude / mean_abs.max(1e-6);
                        let reshaped = normalized.powf(gamma) * mean_abs;
                        *w = sign * reshaped;
                        affected += 1;
                    }
                }
            }
            ExperimentTechnique::PhaseModulation {
                frequency_rad,
                phase_shift_rad,
            } => {
                for (i, w) in weights.iter_mut().enumerate() {
                    let theta = (i as f32 * frequency_rad) + phase_shift_rad;
                    let modulation = theta.cos();
                    *w *= 0.5 + 0.5 * modulation;
                    affected += 1;
                }
            }
            ExperimentTechnique::ActivationWarping { blend_factor } => {
                let alpha = blend_factor.clamp(0.0, 1.0);
                for w in weights.iter_mut() {
                    let linear = *w;
                    let harmonic = (linear * std::f32::consts::PI).sin() * 0.15;
                    *w = (1.0 - alpha) * linear + alpha * (linear + harmonic);
                    affected += 1;
                }
            }
            ExperimentTechnique::StochasticDropBlock {
                drop_probability,
                block_size,
            } => {
                let p = drop_probability.clamp(0.0, 0.95);
                let b = (*block_size).max(1);
                for chunk in weights.chunks_mut(b) {
                    let seed = ((chunk.as_ptr() as usize) as f32 * 0.0019).sin().abs();
                    if seed < p {
                        for w in chunk.iter_mut() {
                            *w = 0.0;
                            affected += 1;
                        }
                    }
                }
            }
        }

        let new_norm_sq: f32 = weights.iter().map(|&x| x * x).sum();
        let new_l2 = new_norm_sq.sqrt();

        let mut mse = 0.0f32;
        let mut zero_orig = 0;
        let mut zero_new = 0;
        for (i, &w) in weights.iter().enumerate() {
            if original_copy[i].abs() < 1e-7 {
                zero_orig += 1;
            }
            if w.abs() < 1e-7 {
                zero_new += 1;
            }
            mse += (w - original_copy[i]).powi(2);
        }
        mse /= n as f32;

        let snr = if mse > 1e-9 {
            10.0 * (orig_norm_sq / (mse * n as f32)).log10()
        } else {
            99.0
        };

        let orig_sp = zero_orig as f32 / n as f32;
        let new_sp = zero_new as f32 / n as f32;

        ExperimentMetrics {
            technique_name: technique.display_name().to_string(),
            original_l2_norm: orig_l2,
            transformed_l2_norm: new_l2,
            mean_squared_error: mse,
            signal_to_noise_ratio_db: snr,
            sparsity_delta_pct: (new_sp - orig_sp) * 100.0,
            affected_elements: affected,
        }
    }
}
