//! Bring Your Own Algorithm (BYOA) Engine for THME
//!
//! Allows researchers, developers, and automated Teacher LLMs to:
//! 1. Design, import, export, and register custom quantization, transformation, and pruning algorithms.
//! 2. Apply custom mathematical transformation rules to model weight matrices and tensor slices.
//! 3. Integrate custom algorithms directly into the Pareto AutoDecider and Surgery engine.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::engine::basis::{walsh_hadamard, walsh_hadamard_blocks};
use crate::engine::numfmt;
use crate::ir::TensorRole;

/// Category of the custom algorithm
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AlgorithmCategory {
    Quantization,
    LinearTransformation,
    SparsityPruning,
    SequenceMixer,
    FeatureDistillation,
}

impl AlgorithmCategory {
    pub fn display_name(&self) -> &'static str {
        match self {
            Self::Quantization => "Quantization",
            Self::LinearTransformation => "Linear Transformation",
            Self::SparsityPruning => "Sparsity & Pruning",
            Self::SequenceMixer => "Sequence Mixer",
            Self::FeatureDistillation => "Feature Distillation",
        }
    }
}

/// Scale normalization mode for quantization
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ScaleMode {
    AbsMax,
    AbsMean,
    RMS,
    StandardDev,
}

/// Parametric mathematical transformation rule executed by a custom algorithm
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", content = "params")]
pub enum TransformationRule {
    /// Discretize into a custom discrete codebook of levels
    QuantizeCodebook {
        levels: Vec<f32>,
        scale_mode: ScaleMode,
    },
    /// Ternary discretization with custom gamma scale factor and zero-band
    AbsmeanTernary { scale_gamma: f32, zero_band: f32 },
    /// Orthogonal basis rotation (e.g. Walsh-Hadamard, Random Orthogonal, Givens)
    OrthogonalRotation { block_size: usize, basis: String },
    /// M:N or structured sparsity (e.g. 2:4 sparsity, block sparsity)
    StructuredSparsity { m: usize, n: usize },
    /// SmoothQuant-style outlier migration balancing weights and activations
    SmoothQuantOutlier { alpha: f32, threshold: f32 },
    /// Low-rank affine decomposition / projection
    AffineProject { target_dim: usize, rank: usize },
    /// Rotate blocks of each weight row with a normalized Walsh-Hadamard transform, apply
    /// `inner`, then rotate back. Equivalent to running `inner` on rotated weights with the
    /// matching rotation applied to activations (QuaRot/QuIP#-style incoherence processing).
    RotatedQuantize {
        block_size: usize,
        inner: Box<TransformationRule>,
    },
    /// AWQ-style activation-aware quantization: scale each input channel by
    /// `mean|x|^alpha`, store in `format` (e.g. `Q4_0`, `Q4_K`), undo the scale, and keep the
    /// alpha from `alpha_grid` with the lowest activation-weighted error. Needs calibration
    /// statistics attached to the tensor (`thme evaluate --calibration`).
    ActivationAware {
        format: String,
        alpha_grid: Vec<f32>,
    },
}

/// Rounds a slice in place to the named block format (e.g. `Q4_0`).
pub type FormatRounder<'a> = dyn Fn(&mut [f32], &str) -> Result<(), String> + 'a;

/// What a rule may use besides the weights. Built once per tensor by the caller; each rule
/// takes what it needs and reports what is missing.
#[derive(Clone, Copy, Default)]
pub struct ApplyContext<'a> {
    /// Width of one weight row (the input dimension of a `[out][in]` linear tensor).
    pub row_width: Option<usize>,
    /// Per-input-channel activation statistics from `thme calibrate`.
    pub calibration: Option<&'a crate::transform::awq::ChannelStats>,
    /// Replaces a slice by what it becomes after storage in the named block format
    /// (e.g. `Q4_0`); supplied by the inference path, which owns the codecs.
    pub round: Option<&'a FormatRounder<'a>>,
}

/// What a rule means for a running model.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WeightContract {
    /// Each weight is replaced by the value the model will use: measurable as written.
    Reconstruct,
    /// Orthogonal change of basis: only valid when compensated (see `RotatedQuantize`).
    Rotation,
    /// Not a weight transform the forward pass can evaluate; the reason is given.
    NotWeightOnly(&'static str),
}

impl TransformationRule {
    /// True for rules that read calibration activations and cannot run on weights alone.
    pub fn needs_calibration(&self) -> bool {
        matches!(self, Self::ActivationAware { .. })
    }

    pub fn contract(&self) -> WeightContract {
        match self {
            Self::AbsmeanTernary { .. }
            | Self::QuantizeCodebook { .. }
            | Self::StructuredSparsity { .. }
            | Self::SmoothQuantOutlier { .. }
            | Self::ActivationAware { .. } => WeightContract::Reconstruct,
            Self::OrthogonalRotation { .. } => WeightContract::Rotation,
            Self::AffineProject { .. } => WeightContract::NotWeightOnly(
                "AffineProject attenuates weights by position; it is not a low-rank projection",
            ),
            Self::RotatedQuantize { inner, .. } => match inner.contract() {
                WeightContract::Reconstruct => WeightContract::Reconstruct,
                _ => WeightContract::NotWeightOnly(
                    "RotatedQuantize needs a reconstructing inner rule",
                ),
            },
        }
    }
}

/// Specification of a custom algorithm (BYOA)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AlgorithmDefinition {
    /// Unique identifier (e.g., "epiphanes.bitnet_b158", "user.my_quant_4bit")
    pub id: String,
    /// Human-readable display name
    pub name: String,
    /// Author / organization
    pub author: String,
    /// Semantic version
    pub version: String,
    /// Category
    pub category: AlgorithmCategory,
    /// Description
    pub description: String,
    /// Effective bits stored per parameter (e.g. 1.58, 2.0, 4.0, 8.0)
    pub effective_bits_per_param: f32,
    /// Projected compression ratio relative to FP16
    pub compression_ratio: f32,
    /// Projected hardware speedup multiplier (e.g. 1.5x, 2.2x)
    pub hardware_speedup_factor: f32,
    /// Target tensor roles where this algorithm is applicable
    pub target_roles: Vec<TensorRole>,
    /// Underlying transformation rule
    pub rule: TransformationRule,
}

/// Transformation output statistics
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TransformResult {
    pub original_elements: usize,
    pub original_l2_norm: f32,
    pub transformed_l2_norm: f32,
    pub mean_squared_error: f32,
    pub signal_to_noise_ratio_db: f32,
    pub sparsity_ratio: f32,
}

impl AlgorithmDefinition {
    /// Deserializes an algorithm definition from a JSON string
    pub fn from_json(json_str: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json_str)
    }

    /// Whether the reference forward pass can measure this algorithm, with the reason if not.
    pub fn measurable(&self) -> Result<(), String> {
        if self.category == AlgorithmCategory::SequenceMixer {
            return Err(format!(
                "'{}' replaces a sequence mixer; that needs an architecture change, not a weight transform",
                self.id
            ));
        }
        match self.rule.contract() {
            WeightContract::Reconstruct => Ok(()),
            WeightContract::Rotation => Err(format!(
                "'{}' rotates weights without rotating activations, which changes what the model computes; wrap a quantizer in RotatedQuantize instead",
                self.id
            )),
            WeightContract::NotWeightOnly(reason) => Err(format!("'{}': {reason}", self.id)),
        }
    }

    /// Checks the declared contract on sample weights: length preserved, outputs finite,
    /// deterministic. Returns the transform statistics.
    pub fn verify_contract(&self, sample: &[f32]) -> Result<TransformResult, String> {
        self.measurable()?;
        let mut first = sample.to_vec();
        let mut second = sample.to_vec();
        let stats = self.apply_to_weights(&mut first)?;
        self.apply_to_weights(&mut second)?;
        if first.len() != sample.len() {
            return Err(format!("'{}' changed the number of weights", self.id));
        }
        if first.iter().any(|v| !v.is_finite()) {
            return Err(format!("'{}' produced non-finite weights", self.id));
        }
        if first != second {
            return Err(format!("'{}' is not deterministic", self.id));
        }
        Ok(stats)
    }

    /// Serializes this algorithm definition to a formatted JSON string
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Applies this algorithm's transformation rule to a float weight slice in-place, with
    /// no context (rules that need calibration or a codec report it as an error).
    pub fn apply_to_weights(&self, weights: &mut [f32]) -> Result<TransformResult, String> {
        self.apply_with(weights, &ApplyContext::default())
    }

    /// Applies this algorithm's rule in place with whatever context the caller has.
    pub fn apply_with(
        &self,
        weights: &mut [f32],
        ctx: &ApplyContext<'_>,
    ) -> Result<TransformResult, String> {
        let n = weights.len();
        if n == 0 {
            return Err("Cannot apply algorithm to empty weight slice".to_string());
        }

        let orig_norm_sq: f32 = weights.iter().map(|&x| x * x).sum();
        let orig_l2 = orig_norm_sq.sqrt();
        let original_copy = weights.to_vec();

        apply_rule(&self.rule, weights, ctx)?;

        // Compute metrics
        let trans_norm_sq: f32 = weights.iter().map(|&x| x * x).sum();
        let trans_l2 = trans_norm_sq.sqrt();

        let mse: f32 = original_copy
            .iter()
            .zip(weights.iter())
            .map(|(&orig, &trans)| (orig - trans).powi(2))
            .sum::<f32>()
            / n as f32;

        let snr_db = if mse > 1e-12 {
            10.0 * ((orig_norm_sq / (mse * n as f32)).max(1e-6)).log10()
        } else {
            100.0
        };

        let zero_count = weights.iter().filter(|&&x| x == 0.0).count();
        let sparsity_ratio = zero_count as f32 / n as f32;

        Ok(TransformResult {
            original_elements: n,
            original_l2_norm: orig_l2,
            transformed_l2_norm: trans_l2,
            mean_squared_error: mse,
            signal_to_noise_ratio_db: snr_db,
            sparsity_ratio,
        })
    }
}

/// Applies one transformation rule in place (shared by plain and composed rules).
fn apply_rule(
    rule: &TransformationRule,
    weights: &mut [f32],
    ctx: &ApplyContext<'_>,
) -> Result<(), String> {
    let n = weights.len();
    let orig_norm_sq: f32 = weights.iter().map(|&x| x * x).sum();
    match rule {
        TransformationRule::ActivationAware { format, alpha_grid } => {
            let stats = ctx.calibration.ok_or(
                "activation-aware rules need calibration statistics; run `thme calibrate` and pass `--calibration` to `thme evaluate`",
            )?;
            let round = ctx.round.ok_or(
                "activation-aware rules need the block-format codec of the inference path",
            )?;
            let in_dim = ctx.row_width.unwrap_or(n);
            let round_in_format = |v: &mut [f32]| round(v, format).map_err(|e| anyhow::anyhow!(e));
            crate::transform::awq::activation_aware_quantize(
                weights,
                in_dim,
                stats,
                &round_in_format,
                alpha_grid,
            )
            .map_err(|e| format!("{e:#}"))?;
        }
        TransformationRule::AbsmeanTernary {
            scale_gamma,
            zero_band,
        } => {
            let gamma = if *scale_gamma > 0.0 {
                *scale_gamma
            } else {
                numfmt::absmean(weights).max(1e-6)
            };
            // The zero band is in scaled units (|w / gamma|), as in BitNet b1.58.
            let band = zero_band.max(0.1);
            for w in weights.iter_mut() {
                *w = f32::from(numfmt::ternary_code(*w / gamma, band)) * gamma;
            }
        }
        TransformationRule::QuantizeCodebook { levels, scale_mode } => {
            if levels.is_empty() {
                return Err("QuantizeCodebook requires at least one level".to_string());
            }

            // Compute scale factor based on mode
            let scale = match scale_mode {
                ScaleMode::AbsMax => weights.iter().map(|&x| x.abs()).fold(0.0f32, f32::max),
                ScaleMode::AbsMean => weights.iter().map(|&x| x.abs()).sum::<f32>() / n as f32,
                ScaleMode::RMS => (orig_norm_sq / n as f32).sqrt(),
                ScaleMode::StandardDev => {
                    let mean = weights.iter().sum::<f32>() / n as f32;
                    let var = weights.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / n as f32;
                    var.sqrt()
                }
            }
            .max(1e-8);

            // Quantize to nearest codebook level
            for w in weights.iter_mut() {
                let norm_val = *w / scale;
                let mut best_level = levels[0];
                let mut min_diff = (norm_val - levels[0]).abs();

                for &lvl in &levels[1..] {
                    let diff = (norm_val - lvl).abs();
                    if diff < min_diff {
                        min_diff = diff;
                        best_level = lvl;
                    }
                }
                *w = best_level * scale;
            }
        }
        TransformationRule::StructuredSparsity { m, n: group_sz } => {
            let m = *m;
            let group_sz = *group_sz;
            if m == 0 || group_sz == 0 || m >= group_sz {
                return Err(
                    "Invalid structured sparsity parameters: requires 0 < M < N".to_string()
                );
            }

            for chunk in weights.chunks_mut(group_sz) {
                if chunk.len() == group_sz {
                    // Find the M smallest absolute elements in the group and zero them out
                    let mut indexed: Vec<(usize, f32)> = chunk
                        .iter()
                        .enumerate()
                        .map(|(i, &x)| (i, x.abs()))
                        .collect();
                    indexed
                        .sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

                    for (idx_to_zero, _) in indexed.iter().take(m) {
                        chunk[*idx_to_zero] = 0.0;
                    }
                }
            }
        }
        TransformationRule::SmoothQuantOutlier { alpha, threshold } => {
            let max_abs = weights
                .iter()
                .map(|&x| x.abs())
                .fold(0.0f32, f32::max)
                .max(1e-8);
            let scale = max_abs.powf(*alpha);

            for w in weights.iter_mut() {
                let s = *w / scale;
                if s.abs() > *threshold {
                    *w = s.clamp(-*threshold, *threshold) * scale;
                }
            }
        }
        TransformationRule::RotatedQuantize { block_size, inner } => {
            let b = *block_size;
            if b < 2 || !b.is_power_of_two() || !n.is_multiple_of(b) {
                return Err(format!(
                    "RotatedQuantize needs a power-of-two block size dividing {n}, got {b}"
                ));
            }
            walsh_hadamard_blocks(weights, b).map_err(|e| e.to_string())?;
            // Rotation mixes input channels, so per-channel calibration no longer applies.
            let rotated = ApplyContext {
                calibration: None,
                ..*ctx
            };
            apply_rule(inner, weights, &rotated)?;
            // The normalized Hadamard matrix is its own inverse.
            walsh_hadamard_blocks(weights, b).map_err(|e| e.to_string())?;
        }
        TransformationRule::OrthogonalRotation { block_size, .. } => {
            let raw_b = (*block_size).max(2);
            let b = if raw_b.is_power_of_two() {
                raw_b
            } else {
                1usize << (usize::BITS - 1 - raw_b.leading_zeros()).max(1)
            };
            for chunk in weights.chunks_mut(b) {
                // A short tail block is rotated only when its length is a power of two.
                if chunk.len() >= 2 && chunk.len().is_power_of_two() {
                    walsh_hadamard(chunk).map_err(|e| e.to_string())?;
                }
            }
        }
        TransformationRule::AffineProject { target_dim, rank } => {
            let r = (*rank).min(n).max(1);
            let _ = target_dim;
            // Truncated projection dampening
            for (i, w) in weights.iter_mut().enumerate() {
                if i >= r {
                    *w *= 0.1; // Energy attenuation on trailing spectral modes
                }
            }
        }
    }
    Ok(())
}

/// Pluggable registry containing built-in and user-defined Bring Your Own Algorithm (BYOA) instances
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AlgorithmRegistry {
    algorithms: HashMap<String, AlgorithmDefinition>,
}

impl Default for AlgorithmRegistry {
    fn default() -> Self {
        let mut registry = Self {
            algorithms: HashMap::new(),
        };
        registry.register_default_presets();
        registry
    }
}

impl AlgorithmRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Populate standard built-in algorithm presets
    fn register_default_presets(&mut self) {
        // 1. BitNet b1.58 Absmean Ternary
        self.register(AlgorithmDefinition {
            id: "thme.bitnet_b158".to_string(),
            name: "BitNet b1.58 Absmean Ternary".to_string(),
            author: "Epiphanes / Microsoft Research".to_string(),
            version: "1.0.0".to_string(),
            category: AlgorithmCategory::Quantization,
            description:
                "Discretizes weights to {-1, 0, 1} scaled by mean absolute magnitude gamma."
                    .to_string(),
            effective_bits_per_param: 1.585,
            compression_ratio: 10.1,
            hardware_speedup_factor: 2.8,
            target_roles: vec![
                TensorRole::MlpGate,
                TensorRole::MlpUp,
                TensorRole::MlpDown,
                TensorRole::AttentionQ,
                TensorRole::AttentionK,
                TensorRole::AttentionV,
                TensorRole::AttentionOut,
            ],
            rule: TransformationRule::AbsmeanTernary {
                scale_gamma: 0.0, // Computed dynamically from weight distribution
                zero_band: 0.5,
            },
        });

        // 1b. Hadamard-rotated variants (rotate, quantize, rotate back). Speedup is not
        // claimed: no kernel for these formats has been measured.
        let linear_roles = vec![
            TensorRole::MlpGate,
            TensorRole::MlpUp,
            TensorRole::MlpDown,
            TensorRole::AttentionQ,
            TensorRole::AttentionK,
            TensorRole::AttentionV,
            TensorRole::AttentionOut,
            TensorRole::AttentionQkv,
        ];
        self.register(AlgorithmDefinition {
            id: "thme.hadamard_ternary".to_string(),
            name: "Hadamard-rotated absmean ternary".to_string(),
            author: "Epiphanes".to_string(),
            version: "1.0.0".to_string(),
            category: AlgorithmCategory::Quantization,
            description: "Rotates 64-weight blocks with a normalized Walsh-Hadamard transform, ternarizes, and rotates back, spreading outliers before rounding.".to_string(),
            effective_bits_per_param: 1.585,
            compression_ratio: 16.0 / 1.585,
            hardware_speedup_factor: 1.0,
            target_roles: linear_roles.clone(),
            rule: TransformationRule::RotatedQuantize {
                block_size: 64,
                inner: Box::new(TransformationRule::AbsmeanTernary {
                    scale_gamma: 0.0,
                    zero_band: 0.5,
                }),
            },
        });
        self.register(AlgorithmDefinition {
            id: "thme.hadamard_int4".to_string(),
            name: "Hadamard-rotated 4-bit codebook".to_string(),
            author: "Epiphanes".to_string(),
            version: "1.0.0".to_string(),
            category: AlgorithmCategory::Quantization,
            description: "Rotates 64-weight blocks, rounds to 15 symmetric levels scaled by the tensor's absolute maximum, and rotates back.".to_string(),
            effective_bits_per_param: 4.0,
            compression_ratio: 4.0,
            hardware_speedup_factor: 1.0,
            target_roles: linear_roles,
            rule: TransformationRule::RotatedQuantize {
                block_size: 64,
                inner: Box::new(TransformationRule::QuantizeCodebook {
                    levels: (-7..=7).map(|k| k as f32 / 7.0).collect(),
                    scale_mode: ScaleMode::AbsMax,
                }),
            },
        });
        for (format, bits) in [("Q4_0", 4.5f32), ("Q4_K", 4.5)] {
            self.register(AlgorithmDefinition {
                id: format!("thme.awq_{}", format.to_ascii_lowercase()),
                name: format!("Activation-aware {format} (AWQ-style)"),
                author: "Epiphanes".to_string(),
                version: "1.0.0".to_string(),
                category: AlgorithmCategory::Quantization,
                description: format!("Scales input channels by calibration mean |x|^alpha before {format} rounding and undoes the scale, choosing alpha by activation-weighted error. Needs `thme evaluate --calibration`."),
                effective_bits_per_param: bits,
                compression_ratio: 16.0 / bits,
                hardware_speedup_factor: 1.0,
                target_roles: vec![
                    TensorRole::AttentionQ,
                    TensorRole::AttentionK,
                    TensorRole::AttentionV,
                    TensorRole::AttentionQkv,
                    TensorRole::AttentionOut,
                    TensorRole::MlpGate,
                    TensorRole::MlpUp,
                    TensorRole::MlpDown,
                ],
                rule: TransformationRule::ActivationAware {
                    format: format.to_string(),
                    alpha_grid: (0..=10).map(|k| k as f32 / 10.0).collect(),
                },
            });
        }
        // 2. Semi-Structured 2:4 Sparsity
        self.register(AlgorithmDefinition {
            id: "thme.sparse_2to4".to_string(),
            name: "Semi-Structured 2:4 Sparsity".to_string(),
            author: "NVIDIA TensorRT / Epiphanes".to_string(),
            version: "1.0.0".to_string(),
            category: AlgorithmCategory::SparsityPruning,
            description: "Enforces 2 zero weights per 4-element window for NVIDIA Ampere/Hopper sparse tensor acceleration."
                .to_string(),
            effective_bits_per_param: 8.0,
            compression_ratio: 2.0,
            hardware_speedup_factor: 1.8,
            target_roles: vec![
                TensorRole::MlpGate,
                TensorRole::MlpUp,
                TensorRole::MlpDown,
                TensorRole::AttentionOut,
            ],
            rule: TransformationRule::StructuredSparsity { m: 2, n: 4 },
        });

        // 3. Walsh-Hadamard Outlier Suppression Rotation
        self.register(AlgorithmDefinition {
            id: "thme.hadamard_wht".to_string(),
            name: "Walsh-Hadamard Orthogonal Rotation".to_string(),
            author: "QuaRot / QuIP# / Epiphanes".to_string(),
            version: "1.0.0".to_string(),
            category: AlgorithmCategory::LinearTransformation,
            description: "Orthonormal basis rotation spreading kurtosis energy across channels to suppress activation outliers."
                .to_string(),
            effective_bits_per_param: 8.0,
            compression_ratio: 2.0,
            hardware_speedup_factor: 1.4,
            target_roles: vec![
                TensorRole::AttentionQ,
                TensorRole::AttentionK,
                TensorRole::AttentionV,
            ],
            rule: TransformationRule::OrthogonalRotation {
                block_size: 64,
                basis: "WalshHadamard".to_string(),
            },
        });

        // 4. SmoothQuant Per-Channel Scaling
        self.register(AlgorithmDefinition {
            id: "thme.smoothquant".to_string(),
            name: "SmoothQuant Alpha Balancing".to_string(),
            author: "MIT Song Han Lab / Epiphanes".to_string(),
            version: "1.0.0".to_string(),
            category: AlgorithmCategory::Quantization,
            description: "Balances quantization difficulty between activations and weights using channel scaling factor alpha=0.5."
                .to_string(),
            effective_bits_per_param: 8.0,
            compression_ratio: 2.0,
            hardware_speedup_factor: 1.6,
            target_roles: vec![
                TensorRole::AttentionQ,
                TensorRole::AttentionK,
                TensorRole::AttentionV,
                TensorRole::MlpGate,
            ],
            rule: TransformationRule::SmoothQuantOutlier {
                alpha: 0.5,
                threshold: 3.0,
            },
        });

        // 5. Gated DeltaNet Sequence Mixer
        self.register(AlgorithmDefinition {
            id: "thme.gated_deltanet".to_string(),
            name: "Gated DeltaNet Sequence Mixer".to_string(),
            author: "Yang, Kautz & Hatamizadeh (ICLR 2025)".to_string(),
            version: "1.0.0".to_string(),
            category: AlgorithmCategory::SequenceMixer,
            description: "Substitutes standard softmax attention with selective delta-rule associative recurrence."
                .to_string(),
            effective_bits_per_param: 2.0,
            compression_ratio: 8.0,
            hardware_speedup_factor: 3.2,
            target_roles: vec![
                TensorRole::AttentionQ,
                TensorRole::AttentionK,
                TensorRole::AttentionV,
                TensorRole::AttentionOut,
            ],
            rule: TransformationRule::AbsmeanTernary {
                scale_gamma: 0.0,
                zero_band: 0.4,
            },
        });
    }

    /// Register a new or imported algorithm
    pub fn register(&mut self, alg: AlgorithmDefinition) {
        self.algorithms.insert(alg.id.clone(), alg);
    }

    /// Retrieve an algorithm by its unique ID
    pub fn get(&self, id: &str) -> Option<&AlgorithmDefinition> {
        self.algorithms.get(id)
    }

    /// Remove an algorithm by its unique ID
    pub fn remove(&mut self, id: &str) -> Option<AlgorithmDefinition> {
        self.algorithms.remove(id)
    }

    /// List all registered algorithms
    pub fn list(&self) -> Vec<&AlgorithmDefinition> {
        let mut list: Vec<&AlgorithmDefinition> = self.algorithms.values().collect();
        list.sort_by(|a, b| a.id.cmp(&b.id));
        list
    }

    /// Import a custom algorithm definition from a JSON string
    pub fn import_from_json(&mut self, json_str: &str) -> Result<String, String> {
        let def: AlgorithmDefinition =
            serde_json::from_str(json_str).map_err(|e| format!("JSON parsing failed: {e}"))?;
        let id = def.id.clone();
        self.register(def);
        Ok(id)
    }

    /// Export an algorithm definition to formatted JSON string
    pub fn export_to_json(&self, id: &str) -> Result<String, String> {
        let alg = self
            .get(id)
            .ok_or_else(|| format!("Algorithm with id '{id}' not found in registry"))?;
        serde_json::to_string_pretty(alg).map_err(|e| format!("JSON serialization failed: {e}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn outlier_weights() -> Vec<f32> {
        // Small weights with a few large outliers, as in real LLM projections.
        (0..256)
            .map(|i| {
                if i % 61 == 0 {
                    4.0
                } else {
                    ((i * 37 % 101) as f32 - 50.0) / 500.0
                }
            })
            .collect()
    }

    #[test]
    fn hadamard_is_its_own_inverse() {
        let original = outlier_weights();
        let mut w = original.clone();
        walsh_hadamard_blocks(&mut w, 64).unwrap();
        assert_ne!(w, original);
        walsh_hadamard_blocks(&mut w, 64).unwrap();
        for (a, b) in w.iter().zip(&original) {
            assert!((a - b).abs() < 1e-5);
        }
    }

    #[test]
    fn contracts_classify_every_built_in() {
        let registry = AlgorithmRegistry::new();
        let get = |id: &str| registry.get(id).unwrap();
        assert!(get("thme.bitnet_b158").measurable().is_ok());
        assert!(get("thme.hadamard_ternary").measurable().is_ok());
        assert!(get("thme.hadamard_int4").measurable().is_ok());
        assert!(get("thme.hadamard_wht")
            .measurable()
            .unwrap_err()
            .contains("activations"));
        assert!(get("thme.gated_deltanet").measurable().is_err());
        let sample = outlier_weights();
        for alg in registry.list() {
            if alg.measurable().is_ok() && !alg.rule.needs_calibration() {
                alg.verify_contract(&sample).unwrap();
            }
        }
    }

    /// THME-3010: a data-needing rule gets everything through `ApplyContext`, testable with a
    /// toy codec and no inference code; missing pieces are reported, not defaulted.
    #[test]
    fn activation_aware_rule_runs_through_the_apply_context() {
        use crate::transform::awq::ChannelStats;
        let rule = AlgorithmDefinition {
            rule: TransformationRule::ActivationAware {
                format: "grid".into(),
                alpha_grid: vec![0.0, 0.5, 1.0],
            },
            ..AlgorithmRegistry::new()
                .get("thme.awq_q4_0")
                .unwrap()
                .clone()
        };
        // Toy codec: round to a 0.25 grid; it records the format name it was asked for.
        let asked = std::cell::RefCell::new(String::new());
        let round = |v: &mut [f32], name: &str| -> Result<(), String> {
            *asked.borrow_mut() = name.to_string();
            v.iter_mut().for_each(|x| *x = (*x * 4.0).round() / 4.0);
            Ok(())
        };
        let mut mean_abs = vec![0.05f32; 8];
        mean_abs[2] = 2.0;
        let stats = ChannelStats {
            rms: mean_abs.clone(),
            mean_abs,
        };
        let w: Vec<f32> = (0..32)
            .map(|i| ((i * 7 % 32) as f32 - 15.5) / 20.0)
            .collect();
        let ctx = ApplyContext {
            row_width: Some(8),
            calibration: Some(&stats),
            round: Some(&round),
        };
        let mut out = w.clone();
        rule.apply_with(&mut out, &ctx).unwrap();
        assert_eq!(*asked.borrow(), "grid");
        assert_ne!(out, w);

        let no_stats = ApplyContext {
            calibration: None,
            ..ctx
        };
        assert!(rule
            .apply_with(&mut w.clone(), &no_stats)
            .unwrap_err()
            .contains("--calibration"));
        let no_codec = ApplyContext { round: None, ..ctx };
        assert!(rule
            .apply_with(&mut w.clone(), &no_codec)
            .unwrap_err()
            .contains("codec"));
        assert!(rule.apply_to_weights(&mut w.clone()).is_err());
    }

    /// The zero band is in scaled units (w / gamma), like BitNet b1.58 round-and-clamp:
    /// with band 0.5 the rule must agree with `MatrixMath::discretize_bitnet_ternary`.
    #[test]
    fn absmean_ternary_zero_band_is_in_scaled_units() {
        let rule = TransformationRule::AbsmeanTernary {
            scale_gamma: 0.0,
            zero_band: 0.5,
        };
        let mut w = vec![1.0f32, -1.0, 0.2, -0.2]; // mean |w| = 0.6; scaled +-1.67, +-0.33
        apply_rule(&rule, &mut w, &ApplyContext::default()).unwrap();
        assert_eq!(w, vec![0.6, -0.6, 0.0, 0.0]);

        let weights: Vec<f32> = (0..64)
            .map(|i| ((i * 37 % 64) as f32 - 31.5) / 40.0)
            .collect();
        let (codes, gamma, _) = crate::transform::MatrixMath::discretize_bitnet_ternary(&weights);
        let mut via_rule = weights.clone();
        apply_rule(&rule, &mut via_rule, &ApplyContext::default()).unwrap();
        let expected: Vec<f32> = codes.iter().map(|&c| f32::from(c) * gamma).collect();
        assert_eq!(via_rule, expected);
    }

    #[test]
    fn rotation_reduces_codebook_error_on_outlier_weights() {
        let registry = AlgorithmRegistry::new();
        let rotated = registry.get("thme.hadamard_int4").unwrap();
        let mut plain = rotated.clone();
        if let TransformationRule::RotatedQuantize { inner, .. } = &rotated.rule {
            plain.rule = (**inner).clone();
        }
        let mut a = outlier_weights();
        let mut b = outlier_weights();
        let rotated_mse = rotated.apply_to_weights(&mut a).unwrap().mean_squared_error;
        let plain_mse = plain.apply_to_weights(&mut b).unwrap().mean_squared_error;
        assert!(
            rotated_mse < plain_mse,
            "rotated {rotated_mse} vs plain {plain_mse}"
        );
    }

    #[test]
    fn rotated_quantize_rejects_bad_block_sizes() {
        let rule = TransformationRule::RotatedQuantize {
            block_size: 48,
            inner: Box::new(TransformationRule::AbsmeanTernary {
                scale_gamma: 0.0,
                zero_band: 0.5,
            }),
        };
        let mut w = vec![0.1; 96];
        assert!(apply_rule(&rule, &mut w, &ApplyContext::default()).is_err());
    }
}
