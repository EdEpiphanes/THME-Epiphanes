// SPDX-License-Identifier: AGPL-3.0-only
use crate::ir::TensorFormat;
use crate::surgery::recipe::{TweakOperation, TweakRecipe};

/// Policy for relaxing aggressive quantization when quality gates fail.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RelaxationLevel {
    /// Step down from BitNet 1.58-bit to 4-bit (e.g. Q4_K or Q4_0).
    ToQ4,
    /// Step down from 4-bit to 8-bit (e.g. Q8_0 or INT8).
    ToQ8,
    /// Restore full precision (FP16/BF16/FP32).
    ToFullPrecision,
}

/// Heuristic relaxation engine that modifies recipe operations to recover perplexity.
pub struct RelaxationEngine;

impl RelaxationEngine {
    /// Relaxes the precision operations in a recipe when a quality gate breach is detected.
    pub fn relax_recipe(recipe: &TweakRecipe) -> (TweakRecipe, bool) {
        let mut relaxed = recipe.clone();
        let mut modified = false;

        for op in &mut relaxed.operations {
            if let TweakOperation::SetPrecision { format, .. } = op {
                match format {
                    TensorFormat::Ternary | TensorFormat::Bit1_5 => {
                        *format = TensorFormat::Q4_0;
                        modified = true;
                    }
                    TensorFormat::Q4_0 | TensorFormat::Q4_K => {
                        *format = TensorFormat::Q8_0;
                        modified = true;
                    }
                    TensorFormat::Q8_0 | TensorFormat::INT8 => {
                        *format = TensorFormat::FP16;
                        modified = true;
                    }
                    _ => {}
                }
            }
        }

        if modified {
            relaxed.description = format!("{} (Relaxed)", relaxed.description);
        }

        (relaxed, modified)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::TensorRole;

    #[test]
    fn test_relaxation_progression() {
        let recipe = TweakRecipe {
            name: "Test Recipe".to_string(),
            version: "1.0.0".to_string(),
            author: "Tester".to_string(),
            description: "Aggressive BitNet".to_string(),
            target_model: "llama".to_string(),
            expected_memory_savings_pct: Some(60.0),
            expected_speedup: Some(1.5),
            operations: vec![TweakOperation::SetPrecision {
                start_layer: 0,
                end_layer: 0,
                role: Some(TensorRole::MlpDown),
                format: TensorFormat::Ternary,
            }],
        };

        let (r1, mod1) = RelaxationEngine::relax_recipe(&recipe);
        assert!(mod1);
        if let TweakOperation::SetPrecision { format, .. } = &r1.operations[0] {
            assert_eq!(*format, TensorFormat::Q4_0);
        }

        let (r2, mod2) = RelaxationEngine::relax_recipe(&r1);
        assert!(mod2);
        if let TweakOperation::SetPrecision { format, .. } = &r2.operations[0] {
            assert_eq!(*format, TensorFormat::Q8_0);
        }
    }
}
