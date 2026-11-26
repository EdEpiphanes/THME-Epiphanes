// SPDX-License-Identifier: AGPL-3.0-only
//! Automated source-to-heterogeneous regime routing (§7.6) and cross-architecture
//! symmetry-preserving model conversion (§7.10).

pub mod arch_graph;
pub mod regime_router;

pub use arch_graph::{
    mha_to_gqa_whitened_pool, permute_rope_vector, sinkhorn_balanced_router_init,
    verify_fp32_oracle_match, ConversionOp, ConversionPlan, ModelArchKind, RopeDialect,
};
pub use regime_router::{
    partition_row_blocks_1024, residual_variance_scale, solve_dynamic_regime_mix,
    solve_weighted_mse_tau_dead, BoundaryFloor, DeterministicSketchPrng, LayerRole, RegimeCodec,
    RegimeMixture, RowSlicePlan, StructuralFingerprint,
};
