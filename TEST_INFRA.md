# E2E Test Infra: THME Roofline & Micro-Kernel Profiler

## Test Philosophy
- Opaque-box, requirement-driven. Derived from user specifications in `ORIGINAL_REQUEST.md`.
- Methodology: Category-Partition + Boundary Value Analysis + Pairwise Combinatorial + Real-World Workload Testing.

## Feature Inventory
| # | Feature | Source (Requirement) | Tier 1 | Tier 2 | Tier 3 |
|---|---------|----------------------|:------:|:------:|:------:|
| 1 | Hardware Target DB & Knee Calculator | R2 / `src/profiling/hardware.rs` | 5 | 5 | ✓ |
| 2 | Layer Arithmetic Intensity Calculator | R2 / `src/profiling/intensity.rs` | 5 | 5 | ✓ |
| 3 | 4-Tier Memory Hierarchy Cache Sim | R2 / `src/profiling/cache_sim.rs` | 5 | 5 | ✓ |
| 4 | Roofline Boundary & Layer Classifier | R2 / `src/profiling/roofline.rs` | 5 | 5 | ✓ |
| 5 | Dynamic TDP & Thermal Throttling | R2 / `src/profiling/thermal.rs` | 5 | 5 | ✓ |
| 6 | CLI Subcommand `thme roofline` | R2 / `src/cli/roofline.rs` | 5 | 5 | ✓ |
| 7 | MCP Tool `thme_profile_roofline` | R2 / `src/cli/mcp.rs` | 5 | 5 | ✓ |

## Test Architecture
- Test Runner: `cargo test --test test_roofline_profiler`
- Target Test File: `tests/test_roofline_profiler.rs`
- Coverage includes:
  1. `test_hardware_database_predefined_targets` (H100, RTX 4090, Apple M3, AMD MI300X, Intel Emerald Rapids)
  2. `test_layer_arithmetic_intensity_linear` (Batch scaling, weight reuse)
  3. `test_layer_arithmetic_intensity_attention_and_flash_attention` (Standard $O(S^2)$ vs FlashAttn-2 $O(S)$)
  4. `test_layer_arithmetic_intensity_swiglu` (SwiGLU FFN gating and projections)
  5. `test_cache_simulator_4_tier` (L1, L2, L3, DRAM hit rates and eviction)
  6. `test_roofline_curve_and_layer_classification` (MemoryBound vs ComputeBound against knee)
  7. `test_thermal_throttling_model` (Nominal, throttling, and critical temperatures)
  8. `test_end_to_end_roofline_model_profiler` (Synthetic 12-layer model full profile)
  9. `test_roofline_cli_json_serialization` (JSON serialization roundtrip)
  10. `test_mcp_roofline_tool_invocation` (Tool schema and execution)

## Real-World Application Scenarios (Tier 4)
| # | Scenario | Features Exercised | Complexity |
|---|----------|--------------------|------------|
| 1 | LLaMA-3.2-1B / SmolLM-135M decode vs prefill profiling | Linear, Attention, SwiGLU, Roofline | Medium |
| 2 | NVIDIA H100 vs RTX 4090 vs Apple M3 Max comparative profiling | Hardware DB, Roofline, Cache Sim | High |
| 3 | Long-context inference ($S=8192$) memory choke analysis | FlashAttention-2 vs Standard Attention | High |
| 4 | Thermal throttling under sustained full-TDP inference | Thermal model, DVFS downclocking | Medium |
| 5 | Automated JSON-RPC MCP Agent Model Inspection | MCP tool, JSON serialization, Recommendations | High |

## Pass/Fail Semantics
- Zero panics, zero unhandled errors.
- Clean exit code 0 on all test targets.
