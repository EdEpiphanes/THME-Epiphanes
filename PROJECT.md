# Project: THME Active Milestone v0.1.7.0 (Roofline & Micro-Kernel Profiler)

## Architecture
The THME (Topological Heterogeneous Manifold Engine) profiling subsystem provides hardware-calibrated arithmetic intensity calculations, multi-tier cache memory hierarchy simulation, empirical roofline ceiling boundaries, and dynamic thermal throttling models.

### Subsystem Boundaries
1. **Documentation Track**:
   - `README.md`: Mermaid flowchart with parallel sub-milestone blocks for active `v0.1.7.0` and planned `v0.1.8.0`, plus updated roadmap overview text.
   - `BACKLOG.md`: Complete tickets `[THME-601]` through `[THME-1309]` covering Sprints 6 through 13 with all 5 mandatory fields (`Component`, `Problem`, `Acceptance Criteria`, `Production Pipeline / CI/CD Impact`, `Done Means`).
2. **Profiling Core Subsystem** (`src/profiling/`):
   - `hardware.rs`: Predefined hardware profiles (H100 SXM5, RTX 4090, Apple M3 Max, AMD MI300X, Intel Emerald Rapids 8592+) and machine balance knee calculation.
   - `intensity.rs`: Layer-by-layer FLOP and byte transfer arithmetic intensity calculator for Linear, Attention, SwiGLU FFN, and FlashAttention-2.
   - `cache_sim.rs`: 4-tier memory hierarchy simulator (L1, L2, L3, DRAM) with hit/miss latency penalties and working set eviction.
   - `roofline.rs`: Empirical Roofline ceiling curve generator and layer classifier (`MemoryBound` vs `ComputeBound`).
   - `thermal.rs`: Dynamic TDP power and thermal throttling model.
   - `mod.rs`: Module exports and facade.
3. **Integration & Interface Layer**:
   - `src/cli/roofline.rs` & `src/cli/mod.rs`: `thme roofline` command with ASCII diagram and JSON output.
   - `src/cli/mcp.rs`: MCP tool `thme_profile_roofline`.
   - `src/gui/app.rs`: Studio Tab 4 interactive Roofline deck.
4. **Testing & Quality Assurance**:
   - `tests/test_roofline_profiler.rs`: 10 comprehensive unit and integration tests.
   - Zero clippy warnings, zero fmt violations, clean tests, zero path leaks.

---

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | README Mermaid Flowchart Sub-Blocks | Render parallel sub-milestone blocks for v0.1.7.0 and v0.1.8.0 | M1 | Survey |
| 2 | README Roadmap Overview Text | Updated text reflecting active v0.1.7 and planned v0.1.8-v0.2.4 | M1 | Survey |
| 3 | BACKLOG.md Sprints 6-13 | 72 complete tickets [THME-601]-[THME-1309] with 5 required fields | M1 | Survey |
| 4 | Hardware Database & Knee Calculator | Predefined hardware specs & machine balance knee point | M2 | Survey |
| 5 | Operational Arithmetic Intensity | Layer FLOP and byte calculations (Linear, Attention, SwiGLU, FlashAttn-2) | M2 | Survey |
| 6 | 4-Tier Cache Hierarchy Simulator | L1, L2, L3, DRAM hit/miss latencies and working set eviction | M2 | Survey |
| 7 | Roofline Ceiling & Layer Classifier | Multi-ceiling curve & MemoryBound vs ComputeBound classification | M2 | Survey |
| 8 | Dynamic TDP & Thermal Throttling | Lumped RC power dissipation, thermal rise, and DVFS throttling | M2 | Survey |
| 9 | CLI Subcommand `thme roofline` | Terminal command with ASCII diagram and JSON output | M3 | Survey |
| 10 | MCP Tool `thme_profile_roofline` | JSON-RPC MCP tool exposing roofline profiling | M3 | Survey |
| 11 | Desktop Studio Tab 4 Deck | Interactive egui roofline canvas and controls in Benchmarks tab | M3 | Survey |
| 12 | Opaque-Box E2E Test Suite | 10 unit and integration tests covering all requirements | E2E / M4 | Survey |
| 13 | Quality Gate Compliance | `cargo fmt --check`, `cargo clippy -- -D warnings`, `cargo test` | M4 | Survey |
| 14 | Path Hygiene Guard | Zero developer username or machine path leaks | M4 | Survey |
| 15 | Git Commit & Push | Conventional commit and push to `origin/main` | M4 | Survey |

---

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M1 | Repository Documentation | `README.md`, `BACKLOG.md` | none | DONE |
| M2 | Core Profiling Engine | `src/profiling/{hardware,intensity,cache_sim,roofline,thermal,mod}.rs` | none | DONE |
| M3 | CLI, MCP & GUI Tab 4 | `src/cli/roofline.rs`, `src/cli/mod.rs`, `src/cli/mcp.rs`, `src/gui/app.rs` | M2 | DONE |
| M4 | Final Milestone: Quality Gate & Push | Full E2E test verification, clippy, fmt, path check, push to `origin/main` | M1, M2, M3 | DONE |

---

## Interface Contracts

### Hardware Spec ↔ Profiling Modules
- `HardwareSpec::from_target(HardwareTarget) -> HardwareSpec`
- `HardwareSpec::machine_balance_knee(PrecisionType) -> f64` (in FLOPs/Byte)
- `HardwareSpec::peak_tflops_fp16: f64`, `HardwareSpec::memory_bandwidth_tbps: f64`, `HardwareSpec::tdp_watts: f64`

### Intensity Calculator ↔ Roofline Model
- `LayerArithmeticIntensity::calculate_linear(...) -> KernelIntensityReport`
- `LayerArithmeticIntensity::calculate_swiglu(...) -> KernelIntensityReport`
- `LayerArithmeticIntensity::calculate_attention(...) -> KernelIntensityReport`
- `LayerArithmeticIntensity::calculate_flash_attention_v2(...) -> KernelIntensityReport`

### Cache Simulator ↔ Working Set
- `CacheSimulator4Tier::simulate_working_set(working_set_bytes: usize) -> CacheSimulationSummary`

### Thermal Model ↔ Throttling
- `ThermalModel::calculate_throttling_factor(temp_c: f32, tdp_watts: f64) -> f32`

### CLI / MCP / GUI ↔ Profiler Engine
- `RooflineModel::profile_model(model: &ModelIR, hw: &HardwareSpec, batch_size: usize, seq_len: usize, flash_attn: bool, thermal_factor: f32) -> Result<RooflineProfileReport>`
- `RooflineProfileReport::format_ascii_report(&self, hw: &HardwareSpec, effective_peak: f64, knee: f64) -> String`
- `RooflineProfileReport` serializable to JSON.

---

## Code Layout
- `src/profiling/` owns all profiling math and hardware specifications.
- `src/cli/` owns CLI subcommands and MCP JSON-RPC tool handlers.
- `src/gui/` owns desktop studio application rendering.
- `tests/` owns integration test files.
- Exclusive ownership: Concurrent workers MUST NOT edit overlapping files.
