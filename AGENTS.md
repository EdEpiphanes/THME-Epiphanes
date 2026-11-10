# AGENTS.md

> Operational Guidelines, Non-Negotiable Standards, and Architecture Playbook for AI Coding Agents and Human Contributors.

---

## 1. Project Identity & Governance

- **Project Name**: THME (Topological Heterogeneous Manifold Engine)
- **Organization / Company**: Epiphanes
- **Repository**: `https://github.com/EdEpiphanes/THME`
- **Backwards-Compatible CLI Alias**: `llm-tweaker`
- **Core Mission**: Provide an open-source, mathematically principled framework for zero-copy LLM memory profiling, layer sensitivity estimation, Pareto precision optimization, and automated Teacher LLM surgery.

---

## 2. Non-Negotiable Rules

All agents and contributors must uphold these rules without exception:

1. **Clean Git Hygiene**:
   - Never commit or print absolute paths from developer machines or personal usernames in code, configurations, logs, or documentation.
   - Use relative workspace paths or forward-slash Unix-style paths.
   - Keep `.cargo/config.toml` path remapping active to avoid embedding developer machine paths in release binaries.

2. **No Unverified Empirical Claims**:
   - Never present simulated numbers or synthetic benchmarks as measured hardware results.
   - Any synthetic telemetry or simulated token flow must be explicitly labeled as simulated or benchmark projections.

3. **Zero Committed Binaries or Model Weights**:
   - Never commit `.safetensors`, `.gguf`, `.bin`, `.pt`, `.onnx`, `.thme`, `.exe`, `.dll`, `.so`, `.dylib`, or `.pdb` files.
   - Use mock generator scripts (`examples/generate_mock_model.rs`) for testing and CI validation.

4. **Infallible Parsers & Graceful Error Handling**:
   - Model parsers, header readers, and tensor mappers must never panic (`panic!`, `unwrap()`, or `expect()` on external I/O).
   - Return structured `Result<T, E>` types with actionable diagnostic contexts.

---

## 3. Code Standards & CI Requirements

Before submitting any Pull Request or committing changes:

```bash
# 1. Format verification
cargo fmt --check

# 2. Linter with zero warnings tolerated
cargo clippy -- -D warnings

# 3. Full test suite execution
cargo test
```

- **Toolchain**: Rust 2021 edition (version >= 1.80).
- **Binary Footprint Constraints**:
  - Release GUI binary (`thme-gui`): <= 15 MB.
  - Release CLI binary (`thme` / `llm-tweaker`): <= 10 MB.
- **Dependencies**: Pure Rust wherever feasible; avoid heavy C/C++ runtimes or external dynamic linking requirements.

---

## 4. Conventional Commits & Branch Naming

- **Branch Naming**:
  - `feature/<name>`: New capabilities or algorithms.
  - `fix/<name>`: Bug fixes and parser hardening.
  - `perf/<name>`: Performance improvements and memory reductions.
  - `docs/<name>`: Documentation, specs, and playbooks.
- **Commit Messages**:
  - `feat(...)`: New user-facing features.
  - `fix(...)`: Bug fixes.
  - `perf(...)`: Performance optimizations.
  - `docs(...)`: Documentation changes.
  - `chore(...)`: Maintenance, dependency updates, repo metadata.
  - `refactor(...)`: Code restructuring without behavioral changes.

---

## 5. Versioning & Production Roadmap (v0.1.0 to v0.2.0)

| Version | Focus Area | Deliverables |
| :--- | :--- | :--- |
| **v0.1.0** | Foundations | Zero-copy SafeTensors/GGUF parsing, CLI inspection, basic memory profiling. |
| **v0.1.1** | Sensitivity Profiling | SVD energy ratio, condition number, Fisher proxy estimation. |
| **v0.1.2** | Pareto Allocation | Dynamic programming precision planner, quality gate enforcement. |
| **v0.1.3** | Live Studio GUI | Native 60 FPS egui/eframe dashboard, telemetry flow, 5 color palettes. |
| **v0.1.4** | Distillation Engine | Truncated SVD, Walsh-Hadamard rotation, SmoothQuant, BitNet b1.58. |
| **v0.1.5** | AutoDecider | Parameter-level algorithm recommendation based on kurtosis and spectral decay. |
| **v0.1.6** | Token Telemetry | Live 13-stage Transformer forward simulator, batch scaling, KV projections. |
| **v0.1.7** | History & Audit Trail | Snapshot rollback checkpoints, cryptographically indexed audit events. |
| **v0.1.8** | Teacher LLM Surgery | Declarative JSON recipes for layer replacement, pruning, and bit-tweaking. |
| **v0.1.9** | Cross-Platform Release | Reproducible CI builds, stripped symbols, multi-platform binaries. |
| **v0.2.0** | Production Hardening | Hardware execution backends (Vulkan / Metal / CUDA kernels) & live distillation. |
