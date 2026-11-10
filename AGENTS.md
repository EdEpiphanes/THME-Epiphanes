# AGENTS.md

> Operational Guidelines, Non-Negotiable Standards, and Architecture Playbook for AI Coding Agents and Human Contributors.

---

## 1. Project Identity & Governance

- **Project Name**: THME (Topological Heterogeneous Manifold Engine)
- **Organization / Company**: Epiphanes
- **Repository**: `https://github.com/EdEpiphanes/THME-Epiphanes`
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

## 5. Versioning & Production Roadmap (v0.1 to v0.2)

| Version | Focus Area | Deliverables | Status |
| :--- | :--- | :--- | :--- |
| **v0.1.0** | Core Foundations | Zero-copy SafeTensors/GGUF parsing, Pareto precision allocator, live 60 FPS studio GUI, matrix distillation engine, token flow simulator, audit logging, and declarative surgery recipes. | Completed |
| **v0.1.1** | Path Hygiene & CI | Rebranded to THME under Epiphanes, 3-OS GitHub Actions CI matrix, automated path-leak guard, release automation, zero linter warnings, surgery JSON schema. | Completed |
| **v0.1.2** | Format Correctness | Full GGUF v2/v3 quant matrix (K-quants, I-quants), zero-allocation block decoders, truncated stream recovery, and infallible parser fuzz suite. | Completed |
| **v0.1.3** | Real Telemetry & SIMD | AVX2 / AVX-512 / ARM NEON vector micro-kernels for matrix transformations, and Criterion benchmarks integrated into CI. | In Progress |
| **v0.1.4** | Empirical Quality Gate | External perplexity evaluation bridge (WikiText-2, C4) with automated rollback gating (Delta PPL <= epsilon). | Planned |
| **v0.1.5** | Surgery Daemon & API | Headless JSON-RPC & REST server allowing remote Teacher LLMs to query model IR and dispatch surgery recipes programmatically. | Planned |
| **v0.2.0** | Production Release | Verified BitNet b1.58 and calibrated scale quantization exporter with end-to-end reproducible small-model demonstration. | Target |
