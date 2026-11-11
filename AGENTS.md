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

### 5.1 Sub-Milestone Execution Architecture
Every major release milestone $v0.1.x$ is governed by a 4-level sub-milestone decomposition:
$$v0.1.x.0 \longrightarrow \left[ v0.1.x.1, v0.1.x.2, v0.1.x.3, \dots, v0.1.x.9 \right] \longrightarrow v0.1.(x+1).0$$
- **Baseline Contract ($v0.1.x.0$)**: Architectural specification, trait contracts, and baseline benchmark definitions.
- **Parallel Sub-Functions ($v0.1.x.1 \dots v0.1.x.9$)**: Exactly 9 interdependent sub-milestones executed concurrently across modular subsystems, feeding continuously into CI/CD quality gates.
- **Convergence ($v0.1.(x+1).0$)**: End-to-end integration and verification gate preceding the subsequent milestone cycle.

### 5.2 Milestone Progress Ledger

| Version | Focus Area | Sub-Milestones [1-9] Summary | Status |
| :--- | :--- | :--- | :--- |
| **v0.1.0.0** | Core Foundations | [v0.1.0.1 - v0.1.0.9] SafeTensors/GGUF mmap, SVD decay, condition number, Fisher proxy, Pareto DP allocator, 60 FPS studio, token simulator, recipe engine. | Completed |
| **v0.1.1.0** | Path Hygiene & CI | [v0.1.1.1 - v0.1.1.9] Rebranding, AGENTS.md governance, 3-OS CI matrix, path-leak guard, tag release pipeline, zero warnings, recipe schema, alias, symbol stripping. | Completed |
| **v0.1.2.0** | Format Correctness | [v0.1.2.1 - v0.1.2.9] 28 GGML quant types, zero-alloc decoders, header recovery, crate modularization, 1k fuzz suite, SafeTensors fuzzing, crates.io prep, diagnostic errors. | Completed |
| **v0.1.3.0** | BYOA & REA-LLM Engine | [v0.1.3.1 - v0.1.3.9] Algorithm registry, JSON import/export, architecture rebuilder, open-weights remapping, EWHA reach profile, memory metric, energy model, layer anatomy, SIMD benches. | Completed |
| **v0.1.4.0** | Empirical Quality Gate | [v0.1.4.1 - v0.1.4.9] External PPL bridge, WikiText-2 harness, C4/PTB corpora, delta-PPL math, automated rollback, Pareto degradation, CLI validate, JSON audit, studio telemetry. | Completed |
| **v0.1.5.0** | Surgery Daemon & 8 Models | [v0.1.5.1 - v0.1.5.9] MCP JSON-RPC server, VS Code .vsix, Copilot LM tools, train_cuda.py, HF config generator, PEFT LoRA merger, 8 model paradigms, sparse upcycling, SLM BitNet. | Completed |
| **v0.1.6.0** | Speculative Decoding | [v0.1.6.1 - v0.1.6.9] Prefix manifold extractor, strided sampler, BitNet draft, rejection sampling, Leviathan speedup math, horizon search, tree topology, 2D attention mask, CLI speculate. | Completed |
| **v0.1.7.0** | Roofline & Kernel Profiler | [v0.1.7.1 - v0.1.7.9] Hardware DB, linear/attn intensity, SwiGLU math, 4-tier cache sim, eviction latency model, roofline boundary, thermal TDP, CLI roofline, Tab 4 deck. | Completed |
| **v0.1.8.0** | Quantization Exporter | [v0.1.8.1 - v0.1.8.9] BitNet packer, ternary unpacker, pure GGUF v3 writer, metadata sync, SafeTensors writer, streaming chunk I/O, MSE scale search, CLI export, downstream testbed. | Completed |
| **v0.1.9.0** | Closed-Loop Co-Surgery | [v0.1.9.1 - v0.1.9.9] Co-surgery state machine, prompt synthesizer, agent bridge, 3D Pareto optimizer, hypervolume indicator, self-healing relaxation, rollback watchdog, CLI co-surgery, Tab 2 dashboard. | Active |
| **v0.2.0.0** | Production Core Release | [v0.2.0.1 - v0.2.0.9] SmolLM-135M pipeline, LLaMA-3.2-1B pipeline, reproducible script, benchmark dossier, multi-platform LTO, Minisign/GPG signing, SHA-256 manifest, manual/man pages, verification harness. | Target |
| **v0.2.1.0** | Direct GPU Driver Engine | [v0.2.1.1 - v0.2.1.9] HAL traits, CUDA Driver API (cudarc), Vulkan 1.3 SPIR-V, Apple Metal 3, fused BitNet GEMM, fused RMSNorm, fused SwiGLU, fused RoPE, CLI run. | Scheduled |
| **v0.2.2.0** | Out-of-Core NVMe Streaming | [v0.2.2.1 - v0.2.2.9] Linux io_uring, Windows Overlapped I/O, double-buffered prefetch, pinned memory staging, REA-LLM tiered allocation, VRAM accountant, 70B in 8GB testbed, sequential dealloc, saturation monitor. | Scheduled |
| **v0.2.3.0** | Heterogeneous Clustering | [v0.2.3.1 - v0.2.3.9] Intra-node TP, inter-node PP, 1F1B schedule, pure Rust async TCP ring all-reduce, RDMA/RoCE fallback, mDNS discovery, 100ms heartbeat, automated re-sharding, CLI cluster. | Scheduled |
| **v0.2.4.0** | Universal Agentic Gateway | [v0.2.4.1 - v0.2.4.9] WebSocket daemon, JSON-RPC router, HMAC-SHA256 attestation, CLI agent adapter, CI/CD harness adapter, Desktop GUI connector, IDE/Copilot bridge, real-time telemetry, CLI daemon. | Scheduled |
