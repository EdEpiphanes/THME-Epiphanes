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
| **v0.1.3** | BYOA & REA-LLM Engine | Bring Your Own Algorithm registry with JSON import/export, architecture rebuilder from scratch, open weights remapping, REA-LLM EWHA routing & energy telemetry, layer anatomy deconstruction, dataset ingestion, and micro-trainer. | Completed |
| **v0.1.4** | Empirical Quality Gate | External perplexity evaluation bridge (WikiText-2, C4) with automated rollback gating (Delta PPL <= epsilon) and CLI validate enhancements. | Completed |
| **v0.1.5** | Surgery Daemon, IDE & Interop | Headless JSON-RPC MCP server (`thme mcp`), VS Code extension (`thme-vscode-0.1.5.vsix`), GitHub Copilot LM Tools, PyTorch/CUDA ecosystem interop bridge with PEFT LoRA zero-copy merge, and Universal 8 Specialized AI Model Paradigm Converter (LLM, LCM, LAM, MoE, VLM, SLM, MLM, SAM). | Completed |
| **v0.1.6** | Speculative Decoding Synthesizer | Draft model manifold extraction from early-exit checkpoints, multi-token tree speculation verification engine, acceptance rate & latency speedup modeling, and Desktop Studio deck. | Active |
| **v0.1.7** | Roofline & Kernel Profiler | Hardware-calibrated Arithmetic Intensity vs Memory Bandwidth roofline model with L1/L2/L3/HBM cache hierarchy simulation. | Planned |
| **v0.1.8** | Binary Quantization Exporter | Direct serialization to deployable GGUF and SafeTensors files with BitNet b1.58 ternary packed weights and calibrated per-channel scale factors. | Planned |
| **v0.1.9** | Closed-Loop Teacher Surgery | Autonomous Teacher-Student co-surgery optimization pipeline: automated recipe generation, delta-PPL verification, and iterative Pareto tuning. | Planned |
| **v0.2.0** | Production Core Release | Verified production deployment on real open weights (SmolLM-135M / LLaMA-3.2-1B) with reproducible before-and-after benchmarks and signed release binaries. | Target |
| **v0.2.1** | Direct GPU Driver Engine | Pure Rust GPU execution via Vulkan 1.3 SPIR-V, Apple Metal 3, and NVIDIA CUDA Driver API bypassing Python runtimes. | Scheduled |
| **v0.2.2** | Tiered Out-of-Core SSD Streaming | Three-tier memory hierarchy (VRAM <-> RAM <-> NVMe SSD) with asynchronous direct I/O for 70B inference in 8 GB VRAM. | Scheduled |
| **v0.2.3** | Multi-Node Heterogeneous Clustering | Distributed Pipeline and Tensor Parallelism across mixed nodes with pure Rust async TCP/RDMA ring all-reduce. | Scheduled |
| **v0.2.4** | Universal Agentic AI Gateway | JSON-RPC 2.0 / WebSocket daemon connecting CLI agents, CI/CD harnesses, desktop studio, and IDEs with HMAC-SHA256 recipe attestation. | Scheduled |
