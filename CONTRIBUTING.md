# Contributing to THME

Thank you for your interest in contributing to **THME (Topological Heterogeneous Manifold Engine)** by **Epiphanes**! This project is an open-source initiative dedicated to advancing modular, reproducible, and verifiable optimization for large language models.

---

## Code of Conduct

All contributors and maintainers are expected to follow our [Code of Conduct](CODE_OF_CONDUCT.md).

---

## Research & Engineering Principles

1. **Verify Before Claiming**: Do not publish claims of universal superiority based on one model or synthetic benchmark. Distinguish verified experimental results from hypotheses.
2. **Deterministic & Reproducible**: All benchmarks, sensitivity measurements, and allocation plans must include exact environment parameters, seeds, and hardware metadata.
3. **Strict Quality Gating**: Any proposed optimization strategy must provide an associated anti-degradation quality metric ($Q_{\text{candidate}} \ge Q_{\text{baseline}} - \epsilon$).
4. **Clean Git Hygiene**: Never commit absolute local file paths, personal usernames, binary model weights (`.safetensors`, `.gguf`), or compiled binaries (`.exe`, `.dll`, `.so`).

---

## Development Setup

1. Install the Rust toolchain (version >= 1.80 recommended):
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   ```
2. Clone the repository:
   ```bash
   git clone https://github.com/EdEpiphanes/THME-Epiphanes.git
   cd THME
   ```
3. Run test suite:
   ```bash
   cargo test
   ```
4. Check formatting and lints:
   ```bash
   cargo fmt --check
   cargo clippy -- -D warnings
   ```

---

## Pull Request Guidelines

1. **Topic Branch**: Create a feature branch with a descriptive name (`git checkout -b feature/qwen-parser`).
2. **Testing**: Add unit and integration tests covering your additions under `tests/`.
3. **Documentation**: Update the README or module-level rustdocs if you introduce new CLI flags or public APIs.
4. **Commits**: Write clear, conventional commit messages (`feat:`, `fix:`, `perf:`, `docs:`, `chore:`).
