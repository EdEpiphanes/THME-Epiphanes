# Third-party code register

This register lists every outside source used by this repository, so the origin, licence,
files involved and the notice that must stay with them can be checked in one place.
Licences are read from each source's own licence text — see `LICENSING.md` for the
project-wide model and `LICENSES/anatomy-MIT.txt` for the anatomy figure kit's full MIT
text. Code under GPL, non-commercial or "do not copy" terms is never copied: the idea is
rewritten from the paper. Tracked files whose header names a non-AGPL licence must have a
row here (enforced by `scripts/third_party.py --check`).

`THIRD_PARTY_LICENSES.html` in release archives is generated from `about.hbs` and
`cargo about generate`, and covers the crates below. The crate licences below were checked
against `cargo metadata` on 19 November 2026.

| Name | URL | Licence | What was reused | Notice kept | Files |
| :--- | :--- | :--- | :--- | :--- | :--- |
| minisign | https://crates.io/crates/minisign | MIT | linked as a dependency, no source copied | `THIRD_PARTY_LICENSES.html` in release archives (`about.hbs`) | `Cargo.toml` |
| flate2 | https://crates.io/crates/flate2 | MIT OR Apache-2.0 | linked as a dependency, no source copied | `THIRD_PARTY_LICENSES.html` in release archives (`about.hbs`) | `Cargo.toml` |
| resvg | https://crates.io/crates/resvg | Apache-2.0 OR MIT | linked as a dependency, no source copied | `THIRD_PARTY_LICENSES.html` in release archives (`about.hbs`) | `Cargo.toml` |
| gif | https://crates.io/crates/gif | MIT OR Apache-2.0 | linked as a dependency, no source copied | `THIRD_PARTY_LICENSES.html` in release archives (`about.hbs`) | `Cargo.toml` |
| wgpu | https://crates.io/crates/wgpu | MIT OR Apache-2.0 | linked as a dependency, no source copied | `THIRD_PARTY_LICENSES.html` in release archives (`about.hbs`) | `Cargo.toml` |
| eframe (egui) | https://crates.io/crates/egui | MIT OR Apache-2.0 | linked as a dependency, no source copied | `THIRD_PARTY_LICENSES.html` in release archives (`about.hbs`) | `Cargo.toml` |
| safetensors | https://crates.io/crates/safetensors | Apache-2.0 | linked as a dependency, no source copied | `THIRD_PARTY_LICENSES.html` in release archives (`about.hbs`) | `Cargo.toml` |
| memmap2 | https://crates.io/crates/memmap2 | MIT OR Apache-2.0 | linked as a dependency, no source copied | `THIRD_PARTY_LICENSES.html` in release archives (`about.hbs`) | `Cargo.toml` |
| tokenizers | https://crates.io/crates/tokenizers | Apache-2.0 | linked as a dependency, no source copied | `THIRD_PARTY_LICENSES.html` in release archives (`about.hbs`) | `Cargo.toml` |
| minijinja | https://crates.io/crates/minijinja | Apache-2.0 | linked as a dependency, no source copied | `THIRD_PARTY_LICENSES.html` in release archives (`about.hbs`) | `Cargo.toml` |
| minijinja-contrib | https://crates.io/crates/minijinja-contrib | Apache-2.0 | linked as a dependency, no source copied | `THIRD_PARTY_LICENSES.html` in release archives (`about.hbs`) | `Cargo.toml` |
| ureq | https://crates.io/crates/ureq | MIT OR Apache-2.0 | linked as a dependency (HTTPS client for `thme download`), no source copied | `THIRD_PARTY_LICENSES.html` in release archives (`about.hbs`) | `Cargo.toml` |
| sha2 | https://crates.io/crates/sha2 | MIT OR Apache-2.0 | linked as a dependency (SHA-256 of downloaded files), no source copied | `THIRD_PARTY_LICENSES.html` in release archives (`about.hbs`) | `Cargo.toml` |
| rustls | https://crates.io/crates/rustls | Apache-2.0 OR ISC OR MIT | linked as a dependency of ureq (TLS), no source copied | `THIRD_PARTY_LICENSES.html` in release archives (`about.hbs`) | `Cargo.toml` |
| ring | https://crates.io/crates/ring | Apache-2.0 AND ISC | linked as a dependency of rustls (TLS crypto); its build script compiles C and assembly, no source copied | `THIRD_PARTY_LICENSES.html` in release archives (`about.hbs`) | `Cargo.toml` |
| Chakra Petch font | https://github.com/m4rc1e/Chakra-Petch.git | SIL Open Font License 1.1 | Bundled font for the GUI; no source copied | `assets/fonts/OFL.txt` | `assets/fonts/ChakraPetch-Bold.ttf`, `assets/fonts/ChakraPetch-Medium.ttf` |
| anatomy figure kit | https://github.com/wheresryan22/anatomy | MIT | `src/engine/iso.rs` ported from the kit's core (TypeScript) to Rust; `assets/iso.css` copied | File headers plus `LICENSES/anatomy-MIT.txt` (see `LICENSING.md`) | `src/engine/iso.rs`, `assets/iso.css` |
| llama.cpp | https://github.com/ggml-org/llama.cpp | MIT | sampler chain logic (temperature, top-k, top-p, min-p, Mirostat v1/v2) translated from src/llama-sampler.cpp to Rust | file header plus `LICENSES/llama.cpp-MIT.txt` | `src/inference/sampler.rs` |
| llm.c | https://github.com/karpathy/llm.c | MIT (licence not re-read from upstream LICENSE in this session: unverified) | hand-written transformer forward and backward structure (one function per operation, cached activations) ported to Rust; structure only, no source copied verbatim | file header attribution in `src/trainer/backward.rs` | `src/trainer/backward.rs` |
| Hugging Face PEFT | https://github.com/huggingface/peft | Apache-2.0 (confirmed 27 November 2026 from `peft-0.22.0.dist-info/LICENSE` and the package metadata) | LoRA formulas only: `W + (alpha / r) B A`, uniform A and zero B initialisation; no source copied; no PEFT code included; owner or counsel to confirm whether any Apache-2.0 NOTICE text applies (a NOTICE file was not looked for) | file header in `src/trainer/lora.rs` | `src/trainer/lora.rs` |
| PyTorch | https://pytorch.org | BSD-3-Clause (confirmed 27 November 2026 from `torch-2.11.0+cu128.dist-info/LICENSE`) | external trainer and gradient oracle, run as a separate program in `C:\Users\...\venvs\thme-train` outside the repository; no source copied, nothing bundled in a THME binary | its own licence stays with the installed package | `scripts/make_torch_gradients.py`, the training bridge (planned) |
| Hugging Face transformers | https://github.com/huggingface/transformers | Apache-2.0 (confirmed 27 November 2026 from the installed package metadata and licence file) | reference LLaMA implementation used as the gradient oracle and, with PEFT, as the external trainer; no source copied; nothing bundled | its own licence stays with the installed package | `scripts/make_torch_gradients.py` |
| llama.cpp prebuilt binaries | https://github.com/ggml-org/llama.cpp | MIT ("The ggml authors", confirmed 27 November 2026 from `llama.cpp/LICENSE-llama.cpp.txt`) | used as an external reference engine and tool (`llama-completion`, `llama-perplexity`, `llama-quantize`) in measurements and the compatibility report; downloaded, not bundled in a THME binary | `LICENSE-llama.cpp.txt` beside the binaries; porting code into Rust follows the existing llama.cpp row above | `scripts/compat_report.py`, `scripts/engine_compare.py` |
