# burn (Tracel AI) survey for THME

Read-only survey of the shallow clone at `<engines>\burn` (paths relative to it).
S, M, L are estimates, not measurements. Nothing was run. The GPU kernels live in the external `cubecl` and `cubek` crates, which are NOT in this clone, so kernel claims are limited to what the burn crates say.

## 1. Licence

- Root files: `LICENSE-MIT` and `LICENSE-APACHE` (dual), plus `NOTICES.md`. `Cargo.toml` line 20: `license = "MIT OR Apache-2.0"`. Each crate checked (`crates/burn-autodiff`) also carries both licence files.
- `LICENSE-MIT` begins "MIT License / Copyright (c) 2022 Nathaniel Simard & Burn Framework Contributors / Permission is hereby granted, free of charge, to any person obtaining a copy ...". `LICENSE-APACHE` is the stock Apache 2.0 text ("Apache License / Version 2.0, January 2004"); its appendix holder was not checked (unverified).
- `README.md` lines 587 to 599: "distributed under the terms of either the MIT License or the Apache License, Version 2.0 ... Third-party materials ... remain subject to their respective licenses. Opening a pull request is assumed to signal agreement."
- `NOTICES.md` exists ("NOTICES AND INFORMATION ... notices and information required by libraries that this repository copied or derived from"). Sections (headings read): PyTorch MNIST Example (BSD 3-Clause), wgpu, BSL 1.0, num-traits, RP, github-device-flow, Candle pickle reader (candle's `pickle.rs`, MIT and Apache), ICU, FreeBSD msun via libm, Cephes Math Library.
- Obligations if THME copies code: keep the MIT copyright line (Simard and contributors) or the Apache notices with a change statement; any code that burn itself derived (the list in `NOTICES.md`) needs that upstream notice too. Compatible with AGPL-3.0-only if recorded in THME's third-party register. The Cephes and msun entries matter if math routines are copied (check them first).

## 2. What it is

A Rust deep-learning framework for training and inference, written as a stack of backend traits. Backends are decorators and runtimes: `burn-autodiff` wraps any backend to add reverse-mode gradients, `burn-fusion` fuses kernels, and the `burn-cubecl` backend runs on CUDA, ROCm, wgpu (Vulkan, Metal, DX12, WebGPU) and CPU runtimes through the external CubeCL compiler. It targets general models (vision, text classification, RL), not LLM serving; there is no LLaMA-class model in `examples/` (listing: text-generation, text-classification, mnist, wgan and others).

## 3. Components

| Component | What it does | Paths opened |
| :--- | :--- | :--- |
| burn-autodiff | reverse-mode autodiff as a backend decorator, first order only; checkpointing; graph; per-op backward | `crates/burn-autodiff/src/lib.rs`, `ops/backward.rs`, `ops/tensor.rs` (matmul at 690 to 744), `ops/module.rs` (attention at 2268), `checkpoint/strategy.rs`, `backend.rs` |
| burn-cubecl | one backend over all CubeCL runtimes; kernels for matmul, attention, conv, reduce, rms_norm, swiglu, quantization, fft | `crates/burn-cubecl/src/kernel/` listing, `kernel/attention/backward.rs` (head) |
| burn-wgpu | wgpu runtime: Vulkan, Metal, DX12, OpenGL, WebGPU; autotune, fusion; hand-written WGSL via `template` feature | `crates/burn-wgpu/src/lib.rs`, `Cargo.toml` |
| burn-cuda | CUDA runtime for the same backend; driver loaded at run time, no toolkit needed to build | `crates/burn-cuda/src/lib.rs` |
| burn-backend-tests | shared test suite run against every backend, including `autodiff/` | `crates/burn-backend-tests/tests/autodiff/matmul.rs`, `attention.rs`, `ctc.rs` (head) |
| burn-store | weight loading: safetensors, PyTorch `.pt`, burnpack, key remapping | `crates/burn-store/Cargo.toml`, `src/pytorch/mod.rs` (head); no GGUF string found in `Cargo.toml` |
| burn-std | quantization schemes (`QuantScheme`, block scales, ternary BitNet-style mode) | `crates/burn-std/src/tensor/quantization/base.rs` (lines 57 to 118) |
| burn-nn | layers: attention, transformer, rope_encoding, norm, losses | directory listing only |

How autodiff works: each differentiable op implements a stateless `Backward<B, N>` trait with `backward` and `prepare` (`ops/backward.rs` lines 10 to 50). Backward is written with ordinary backend tensor ops, for example matmul's gradient is two `B::float_matmul` calls (`ops/tensor.rs` lines 713, 719). Because the decorator is generic over `B`, the gradient runs on whatever backend runs the forward (CPU, wgpu, CUDA). Memory: a `CheckpointStrategy` trait chooses what to recompute (`checkpoint/strategy.rs`; the default `NoCheckpointing` is imported in `backend.rs` line 2). Attention has two paths: a native fused backward when the backend says `has_attention_backward()`, otherwise a decomposed fallback built from differentiable ops (`ops/module.rs` lines 2277 to 2290); the CubeCL fused backward is in `kernel/attention/backward.rs` (147 lines, goes through external `cubek`).

How it is tested: `burn-backend-tests/tests/autodiff/` has about 80 per-op files (listing: add, matmul, softmax-style activations, conv, attention, cross_entropy, checkpoint, quantization and more). Oracles seen: hand-computed expected gradients (`matmul.rs`: expected [[11,5],[11,5]] and [[3,3],[10,10]]), the same computation spelled out in composed ops (`attention.rs`: "Attention gradients match the same computation spelled out as tensor ops"), and PyTorch reference fixtures (`ctc.rs` comments mention a PyTorch fixture generator). Finite differences: not found in the files opened (unverified).

## 4. What THME lacks that burn has (ranked)

THME today: CPU-only hand-written LLaMA backward checked against central finite differences and Hugging Face (`src/trainer/backward.rs`, 1189 lines, header lines 1 to 12), CUDA via embedded PTX and wgpu WGSL forward kernels (`src/inference/gpu/`, with `shaders/gemm_f32.wgsl`, `attention_tiled.wgsl`, `decoder.cu`), LoRA and AdamW on CPU (`src/trainer/lora.rs`).

Can autodiff give THME a cheap GPU backward pass? Short answer: not by adopting burn, yes by copying its design. Adopting burn means adopting its tensor, its CubeCL kernels and a git-pinned cubecl dependency (`Cargo.toml` lines 227 to 230), replacing THME's own kernels and its AGPL-checked quantized formats. The design lesson is cheap: write each backward as a composition of forward GPU ops THME already has (matmul gradient is two matmuls and a transpose), and keep `backward.rs` as the oracle.

1. **Backward-from-forward-ops design for a GPU training path.**
   - Why: removes the CPU-only limit without a new framework.
   - Paths: `crates/burn-autodiff/src/ops/backward.rs`, `ops/tensor.rs` (matmul 690 to 744), `ops/module.rs` (attention fallback, 2277 to 2290).
   - Translate: for THME's LLaMA block only (RMSNorm, rotary, causal grouped attention, SwiGLU, cross entropy): implement each backward over `src/inference/gpu` kernels, starting with GEMM backward (two GEMMs) and the attention fallback (decomposed, no new fused kernel).
   - Drags in: nothing external. Effort: M to L (estimate; the elementwise and reduction backward kernels need new WGSL and CUDA). Risk: medium (numerics in f16, memory for activations).
   - Test: oracle is THME's own `backward.rs` in `f64` plus Hugging Face gradients; per-op checks against central finite differences, as that file already does.
2. **Activation checkpointing strategy.**
   - Why: GPU training memory. Paths: `crates/burn-autodiff/src/checkpoint/` (`strategy.rs`, `builder.rs`, `retro_forward.rs`, `state.rs`; only `strategy.rs` read). Translate: recompute a block's forward during backward; for a fixed LLaMA block this is a small loop, not the generic graph. Effort: S to M. Risk: low. Test: gradients with and without checkpointing must match exactly (same oracle).
3. **Fused attention backward (logsumexp-saving forward).**
   - Why: memory and speed in backward. Path: `crates/burn-cubecl/src/kernel/attention/backward.rs` calls external `cubek`, so the kernel itself is not here (unverified). Effort: L. Risk: high. Defer until item 1 works with the decomposed fallback. Test: oracle is the decomposed gradient, as burn does in `tests/autodiff/attention.rs`.
4. **Backend-agnostic kernel testing pattern.** `crates/burn-backend-tests/` runs one test suite on every backend. THME could run each GPU op test on CPU, CUDA and wgpu against the CPU reference. Effort: S. Risk: low. Test: itself.
5. **Quantization scheme vocabulary** (per-tensor or per-block scales, BitNet-style ternary mode with scale mean(|W|)): `crates/burn-std/src/tensor/quantization/base.rs`. THME already has TQ1_0 and block formats; useful only as a cross-check of definitions. Effort: S. Risk: low.
6. **Autotune and kernel fusion** (`burn-fusion`, `autotune` features in `crates/burn-wgpu/Cargo.toml`): picks the fastest kernel variant at run time; matches THME's measurement-driven style. Depends on CubeCL's machinery, so only the idea ports: time two kernel variants on first use and cache the winner. Effort: M. Risk: medium.
7. **PyTorch `.pt` reader and key remapping** (`crates/burn-store/src/pytorch/`, `keyremapper.rs`): THME has a scanner and SafeTensors; a safe pickle reader is attractive but the Burn one derives from candle's (`NOTICES.md` Candle section). Not recommended.

## 5. What NOT to port

- The whole `burn` tensor stack, `burn-train`, `burn-nn`, `burn-vision`, `burn-rl`, datasets and the burn-book: THME is not a general framework.
- CubeCL kernels and the cubek attention: external, git-pinned dependency (`Cargo.toml` lines 227 to 230), and they would replace THME's embedded-PTX approach.
- Remote and distributed training (`burn-remote`, `burn-communication`, `burn-shard`, p2p examples): out of scope.
- Router, dispatch and fusion IR crates (`burn-router`, `burn-dispatch`, `burn-ir`, `burn-fusion`): only worth studying if THME ever builds a graph compiler.
- Second-order gradients: burn does not support them either (`lib.rs` line 9), and THME does not need them.
