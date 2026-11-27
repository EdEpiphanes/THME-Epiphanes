# Unified engine port: plan

27 November 2026. Owner direction: take the best features of the top open-source inference engines
into one unified Rust port inside THME, to cut time on coding, testing and research, and add the good
parts for whatever is missing in THME's own engine. Status: surveys running; nothing ported yet.

## The rule: gap-driven

A feature is ported only if THME lacks it or does it measurably worse. The starting point is THME's own
capabilities (README feature tables, `docs/subsystems.md`, the code), not the other engines' feature
lists. For every candidate the question is "what does THME not have that this engine does well, and what
does it cost to get?", not "what does this engine have?".

## Selection criteria, in order

1. **Gap.** THME has no equivalent, or its equivalent is weaker by a measurement we can name.
2. **Licence.** Permissive only (MIT, Apache-2.0, BSD, ISC, Zlib, CC0, Unlicense), read from the LICENSE
   file in the clone; Apache-2.0 notice and patent terms noted; copyleft, non-commercial, custom and
   unlicensed code is not used. Licence decisions beyond that stay with the owner and counsel.
3. **Fit.** It must keep the `thme` binary under 16 MiB and the Studio under 25 MiB, add no heavy
   dependency, and work on THME's backends (CPU, CUDA with embedded PTX, wgpu).
4. **Testable against the source.** The original engine is the oracle: a command or function whose output
   THME's port must reproduce (bit for bit where the maths is exact, within a stated tolerance otherwise).
5. **Effort and risk**, estimated S, M or L by the survey and then corrected after a first slice.

## The process for each feature

1. Write a failing test whose expected value comes from the source engine, not from the new code.
2. Port the idea or the code to Rust, with an attribution header and a row in `docs/third-party.md`.
3. Run the full gate and the oracle comparison; record what is and is not covered.
4. Measure the effect on THME (quality, speed, memory) under the drift protocol and commit the report.
5. Only then mark the feature done, with the evidence.

## What is already ported or built from these sources

- Sampler chain from llama.cpp (temperature, top-k, top-p, min-p, Mirostat), MIT, in `src/inference/sampler.rs`.
- Transformer backward-pass structure from llm.c (MIT) and LoRA formulas from PEFT (Apache-2.0).
- llama.cpp, PyTorch, PEFT and transformers as external tools, not bundled.

## Candidates and ranking

Compiled on 27 November 2026 from nine read-only surveys (`docs/research/engines/*.md`). Effort is the
surveys' estimate (S, M, L), not a measurement; each note marks what it did not verify. "Oracle" is the
reference output the port must reproduce.

### Tier 1: small, measurable, testable on this machine

| # | Feature | From | THME today | Effort | Oracle |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | Hadamard rotation before KV-cache quantization (keys first) | exllamav2 (Q4 cache), vLLM (`int4_per_token_head.py`) | int8 and int4 KV and a Walsh-Hadamard transform exist; no rotation before quantizing. Measured: on Pythia-14M the keys cause all the quantization loss | S-M | perplexity with and without rotation (Pythia-14M int8 keys: 245.2 against 190.8 at f32) |
| 2 | Importance-matrix (imatrix) collection and imatrix-weighted K-quants | llama.cpp, candle, mistral.rs | none (`src/inference/calibration.rs` overlap unverified) | S-M | `llama-imatrix` and `llama-quantize --imatrix` output (the binaries are on this machine) |
| 3 | Missing encoders: Q2_K, Q3_K, Q4_1, Q5_0, Q5_1, IQ4_NL, IQ4_XS | llama.cpp | encodes Q4_K, Q5_K, Q6_K; refuses Q2_K | S-M | byte comparison with `llama-quantize` output |
| 4 | Run-time RoPE scaling: linear, YaRN, longrope | llama.cpp | forward pass reads only the base frequency | S-M | `tests/test-rope.cpp` values and llama.cpp logits |
| 5 | Remaining samplers: typical, XTC, DRY, penalties | llama.cpp | temperature, top-k, top-p, min-p, Mirostat | S each | `tests/test-sampling.cpp` |
| 6 | Read Ollama's local manifest and blob store, Modelfile import | ollama | none | S each | `ollama show --modelfile` |

### Tier 2: medium

| # | Feature | From | Effort | Oracle |
| :--- | :--- | :--- | :--- | :--- |
| 7 | Speculative decoding runner with a rejection-sampling verifier (turns the "Model" status into "Measured") | mistral.rs | M | greedy output equals the target-only run |
| 8 | Prefix cache (block hash map, radix tree); needs a block-structured KV cache first | vLLM, SGLang | M | greedy output with the cache equals output without it |
| 9 | Constrained generation (a per-token logit mask is easy; compiling grammars is the hard part) and GBNF | llama.cpp, SGLang, vLLM | M-L | a schema validator on every output |
| 10 | GPU backward pass built from forward operations (design only, for the LLaMA block) | burn, candle | M-L | `src/trainer/backward.rs` and the Hugging Face gradients (THME-3107) |

### Tier 3: large, later

Paged attention with a continuous-batching scheduler (vLLM, mistral.rs); intra-tensor mixed-bit formats
and their GEMV kernels (EXL2); CUDA quantized-matmul kernels (candle, from llama.cpp); an HQQ-style
quantizer and activation checkpointing as cheaper runners-up.

### Not ported, and why

- The `tch` and `pyo3`-based parts of SGLang, and anything that needs libtorch.
- candle's model code and mistral.rs's model code: written against candle's tensor type, so only ideas travel.
- Ollama's scheduler: useful only if THME gets a server. Ollama bundles a pinned llama.cpp, so it adds
  packaging, not new numerics.
- EXL2's simulated-annealing allocator: stochastic; THME's exact dynamic-programming allocator is better. Its
  per-module output-error cost proxy and measurement file are worth reading.
- burn as a dependency: its GPU kernels sit in separate pinned crates that were not available to check, and
  it would break the size cap.

### Licence cautions found by the surveys

- The ExLlamaV2 MIT file and candle's `LICENSE-MIT` carry no copyright line, so the holder is unverified:
  confirm it before copying anything from them.
- MLC-LLM has a NOTICE file that must be carried into any derivative that includes it.
- Apache-2.0 sources need their licence text and any NOTICE kept; candle's CUDA quantization kernels are
  adapted from llama.cpp and would bring a second notice.
- Nested licences exist (MIT and BSD-3 folders inside vLLM and SGLang); none was read beyond the survey.

## Scheduling

After the v0.2 merge. First slice: item 1 (rotate before quantizing the KV cache). It is small, uses the
harness and the failing case we already have (Pythia-14M keys), and has a clear pass test: int8-key perplexity
moves from 245.2 toward the f32 value of 190.8. Second slice: items 2 and 3 together (imatrix and the missing
encoders), tested against the `llama-quantize` binary already on this machine. The first slice's real effort
corrects the estimates for the rest.
