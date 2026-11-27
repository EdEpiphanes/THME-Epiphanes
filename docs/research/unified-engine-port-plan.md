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

Filled in from `docs/research/engines/*.md` once the three surveys finish. Each row will carry: feature,
source engine and licence, source paths, THME's current state, effort, risk, oracle.

## Scheduling

After the v0.2 merge. First slice: the highest-ranked item that is small (S) and measurable, to correct the
effort estimates before committing to the rest.
