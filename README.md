<p align="center">
  <img src="assets/brand/thme-banner.gif" alt="THME title spread in black-and-white manga style: the mascot breathing in the left panel; in the right panel a Calabi-Yau manifold turning in 3D and through its fourth dimension inside focus lines, under the THME wordmark" width="900" />
</p>

<h1 align="center">THME</h1>

<p align="center">
  <strong>Topological Heterogeneous Manifold Engine</strong> by Epiphanes<br>
  A pure-Rust workbench for taking language models apart, changing how they are stored and run,<br>
  and measuring what each change costs.
</p>

<p align="center">
  <a href="https://github.com/EdEpiphanes/THME-Epiphanes/actions/workflows/ci.yml"><img src="https://github.com/EdEpiphanes/THME-Epiphanes/actions/workflows/ci.yml/badge.svg" alt="CI" /></a>
  <img src="https://img.shields.io/badge/Rust-2021%20%C2%B7%20MSRV%201.89-orange?logo=rust&style=flat-square" alt="Rust 2021, MSRV 1.89" />
  <img src="https://img.shields.io/badge/Formats-SafeTensors%20%C2%B7%20GGUF%20v2%2Fv3-10b981?style=flat-square" alt="Formats" />
  <img src="https://img.shields.io/badge/Measured%20on-SmolLM--135M%20%C2%B7%20Pythia--14M-2a78d6?style=flat-square" alt="Measured models" />
  <a href="LICENSING.md"><img src="https://img.shields.io/badge/License-AGPL--3.0-blue?style=flat-square" alt="License: AGPL-3.0; commercial license planned" /></a>
</p>

<p align="center">
  <a href="#now-building">Now building</a> ·
  <a href="#measured-results">Measured results</a> ·
  <a href="#what-is-real-today">What is real today</a> ·
  <a href="#thme-studio">Studio</a> ·
  <a href="#quickstart">Quickstart</a> ·
  <a href="#architecture">Architecture</a> ·
  <a href="#status">Status</a> ·
  <a href="#roadmap">Roadmap</a> ·
  <a href="docs/benchmark_dossier.md">Benchmark dossier</a> ·
  <a href="docs/research/THME_measured_findings_2026-11.md">Research notes</a>
</p>

---

## Why THME

Most compression tools apply one format to every layer and report a size. THME is built on
the opposite premise: every tensor can have its own representation, chosen by measuring what
it costs. You describe a change as a declarative **recipe** (or bring your own algorithm via
the **BYOA** registry), THME applies it, runs the model, and a quality gate accepts or rolls
it back.

Every number below comes from a real forward pass and can be regenerated with one command.
Anything THME cannot yet measure is labelled as an estimate or a simulation.

## Now building

THME is open source (AGPL-3.0-only) and is finishing its **engine and tools first**; a
commercial release follows later. This week's work, all tested on synthetic data:

| Track | Shipped | In progress |
| :--- | :--- | :--- |
| **Epiphanes model format** | [`.epiph` v1.0](docs/formats/epiph-v1.md): one verifiable file with weights, model description, recipe, provenance, quality results and calibration (`thme epiph pack`, `thme epiph verify`); signed with Minisign (`thme epiph sign`); stores recipe candidates quantized (`--recipe R --quantize`); keeps GGUF metadata and tokenizer files (`thme epiph extract`); converts to GGUF and SafeTensors | Streaming for large models |
| **Model security** | `thme scan PATH`: finds code-executing pickles, chat-template injection, `trust_remote_code` files, hidden executables and malformed headers without loading anything; never runs the code it finds | More formats; scan on import |
| **Math engine** | `src/engine`: one shared Walsh-Hadamard transform; FP16, BF16 and FP8 with one tested rounding routine; integer, ternary, sign, codebook and phase quantizers with stated error bounds; an exact DP allocator (`thme search --solver dp`) beside greedy and thresholds | Transport, ODE steppers, SVG visual engine |
| **Quality and evidence** | [Versioned quality gates](docs/quality-gate.md) with custom criteria and a hash in every report (`--gate FILE`); every figure labelled Measured, Tested, Estimate or Simulated; glossary in [CONTEXT.md](CONTEXT.md); [benchmark registry](docs/benchmark-registry.md): task files and result records (`thme bench`); metrics and statistics libraries checked against the dossier's oracle values | Three-state gate; benchmark runner |

Model runs are paused, so work that needs a real-model measurement is marked *measurement
pending*. Full plan: **[Now, next, later](ROADMAP.md#now-next-later)**.

## Measured results

Measured on the full WikiText-2 test split with THME's own CPU forward pass. No retraining or
calibration. Quality gate: perplexity may rise by at most 0.45.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/benchmarks/smollm-135m/charts/quality-gate-dark.svg">
  <img alt="SmolLM-135M perplexity: source 19.80, exported GGUF Q8_0 19.83 inside the gate, surgery recipe 48.71 outside the gate" src="docs/benchmarks/smollm-135m/charts/quality-gate-light.svg" width="760">
</picture>

| Model | Source | Exported GGUF, uniform Q8_0 | Surgery recipe | Scored tokens |
| :--- | ---: | ---: | ---: | ---: |
| SmolLM-135M | 19.80 | 19.83 (Δ +0.03, **passes**) | 48.71 (Δ +28.91, fails) | 303,118 |
| Pythia-14M | 114.29 | 115.12 (Δ +0.83, fails) | 618.53 (Δ +504.24, fails) | 286,878 |

The exported SmolLM-135M GGUF is 136 MiB, against a 513 MiB FP32 source (about 257 MiB at FP16, so about 47% smaller than FP16).

### What the experiments show

**1. Where a model is fragile is measurable, and it is not where the heuristics guessed.**
Ternarizing one SmolLM-135M layer at a time costs +0.9 to +6.8 perplexity in the middle of
the network, +13 at the last layer, and +1,429 at layer 0.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/benchmarks/smollm-135m/charts/layer-sensitivity-dark.svg">
  <img alt="Per-layer perplexity cost of ternarization in SmolLM-135M, log scale; layer 0 is far above the rest" src="docs/benchmarks/smollm-135m/charts/layer-sensitivity-light.svg" width="760">
</picture>

**2. Tensor roles differ by an order of magnitude.** The attention value projection holds
2.5% of the weights and is the most fragile role; ternarizing it alone raises perplexity by
+359.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/benchmarks/smollm-135m/charts/role-sensitivity-dark.svg">
  <img alt="Perplexity cost of ternarizing each tensor role in SmolLM-135M; attention V is highest" src="docs/benchmarks/smollm-135m/charts/role-sensitivity-light.svg" width="760">
</picture>

**3. Placement matters.** With identical formats and tensor counts, the sensitivity-guided
recipe beat three random placements. This is exploratory: n = 3, and every row still fails
the gate.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/benchmarks/smollm-135m/charts/placement-dark.svg">
  <img alt="Recipe +25.5 perplexity versus random placements +53.3, +82.3 and +173.2" src="docs/benchmarks/smollm-135m/charts/placement-light.svg" width="760">
</picture>

**4. Costs do not add up.** The per-layer costs sum to +1,509. Ternarizing all of those
layers together measures +15.9 million. An allocator that adds per-layer costs cannot rank
aggressive candidates; that is why THME evaluates whole candidates.

**5. Early exit is not free.** Read through SmolLM-135M's own output head, 1.6% of tokens
have settled by 30% depth and 28.7% by 80%. Exits need training before they pay off.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/benchmarks/smollm-135m/charts/depth-settling-dark.svg">
  <img alt="Share of tokens whose top prediction has settled, by depth; 1.6% at 30% depth and 28.7% at 80% depth" src="docs/benchmarks/smollm-135m/charts/depth-settling-light.svg" width="760">
</picture>

<details>
<summary><b>Pythia-14M charts</b> (fast iteration model, about 1 minute per full-split evaluation)</summary>

Pythia-14M is 92% embeddings: quantizing every linear layer changes only 8.4% of its weights.
Use it for fast quality checks, not for memory-reduction studies.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/benchmarks/pythia-14m/charts/quality-gate-dark.svg">
  <img alt="Pythia-14M perplexity: source 114.29, Q8_0 115.12, recipe 618.53" src="docs/benchmarks/pythia-14m/charts/quality-gate-light.svg" width="760">
</picture>
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/benchmarks/pythia-14m/charts/role-sensitivity-dark.svg">
  <img alt="Pythia-14M role sensitivity" src="docs/benchmarks/pythia-14m/charts/role-sensitivity-light.svg" width="760">
</picture>
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/benchmarks/pythia-14m/charts/layer-sensitivity-dark.svg">
  <img alt="Pythia-14M layer sensitivity" src="docs/benchmarks/pythia-14m/charts/layer-sensitivity-light.svg" width="760">
</picture>

</details>

Full tables, method, provenance and hashes: **[benchmark dossier](docs/benchmark_dossier.md)**.
Interpretation against the research programme: **[measured findings](docs/research/THME_measured_findings_2026-11.md)**.

## What is real today

| Capability | Status | Evidence |
| :--- | :--- | :--- |
| SafeTensors and GGUF v2/v3 import, 28 GGML types | Implemented | Unit and fuzz tests |
| Reference forward pass (LLaMA family, GPT-NeoX) | **Measured** | WikiText-2 perplexity on 2 models |
| Precision round trips: FP16, BF16, Q8_0, Q4_0, ternary TQ1_0 | **Measured** | [Precision sweeps](docs/benchmark_dossier.md) |
| GGUF v3 export | **Measured** | Uniform Q8_0 re-imported and evaluated; per-tensor formats (Q4_K, Q5_K, Q6_K, ternary) Tested |
| Precision sweep, random-placement control, depth probe | **Measured** | `examples/precision_sweep.rs`, `examples/depth_probe.rs` |
| `thme validate --report`, `thme co-surgery MODEL --tokens` | **Measured** | Gate measured reports; co-surgery scores candidates with the forward pass |
| Surgery recipes, BYOA registry | Implemented | Recipes are measured through the verification pipeline and co-surgery |
| Layer sensitivity profiler (`thme profile`) | Estimate | Analytic model; not yet calibrated against measured sensitivity |
| `thme evaluate --recipe`, `thme search` | **Measured** | Any recipe through the gate; mixed-precision search on measured costs (greedy or exact DP solver, `--solver`) with calibration/validation split and random controls |
| Pareto allocator (`thme plan`) | Estimate | Analytic costs; superseded by `thme search` for measured allocation |
| Co-surgery without tokens, Studio Co-Surgery tab | **Simulated** | Analytic proxy, labelled as simulated in the output |
| Speculative decoding, roofline, energy | Model | Formulas over published hardware specs; no timing measured |
| Paradigm converter | Implemented | Rewrites the IR; converted models are not evaluated |
| PyTorch/CUDA scripts, LoRA merge, MCP server, VS Code extension | Implemented | Unit tests |
| BYOA algorithms in recipes | **Measured** | Contract-checked; Hadamard-rotated ternary and 4-bit built-ins; invalid transforms refused |
| Studio Live 3D view | **Measured** | Per-tensor activity, residual updates and anomalies from the forward-pass probe (`src/inference/probe.rs`) |
| `.epiph` model files | Tested | Round trip, corruption and truncation, GGUF conversion byte-identical to the source |
| `thme scan` security scanner | Tested | Built fixtures for each rule; nothing is loaded or run |
| Engine primitives (`src/engine`) | Tested | Oracles: explicit Hadamard matrix, exhaustive search, round-and-clamp |
| Other Studio tabs | Implemented | Real screenshots below; telemetry uses a synthetic demo model |

## THME Studio

A native desktop workbench (`thme-gui`, egui) in a black-and-white manga style: paper, ink,
screentone and glass panels, set in Chakra Petch. Every image here is the actual app, captured by
`scripts/capture_studio.ps1` and `scripts/record_gif.ps1`.

### Live 3D model view

Watch a model compute. The default **Manifold** view binds each of the 25 patches of a
Calabi-Yau quintic cross-section to a layer: a patch's tone is the layer's format family and its
light is the layer's **measured** residual update in the current window. The surface turns
through its fourth dimension while token particles ride across it. The geometry is
illustrative; the tones and light are measured. **Layer slabs** shows the detail: every slab is
one applied layer, every row a linear tensor split into output-channel groups, lit by measured
output magnitude, with particles carrying the residual stream between layers. Non-finite
values, exploding residuals and dead MLP units are marked and listed beside the view. Drag to
orbit, scroll to zoom, hover for statistics. Point it at any local LLaMA-family or GPT-NeoX
folder, or run the built-in synthetic demo shown here.

<p align="center">
  <img src="assets/gifs/live-3d-manifold.gif" alt="Recording of the Live 3D manifold view: a black-and-white manga-shaded Calabi-Yau surface turning and morphing through its fourth dimension, patches toned by format family and lit by measured layer activity, token particles flowing" width="760" />
</p>

<table>
  <tr>
    <td width="50%"><img src="assets/gifs/live-3d-slabs.gif" alt="Recording of the Live 3D layer-slab view with measured per-channel activity and token-flow particles" /><br><sub><b>Layer slabs</b>: tensors, channel groups, residual flow</sub></td>
    <td width="50%"><img src="assets/gifs/dashboard.gif" alt="Recording of the Studio dashboard in the Manga Panel theme" /><br><sub><b>Manga Panel</b> theme (default); <b>Manga Night</b> and <b>Manga Ink</b> keep the red</sub></td>
  </tr>
</table>

<sub>Recorded from the real app with `scripts/record_gif.ps1` while it runs the synthetic demo
(pseudo-random weights, mixed formats), so perplexity is not meaningful; the activity is real
computation. Binary and phase families and adaptive routing are in the legend as not implemented.</sub>

### Workspaces

The other tabs below show the Studio's built-in synthetic demo model, so every number on screen
is illustrative rather than measured.

<table>
  <tr>
    <td width="50%"><img src="assets/screenshots/00-dashboard.png" alt="Studio dashboard tab" /><br><sub><b>Dashboard</b>: model summary, sensitivity allocation, audit stream</sub></td>
    <td width="50%"><img src="assets/screenshots/02-layer-tweaker.png" alt="Studio layer tweaker tab" /><br><sub><b>Layer Tweaker</b>: per-layer anatomy, α/β/γ ratios, BYOA sub-block algorithms</sub></td>
  </tr>
  <tr>
    <td><img src="assets/screenshots/03-matrix-engine.png" alt="Studio matrix engine tab" /><br><sub><b>Matrix Engine</b>: per-parameter transformation decisions</sub></td>
    <td><img src="assets/screenshots/04-benchmarks.png" alt="Studio benchmarks tab" /><br><sub><b>Benchmarks</b>: flow simulator, KV-cache scaling, roofline deck</sub></td>
  </tr>
</table>

<details>
<summary>All ten tabs</summary>

| | |
| :---: | :---: |
| <img src="assets/screenshots/01-models.png" alt="Models tab" /><br><sub>Models</sub> | <img src="assets/screenshots/05-history.png" alt="History tab" /><br><sub>History</sub> |
| <img src="assets/screenshots/06-audit-log.png" alt="Audit log tab" /><br><sub>Audit Log</sub> | <img src="assets/screenshots/07-surgery-export.png" alt="Surgery and export tab" /><br><sub>Surgery & Export</sub> |
| <img src="assets/screenshots/08-dev-lab.png" alt="Dev lab tab" /><br><sub>Dev Lab & Training</sub> | <img src="assets/screenshots/09-co-surgery.png" alt="Co-surgery tab" /><br><sub>Co-Surgery</sub> |

</details>

## Quickstart

Requires Rust 1.89 or newer. Measuring perplexity also needs Python with `tokenizers`.

```bash
git clone https://github.com/EdEpiphanes/THME-Epiphanes.git
cd THME-Epiphanes
cargo build --release
```

**Explore without any download** (synthetic fixture):

```bash
cargo run --example generate_mock_model
./target/release/thme inspect examples/mock_llama
./target/release/thme scan examples/mock_llama                       # security scan, loads nothing
./target/release/thme epiph pack examples/mock_llama --out target/mock.epiph
./target/release/thme epiph verify target/mock.epiph                 # digests, shapes, types
./target/release/thme gui
```

**Measure a real model.** Put a Hugging Face checkpoint (`model.safetensors` + `config.json`)
in `models/<name>/`. Weights are never committed.

```bash
pip install tokenizers
python scripts/prepare_eval_tokens.py --repo EleutherAI/pythia-14m

# Quality gate: source vs exported GGUF vs surgery recipe
cargo run --release --example production_verification -- \
    --model-path models/Pythia-14M --tokens target/eval/wikitext2_test.pythia-14m.u32

# Gate the report it wrote
./target/release/thme validate --report target/verification_report.json

# What does each format, role and layer cost?
cargo run --release --example precision_sweep -- \
    --model-path models/Pythia-14M --tokens target/eval/wikitext2_test.pythia-14m.u32 --layer-scan

# How early do predictions settle; what do repeated or skipped layers do?
cargo run --release --example depth_probe -- \
    --model-path models/Pythia-14M --tokens target/eval/wikitext2_test.pythia-14m.u32

# Write the recipe as a GGUF file, each tensor in its own format
./target/release/thme export models/SmolLM-135M -o smollm-mixed.gguf --recipe recipe.json

# Search a mixed-precision recipe under a size budget, then re-check it
./target/release/thme search models/Pythia-14M --tokens target/eval/wikitext2_test.pythia-14m.u32 \
    --budget 0.5 --formats q8_0,q4_0 --emit-recipe target/searched.json
./target/release/thme evaluate models/Pythia-14M --recipe target/searched.json \
    --tokens target/eval/wikitext2_test.pythia-14m.u32

# Closed-loop co-surgery with measured perplexity
./target/release/thme co-surgery models/Pythia-14M --tokens target/eval/wikitext2_test.pythia-14m.u32
```

Each run writes a JSON report. `scripts/generate_dossier.py` and `scripts/generate_charts.py`
turn reports into the dossier and the charts above; neither script contains numbers of its own.

| Model | Role | Full-split evaluation on a 6-core desktop CPU |
| :--- | :--- | :--- |
| [Pythia-14M](https://huggingface.co/EleutherAI/pythia-14m) | Fast iteration on techniques | About 50 s |
| [SmolLM-135M](https://huggingface.co/HuggingFaceTB/SmolLM-135M) | Release verification | About 9 min |

## Architecture

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'fontFamily': 'Chakra Petch, Bahnschrift, Arial Narrow, sans-serif', 'primaryColor': '#ffffff', 'primaryTextColor': '#0b0b0b', 'primaryBorderColor': '#0b0b0b', 'lineColor': '#0b0b0b', 'secondaryColor': '#f6f6f4', 'tertiaryColor': '#ffffff', 'clusterBkg': '#f6f6f4', 'clusterBorder': '#0b0b0b', 'edgeLabelBackground': '#ffffff', 'noteBkgColor': '#ffffff', 'noteBorderColor': '#0b0b0b', 'actorBkg': '#ffffff', 'actorBorder': '#0b0b0b', 'signalColor': '#0b0b0b', 'labelBoxBkgColor': '#ffffff'}}}%%
flowchart LR
    classDef measured fill:#0b0b0b,stroke:#0b0b0b,stroke-width:2px,color:#ffffff;
    classDef impl fill:#ffffff,stroke:#0b0b0b,stroke-width:1.5px,color:#0b0b0b;

    subgraph In["Import"]
        ST["SafeTensors"]:::impl
        GG["GGUF v2/v3"]:::impl
    end
    IR["ModelIR<br/>tensors, roles, sources"]:::impl
    subgraph Change["Change"]
        R["Surgery recipe / BYOA"]:::impl
        P["Profiler and planner"]:::impl
    end
    subgraph Measure["Measure"]
        F["Reference forward pass<br/>LLaMA · GPT-NeoX"]:::measured
        G["Perplexity gate<br/>accept or roll back"]:::measured
    end
    E["GGUF export"]:::measured
    D["Reports → dossier, charts"]:::measured

    ST & GG --> IR --> P --> R --> F --> G
    G -- accept --> E --> F
    G --> D
```

Solid ink stages are backed by measurements today; white stages are implemented and tested. The full subsystem reference (profiler,
allocator, speculative decoding, roofline, paradigm converter, PyTorch bridge, MCP, CLI flags)
is in **[docs/subsystems.md](docs/subsystems.md)**.

## Status

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/status/status-dark.svg">
  <img alt="Project status board: milestones and sub-milestones done, in progress and open for the v0.1 and v0.2 series, active backlog tickets by area, and tests by module" src="docs/status/status-light.svg" width="900">
</picture>

<sub>Generated by `scripts/generate_status.py` from [ROADMAP.md](ROADMAP.md),
[BACKLOG.md](BACKLOG.md) and the test suite. CI regenerates it and fails if the committed board is stale.</sub>

## Roadmap

Two views of one plan. The **[now, next, later](ROADMAP.md#now-next-later)** view groups work by
track (math engine, Epiphanes model format, security, quality and evidence). The numbered
milestones below track the measured-model programme, which waits while model runs are paused.

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'fontFamily': 'Chakra Petch, Bahnschrift, Arial Narrow, sans-serif', 'primaryColor': '#ffffff', 'primaryTextColor': '#0b0b0b', 'primaryBorderColor': '#0b0b0b', 'lineColor': '#0b0b0b', 'secondaryColor': '#f6f6f4', 'tertiaryColor': '#ffffff', 'clusterBkg': '#f6f6f4', 'clusterBorder': '#0b0b0b', 'edgeLabelBackground': '#ffffff', 'noteBkgColor': '#ffffff', 'noteBorderColor': '#0b0b0b', 'actorBkg': '#ffffff', 'actorBorder': '#0b0b0b', 'signalColor': '#0b0b0b', 'labelBoxBkgColor': '#ffffff'}}}%%
flowchart LR
    classDef done fill:#0b0b0b,stroke:#0b0b0b,stroke-width:2px,color:#ffffff;
    classDef active fill:#ffffff,stroke:#0b0b0b,stroke-width:4px,color:#0b0b0b;
    classDef next fill:#ffffff,stroke:#0b0b0b,stroke-width:1.5px,stroke-dasharray:4 3,color:#52514e;

    A["v0.1.x<br/>Foundations<br/>implemented"]:::done
    B["v0.2.0<br/>Measured core"]:::active
    C["v0.2.1<br/>Recipe and BYOA lab<br/>done"]:::done
    D["v0.2.2<br/>Formats and calibration"]:::active
    E["v0.2.3<br/>Recovery and distillation"]:::next
    F["v0.2.4<br/>Runtime, speed, KV cache"]:::next
    H["v0.2.5<br/>Math engine"]:::active
    I["v0.2.6<br/>Epiphanes model format"]:::active
    J["v0.2.7<br/>Security and trust"]:::active
    K["v0.2.8<br/>Quality gate,<br/>benchmarks"]:::next
    L["v0.2.9<br/>Translation layer,<br/>convergence"]:::next
    G["v0.3<br/>Routing, distillation,<br/>GPU, scale-out"]:::next
    A --> B --> C --> D --> E --> F --> H --> I --> J --> K --> L --> G
```

<sub>Solid ink: done · thick frame: in progress · dashed: next. Live counts are on the [status board](#status).</sub>

| Milestone | Goal | Done when |
| :--- | :--- | :--- |
| **v0.2.0 Measured core** (active; the HF cross-check is paused) | Real forward pass, measured gate, honest dossier | Every published figure regenerates from a report; analytic proxy retired or labelled everywhere |
| **v0.2.1 Recipe and BYOA lab** (done) | Any recipe or BYOA algorithm runs through the measured gate | A searched recipe beats uniform and random placement at matched size |
| **v0.2.2 Formats and calibration** (in progress) | 4-bit K-quants, activation-aware scales, Hadamard rotation, per-tensor export | A heterogeneous artifact smaller than uniform Q8_0 passes the gate |
| v0.2.3 Recovery | Distillation and LoRA recovery for aggressive formats | A recovered ternary-heavy candidate passes the gate |
| v0.2.4 Runtime | Incremental decode, measured speed, memory and KV cache | Speed and memory claims are measured on named hardware |
| **v0.2.5 Math engine** (in progress) | Validated primitives with oracle tests, one implementation per method | Every primitive passes its oracle tests; no duplicates |
| **v0.2.6 Epiphanes model format** (in progress) | `.epiph` with signatures, quantized payloads, tokenizer, placement | A quantized candidate packs, signs, verifies and runs from one file |
| **v0.2.7 Security and trust** (in progress) | Scanner, permission record, supply chain; sandbox reserved for the commercial version | Every import scans first; untrusted code is never run |
| v0.2.8 Quality gate and benchmarks | Versioned gate, benchmark harness, statistics, charts | A task file runs to a result with an interval and a three-state verdict |
| v0.2.9 Translation layer and convergence | Operation graph, exact symmetries, compatibility reports | Every v0.2 exit criterion met or deferred with a reason |

Full plan with sub-milestones: **[ROADMAP.md](ROADMAP.md)** · Tickets: **[BACKLOG.md](BACKLOG.md)**

## Research

<img src="assets/brand/thme-mascot-loop.gif" align="right" width="360" alt="The THME mascot, drawn in black-and-white manga style, breathing slowly while a wireframe Calabi-Yau manifold turns in her hands" />

THME is also a research programme into heterogeneous representations for language models.

- [Research Programme v2](docs/research/THME_Research_Programme_v2.pdf): hypotheses, method, controls
- [Measured findings, November 2026](docs/research/THME_measured_findings_2026-11.md): what the first experiments support and refute
- [Research Dossier v3.1](docs/research/THME_Research_Dossier_v3.1_2026-11-13.pdf) (theoretical proposal and engineering brief; no measurements of its own) and its [review against the repository](docs/research/REVIEW_dossier_v3.1.md)

## Contributing

Read [AGENTS.md](AGENTS.md) (rules for humans and AI agents), [CONTRIBUTING.md](CONTRIBUTING.md)
and the [production workflow](docs/WORKFLOW.md) (backlog, pull requests, CI gates, releases).
Planned work lives in [BACKLOG.md](BACKLOG.md).
Before any change: `cargo fmt --check`, `cargo clippy --all-targets -- -D warnings`, `cargo test`.
New benchmark numbers must come from a committed report under `docs/benchmarks/`.

Security: [SECURITY.md](SECURITY.md) · Conduct: [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) ·
License: [AGPL-3.0](LICENSE). A commercial license is planned for a later release ([LICENSING.md](LICENSING.md)).
