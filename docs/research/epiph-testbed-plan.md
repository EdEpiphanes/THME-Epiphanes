<!-- SPDX-License-Identifier: AGPL-3.0-only -->
# A small .epiph testbed for the Epiphanes math

27 November 2026. Owner decision: build a very small model that lives as `.epiph` from the start and use it to test
every formula and parameter of the research paper (`docs/research/paper/THME_Epiphanes_Paper_v5.1.md`), find the
parameters the paper leaves open, and derive extra formulas from trial and error. Claude trains and teaches it.
Ticket: THME-3110.

## Why a model we train ourselves

The measurements so far use two pretrained models whose training history we do not know. A model we train has a
known corpus, seed, step count and checkpoint series, so a change in the gate, allocator or drift estimate can be
tied to a cause. It also gives something no pretrained model gives: the same weights at many points in training.

Expectation, stated first: a 14M-parameter model trained on WikiText-103 will be much worse than the pretrained
Pythia-14M (trained on the Pile). The testbed is for testing the math, not for quality.

## Two arms

| Arm | What it is | Why |
| :--- | :--- | :--- |
| **A: scratch** | Pythia-14M's shape (6 layers, width 128, 4 heads, GPT-NeoX blocks), random initialisation with a fixed seed, trained by us | Fully ours; THME's existing GPT-NeoX forward pass runs it unchanged |
| **B: Pythia start** | The same shape, initialised from the published Pythia-14M weights (Apache-2.0), trained on the same data | Control: separates the effect of the math from the effect of the data history |

A third arm (own architecture, sized by the paper's parameters) waits until A and B show which parameters matter.

## Data, in two stages

1. **Stage 1: WikiText-103 train** (CC BY-SA 3.0, text only). WikiText-2 test stays the held-out evaluation set, as in every
   committed report; the training windows are checked disjoint from it by hash.
2. **Stage 2: lessons written by Claude** (test-only, local). Short, structured texts on chosen topics, generated in
   batches, hashed and kept under the ignored `distill/` folder. They never leave the machine; any release of a model
   taught this way needs the provider's permission (guard deferred, see `docs/research/distillation-pipeline-design.md`).

## What gets trained and stored

- PyTorch trainer from v0.2.3 (`scripts/train/`), one job on the GPU at a time (RTX 4060 Ti, 16 GB; 14M parameters fit
  with room to spare). Seeds, data hashes, learning rates and step counts are written to a report before the run.
- A checkpoint every fixed number of steps, each converted to **`.epiph`** (`thme epiph pack`) with provenance and the
  recipe field empty. The series is the testbed's main asset.
- Each checkpoint is evaluated by `thme evaluate` (WikiText-2 test) and cross-checked against Hugging Face on a few
  prompts, so the engine and the trainer agree before any claim is made.

## What it is used to test

Each item is pre-registered (decision rule committed before the run), measured on the GPU with the drift protocol
(`docs/research/measurement-drift-protocol.md`), and written into the findings file below.

| Paper part | Question the testbed answers | Needs |
| :--- | :--- | :--- |
| Gate, section 3.3 and 3.6 | Do the e-process confidence sequences keep their error rate when the candidate is a known-degraded checkpoint? Calibrate `r_max`, the switch point and the declared range `M` | Paired per-block losses (THME-3109) |
| Block partition, section 3.2 | Is the autocorrelation-separated block length stable across checkpoints? | Per-block losses |
| PS-16 bands, section 3.8 | Run the full relative-accuracy statistic, bias, variance, correlation and three levels on checkpoint pairs; decide the absolute limit | Per-block losses |
| Allocator | Calibrate the link constant C (THME-3100); compare exact and greedy allocation on real cost tables | Cost tables per checkpoint |
| Depth probe and exit heads | Settling depth as training proceeds; exit heads trained per depth | Checkpoint series |
| Recovery, v0.2.3 | QAD-LoRA on a model whose training data we know | Stage 1 model |
| Rotations and KV precision | Why Hadamard failed on both pretrained models; whether outlier structure appears during training | Checkpoint series |
| Teaching | Does stage 2 change held-out loss, and by how much against noise? | Stage 2 lessons |

## Findings file and new formulas

`docs/research/epiph-testbed-findings.md` (created with the first result) holds one entry per experiment: the
pre-registered rule, the committed report, the outcome, and any parameter or formula the result suggests. A formula
suggested by data is labelled **Proposed** until a second, separately registered experiment confirms it; only then does
it move into the paper as a new version.

## Order of work

1. Fetch Pythia-14M and WikiText-103 into untracked folders; record hashes and licences in `docs/third-party.md`.
2. Trainer config for the shared shape; a short smoke run; HF versus THME cross-check on the smoke checkpoint.
3. Arm A and arm B stage 1, checkpoint series as `.epiph`, committed evaluation reports.
4. Paired per-block losses from `thme evaluate` (the missing piece for the gate and PS-16 work).
5. The experiments in the table, one pre-registration at a time.
6. Stage 2 lessons and the teaching experiment.

## Limits

One tiny model family, one tokenizer (Pythia's), one evaluation corpus. Results say how the math behaves here; a claim
about larger models needs a larger testbed.
