# Teacher-to-student distillation pipeline: design

27 November 2026. Design only: nothing here is built or measured except where a section says so.

## Goal

A frontier model (the teacher) trains a smaller open-weight model (the student) quickly, and THME
measures the result with its own gate. The aim is to cut the time and cost of getting a small model
to do one job well, and to ship the result as a single verifiable `.epiph` file with its provenance.

The page that prompted this (the "Knowledge Distiller" skill listing on mcpmarket.com) is a
text-summarising skill and contains no model training. It is not a source for this design.

## What THME already has

| Piece | Where | State |
| :--- | :--- | :--- |
| Teacher bridge: run an external command or CLI agent with a timeout, mock for tests | `src/surgery/agent_bridge.rs` | Tested; used by co-surgery only |
| MCP server so an agent can drive THME tools | `src/cli/mcp.rs` | Tested |
| Measured quality gate with a pre-registered decision rule | `src/quality/`, `docs/quality-gate.md` | Measured on two small models |
| Perplexity and token-by-token evaluation | `thme evaluate`, `quality::kv_perplexity` | Measured |
| `.epiph` with recipe, provenance, quality entries, attachments | `src/epiph/` | Tested; compression measured |
| LoRA merge | `src/transform/` | Tested |
| Real backward pass and LoRA training | `agent/v02-train` branch | In progress, not merged |
| PyTorch and PEFT environment as an external trainer | `C:\Users\TechTroniX\venvs\thme-train` | Installing |

## The pipeline

1. **Task.** A short description of what the student must do, plus a seed set of prompts (written by
   the user, or proposed by the teacher and reviewed). A held-out evaluation set is split off first
   and its hash recorded; nothing from it ever reaches training (ROADMAP v0.2.3.8).
2. **Teacher outputs.** One of three modes, chosen by what the teacher can give:
   - *Text only* (a hosted frontier model through a command, an API wrapper or MCP): the teacher
     answers each prompt; the student learns from those answers (sequence-level distillation).
   - *Logits* (an open-weight teacher THME can run): the student is trained to match the teacher's
     token probabilities (logit distillation), which carries more signal per example.
   - *Quantization-aware* (teacher = the unquantized source model): the student is the quantized
     version of the same model and learns to match the source's logits (ROADMAP v0.2.3.2).
3. **Dataset.** Prompt, teacher output, teacher identity and version, date, hashes. Cached and
   resumable, so a failed call is not paid for twice.
4. **Training.** LoRA on the student: the in-Rust backward pass when it is merged, or PyTorch with
   PEFT as the bridge for larger models and the GPU. Cost accounting is recorded: tokens, wall time,
   and energy only where measurable (v0.2.3.7).
5. **Evaluation.** THME's gate on the held-out set, against the untuned student and against the
   teacher's own outputs where that makes sense. The decision rule is written down before the run.
6. **Package.** An `.epiph` holding the student, the adapter, the recipe, the teacher identity, the
   dataset hash and the quality entries with their evidence labels.

## Where the speed comes from

Small student; LoRA instead of full fine-tuning; short sequences; cached teacher outputs; the GPU
through the PyTorch bridge for anything above a few hundred million parameters; the quantized student
trained directly when the target is a quantized file. None of these figures is measured yet.

## Decisions that are not engineering

- **Terms of the teacher.** Some hosted model providers restrict using their outputs to train other
  models. Which teachers are allowed is a licensing decision for the owner and counsel. The pipeline
  records the teacher and its terms note in every dataset and refuses to run without one. The
  default allowed teachers are open-weight models whose licences permit it.
- **Keys and cost.** Calling a hosted teacher needs a key and costs money. THME never stores a key
  and never calls a hosted teacher unless the user configured the command and asked for the run.
- **Data.** Prompts from the user may be private. The dataset file is local; nothing is uploaded.

## First slice

Quantization-aware distillation on SmolLM-135M with the source model as teacher (no hosted teacher,
no terms question). Measure perplexity against the Q4_0 baseline, round-to-nearest 23.70 and AWQ 22.04
already in the repository, using the pre-registered rule style of the GPTQ comparison.
