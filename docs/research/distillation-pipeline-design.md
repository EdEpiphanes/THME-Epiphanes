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
| PyTorch and PEFT environment as an external trainer | `<venvs>/thme-train` | Installing |

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

## Scope: functionality testing only

Owner decision, 27 November 2026: distillation done with a hosted teacher (Claude) exists to test that
the engine works. A model taught this way is a test artifact. It is not released to the public and not
used commercially, and the goal is a solid multi-feature engine, not a product model.

What that means in practice:
- Datasets, adapters and `.epiph` files made with a hosted teacher stay on the machine that made them.
  The repository ignores the `distill/` folder, and no such file goes into a commit, a release archive,
  a benchmark folder or a published page. Reports may state numbers and method, never the teacher's
  outputs.
- Every such file carries the annotation `use=test-only` and the teacher's identity, so a later
  reader cannot mistake it for a releasable model.
- A model meant for release or commercial use must be taught by a teacher whose terms allow it
  (an open-weight teacher with a permissive licence, or the source model itself). Turning a test
  artifact into a release is a new decision for the owner and counsel, not a flag to flip.
- This narrows the terms question below without removing it: the provider's terms may still apply to
  testing, which counsel can confirm when convenient. It does not block the engine work.

## Permission to release (owner statement, 27 November 2026)

Using a hosted model such as Claude to teach another model for release to the public or for commercial
use requires permission from that model's provider. The person who plans the release must obtain and
hold it; THME cannot grant it and does not assume it. That is why the work here is limited to testing
that the engine works and to proving the claims THME makes.

The engine should enforce this, not just document it. Planned guard (deferred by the owner on 27 November 2026; to be installed later, not built):
- A file annotated `use=test-only` cannot be signed (`thme epiph sign`), exported to GGUF or
  SafeTensors for sharing, or packed into a release archive; the command refuses and says why.
- The way past it is an explicit `--permission-note TEXT` that is written into the file's provenance
  with the date, so the claim of permission is visible and can be audited. THME records the note; it
  does not check it.
- Models taught only by open-weight teachers with permissive licences, or by the source model, carry no
  such annotation.

## Teachers and subjects

The student goes to class: it has several subjects, and each subject can have its own teacher.

- **First teacher: Claude.** Decided by the owner on 27 November 2026. Claude answers the prompts of a
  subject, either through a command the teacher bridge runs (for example the `claude` command line in
  print mode, with a timeout) or through the session that drives THME over MCP. Claude gives text, not
  probabilities, so Claude-taught subjects use sequence-level distillation.
- **Several teachers later.** A teacher registry file lists each teacher: an id, its kind (command,
  open-weight model THME runs, mock for tests), the subjects it teaches, a timeout, and a terms note.
- **A curriculum.** A list of subjects. Each subject has its own seed prompts, its own teacher, its own
  mixing weight, and its own held-out evaluation set. Dataset records carry the subject and the teacher.
- **Report per subject.** The gate result is reported for each subject separately and for the mix, so a
  teacher or a subject that does not help is visible instead of averaged away.

Open question for the owner and counsel, recorded rather than assumed: the usage terms of the teacher's
provider may restrict training other models on its outputs. The pipeline will not run a hosted teacher
unless its terms note is filled in. The first slice below does not need a hosted teacher.

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
