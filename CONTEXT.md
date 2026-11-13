# THME

THME loads language models, changes how their weights are stored or computed, and measures what
each change costs. These are the words the project uses for that work.

## Evidence

**Measured**:
A figure produced by running THME's own forward pass or reading a written artifact, regenerated from a committed report.
_Avoid_: benchmarked, real (without saying how)

**Tested**:
A claim checked by an automated test against an oracle outside the code under test, on synthetic data.
_Avoid_: verified, validated

**Estimate**:
A figure from a stated formula or analytic model, never from a run.
_Avoid_: projection presented as a result

**Simulated**:
A figure from a toy or synthetic simulation; says nothing about a trained model.

**Oracle**:
An answer known independently of the code under test: an analytic solution, an exhaustive search, a published value, or a conservation law.

## Models and formats

**Source model**:
The model exactly as stored on disk, before any change.
_Avoid_: original, base (when it means the source)

**Candidate**:
A model after a recipe or allocation is applied, awaiting the quality gate.
_Avoid_: student (reserved for distillation)

**Stored format**:
The number format a tensor has on disk.

**Selected format**:
The number format a tensor is assigned for a candidate.
_Avoid_: target format, quantization level

**Recipe**:
A versioned, declarative list of operations that turns a source model into a candidate.
_Avoid_: config, plan (a plan is an allocator's output)

**Algorithm**:
A bring-your-own-algorithm definition: identity, metadata and one rule, applied to tensors as data.
_Avoid_: plugin (reserved for out-of-process extensions)

**Rule**:
The transformation an algorithm performs on a tensor's weights, with a weight contract.

**Epiphanes model file**:
A single `.epiph` file holding a model's weights, its description and everything known about it, each payload with a digest.
_Avoid_: epi file, THME file, checkpoint (for this format)

**Calibration statistics**:
Per-input-channel activation statistics of each linear tensor, collected by running the source model over calibration text.
_Avoid_: calibration data, activations

## Size

**Nominal bits per weight**:
Format bit-widths averaged over every parameter of the model, with unchanged tensors counted at their stored width; arithmetic, not a size.
_Avoid_: normal bits, bits per weight (unqualified), compression

**Measured bits per weight**:
A written artifact's size in bits divided by the parameter count, including block scales.
_Avoid_: BPW (unqualified)

## Quality

**Quality gate**:
A versioned set of one or more criteria that a candidate must meet against its source model to be accepted.
_Avoid_: gate (unqualified), threshold

**Criterion**:
One metric of a quality gate and the degradation it allows, as an absolute and/or a relative margin.
_Avoid_: tolerance, gate value

**Decision**:
The outcome of a quality gate: pass, fail, or inconclusive when a criterion's degradation interval straddles its margin.
_Avoid_: verdict (for the three-way outcome), result

**Benchmark task**:
A declarative file naming a dataset by hash, its split and licence, a prompt template, an answer-extraction rule and a metric; identified by `id@vN`.
_Avoid_: eval, test (for this)

**Result record**:
The outcome of one benchmark task on one model, with per-item scores, an interval, the run mode and the evidence status.
_Avoid_: score file, eval result

**Run mode**:
How a result record's numbers were obtained: native (THME's forward pass), bridged (an external tool) or imported (supplied by a user, labelled imported, never measured).

**Gate unit**:
A gating function inside a model's weights, such as the gate of a gated feed-forward layer.
_Avoid_: gate (unqualified)

**Router**:
A function inside or attached to a model that chooses among computation paths.

**Needle**:
A very small model attached to a frozen model to make one routing decision.

## Allocation

**Allocator**:
Anything that assigns a selected format to each tensor group under a size budget.

**Budget**:
The target nominal size of the linear tensors, as a fraction of their stored size.

**Cost table**:
The measured quality loss of each tensor group in each candidate format, one group changed at a time.

**Solver**:
The method an allocator uses to choose: sensitivity thresholds, greedy by saving per unit cost, or an exact dynamic program over the cost table.
_Avoid_: optimizer (implies the result is optimal)

## Security

**Untrusted code**:
Code that arrives with a model, benchmark or plugin and that nobody has reviewed; it runs only in the sandbox, and only when the user chooses to run it.
_Avoid_: remote code (for the general case), payload

**Sandbox**:
An isolated process with no network, write access only to its work folder, and caps on time, memory, processes and output; it refuses to start when a layer cannot be enforced.
_Avoid_: jail, container (unless it is one)

**Escape test**:
A test that runs a probe inside the sandbox, attempts one forbidden action and asserts that it fails.

## Engine

**Engine primitive**:
A numerical building block in the engine with a stated interface, explicit parameters and an oracle test.
_Avoid_: helper, util
