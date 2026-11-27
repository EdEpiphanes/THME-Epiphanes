# THME Domain Glossary

THME (Topological Heterogeneous Manifold Engine) loads language models, modifies tensor formats zero-copy, and measures computational cost and representation quality.

## Evidence

**Measured**:
A figure produced by running THME's native forward pass or reading a written artifact, regenerated from a committed report.
_Avoid_: benchmarked, real (without specifying measurement mode)

**Tested**:
A claim checked by an automated test against an independent oracle outside the code under test on synthetic data.
_Avoid_: verified, validated

**Estimate**:
A figure derived from a stated formula or analytic model, never from an empirical execution run.
_Avoid_: projection presented as a result

**Simulated**:
A figure produced from a synthetic simulation; carries no empirical weight on a trained model.

**Oracle**:
An answer known independently of the code under test: an analytic solution, an exhaustive search, a published figure, or a physical conservation law.

## Models and Formats

**Source model**:
The model file exactly as stored on disk, before any transformation or surgery.
_Avoid_: original, base model

**Candidate**:
A model representation after a recipe or allocation is applied, awaiting quality gate decision.
_Avoid_: student (reserved exclusively for distillation)

**Stored format**:
The numerical representation format a tensor holds on disk.

**Selected format**:
The numerical representation format assigned to a tensor for a candidate model.
_Avoid_: target format, quantization level

**Recipe**:
A versioned, declarative JSON specification of operations transforming a source model into a candidate.
_Avoid_: config, plan (plan is an allocator output)

**Algorithm**:
A bring-your-own-algorithm definition: unique identity, metadata, and one transformation rule applied to tensors as data.
_Avoid_: plugin (reserved for out-of-process extensions)

**Rule**:
The numerical transformation an algorithm executes on tensor weights under a formal contract.

**Epiphanes model file**:
A single `.epiph` container holding model weights, architecture descriptors, and digests for each payload.
_Avoid_: epi file, THME file, checkpoint

**Calibration statistics**:
Per-input-channel activation distributions collected by running the source model over calibration text.
_Avoid_: calibration data, raw activations

## Allocation and Metrics

**Nominal bits per weight**:
Arithmetic average of format bit-widths across all model parameters, with unmodified tensors counted at stored width.
_Avoid_: normal bits, compression ratio

**Measured bits per weight**:
Total artifact size in bits divided by parameter count, including quantization scales and block overheads.
_Avoid_: bare BPW

**Allocator**:
Subsystem that assigns selected formats to tensor groups under a nominal size budget.

**Budget**:
Target nominal size of linear tensors, expressed as a fraction of their stored size.

**Cost table**:
Measured quality degradation for each tensor group across candidate formats, varying one group at a time.

**Solver**:
Algorithmic technique an allocator uses: sensitivity thresholds, greedy cost-benefit, or exact dynamic programming over the cost table.
_Avoid_: optimizer (implies universal optimality)

## Quality and Verification

**Quality gate**:
A versioned set of criteria that a candidate must satisfy against the source model to be accepted.
_Avoid_: bare gate, threshold

**Criterion**:
A single metric within a quality gate specifying allowable absolute or relative degradation margin.
_Avoid_: tolerance, gate value

**Decision**:
The three-state outcome of a quality gate evaluation: pass, fail, or inconclusive (when intervals straddle margins).
_Avoid_: verdict

**Benchmark task**:
A declarative evaluation file naming a dataset by hash, split, license, prompt template, extraction rule, and metric.
_Avoid_: eval, test

## Security

**Untrusted code**:
Code supplied with external models or plugins that has not been audited; constrained to execute inside the sandbox.
_Avoid_: remote code, general payload

**Sandbox**:
An isolated process with no network access, write bounds restricted to a transient folder, and hard limits on execution time, RAM, and processes.
_Avoid_: jail, generic container

**Escape test**:
A security validation test attempting forbidden host actions from within the sandbox and asserting execution failure.

## Run planning and storage

**Predicted**:
An evidence label for a figure computed by arithmetic from a model header (for example the memory plan from `thme pathfinder`), not measured; nothing was run or measured.

**Run plan**:
The settings for one run (context length, KV cache precision, weight width) with the predicted weight, KV cache, and scratch bytes they need, from `thme pathfinder`.

**Shrink order**:
The order in which the planner lowers settings to fit a byte budget: KV cache precision first, then context length (never below the floor, never raised), then weight width. A setting the user fixed is never changed, and if nothing fits the planner reports what it tried.

**KV cache precision**:
How stored keys and values are encoded (f32, int8, or int4), chosen with `thme chat --kv-precision`. Both int8 and int4 use groups of 32 values with one f32 scale per group.

**llama.cpp layout**:
The GGUF form llama.cpp reads, written by `thme export --llama-cpp-layout`: GGML tensor names, norm weights as F32, q and k rows in interleaved-pair order, a rope dimension key, and the tokenizer. THME's own reader undoes the row order for llama-architecture files.

**Switch point**:
In quality gate v2, the baseline perplexity (20.0) below which only the absolute margin applies and at or above which only the relative margin applies.

**Fallback norm epsilon**:
The RMS-norm epsilon (1e-6) assumed, and written into exported GGUF files, when a model records none.
