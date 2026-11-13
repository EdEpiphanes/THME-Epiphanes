# Benchmark registry

A benchmark in THME is **data**: a task file says what to run and how to score it, and a result
record says what came out, with enough detail to compute intervals and paired comparisons later
without running again. The registry reads and checks both. Running tasks is the benchmark runner
(v0.2.8.3); this page covers the formats.

```bash
thme bench list benchmarks/tasks            # every task in a folder
thme bench verify-dataset benchmarks/tasks/gsm8k.json
thme bench validate run.json --task benchmarks/tasks/gsm8k.json
```

## A task file

```json
{
  "schema": "thme-bench-task/v1",
  "id": "gsm8k-main",
  "version": 1,
  "family": "checkable-answer",
  "dataset": {
    "path": "data/gsm8k-test.jsonl",
    "sha256": "<64 hex characters>",
    "split": "test",
    "licence": "MIT"
  },
  "prompt_template": "Question: {input}\nAnswer:",
  "extract": { "kind": "after-marker", "marker": "####" },
  "metric": { "kind": "numeric-tolerance", "abs": 0.0 }
}
```

- **`id@vN` is the identity.** Changing anything that affects scores means a new version; two
  files with the same `id@vN` refuse the whole folder.
- **The dataset is pinned by hash.** `path` is relative to the task file, uses `/`, and cannot
  leave the task's folder, so task files are portable and carry no machine paths.
- **The licence is required.** It is copied into every result record.
- **Strict.** Unknown or missing fields are refused, never guessed.

The dataset is JSONL, one item per line: a string `id` (unique), a string `input`, an
optional string `target`, and optional `choices` (a list of strings). For likelihood tasks the
target is the index of the correct choice.

| Family | Extract | Metric |
| :--- | :--- | :--- |
| `multiple-choice` | `none` | `choice-by-likelihood` |
| `multiple-choice` | `choice-letter`, `exact`, `first-line` | `exact-match`, `normalized-match` |
| `checkable-answer` | `exact`, `first-line`, `last-number`, `after-marker` | `exact-match`, `normalized-match`, `numeric-tolerance` |
| `code` | `none` | `pass-at-k` (`k`, `samples`); accepted by the registry, but running it needs the [sandbox](sandbox.md), which is reserved for the commercial version |

Other combinations are refused. Every metric gives each item a score in [0, 1]. Perplexity is
not a registry task: it stays with `thme evaluate` and its measured reports.

## A result record

Every field of dossier Table R2:

| Field | Purpose |
| :--- | :--- |
| `benchmark`: id, version, task hash, dataset hash, split, licence | Which test exactly |
| `model`: hash, transformation | Which model and which conversion |
| `harness`: version, run mode, prompt-template hash | How it was run |
| `decoding`: temperature, seed, max new tokens | Reproducibility; sampling needs a seed |
| `items`: id, output, score per item | Intervals and paired tests |
| `score`, `interval` (low, high, method, level) | The result with its uncertainty |
| `evidence` | Measured, imported, estimate or simulated |
| `hardware`, `run_date`, `duration_s` | Context |

Checks on every record:

- `score` is the mean of the item scores; item ids are unique; the interval lies in [0, 1] and
  contains the score.
- **Run mode and evidence agree.** Numbers supplied by a user (`imported` run mode) are labelled
  `imported`, never `measured`; only an imported run can be labelled imported.
- With `--task`, the record's task hash, dataset hash, split, licence and prompt-template hash
  must equal the task's, so a record cannot be filed under a task it did not run.

## Scoring outputs from any tool (imported mode)

```bash
thme bench score --task T.json --outputs outputs.jsonl --model-sha256 HASH --out record.json
```

The outputs file has one JSONL line per dataset item:

| Task metric | Line |
| :--- | :--- |
| Generated answer (match, numeric tolerance) | `{"id": "q1", "output": "... #### 72"}` |
| Likelihood choice | `{"id": "q1", "log_likelihoods": [-3.1, -0.4, -2.2]}` |
| pass@k | `{"id": "p1", "samples": 6, "passed": 2}` (tests run elsewhere; THME runs no code) |

THME extracts and scores each output with the task's own rules, computes the interval (Wilson
for 0/1 scores, bootstrap otherwise) and writes a complete result record. Its run mode and
evidence are `imported`: THME scored the outputs, it did not produce them. Every dataset item
needs exactly one output; a partial, duplicated or mismatched file is refused rather than scored
on a subset. Decoding settings and hardware are recorded as you state them (`--temperature`,
`--seed`, `--max-new-tokens`, `--hardware`). Native and bridged modes, which produce the
outputs, wait until model runs resume.

## Charts

```bash
thme bench chart --record base.json --record q4.json --baseline base.json --out charts/
```

Writes light and dark SVG through the visual engine (`engine::viz`): scores with their
intervals (when all records are on one benchmark), a model-by-benchmark score heatmap, and, with
`--baseline`, each record's change against the baseline with a paired-bootstrap interval and a
zero line. Every chart prints the records' evidence status. Paired changes need comparable
records, the same rule as `thme bench gate`.

## Contamination check

A score is only clean if its items were not in the model's training or teacher-written data
(dossier WP36, Q.6). Check before scoring:

```bash
thme bench contamination --task benchmarks/tasks/gsm8k.json --corpus train-part-1.txt --corpus teacher.jsonl
```

Items and corpora are compared as 13-word n-grams after lower-casing and dropping punctuation
(`--n` changes the length; an item shorter than that is matched whole). Corpora are streamed, so
they can be larger than memory. The command lists every item with an overlap, the share of its
n-grams found and the files they were found in, writes the list with `--report`, and exits with
an error when any item overlaps. It finds verbatim and lightly reformatted copies, not
paraphrases.

## Not yet

The metrics are in `thme::bench::metrics` (v0.2.8.4) and the intervals and tests in
`thme::bench::stats` (v0.2.8.5; `interval_for` fills a record's `interval`). Running tasks
(v0.2.8.3) builds on these, and the three-state gate (v0.2.8.6) judges two records with
`thme bench gate`. JSON Schemas: [bench-task-v1](schemas/bench-task-v1.json) and
[bench-record-v1](schemas/bench-record-v1.json), each with an example under `schemas/examples/`.
