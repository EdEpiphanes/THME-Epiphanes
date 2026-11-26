# Notebooks and Python

THME has no Python package yet. A notebook drives it through the `thme` command line: run the
command with `--json`, read stdout with `json.loads`. Nothing else is needed.

Start with [examples/notebooks/quickstart.ipynb](../examples/notebooks/quickstart.ipynb). It runs
`thme doctor --json`, `thme discover --json` and `thme inspect` on a path you fill in. The notebook
is saved without outputs, and `scripts/check_notebook.py` checks that in CI.

```python
import json
import subprocess

run = subprocess.run(["thme", "doctor", "--json"], capture_output=True, text=True, check=True)
facts = json.loads(run.stdout)
```

Rules that hold for every command below:

- With `--json`, stdout holds the JSON object and nothing else. Progress text and warnings are not
  mixed in.
- A failed command exits with a non-zero code and prints `Error: ...` on stderr. `check=True` turns
  that into an exception, and `exc.stderr` has the message.
- `thme` must be on your PATH. See [getting-started.md](getting-started.md) for installing it.
- The keys below are the top-level keys. Nested fields are listed only where a notebook needs them.

## Commands that print JSON

| Command | Keys at the top level | Checked by |
| :--- | :--- | :--- |
| `thme inspect MODEL --json` | `name`, `architecture`, `num_parameters`, `original_byte_size`, `num_layers`, `hidden_size`, `num_attention_heads`, `num_key_value_heads`, `head_dim`, `vocab_size`, `max_position_embeddings`, `tensors` (list; each has `name`, `role`, `shape`, `original_dtype`, `byte_size`, `layer_index`), `norm_eps`, `rope_theta`, `rope_scaling_type`, `rope_scaling_factor` | run on a small SafeTensors folder, and `tests/test_notebook_json.rs` |
| `thme profile MODEL --json` | `model`, `layers` (list; `layer_index`, `average_sensitivity`, `recommended_format`, `total_bytes`, `total_parameters`), `kv_cache` (list; `context_length`, `batch_size`, `bytes_fp16`, `bytes_fp8`, `bytes_q4`) | run on a small SafeTensors folder, and `tests/test_notebook_json.rs` |
| `thme doctor --json` | `os`, `arch`, `memory_total_bytes`, `memory_available_bytes`, `disk_free_bytes`, `cpu_threads`, `gpu_adapter`, `models` (`total`, `shown`), `next` (list of text lines) | run |
| `thme discover --json` | `models` (list; `name`, `source`, `format`, `size_bytes`, `path`), `warnings` (list of text) | run; `--path DIR` adds a folder, and the flag can repeat |
| `thme telemetry --json` | `timestamp_epoch_secs`, `cpu`, `gpus` | run. With `--watch`, one compact JSON object per line |
| `thme validate --json` | depends on the mode, see below | run for two modes |
| `thme convert --to PARADIGM --json` | `source_paradigm`, `target_paradigm`, `original_parameters`, `new_parameters`, `parameter_delta_pct`, `original_byte_size`, `new_byte_size`, `storage_savings_pct`, `new_tensors_added`, `tensors_restructured`, `description` | run on the built-in synthetic baseline (no model argument) |
| `thme speculate --json` | `provenance`, `synthesis`, `telemetry`, `tree_speculation` (null unless `--tree`) | run with default options |
| `thme roofline --json` | `hardware_name`, `precision`, `peak_compute_tflops`, `memory_bandwidth_gbps`, `knee_intensity_flop_per_byte`, `points` (list) | run with default options |
| `thme forward MODEL --json` | `model`, `architecture`, `device`, `vocab_size`, `tokens`, `top1_token_id`, `top1_logit`, `limit`, `logits` | read from the code; not run, because it needs a model with all weights |
| `thme co-surgery MODEL --json` | `total_iterations`, `successful_commits`, `total_rollbacks`, `total_relaxations`, `final_state`, `baseline_ppl`, `final_ppl`, `original_bytes`, `final_bytes`, `final_reduction_ratio`, `history` (list), `quality_source` | read from the code; not run |
| `thme tokenize MODEL_DIR TEXT --json` | `ids`, `decoded` | read from the code; not run, because it needs a `tokenizer.json` |

`thme validate --json` has three shapes, one per input mode:

- `--baseline S --candidate S` (benchmark scores): `decision`, `degradation`,
  `max_allowed_degradation`, `task_evaluations`, `summary`. Run.
- `--baseline-ppl P --candidate-ppl P` (perplexities): `decision`, `baseline_ppl`,
  `candidate_ppl`, `delta_ppl`, `max_allowed_delta`, `degradation_pct`, `summary`,
  `recommended_action`. Run.
- `--report FILE` (a measured report): `source`, `gate`, `variants`, `passed`. From the code; not run.

`thme inspect` and `thme profile` refuse a model whose safety scan finds high or critical issues,
unless you add the global `--allow-unsafe-model` flag. The error says why.

## Commands without `--json`

These print text only. Where a command writes a JSON file, the notebook can read that file with
`json.load` after the command runs.

| Command | What it does with JSON |
| :--- | :--- |
| `thme gui` | Opens the Studio (no output) |
| `thme plan` | Writes `optimization_plan.json` by default (`-o` to change it) |
| `thme evaluate` | Writes a measured report to `--report` (default `target/evaluate_report.json`) |
| `thme search` | Writes a measured report to `--report` (default `target/search_report.json`) |
| `thme calibrate` | Writes statistics to `--out` (default `target/calibration.json`) |
| `thme tweak` | Text only; `-o` saves a plan file |
| `thme report` | Text, or Markdown with `--markdown` |
| `thme export` | Writes a model file (GGUF or SafeTensors); `--checksum` prints a SHA-256 |
| `thme epiph` | Text; `pack`, `verify`, `sign` and `extract` work on `.epiph` files |
| `thme scan PATH` | Text; `--report FILE` also writes the findings as JSON |
| `thme bench` | Text; subcommands `list`, `validate`, `verify-dataset`, `score`, `chart`, `contamination`, `gate` |
| `thme figure` | Writes SVG (and GIF with `--gif`) files |
| `thme interop` | Writes a training script and an optional config.json |
| `thme mcp` | A JSON-RPC server on stdin and stdout for editors and agents, not a `--json` flag |
| `thme chat-template MODEL_DIR --messages FILE` | Text: the rendered prompt |

If a notebook needs JSON from one of these, the clean route is a request for `--json` on that
command. Do not parse its text output; the text format is for people and can change.

## Adding a command to this list

1. Give the command a `--json` flag that prints one `serde_json` object to stdout.
2. Add a test that runs the binary, parses stdout and checks the keys (see
   `tests/test_notebook_json.rs`).
3. Add a row here with the keys you checked.
