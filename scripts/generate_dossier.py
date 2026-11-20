#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""
THME Benchmark Dossier Generator (Milestone v0.2.0.4)

Renders the machine-readable report written by the verification pipeline
(`examples/production_verification.rs`) into Markdown and CSV.

Every figure in the dossier is read from a report produced by a real forward
pass. This script contains no benchmark numbers of its own: if a report is
missing or holds no measured evaluation, generation fails instead of emitting
placeholder values.
"""

import argparse
import glob
import csv
import json
import math
import os
import statistics
import sys
import time


def load_reports(paths):
    reports = []
    for path in paths:
        if not os.path.exists(path):
            raise SystemExit(
                f"error: report not found: {path}\n"
                "Run the verification pipeline with --model-path and --tokens first."
            )
        with open(path, "r", encoding="utf-8") as f:
            report = json.load(f)
        if not report.get("evaluation", {}).get("measured"):
            raise SystemExit(
                f"error: {path} contains no measured evaluation; refusing to generate a dossier from it."
            )
        reports.append(report)
    return reports


def mib(n):
    return f"{n / (1024 * 1024):.1f} MiB"


def verdict(variant):
    if variant["accepted"] is None:
        return "baseline"
    return "PASSED [ACCEPT]" if variant["accepted"] else "FAILED [ROLLBACK]"


def sweep_section(sweep, heading):
    """Renders the measured precision sweep (examples/precision_sweep.rs)."""
    rows = sweep["variants"]
    by_group = {}
    for row in rows:
        by_group.setdefault(row["group"], []).append(row)
    base = by_group["baseline"][0]
    lines = [
        heading,
        "",
        f"Model: {sweep['model_name']}. Each row is a separate real forward pass over the first "
        f"{sweep['corpus_tokens_used']:,} corpus tokens (context {sweep['context_len']}), so absolute values differ "
        "from the full-split figures in section 1. Tensors a row does not name keep their stored values. "
        "No retraining or calibration is applied. Bits per weight are nominal arithmetic, not artifact sizes: format bit-widths averaged over every parameter, with tensors a row does not change counted at their stored width (32 bits for FP32 embeddings), so the figure is a whole-model average, not the bits of the changed tensors.",
        "",
        f"Baseline perplexity on this subset: {base['perplexity']:.4f}",
        "",
    ]

    def table(title, group_rows):
        out = [
            f"**{title}**",
            "",
            "| Variant | Perplexity | Δ PPL | Parameters changed | Nominal bits/weight |",
            "| :--- | ---: | ---: | ---: | ---: |",
        ]
        for r in group_rows:
            out.append(
                f"| {r['description']} | {fmt_ppl(r['perplexity'])} | {r['delta_ppl']:+,.4f} | "
                f"{100 * r['targeted_param_fraction']:.1f}% | {r['nominal_bits_per_weight']:.2f} |"
            )
        out.append("")
        return out

    if "uniform" in by_group:
        lines += table("Uniform formats", by_group["uniform"])
    if "role" in by_group:
        lines += table("One tensor role ternarized at a time", sorted(by_group["role"], key=lambda r: r["delta_ppl"]))
    if "recipe" in by_group and "random" in by_group:
        recipe = by_group["recipe"][0]
        randoms = by_group["random"]
        lines += table("Recipe versus random placement with the same tensor counts and shapes", [recipe] + randoms)
        deltas = [r["delta_ppl"] for r in randoms]
        mean = sum(deltas) / len(deltas)
        if recipe["delta_ppl"] < min(deltas):
            outcome = "The recipe is better than every random placement tested."
        elif recipe["delta_ppl"] > max(deltas):
            outcome = "The recipe is worse than every random placement tested."
        else:
            outcome = "The recipe falls inside the range of the random placements tested."
        lines += [
            f"Recipe Δ {recipe['delta_ppl']:+.4f}; random placements Δ mean {mean:+.4f} "
            f"(min {min(deltas):+.4f}, max {max(deltas):+.4f}, n={len(deltas)}). {outcome} "
            f"With n={len(deltas)} this is exploratory, not a statistical claim.",
            "",
        ]
    if "layer" in by_group:
        lines += [
            "**All linear tensors of one layer ternarized at a time**",
            "",
            "| Layer | Perplexity | Δ PPL |",
            "| ---: | ---: | ---: |",
        ]
        for r in by_group["layer"]:
            lines.append(f"| {int(r['id'].rsplit('_', 1)[1])} | {fmt_ppl(r['perplexity'])} | {r['delta_ppl']:+,.4f} |")
        lines.append("")
        uniform = next((r for r in by_group.get("uniform", []) if r["id"] == "uniform_ternary_linear"), None)
        if uniform:
            additive = sum(r["delta_ppl"] for r in by_group["layer"])
            lines += [
                f"Additivity check: the per-layer deltas sum to {additive:+,.1f}, while ternarizing all of those "
                f"layers together measures {uniform['delta_ppl']:+,.1f}. Per-layer effects do not add.",
                "",
            ]
    return lines


def fmt_ppl(value):
    return f"{value:.2e}" if value >= 1e7 else f"{value:,.3f}"


def depth_section(probe, heading):
    """Renders the measured depth probe (examples/depth_probe.rs)."""
    rows = probe["exit_readout"]
    depth = probe["num_layers"]
    lines = [
        heading,
        "",
        f"Model: {probe['model_name']} ({depth} layers), first {probe['corpus_tokens_used']:,} corpus tokens, "
        f"{probe['scored_tokens']:,} scored. Stored weights, no retraining.",
        "",
        "**Readout after each layer.** The model's own final norm and output head are applied to the state at "
        "that depth. The model was not trained to be read early, so this is a diagnostic of the stored weights, "
        "not the behaviour of a model trained with exit heads.",
        "",
        "| Layers applied | Perplexity | Top prediction equals full-depth | Top prediction correct | Settled |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        lines.append(
            f"| {r['layers_applied']} | {fmt_ppl(r['perplexity'])} | {100 * r['top1_agrees_with_final']:.1f}% | "
            f"{100 * r['top1_accuracy']:.1f}% | {100 * r['settled_fraction']:.1f}% |"
        )
    lines += [
        "",
        "\"Settled\" is the share of tokens whose top prediction equals the full-depth top prediction at this "
        "depth and at every later depth.",
        "",
        "**Layer schedule changes.** Layers repeated or skipped, read out at the end.",
        "",
        "| Schedule | Layer applications | Perplexity | Δ PPL |",
        "| :--- | ---: | ---: | ---: |",
    ]
    for r in probe["schedule_experiments"]:
        lines.append(
            f"| {r['description']} | {r['layer_applications']} | {fmt_ppl(r['perplexity'])} | {r['delta_ppl']:+,.3f} |"
        )
    lines.append("")
    return lines


def search_section(report, heading):
    """Renders a measured recipe search (thme search) and its Pareto front."""
    s = report["search"]
    ev = report["evaluation"]
    lines = [
        heading,
        "",
        f"Model: {report['model_name']}. Formats {', '.join(s['formats'])}, grouped by {s['granularity']}. "
        f"Decisions used {s['calibration_tokens']:,} calibration tokens; every figure below is measured on "
        f"{s['validation_tokens']:,} separate validation tokens. Gate: Δ PPL ≤ {ev['max_ppl_delta']}.",
        "",
        f"On calibration, the sum of per-group costs predicted Δ {s['predicted_delta_calibration']:+.4f}; "
        f"the whole candidate measured Δ {s['measured_delta_calibration']:+.4f} "
        f"({s['moves_reverted']} moves reverted by refinement).",
        "",
        "| Budget | Achieved size | Searched Δ PPL | Random placement Δ PPL (min / mean / max) | Gate |",
        "| ---: | ---: | ---: | :--- | :--- |",
    ]
    for p in sorted(s["pareto"], key=lambda p: -p["achieved_fraction"]):
        controls = p["control_deltas"]
        spread = (f"{min(controls):+.3f} / {sum(controls) / len(controls):+.3f} / {max(controls):+.3f}"
                  if controls else "none distinct")
        verdict = "PASSED" if p["validation_delta"] <= ev["max_ppl_delta"] else "FAILED"
        if p["achieved_fraction"] > p["budget"] + 1e-9:
            verdict += " (BUDGET NOT MET)"
        lines.append(
            f"| {100 * p['budget']:.0f}% | {100 * p['achieved_fraction']:.1f}% | {p['validation_delta']:+.4f} | "
            f"{spread} | {verdict} |"
        )
    lines.append("")
    if s.get("test"):
        t = s["test"]
        lines += [f"Held-out test slice ({t['tokens']:,} tokens, never used by the search): Δ {t['delta_ppl']:+.4f}.", ""]
    lines += ["Sizes are nominal (bits per weight of the linear tensors), not measured artifact sizes.", ""]
    return lines

AWQ_SEEDS = 5
AWQ_GATE_DELTA = 0.45
AWQ_RULE_FACTOR = 2.0
AWQ_RESULT_ARMS = ["ceiling", "rtn", "awq", "control_a", "control_b1", "control_b2"]


def awq_stats(values):
    values = [float(v) for v in values]
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "sd": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def awq_is_measured(report):
    """False while the report is pending; refuses a report that is only partly measured."""
    if report.get("status") == "pending":
        return False
    if report.get("status") != "measured":
        raise ValueError(f"awq report status must be 'pending' or 'measured', not {report.get('status')!r}")
    for fmt in report["formats"]:
        for arm in AWQ_RESULT_ARMS:
            if report["results"][fmt][arm].get("status") != "measured":
                raise ValueError(f"awq report is marked measured but {fmt} {arm} is not measured")
    return True


def awq_runs(arm):
    return [r["perplexity"] for r in arm["runs"]]


def awq_evaluate(block, ceiling, rtn):
    """Applies the pre-registered rule (report preregistration.decision_rule) to one format."""
    awq = awq_runs(block["awq"])
    seeded = {"control_a": awq_runs(block["control_a"]), "control_b1": awq_runs(block["control_b1"])}
    if any(len(values) != AWQ_SEEDS for values in [awq, *seeded.values()]):
        raise ValueError(f"the pre-registered rule needs {AWQ_SEEDS} runs in every seeded arm")
    pooled_sd = math.sqrt(statistics.fmean([statistics.variance(v) for v in [awq, *seeded.values()]]))
    mean_awq = statistics.fmean(awq)
    threshold = AWQ_RULE_FACTOR * pooled_sd
    all_below = all(p < rtn for p in awq)
    controls = {}
    for name, values in seeded.items():
        gap = statistics.fmean(values) - mean_awq
        paired = all(c > a for a, c in zip(awq, values))
        controls[name] = {
            "mean": statistics.fmean(values),
            "gap": gap,
            "awq_beats_control": gap > threshold and paired,
        }
    return {
        "awq_mean": mean_awq,
        "pooled_sd": pooled_sd,
        "gap": rtn - mean_awq,
        "threshold": threshold,
        "all_seeds_below_rtn": all_below,
        "awq_wins": (rtn - mean_awq) > threshold and all_below,
        "gate_pass": (mean_awq - ceiling) <= AWQ_GATE_DELTA,
        "controls": controls,
    }


def awq_section(report, heading):
    """Renders the calibration-aware (AWQ-style) comparison; pending until measured."""
    measured = awq_is_measured(report)
    ev = report["evaluation"]
    cal = report["calibration"]
    lines = [
        heading,
        "",
        f"Model: {report['model_name']} ({report['model_repo']}). Formats {', '.join(report['formats'])}. "
        f"Scored on WikiText-2 test ({ev['token_count']:,} tokens, context {ev['context_len']}). "
        f"Calibration statistics come from WikiText-2 train: {cal['windows']} windows of {cal['window_len']} tokens "
        f"({cal['tokens']:,} tokens) per calibration seed {', '.join(str(s) for s in cal['seeds'])}. "
        "Calibration never reads the scoring split.",
        "",
        "Arms: ceiling (unquantized); round-to-nearest without scaling (floor); AWQ-style scaling; control A "
        "(each layer's real per-channel scales permuted across input channels); control B1 (the same train "
        "tokens with their order shuffled); control B2 (uniform random token ids, a stress arm and not a "
        "baseline). Every result carries its benchmark id and version.",
        "",
    ]
    if not measured:
        lines += [
            f"**Status: PENDING.** {report['status_note']}",
            "",
            "Pre-registered decision rule, fixed before any measured run and applied unchanged to whatever is measured:",
            "",
        ]
        lines += [f"- {rule}" for rule in report["preregistration"]["decision_rule"]]
        lines += ["", "Still pending:", ""]
        lines += [f"- {item}" for item in report["pending"]]
        lines.append("")
        return lines

    for fmt in report["formats"]:
        block = report["results"][fmt]
        ceiling = block["ceiling"]["perplexity"]
        rtn = block["rtn"]["perplexity"]
        result = awq_evaluate(block, ceiling, rtn)
        lines += [
            f"**{fmt}**",
            "",
            "| Arm | n | Mean PPL | Sample SD | Min | Max | Δ vs RTN | Δ vs ceiling |",
            "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        rows = [
            ("Ceiling (unquantized)", [ceiling]),
            ("Round-to-nearest (floor)", [rtn]),
            ("AWQ-style, 5 calibration seeds", awq_runs(block["awq"])),
            ("Control A, 5 permutation seeds", awq_runs(block["control_a"])),
            ("Control B1, 5 shuffle seeds", awq_runs(block["control_b1"])),
            ("Control B2 (stress arm, not a baseline)", awq_runs(block["control_b2"])),
        ]
        for label, values in rows:
            s = awq_stats(values)
            sd = f"{s['sd']:.4f}" if s["n"] > 1 else "n/a"
            lines.append(
                f"| {label} | {s['n']} | {s['mean']:.4f} | {sd} | {s['min']:.4f} | {s['max']:.4f} | "
                f"{s['mean'] - rtn:+.4f} | {s['mean'] - ceiling:+.4f} |"
            )
        alphas = ", ".join(f"seed {r['seed']}: {r['alpha']}" for r in block["awq"]["runs"])
        lines += [
            "",
            f"Chosen alpha per calibration seed (grid {report['preregistration']['alpha_grid'][0]} to "
            f"{report['preregistration']['alpha_grid'][-1]}): {alphas}.",
            "",
            f"AWQ beats round-to-nearest: {'yes' if result['awq_wins'] else 'no'}. Gap RTN minus AWQ mean "
            f"{result['gap']:+.4f} against 2 x pooled SD {result['threshold']:.4f} (pooled SD "
            f"{result['pooled_sd']:.4f}); every seed below RTN: {'yes' if result['all_seeds_below_rtn'] else 'no'}.",
            "",
            f"Control A: {'AWQ beats it' if result['controls']['control_a']['awq_beats_control'] else 'not beaten'} "
            f"(mean gap {result['controls']['control_a']['gap']:+.4f}). Control B1: "
            f"{'AWQ beats it' if result['controls']['control_b1']['awq_beats_control'] else 'not beaten'} "
            f"(mean gap {result['controls']['control_b1']['gap']:+.4f}).",
            "",
            f"Gate (mean dPPL vs unquantized at most {AWQ_GATE_DELTA}): "
            f"{'pass' if result['gate_pass'] else 'fail'} at {result['awq_mean'] - ceiling:+.4f}.",
            "",
        ]
    lines += [
        "Null and negative results stand as reported. Each arm is measured on the same scoring split as the "
        "quantization gate above.",
        "",
    ]
    return lines


def generate_markdown(reports, output_md, sweep=None, probe=None, searches=None, awq=None):
    section = iter(range(1, 100))

    def h(title):
        return f"## {next(section)}. {title}"

    lines = [
        "# THME Benchmark Dossier",
        "",
        "> Measured results for THME (Topological Heterogeneous Manifold Engine).",
        "> Every number below was produced by a real forward pass on the host listed under Provenance.",
        "> Nothing in this document is simulated, projected, or estimated unless the row says so.",
        "",
        "> These reports were measured with the code after the FP16 rounding fix (THME-3023, 13 November",
        "> 2026). Reports measured before that fix were removed and re-run, not patched.",
        "",
        h("Perplexity Quality Gate"),
        "",
    ]

    for report in reports:
        ev = report["evaluation"]
        corpus = ev.get("corpus") or {}
        first = ev["variants"][0]
        full = corpus.get("token_count") == first["corpus_tokens_used"]
        coverage = (
            "complete split"
            if full
            else f"first {first['corpus_tokens_used']:,} tokens of the split (partial run)"
        )
        lines.extend([
            f"### {report['model_name']}",
            "",
            f"- Corpus: {corpus.get('corpus', 'unrecorded')}, {coverage}",
            f"- Scored tokens: {first['scored_tokens']:,} in {first['windows']} windows of {ev['context_len']}",
            f"- Gate tolerance: ΔPPL <= {ev['max_ppl_delta']}",
            "",
            "| Weights | Perplexity | Δ PPL | Cross-entropy (nats) | Gate |",
            "| :--- | ---: | ---: | ---: | :--- |",
        ])
        for v in ev["variants"]:
            delta = "" if v["delta_ppl"] is None else f"{v['delta_ppl']:+.4f}"
            lines.append(
                f"| {v['description']} | {v['perplexity']:.4f} | {delta} | "
                f"{v['cross_entropy_nats']:.4f} | {verdict(v)} |"
            )
        lines.append("")

    for s in sweep or []:
        lines.extend(sweep_section(s, h(f"Measured Precision Sensitivity: {s['model_name']}")))
    for r in searches or []:
        lines.extend(search_section(r, h(f"Measured Recipe Search: {r['model_name']}")))
    for d in probe or []:
        lines.extend(depth_section(d, h(f"Measured Depth Behaviour: {d['model_name']}")))
    for a in awq or []:
        lines.extend(awq_section(a, h(f"Activation-Aware Scaling (AWQ-style): {a['model_name']}")))

    lines.extend([h("Sizes"), ""])
    for report in reports:
        export = report.get("export")
        recipe = report["recipe"]
        lines.extend([
            f"### {report['model_name']}",
            "",
            "| Item | Size | Basis |",
            "| :--- | ---: | :--- |",
            f"| Source tensor payload | {mib(report['source_payload_bytes'])} | measured (bytes on disk) |",
        ])
        if export:
            lines.append(
                f"| Exported {export['container']}, {export['quant_type']} | {mib(export['bytes'])} | "
                f"measured (file size, sha256 `{(export.get('sha256') or 'n/a')[:16]}…`) |"
            )
        counts = ", ".join(f"{n} × {name}" for name, n in sorted(recipe["format_counts"].items()))
        lines.extend([
            f"| Recipe `{recipe['name']}` ({counts}) | {mib(recipe['estimated_bytes'])} | "
            "**estimate** from nominal bits per weight; no artifact of this size is written |",
            "",
        ])

    lines.extend([
        h("Method"),
        "",
        "- Forward pass: THME reference CPU implementation (`src/inference/`), dense f32 arithmetic,",
        "  RMSNorm, rotary position embeddings, grouped-query causal attention, SwiGLU MLP, tied output head.",
        "- The corpus is tokenized once with the model's own tokenizer, without special tokens",
        "  (`scripts/prepare_eval_tokens.py`), then cut into consecutive non-overlapping windows.",
        "- Supported architectures: LLaMA family and GPT-NeoX (Pythia). The forward pass is unit-tested",
        "  (causality, kernels, format round trips) but has not yet been cross-checked against the",
        "  Hugging Face reference implementation; that check is an open roadmap item.",
        "- Every token except the first of each window is scored given the earlier tokens of that window.",
        "  Perplexity is exp(mean negative log-likelihood) over all scored tokens.",
        "- This differs from overlapping sliding-window protocols, which give each token more context",
        "  and therefore report lower perplexity. Figures are comparable to each other, not to other protocols.",
        "- \"Exported artifact\" rows decode the written file back to f32 and run the same forward pass.",
        "- \"Recipe\" rows round-trip each tensor through its selected format in memory",
        "  (Q8_0: GGML 32-value blocks; Ternary: GGML TQ1_0 256-value blocks with MSE-optimal scale)",
        "  with no retraining or calibration.",
        "",
        h("Provenance"),
        "",
    ])
    for report in reports:
        ev = report["evaluation"]
        corpus = ev.get("corpus") or {}
        host = report.get("host", {})
        stamp = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(report.get("generated_unix_secs", 0)))
        elapsed = sum(v["elapsed_secs"] for v in ev["variants"])
        rate = ev["variants"][0]["tokens_per_sec"]
        lines.extend([
            f"### {report['model_name']}",
            "",
            f"- THME version: {report['thme_version']}; run completed {stamp}",
            f"- Host: {host.get('cpu', 'unknown CPU')} ({host.get('logical_cpus', '?')} logical CPUs), "
            f"{host.get('os', '?')}/{host.get('arch', '?')}",
            f"- Backend: {host.get('backend', 'unrecorded')}",
            f"- Evaluation wall time: {elapsed / 60:.1f} min total; baseline scored at {rate:.0f} tokens/s "
            "(reference implementation on a machine in normal use; not a performance benchmark)",
            f"- Corpus source: {corpus.get('corpus_source', 'unrecorded')} (`{corpus.get('corpus_member', '')}`)",
            f"- Corpus sha256: `{corpus.get('corpus_sha256', 'unrecorded')}`",
            f"- Tokenizer: {corpus.get('tokenizer_repo', 'unrecorded')}, sha256 `{corpus.get('tokenizer_sha256', 'unrecorded')}`",
            f"- Token file: {corpus.get('token_count', 0):,} tokens, sha256 `{corpus.get('token_file_sha256', 'unrecorded')}`",
            "",
        ])

    lines.extend([
        h("Not Measured"),
        "",
        "The following have no measurements and are deliberately absent from this dossier:",
        "",
        "- Inference throughput, energy, or memory on any GPU or accelerator.",
        "- Any speedup of THME-processed models over a baseline runtime.",
        "- Models other than those listed in section 1.",
        "- Any effect of retraining, distillation, or calibration on the quantized variants.",
        "",
        h("Reproduce"),
        "",
        "```bash",
        "pip install tokenizers",
        "python scripts/prepare_eval_tokens.py",
        "python scripts/prepare_eval_tokens.py --split train --calibration-windows 16 --window-len 2048 --seed 1234",
        "cargo run --release --example production_verification -- \\",
        "    --model-path models/SmolLM-135M \\",
        "    --tokens target/eval/wikitext2_test.SmolLM-135M.u32",
        "cargo run --release --example depth_probe -- \\",
        "    --model-path models/SmolLM-135M \\",
        "    --tokens target/eval/wikitext2_test.SmolLM-135M.u32",
        "cargo run --release --example precision_sweep -- \\",
        "    --model-path models/SmolLM-135M \\",
        "    --tokens target/eval/wikitext2_test.SmolLM-135M.u32 --layer-scan",
        "python scripts/generate_dossier.py",
        "python scripts/generate_charts.py --data-dir docs/benchmarks/<model>/data --out-dir docs/benchmarks/<model>/charts",
        "```",
    ])

    with open(output_md, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Generated Markdown dossier: {output_md}")


def generate_csv(reports, output_csv):
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow([
            "model", "variant", "perplexity", "delta_ppl", "cross_entropy_nats",
            "scored_tokens", "context_len", "accepted", "tokens_per_sec",
        ])
        for report in reports:
            ev = report["evaluation"]
            for v in ev["variants"]:
                writer.writerow([
                    report["model_name"], v["id"], f"{v['perplexity']:.6f}",
                    "" if v["delta_ppl"] is None else f"{v['delta_ppl']:.6f}",
                    f"{v['cross_entropy_nats']:.6f}", v['scored_tokens'], ev['context_len'],
                    "" if v["accepted"] is None else str(v["accepted"]).lower(),
                    f"{v['tokens_per_sec']:.1f}",
                ])
    print(f"Generated CSV dossier: {output_csv}")


def main():
    parser = argparse.ArgumentParser(description="THME Benchmark Dossier Generator")
    parser.add_argument("--input", nargs="+", default=[
        "docs/benchmarks/smollm-135m/data/verification_report.json",
        "docs/benchmarks/pythia-14m/data/verification_report.json",
    ],
                        help="Verification report JSON file(s) from production_verification")
    parser.add_argument("--sweep", nargs="*", default=[
        "docs/benchmarks/smollm-135m/data/precision_sweep.json",
        "docs/benchmarks/pythia-14m/data/precision_sweep.json",
    ],
                        help="Precision sweep JSON from precision_sweep (included when the file exists)")
    parser.add_argument("--depth", nargs="*", default=[
        "docs/benchmarks/smollm-135m/data/depth_probe.json",
        "docs/benchmarks/pythia-14m/data/depth_probe.json",
    ],
                        help="Depth probe JSON from depth_probe (included when the file exists)")
    parser.add_argument("--search", nargs="*", default=None,
                        help="thme search reports (default: docs/benchmarks/*/data/search_report.json)")
    parser.add_argument("--awq", nargs="*", default=None,
                        help="AWQ calibration reports (default: docs/benchmarks/*/data/awq_calibration.json)")
    parser.add_argument("--output-md", default="docs/benchmark_dossier.md", help="Output Markdown file")
    parser.add_argument("--output-csv", default="docs/benchmark_dossier.csv", help="Output CSV file")
    args = parser.parse_args()

    if not any(os.path.exists(path) for path in args.input):
        print("no committed reports, nothing to render")
        return 0

    reports = load_reports(args.input)
    os.makedirs(os.path.dirname(args.output_md) or ".", exist_ok=True)
    def load_all(paths):
        found = []
        for path in paths or []:
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    found.append(json.load(f))
        return found

    sweep = load_all(args.sweep)
    probe = load_all(args.depth)
    searches = load_all(args.search or sorted(glob.glob("docs/benchmarks/*/data/search_report.json")))
    awq = load_all(args.awq or sorted(glob.glob("docs/benchmarks/*/data/awq_calibration.json")))
    generate_markdown(reports, args.output_md, sweep, probe, searches, awq)
    generate_csv(reports, args.output_csv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
