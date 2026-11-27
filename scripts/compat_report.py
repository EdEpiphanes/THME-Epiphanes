#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""
Compatibility report for one model across runtimes (ROADMAP v0.2.6.9, WP18). Standard library only.

For every runtime it records four facts, each read from a command's own output:
  loads   the process started, loaded the model and exited 0
  runs    it produced the requested 20 greedy tokens
  agrees  those tokens equal the THME CPU reference (the logits too, for THME runtimes)
  gate    for THME runtimes: perplexity on the same token slice differs from the CPU reference by
          at most the repository gate margin (0.45); for llama.cpp the windowing differs, so the
          gate is not evaluated and the field says so

Runtimes: THME on the CPU, on CUDA with the embedded PTX, on wgpu over Vulkan and over DX12, and
llama.cpp (CUDA build on the CPU and on the GPU, Vulkan build) reading a GGUF that THME exported
with --llama-cpp-layout. Nothing is simulated: a runtime that fails is recorded as failed.

  py scripts/compat_report.py --model-dir models/SmolLM-135M --tokens wikitext2_test.SmolLM-135M.u32 \
      --thme target/release/thme --llama-dir llama.cpp --work-dir target/compat \
      --cublas-dir cuda12-rt/nvidia/cublas/bin --out docs/benchmarks/smollm-135m/data/compatibility.json
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time

PROMPT = "The capital of France is"
NEW_TOKENS = 20
GATE_MARGIN = 0.45


def sh(cmd, env=None, timeout=1800, cwd=None):
    t0 = time.time()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=timeout, cwd=cwd,
                           encoding="utf-8", errors="replace")
        return p.returncode, p.stdout, p.stderr, time.time() - t0
    except subprocess.TimeoutExpired:
        return 124, "", "timeout", time.time() - t0


def tail(text, n=400):
    text = re.sub(r"\x1b\[[0-9;]*m", "", text or "").strip()
    return text[-n:]


def thme_json(thme, args, env):
    code, out, err, secs = sh([thme] + args, env=env)
    if code != 0 or "{" not in out:
        return None, tail(err or out), secs
    try:
        return json.loads(out[out.index("{"):]), "", secs
    except ValueError:
        return None, tail(out), secs


def encode(thme, model, text):
    code, out, err, _ = sh([thme, "tokenize", model, text, "--ids-only"])
    if code != 0:
        raise SystemExit(f"error: tokenize failed: {tail(err)}")
    return [int(x) for x in out.split()]


def thme_runtime(name, thme, model, ids, env, device, slice_tokens, tokens_file, work, identity):
    """Load, run, logits, greedy tokens and slice perplexity for one THME runtime."""
    row = {"runtime": name, "engine": "thme", "loads": False, "runs": False}
    cur = list(ids)
    first = None
    device_used = None
    secs_total = 0.0
    for _ in range(NEW_TOKENS):
        j, why, secs = thme_json(thme, ["forward", model, "--tokens", ",".join(map(str, cur)),
                                        "--limit", "200", "--device", device, "--json"], env)
        secs_total += secs
        if j is None:
            row["failure"] = why
            return row, None, None
        if first is None:
            first = j
            row["loads"] = True
            device_used = j.get("device")
        cur.append(j["top1_token_id"])
    row["runs"] = True
    row["device_reported"] = device_used
    row["seconds_for_20_forward_passes_including_loads"] = round(secs_total, 1)
    report = os.path.join(work, f"report-{name}.json")
    if not os.path.exists(report):
        code, out, err, secs = sh([thme, "evaluate", model, "--recipe", identity, "--tokens", tokens_file,
                                   "--eval-max-tokens", str(slice_tokens), "--report", report,
                                   "--device", device, "--no-baseline-cache"], env=env)
    ppl = None
    adapter = None
    if os.path.exists(report):
        with open(report, encoding="utf-8") as f:
            doc = json.load(f)
        base = next((v for v in doc["evaluation"]["variants"] if v.get("id") == "baseline"), {})
        ppl = base.get("perplexity")
        host = doc.get("host", {})
        adapter = {"device": host.get("device"), "backend": host.get("backend")}
    return row, first, (cur[len(ids):], ppl, adapter)


def llama_runtime(name, exe, gguf, ngl, ids, thme, model, cwd):
    row = {"runtime": name, "engine": "llama.cpp", "loads": False, "runs": False}
    code, out, err, secs = sh([exe, "-m", gguf, "-p", PROMPT, "-n", str(NEW_TOKENS), "-ngl", str(ngl),
                               "--temp", "0", "-no-cnv", "--no-display-prompt"], cwd=cwd)
    row["seconds"] = round(secs, 1)
    row["exit_code"] = code
    if code != 0:
        row["failure"] = tail(err or out)
        return row, None
    row["loads"] = True
    text = re.sub(r"\x1b\[[0-9;]*m", "", out)
    row["text"] = text.strip()
    got = encode(thme, model, PROMPT + text.rstrip("\n"))
    cont = got[len(ids):]
    row["runs"] = len(cont) >= NEW_TOKENS - 2
    return row, cont[:NEW_TOKENS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--thme", required=True)
    ap.add_argument("--llama-dir", required=True)
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--cublas-dir", default="")
    ap.add_argument("--slice", type=int, default=16384)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.work_dir, exist_ok=True)

    identity = os.path.join(args.work_dir, "identity.json")
    with open(identity, "w", encoding="utf-8") as f:
        json.dump({"name": "fp32-identity", "author": "Epiphanes", "target_model": "compat", "version": "1.0",
                   "description": "SetPrecision FP32 leaves every weight unchanged, so the report measures the runtime",
                   "operations": [{"type": "SetPrecision", "params": {"start_layer": 0, "end_layer": 1000,
                                                                       "role": None, "format": "FP32"}}]}, f)

    env = dict(os.environ)
    if args.cublas_dir:
        env["PATH"] = args.cublas_dir + os.pathsep + env.get("PATH", "")
    ids = encode(args.thme, args.model_dir, PROMPT)

    variants = [("thme-cpu", "cpu", {}), ("thme-cuda-ptx", "gpu", {}),
                ("thme-wgpu-vulkan", "gpu", {"THME_GPU_BACKEND": "vulkan"}),
                ("thme-wgpu-dx12", "gpu", {"THME_GPU_BACKEND": "dx12"})]
    rows, ref_tokens, ref_ppl, ref_logits = [], None, None, None
    for name, device, extra in variants:
        e = dict(env)
        e.update(extra)
        row, first, rest = thme_runtime(name, args.thme, args.model_dir, ids, e, device, args.slice,
                                        args.tokens, args.work_dir, identity)
        if rest is not None:
            toks, ppl, adapter = rest
            row["adapter"] = adapter
            row["greedy_tokens"] = toks
            row["perplexity_on_slice"] = ppl
            row["top1_logit"] = first["top1_logit"]
            if name == "thme-cpu":
                ref_tokens, ref_ppl, ref_logits = toks, ppl, first["logits"]
            else:
                row["max_abs_logit_difference_first_200"] = (
                    max(abs(a - b) for a, b in zip(first["logits"], ref_logits)) if ref_logits else None)
        rows.append(row)
    for row in rows:
        if row["runtime"] == "thme-cpu":
            row["agrees"] = bool(row["runs"])
            row["agrees_note"] = "this is the reference"
            row["gate"] = "reference" if row["runs"] else None
            continue
        row["agrees"] = bool(row.get("greedy_tokens")) and row["greedy_tokens"] == ref_tokens
        ppl = row.get("perplexity_on_slice")
        row["gate"] = (None if ppl is None or ref_ppl is None
                       else "passed" if abs(ppl - ref_ppl) <= GATE_MARGIN else "failed")
        if row["gate"]:
            row["gate_basis"] = f"|perplexity - CPU perplexity| <= {GATE_MARGIN} on the same {args.slice}-token slice"

    # llama.cpp reading a GGUF THME wrote
    gguf = os.path.join(args.work_dir, "model-f16-llama.gguf")
    code, out, err, _ = sh([args.thme, "export", args.model_dir, "-o", gguf, "--quant", "fp16", "--llama-cpp-layout"])
    lrows = []
    if code != 0:
        lrows.append({"runtime": "llama.cpp", "engine": "llama.cpp", "loads": False, "runs": False,
                      "failure": "export failed: " + tail(err or out)})
    else:
        for name, sub, ngl in (("llama.cpp-cuda-build-cpu", "cuda", 0), ("llama.cpp-cuda-build-gpu", "cuda", 99),
                               ("llama.cpp-vulkan-build-gpu", "vulkan", 99)):
            d = os.path.join(args.llama_dir, sub)
            exe = os.path.join(d, "llama-completion.exe")
            row, cont = llama_runtime(name, exe, gguf, ngl, ids, args.thme, args.model_dir, d)
            row["agrees"] = bool(cont) and ref_tokens is not None and cont == ref_tokens[:len(cont)] and len(cont) >= NEW_TOKENS - 2
            row["greedy_tokens"] = cont
            row["gate"] = "not evaluated: llama-perplexity windows the text differently from `thme evaluate`"
            lrows.append(row)

    doc = {
        "report": "compatibility", "report_version": 1, "date": time.strftime("%Y-%m-%d"),
        "model": os.path.basename(os.path.normpath(args.model_dir)), "prompt": PROMPT,
        "prompt_token_ids": ids, "new_tokens": NEW_TOKENS, "reference": "thme-cpu",
        "reference_greedy_tokens": ref_tokens, "reference_perplexity_on_slice": ref_ppl,
        "slice_tokens": args.slice, "gate_margin": GATE_MARGIN,
        "runtimes": rows + lrows,
        "reading": "loads: process exited 0; runs: produced the tokens; agrees: equals the THME CPU reference; gate: see gate_basis",
        "known_gaps": ["one model, one prompt, F16 export for llama.cpp",
                       "llama.cpp perplexity is not compared (different windowing)",
                       "no speed is claimed here"],
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(doc, f, indent=2)
        f.write("\n")
    for r in doc["runtimes"]:
        print(f"{r['runtime']:28} loads={r['loads']} runs={r['runs']} agrees={r.get('agrees')} gate={str(r.get('gate'))[:30]}")


if __name__ == "__main__":
    main()
