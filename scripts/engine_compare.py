#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""
One model through two engines and two formats (ROADMAP v0.2.6.9, v0.2.9.6). Standard library only.

  A  THME reading the original SafeTensors folder
  B  THME reading the .epiph made from it (`thme epiph pack`)
  C  llama.cpp reading a GGUF that THME converted from that same .epiph (`thme export --llama-cpp-layout`)

For several prompts, 32 greedy tokens each. A against B isolates the file format (the weights are the
same bytes, so the tokens must be identical). B against C isolates the engine and the conversion.
Tokens of C are read back by re-encoding its text with the model's tokenizer, so a tokenizer mismatch
in the GGUF would show as a disagreement and is reported as such, not hidden.

  py scripts/engine_compare.py --model-dir models/SmolLM-135M --thme target/release/thme \
      --llama-dir llama.cpp --work-dir target/engine-compare --device gpu \
      --cublas-dir cuda12-rt/nvidia/cublas/bin --out docs/benchmarks/smollm-135m/data/engine_compare.json
"""

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time

PROMPTS = [
    "The capital of France is",
    "In 1969, the first humans",
    "To make a cup of tea, you first",
    "def fibonacci(n):",
    "The theory of relativity states that",
]
NEW = 32


def sh(cmd, env=None, cwd=None, timeout=1800):
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=cwd, timeout=timeout,
                       encoding="utf-8", errors="replace")
    return p.returncode, p.stdout, p.stderr, time.time() - t0


def encode(thme, model, text):
    code, out, err, _ = sh([thme, "tokenize", model, text, "--ids-only"])
    if code != 0:
        raise SystemExit(f"tokenize failed: {err[-300:]}")
    return [int(x) for x in out.split()]


def greedy(thme, source, ids, env, device):
    cur = list(ids)
    for _ in range(NEW):
        code, out, err, _ = sh([thme, "forward", source, "--tokens", ",".join(map(str, cur)), "--limit", "1",
                                "--device", device, "--json"], env=env)
        if code != 0 or "{" not in out:
            return None, err[-300:]
        cur.append(json.loads(out[out.index("{"):])["top1_token_id"])
    return cur[len(ids):], ""


def first_difference(a, b):
    """Index of the first differing token over the common length, or None if they agree there."""
    if a is None or b is None:
        return None
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None


def length_note(a, b, whitespace_ids):
    """Explains a length difference: llama.cpp's text has its trailing newlines stripped before it is
    re-encoded, so extra THME tokens that are only newlines are not a disagreement."""
    if a is None or b is None or len(a) == len(b):
        return None
    longer = a if len(a) > len(b) else b
    extra = longer[min(len(a), len(b)):]
    return {"thme_epiph_tokens": len(a), "llama_cpp_tokens": len(b), "extra_tokens": extra,
            "extra_tokens_are_only_newlines": all(x in whitespace_ids for x in extra)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--thme", required=True)
    ap.add_argument("--llama-dir", required=True)
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="gpu")
    ap.add_argument("--cublas-dir", default="")
    args = ap.parse_args()
    os.makedirs(args.work_dir, exist_ok=True)
    env = dict(os.environ)
    if args.cublas_dir:
        env["PATH"] = args.cublas_dir + os.pathsep + env.get("PATH", "")

    epiph = os.path.join(args.work_dir, "model.epiph")
    gguf = os.path.join(args.work_dir, "from-epiph.gguf")
    steps = {}
    for label, cmd in (("pack", [args.thme, "epiph", "pack", args.model_dir, "--out", epiph]),
                       ("verify", [args.thme, "epiph", "verify", epiph]),
                       ("convert", [args.thme, "export", epiph, "-o", gguf, "--quant", "fp16", "--llama-cpp-layout"])):
        code, out, err, secs = sh(cmd, env=env)
        steps[label] = {"exit_code": code, "stderr_tail": err[-300:] if code else ""}
        if code != 0:
            print(f"{label} failed: {err[-400:]}")
            break
    doc = {"report": "engine_compare", "report_version": 1, "date": time.strftime("%Y-%m-%d"),
           "model": os.path.basename(os.path.normpath(args.model_dir)), "device": args.device,
           "steps": steps, "new_tokens": NEW, "prompts": [], "known_gaps": [
               "one model; THME device is the one named above, llama.cpp runs on its own backends",
               "llama.cpp tokens are re-encoded from its text; a text that does not round-trip would show as a disagreement",
               "the GGUF has no tokenizer.ggml.pre (THME-3102)"]}
    if all(s["exit_code"] == 0 for s in steps.values()) and len(steps) == 3:
        for text in PROMPTS:
            ids = encode(args.thme, args.model_dir, text)
            whitespace_ids = set()
            for run in ("\n", "\n\n", "\n\n\n"):
                whitespace_ids.update(encode(args.thme, args.model_dir, run))
            a, ea = greedy(args.thme, args.model_dir, ids, env, args.device)
            b, eb = greedy(args.thme, epiph, ids, env, args.device)
            row = {"prompt": text, "prompt_token_ids": ids, "thme_safetensors": a, "thme_epiph": b,
                   "safetensors_vs_epiph_first_difference": first_difference(a, b), "llama_cpp": {}}
            if ea or eb:
                row["failure"] = ea or eb
            for name, sub, ngl in (("cuda-build-cpu", "cuda", 0), ("cuda-build-gpu", "cuda", 99), ("vulkan-build-gpu", "vulkan", 99)):
                d = os.path.join(args.llama_dir, sub)
                code, out, err, _ = sh([os.path.join(d, "llama-completion.exe"), "-m", gguf, "-p", text, "-n", str(NEW),
                                        "-ngl", str(ngl), "--temp", "0", "-no-cnv", "--no-display-prompt"], cwd=d)
                if code != 0:
                    row["llama_cpp"][name] = {"failure": err[-300:]}
                    continue
                txt = re.sub(r"\x1b\[[0-9;]*m", "", out)
                toks = encode(args.thme, args.model_dir, text + txt.rstrip("\n"))[len(ids):][:NEW]
                row["llama_cpp"][name] = {"tokens": toks, "text": txt.strip()[:200],
                                          "epiph_vs_llama_first_difference": first_difference(b, toks),
                                          "length_difference": length_note(b, toks, whitespace_ids)}
            doc["prompts"].append(row)
        diffs = [p["safetensors_vs_epiph_first_difference"] for p in doc["prompts"]]
        doc["summary"] = {
            "safetensors_equals_epiph_on_every_prompt": all(d is None for d in diffs),
            "llama_cpp_equals_epiph_on_the_common_length": {n: sum(1 for p in doc["prompts"] if p["llama_cpp"].get(n, {}).get("epiph_vs_llama_first_difference", 0) is None)
                                                            for n in ("cuda-build-cpu", "cuda-build-gpu", "vulkan-build-gpu")},
            "llama_cpp_length_differences_that_are_not_only_newlines": sum(
                1 for p in doc["prompts"] for v in p["llama_cpp"].values()
                if v.get("length_difference") and not v["length_difference"]["extra_tokens_are_only_newlines"]),
            "prompts": len(doc["prompts"])}
        doc["epiph_sha256"] = hashlib.sha256(open(epiph, "rb").read()).hexdigest()
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(doc, f, indent=2)
        f.write("\n")
    print(json.dumps(doc.get("summary", steps), indent=2))


if __name__ == "__main__":
    main()
