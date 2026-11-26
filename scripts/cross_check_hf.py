#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""
THME forward pass reference cross-check against Hugging Face.

Compares logits produced by THME's reference forward pass against the
Hugging Face PyTorch reference implementation (AutoModelForCausalLM) on fixed prompts.
Asserts that max absolute difference across logits stays within tolerance.

Usage:
  python scripts/cross_check_hf.py --model-path models/SmolLM-135M --tolerance 0.005
  python scripts/cross_check_hf.py --allow-skip  # Exits 77 if torch/transformers missing
"""

import argparse
import json
import os
import subprocess
import sys

DEFAULT_PROMPTS = [
    "The capital of France is",
    "In mathematics, a manifold is",
    "Deep learning models require careful precision tuning because",
    "Once upon a time in a distant topological realm,",
]


def run_thme_forward_cli(thme_bin, model_path, tokens, limit=100):
    """
    Invokes `thme forward` to compute real forward pass logits on token IDs.
    """
    tokens_str = ",".join(str(t) for t in tokens)
    cmd = [
        thme_bin,
        "forward",
        model_path,
        "--tokens",
        tokens_str,
        "--limit",
        str(limit),
        "--json",
    ]
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=180,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"THME forward CLI failed (exit {proc.returncode}):\nSTDOUT: {proc.stdout}\nSTDERR: {proc.stderr}"
        )
    data = json.loads(proc.stdout)
    logits = data.get("logits", [])
    if not logits:
        raise RuntimeError(f"THME returned empty logits: {proc.stdout}")
    return logits


def run_thme_forward_mcp(thme_bin, model_path, tokens, limit=100):
    """
    Invokes `thme mcp` via JSON-RPC to query `thme_forward_pass` tool.
    """
    proc = subprocess.Popen(
        [thme_bin, "mcp", "--model", model_path],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    reqs = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "thme_forward_pass",
                "arguments": {
                    "model_path": model_path,
                    "tokens": tokens,
                    "limit": limit,
                },
            },
        },
    ]
    stdin_str = "\n".join(json.dumps(r) for r in reqs) + "\n"
    stdout, stderr = proc.communicate(input=stdin_str, timeout=180)
    for line in stdout.strip().split("\n"):
        if not line:
            continue
        try:
            msg = json.loads(line)
            if msg.get("id") == 2:
                res = msg.get("result", {})
                content = res.get("content", [{}])[0].get("text", "{}")
                payload = json.loads(content)
                logits = payload.get("logits", [])
                if logits:
                    return logits
        except Exception:
            continue
    raise RuntimeError(
        f"Failed to obtain logits over MCP:\nSTDOUT: {stdout}\nSTDERR: {stderr}"
    )


def compare_logits(hf_logits, thme_logits, tolerance=0.005):
    """
    Computes absolute difference statistics between HF and THME logits.
    """
    if len(hf_logits) != len(thme_logits):
        min_len = min(len(hf_logits), len(thme_logits))
        hf_logits = hf_logits[:min_len]
        thme_logits = thme_logits[:min_len]

    diffs = [abs(h - t) for h, t in zip(hf_logits, thme_logits)]
    max_diff = max(diffs) if diffs else 0.0
    mean_diff = sum(diffs) / len(diffs) if diffs else 0.0
    passed = max_diff <= tolerance
    return {
        "max_abs_diff": max_diff,
        "mean_abs_diff": mean_diff,
        "tolerance": tolerance,
        "passed": passed,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Cross-check THME reference forward pass against Hugging Face"
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="HuggingFaceTB/SmolLM-135M",
        help="Hugging Face model ID or path",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Local path to SafeTensors or GGUF model file",
    )
    parser.add_argument(
        "--thme-bin",
        type=str,
        default="target/debug/thme.exe" if os.name == "nt" else "target/debug/thme",
        help="Path to thme executable",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.005,
        help="Maximum allowable absolute logit difference",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Number of logits to compare per prompt (default 100)",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["cli", "mcp"],
        default="cli",
        help="THME invocation method ('cli' or 'mcp')",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to write JSON comparison report",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for the Hugging Face reference (cpu or cuda); fp32 either way",
    )
    parser.add_argument(
        "--allow-skip",
        action="store_true",
        help="Exit with status code 77 (SKIP) if dependencies or models are missing",
    )
    args = parser.parse_args()

    model_target = args.model_path or args.model_id
    print("=== THME Reference Forward Pass Cross-Check ===")
    print(f"Target Model: {model_target}")
    print(f"Tolerance:    {args.tolerance}")
    print(f"Method:       {args.method.upper()}")

    # Check for PyTorch & Transformers
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        has_torch = True
    except ImportError as e:
        has_torch = False
        print(
            f"\n[ERROR] Required Python packages ('torch', 'transformers') are not installed: {e}",
            file=sys.stderr,
        )
        print(
            "       Install dependencies with: pip install torch transformers\n",
            file=sys.stderr,
        )

    if not has_torch:
        report = {
            "model": model_target,
            "tolerance": args.tolerance,
            "status": "SKIPPED_MISSING_DEPENDENCIES",
            "overall_passed": False,
            "prompts": [],
        }
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
            print(f"Report written to: {args.output}")

        if args.allow_skip:
            print("Notice: --allow-skip enabled; exiting with code 77 (SKIP).")
            return 77
        return 1

    # Verify THME executable exists
    if not os.path.isfile(args.thme_bin):
        print(
            f"\n[ERROR] THME binary not found at '{args.thme_bin}'. Run 'cargo build --bin thme' first.",
            file=sys.stderr,
        )
        if args.allow_skip:
            return 77
        return 1

    # Load HF model & tokenizer
    try:
        print(f"Loading Hugging Face model & tokenizer for '{model_target}'...")
        tokenizer = AutoTokenizer.from_pretrained(model_target)
        model = AutoModelForCausalLM.from_pretrained(
            model_target, dtype=torch.float32  # newer transformers default to the stored dtype (fp16 here)
        )
        model.eval()
        model.to(args.device)
    except Exception as e:
        print(
            f"\n[ERROR] Failed to load reference model '{model_target}': {e}",
            file=sys.stderr,
        )
        report = {
            "model": model_target,
            "tolerance": args.tolerance,
            "status": "ERROR_MODEL_LOAD",
            "error": str(e),
            "overall_passed": False,
            "prompts": [],
        }
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
        if args.allow_skip:
            return 77
        return 1

    report = {
        "model": model_target,
        "tolerance": args.tolerance,
        "method": args.method,
        "prompts": [],
        "overall_passed": True,
        "status": "PASSED",
    }

    for prompt in DEFAULT_PROMPTS:
        print(f"\nEvaluating prompt: '{prompt}'")
        inputs = tokenizer(prompt, return_tensors="pt").to(args.device)
        token_ids = inputs["input_ids"][0].tolist()

        with torch.no_grad():
            hf_out = model(**inputs)
            hf_logits = hf_out.logits[0, -1, : args.limit].float().cpu().tolist()

        try:
            if args.method == "mcp":
                thme_logits = run_thme_forward_mcp(
                    args.thme_bin, model_target, token_ids, limit=args.limit
                )
            else:
                thme_logits = run_thme_forward_cli(
                    args.thme_bin, model_target, token_ids, limit=args.limit
                )
        except Exception as e:
            print(f"  [ERROR] THME forward execution failed: {e}", file=sys.stderr)
            report["prompts"].append(
                {
                    "prompt": prompt,
                    "tokens": token_ids,
                    "error": str(e),
                    "passed": False,
                }
            )
            report["overall_passed"] = False
            continue

        stats = compare_logits(hf_logits, thme_logits, args.tolerance)
        print(f"  Max Absolute Logit Diff: {stats['max_abs_diff']:.6f}")
        print(f"  Tolerance:               {stats['tolerance']:.6f}")
        print(f"  Verdict:                 {'PASS' if stats['passed'] else 'FAIL'}")

        report["prompts"].append(
            {
                "prompt": prompt,
                "tokens": token_ids,
                "stats": stats,
            }
        )
        if not stats["passed"]:
            report["overall_passed"] = False

    report["status"] = "PASSED" if report["overall_passed"] else "FAILED"
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\nReport written to: {args.output}")

    print(f"\nFinal Cross-Check Status: {report['status']}")
    return 0 if report["overall_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
