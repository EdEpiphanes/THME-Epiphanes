#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""
S0 of the arm D plan (THME-3110, docs/research/epiph-testbed-arm-d-prep.md): fake-quantise a trained checkpoint, one
layer at a time and then the middle layers together, with each weight regime of the paper (binary, ternary, phase) and
a plain 2-bit control, and measure WikiText-2 test perplexity without any training.

Quantised weights are written back as ordinary float tensors ("fake quantisation"), so the result is a normal
GPT-NeoX model. Screening runs in PyTorch (float32, eager attention); the numbers that go into a report are re-measured
with `thme evaluate` on the exported folders. Bits per weight are computed from the formulas, not from files.

  python scripts/testbed_s0.py --model target/testbed/arm-scratch/step-012000 \
      --tokens target/testbed/eval/wikitext2_test.pythia-14m.u32 --json docs/benchmarks/epiph-testbed/data/s0_arm_a.json
  python scripts/testbed_s0.py --selftest
"""

import argparse
import json
import math

import numpy as np
import torch
from transformers import GPTNeoXForCausalLM

GROUP = 1024  # weights per scale, as in the paper's row blocks


def groups(w):
    """Flatten to (n_groups, GROUP); the last partial group is padded with zeros and trimmed after."""
    flat = w.reshape(-1)
    pad = (-flat.numel()) % GROUP
    return torch.nn.functional.pad(flat, (0, pad)).reshape(-1, GROUP), flat.numel()


def ungroup(g, n, shape):
    return g.reshape(-1)[:n].reshape(shape)


def ternary(w):
    """BITCOS-style: dead zone 0.7 E|w|, one scale per group (mean magnitude of the kept weights)."""
    g, n = groups(w)
    tau = 0.7 * g.abs().mean(dim=1, keepdim=True)
    keep = g.abs() > tau
    scale = (g.abs() * keep).sum(1, keepdim=True) / keep.sum(1, keepdim=True).clamp(min=1)
    return ungroup(torch.sign(g) * keep * scale, n, w.shape), math.log2(3) + 16 / GROUP


def binary(w, rank):
    """Sign bits with one scale per group, plus a rank-`rank` SVD of the residual stored in f16 (LittleBit-like)."""
    g, n = groups(w)
    q = ungroup(torch.sign(g) * g.abs().mean(1, keepdim=True), n, w.shape)
    bits = 1 + 16 / GROUP
    if rank > 0 and w.dim() == 2:
        u, s, vh = torch.linalg.svd(w - q, full_matrices=False)
        r = min(rank, s.numel())
        q = q + (u[:, :r] * s[:r]) @ vh[:r]
        bits += 16 * r * sum(w.shape) / w.numel()
    return q, bits


def scalar2(w):
    """Control: 2 bits per weight, four uniform levels per group with the step that is optimal for a Gaussian
    (0.996 standard deviations), so the control is not made weak on purpose."""
    g, n = groups(w)
    d = 0.9957 * g.std(1, keepdim=True).clamp(min=1e-12)
    q = (torch.clamp(torch.round(g / d + 0.5) - 0.5, -1.5, 1.5)) * d
    return ungroup(q, n, w.shape), 2 + 16 / GROUP


def lloyd(r, levels, iters=8):
    """1-D Lloyd-Max levels for the magnitudes of one group."""
    c = torch.quantile(r, torch.linspace(0.1, 0.9, levels, device=r.device))
    for _ in range(iters):
        idx = (r[:, None] - c[None]).abs().argmin(1)
        for k in range(levels):
            m = idx == k
            if m.any():
                c[k] = r[m].mean()
    return c


def phase(w, angle_bits, radius_bits):
    """Adjacent weight pairs as (radius, angle). Angle on a uniform grid; radius is the group RMS when
    radius_bits is 0, else Lloyd-Max levels per group. Radius is rescaled so the group's second moment is kept."""
    g, n = groups(w)
    a, b = g[:, 0::2], g[:, 1::2]
    r, th = torch.sqrt(a * a + b * b), torch.atan2(b, a)
    k = 2 ** angle_bits
    th_q = (torch.round((th + math.pi) / (2 * math.pi) * k) % k) / k * 2 * math.pi - math.pi
    r_q = torch.empty_like(r)
    for i in range(r.shape[0]):
        if radius_bits == 0:
            r_q[i] = torch.sqrt((r[i] ** 2).mean())
        else:
            c = lloyd(r[i], 2 ** radius_bits)
            r_q[i] = c[(r[i][:, None] - c[None]).abs().argmin(1)]
            e = (r[i] ** 2).mean() / ((r_q[i] ** 2).mean() + 1e-20)
            r_q[i] = r_q[i] * torch.sqrt(e)
    out = torch.empty_like(g)
    out[:, 0::2], out[:, 1::2] = r_q * torch.cos(th_q), r_q * torch.sin(th_q)
    return ungroup(out, n, w.shape), (angle_bits + radius_bits) / 2 + 16 / GROUP


REGIMES = {
    "ternary": lambda w: ternary(w),
    "binary_r0": lambda w: binary(w, 0),
    "binary_r8": lambda w: binary(w, 8),
    "binary_r32": lambda w: binary(w, 32),
    "scalar2_control": lambda w: scalar2(w),
    "phase_P1_a3_r0": lambda w: phase(w, 3, 0),
    "phase_P2_a3_r1": lambda w: phase(w, 3, 1),
    "phase_P3_a4_r0": lambda w: phase(w, 4, 0),
    "phase_P4_a4_r2": lambda w: phase(w, 4, 2),
}
LAYER_WEIGHTS = ("attention.query_key_value", "attention.dense", "mlp.dense_h_to_4h", "mlp.dense_4h_to_h")


def load_windows(path, context):
    ids = np.fromfile(path, dtype="<u4").astype(np.int64)
    return [ids[i:i + context] for i in range(0, len(ids), context) if len(ids[i:i + context]) >= 2]


@torch.no_grad()
def perplexity(model, windows, device):
    total, count = 0.0, 0
    for w in windows:
        x = torch.from_numpy(w).to(device)[None]
        logits = model(x[:, :-1]).logits.float()
        total += torch.nn.functional.cross_entropy(logits[0], x[0, 1:], reduction="sum").item()
        count += x.shape[1] - 1
    return math.exp(total / count)


def quantise_layers(model, layers, regime):
    """Replace the layer weights in place; returns (original tensors, weight-count-weighted bits per weight)."""
    saved, bits_sum, n_sum = {}, 0.0, 0
    for li in layers:
        layer = model.gpt_neox.layers[li]
        for name in LAYER_WEIGHTS:
            mod = layer
            for part in name.split("."):
                mod = getattr(mod, part)
            saved[(li, name)] = mod.weight.data.clone()
            q, bits = REGIMES[regime](mod.weight.data.float())
            mod.weight.data = q.to(mod.weight.dtype)
            bits_sum += bits * q.numel()
            n_sum += q.numel()
    return saved, bits_sum / max(1, n_sum)


def restore(model, saved):
    for (li, name), w in saved.items():
        mod = model.gpt_neox.layers[li]
        for part in name.split("."):
            mod = getattr(mod, part)
        mod.weight.data = w


def selftest():
    torch.manual_seed(0)
    w = torch.randn(128, 512) * 0.02
    err = {}
    for name, f in REGIMES.items():
        q, bits = f(w)
        assert q.shape == w.shape and torch.isfinite(q).all() and bits > 0, name
        err[name] = ((q - w) ** 2).mean().item() / (w ** 2).mean().item()
    q, _ = ternary(w)
    g, n = groups(q)
    assert all(len(torch.unique(row.abs())) <= 2 for row in g), "ternary has more than one magnitude per group"
    assert err["phase_P4_a4_r2"] < err["phase_P1_a3_r0"], "more phase bits should lower the error"
    assert err["binary_r32"] < err["binary_r0"], "a residual should lower the binary error"
    print("selftest ok", {k: round(v, 3) for k, v in err.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--tokens")
    ap.add_argument("--json")
    ap.add_argument("--context", type=int, default=1024)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    device = "cuda" if torch.cuda.is_available() and not a.cpu else "cpu"
    model = GPTNeoXForCausalLM.from_pretrained(a.model, torch_dtype=torch.float32).to(device).eval()
    windows = load_windows(a.tokens, a.context)
    base = perplexity(model, windows, device)
    print("baseline", round(base, 3), flush=True)
    n_layers = model.config.num_hidden_layers
    cases = [([li], f"layer{li}") for li in range(n_layers)] + [(list(range(1, n_layers - 1)), "middle_1_to_%d" % (n_layers - 2))]
    rows = []
    for layers, label in cases:
        for regime in REGIMES:
            saved, bpw = quantise_layers(model, layers, regime)
            ppl = perplexity(model, windows, device)
            restore(model, saved)
            rows.append({"layers": layers, "scope": label, "regime": regime, "bits_per_weight": round(bpw, 3),
                         "perplexity": ppl, "delta_ppl": ppl - base})
            print(label, regime, round(bpw, 2), round(ppl, 2), flush=True)
    out = {"report": "s0_fake_quantise", "model": a.model.replace("\\", "/").split("/")[-2:], "device": device,
           "precision": "float32 weights, eager attention", "context_len": a.context, "windows": len(windows),
           "baseline_perplexity": base,
           "note": "post-hoc fake quantisation, no training; screening values, re-measure reported ones with thme evaluate",
           "rows": rows}
    with open(a.json, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
