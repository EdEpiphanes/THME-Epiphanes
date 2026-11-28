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
from transformers import AutoModelForCausalLM

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


def scalar3(w):
    """Control: 3 bits per weight, eight uniform levels, Gaussian-optimal step (0.586 standard deviations)."""
    g, n = groups(w)
    d = 0.586 * g.std(1, keepdim=True).clamp(min=1e-12)
    q = torch.clamp(torch.round(g / d + 0.5) - 0.5, -3.5, 3.5) * d
    return ungroup(q, n, w.shape), 3 + 16 / GROUP


def scalar_3level(w):
    """Control at ternary's rate with no dead-zone rule: three uniform levels, Gaussian-optimal step (1.224 sd)."""
    g, n = groups(w)
    d = 1.224 * g.std(1, keepdim=True).clamp(min=1e-12)
    return ungroup(torch.clamp(torch.round(g / d), -1, 1) * d, n, w.shape), math.log2(3) + 16 / GROUP


def codes(w, regime):
    """Integer code of each weight for the regimes that have one; used to count codes that change in the f16 cast."""
    g, _ = groups(w)
    if regime == "ternary":
        return torch.sign(g) * (g.abs() > 0.7 * g.abs().mean(1, keepdim=True))
    if regime == "binary_r0":
        return torch.sign(g)
    return None


def f16_flip_fraction(w, regime):
    """Share of weights whose code changes when the float32 weight is first rounded to float16 (paper 2610.07853)."""
    a = codes(w, regime)
    if a is None:
        return None
    b = codes(w.half().float(), regime)
    return (a != b).float().mean().item()


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


def lloyd_batched(r, levels, iters=8):
    """lloyd() for every row of r at once (rows are groups)."""
    c = torch.quantile(r, torch.linspace(0.1, 0.9, levels, device=r.device), dim=1).T.contiguous()
    for _ in range(iters):
        onehot = (r[:, :, None] - c[:, None, :]).abs().argmin(2)[:, :, None] == torch.arange(levels, device=r.device)
        cnt = onehot.sum(1)
        c = torch.where(cnt > 0, (onehot * r[:, :, None]).sum(1) / cnt.clamp(min=1), c)
    return c


def phase(w, angle_bits, radius_bits):
    """Adjacent weight pairs as (radius, angle). Angle on a uniform grid; radius is the group RMS when
    radius_bits is 0, else Lloyd-Max levels per group. Radius is rescaled so the group's second moment is kept."""
    g, n = groups(w)
    a, b = g[:, 0::2], g[:, 1::2]
    r, th = torch.sqrt(a * a + b * b), torch.atan2(b, a)
    k = 2 ** angle_bits
    th_q = (torch.round((th + math.pi) / (2 * math.pi) * k) % k) / k * 2 * math.pi - math.pi
    if radius_bits == 0:
        r_q = torch.sqrt((r ** 2).mean(1, keepdim=True)).expand_as(r)
    else:
        c = lloyd_batched(r, 2 ** radius_bits)
        r_q = torch.gather(c, 1, (r[:, :, None] - c[:, None, :]).abs().argmin(2))
        e = (r ** 2).mean(1, keepdim=True) / ((r_q ** 2).mean(1, keepdim=True) + 1e-20)
        r_q = r_q * torch.sqrt(e)
    out = torch.empty_like(g)
    out[:, 0::2], out[:, 1::2] = r_q * torch.cos(th_q), r_q * torch.sin(th_q)
    return ungroup(out, n, w.shape), (angle_bits + radius_bits) / 2 + 16 / GROUP


def lloyd3(w, iters=10):
    """Stronger 3-bit control: eight non-uniform Lloyd-Max levels per group of 1,024 weights (levels stored in f16, so
    the rate is 3 + 128/1024 bits). Batched over groups."""
    g, n = groups(w)
    c = torch.quantile(g, torch.linspace(0.0625, 0.9375, 8, device=g.device), dim=1).T.contiguous()  # (G, 8)
    for _ in range(iters):
        idx = (g[:, :, None] - c[:, None, :]).abs().argmin(2)
        for k in range(8):
            m = (idx == k).float()
            c[:, k] = torch.where(m.sum(1) > 0, (g * m).sum(1) / m.sum(1).clamp(min=1), c[:, k])
    idx = (g[:, :, None] - c[:, None, :]).abs().argmin(2)
    return ungroup(torch.gather(c, 1, idx), n, w.shape), 3 + 128 / GROUP


REGIMES = {
    "ternary": lambda w: ternary(w),
    "binary_r0": lambda w: binary(w, 0),
    "binary_r8": lambda w: binary(w, 8),
    "binary_r32": lambda w: binary(w, 32),
    "scalar2_control": lambda w: scalar2(w),
    "scalar3_control": lambda w: scalar3(w),
    "scalar_3level_control": lambda w: scalar_3level(w),
    "phase_P1_a3_r0": lambda w: phase(w, 3, 0),
    "phase_P2_a3_r1": lambda w: phase(w, 3, 1),
    "phase_P3_a4_r0": lambda w: phase(w, 4, 0),
    "phase_P4_a4_r2": lambda w: phase(w, 4, 2),
    "phase_P5_a3_r3": lambda w: phase(w, 3, 3),
    "phase_P6_a5_r1": lambda w: phase(w, 5, 1),
    "lloyd3_control": lambda w: lloyd3(w),
}
FAST_REGIMES = ("ternary", "binary_r8", "scalar2_control", "scalar3_control", "scalar_3level_control",
                "phase_P2_a3_r1", "phase_P4_a4_r2", "lloyd3_control")
LLAMA_WEIGHTS = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                 "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
LAYER_WEIGHTS = ("attention.query_key_value", "attention.dense", "mlp.dense_h_to_4h", "mlp.dense_4h_to_h")


def load_windows(path, context):
    ids = np.fromfile(path, dtype="<u4").astype(np.int64)
    return [ids[i:i + context] for i in range(0, len(ids), context) if len(ids[i:i + context]) >= 2]


@torch.no_grad()
def perplexity(model, windows, device, batch=8):
    """Returns (perplexity, per-window summed negative log-likelihood in nats, per-window scored tokens).
    Equal-length windows go through the model `batch` at a time; the short last window goes alone."""
    per, cnt = [], []
    i = 0
    while i < len(windows):
        n = 1
        while i + n < len(windows) and n < batch and len(windows[i + n]) == len(windows[i]):
            n += 1
        x = torch.from_numpy(np.stack(windows[i:i + n])).to(device)
        logits = model(x[:, :-1]).logits.float()
        nll = torch.nn.functional.cross_entropy(logits.transpose(1, 2), x[:, 1:], reduction="none").sum(1)
        per += [round(v, 4) for v in nll.tolist()]
        cnt += [x.shape[1] - 1] * n
        i += n
    return math.exp(sum(per) / sum(cnt)), per, cnt


def layer_list(model):
    return (model.gpt_neox if hasattr(model, "gpt_neox") else model.model).layers


def weight_names(model):
    return LAYER_WEIGHTS if hasattr(model, "gpt_neox") else LLAMA_WEIGHTS


def quantise_layers(model, layers, regime):
    """Replace the layer weights in place; returns (original tensors, weight-count-weighted bits per weight)."""
    saved, bits_sum, n_sum, flip_sum = {}, 0.0, 0, 0.0
    for li in layers:
        layer = layer_list(model)[li]
        for name in weight_names(model):
            mod = layer
            for part in name.split("."):
                mod = getattr(mod, part)
            saved[(li, name)] = mod.weight.data.clone()
            q, bits = REGIMES[regime](mod.weight.data.float())
            fl = f16_flip_fraction(mod.weight.data.float(), regime)
            flip_sum = None if fl is None or flip_sum is None else flip_sum + fl * q.numel()
            mod.weight.data = q.to(mod.weight.dtype)
            bits_sum += bits * q.numel()
            n_sum += q.numel()
    return saved, bits_sum / max(1, n_sum), None if flip_sum is None else flip_sum / max(1, n_sum)


def restore(model, saved):
    for (li, name), w in saved.items():
        mod = layer_list(model)[li]
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
    ap.add_argument("--fast", action="store_true", help="only the regimes still in play after entry 6 of the findings")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    device = "cuda" if torch.cuda.is_available() and not a.cpu else "cpu"
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float32, attn_implementation="eager").to(device).eval()
    windows = load_windows(a.tokens, a.context)
    base, base_nll, window_tokens = perplexity(model, windows, device)
    print("baseline", round(base, 3), flush=True)
    n_layers = model.config.num_hidden_layers
    cases = [([li], f"layer{li}") for li in range(n_layers)] + [(list(range(1, n_layers - 1)), "middle_1_to_%d" % (n_layers - 2))]
    rows = []
    regimes = list(FAST_REGIMES) if a.fast else list(REGIMES)
    for layers, label in cases:
        for regime in regimes:
            saved, bpw, flips = quantise_layers(model, layers, regime)
            ppl, nll_w, _ = perplexity(model, windows, device)
            restore(model, saved)
            rows.append({"layers": layers, "scope": label, "regime": regime, "bits_per_weight": round(bpw, 3),
                         "f16_code_flip_fraction": flips,
                         "perplexity": ppl, "delta_ppl": ppl - base, "window_nll": nll_w})
            print(label, regime, round(bpw, 2), round(ppl, 2), flush=True)
    out = {"report": "s0_fake_quantise", "model": a.model.replace("\\", "/").split("/")[-2:], "device": device,
           "precision": "float32 weights, eager attention", "context_len": a.context, "windows": len(windows),
           "baseline_perplexity": base, "baseline_window_nll": base_nll, "window_tokens": window_tokens,
           "note": "post-hoc fake quantisation, no training; screening values, re-measure reported ones with thme evaluate",
           "rows": rows}
    with open(a.json, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
