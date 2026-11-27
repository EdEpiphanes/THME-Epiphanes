# SPDX-License-Identifier: AGPL-3.0-only
"""Independent PyTorch oracle for the hand-written backward pass (THME-3036 follow-up).

Reads the JSON that `cargo run --example dump_tiny_gradients -- <out.json>` writes (tiny model
weights under their Hugging Face names, the token ids, and the Rust loss and gradients), loads the
weights into transformers LlamaForCausalLM in float64 on the CPU, and compares:

- the loss: mean next-token cross-entropy, logits[:-1] against tokens[1:] (the Rust `loss`);
- per tensor: the maximum relative error of the gradient, |rust - torch| / max(|torch|, 1e-9).

If the losses disagree the script stops before any gradient is compared and writes nothing.
Otherwise it writes the fixture `tests/fixtures/torch_gradients_tiny.json` with the PyTorch loss
and, per tensor, the gradient norm and its first eight values.

    python scripts/make_torch_gradients.py <dump.json> [--out <fixture.json>]

Standard library plus the torch and transformers already installed in the venv. Nothing is
downloaded. The GPU is hidden on purpose (CUDA_VISIBLE_DEVICES is set empty before torch loads).
"""
import argparse
import json
import os
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "tests" / "fixtures" / "torch_gradients_tiny.json"
LOSS_TOL = 1e-9  # relative, forward loss: below this the two forward passes agree
GRAD_TOL = 1e-6  # relative, per tensor: the verdict threshold, same as the Rust fixture test
ABS_FLOOR = 1e-9


def rel_error(a, b):
    """|a - b| / max(|b|, ABS_FLOOR), for two float lists of equal length, taking the maximum."""
    return max(abs(x - y) / max(abs(y), ABS_FLOOR) for x, y in zip(a, b))


def use_float64_ops(model, torch, theta, head_dim):
    """Runs the rotary embedding and RMSNorm in float64, with the same formulas as transformers.

    shortcut: transformers 5.24 builds the rotary inv_freq and angles in float32 (modeling_llama.py
    line 106 and 111) and its RMSNorm casts to float32 (line 64). Those casts move the loss by
    about 7e-9 relative, which is the disagreement the first STOP reported (see --hf-precision).
    Upgrade when transformers keeps these in the model dtype. Attention is left to transformers'
    default SDPA path; forcing it explicitly gave the same loss, so it adds no float32 step here.
    """
    inv_freq = 1.0 / theta ** (torch.arange(0, head_dim, 2, dtype=torch.float64) / head_dim)

    def rope_forward(x, position_ids):
        angles = position_ids[..., None].to(torch.float64) * inv_freq
        emb = torch.cat((angles, angles), dim=-1)
        return emb.cos().to(x.dtype), emb.sin().to(x.dtype)

    def norm_forward(self, hidden):
        variance = hidden.pow(2).mean(-1, keepdim=True)
        return self.weight * (hidden * torch.rsqrt(variance + self.variance_epsilon))

    model.model.rotary_emb.forward = rope_forward
    for module in model.modules():
        if type(module).__name__ == "LlamaRMSNorm":
            module.forward = types.MethodType(norm_forward, module)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("dump", help="JSON from examples/dump_tiny_gradients.rs")
    parser.add_argument("--out", default=str(FIXTURE), help="fixture to write")
    parser.add_argument(
        "--hf-precision",
        action="store_true",
        help="keep transformers' float32 rotary and RMSNorm casts (control run, expect a STOP)",
    )
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # CPU only; another job may be using the GPU
    try:
        import torch
        import transformers
        from transformers import LlamaConfig, LlamaForCausalLM
    except ImportError as err:
        print(f"torch and transformers are needed: {err}")
        return 2

    dump = json.loads(Path(args.dump).read_text(encoding="utf-8"))
    cfg = dump["config"]
    tokens = dump["tokens"]
    tied = cfg["tie_word_embeddings"]

    # Rust: pair j of a head rotates dims j and j + head_dim/2 by pos * theta^(-2j/head_dim),
    # which is the Hugging Face rotate_half default. Epsilon is the f32 value widened to f64.
    hf_config = LlamaConfig(
        vocab_size=cfg["vocab_size"],
        hidden_size=cfg["hidden_size"],
        intermediate_size=cfg["intermediate_size"],
        num_hidden_layers=cfg["num_hidden_layers"],
        num_attention_heads=cfg["num_attention_heads"],
        num_key_value_heads=cfg["num_key_value_heads"],
        head_dim=cfg["hidden_size"] // cfg["num_attention_heads"],
        rms_norm_eps=cfg["rms_norm_eps"],
        rope_theta=cfg["rope_theta"],
        tie_word_embeddings=tied,
        attention_bias=False,
        mlp_bias=False,
        hidden_act="silu",
        max_position_embeddings=4096,
    )
    model = LlamaForCausalLM(hf_config).double().eval()
    if not args.hf_precision:
        use_float64_ops(model, torch, float(cfg["rope_theta"]), hf_config.head_dim)

    state = {
        name: torch.tensor(t["values"], dtype=torch.float64).reshape(t["shape"])
        for name, t in dump["tensors"].items()
    }
    loaded = model.load_state_dict(state, strict=False)
    unexpected = loaded.unexpected_keys
    missing = [k for k in loaded.missing_keys if not (tied and k == "lm_head.weight")]
    if unexpected or missing:
        print(f"weight names do not match: unexpected {unexpected}, missing {missing}")
        return 1
    if tied and model.lm_head.weight.data_ptr() != model.model.embed_tokens.weight.data_ptr():
        print("tied head is not shared with the embedding")
        return 1

    ids = torch.tensor([tokens], dtype=torch.long)
    logits = model(input_ids=ids, use_cache=False).logits
    # Rust scores logits[0..T-1] against tokens[1..T], averaged over the T-1 rows.
    loss = torch.nn.functional.cross_entropy(logits[0, :-1], ids[0, 1:], reduction="mean")
    loss.backward()
    torch_loss = loss.item()
    rust_loss = dump["loss"]
    loss_rel = abs(rust_loss - torch_loss) / abs(torch_loss)

    print(f"torch {torch.__version__}, transformers {transformers.__version__}, float64, CPU")
    print(f"loss   rust {rust_loss:.15g}   torch {torch_loss:.15g}   rel diff {loss_rel:.3e}")
    if loss_rel > LOSS_TOL:
        print(f"STOP: forward losses differ by {loss_rel:.3e} (limit {LOSS_TOL:.0e}).")
        print("Nothing was compared or written. Check epsilon, rope convention, positions, mean.")
        return 1

    named = dict(model.named_parameters())
    fixture = {}
    worst = 0.0
    print(f"{'tensor':40} {'shape':>12} {'torch norm':>13} {'rust norm':>13} {'max rel err':>12}")
    for name, t in dump["tensors"].items():
        param = named[name]
        if list(param.shape) != t["shape"]:
            print(f"{name}: shape {list(param.shape)} vs {t['shape']}")
            return 1
        torch_grad = param.grad.reshape(-1).tolist()
        rust_grad = t["grad"]
        err = rel_error(rust_grad, torch_grad)
        worst = max(worst, err)
        torch_norm = sum(v * v for v in torch_grad) ** 0.5
        rust_norm = sum(v * v for v in rust_grad) ** 0.5
        print(
            f"{name:40} {str(t['shape']):>12} {torch_norm:13.6e} {rust_norm:13.6e} {err:12.3e}"
        )
        fixture[name] = {
            "shape": t["shape"],
            "norm": torch_norm,
            "first8": torch_grad[:8],
        }

    verdict = "met" if worst <= GRAD_TOL and loss_rel <= LOSS_TOL else "NOT met"
    print(f"worst per-tensor max relative error {worst:.3e} (limit {GRAD_TOL:.0e}): {verdict}")

    out = {
        "about": "PyTorch float64 CPU oracle for tests/common/tiny_oracle.rs. Written by "
        "scripts/make_torch_gradients.py; regenerate rather than edit.",
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "config": cfg,
        "tokens": tokens,
        "loss": torch_loss,
        "tensors": fixture,
    }
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")
    return 0 if verdict == "met" else 1


if __name__ == "__main__":
    sys.exit(main())
