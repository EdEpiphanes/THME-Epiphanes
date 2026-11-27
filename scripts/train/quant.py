# SPDX-License-Identifier: AGPL-3.0-only
"""THME's Q4_0 round trip in torch, and the list of tensors a `SetPrecision` Q4_0 recipe touches.

The arithmetic is src/exporter/conversion.rs `encode_q4_0_block` followed by the decoder
`GgufBlockDecoder::unpack_q4_0`, all in float32 except the stored scale, which is FP16:

    extreme = the value of largest magnitude in the 32-value block (first one on a tie)
    scale   = extreme / -8                       (float32)
    stored  = FP16(scale), round to nearest even
    code    = clamp(trunc(v * (1/scale) + 8.5), 0, 15)    (1/scale uses the float32 scale)
    result  = float32(stored) * (code - 8)

Blocks are 32 consecutive values of the row-major flattened tensor, as in `fake_quantize`.
"""

import torch

BLOCK = 32
# src/ir/tensor.rs `extract_layer_index`: the name segment after one of these is the layer.
_LAYER_MARKERS = ("layers", "layer", "h", "blocks", "block", "blk")


def q4_0_roundtrip(weight):
    """Returns `weight` quantized to Q4_0 and dequantized, float32, same shape. Raises ValueError
    where THME's encoder returns an error (scale outside FP16 range or underflowing to zero)."""
    w = weight.detach().to(torch.float32)
    if w.numel() % BLOCK:
        raise ValueError(f"Q4_0 requires an element count divisible by {BLOCK}, found {w.numel()}")
    if not torch.isfinite(w).all():
        raise ValueError("Q4_0 input holds a NaN or infinity")
    b = w.reshape(-1, BLOCK)
    extreme = b.gather(1, b.abs().argmax(dim=1, keepdim=True))
    extreme = torch.where(extreme == 0, torch.zeros_like(extreme), extreme)  # THME folds from +0.0
    scale = extreme / -8.0
    half = scale.to(torch.float16)
    stored = half.to(torch.float32)
    if not torch.isfinite(stored).all():
        raise ValueError("Q4_0 scale exceeds finite FP16 range")
    if ((scale != 0) & (half == 0)).any():
        raise ValueError("Q4_0 scale underflows to zero in FP16")
    inv = torch.where(scale == 0, torch.zeros_like(scale), 1.0 / scale)
    code = (b * inv + 8.5).to(torch.int32).clamp(0, 15)  # int cast truncates, like Rust `as i32`
    return (stored * (code - 8).to(torch.float32)).reshape(w.shape)


def q4_0_param_names(state):
    """Names in a HF LLaMA state dict (name -> tensor) that THME rounds to Q4_0 for a recipe of
    type SetPrecision, format Q4_0, every layer, role null.

    THME selects a tensor when its name carries a layer index (`model.layers.<n>.`), and the
    measured path then rounds only tensors with two or more dimensions
    (`load_tensor`: "1-D tensors (norms, biases) stay as stored"). For LLaMA that is exactly
    q, k, v, o, gate, up and down per layer. Embeddings, the final norm and `lm_head` have no layer
    index, so a role-null recipe never reaches them."""
    return sorted(n for n, t in state.items() if _layer_index(n) is not None and t.ndim >= 2)


def _layer_index(name):
    parts = name.split(".")
    for i in range(len(parts) - 1):
        if parts[i] in _LAYER_MARKERS and parts[i + 1].isdigit():
            return int(parts[i + 1])
    return None


def quantize_model_(model):
    """Replaces, in place, every weight THME quantizes by its Q4_0 round trip. Returns the names."""
    names = q4_0_param_names(model.state_dict())
    params = dict(model.named_parameters())
    with torch.no_grad():
        for n in names:
            params[n].copy_(q4_0_roundtrip(params[n]))
    return names
