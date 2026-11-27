# SPDX-License-Identifier: AGPL-3.0-only
"""Test helper: a tiny randomly initialised LLaMA (2 layers, hidden 64, vocab 128). Not a model
anyone should draw conclusions from; it only makes the tests run in seconds on the CPU."""

import torch
from transformers import LlamaConfig, LlamaForCausalLM


def tiny_llama(seed=0):
    torch.manual_seed(seed)
    cfg = LlamaConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        tie_word_embeddings=True,
        initializer_range=0.1,  # peaked logits, so the quantization KL is clearly above zero
    )
    return LlamaForCausalLM(cfg).eval()
