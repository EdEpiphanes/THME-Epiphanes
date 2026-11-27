#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Adds the cross-check and the reading to the measured QAD-LoRA report (run once after run_qad_comparison.py)."""

import json
import sys

path = "docs/benchmarks/smollm-135m/data/qad_lora.json"
doc = json.load(open(path, encoding="utf-8"))
thme = float(sys.argv[1])
torch_ppl = doc["results"]["scored_runs"][0]["perplexity"]
doc["results"]["cross_check"] = {
    "what": "thme evaluate on the merged float16 model of seed 1 (FP32 identity recipe, same test token file, THME CUDA path) against the PyTorch evaluation of the same run",
    "thme_perplexity": thme,
    "pytorch_perplexity": torch_ppl,
    "relative_difference": (thme - torch_ppl) / torch_ppl,
    "allowed": 0.01,
    "passed": abs(thme - torch_ppl) / torch_ppl <= 0.01,
}
doc["results"]["reading"] = (
    "With the pre-registered rule, LoRA recovery on the Q4_0 student beats round-to-nearest (by 3.29), GPTQ (by 1.82) and AWQ (by 1.63) on the SmolLM-135M test split, "
    "each far above 2 sigma (0.065); the three seeds agree to 0.018. It recovers most of the quantization loss (round-to-nearest 3.90 above the ceiling, this 0.61 above) "
    "but still fails the gate's 0.45 margin. This is a comparison against methods that add no parameters: the recovered model is Q4_0 plus a float32 adapter of "
    "19,593,064 bytes (4,884,480 trainable parameters, about 3.5 percent of the model), so it is larger than Q4_0 alone, and the adapter was not folded back into 4-bit weights."
)
doc["results"]["limits_found"] = [
    "the pilot's best learning rate (1e-3) is the edge of the pre-registered grid, so a higher rate might do better; the grid was not extended after the result, per the protocol",
    "one model, one corpus, one configuration, 256 steps; no claim about other models or longer training",
    "the adapter is evaluated unmerged in float32; the float16 merge adds its own small rounding (checked: THME on the merged file agrees to 0.002 percent)",
    "training scores every position while evaluation drops each window's first token (stated in the protocol)",
]
json.dump(doc, open(path, "w", encoding="utf-8", newline="\n"), indent=2)
open(path, "a").write("\n")
print("cross-check", doc["results"]["cross_check"]["relative_difference"], doc["results"]["cross_check"]["passed"])
