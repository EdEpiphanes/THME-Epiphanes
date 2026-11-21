# GPU tensor-core and int8 routes: pre-registered rules (THME-3087, THME-3088)

Written before any measurement of these routes. Device: RTX 4060 Ti 16 GB, Vulkan, driver 616.92.
Reference for context: llama.cpp b12584 on the same machine, pp1024 about 11,300 to 14,300 tokens
per second (`docs/benchmarks/smollm2-1.7b/data/llama_cpp_reference.json` on branch
`feat/llama-cpp-reference`). Current THME path: about 2,100 tokens per second, matmul about 9.5 TFLOPS.

Method for every step: 16,384-token slice of the WikiText-2 test split, SmolLM2-1.7B, GPU only,
`examples/gpu_speed.rs` (`gemm`, `profile`, `ppl`), release build, same binary with flags,
one change at a time, short runs because the card reaches 83 C on long passes.

## B. Cooperative matrices (f16 inputs, f32 accumulate)

Go if all hold, otherwise no-go with the numbers:

- at least 25 TFLOPS isolated on the large 1.7B shapes (`gpu_speed gemm`);
- at least 6,000 tokens per second on the 16,384-token slice;
- perplexity within 0.01 nats of the f32 path on that slice.

Expected parity: f16 inputs give about 1e-3 relative error per product; the report states the
measured logit and perplexity difference.

## C. Packed 4 x 8 integer dot product

Weights Q8_0 (and Q4_0 unpacked to int8), activations quantized per 32-block to int8 on the fly,
integer accumulate with `dot4I8Packed`, scale to f32. Go if all hold:

- at least 20 TOPS-equivalent isolated;
- at least 6,000 tokens per second on the slice;
- perplexity change on the slice reported next to the Q8_0 f32-activation figure (8.377).

## D. Next cost

Profile again after the best route, fix the largest remaining cost, stop at 9,000 tokens per second
on the slice or when a step gives under 5 percent.

## Results (20 November 2026)

Reports: `data/gpu_coop_gemm_*.json`, `data/gpu_coop_slice_*.json`, `data/gpu_coop_profile_*.json`.
RTX 4060 Ti, Vulkan, driver 616.92, release build, GPU only, two runs per slice arm, shared machine.

- **B (cooperative matrices): go.** Isolated 28.0, 35.7, 26.1, 40.0, 27.8 TFLOPS on the five shapes
  (rule: at least 25 on large shapes); slice 7,000 to 7,020 tokens per second with f16 weights
  (rule: 6,000); perplexity 8.37286 against 8.37288 (rule: within 0.01 nats).
- **C (int8 dot product): no-go.** Isolated 10.1 to 12.9 TOPS-equivalent (rule: 20); slice 3,028
  (Q8_0) and 3,045 (Q4_0) tokens per second (rule: 6,000); perplexity +0.0017 and +0.0020 nats.
- **D:** the next cost after the matmul was attention (58 ms of a 1,024-token window); a cooperative
  attention for head size 64 took it to 13 ms. The slice stands at about 7,000 tokens per second,
  below the 9,000 stop line; the largest remaining costs are the matmul itself (99 ms) and the
  repeated activation conversions plus SwiGLU (about 17 ms together).
