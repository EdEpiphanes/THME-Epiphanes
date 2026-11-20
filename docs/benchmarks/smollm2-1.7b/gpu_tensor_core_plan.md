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
