# Empirical Roofline Profiling with THME: Hardware Ceilings, Arithmetic Intensity, and 4-Tier Cache Simulation

## Executive Summary

Hardware Roofline profiling models the relationship between operational arithmetic intensity (FLOPs per byte transferred from DRAM) and the maximum attainable execution throughput (TFLOPS) on a target processor. It establishes whether neural network layers are bottlenecked by raw arithmetic execution throughput (**Compute-Bound**) or off-chip memory bandwidth (**Memory-Bound**).

THME features an integrated empirical Roofline and micro-kernel profiling subsystem. It evaluates the exact tensor operations of Transformer models (Linear GEMM projections, Scaled Dot-Product Attention, FlashAttention-2 tiling, SwiGLU FFN blocks, and RMSNorm), calculates machine balance knee boundaries across modern accelerators, simulates multi-level cache residency through a 4-tier memory hierarchy (L1, L2, L3, DRAM), and models dynamic TDP power dissipation with thermal throttling.

---

## Theoretical Foundations and Physics

### 1. Operational Arithmetic Intensity

Arithmetic intensity $I$ measures the computational work performed per byte of data transferred across the memory bus:

$$I = \frac{\text{Total Floating-Point Operations (FLOPs)}}{\text{Total Memory Traffic (DRAM Bytes)}} \quad [\text{FLOPs/byte}]$$

### 2. The Machine Balance Knee Point

Every accelerator architecture possesses a characteristic machine balance knee point $I^*$, defined as the ratio of peak arithmetic compute throughput $P_{\text{peak}}$ to peak memory bandwidth $B_{\text{dram}}$:

$$I^* = \frac{P_{\text{peak}} \text{ (FLOP/s)}}{B_{\text{dram}} \text{ (Bytes/s)}} = \frac{P_{\text{peak}} \text{ (TFLOP/s)}}{B_{\text{dram}} \text{ (TB/s)}} \quad [\text{FLOPs/byte}]$$

The theoretical attainable throughput ceiling $P(I)$ is bounded by:

$$P(I) = \min\left(P_{\text{peak}}, I \cdot B_{\text{dram}}\right)$$

- **Memory-Bound Regime ($I < I^*$)**: Attainable throughput scales linearly with arithmetic intensity: $P(I) = I \cdot B_{\text{dram}}$. The compute units are starved waiting for memory loads. Speedups require reducing memory traffic (e.g., quantization, BitNet b1.58, kernel fusion).
- **Compute-Bound Regime ($I \ge I^*$)**: Attainable throughput is saturated at peak compute capacity: $P(I) = P_{\text{peak}}$. Memory bandwidth is no longer the limiting factor. Speedups require algorithmic simplification, pruning, or structural architectural modifications.

### 3. Layer Complexity Formulations

Transformer operations exhibit starkly different arithmetic intensities between token generation (decode) and context ingestion (prefill):

| Kernel Type | Mathematical FLOPs | Memory Traffic (Bytes) | Operational Regime |
|:---|:---|:---|:---|
| **Linear GEMM (Decode $B=1$)** | $2 \cdot 1 \cdot K \cdot N$ | $\text{size}(W) + \text{size}(X) + \text{size}(Y) \approx 2 K N$ bytes | $I \approx 1.0$ FLOP/Byte (Strictly Memory-Bound) |
| **Linear GEMM (Prefill $S \ge 2048$)** | $2 \cdot S \cdot K \cdot N$ | $\text{size}(W) + 2 S K + 2 S N \approx 2 K N$ bytes | $I \approx S$ FLOP/Byte (Compute-Bound) |
| **Standard SDPA Attention** | $4 B H S^2 D + 3 B H S^2$ | $O(S^2)$ quadratic score matrix DRAM traffic | Memory-Bandwidth & VRAM Capacity Bottleneck |
| **FlashAttention-2** | $4 B H S^2 D + 3 B H S^2$ | $O(S)$ linear DRAM traffic (SRAM tiled execution) | Compute-Bound across moderate to large context lengths |
| **SwiGLU FFN Block** | $6 B S D D_{\text{ffn}} + 3 B S D_{\text{ffn}}$ | $3 D D_{\text{ffn}} \cdot b_w + \text{activations}$ | High arithmetic density due to 3 parallel matrices |
| **RMSNorm / LayerNorm** | $3 B S D$ | $2 B S D \cdot b_w$ (read input, write output) | $I \approx 0.5 - 0.75$ FLOP/Byte (Memory-Bound) |

---

## Predefined Hardware Target Database

THME includes a hardware database spanning modern enterprise and client accelerators:

| Hardware Target | Code Identifier | Architecture | Peak Compute (FP16) | Memory Bandwidth | Machine Balance Knee ($I^*$) |
|:---|:---|:---|:---|:---|:---|
| **NVIDIA H100 SXM5** | `h100` | Hopper GH100 | 989.4 TFLOPS | 3350.0 GB/s (HBM3) | 295.34 FLOP/Byte |
| **NVIDIA GeForce RTX 4090** | `rtx4090` | Ada Lovelace AD102 | 165.2 TFLOPS | 1008.0 GB/s (GDDR6X)| 163.89 FLOP/Byte |
| **Apple M3 Max** | `m3_max` | Apple Silicon | 32.5 TFLOPS | 400.0 GB/s (Unified) | 81.25 FLOP/Byte |
| **AMD Instinct MI300X** | `mi300x` | CDNA 3 | 1307.4 TFLOPS | 5300.0 GB/s (HBM3) | 246.68 FLOP/Byte |
| **Intel Emerald Rapids** | `emerald_rapids`| Xeon Platinum 8592+ | 8.2 TFLOPS | 307.0 GB/s (DDR5) | 26.71 FLOP/Byte |

---

## Step 1: Default Profiling on NVIDIA H100 SXM5

Execute Roofline profiling using the default target (`h100`) against the 12-layer synthetic baseline:

```bash
./target/release/thme roofline
```

Expected output:

```text
Using default 12-layer synthetic LLM baseline for roofline profiling.
================ ROOFLINE CEILING MODEL [NVIDIA H100 SXM5] ================
Peak Compute: 989.4 TFLOPS (FP16) | Bandwidth: 3350.0 GB/s | Knee: 295.34 FLOP/Byte

   TFLOPS
  989.4 |-------------------------------+-----------------------------+ (Peak Compute Ceiling)
        |                              /|
        |                             / |
        |                            /  |
        |                           /   |
        |       MEMORY BOUND       /    |        COMPUTE BOUND
        |                         /     |
        |                        /      |
        |                       /       |
        |                      /        |
        |                     /         |
        |                    /          |
        |                   /           |
        |                  /            |
        |                 /             |
        |                /              |
        |               /               |
    0.1 +--------------+----------------+-----------------------------+---> Arithmetic Intensity (FLOP/Byte)
       0.1            1.0             295.3 (Knee)                   1000.0

OPERATIONAL KERNEL MAPPINGS:
 [1] Q Projection                 | Int:  682.67 FLOP/B | Attain:  989.4 TFLOPS | COMPUTE-BOUND
 [2] K Projection                 | Int:  682.67 FLOP/B | Attain:  989.4 TFLOPS | COMPUTE-BOUND
 [3] V Projection                 | Int:  682.67 FLOP/B | Attain:  989.4 TFLOPS | COMPUTE-BOUND
 [4] Output Projection            | Int:  682.67 FLOP/B | Attain:  989.4 TFLOPS | COMPUTE-BOUND
 [5] FlashAttention-2 Kernel      | Int: 1036.00 FLOP/B | Attain:  989.4 TFLOPS | COMPUTE-BOUND
 [6] SwiGLU FFN Sub-Block         | Int: 1068.77 FLOP/B | Attain:  989.4 TFLOPS | COMPUTE-BOUND
 [7] RMSNorm Kernel               | Int:    0.75 FLOP/B | Attain:    2.5 TFLOPS | MEMORY-BOUND

--------------------------------------------------------------------------------
EXECUTION PROFILE SUMMARY:
 - Target Device       : NVIDIA H100 SXM5 (Architecture: Hopper GH100)
 - Peak Compute Ceiling: 989.4 TFLOPS (FP16)
 - Memory Bandwidth    : 3350.0 GB/s (3.35 TB/s)
 - Machine Balance Knee: 295.34 FLOPs/Byte
 - Thermal Throttling  : 0.0% (Throttle Factor: 1.00)
 - Bound Breakdown     : 1 Memory-Bound (14.3%), 6 Compute-Bound (85.7%)
 - Cache Simulation    : Effective Latency: 2.11 ns | Effective BW: 19517.6 GB/s | DRAM Spill: 0.08 MB
================================================================================
```

---

## Step 2: Profiling Across Heterogeneous Accelerators

Target alternative hardware architectures by supplying the `--hardware` flag.

Profile for an NVIDIA GeForce RTX 4090:

```bash
./target/release/thme roofline --hardware rtx4090
```

Expected output:

```text
================ ROOFLINE CEILING MODEL [NVIDIA GeForce RTX 4090] ================
Peak Compute: 165.2 TFLOPS (FP16) | Bandwidth: 1008.0 GB/s | Knee: 163.89 FLOP/Byte
...
EXECUTION PROFILE SUMMARY:
 - Target Device       : NVIDIA GeForce RTX 4090 (Architecture: Ada Lovelace AD102)
 - Peak Compute Ceiling: 165.2 TFLOPS (FP16)
 - Memory Bandwidth    : 1008.0 GB/s (1.01 TB/s)
 - Machine Balance Knee: 163.89 FLOPs/Byte
 - Thermal Throttling  : 0.0% (Throttle Factor: 1.00)
 - Bound Breakdown     : 1 Memory-Bound (14.3%), 6 Compute-Bound (85.7%)
 - Cache Simulation    : Effective Latency: 11.77 ns | Effective BW: 4214.1 GB/s | DRAM Spill: 0.86 MB
================================================================================
```

Profile for Apple Silicon (M3 Max with unified memory):

```bash
./target/release/thme roofline --hardware m3_max
```

Profile for AMD Instinct MI300X:

```bash
./target/release/thme roofline --hardware mi300x
```

---

## Step 3: Evaluating Batch Size and Context Scaling

The transition between token generation (autoregressive decode) and prompt evaluation (prefill) shifts a model's operational characteristics completely.

### Autoregressive Decode ($B=1, S=1$)

During single-token generation, weights must be read from off-chip DRAM for every emitted token, resulting in low arithmetic intensity:

```bash
./target/release/thme roofline --batch-size 1 --seq-len 1
```

Expected output:

```text
OPERATIONAL KERNEL MAPPINGS:
 [1] Q Projection                 | Int:    1.00 FLOP/B | Attain:    3.3 TFLOPS | MEMORY-BOUND
 [2] K Projection                 | Int:    1.00 FLOP/B | Attain:    3.3 TFLOPS | MEMORY-BOUND
 [3] V Projection                 | Int:    1.00 FLOP/B | Attain:    3.3 TFLOPS | MEMORY-BOUND
 [4] Output Projection            | Int:    1.00 FLOP/B | Attain:    3.3 TFLOPS | MEMORY-BOUND
 [5] FlashAttention-2 Kernel      | Int:    0.51 FLOP/B | Attain:    1.7 TFLOPS | MEMORY-BOUND
 [6] SwiGLU FFN Sub-Block         | Int:    1.00 FLOP/B | Attain:    3.4 TFLOPS | MEMORY-BOUND
 [7] RMSNorm Kernel               | Int:    0.50 FLOP/B | Attain:    1.7 TFLOPS | MEMORY-BOUND

--------------------------------------------------------------------------------
EXECUTION PROFILE SUMMARY:
 - Bound Breakdown     : 7 Memory-Bound (100.0%), 0 Compute-Bound (0.0%)
```

During decode, all kernels are **100% Memory-Bound**. Attainable throughput is limited to 3.3 TFLOPS out of the H100's 989.4 TFLOPS theoretical peak.

### Batched Prefill ($B=4, S=2048$)

When evaluating prompts, weights are reused across $B \times S = 8192$ tokens:

```bash
./target/release/thme roofline --batch-size 4 --seq-len 2048
```

Arithmetic intensity jumps to over 2,000 FLOPs/Byte, saturating the H100 compute ceiling at 989.4 TFLOPS.

---

## Step 4: FlashAttention-2 vs Standard Attention Tiling

Standard scaled dot-product attention writes the intermediate $S \times S$ score matrix to DRAM, incurring quadratic memory traffic $O(S^2)$. FlashAttention-2 computes softmax reduction incrementally within on-chip SRAM blocks, keeping DRAM traffic at $O(S)$.

By default, THME enables FlashAttention-2 modeling. To observe the degradation caused by un-tiled standard attention:

```bash
./target/release/thme roofline --flash-attn=false --seq-len 4096
```

At $S = 4096$, disabling FlashAttention causes attention arithmetic intensity to collapse due to massive intermediate DRAM score matrix traffic, shifting the attention kernel into the memory-bound regime.

---

## Step 5: 4-Tier Memory Hierarchy Cache Simulation

THME includes an analytical cache simulator modeling the memory hierarchy across four tiers:
1. **L1 Cache**: 0.5 ns latency, 128 KB per core.
2. **L2 Cache**: 4.0 ns latency, 50 MB chip-wide.
3. **L3 Cache**: 15.0 ns latency (or system level cache).
4. **DRAM (HBM / GDDR)**: 60.0 ns latency.

The simulator evaluates the active working set size $W_{\text{active}} = \text{Weights} + \text{Activations} + \text{KV Cache}$:
- If $W_{\text{active}} \le \text{Capacity}_{\text{L2}}$, the layer executes with high effective bandwidth and low latency penalty.
- If $W_{\text{active}} > \text{Capacity}_{\text{L2}}$, working set eviction forces DRAM stalls, reported in the telemetry as **DRAM Spill Volume** and elevated **Effective Latency**.

---

## Step 6: Dynamic TDP Power and Thermal Throttling

Under sustained load or elevated ambient temperatures, GPU Dynamic Voltage and Frequency Scaling (DVFS) throttles core clocks to prevent thermal runaway.

Simulate an RTX 4090 running at an elevated die temperature of 85.0 Celsius:

```bash
./target/release/thme roofline --hardware rtx4090 --thermal --temperature 85.0
```

Expected output:

```text
================ ROOFLINE CEILING MODEL [NVIDIA GeForce RTX 4090] ================
Peak Compute: 148.7 TFLOPS (FP16) | Bandwidth: 1008.0 GB/s | Knee: 163.89 FLOP/Byte
...
EXECUTION PROFILE SUMMARY:
 - Target Device       : NVIDIA GeForce RTX 4090 (Architecture: Ada Lovelace AD102)
 - Peak Compute Ceiling: 148.7 TFLOPS (FP16)
 - Memory Bandwidth    : 1008.0 GB/s (1.01 TB/s)
 - Machine Balance Knee: 163.89 FLOPs/Byte
 - Thermal Throttling  : 10.0% (Throttle Factor: 0.90)
 - Bound Breakdown     : 1 Memory-Bound (14.3%), 6 Compute-Bound (85.7%)
================================================================================
```

The thermal model applies a throttle factor of $0.90$ (10.0% clock reduction), reducing peak compute capacity from 165.2 TFLOPS down to 148.7 TFLOPS.

---

## Step 7: Machine-Readable JSON Telemetry for CI/CD

Integrate Roofline metrics into automated CI performance regressions:

```bash
./target/release/thme roofline --hardware h100 --batch-size 1 --seq-len 2048 --json
```

Expected output:

```json
{
  "hardware_name": "NVIDIA H100 SXM5",
  "precision": "FP16",
  "peak_compute_tflops": 989.4,
  "memory_bandwidth_gbps": 3350.0,
  "knee_intensity_flop_per_byte": 295.34328358208955,
  "points": [
    {
      "name": "Q Projection",
      "arithmetic_intensity": 682.6666666666666,
      "attainable_tflops": 989.4,
      "classification": {
        "ComputeBound": {
          "compute_saturation_ratio": 1.0
        }
      },
      "efficiency_pct": 100.0,
      "recommended_optimizations": [
        "Apply 2:4 structured sparsity to double effective tensor compute.",
        "Utilize low-precision Tensor Cores (INT8/FP8) for compute speedup."
      ]
    },
    {
      "name": "FlashAttention-2 Kernel",
      "arithmetic_intensity": 1036.0,
      "attainable_tflops": 989.4,
      "classification": {
        "ComputeBound": {
          "compute_saturation_ratio": 1.0
        }
      },
      "efficiency_pct": 100.0,
      "recommended_optimizations": [
        "Apply 2:4 structured sparsity to double effective tensor compute.",
        "Utilize low-precision Tensor Cores (INT8/FP8) for compute speedup."
      ]
    },
    {
      "name": "SwiGLU FFN Sub-Block",
      "arithmetic_intensity": 1068.7740995509698,
      "attainable_tflops": 989.4,
      "classification": {
        "ComputeBound": {
          "compute_saturation_ratio": 1.0
        }
      },
      "efficiency_pct": 100.0,
      "recommended_optimizations": [
        "Apply 2:4 structured sparsity to double effective tensor compute.",
        "Utilize low-precision Tensor Cores (INT8/FP8) for compute speedup."
      ]
    },
    {
      "name": "RMSNorm Kernel",
      "arithmetic_intensity": 0.7498169392238223,
      "attainable_tflops": 2.5118867463998047,
      "classification": {
        "MemoryBound": {
          "memory_bound_factor": 0.002538798005255513,
          "headroom_to_knee": 294.59346664286574
        }
      },
      "efficiency_pct": 0.2538798005255513,
      "recommended_optimizations": [
        "Quantize weights to BitNet b1.58 or Q4_K to reduce memory traffic.",
        "Apply kernel fusion (SwiGLU, LayerNorm) to keep intermediates in SRAM."
      ]
    }
  ]
}
```

Verify that FlashAttention remains compute-bound via `jq`:

```bash
./target/release/thme roofline --json | jq '.points[] | select(.name=="FlashAttention-2 Kernel") | .classification.ComputeBound != null'
```

---

## Step 8: Interactive Studio GUI Tab 4 Roofline Deck

Launch the desktop studio to interact with the visual Roofline deck:

```bash
./target/release/thme gui
```

Navigate to **Tab 4: Hardware Roofline & Speculative Deck**:
- Select target hardware from the dropdown (`H100`, `RTX 4090`, `M3 Max`, `MI300X`, `Emerald Rapids`).
- View the real-time log-log Roofline plot with kernel coordinates mapped dynamically.
- Drag the **Batch Size** ($B$) and **Sequence Length** ($S$) sliders to watch kernels transition between the memory-bound diagonal slope and the horizontal compute ceiling.
- Toggle **Simulate Thermal Throttling** and adjust the ambient temperature slider to observe real-time ceiling contraction.

---

## CLI Parameter Reference

| Flag | Short | Type | Default | Description |
|:---|:---|:---|:---|:---|
| `--model` | `-m` | Path | None | Path to model file (`.safetensors` or `.gguf`). Uses synthetic baseline if omitted |
| `--hardware` | `-w` | String | "h100" | Hardware target (`h100`, `rtx4090`, `m3_max`, `mi300x`, `emerald_rapids`) |
| `--batch-size` | `-b` | Integer | 1 | Inference batch size $B$ |
| `--seq-len` | `-s` | Integer | 2048 | Sequence context length $S$ |
| `--flash-attn` | | Boolean | true | Model FlashAttention-2 SRAM tiling arithmetic intensity |
| `--thermal` | | Flag | false | Simulate dynamic TDP power and thermal throttling |
| `--temperature`| | Float | None | Operating temperature in Celsius (e.g. 25.0, 75.0, 85.0) |
| `--json` | | Flag | false | Output report in machine-readable JSON format |

---

## Troubleshooting and Common Issues

### Issue 1: Unrecognized Hardware Target
- **Symptom**: CLI defaults to NVIDIA H100 SXM5 when passing a custom target.
- **Cause**: The hardware identifier string did not match supported keys.
- **Remedy**: Use one of the five exact identifiers: `h100`, `rtx4090`, `m3_max`, `mi300x`, or `emerald_rapids`.

### Issue 2: Excessive DRAM Spill
- **Symptom**: `Cache Simulation: DRAM Spill: 142.5 MB`.
- **Cause**: Sequence length or hidden dimension exceeds on-chip cache capacity, forcing high latency DRAM thrashing.
- **Remedy**: Reduce batch size or apply quantization (`INT8` / `Ternary`) to reduce activation working set footprint.

---

## Next Steps and Related Guides

- [Speculative Decoding Tutorial](speculative_decoding.md): Break decode memory bandwidth bottlenecks with draft model speculative execution.
- [PyTorch & CUDA Interoperability Guide](pytorch_cuda_interop.md): Generate CUDA training harnesses and merge LoRA adapters.
- [Quickstart Guide](quickstart.md): Zero-copy inspection, layer sensitivity profiling, and quality gates.
- [README Overview](../README.md): Central architectural documentation and production roadmap.
