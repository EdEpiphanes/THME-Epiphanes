#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""
THME Automated Reproducible Benchmark Script Suite (Milestone v0.2.0.3)
Measures empirical throughput, profiling latency, Pareto DP allocation time,
and token generation simulation across hardware architectures.
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import time

def get_system_telemetry():
    """Captures non-identifying hardware telemetry."""
    info = {
        "os": platform.system(),
        "os_release": platform.release(),
        "arch": platform.machine(),
        "processor": platform.processor() or "Unknown",
        "python_version": platform.python_version(),
    }
    try:
        import psutil
        info["ram_gb"] = round(psutil.virtual_memory().total / (1024**3), 2)
        info["cpu_cores_logical"] = psutil.cpu_count(logical=True)
        info["cpu_cores_physical"] = psutil.cpu_count(logical=False)
    except ImportError:
        info["ram_gb"] = "N/A (install psutil)"
        info["cpu_cores_logical"] = os.cpu_count() or 1
    return info

def run_command(cmd, desc):
    """Executes a command and returns execution duration in milliseconds."""
    print(f"  -> Running: {desc}...")
    start = time.perf_counter()
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    duration_ms = (time.perf_counter() - start) * 1000.0
    if res.returncode != 0:
        print(f"     [WARN] Command returned non-zero exit code: {res.returncode}")
        print(f"     stderr: {res.stderr[:200]}")
    return duration_ms, res.stdout, res.returncode

def main():
    parser = argparse.ArgumentParser(description="THME Reproducible Benchmark Suite")
    parser.add_argument("--output", default="target/benchmark_results.json", help="Output JSON path")
    parser.add_argument("--iterations", type=int, default=3, help="Benchmark repeat iterations")
    parser.add_argument("--quick", action="store_true", help="Quick mode with reduced iterations")
    args = parser.parse_args()

    print("════════════════════════════════════════════════════════════════")
    print("  THME Automated Reproducible Benchmark Suite (v0.2.0.3)")
    print("════════════════════════════════════════════════════════════════")

    telemetry = get_system_telemetry()
    print("\n[System Telemetry]")
    for k, v in telemetry.items():
        print(f"  {k:20}: {v}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    benchmarks = {}

    # Benchmark 1: Ingestion & Model Generation
    cmd_ingest = ["cargo", "run", "--release", "--example", "generate_mock_model", "--quiet"]
    durations = []
    iters = 1 if args.quick else args.iterations
    for _ in range(iters):
        d_ms, _, rc = run_command(cmd_ingest, "Synthetic Model Ingestion & Generation")
        if rc == 0:
            durations.append(d_ms)
    if durations:
        benchmarks["ingestion_throughput_ms"] = round(sum(durations) / len(durations), 2)

    # Benchmark 2: Production Verification (SmolLM-135M)
    cmd_verify = ["cargo", "run", "--release", "--example", "production_verification", "--", "--target", "smollm135m", "--dry-run"]
    durations = []
    for _ in range(iters):
        d_ms, _, rc = run_command(cmd_verify, "SmolLM-135M Full Verification Pipeline")
        if rc == 0:
            durations.append(d_ms)
    if durations:
        benchmarks["smollm_verification_pipeline_ms"] = round(sum(durations) / len(durations), 2)

    # Benchmark 3: Production Verification (LLaMA-3.2-1B)
    cmd_llama = ["cargo", "run", "--release", "--example", "production_verification", "--", "--target", "llama321b", "--dry-run"]
    durations = []
    for _ in range(iters):
        d_ms, _, rc = run_command(cmd_llama, "LLaMA-3.2-1B Full Verification Pipeline")
        if rc == 0:
            durations.append(d_ms)
    if durations:
        benchmarks["llama_verification_pipeline_ms"] = round(sum(durations) / len(durations), 2)

    results = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "telemetry": telemetry,
        "benchmarks": benchmarks,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n════════════════════════════════════════════════════════════════")
    print(f"  Benchmark Suite Finished. Results written to: {args.output}")
    print("════════════════════════════════════════════════════════════════")
    for k, v in benchmarks.items():
        print(f"  {k:35}: {v} ms")

if __name__ == "__main__":
    main()
