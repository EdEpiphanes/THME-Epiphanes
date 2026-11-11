#!/usr/bin/env bash
# THME Benchmark Suite Shell Runner (Milestone v0.2.0.3)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

echo "Starting THME Benchmark Suite..."
if command -v python3 >/dev/null 2>&1; then
    python3 scripts/benchmark_suite.py "$@"
elif command -v python >/dev/null 2>&1; then
    python scripts/benchmark_suite.py "$@"
else
    echo "Error: Python 3 is required to run the benchmark suite." >&2
    exit 1
fi
