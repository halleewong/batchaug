#!/bin/bash
# Run all benchmarks sequentially and save outputs to .log files.
# Usage: conda run -n interseg3d bash examples/run_all_benchmarks.sh

set -e
cd "$(dirname "$0")/.."

echo "=== benchmark_time.py ==="
python examples/benchmark_time.py 2>&1 | tee examples/benchmark_time.log

echo "=== benchmark_memory.py ==="
python examples/benchmark_memory.py 2>&1 | tee examples/benchmark_memory.log

echo "=== benchmark_pipeline.py ==="
python examples/benchmark_pipeline.py 2>&1 | tee examples/benchmark_pipeline.log

echo "=== benchmark_triton.py ==="
python examples/benchmark_triton.py 2>&1 | tee examples/benchmark_triton.log

echo "=== benchmark_triton_pipeline.py ==="
python examples/benchmark_triton_pipeline.py 2>&1 | tee examples/benchmark_triton_pipeline.log

echo "=== benchmark_fused.py ==="
python examples/benchmark_fused.py 2>&1 | tee examples/benchmark_fused.log

echo "All benchmarks complete."
