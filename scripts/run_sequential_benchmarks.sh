#!/usr/bin/env bash
set -e

echo "========================================================="
echo " Starting Full Sequential Benchmark Pipeline (Large & Medium)"
echo " Date: $(date)"
echo "========================================================="

export PYTHONUNBUFFERED=1
export JAX_PLATFORMS=cuda,cpu
mkdir -p outputs results/benchmarks

# 1. Benchmark on Large Maze
echo ">>> [1/2] Launching AntMaze Large Benchmark (10 Seeds, 5 Tasks)..."
python scripts/benchmark_translators.py \
    --split=large \
    --num_tasks=5 \
    --episodes_per_task=10 \
    --start_seed=1 \
    --end_seed=10 \
    --output_dir=results/benchmarks 2>&1 | tee outputs/benchmark_large_sequential.log

# 2. Benchmark on Medium Maze
echo ">>> [2/2] Launching AntMaze Medium Benchmark (10 Seeds, 5 Tasks)..."
python scripts/benchmark_translators.py \
    --split=medium \
    --num_tasks=5 \
    --episodes_per_task=10 \
    --start_seed=1 \
    --end_seed=10 \
    --output_dir=results/benchmarks 2>&1 | tee outputs/benchmark_medium_sequential.log

echo "========================================================="
echo " All Benchmarks Completed Successfully!"
echo " Results:"
echo " - Large:  results/benchmarks/benchmark_summary_10seeds_large.csv"
echo " - Medium: results/benchmarks/benchmark_summary_10seeds_medium.csv"
echo " Date: $(date)"
echo "========================================================="
