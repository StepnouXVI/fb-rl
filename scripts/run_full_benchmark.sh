#!/usr/bin/env bash
set -e
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

if [ -f "/home/savvatej/Shared/conda/envs/fb-rl/bin/python" ]; then
    PYTHON="/home/savvatej/Shared/conda/envs/fb-rl/bin/python"
else
    PYTHON="python3"
fi

mkdir -p outputs

echo "================================================================="
echo "  [1/2] FULL BENCHMARK: MEDIUM SPLIT (10 SEEDS, 10 EPISODES/TASK)"
echo "================================================================="
$PYTHON -u scripts/benchmark_translators.py \
    --split=medium \
    --num_tasks=10 \
    --episodes_per_task=10 \
    --start_seed=1 \
    --end_seed=10 \
    --methods "baseline" "enhanced sequence" 2>&1 | tee outputs/benchmark_10seeds_medium_10ep.log

echo "================================================================="
echo "  [2/2] FULL BENCHMARK: LARGE SPLIT (10 SEEDS, 10 EPISODES/TASK)"
echo "================================================================="
$PYTHON -u scripts/benchmark_translators.py \
    --split=large \
    --num_tasks=10 \
    --episodes_per_task=10 \
    --start_seed=1 \
    --end_seed=10 \
    --methods "baseline" "enhanced sequence" 2>&1 | tee outputs/benchmark_10seeds_large_10ep.log

echo "================================================================="
echo "  ALL BENCHMARKS (MEDIUM + LARGE, 10 SEEDS, 10 EPISODES) FINISHED!"
echo "================================================================="
