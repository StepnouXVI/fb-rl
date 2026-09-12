#!/usr/bin/env bash
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="/home/savvatej/Shared/conda/envs/fb-rl/bin/python"
if [ ! -f "$PYTHON_BIN" ]; then
    PYTHON_BIN="python"
fi

export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
mkdir -p outputs outputs/checkpoints results/benchmarks

echo "=============================================================================="
echo "=== Phase 1: Training Sequence Attention Translator on Medium Maze ==="
echo "=============================================================================="
$PYTHON_BIN -u scripts/train_waypoint_translators.py \
    split=medium \
    mode=sequence_attention \
    n_pairs=25000 \
    epochs=500 \
    batch_size=256 \
    lr=3e-4 \
    2>&1 | tee outputs/train_sequence_attention_medium.log

echo "=============================================================================="
echo "=== Phase 2: Comprehensive 10-Seed Benchmark (5 Tasks, 10 Episodes/Task) ==="
echo "=============================================================================="
$PYTHON_BIN -u scripts/benchmark.py \
    --split=medium \
    --num_tasks=5 \
    --episodes_per_task=10 \
    --start_seed=1 \
    --end_seed=10 \
    2>&1 | tee outputs/benchmark_medium_10seeds.log

echo "=============================================================================="
echo "=== Pipeline Completed Successfully on $(date) ==="
echo "=============================================================================="
