#!/usr/bin/env bash
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-$(which python3 2>/dev/null || which python 2>/dev/null)}"
if [ -f "/opt/homebrew/Caskroom/miniconda/base/envs/fb-rl/bin/python" ]; then
    PYTHON_BIN="/opt/homebrew/Caskroom/miniconda/base/envs/fb-rl/bin/python"
fi

export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
mkdir -p outputs outputs/checkpoints results/checkpoints results/benchmarks results/data

echo "=============================================================================="
echo "=== Phase 1: Medium Maze Training (Sequence Attention Translator) ==="
echo "=============================================================================="
$PYTHON_BIN -u scripts/train_waypoint_translators.py \
    split=medium \
    mode=sequence_attention \
    n_pairs=25000 \
    epochs=500 \
    batch_size=256 \
    lr=3e-4 \
    2>&1 | tee outputs/train_seq_attn_medium.log

echo "=============================================================================="
echo "=== Phase 2: Medium Maze Benchmark (10 Seeds, 5 Tasks, 10 Episodes/Task) ==="
echo "=============================================================================="
$PYTHON_BIN -u scripts/benchmark.py \
    --split=medium \
    --num_tasks=5 \
    --episodes_per_task=10 \
    --start_seed=1 \
    --end_seed=10 \
    2>&1 | tee outputs/benchmark_medium_10seeds.log

echo "=============================================================================="
echo "=== Phase 3: Large Maze Distillation Training (Gated-Attn) ==="
echo "=============================================================================="
$PYTHON_BIN -u scripts/train_distillation.py \
    --split=large \
    --model_type=gated_attn \
    --epochs=100 \
    --batch_size=256 \
    --n_pairs=25000 \
    --lr=3e-4 \
    2>&1 | tee outputs/train_distilled_jax_large.log

echo "=============================================================================="
echo "=== Phase 4: Large Maze Single Waypoint Translator Training ==="
echo "=============================================================================="
$PYTHON_BIN -u scripts/train_waypoint_translators.py \
    split=large \
    mode=single_wp \
    n_pairs=60000 \
    epochs=1000 \
    batch_size=256 \
    lr=3e-4 \
    2>&1 | tee outputs/train_single_wp_large.log

echo "=============================================================================="
echo "=== Phase 5: Large Maze Enhanced Sequence Attention Transformer Training ==="
echo "=============================================================================="
$PYTHON_BIN -u scripts/train_waypoint_translators.py \
    split=large \
    mode=enhanced_seq_attn \
    n_pairs=60000 \
    epochs=1000 \
    batch_size=256 \
    lr=3e-4 \
    2>&1 | tee outputs/train_enhanced_seq_attn_large.log

echo "=============================================================================="
echo "=== Phase 6: Large Maze Benchmark (10 Seeds, 5 Tasks, 10 Episodes/Task) ==="
echo "=============================================================================="
$PYTHON_BIN -u scripts/benchmark.py \
    --split=large \
    --num_tasks=5 \
    --episodes_per_task=10 \
    --start_seed=1 \
    --end_seed=10 \
    2>&1 | tee outputs/benchmark_large_10seeds.log

echo "=============================================================================="
echo "=== All Training & Benchmarks Completed Successfully on $(date) ==="
echo "=============================================================================="
