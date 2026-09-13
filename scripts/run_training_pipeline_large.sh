#!/usr/bin/env bash
# ==============================================================================
# Master Training & Benchmark Pipeline for AntMaze-Large
# ==============================================================================
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-$(which python3 2>/dev/null || which python 2>/dev/null)}"
if [ -f "/opt/homebrew/Caskroom/miniconda/base/envs/fb-rl/bin/python" ]; then
    PYTHON_BIN="/opt/homebrew/Caskroom/miniconda/base/envs/fb-rl/bin/python"
fi

export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
mkdir -p outputs/checkpoints outputs/cache results/benchmarks

echo "=============================================================================="
echo "=== Starting Full Training & Benchmark Pipeline for Large Maze on $(date) ==="
echo "=============================================================================="

# ------------------------------------------------------------------------------
# Phase 1: Train Direct Intention Gated-Attention on Large
# ------------------------------------------------------------------------------
echo ""
echo ">>> [Phase 1/4] Training Direct Intention GatedAttn on Large..."
$PYTHON_BIN scripts/train_distillation.py \
    --split=large \
    --model_type=gated_attn \
    --epochs=100 \
    --batch_size=256 \
    --n_pairs=25000 \
    --lr=3e-4 \
    2>&1 | tee outputs/train_distilled_jax_large.log

# ------------------------------------------------------------------------------
# Phase 2: Train Single Waypoint Translator on Large
# ------------------------------------------------------------------------------
echo ""
echo ">>> [Phase 2/4] Training Single Waypoint Translator on Large..."
$PYTHON_BIN scripts/train_waypoint_translators.py \
    split=large \
    mode=single_wp \
    n_pairs=60000 \
    epochs=1000 \
    batch_size=256 \
    lr=3e-4 \
    2>&1 | tee outputs/train_single_wp_large.log

# ------------------------------------------------------------------------------
# Phase 3: Train Enhanced Sequence Attention Transformer on Large
# ------------------------------------------------------------------------------
echo ""
echo ">>> [Phase 3/4] Training Enhanced Sequence Attention Transformer on Large..."
$PYTHON_BIN scripts/train_waypoint_translators.py \
    split=large \
    mode=enhanced_seq_attn \
    n_pairs=60000 \
    epochs=1000 \
    batch_size=256 \
    lr=3e-4 \
    2>&1 | tee outputs/train_enhanced_seq_attn_large.log

# ------------------------------------------------------------------------------
# Phase 4: Run 10-Seed Comprehensive Benchmark on Large
# ------------------------------------------------------------------------------
echo ""
echo ">>> [Phase 4/4] Running Comprehensive 10-Seed Benchmark on Large..."
$PYTHON_BIN scripts/benchmark.py \
    --split=large \
    --num_tasks=5 \
    --episodes_per_task=10 \
    2>&1 | tee outputs/benchmark_large_10seeds.log

echo ""
echo "=============================================================================="
echo "=== Pipeline for Large Maze Completed Successfully on $(date) ==="
echo "=============================================================================="
