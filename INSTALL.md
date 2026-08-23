# Installation and Setup Guide

This document provides step-by-step instructions to set up the environment, install dependencies, and execute benchmarks and training routines on **NixOS / Linux** and **macOS**.

---

## 1. NixOS / Linux Setup

On NixOS or Linux systems with NVIDIA GPUs:

### A. Prerequisites (NixOS environment)
If running under NixOS, enter a shell with system development libraries, or use standard conda/micromamba:

```bash
# If using nix-shell:
nix-shell -p python310 git gcc zlib libGL glstandard mesa

# Or create a fresh Conda environment:
conda create -n fb-rl python=3.10 -y
conda activate fb-rl
```

### B. Install JAX, PyTorch, and Core Libraries
```bash
# 1. JAX with CUDA 12 support
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# 2. PyTorch with CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 3. Core dependencies
pip install flax optax distrax ml_collections tensorflow-probability h5py hydra-core omegaconf gymnasium mujoco tqdm matplotlib pandas plotly scipy scikit-learn tabulate pytest

# 4. Install OGBench
pip install ogbench
```

---

## 2. macOS (Apple Silicon / CPU) Setup

```bash
# 1. Create Conda environment
conda create -n fb-rl python=3.10 -y
conda activate fb-rl

# 2. Install JAX and PyTorch
pip install --upgrade "jax[cpu]"
pip install torch torchvision

# 3. Install core dependencies
pip install flax optax distrax ml_collections tensorflow-probability h5py hydra-core omegaconf gymnasium mujoco tqdm matplotlib pandas plotly scipy scikit-learn tabulate pytest ogbench
```

---

## 3. Verify Installation

Run the complete unit test suite to verify that all modules and environments function correctly:

```bash
pytest tests/
```
Expected output: `34 passed in ~30s`.

---

## 4. Execution Workflow

### A. Run Benchmarks Sequentially
```bash
# AntMaze Medium
python scripts/benchmark_translators.py --split=medium --num_tasks=5 --episodes_per_task=10 --output_dir=results/benchmarks

# AntMaze Large
python scripts/benchmark_translators.py --split=large --num_tasks=5 --episodes_per_task=10 --output_dir=results/benchmarks
```

All benchmark summaries are cleanly saved to:
- `results/benchmarks/benchmark_summary_10seeds_medium.csv` (.md)
- `results/benchmarks/benchmark_summary_10seeds_large.csv` (.md)
- Artifacts do **not** overwrite each other.

### B. Master Execution Script in Tmux
To run the full sequential benchmark pipeline unattended in `tmux`:

```bash
tmux new-session -s benchmark -d "
conda activate fb-rl
python scripts/benchmark_translators.py --split=medium --num_tasks=5 --episodes_per_task=10 --output_dir=results/benchmarks
python scripts/benchmark_translators.py --split=large --num_tasks=5 --episodes_per_task=10 --output_dir=results/benchmarks
echo '=== ALL BENCHMARKS COMPLETED ==='
"
```
