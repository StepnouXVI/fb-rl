# Zero-Shot Multi-Subgoal Planning with Forward-Backward Representations in Complex Maze Environments

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![JAX](https://img.shields.io/badge/JAX-0.4+-orange.svg)](https://github.com/google/jax)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Flax Linen](https://img.shields.io/badge/Flax-Linen-purple.svg)](https://github.com/google/flax)
[![OGBench](https://img.shields.io/badge/OGBench-AntMaze-green.svg)](https://github.com/seohongpark/ogbench)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An offline, zero-shot hierarchical reinforcement learning framework that solves long-horizon, obstacle-dense continuous control navigation tasks by combining **Forward-Backward (FB) successor measure representations** with **Topological Graph Dijkstra Planning**, **ALiBi Sequence Attention Transformers**, and **Differentiable Physics-Grounded Policy Distillation**.

---

## 🚀 Key Highlights

- **Pure FB Geometry & Zero Map Cheating**: Operates with **zero access** to the environment's `maze_map`, wall coordinates, or simulator ground-truth collision geometries. All topological graphs and transitions are constructed strictly from offline buffer observations and the learned bilinear successor measure $F(s, z)^\top B(s')$.
- **State-of-the-Art Benchmark Results (10 Random Seeds, 5 Tasks, 10-15 Episodes/Task)**:
  - **AntMaze-Medium**:
    - **93.3% Success Rate** (Dijkstra + Enhanced Sequence Attention)
    - **86.2% Success Rate** ($O(1)$ Distilled JAX GatedAttn, **$0.10\,\text{ms}$/step**)
    - Baseline: $42.5\%$.
  - **AntMaze-Large**:
    - **74.0% Success Rate** (Dijkstra + Enhanced Sequence Attention Transformer)
    - **68.0% Success Rate** (Dijkstra + Single WP Translator)
    - **56.0% Success Rate** ($O(1)$ Distilled JAX GatedAttn)
    - Baseline: $24.0\%$.
- **ALiBi Sequence Attention Transformer**: Incorporates distance-decay attention bias ($-\lambda \cdot m_h \cdot k$), local-global receptive windowing, and 2D trajectory curvature token encoding ($\cos \theta_k$) to eliminate wall phase-through artifacts.
- **Ultra-Fast JIT Inference**: Full fusion in JAX/Flax delivers sub-millisecond execution ($< 0.10\,\text{ms}$/step for distilled policies, $< 0.40\,\text{ms}$ for sequence transformers).
- **Clean, Modular Codebase**: Strict adherence to software engineering standards ($\le 50$ lines per function, 100% pytest coverage).

---

## 📊 Benchmark Summary

### 1. AntMaze-Large Benchmark (10 Random Seeds, 5 Test Tasks, 10 Episodes/Task)

| # | Method | Success Rate (%) | Latency (ms/step) | Task 01 (%) | Task 02 (%) | Task 03 (%) | Task 04 (%) | Task 05 (%) |
|---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | **Single-Intention Baseline** | $24.0 \pm 8.0\%$ | $\mathbf{0.07\,\text{ms}}$ | $30.0\%$ | $10.0\%$ | $40.0\%$ | $20.0\%$ | $20.0\%$ |
| 2 | **Dijkstra Teacher (high_actor)** | $60.0 \pm 8.9\%$ | $1.25\,\text{ms}$ | $80.0\%$ | $50.0\%$ | $60.0\%$ | $60.0\%$ | $50.0\%$ |
| 3 | **Dijkstra + Single WP Translator** | $68.0 \pm 9.8\%$ | $0.22\,\text{ms}$ | $90.0\%$ | $60.0\%$ | $70.0\%$ | $70.0\%$ | $50.0\%$ |
| 5 | **Dijkstra + Enhanced Sequence Transformer** | $\mathbf{74.0 \pm 8.0\%}$ | $0.38\,\text{ms}$ | $\mathbf{100.0\%}$ | $\mathbf{70.0\%}$ | $\mathbf{80.0\%}$ | $\mathbf{70.0\%}$ | $\mathbf{50.0\%}$ |
| 6 | **Distilled JAX GatedAttn [$O(1)$]** | $56.0 \pm 8.0\%$ | $\mathbf{0.09\,\text{ms}}$ | $70.0\%$ | $50.0\%$ | $60.0\%$ | $60.0\%$ | $40.0\%$ |

---

### 2. AntMaze-Medium Benchmark (10 Random Seeds, 5 Test Tasks, 15 Episodes/Task)

| # | Method | Success Rate (%) | Latency (ms/step) | Task 01 (%) | Task 02 (%) | Task 03 (%) | Task 04 (%) | Task 05 (%) |
|---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | **Single-Intention Baseline** | $42.5 \pm 3.1\%$ | $\mathbf{0.08\,\text{ms}}$ | $26.7\%$ | $20.0\%$ | $73.3\%$ | $60.0\%$ | $33.3\%$ |
| 2 | **Buffer Graph Dijkstra** | $93.3 \pm 1.4\%$ | $1.02\,\text{ms}$ | $100.0\%$ | $100.0\%$ | $80.0\%$ | $86.7\%$ | $100.0\%$ |
| 3 | **Dijkstra + Enhanced Sequence Attention** | $\mathbf{94.7 \pm 1.2\%}$ | $0.35\,\text{ms}$ | $\mathbf{100.0\%}$ | $\mathbf{100.0\%}$ | $\mathbf{86.7\%}$ | $\mathbf{86.7\%}$ | $\mathbf{100.0\%}$ |
| 4 | **Distilled JAX GatedAttn [$O(1)$]** | $86.2 \pm 1.8\%$ | $\mathbf{0.10\,\text{ms}}$ | $100.0\%$ | $93.3\%$ | $80.0\%$ | $73.3\%$ | $86.7\%$ |

---

## 🛠️ Quick Installation Guide

### Prerequisites
- Python 3.10
- Conda / Micromamba
- Linux (Ubuntu / NixOS / Debian) or macOS (Apple Silicon / Intel)
- NVIDIA GPU with CUDA 12+ (optional for training acceleration; CPU inference fully supported)

### Step-by-Step Setup

```bash
# 1. Clone repository
git clone git@github.com:StepnouXVI/fb-rl.git
cd fb-rl

# 2. Create and activate Conda environment
conda create -n fb-rl python=3.10 -y
conda activate fb-rl

# 3. Install PyTorch & JAX
# For CUDA 12 (Linux / NixOS):
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# For macOS / CPU:
# pip install --upgrade "jax[cpu]"
# pip install torch torchvision

# 4. Install Dependencies & OGBench
pip install flax optax hydra-core omegaconf gymnasium mujoco tqdm matplotlib pandas plotly scipy scikit-learn tabulate pytest

# Install OGBench from submodule or pip
pip install ogbench
```

---

## 💻 How to Run

### 1. Run Complete 10-Seed Benchmark

Evaluate all methods (`Single-Intention Baseline`, `Dijkstra Teacher`, `Single WP Translator`, `Enhanced Sequence Attention`, `Distilled JAX GatedAttn`) on 10 seeds:

```bash
# Benchmark on AntMaze Medium
python scripts/benchmark_translators.py --split=medium --num_tasks=5 --episodes_per_task=10 --output_dir=results/benchmarks

# Benchmark on AntMaze Large
python scripts/benchmark_translators.py --split=large --num_tasks=5 --episodes_per_task=10 --output_dir=results/benchmarks
```

Output tables and summaries will be saved in `results/benchmarks/benchmark_summary_10seeds_{split}.csv` and `*.md`.

---

### 2. Train Models from Scratch

#### A. Train Differentiable JAX Latent Distillation ($O(1)$ Amortized Planner)
```bash
# Medium maze
python scripts/train_jax_distillation.py --split=medium --model_type=gated_attn --epochs=35 --n_pairs=25000

# Large maze
python scripts/train_jax_distillation.py --split=large --model_type=gated_attn --epochs=40 --n_pairs=60000
```

#### B. Train Single Waypoint Translator
```bash
python scripts/train_waypoint_translators.py split=medium mode=single_wp epochs=500 n_pairs=25000
python scripts/train_waypoint_translators.py split=large mode=single_wp epochs=800 n_pairs=60000
```

#### C. Train Enhanced Sequence-Aware Attention Transformer (Best SOTA)
```bash
python scripts/train_waypoint_translators.py split=medium mode=enhanced_seq_attn epochs=600 n_pairs=25000
python scripts/train_waypoint_translators.py split=large mode=enhanced_seq_attn epochs=1000 n_pairs=60000
```

---

### 3. Generate Trajectory Visualizations

```bash
# Rollout and export all trajectory plots organized into results/{method_name}/{split}/{success,failed}/
python scripts/generate_all_method_rollouts_and_plots.py --split=medium --num_tasks=5 --episodes_per_task=3
python scripts/generate_all_method_rollouts_and_plots.py --split=large --num_tasks=5 --episodes_per_task=3
```

---

### 4. Run Intention & Low-Level Control Physics Experiments

```bash
# Run multi-mode intention scenarios
python scripts/experiment_intention.py env=antmaze_medium scenario_name=all

# Run low-level physics control & straight-line waypoint comparison
python scripts/experiment_lowlevel_control.py
```

---

### 5. Run Unit Tests

Verify 100% passing test suite:
```bash
pytest tests/
```

---

## 📁 Repository Directory Structure

```
fb-rl/
├── configs/                       # Hydra configuration files
│   ├── config.yaml                # General benchmark config
│   ├── experiment.yaml            # Intention scenario experiments
│   └── train_translator.yaml      # Waypoint translator training config
├── report/                        # LaTeX report & TikZ figures
│   ├── figures/                   # Clean Russian/math TikZ vector schemes
│   │   ├── fig1_framework_overview.tex
│   │   ├── fig2_buffer_graph_dijkstra.tex
│   │   ├── fig3_distilled_jax.tex
│   │   ├── fig4_single_wp_translator.tex
│   │   └── fig5_enhanced_sequence_transformer.tex
│   ├── main.tex                   # Comprehensive Russian research paper
│   └── report.pdf                 # Compiled paper PDF
├── results/                       # Standardized output artifacts
│   ├── benchmarks/                # JSON, CSV, MD summary tables
│   ├── checkpoints/               # Trained Flax & PyTorch weights (.pkl, .pt)
│   ├── data/                      # Trajectory step telemetry (.csv)
│   ├── datasets/                  # Cached golden demonstration datasets (.npz)
│   ├── experiments/               # Intention & low-level physics plots
│   └── plots/                     # Publication comparison figures
├── scripts/                       # Modular CLI & training execution scripts (<50 lines/fn)
│   ├── benchmark_translators.py   # 10-seed multi-task benchmark runner
│   ├── experiment_intention.py    # Multi-scenario intention testing
│   ├── experiment_lowlevel_control.py # 8-joint torque physics controller analysis
│   ├── generate_all_method_rollouts_and_plots.py # Batch trajectory renderer
│   ├── generate_plots.py          # Summary Pareto and bar charts
│   ├── run_benchmark.py           # Parallel multi-process benchmark
│   ├── train_jax_distillation.py  # JAX/Flax differentiable distillation trainer
│   ├── train_waypoint_translators.py # Hydra-based translator trainer
│   └── visualize_trajectories.py  # 2D AntMaze trajectory & portal visualizer
├── src/                           # Core library modules (<50 lines/fn)
│   ├── agent_loader.py            # Pretrained FB checkpoint loader
│   ├── evaluator.py               # Zero-shot evaluation engine
│   ├── models.py                  # PyTorch student architectures
│   └── waypoint_translators.py    # Flax models, ALiBi Attention, Loss steps
└── tests/                         # Pytest test suite (34/34 passing)
```

---

## 📜 Citation & License

This project is licensed under the MIT License. If you use this codebase or methodology in your research, please cite our technical report:

```bibtex
@article{savvateev2026zeroshot,
  title={Zero-Shot Multi-Subgoal Planning with Forward-Backward Representations in Complex Maze Environments},
  author={Savvateev, Iaroslav},
  journal={arXiv preprint},
  year={2026}
}
```
