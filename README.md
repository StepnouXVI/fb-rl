# Planning over Sequences of Intentions on Forward-Backward Representations in Complex Navigation Tasks

An offline zero-shot hierarchical reinforcement learning framework for long-horizon, obstacle-dense continuous control navigation tasks in AntMaze. The approach combines Forward-Backward (FB) successor representations with topological graph search on replay buffer observations, sequence-aware attention translators with ALiBi and curvature tokens, and amortized JAX policy distillation.

---
## PDF report
[report.pdf](report.pdf)

## Overview of Methods

The framework investigates multi-subgoal planning over learned FB bilinear embeddings $F(s, z)^\top B(s')$ without environment map access or ground-truth simulator collision geometry:

1. **Single-Intention Baseline ($\pi$-switch)**: Hierarchical baseline with a low-level continuous controller $\pi^\ell(a \mid s, z)$ and high-level goal-conditioned intention selector $\pi^h(s, B(g)) \to z$.
2. **Recursive Bisection**: Divide-and-conquer intermediate waypoint selection by maximizing reachability sum $\arg\max_w [\log F(s, B(w))^\top B(w) + \log F(w, B(g))^\top B(g)]$.
3. **Buffer Graph Dijkstra**: Directed topological reachability graph built on $N$ offline replay buffer states with distance/reachability threshold filtering and continuous sliding lookahead ($L = 2.6\,\text{m}$).
4. **Distilled JAX Gated-Attention**: $O(1)$ amortized planner trained on teacher trajectories with a 4-component differentiable loss ($\mathcal{L}_{\text{BC}}$, $\mathcal{L}_{\text{Action}}$, $\mathcal{L}_{\text{Reach}}$, $\mathcal{L}_{\text{Goal}}$).
5. **Single Waypoint Translator**: Neural translator $T_\theta(s_t, B(w_{\text{lookahead}})) \to z_{\text{cmd}}$ with adaptive gating and residual connections.
6. **Enhanced Sequence Waypoint Attention Transformer**: Trajectory attention transformer over waypoint sequence $\mathcal{S} = [B(w_1), \dots, B(w_K), B(g)]$ with ALiBi position decay, turning angle tokens ($\cos \theta_k$), localized attention window ($4+1$), and auxiliary local anchor loss $\mathcal{L}_{\text{Aux}}$.

---

## Benchmark Results

Evaluation across 20 independent random seeds (5 test tasks, 10 episodes per task, 1,000 total episodes per method) on AntMaze Medium and Large environments. In Medium, episodes were limited to 1,000 steps; in Large, to 1,500 steps.

### Main Summary Table

| # | Method | AntMaze Medium SR (%) | Latency (ms) | AntMaze Large SR (%) | Latency (ms) |
|---|:---|:---:|:---:|:---:|:---:|
| 1 | Single-Intention Baseline | 81.4 ± 4.5% | 0.95 | 49.3 ± 5.2% | 0.95 |
| 2 | Recursive Bisection | 75.9 ± 6.4% | 2.82 | 46.5 ± 8.0% | 2.36 |
| 3 | Dijkstra Teacher (`high_actor`) | 83.9 ± 3.2% | 0.85 | 48.7 ± 9.2% | 0.87 |
| 4 | Distilled JAX GatedAttn [$O(1)$] | **84.3 ± 4.8%** | 0.55 | 45.6 ± 5.8% | 0.57 |
| 5 | Dijkstra + Single WP Translator | 83.3 ± 5.0% | **0.53** | 57.1 ± 5.6% | **0.54** |
| 6 | **Dijkstra + Enhanced Sequence Attn** | 83.6 ± 3.8% | 0.95 | **60.9 ± 6.9%** | 0.97 |

### AntMaze Large Breakdown (20 Seeds, 1000 Episodes)

| # | Method | Success Rate (%) | Latency (ms) | Task 01 (%) | Task 02 (%) | Task 03 (%) | Task 04 (%) | Task 05 (%) |
|---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | Single-Intention Baseline | 49.3 ± 5.2% | 0.95 | 44.5% | 54.5% | 79.5% | 36.5% | 31.5% |
| 2 | Recursive Bisection Planner | 46.5 ± 8.0% | 2.36 | 29.5% | 59.0% | 85.5% | 25.5% | 33.0% |
| 3 | Dijkstra Teacher (`high_actor`) | 48.7 ± 9.2% | 0.87 | 56.0% | 65.5% | 55.5% | 32.0% | 34.5% |
| 4 | Distilled JAX GatedAttn [$O(1)$] | 45.6 ± 5.8% | 0.57 | 50.0% | 68.0% | 55.0% | 28.0% | 27.0% |
| 5 | Dijkstra + Single WP Translator | 57.1 ± 5.6% | **0.54** | 53.0% | 74.5% | 58.0% | 44.0% | 56.0% |
| 6 | **Dijkstra + Enhanced Sequence Attn** | **60.9 ± 6.9%** | 0.97 | 53.0% | 66.5% | 72.0% | 54.0% | 59.0% |

### AntMaze Medium Breakdown (20 Seeds, 1000 Episodes)

| # | Method | Success Rate (%) | Latency (ms) | Task 01 (%) | Task 02 (%) | Task 03 (%) | Task 04 (%) | Task 05 (%) |
|---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | Single-Intention Baseline | 81.4 ± 4.5% | 0.95 | 80.5% | 86.0% | 81.0% | 69.0% | 90.5% |
| 2 | Recursive Bisection Planner | 75.9 ± 6.4% | 2.82 | 69.5% | 86.0% | 70.0% | 65.0% | 89.0% |
| 3 | Dijkstra Teacher (`high_actor`) | 83.9 ± 3.2% | 0.85 | 72.0% | 88.5% | 89.5% | 80.5% | 89.0% |
| 4 | Distilled JAX GatedAttn [$O(1)$] | **84.3 ± 4.8%** | 0.55 | 83.0% | 95.0% | 81.0% | 73.0% | 89.5% |
| 5 | Dijkstra + Single WP Translator | 83.3 ± 5.0% | **0.53** | 82.5% | 86.5% | 87.0% | 69.0% | 91.5% |
| 6 | **Dijkstra + Enhanced Sequence Attn** | 83.6 ± 3.8% | 0.95 | 87.5% | 85.0% | 87.5% | 70.0% | 88.0% |

---

## Installation

### Prerequisites
- Python 3.10
- Conda / Micromamba
- MuJoCo and continuous control physics backend

### Setup Instructions

#### Option 1: Quick Setup with `conda` and `requirements.txt` (Recommended)

```bash
# 1. Clone repository
git clone git@github.com:StepnouXVI/fb-rl.git
cd fb-rl

# 2. Create and activate Conda environment
conda create -n fb-rl python=3.10 -y
conda activate fb-rl

# 3. Install PyTorch & JAX (Select according to your hardware)

# For Linux / Windows with CUDA 12:
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# For macOS (Apple Silicon MPS / CPU) or generic CPU:
pip install --upgrade "jax[cpu]"
pip install torch torchvision

# 4. Install all dependencies from requirements.txt
pip install -r requirements.txt

# 5. Verify the environment setup
python verify_env.py
```

#### Option 2: Setup from `environment.yml`

```bash
# Create environment from file and install dependencies
conda env create -f environment.yml
conda activate fb-rl

# Verify the environment setup
python verify_env.py
```

---

## Usage

### 1. Run Benchmarks

```bash
# AntMaze Medium benchmark (5 tasks, 10 episodes/task across all candidate planners)
python scripts/benchmark_translators.py --split=medium --num_tasks=5 --episodes_per_task=10 --output_dir=results/benchmarks

# AntMaze Large benchmark (5 tasks, 10 episodes/task across all candidate planners)
python scripts/benchmark_translators.py --split=large --num_tasks=5 --episodes_per_task=10 --output_dir=results/benchmarks
```

Summary tables are automatically exported in Markdown and CSV to `results/benchmarks/`.

### 2. Train Models

#### Differentiable JAX Distillation (Amortized $O(1)$ Gated-Attention Policy)
```bash
# Medium
python scripts/train_jax_distillation.py --split=medium --model_type=gated_attn --epochs=35 --n_pairs=25000

# Large
python scripts/train_jax_distillation.py --split=large --model_type=gated_attn --epochs=40 --n_pairs=60000
```

#### Single Waypoint Translator
```bash
python scripts/train_waypoint_translators.py split=medium mode=single_wp epochs=500 n_pairs=25000
python scripts/train_waypoint_translators.py split=large mode=single_wp epochs=800 n_pairs=60000
```

#### Enhanced Sequence Attention Transformer
```bash
python scripts/train_waypoint_translators.py split=medium mode=enhanced_seq_attn epochs=600 n_pairs=25000
python scripts/train_waypoint_translators.py split=large mode=enhanced_seq_attn epochs=1000 n_pairs=60000
```

### 3. Generate Trajectory Visualizations

```bash
# Collect rollouts and generate per-method trajectory maze maps
python scripts/generate_all_method_rollouts_and_plots.py --split=medium --num_tasks=5 --episodes_per_task=3
python scripts/generate_all_method_rollouts_and_plots.py --split=large --num_tasks=5 --episodes_per_task=3
```

### 4. Run Physics & Intention Experiments

```bash
# Multi-mode intention experiments (direct latent, Dijkstra planner, custom waypoints)
python scripts/experiment_intention.py env=antmaze_medium scenario_name=all

# Low-level continuous torque controller comparative analysis
python scripts/experiment_lowlevel_control.py
```

### 5. Generate Summary Plots & Pareto Curves

```bash
python scripts/generate_plots.py
```

### 6. Run Test Suite

```bash
pytest tests/
```

---

## Directory Structure

```
fb-rl/
├── configs/                       # Hydra configuration files
│   ├── config.yaml                # Benchmark settings
│   ├── experiment.yaml            # Intention scenario configurations
│   └── train_translator.yaml      # Translator training parameters
├── report/                        # Research report and TikZ sources
│   ├── figures/                   # TikZ schemes and loss curves
│   ├── inc/                       # Report section TeX sources
│   ├── main.tex                   # Technical report LaTeX entrypoint
│   ├── preamble.tex               # LaTeX preamble and packages
│   └── main.pdf                   # Compiled technical report PDF
├── results/                       # Evaluation outputs and model weights
│   ├── benchmarks/                # Benchmark JSON and CSV results
│   ├── checkpoints/               # Trained Flax and PyTorch model weights
│   ├── data/                      # Rollout telemetry data
│   ├── datasets/                  # Cached teacher demonstration datasets
│   └── plots/                     # Trajectory visualization plots
├── scripts/                       # Training, benchmarking, and visualization scripts
│   ├── benchmark_translators.py   # Multi-task benchmark runner
│   ├── experiment_intention.py    # Intention analysis experiment
│   ├── experiment_lowlevel_control.py # Low-level controller experiment
│   ├── generate_all_method_rollouts_and_plots.py # Batch trajectory visualizer
│   ├── generate_plots.py          # Summary chart generation
│   ├── generate_report_loss_curves.py # Training curve generator
│   ├── train_jax_distillation.py  # JAX policy distillation trainer
│   ├── train_waypoint_translators.py # Waypoint translator trainer
│   └── visualize_trajectories.py  # Trajectory plotting utility
├── src/                           # Core library modules
│   ├── agent_loader.py            # Pretrained FB checkpoint loader
│   ├── evaluator.py               # Evaluation engine
│   ├── metrics.py                 # Telemetry and success metrics
│   ├── models.py                  # PyTorch student architectures
│   ├── planners.py                # Graph construction and Dijkstra planner
│   └── waypoint_translators.py    # Flax models, ALiBi attention, and loss functions
└── tests/                         # Pytest test suite
```

---

## License

MIT License.

---