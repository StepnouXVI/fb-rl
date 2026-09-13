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
4. **Direct Intention Gated-Attention**: $O(1)$ amortized planner trained on teacher trajectories with a 4-component differentiable loss ($\mathcal{L}_{\text{BC}}$, $\mathcal{L}_{\text{Action}}$, $\mathcal{L}_{\text{Reach}}$, $\mathcal{L}_{\text{Goal}}$).
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
| 4 | Direct Intention GatedAttn [$O(1)$] | **84.3 ± 4.8%** | 0.55 | 45.6 ± 5.8% | 0.57 |
| 5 | Dijkstra + Single WP Translator | 83.3 ± 5.0% | **0.53** | 57.1 ± 5.6% | **0.54** |
| 6 | **Dijkstra + Enhanced Sequence Attn** | 83.6 ± 3.8% | 0.95 | **60.9 ± 6.9%** | 0.97 |

### AntMaze Large Breakdown (20 Seeds, 1000 Episodes)

| # | Method | Success Rate (%) | Latency (ms) | Task 01 (%) | Task 02 (%) | Task 03 (%) | Task 04 (%) | Task 05 (%) |
|---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | Single-Intention Baseline | 49.3 ± 5.2% | 0.95 | 44.5% | 54.5% | 79.5% | 36.5% | 31.5% |
| 2 | Recursive Bisection Planner | 46.5 ± 8.0% | 2.36 | 29.5% | 59.0% | 85.5% | 25.5% | 33.0% |
| 3 | Dijkstra Teacher (`high_actor`) | 48.7 ± 9.2% | 0.87 | 56.0% | 65.5% | 55.5% | 32.0% | 34.5% |
| 4 | Direct Intention GatedAttn [$O(1)$] | 45.6 ± 5.8% | 0.57 | 50.0% | 68.0% | 55.0% | 28.0% | 27.0% |
| 5 | Dijkstra + Single WP Translator | 57.1 ± 5.6% | **0.54** | 53.0% | 74.5% | 58.0% | 44.0% | 56.0% |
| 6 | **Dijkstra + Enhanced Sequence Attn** | **60.9 ± 6.9%** | 0.97 | 53.0% | 66.5% | 72.0% | 54.0% | 59.0% |

### AntMaze Medium Breakdown (20 Seeds, 1000 Episodes)

| # | Method | Success Rate (%) | Latency (ms) | Task 01 (%) | Task 02 (%) | Task 03 (%) | Task 04 (%) | Task 05 (%) |
|---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | Single-Intention Baseline | 81.4 ± 4.5% | 0.95 | 80.5% | 86.0% | 81.0% | 69.0% | 90.5% |
| 2 | Recursive Bisection Planner | 75.9 ± 6.4% | 2.82 | 69.5% | 86.0% | 70.0% | 65.0% | 89.0% |
| 3 | Dijkstra Teacher (`high_actor`) | 83.9 ± 3.2% | 0.85 | 72.0% | 88.5% | 89.5% | 80.5% | 89.0% |
| 4 | Direct Intention GatedAttn [$O(1)$] | **84.3 ± 4.8%** | 0.55 | 83.0% | 95.0% | 81.0% | 73.0% | 89.5% |
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
pip install --upgrade "jax[cuda12]"
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# For macOS (Apple Silicon MPS / CPU) or generic CPU:
pip install --upgrade "jax[cpu]"
pip install torch torchvision

# 4. Install all dependencies from requirements.txt
pip install -r requirements.txt

# 5. Verify the environment setup
pytest tests/
```

#### Option 2: Setup from `environment.yml`

```bash
# Create environment from file and install dependencies
conda env create -f environment.yml
conda activate fb-rl

# Verify the environment setup
pytest tests/
```

---

## Usage

### 1. Run Benchmarks

```bash
# AntMaze Medium benchmark (5 tasks, 10 episodes/task across all candidate planners)
python scripts/benchmark.py --split=medium --num_tasks=5 --episodes_per_task=10 --output_dir=results/benchmarks

# AntMaze Large benchmark (5 tasks, 10 episodes/task across all candidate planners)
python scripts/benchmark.py --split=large --num_tasks=5 --episodes_per_task=10 --output_dir=results/benchmarks
```

Summary tables are automatically exported in Markdown and CSV to `results/benchmarks/`.

### 2. Train Models

#### Direct Intention Distillation (Amortized $O(1)$ Gated-Attention Policy)
```bash
# Medium
python scripts/train_distillation.py --split=medium --model_type=gated_attn --epochs=35 --n_pairs=25000

# Large
python scripts/train_distillation.py --split=large --model_type=gated_attn --epochs=40 --n_pairs=60000
```

#### Single Waypoint Translator
```bash
python scripts/train_waypoint_translators.py split=medium mode=single_wp epochs=500 n_pairs=25000
python scripts/train_waypoint_translators.py split=large mode=single_wp epochs=800 n_pairs=60000
```

#### Sequence Attention Transformer
```bash
python scripts/train_waypoint_translators.py split=medium mode=enhanced_seq_attn epochs=600 n_pairs=25000
python scripts/train_waypoint_translators.py split=large mode=enhanced_seq_attn epochs=1000 n_pairs=60000
```

### 3. Master Training & Benchmark Pipeline

To run the complete end-to-end pipeline (training Sequence Attention, Single Waypoint, and Distillation models followed by multi-seed benchmarks across both Medium and Large mazes) in an isolated session:

```bash
# Run the master pipeline (runs in background or tmux)
bash scripts/run_master_pipeline.sh
```

---

## Interactive Telemetry & Experiment Tracking (Aim + SQLite)

The framework employs a **Dual Telemetry Architecture** designed for zero runtime latency, complete episode replayability, and rich interactive visual analysis:

1. **Relational Episode Database (SQLite `results/data/telemetry.db`)**:
   - Stores episode headers, start/goal coordinates, total steps, cross-track errors (CTE), spatial self-intersections, and full $(x, y, z, v_x, v_y, \text{action torques}, \text{attention weights})$ trajectory logs with WAL (Write-Ahead Logging) mode.
2. **Interactive Telemetry Dashboard (Aim `results/aim`)**:
   - Tracks real-time training losses ($\mathcal{L}_{\text{total}}$, $\mathcal{L}_{\cos}$, $\mathcal{L}_{\text{action}}$, $\mathcal{L}_{\text{reach}}$, $\mathcal{L}_{\text{goal}}$), cosine similarities, validation metrics, and learning rate schedules.
   - Stores interactive **Plotly** figures:
     - **Pareto Frontiers**: Inference Latency (ms) vs. Success Rate (%) trade-offs with standard error bars.
     - **Task-by-Task Breakdowns**: Grouped bar charts comparing all architectures across tasks 1 to 5.
     - **Trajectory & Attention Maps**: Ant trajectories overlayed with Dijkstra paths and transformer attention weights.

### How to Launch Local Aim UI

To start the local Aim web dashboard and inspect all training runs, benchmarks, and interactive Plotly figures:

```bash
# Start Aim UI server locally
aim up --repo results/aim --port 43800
```

Once started, open your browser and navigate to:
```
http://localhost:43800
```

### Remote Server Sync (Optional)

To synchronize local telemetry data (`results/aim/`) to a centralized Aim server or VPS:

```bash
# Export and sync local Aim runs to remote server container
bash scripts/sync_aim.sh
```

---

## Detailed Technical & Mathematical Guide

For an exhaustive, pedagogical explanation of every mathematical formula, symbol breakdown table, physical interpretation, and architecture design decision, see:
- [Complete Guide to FB-RL (Markdown)](docs/fb_rl_comprehensive_guide.md)
  - **Explain Like I'm 5**: Metaphors and intuitive analogies for multi-joint locomotion and hierarchical planning.
  - **Symbol-by-Symbol Breakdowns**: 46 analytical tables explaining every variable, index, and operator.
  - **Mathematical Proofs & Derivations**: Successor measure factorization, $\sqrt{d}$ sphere geometry, TD-LSIF loss, and analytical gradients through the frozen actor $\nabla_\theta \mathcal{L}_{\text{action}}$.
  - **Oral Defense & Exam Guide**: 10 comprehensive answers to challenging questions about FB representation theory and neural intention translation.

---

## Directory Structure

```
fb-rl/
├── configs/                       # Hydra and OmegaConf configurations
│   ├── env/                       # AntMaze environment YAMLs (medium, large)
│   └── train_translator.yaml      # Translator training hyperparameters
├── docs/                          # Comprehensive technical documentation
│   └── fb_rl_comprehensive_guide.md # Complete FB-RL theory, math, and defense guide
├── report/                        # Research report and TikZ sources
│   ├── figures/                   # TikZ schemes and loss curves
│   ├── inc/                       # Report section TeX sources
│   ├── main.tex                   # Technical report LaTeX entrypoint
│   ├── preamble.tex               # LaTeX preamble and packages
│   └── main.pdf                   # Compiled technical report PDF
├── results/                       # Evaluation outputs and model weights
│   ├── aim/                       # Local Aim experiment tracking repository (.aim)
│   ├── benchmarks/                # Multi-seed benchmark summary tables (CSV/MD)
│   ├── checkpoints/               # Trained neural network weights (.pkl)
│   ├── data/                      # SQLite relational telemetry database (telemetry.db)
│   └── datasets/                  # Offline demonstration datasets (npz)
├── scripts/                       # Training, benchmarking, and sync scripts
│   ├── benchmark.py               # Multi-seed StagedAgent benchmark runner
│   ├── run_master_pipeline.sh     # Master end-to-end training & evaluation pipeline
│   ├── sync_aim.sh                # RSync and central Aim server ingestion script
│   ├── train_distillation.py      # Direct intention policy distillation trainer
│   └── train_waypoint_translators.py # Sequence Attention & Single WP translator trainer
├── src/                           # Core library modules
│   ├── agent.py                   # Modular StagedAgent pipeline architecture
│   ├── agent_loader.py            # Pretrained FB checkpoint loader
│   ├── contexts.py                # Pipeline context data structures
│   ├── evaluator.py               # Deterministic zero-shot evaluation engine
│   ├── networks.py                # Flax Linen neural network modules
│   ├── stages.py                  # Modular pipeline execution stages
│   ├── telemetry/                 # Telemetry, metrics, and tracking modules
│   │   ├── aim_fast.py            # Batched remote client transport patch
│   │   ├── aim_tracker.py         # Aim Run lifecycle and metric tracker wrapper
│   │   ├── db.py                  # SQLite database engine (TelemetryDatabase)
│   │   ├── metrics.py             # CTE and geometric self-intersection metrics
│   │   ├── plotting.py            # Dark-theme Plotly figure builders
│   │   └── profiler.py            # High-resolution stage latency profiler
│   ├── topology.py                # Graph construction and Dijkstra search
│   └── training.py                # JIT-compiled differentiable loss steps
└── tests/                         # Pytest test suite (59 unit tests)
```

---

## License

MIT License.

---