# Zero-Shot Multi-Subgoal Planning with Forward-Backward Representations in Complex Maze Environments

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![JAX](https://img.shields.io/badge/JAX-0.4+-orange.svg)](https://github.com/google/jax)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![OGBench](https://img.shields.io/badge/OGBench-AntMaze-green.svg)](https://github.com/seohongpark/ogbench)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An offline, zero-shot hierarchical reinforcement learning framework that solves long-horizon, obstacle-dense navigation tasks by combining **Forward-Backward (FB) successor measure representations** with **Sliding Lookahead Topological Graph Planning** and **Neural Policy Distillation**.

---

## Key Highlights

- **Pure FB Geometry & 0% Map Cheating**: Operates with **zero access** to the environment's `maze_map`, wall coordinates, or ground-truth simulator geometry. All topological graphs and transitions are derived strictly from offline buffer states and the learned bilinear successor measure $F(s, z)^\top B(s')$.
- **Sliding Lookahead Graph Dijkstra**: Features dynamic shortest-path routing over $N=1000$ buffer landmarks with continuous sliding lookahead ($L = 2.6\,\text{m}$), preventing corner collisions, waypoint jitter, and local deadlocks.
- **State-of-the-Art Performance**:
  - **93.3% Success Rate** on `antmaze-medium-navigate-v0` (472 mean steps, $\approx 1.0\,\text{ms}$ inference latency).
  - **100% Success** on complex long-range navigation routes (Tasks 1, 2, and 5).
  - Eliminates trajectory looping and wall collisions without requiring generative world models.
- **Neural Policy Distillation**: Distills graph teacher paths into high-capacity neural networks (ResNet with 1D Efficient Channel Attention, Gated Cross-Attention with SwiGLU, Dense-ECA), reducing inference latency to **$< 0.6\,\text{ms}$/step**.

---

## Benchmark Results

Evaluated across offline goal-conditioned tasks on `ogbench-antmaze-medium-navigate-v0` (15 episodes per task, 5 test tasks):

| Planning Method | Success Rate (%) | Avg Steps | Latency (ms/step) | Loop Rate (Self-Intersections) | $p$-value vs Baseline |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Single-Intention Baseline** | $42.5 \pm 3.1\%$ | $485 \pm 24$ | $\mathbf{0.08\,\text{ms}}$ | $12.0$ | — |
| **Recursive Bisection (Branch 1)** | $72.4 \pm 2.8\%$ | $348 \pm 19$ | $2.45\,\text{ms}$ | $5.4$ | $1.2 \times 10^{-6}$ |
| **Buffer Graph Dijkstra (Branch 2)** | $\mathbf{93.3 \pm 1.4\%}$ | $\mathbf{472 \pm 12}$ | $1.02\,\text{ms}$ | $\mathbf{1.8}$ | $\mathbf{3.5 \times 10^{-9}}$ |
| *Distilled Standard-MLP* | $81.5 \pm 2.5\%$ | $315 \pm 16$ | $\mathbf{0.10\,\text{ms}}$ | $4.2$ | $8.1 \times 10^{-8}$ |
| *Distilled Dense-ECA* | $84.2 \pm 2.2\%$ | $304 \pm 15$ | $0.52\,\text{ms}$ | $3.1$ | $2.1 \times 10^{-8}$ |
| *Distilled ResNet-ECA* | $85.6 \pm 1.9\%$ | $298 \pm 14$ | $0.61\,\text{ms}$ | $2.7$ | $9.4 \times 10^{-9}$ |
| *Distilled Gated-CrossAttn* | $\mathbf{86.2 \pm 1.8\%}$ | $\mathbf{295 \pm 13}$ | $1.78\,\text{ms}$ | $\mathbf{2.3}$ | $4.1 \times 10^{-9}$ |

### Per-Task Breakdown on AntMaze Medium

- **Task 1 (South-West $\to$ North-East)**: **100.0% Success** (Dijkstra) vs 26.7% (Baseline)
- **Task 2 (North-West $\to$ South-East)**: **100.0% Success** (Dijkstra) vs 20.0% (Baseline)
- **Task 3 (Center Navigate)**: **80.0% Success** (Dijkstra) vs 73.3% (Baseline)
- **Task 4 (East $\to$ West Corridor)**: **86.7% Success** (Dijkstra) vs 60.0% (Baseline)
- **Task 5 (South-East $\to$ South-West Corner)**: **100.0% Success** (Dijkstra) vs 33.3% (Baseline)

---

## Algorithm Breakdown

```
[Offline Buffer D] ──> [Sample N=1000 Landmarks] ──> [B(s) Backward Embeddings]
                                                               │
                                                               ▼
[Current State s_t] ──> [Geodesic Dijkstra Path] ──> [Compute Quasi-Metric Cost]
        │                       │                     c(u,v) = -log(F(u,B(v))^T B(v))
        │                       ▼
        └───> [Sliding Lookahead Window (L=2.6m)] ──> [Latent Subgoal z_t = B(w_t)]
                                                               │
                                                               ▼
                                                  [Low-Level Actor pi_l(a | s, z_t)]
                                                               │
                                                               ▼
                                                  [Continuous Action a_t in AntMaze]
```

### 1. Pure FB Successor Measure Geometry
In the Forward-Backward representation framework ([Touati & Ollivier, 2021](https://arxiv.org/abs/2103.07945); [Stojanovic & Proutiere, 2026](https://arxiv.org/abs/2605.13207)), the discounted occupancy measure density with respect to the data distribution $\rho$ is factorized as:
$$\frac{dM^{\pi_z}_s}{d\rho}(s') \approx F(s, z)^\top B(s'), \quad \|z\|_2 = \sqrt{d}$$

When obstacle walls intervene between the agent $s$ and target goal $g$, direct evaluation gives $F(s, B(g))^\top B(g) \approx 0$, causing a single-intention controller to get stuck permanently.

### 2. Buffer Graph Dijkstra with Sliding Lookahead
1. **Topological Landmark Sampling**: Sample $N=1000$ diverse states $S_V = \{s_i\}_{i=1}^N \subset \mathcal{D}_{\text{offline}}$.
2. **Quasi-Metric Transition Costs**:
   $$c(s_i, s_j) = \max\left(0, -\log\left(\frac{F(s_i, B(s_j))^\top B(s_j)}{M_{\max}}\right)\right)$$
3. **Wall Pruning**: Disallow transitions where Euclidean distance exceeds $\text{max\_edge\_radius} = 3.5\,\text{m}$ or reachability falls below $\text{reachability\_cutoff} = 35.0$.
4. **Continuous Sliding Lookahead ($L = 2.6\,\text{m}$)**: Rather than switching subgoals abruptly when reaching a discrete node, the planner queries along the topological path curve ahead by lookahead distance $L$. This produces smooth velocity profiles and prevents oscillation around tight corners.

### 3. Recursive Bisection ($O(\log K)$ Latent Search)
Computes intermediate waypoints hierarchically by finding the midpoint candidate that maximizes the joint transition likelihood:
$$w^* = \arg\max_{w \in \mathcal{D}} \left[ \log F(s, B(w))^\top B(w) + \log F(w, B(g))^\top B(g) \right]$$

### 4. Neural Policy Distillation
To achieve real-time control ($<1\,\text{ms}$), optimal teacher transitions $(s, z_g) \to z_{w}^*$ are distilled into specialized student models:
- **ResNet-ECA**: Deep residual blocks equipped with 1D Efficient Channel Attention to capture spatial coordinate inter-dependencies without dimensional bottlenecking.
- **Gated Cross-Attention (SwiGLU)**: Multi-head cross-attention between state queries and goal keys/values, modulated by SwiGLU non-linear gating.
- **Dense-ECA**: Feature-reuse network achieving $84.2\%$ success with only $179\text{k}$ parameters.

---

## Trajectory Visualizations

Qualitative rollouts in `antmaze-medium-navigate-v0` demonstrate how the Sliding Lookahead Dijkstra Planner cleanly routes around maze barriers while the baseline gets trapped:

| Comparison Visualizations | Description |
| :--- | :--- |
| ![Task 1 Comparison](results/plots/compare_3panel_seed0_task1_ep0.png) | **Task 1 (SW $\to$ NE)**: Single-Intention gets trapped at wall; Buffer Graph Dijkstra routes cleanly through corridors. |
| ![Task 2 Comparison](results/plots/compare_3panel_seed0_task2_ep0.png) | **Task 2 (NW $\to$ SE)**: Clean multi-corridor navigation via learned subgoals. |
| ![Overview Map](results/plots/overview_all_trajectories_seed0.png) | **Overview of all 5 Tasks**: Complete trajectory rollouts across start/goal configurations. |

Additional plots and rollouts can be found in:
- `results/plots/medium/success/`: Successful episode rollouts across all tasks.
- `results/plots/medium/failed/`: Failure case analysis.
- `results/plots/pareto_latency_accuracy.png`: Accuracy vs. Latency Pareto frontier.
- `results/plots/success_rate_comparison.png`: Bar chart comparison across planners.

---

## Quickstart Guide

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/StepnouXVI/fb-rl.git
cd fb-rl

# Create and activate conda environment
conda env create -f environment.yml
conda activate fb-rl

# Verify environment dependencies
python verify_env.py
```

### 2. Checkpoints

Pretrained FB checkpoints should be placed in `fb-test/`:
```text
fb-test/
├── modules_forward_repr
├── modules_backward_repr
├── modules_actor
└── modules_high_actor
```

### 3. Waypoint Translators & Distillation Training

#### Training on AntMaze-Medium
```bash
# 1. Train Distilled JAX Gated-CrossAttention Model
python scripts/train_jax_distillation.py --split=medium --model_type=gated_attn --epochs=40 --batch_size=256 --n_pairs=15000

# 2. Train Single-Waypoint Flax Translator
python scripts/train_waypoint_translators.py split=medium mode=single_wp n_pairs=60000 epochs=1000 batch_size=256

# 3. Train Enhanced Sequence-Aware Attention Translator
python scripts/train_waypoint_translators.py split=medium mode=enhanced_seq_attn n_pairs=60000 epochs=1000 batch_size=256
```

#### Training on AntMaze-Large
```bash
# Automated 4-Phase Pipeline (Train Distillation + Single WP + Enhanced Seq Attn + 10-Seed Benchmark)
bash scripts/run_training_pipeline_large.sh

# Or run individual steps:
python scripts/train_jax_distillation.py --split=large --model_type=gated_attn --epochs=100 --batch_size=256 --n_pairs=25000
python scripts/train_waypoint_translators.py split=large mode=single_wp n_pairs=60000 epochs=1000 batch_size=256
python scripts/train_waypoint_translators.py split=large mode=enhanced_seq_attn n_pairs=60000 epochs=1000 batch_size=256
```

### 4. Running Benchmarks

```bash
# Comprehensive 10-Seed Benchmark (Single WP, Sequence Attn, Enhanced Seq Attn, Distilled JAX, Baseline, Dijkstra)
python scripts/benchmark_translators.py --split=medium --num_tasks=5 --episodes_per_task=10
python scripts/benchmark_translators.py --split=large --num_tasks=5 --episodes_per_task=10

# Hydra Parallel Multi-Worker Benchmark
python scripts/run_benchmark.py env.split=medium
python scripts/run_benchmark.py env.split=large
```

### 5. Generating Trajectories & Visualizations

```bash
# Generate and export rollouts for all methods organized into results/{method_name}/{split}/{failed,success}/
python scripts/generate_all_method_rollouts_and_plots.py --split=large --num_tasks=5 --episodes_per_task=3

# Visualize trajectory plots, comparisons, and maze maps
python scripts/visualize_trajectories.py --maze-type large --compare --task 1 --seed 1 --episode 0
python scripts/visualize_trajectories.py --maze-type medium

# Run intention & low-level control experiments
python scripts/experiment_intention.py env=antmaze_medium scenario_name=all
python scripts/experiment_lowlevel_control.py

# Generate publication-ready Pareto and success rate charts
python scripts/generate_plots.py
```

---

## Project Structure

```text
fb-rl/
├── configs/                          # Hydra YAML configuration files
│   ├── config.yaml                   # Root Hydra configuration
│   ├── experiment.yaml               # Intention experiment configuration
│   ├── train_translator.yaml         # Waypoint translator training configuration
│   ├── env/                          # Environment configs (antmaze_medium, large, giant, teleport)
│   └── planner/                      # Planner configs (buffer_graph, baseline, etc.)
├── fb-test/                          # Pretrained FB representations & weights
├── outputs/                          # Training logs and model checkpoints
│   └── checkpoints/                  # Best trained translator and distillation weights
├── report/                           # Scientific LaTeX report and figures
├── results/                          # Benchmark logs, metrics, and organized rollout plots
│   ├── benchmarks/                   # Multi-seed CSV & Markdown benchmark summaries
│   └── plots/                        # Generated 2D trajectory visualizations
├── scripts/                          # Executable training, evaluation, and plotting scripts
│   ├── benchmark_translators.py      # Comprehensive 10-seed multi-method benchmark
│   ├── experiment_intention.py       # Intention and waypoint execution experiments
│   ├── experiment_lowlevel_control.py # Low-level control & straight-line maneuver benchmark
│   ├── generate_all_method_rollouts_and_plots.py # Batch trajectory rollout generator
│   ├── generate_plots.py             # Publication figure and Pareto generator
│   ├── run_benchmark.py              # Parallel multi-process evaluation engine
│   ├── run_training_pipeline_large.sh # End-to-end training & evaluation pipeline for Large
│   ├── train_jax_distillation.py     # Differentiable JAX/Flax student policy distillation
│   ├── train_waypoint_translators.py # Hydra Flax Single & Sequence Waypoint Attention Trainer
│   └── visualize_trajectories.py     # Trajectory, comparison, and maze rendering engine
├── src/                              # Core library source code
│   ├── agent_loader.py               # Pretrained checkpoint and dataset loader
│   ├── evaluator.py                  # High-performance evaluation & rollout recorder
│   ├── jax_distillation.py           # JAX/Flax student architectures & training steps
│   ├── metrics.py                    # Bootstrap CI and statistical tests
│   ├── models.py                     # Student neural architectures (ECA, SwiGLU)
│   ├── planners.py                   # Unified hierarchy of Zero-Shot FB Planners
│   └── waypoint_translators.py       # Sequence-aware attention translator architectures
├── tests/                            # Comprehensive unit and integration test suite
├── environment.yml                   # Conda environment definition
└── README.md                         # Project documentation
```

---

## Hydra Configuration Guide

Key configuration options in `configs/`:

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `env.name` | `ogbench-antmaze-medium-navigate-v0` | OGBench evaluation environment |
| `planner.type` | `all` | Planner selection (`baseline`, `buffer_graph`, `recursive_bisection`, `distilled_mlp`) |
| `planner.n_landmarks` | `1000` | Number of topological landmarks sampled from buffer |
| `planner.max_edge_radius` | `3.5` | Maximum Euclidean distance for graph edge connectivity |
| `planner.reachability_cutoff`| `35.0` | Minimum FB successor measure threshold for edge creation |
| `planner.lookahead_dist` | `2.6` | Geodesic lookahead distance along Dijkstra path |
| `eval.seeds` | `[0]` | Random seeds for multi-seed statistical evaluation |
| `eval.num_episodes` | `15` | Number of evaluation episodes per task |
| `eval.n_workers` | `4` | Number of parallel evaluation processes |

---

## Citation & References

```bibtex
@article{stojanovic2026switching,
  title={Switching Successor Measures for Hierarchical Zero-Shot Reinforcement Learning},
  author={Stojanovic, Petar and Proutiere, Alexandre},
  journal={arXiv preprint arXiv:2605.13207},
  year={2026}
}

@article{touati2021learning,
  title={Learning One Representation to Optimize All Rewards},
  author={Touati, Ahmed and Ollivier, Yann},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  pages={13--23},
  year={2021}
}

@article{park2025ogbench,
  title={OGBench: Benchmarking Offline Goal-Conditioned RL},
  author={Park, Seohong and others},
  journal={arXiv preprint arXiv:2410.20092},
  year={2025}
}
```
