# Codebase Architecture & System Design

This document details the software architecture, modular class hierarchy, configuration schema, and execution pipeline of the Forward-Backward Reinforcement Learning (FB-RL) zero-shot planning repository.

---

## 1. System Architecture Overview

The system is organized into modular packages separating neural model loading, planning strategies, evaluation engines, configuration management, and visualization:

```
fb-rl/
├── configs/                  # Hydra hierarchical configuration trees
│   ├── config.yaml           # Root configuration entry point
│   ├── env/                  # Environment definitions (medium, large, giant, teleport)
│   └── planner/              # Planner algorithms and hyperparameter presets
├── src/                      # Core runtime implementation
│   ├── agent_loader.py       # Pretrained FB checkpoint & environment deserializer
│   ├── evaluator.py          # Deterministic zero-shot task evaluation engine
│   ├── metrics.py            # Statistical analysis, bootstrap CI, LaTeX table export
│   ├── models.py             # PyTorch distilled student architectures
│   └── planners.py           # Unified hierarchy of Zero-Shot FB Planners
├── scripts/                  # Executable CLI benchmarks and visualization
│   ├── run_benchmark.py      # Multi-process parallel evaluation orchestrator
│   ├── visualize_trajectories.py # High-resolution publication plotting suite
│   ├── train_distillation.py # Distillation training pipeline
│   └── generate_plots.py     # Batch plot generation utility
└── docs/                     # Technical specifications and benchmark reports
```

---

## 2. Planners Class Hierarchy (`src/planners.py`)

All planning strategies inherit from [`BasePlanner`](file:///Users/savvatej/source/fb-rl/src/planners.py#L64) and implement a common stateless/stateful interface:
- `reset(obs, goal_latent)`: Called at the start of each evaluation episode.
- `sample_action(obs, goal_latent, step, seed, temperature)`: Called at each environment control step.
- `get_subgoal_info()`: Returns telemetry including active subgoals, planned waypoints, and direct-goal flags for logging.

```
                         +-------------------+
                         |    BasePlanner    |
                         +-------------------+
                                   |
         +-----------------+-------+-------+--------------------+
         |                 |               |                    |
+-----------------+ +--------------+ +---------------+ +------------------+
| BaselinePlanner | | BufferGraph- | | RecursiveBi-  | | DistilledMLP-    |
|                 | | Planner      | | sectionPlanner| | Planner          |
+-----------------+ +--------------+ +---------------+ +------------------+
```

### 2.1. [`BasePlanner`](file:///Users/savvatej/source/fb-rl/src/planners.py#L64)
- **Role**: Base abstract planner providing standard fallback execution and subgoal telemetry interface.
- **Key Method**: `sample_action` directly queries `_jit_baseline_step`.

### 2.2. [`BaselinePlanner`](file:///Users/savvatej/source/fb-rl/src/planners.py#L83)
- **Role**: Single-Intention Baseline executing the two-level hierarchy ($\pi_{\text{high}}$ and $\pi_{\text{low}}$) conditioned directly on the downstream goal latent $z_{\text{goal}}$.
- **Latent Decoding**: Optionally projects $z_{\text{subgoal}}$ onto the nearest offline state landmark for visualization via [`_jit_decode_latent_to_coords`](file:///Users/savvatej/source/fb-rl/src/planners.py#L52-L58).

### 2.3. [`BufferGraphPlanner`](file:///Users/savvatej/source/fb-rl/src/planners.py#L131)
- **Role**: Branch 2 Topological Shortest-Path Planner.
- **Components**:
  - Offline landmark sampling ($N=1000$).
  - Chunked JAX reachability matrix evaluation [`_jit_batch_reach`](file:///Users/savvatej/source/fb-rl/src/planners.py#L32-L39).
  - Geometry-aware topological masking ($r \le 3.5\text{ m}$, $\text{reach} \ge 35.0$).
  - SciPy sparse Dijkstra all-pairs shortest paths.
  - Continuous Sliding Lookahead tracking ($L = 2.6\text{ m}$).
  - Stuck recovery detection via rolling position buffer.

### 2.4. [`RecursiveBisectionPlanner`](file:///Users/savvatej/source/fb-rl/src/planners.py#L320)
- **Role**: Branch 1 Recursive Midpoint Bisection Planner.
- **Components**: Evaluates dual-reachability score $\log \text{reach}(s, w) + \log \text{reach}(w, g)$ over a candidate set of $K=200$ buffer states, resetting subgoals when reaching proximity threshold $\tau_{\text{hit}} \ge 35.0$.

### 2.5. [`DistilledMLPPlanner`](file:///Users/savvatej/source/fb-rl/src/planners.py#L380)
- **Role**: Branch 3 Neural Student Policy.
- **Components**: Directly predicts the optimal intermediate latent intention $z_{\text{subgoal}} = f_\theta(s, z_{\text{goal}})$ using a lightweight PyTorch neural network ([`DenseECANetwork`](file:///Users/savvatej/source/fb-rl/src/models.py#L34), [`GatedCrossAttentionNetwork`](file:///Users/savvatej/source/fb-rl/src/models.py#L61), or [`StandardMLP`](file:///Users/savvatej/source/fb-rl/src/models.py#L20)), executing in $<0.05\text{ ms}$.

---

## 3. Core Supporting Modules

### 3.1. Agent Loader ([`src/agent_loader.py`](file:///Users/savvatej/source/fb-rl/src/agent_loader.py))
The [`load_pretrained_agent`](file:///Users/savvatej/source/fb-rl/src/agent_loader.py#L7-L40) function deserializes Flax FB checkpoints and initializes the OGBench environment:
- Loads hyperparameters from `flags.json` and weights from `params.pkl`.
- Disables goal-noise injection (`env.unwrapped._add_noise_to_goal = False`).
- Enforces `max_episode_steps` recursively across nested Gym wrapper layers.
- Restores the actor, high-level actor, forward representation, and backward representation networks into an active `FBpiSwitchAgent`.

### 3.2. Zero-Shot Evaluator ([`src/evaluator.py`](file:///Users/savvatej/source/fb-rl/src/evaluator.py))
The [`ZeroShotEvaluator`](file:///Users/savvatej/source/fb-rl/src/evaluator.py#L12-L185) orchestrates deterministic benchmarking across tasks:
- **Latent Inference Caching**: Computes the zero-shot task latent $z_{\text{goal}} = \text{infer\_latent}(\mathcal{D}_{\text{relabelled}})$ once per $(task\_id, seed)$ and caches it in memory.
- **Detailed Step Telemetry**: Logs $(x, y)$ agent coordinates, reward signals, step indices, active subgoals, and planned waypoints into structured DataFrames.
- **Self-Intersection Analysis**: Calls [`count_self_intersections`](file:///Users/savvatej/source/fb-rl/src/evaluator.py#L141-L142) on trajectory paths to monitor looping behavior.

### 3.3. Statistical Metrics Aggregator ([`src/metrics.py`](file:///Users/savvatej/source/fb-rl/src/metrics.py))
- [`bootstrap_ci`](file:///Users/savvatej/source/fb-rl/src/metrics.py#L7): Computes 95% non-parametric bootstrap confidence intervals across $B=2000$ resamples.
- [`aggregate_runs`](file:///Users/savvatej/source/fb-rl/src/metrics.py#L16): Aggregates per-seed metrics and computes Welch's two-sample $t$-test against baseline.
- [`export_latex_table`](file:///Users/savvatej/source/fb-rl/src/metrics.py#L49): Automatically compiles publication-ready LaTeX tables (`summary_table.tex`).

---

## 4. Hydra Configuration Hierarchy

The project utilizes [Hydra](https://hydra.cc/) for configuration management. Configurations are dynamically composed from `configs/config.yaml`:

```
configs/
├── config.yaml
├── env/
│   ├── antmaze_medium.yaml
│   ├── antmaze_large.yaml
│   ├── antmaze_giant.yaml
│   └── antmaze_teleport.yaml
└── planner/
    ├── all.yaml
    ├── baseline.yaml
    ├── baseline_vs_dijkstra.yaml
    ├── buffer_graph.yaml
    ├── distilled_mlp.yaml
    └── recursive_bisection.yaml
```

### 4.1. Root Config ([`configs/config.yaml`](file:///Users/savvatej/source/fb-rl/configs/config.yaml))
```yaml
defaults:
  - env: antmaze_medium
  - planner: all
  - _self_

eval:
  checkpoint_dir: "fb-test"
  seeds: [0]
  num_episodes: 15
  eval_temperature: 0.0
  video_episodes: 0
  video_frame_skip: 3
  output_dir: "results"
  n_workers: 4
  device: "auto"

distillation:
  model_type: "dense_eca"
  n_pairs: 2000
  batch_size: 128
  epochs: 20
  lr: 1e-3
  hidden_dim: 256
  n_layers: 3
  num_heads: 4
  device: "auto"
```

### 4.2. Overriding via CLI
Hydra enables flexible composition and parameter overrides directly from the command line:
```bash
# Run baseline vs dijkstra on AntMaze Medium with 8 parallel workers
python scripts/run_benchmark.py planner=baseline_vs_dijkstra eval.n_workers=8

# Run evaluation on AntMaze Large with custom episode limit
python scripts/run_benchmark.py env=antmaze_large env.max_episode_steps=1500

# Evaluate only the Buffer Graph Dijkstra planner with custom lookahead
python scripts/run_benchmark.py planner=buffer_graph planner.lookahead_dist=3.0
```

---

## 5. Dynamic Parameter Passing & Environment Unwrapping

Gymnasium and OGBench wrap simulation environments inside multiple nested wrapper layers (`TimeLimit`, `OrderEnforcing`, `PixelObservationWrapper`, etc.). To guarantee that user-specified `max_episode_steps` are strictly respected without premature truncation or infinite loops:

1. **Loader-Level Propagation** ([`src/agent_loader.py`](file:///Users/savvatej/source/fb-rl/src/agent_loader.py#L25-L31)):
   ```python
   if max_episode_steps is not None:
       curr = env
       while curr is not None:
           if hasattr(curr, "_max_episode_steps"):
               curr._max_episode_steps = int(max_episode_steps)
           curr = getattr(curr, "env", None)
   ```

2. **Evaluator-Level Step Guard** ([`src/evaluator.py`](file:///Users/savvatej/source/fb-rl/src/evaluator.py#L105-L107)):
   ```python
   step += 1
   if max_steps is not None and step >= max_steps:
       truncated = True
   done = terminated or truncated
   ```

This dual enforcement ensures consistent episode termination across all environments and custom configurations.
