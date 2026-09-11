# Design Specification: StagedAgent Architecture & Unified Telemetry Refactoring

## 1. Overview & Motivation
This document specifies the refactoring of the `fb-rl` and `fb-rl-video` codebases to:
1. Replace monolithic, tightly-coupled planner classes with a modular, composable `StagedAgent` pipeline.
2. Formalize agent context structures via typed hierarchies (`BaseMazeContext`, `TopologicalPathContext`, `SequenceAttentionContext`).
3. Transition from fragmented 800MB CSV files to a 2-tier telemetry architecture:
   - Tier 1: High-performance SQLite database (`results/data/telemetry.db`) in WAL mode for full per-step telemetry, dynamic replanning paths, and terminal metrics.
   - Tier 2: Aim 3.x dashboard (`results/aim`) for high-level experiment metrics, hyperparameter tracking, and media artifacts (trajectory PNGs, Manim MP4 videos).
4. Eliminate framework-leaking naming conventions (`DistilledJAXPlanner` -> `DirectIntentionPlanner`, `Flax...` -> domain names).
5. Clean up dead code, redundant scripts, and unnecessary dependencies across the repository.
6. Enforce code quality standards: no functions exceeding 60 lines, single responsibility principle (SRP) for all classes, accurate latency profiling (excluding environment physics and serialization), and strictly NO comments in code except docstrings.

---

## 2. Component Architecture

### 2.1. StagedAgent Pipeline
An instance of `StagedAgent` represents any navigation agent constructed from an ordered list of pipeline handlers:
`agent = StagedAgent(stages=[...], context_cls=..., name=...)`

Execution lifecycle per step:
1. Initialize typed `context_cls(obs, goal_latent, step, seed, temperature)`.
2. Carry forward persistent state (e.g. topological path coordinates, path indices) from the previous step.
3. Sequentially invoke each stage: `stage(ctx)`.
4. Return `ctx.action` (an 8-dimensional torque vector).

### 2.2. Pipeline Handlers (No "Stage" suffix in naming)
- `DijkstraPathBuilder`: Graph maintenance, Dijkstra search, deviation check (>4.8m trigger replan), stuck state detection.
- `PathLogger`: Logs initial and dynamic replan paths into SQLite (`planned_paths` table) only when triggered.
- `AttentionFilter`: Continuous sliding lookahead marching (2.6m) and selection of 4 downstream landmarks + goal.
- `LandMarksForAttentionLogger`: Populates telemetry fields for lookahead coordinate, targets, and attention weights.
- `SequenceAttentionTranslator`: Neural network inference computing intention vector $z_{cmd}$ via sequence cross-attention with ALiBi and curvature embeddings.
- `DirectIntentionTranslator`: $O(1)$ amortized neural network predicting $z_{cmd}$ directly from current state and goal.
- `HighLevelActor`: Baseline high-level actor predicting intention from goal.
- `LowLevelActor`: Evaluates low-level policy $\pi^\ell(a \mid s, z_{cmd})$ to produce motor torque actions.

### 2.3. Context Hierarchy
- `BaseMazeContext`: Common fields (`obs`, `goal_latent`, `step`, `seed`, `temperature`, `action`, `z_cmd`, `telemetry`).
- `TopologicalPathContext(BaseMazeContext)`: Path fields (`path_coords`, `path_latents`, `waypoint_coords`, `current_path_idx`, `replan_triggered`, `lookahead_xy`, `dist_to_lookahead`).
- `SequenceAttentionContext(TopologicalPathContext)`: Attention fields (`attention_targets`, `attention_weights`, `pad_seq`, `seq_mask`, `curv_arr`).

### 2.4. Telemetry Architecture
- `src/telemetry/db.py`: `TelemetryDatabase` wrapping SQLite with WAL mode, normal synchronous, 64MB cache, foreign keys enabled.
  - Tables: `experiments`, `runs`, `landmarks`, `episodes`, `planned_paths`, `steps`.
- `src/telemetry/metrics.py`:
  - `compute_cross_track_errors`: Vectorized point-to-segment distance with progress windowing to prevent wall penetration projection.
  - `count_spatial_self_intersections`: Spatial subsampling (min_step_dist=0.35m) with CCW orientation predicate and AABB fast bounding box rejection.
- `src/telemetry/aim_tracker.py`:
  - Wrapper around `aim.Run` tracking scalars, hyperparameters, image artifacts, and video artifacts. Graceful degradation if disabled or uninstalled.
- `fb-rl-video/fb_video/data/sqlite_reader.py`:
  - `SQLiteRunReader` in video repo loading `AlgorithmDataRecord` directly from `telemetry.db` without importing RL simulation packages (no Gym, JAX, PyTorch, MuJoCo).

---

## 3. Code Standards
- Function length: strictly <= 60 lines.
- Comments: NO comments in code except docstrings (including no `# ponytail:` comments).
- Latency profiling: isolated to model inference only; excludes `env.step()` and telemetry serialization. Includes warm-up step to exclude JIT compilation.
- Clean dependencies: remove PyTorch, Torchvision, Plotly, MLflow, Tabulate from `requirements.txt`.
