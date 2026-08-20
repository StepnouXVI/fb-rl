# Trajectory Visualizer Guide: `scripts/visualize_trajectories.py`

This guide details the features, CLI options, rendering modes, and practical usage recipes of the high-resolution trajectory and subgoal visualizer suite in [`scripts/visualize_trajectories.py`](file:///Users/savvatej/source/fb-rl/scripts/visualize_trajectories.py).

---

## 1. Overview and Capabilities

The visualizer tool transforms CSV telemetry logged during benchmark evaluation (`trajectories.csv`, `subgoals.csv`, and `landmarks.csv`) into publication-grade Matplotlib plots.

### Key Capabilities:
- **Exact Grid Alignment**: Renders maze walls, open corridors, start locations (green circle), and target goals (red star) aligned to physical simulation coordinates.
- **Subgoal & Waypoint Overlays**: Visualizes high-level subgoals (magenta diamonds), topological planned waypoints (cyan crosses), and connecting lookahead lines.
- **Teleport Portals & Curved Jump Links**: Automatically renders teleport entrances (blue circles), exits (purple squares), and curved jump trajectories for teleportation mazes.
- **Outcome Organization**: Automatically organizes output plots into `success/` and `failed/` directories.

---

## 2. Command-Line Reference

```bash
python scripts/visualize_trajectories.py [OPTIONS]
```

### Table of CLI Arguments

| Argument | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `--traj_file` | `str` | `"results/trajectories.csv"` | Path to input trajectories CSV file. |
| `--sg_file` | `str` | `"results/subgoals.csv"` | Path to subgoals CSV file (optional). |
| `--lm_file` | `str` | `"results/landmarks.csv"` | Path to sampled landmarks CSV file (optional). |
| `--output_dir` | `str` | `"results/plots"` | Base output directory for exported plots. |
| `--maze_type` | `str` | `"auto"` | Maze environment layout: `auto`, `medium`, `large`, `giant`, `teleport`. |
| `--unit_size` | `float` | `4.0` | Grid cell size in physical meters (default: 4.0). |
| `--custom_layout`| `str` | `None` | Path to custom text/JSON maze grid layout file. |
| `--method` | `str` | `None` | Filter by planner method name (e.g. `"Buffer Graph Dijkstra (Branch 2)"`). |
| `--seed` | `int` | `0` | Random evaluation seed index. |
| `--task` | `int` | `1` | Task ID to visualize (1 through 5). |
| `--episode` | `int` | `0` | Specific episode index (0 through 14). |
| `--compare` | `flag`| `False` | Renders a 3-panel comparison (Baseline vs. Dijkstra vs. Landmarks). |
| `--export_comparisons`| `flag`| `False` | Batch exports 3-panel comparisons for all 75 episodes into `success/` and `failed/`. |
| `--all_episodes`| `flag`| `False` | Overlays all 15 evaluation episodes for a given task and seed onto a single plot. |
| `--overview` | `flag`| `False` | Generates a complete $5 \times M$ grid of all tasks and methods for the given seed. |
| `--no_portals` | `flag`| `False` | Disables rendering of teleport portal markers. |
| `--no_portal_links`| `flag`| `False`| Disables curved arrow rendering between teleport in/out portals. |
| `--save_path` | `str` | `None` | Custom path to save the generated image file. |

---

## 3. Visualization Modes & Usage Recipes

### 3.1. Mode 1: Single Trajectory Plotting
Plots a single episode trajectory with step-by-step path coloring, start/goal markers, and active subgoals.

```bash
# Plot Episode 0 of Task 1 for Buffer Graph Dijkstra
python scripts/visualize_trajectories.py \
  --method "Buffer Graph Dijkstra (Branch 2)" \
  --task 1 \
  --episode 0 \
  --seed 0

# Plot Episode 2 of Task 4 for the Baseline Planner
python scripts/visualize_trajectories.py \
  --method "Single-Intention Baseline" \
  --task 4 \
  --episode 2 \
  --seed 0
```
*Output Path*: `results/plots/medium/success/Buffer_Graph_Dijkstra_Branch_2_medium_seed0_task1_ep0.png`

---

### 3.2. Mode 2: 3-Panel Side-by-Side Comparison (`--compare`)
Generates a horizontal 3-panel comparison figure:
1. **Left Panel**: Single-Intention Baseline trajectory and decoded subgoals.
2. **Center Panel**: Buffer Graph Dijkstra trajectory with sliding lookahead targets and planned topological waypoints.
3. **Right Panel**: Full buffer landmark graph overlay ($N=1000$ points) illustrating topological coverage.

```bash
# Compare Baseline vs. Dijkstra on Task 1, Episode 0
python scripts/visualize_trajectories.py \
  --compare \
  --task 1 \
  --episode 0 \
  --seed 0

# Compare on Task 5, Episode 3
python scripts/visualize_trajectories.py \
  --compare \
  --task 5 \
  --episode 3 \
  --seed 0
```
*Output Path*: `results/plots/medium/success/compare_medium_seed0_task1_ep0.png`

---

### 3.3. Mode 3: All-Episodes Overlay (`--all_episodes`)
Overlays all 15 evaluation episodes for a specific method and task onto a single maze figure. Successful trajectories are rendered in dark blue, while unsuccessful/timed-out trajectories are rendered in red.

```bash
# Overlay all 15 episodes of Task 1 for Buffer Graph Dijkstra
python scripts/visualize_trajectories.py \
  --all_episodes \
  --method "Buffer Graph Dijkstra (Branch 2)" \
  --task 1 \
  --seed 0

# Overlay all 15 episodes of Task 2 for Single-Intention Baseline
python scripts/visualize_trajectories.py \
  --all_episodes \
  --method "Single-Intention Baseline" \
  --task 2 \
  --seed 0
```
*Output Path*: `results/plots/medium/Buffer_Graph_Dijkstra_Branch_2_medium_seed0_task1_all_episodes.png`

---

### 3.4. Mode 4: Full Overview Grid (`--overview`)
Generates a comprehensive grid containing rows for all 5 tasks and columns for all evaluated planning methods for a complete visual summary.

```bash
# Generate complete multi-task, multi-method overview grid
python scripts/visualize_trajectories.py --overview --seed 0
```
*Output Path*: `results/plots/medium/overview_medium_seed0.png`

---

### 3.5. Mode 5: Batch Comparison Export (`--export_comparisons`)
Iterates over all 75 benchmark episodes across all tasks and automatically generates 3-panel comparison plots, sorting them into outcome folders:

```bash
python scripts/visualize_trajectories.py --export_comparisons --seed 0
```
*Output Hierarchy*:
```
results/plots/medium/
├── success/
│   ├── compare_medium_seed0_task1_ep0.png
│   ├── compare_medium_seed0_task1_ep1.png
│   └── ...
└── failed/
    ├── compare_medium_seed0_task4_ep2.png
    └── ...
```

---

## 4. Custom Layouts and Advanced Environments

### Custom Maze Grids
To visualize custom environments or altered maze geometries, pass a JSON or text matrix file:
```bash
python scripts/visualize_trajectories.py \
  --custom_layout my_custom_maze.json \
  --task 1 \
  --episode 0
```

### Teleportation Environments
For `ogbench-antmaze-teleport-navigate-v0`, the visualizer automatically detects portal jumps and renders colored entrance rings, exit squares, and dashed arc linkages. Portals can be suppressed with `--no_portals` and `--no_portal_links` if desired.
