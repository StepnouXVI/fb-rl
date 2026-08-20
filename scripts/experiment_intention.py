import os
import sys
import json
import time
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
import hydra
from omegaconf import DictConfig, OmegaConf
import jax
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.agent_loader import load_pretrained_agent
from src.planners import BufferGraphPlanner, _jit_baseline_step, _jit_batch_reach

# ==============================================================================
# Maze Layout Definitions
# ==============================================================================

MAZE_LAYOUTS = {
    "medium": np.array([
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 1, 1, 0, 0, 1],
        [1, 0, 0, 1, 0, 0, 0, 1],
        [1, 1, 0, 0, 0, 1, 1, 1],
        [1, 0, 0, 1, 0, 0, 0, 1],
        [1, 0, 1, 0, 0, 1, 0, 1],
        [1, 0, 0, 0, 1, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1],
    ]),
    "large": np.array([
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
        [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
        [1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
        [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1],
        [1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1],
        [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    ]),
    "giant": np.array([
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1],
        [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1],
        [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
        [1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1],
        [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1],
        [1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1],
        [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
        [1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1],
        [1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1],
        [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    ]),
    "teleport": np.array([
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1],
        [1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1],
        [1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1],
        [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        [1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    ]),
}


def normalize_maze_type(maze_type):
    if maze_type is None:
        return "medium"
    s = str(maze_type).lower().strip()
    s = s.replace("antmaze-", "").replace("pointmaze-", "").replace("humanoidmaze-", "")
    s = s.replace("-navigate-v0", "").replace("-singletask-v0", "").replace("-v0", "")
    return s if s in MAZE_LAYOUTS else ("teleport" if "teleport" in s else ("giant" if "giant" in s else ("large" if "large" in s else "medium")))


def draw_maze(ax, maze_type="medium", unit_size=4.0, offset_x=4.0, offset_y=4.0):
    norm_type = normalize_maze_type(maze_type)
    grid = MAZE_LAYOUTS.get(norm_type, MAZE_LAYOUTS["medium"])
    h, w = grid.shape
    for i in range(h):
        for j in range(w):
            if grid[i, j] == 1:
                x = j * unit_size - offset_x - unit_size / 2.0
                y = i * unit_size - offset_y - unit_size / 2.0
                rect = patches.Rectangle(
                    (x, y), unit_size, unit_size, linewidth=1, edgecolor="#333333", facecolor="#d0d0d0", zorder=1
                )
                ax.add_patch(rect)

    x_min = -offset_x - unit_size / 2.0
    x_max = (w - 1) * unit_size - offset_x + unit_size / 2.0
    y_min = -offset_y - unit_size / 2.0
    y_max = (h - 1) * unit_size - offset_y + unit_size / 2.0

    ax.set_xlim(x_min - 0.5, x_max + 0.5)
    ax.set_ylim(y_min - 0.5, y_max + 0.5)
    ax.set_aspect("equal")
    ax.set_facecolor("#fafafa")
    ax.grid(True, linestyle=":", alpha=0.35, color="#888888")


# ==============================================================================
# Helper Functions: State & Coordinate Encoding
# ==============================================================================

def encode_coord_to_z(agent, dataset_observations, target_xy):
    """Finds nearest dataset state to target_xy and encodes it via backward representation."""
    coords = dataset_observations[:, :2]
    dists = np.linalg.norm(coords - np.asarray(target_xy), axis=-1)
    best_idx = int(np.argmin(dists))
    best_state = dataset_observations[best_idx : best_idx + 1]
    z = np.asarray(agent.normalize_z(agent.network.select("backward_repr")(jnp.asarray(best_state)))[0])
    return z, dataset_observations[best_idx]


# ==============================================================================
# Core Experiment Execution Engine
# ==============================================================================

def run_scenario(
    agent,
    env,
    train_ds,
    config,
    scenario_cfg,
    eval_temperature=0.0,
    target_radius=1.0,
    terminate_on_goal=True,
    seed=0,
    cached_planner=None,
):
    """
    Executes a single intention experiment scenario with exact initial ant placement,
    step-by-step telemetry collection, and mode-specific intention generation.
    """
    name = scenario_cfg.get("name", "unnamed_scenario")
    start_xy = list(scenario_cfg.get("start_xy", [0.0, 0.0]))
    target_xy = list(scenario_cfg.get("target_xy", [0.0, 0.0]))
    mode = str(scenario_cfg.get("mode", "planner")).lower()
    max_steps = int(scenario_cfg.get("max_steps", 500))
    waypoint_threshold = float(scenario_cfg.get("waypoint_threshold", 1.8))
    planner_cfg = scenario_cfg.get("planner_cfg", {})

    # Reset environment & strictly set target goal and ant position
    env.action_space.seed(seed)
    np.random.seed(seed)
    env.reset(seed=seed)

    # Set target goal in unwrapped env to prevent premature completion at reset state
    if hasattr(env.unwrapped, "set_goal"):
        env.unwrapped.set_goal(goal_xy=np.asarray(target_xy, dtype=np.float32))

    # Strictly position ant at start_xy and synchronize Mujoco observation
    env.unwrapped.set_xy(np.asarray(start_xy, dtype=np.float32))
    obs = env.unwrapped.get_ob()

    # Pre-encode target_z
    target_z, _ = encode_coord_to_z(agent, train_ds["observations"], target_xy)

    # Initialize mode-specific planning state
    waypoints = []
    planner = None
    cw_latents = []
    cw_list = []
    z_vector = None

    if mode == "planner":
        p_landmarks = int(planner_cfg.get("n_landmarks", 1000))
        p_max_edge = float(planner_cfg.get("max_edge_radius", 3.5))
        p_reach_cut = float(planner_cfg.get("reachability_cutoff", 35.0))
        p_lookahead = float(planner_cfg.get("lookahead_dist", 2.6))

        if (
            cached_planner is not None
            and cached_planner.n_landmarks == p_landmarks
            and cached_planner.max_edge_radius == p_max_edge
            and cached_planner.reachability_cutoff == p_reach_cut
            and cached_planner.lookahead_dist == p_lookahead
        ):
            planner = cached_planner
        else:
            planner = BufferGraphPlanner(
                agent,
                train_ds["observations"],
                n_landmarks=p_landmarks,
                max_edge_radius=p_max_edge,
                reachability_cutoff=p_reach_cut,
                lookahead_dist=p_lookahead,
            )

        planner.reset(obs, target_z)
        waypoints = [c.tolist() if isinstance(c, np.ndarray) else list(c) for c in planner.waypoint_coords]

    elif mode == "direct_latent":
        waypoints = [target_xy]

    elif mode == "custom_waypoints":
        raw_cw = scenario_cfg.get("custom_waypoints", [])
        cw_list = [list(pt) for pt in raw_cw]
        if not cw_list:
            cw_list = [target_xy]
        elif np.linalg.norm(np.asarray(cw_list[-1]) - np.asarray(target_xy)) > 0.5:
            cw_list.append(target_xy)

        cw_latents = [encode_coord_to_z(agent, train_ds["observations"], pt)[0] for pt in cw_list]
        waypoints = cw_list

    elif mode == "custom_z":
        raw_z = scenario_cfg.get("custom_z", None)
        if raw_z is not None:
            z_arr = np.asarray(raw_z, dtype=np.float32)
            z_vector = np.asarray(agent.normalize_z(jnp.asarray(z_arr)))
        else:
            z_vector = target_z
        waypoints = [target_xy]
    else:
        raise ValueError(f"Unknown intention mode: {mode}")

    # Telemetry storage
    records = []
    current_wp_idx = 0
    step = 0

    # Initial state (Step 0)
    cur_pos = np.asarray(obs[:2])
    if mode == "planner":
        sg_info = planner.get_subgoal_info()
        sg_xy = sg_info.get("subgoal_xy") or target_xy
        z_0 = planner.get_subgoal_latent(obs, target_z, step=0)
    elif mode == "direct_latent":
        sg_xy = target_xy
        z_0 = target_z
    elif mode == "custom_waypoints":
        sg_xy = cw_list[current_wp_idx]
        z_0 = cw_latents[current_wp_idx]
    elif mode == "custom_z":
        sg_xy = target_xy
        z_0 = z_vector

    d_sg_0 = float(np.linalg.norm(cur_pos - np.asarray(sg_xy)))
    d_goal_0 = float(np.linalg.norm(cur_pos - np.asarray(target_xy)))

    records.append({
        "scenario_name": name,
        "mode": mode,
        "seed": int(seed),
        "step": 0,
        "x": float(cur_pos[0]),
        "y": float(cur_pos[1]),
        "action_norm": 0.0,
        "speed": 0.0,
        "subgoal_x": float(sg_xy[0]),
        "subgoal_y": float(sg_xy[1]),
        "dist_to_subgoal": d_sg_0,
        "dist_to_goal": d_goal_0,
        "alignment_cos": 1.0,
        "reached_goal": bool(d_goal_0 <= target_radius),
        "reward": 0.0,
        "done": False,
    })

    done = False
    goal_ever_reached = bool(d_goal_0 <= target_radius)

    while step < max_steps and not done:
        seed_k = jax.random.PRNGKey(seed * 10000 + step)
        prev_pos = cur_pos.copy()

        # Step 1: Intention & Action sampling
        if mode == "planner":
            action = planner.sample_action(
                obs, target_z, step=step, seed=seed_k, temperature=eval_temperature
            )
            sg_info = planner.get_subgoal_info()
            sg_xy = sg_info.get("subgoal_xy") or target_xy
            z_t = planner.get_subgoal_latent(obs, target_z, step=step)

        elif mode == "direct_latent":
            action, _ = _jit_baseline_step(
                agent, jnp.asarray(obs), jnp.asarray(target_z), seed=seed_k, temperature=eval_temperature
            )
            action = np.asarray(action)
            sg_xy = target_xy
            z_t = target_z

        elif mode == "custom_waypoints":
            dist_to_cur_wp = float(np.linalg.norm(cur_pos - np.asarray(cw_list[current_wp_idx])))
            if dist_to_cur_wp <= waypoint_threshold and current_wp_idx < len(cw_list) - 1:
                current_wp_idx += 1

            active_z = cw_latents[current_wp_idx]
            sg_xy = cw_list[current_wp_idx]
            action, _ = _jit_baseline_step(
                agent, jnp.asarray(obs), jnp.asarray(active_z), seed=seed_k, temperature=eval_temperature
            )
            action = np.asarray(action)
            z_t = active_z

        elif mode == "custom_z":
            action, _ = _jit_baseline_step(
                agent, jnp.asarray(obs), jnp.asarray(z_vector), seed=seed_k, temperature=eval_temperature
            )
            action = np.asarray(action)
            sg_xy = target_xy
            z_t = z_vector

        # Step 2: Environment Step
        next_obs, reward, terminated, truncated, info = env.step(action)
        step += 1

        cur_pos = np.asarray(next_obs[:2])
        obs = next_obs

        # Step 3: Compute Telemetry
        disp = cur_pos - prev_pos
        speed = float(np.linalg.norm(disp))
        subgoal_vec = np.asarray(sg_xy) - prev_pos
        subgoal_mag = float(np.linalg.norm(subgoal_vec))

        if speed > 1e-4 and subgoal_mag > 1e-4:
            cos_theta = float(np.dot(disp, subgoal_vec) / (speed * subgoal_mag))
            cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
        else:
            cos_theta = 1.0 if float(np.linalg.norm(cur_pos - np.asarray(target_xy))) <= target_radius else 0.0

        d_sg = float(np.linalg.norm(cur_pos - np.asarray(sg_xy)))
        d_goal = float(np.linalg.norm(cur_pos - np.asarray(target_xy)))
        reached_now = bool(d_goal <= target_radius)
        if reached_now:
            goal_ever_reached = True

        if terminate_on_goal and reached_now:
            done = True
        elif terminated or truncated or step >= max_steps:
            done = True

        records.append({
            "scenario_name": name,
            "mode": mode,
            "seed": int(seed),
            "step": step,
            "x": float(cur_pos[0]),
            "y": float(cur_pos[1]),
            "action_norm": float(np.linalg.norm(action)),
            "speed": speed,
            "subgoal_x": float(sg_xy[0]),
            "subgoal_y": float(sg_xy[1]),
            "dist_to_subgoal": d_sg,
            "dist_to_goal": d_goal,
            "alignment_cos": cos_theta,
            "reached_goal": reached_now,
            "reward": float(reward),
            "done": bool(done),
        })

    df_telemetry = pd.DataFrame(records)

    summary = {
        "scenario_name": name,
        "mode": mode,
        "seed": int(seed),
        "split": str(config.get("split", "medium")),
        "start_x": float(start_xy[0]),
        "start_y": float(start_xy[1]),
        "target_x": float(target_xy[0]),
        "target_y": float(target_xy[1]),
        "num_steps": int(step),
        "reached_goal": bool(goal_ever_reached),
        "min_dist_to_goal": float(df_telemetry["dist_to_goal"].min()),
        "final_dist_to_goal": float(df_telemetry["dist_to_goal"].iloc[-1]),
        "mean_speed": float(df_telemetry["speed"].mean()),
        "max_speed": float(df_telemetry["speed"].max()),
        "mean_subgoal_alignment": float(df_telemetry["alignment_cos"].mean()),
    }

    return summary, df_telemetry, waypoints, planner


# ==============================================================================
# Visualization Engine (High-Resolution Dual-Panel Plots)
# ==============================================================================

def plot_scenario_results(summary, df_telemetry, waypoints, output_path, maze_type="medium", target_radius=1.0):
    """
    Renders high-resolution dual-panel plot:
    - Left Panel: 2D Maze, trajectory with colormap, start/target markers, waypoints, and intention arrows.
    - Right Panel: 3 subplots (Distance to Target/Subgoal vs step, Speed vs step, Alignment cos vs step).
    """
    plt.close("all")
    fig = plt.figure(figsize=(16, 7.5), dpi=200)
    gs = GridSpec(3, 2, width_ratios=[1.15, 1.0], wspace=0.28, hspace=0.38)

    # --------------------------------------------------------------------------
    # Left Panel: 2D Maze + Trajectory
    # --------------------------------------------------------------------------
    ax_maze = fig.add_subplot(gs[:, 0])
    draw_maze(ax_maze, maze_type=maze_type)

    start_x, start_y = summary["start_x"], summary["start_y"]
    target_x, target_y = summary["target_x"], summary["target_y"]

    # Target radius tolerance circle
    target_circle = patches.Circle(
        (target_x, target_y),
        target_radius,
        facecolor="#ffd54f",
        edgecolor="#ff8f00",
        linestyle="--",
        linewidth=1.2,
        alpha=0.35,
        zorder=2,
    )
    ax_maze.add_patch(target_circle)

    # Waypoints chain (if multiple)
    if waypoints and len(waypoints) > 1:
        wps_arr = np.asarray(waypoints)
        ax_maze.plot(
            wps_arr[:, 0],
            wps_arr[:, 1],
            color="#8e24aa",
            linestyle="--",
            linewidth=1.5,
            alpha=0.65,
            zorder=3,
            label="Waypoints",
        )
        ax_maze.scatter(
            wps_arr[:, 0],
            wps_arr[:, 1],
            color="#ab47bc",
            edgecolors="#4a148c",
            s=45,
            marker="D",
            zorder=4,
        )

    # Ant Trajectory
    xs = df_telemetry["x"].values
    ys = df_telemetry["y"].values
    steps = df_telemetry["step"].values

    if len(xs) > 1:
        # Subtle connection line
        ax_maze.plot(xs, ys, color="#424242", linewidth=1.2, alpha=0.4, zorder=4)
        # Step-colored scatter points
        sc = ax_maze.scatter(
            xs,
            ys,
            c=steps,
            cmap="plasma",
            s=18,
            alpha=0.85,
            edgecolors="none",
            zorder=5,
            label="Trajectory ($t=0..T$)",
        )
        cbar = fig.colorbar(sc, ax=ax_maze, orientation="horizontal", pad=0.06, fraction=0.04, aspect=30)
        cbar.set_label("Step $t$", fontsize=9)
        cbar.ax.tick_params(labelsize=8)

    # Subgoal Intention Arrows (sampled every k steps)
    sample_stride = max(1, len(df_telemetry) // 18)
    for idx in range(0, len(df_telemetry), sample_stride):
        row = df_telemetry.iloc[idx]
        px, py = row["x"], row["y"]
        gx, gy = row["subgoal_x"], row["subgoal_y"]
        dx, dy = gx - px, gy - py
        dist_sg = math.hypot(dx, dy)
        if dist_sg > 0.4:
            arrow_len = min(dist_sg, 1.8)
            norm_dx = (dx / dist_sg) * arrow_len
            norm_dy = (dy / dist_sg) * arrow_len
            ax_maze.annotate(
                "",
                xy=(px + norm_dx, py + norm_dy),
                xytext=(px, py),
                arrowprops=dict(
                    arrowstyle="->,head_width=0.25,head_length=0.35",
                    color="#00897b",
                    lw=1.3,
                    alpha=0.75,
                ),
                zorder=6,
            )

    # Start & Target markers
    ax_maze.scatter(
        [start_x],
        [start_y],
        c="#2e7d32",
        s=140,
        marker="o",
        edgecolors="black",
        linewidths=1.5,
        zorder=7,
        label="Start $s_0$",
    )
    ax_maze.scatter(
        [target_x],
        [target_y],
        c="#fbc02d",
        s=220,
        marker="*",
        edgecolors="#d84315",
        linewidths=1.5,
        zorder=7,
        label="Target $g$",
    )

    status_str = "SUCCESS (Goal Reached)" if summary["reached_goal"] else "INCOMPLETE"
    status_color = "#1b5e20" if summary["reached_goal"] else "#b71c1c"
    seed_txt = f" | Seed: {summary['seed']}" if "seed" in summary else ""

    ax_maze.set_title(
        f"Scenario: {summary['scenario_name']}{seed_txt} [Mode: {summary['mode']}]\nStatus: {status_str} | Steps: {summary['num_steps']} | Final Dist: {summary['final_dist_to_goal']:.2f}m",
        fontsize=11,
        fontweight="bold",
        color=status_color,
        pad=10,
    )
    ax_maze.legend(loc="upper left", fontsize=8.5, framealpha=0.92)

    # --------------------------------------------------------------------------
    # Right Top Panel: Distance to Goal & Subgoal vs Step
    # --------------------------------------------------------------------------
    ax_dist = fig.add_subplot(gs[0, 1])
    ax_dist.plot(steps, df_telemetry["dist_to_goal"], color="#d32f2f", linewidth=1.8, label="Dist to Goal")
    ax_dist.plot(
        steps,
        df_telemetry["dist_to_subgoal"],
        color="#1976d2",
        linewidth=1.3,
        linestyle="--",
        alpha=0.75,
        label="Dist to Subgoal",
    )
    ax_dist.axhline(target_radius, color="#388e3c", linestyle=":", linewidth=1.4, label=f"Goal Radius ({target_radius}m)")
    ax_dist.set_ylabel("Distance (m)", fontsize=9, fontweight="bold")
    ax_dist.grid(True, linestyle=":", alpha=0.5)
    ax_dist.set_title("Distance to Goal / Subgoal vs Step", fontsize=10, fontweight="bold")
    ax_dist.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax_dist.tick_params(labelsize=8.5)

    # --------------------------------------------------------------------------
    # Right Middle Panel: Speed vs Step
    # --------------------------------------------------------------------------
    ax_speed = fig.add_subplot(gs[1, 1])
    speeds = df_telemetry["speed"].values
    ax_speed.plot(steps, speeds, color="#0288d1", alpha=0.45, linewidth=1.0, label="Instantaneous $v_t$")
    if len(speeds) > 5:
        rolling_speed = pd.Series(speeds).rolling(window=min(10, len(speeds)), min_periods=1).mean()
        ax_speed.plot(steps, rolling_speed, color="#01579b", linewidth=1.8, label="Rolling Mean (w=10)")
    ax_speed.set_ylabel("Speed (m/step)", fontsize=9, fontweight="bold")
    ax_speed.grid(True, linestyle=":", alpha=0.5)
    ax_speed.set_title(f"Agent Velocity vs Step (Mean: {summary['mean_speed']:.2f} m/step)", fontsize=10, fontweight="bold")
    ax_speed.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax_speed.tick_params(labelsize=8.5)

    # --------------------------------------------------------------------------
    # Right Bottom Panel: Subgoal Vector Alignment (cos theta) vs Step
    # --------------------------------------------------------------------------
    ax_align = fig.add_subplot(gs[2, 1])
    aligns = df_telemetry["alignment_cos"].values
    ax_align.plot(steps, aligns, color="#7b1fa2", alpha=0.45, linewidth=1.0, label=r"Cosine $\cos(\theta_t)$")
    if len(aligns) > 5:
        rolling_align = pd.Series(aligns).rolling(window=min(10, len(aligns)), min_periods=1).mean()
        ax_align.plot(steps, rolling_align, color="#4a148c", linewidth=1.8, label="Rolling Mean (w=10)")
    ax_align.axhline(0.0, color="#757575", linestyle=":", linewidth=1.0)
    ax_align.set_ylim(-1.05, 1.05)
    ax_align.set_xlabel("Step $t$", fontsize=9, fontweight="bold")
    ax_align.set_ylabel(r"$\cos(\theta)$", fontsize=9, fontweight="bold")
    ax_align.grid(True, linestyle=":", alpha=0.5)
    ax_align.set_title(
        f"Intention Vector Alignment vs Step (Mean: {summary['mean_subgoal_alignment']:.2f})",
        fontsize=10,
        fontweight="bold",
    )
    ax_align.legend(loc="lower right", fontsize=8, framealpha=0.9)
    ax_align.tick_params(labelsize=8.5)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=200)
    plt.close(fig)


# ==============================================================================
# Hydra CLI Entrypoint
# ==============================================================================

@hydra.main(version_base=None, config_path="../configs", config_name="experiment")
def main(cfg: DictConfig):
    print("=" * 70)
    print(f"=== Ant Intention & Navigation Experiments on {cfg.env.name} ===")
    print("=" * 70)

    checkpoint_dir = str(cfg.get("checkpoint_dir", "fb-test"))
    split = str(cfg.env.get("split", "medium"))
    eval_temperature = float(cfg.get("eval_temperature", 0.0))
    target_radius = float(cfg.get("target_radius", 1.0))
    terminate_on_goal = bool(cfg.get("terminate_on_goal", True))
    output_dir = str(cfg.get("output_dir", "results/experiments"))
    target_scenario = cfg.get("scenario_name", None)

    # Parse seeds
    if "seeds" in cfg and cfg.seeds is not None:
        try:
            seeds = [int(s) for s in cfg.seeds]
        except (TypeError, ValueError):
            seeds = [int(cfg.seeds)]
    elif "seed" in cfg and cfg.seed is not None:
        seeds = [int(cfg.seed)]
    else:
        seeds = [0]

    os.makedirs(output_dir, exist_ok=True)

    print(f"Seeds: {seeds}")
    print(f"Loading pretrained agent from {checkpoint_dir}/{split} (initial seed {seeds[0]})...")
    agent, env, train_ds, val_ds, fb_cfg = load_pretrained_agent(checkpoint_dir, split, seed=seeds[0])
    print("Agent and environment loaded successfully.")

    # Filter scenarios if specified
    all_scenarios = list(cfg.get("scenarios", []))
    if target_scenario is not None and str(target_scenario).strip().lower() not in ("", "all", "none"):
        selected_scenarios = [s for s in all_scenarios if s.name == target_scenario]
        if not selected_scenarios:
            raise ValueError(
                f"Scenario '{target_scenario}' not found in configuration! Available: {[s.name for s in all_scenarios]}"
            )
    else:
        selected_scenarios = all_scenarios

    total_runs = len(selected_scenarios) * len(seeds)
    print(f"Total scenarios: {len(selected_scenarios)} | Total runs: {total_runs}")
    print("-" * 70)

    summaries = []
    cached_planner = None

    run_idx = 0
    for sc in selected_scenarios:
        for seed in seeds:
            run_idx += 1
            seed_suffix = f"_seed{seed}" if len(seeds) > 1 else ""
            seed_info = f" (seed: {seed})" if len(seeds) > 1 else ""
            print(f"[{run_idx}/{total_runs}] Running scenario '{sc.name}'{seed_info} (mode: {sc.mode})...")
            t0 = time.perf_counter()

            summary, df_telemetry, waypoints, planner_obj = run_scenario(
                agent=agent,
                env=env,
                train_ds=train_ds,
                config=fb_cfg,
                scenario_cfg=sc,
                eval_temperature=eval_temperature,
                target_radius=target_radius,
                terminate_on_goal=terminate_on_goal,
                seed=seed,
                cached_planner=cached_planner,
            )
            if planner_obj is not None:
                cached_planner = planner_obj

            elapsed = time.perf_counter() - t0

            # Save individual telemetry CSV
            telemetry_file = os.path.join(output_dir, f"telemetry_{sc.name}{seed_suffix}.csv")
            df_telemetry.to_csv(telemetry_file, index=False)

            # Generate plot
            plot_file = os.path.join(output_dir, f"{sc.name}{seed_suffix}.png")
            plot_scenario_results(
                summary=summary,
                df_telemetry=df_telemetry,
                waypoints=waypoints,
                output_path=plot_file,
                maze_type=split,
                target_radius=target_radius,
            )
            summary["plot_path"] = plot_file
            summary["telemetry_path"] = telemetry_file
            summaries.append(summary)

            status = "SUCCESS" if summary["reached_goal"] else "INCOMPLETE"
            print(
                f"   -> {status} in {summary['num_steps']} steps ({elapsed:.2f}s). Final dist: {summary['final_dist_to_goal']:.2f}m. Plot: {plot_file}"
            )

    # Save summary CSV
    df_summary = pd.DataFrame(summaries)
    summary_csv_path = os.path.join(output_dir, "summary_experiments.csv")
    df_summary.to_csv(summary_csv_path, index=False)

    print("=" * 70)
    print(f"All experiments completed! Summary saved to {summary_csv_path}")
    print("=" * 70)
    cols = ["scenario_name", "mode", "seed", "num_steps", "reached_goal", "min_dist_to_goal", "final_dist_to_goal"]
    existing_cols = [c for c in cols if c in df_summary.columns]
    print(df_summary[existing_cols].to_string(index=False))
    print("=" * 70)


if __name__ == "__main__":
    main()
