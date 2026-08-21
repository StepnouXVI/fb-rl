#!/usr/bin/env python3
"""
Low-Level Control & Straight-Line Waypoint Execution Experiment in FB-RL

Compares 4 low-level control strategies for Ant in MuJoCo AntMaze:
1. Baseline (high_actor): pi_h(s_t, B(w)) -> pi_l(s_t, z_subgoal)
2. FB Gradient Maximization (Latent Optimization): argmax_z F(s_t, z)^T B(w) via projected gradient ascent on S^{d-1}
3. Heading Direction Alignment Intent: Local lookahead velocity vector projected into FB latent space
4. Heuristic Joint Controller (Naive PD): Direct coordinate error mapping onto joint torques

Evaluates 3 benchmark scenarios:
- straight_corridor (8m straight line)
- sharp_90deg_turn (90-degree corner)
- s_curve_maneuver (Double-turn S-curve)

Computes:
- Success rate (%)
- Mean linear speed (m/s)
- Fall/flip count
- Trajectory Jerk (m/s^3) & Total Curvature (rad)
- Distance & Reachability telemetry
"""

import os
import sys
import math
import time
import json
import numpy as np
import pandas as pd
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import jax
import jax.numpy as jnp
import flax

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.agent_loader import load_pretrained_agent
from scripts.experiment_intention import MAZE_LAYOUTS, draw_maze

# ==============================================================================
# JIT-Compiled Execution Primitives
# ==============================================================================

@jax.jit
def _jit_baseline_step(agent, obs, goal_latent):
    """Fused high_actor + actor execution."""
    obs_b = obs[None, :] if obs.ndim == 1 else obs
    z_b = goal_latent[None, :] if goal_latent.ndim == 1 else goal_latent
    high_dist = agent.network.select("high_actor")(obs_b, z_b, goal_encoded=True, temperature=0.0)
    subgoal_z = agent.normalize_z(high_dist.mode())
    low_dist = agent.network.select("actor")(obs_b, subgoal_z, goal_encoded=True, temperature=0.0)
    return jnp.clip(low_dist.mode()[0], -1.0, 1.0), subgoal_z[0]


@jax.jit
def _jit_direct_actor_step(agent, obs, z_intent):
    """Direct low-level actor execution on a specified intention vector."""
    obs_b = obs[None, :] if obs.ndim == 1 else obs
    z_b = z_intent[None, :] if z_intent.ndim == 1 else z_intent
    z_norm = agent.normalize_z(z_b)
    low_dist = agent.network.select("actor")(obs_b, z_norm, goal_encoded=True, temperature=0.0)
    return jnp.clip(low_dist.mode()[0], -1.0, 1.0), z_norm[0]


@jax.jit
def _jit_opt_z_step(agent, obs, b_goal, z_init, n_steps=25, lr=0.15):
    """
    Projected gradient ascent in latent space S^{d-1} to maximize reachability F(s, z)^T B(w).
    Directly solves z^* = argmax_z F(s, z)^T B(w) online.
    """
    obs_b = obs[None, :] if obs.ndim == 1 else obs
    bg_b = b_goal[None, :] if b_goal.ndim == 1 else b_goal

    def step_fn(i, z):
        def obj_fn(z_cur):
            z_n = agent.normalize_z(z_cur)
            f_rep = agent.network.select("forward_repr")(obs_b, z_n, goal_encoded=True)
            f_mean = jnp.mean(f_rep, axis=0) # ensemble mean (1, 128)
            return jnp.sum(f_mean * bg_b)
        grad_z = jax.grad(obj_fn)(z)
        z_new = z + lr * grad_z
        return agent.normalize_z(z_new)

    z_opt = jax.lax.fori_loop(0, n_steps, step_fn, z_init[None, :] if z_init.ndim == 1 else z_init)
    low_dist = agent.network.select("actor")(obs_b, z_opt, goal_encoded=True, temperature=0.0)
    action = jnp.clip(low_dist.mode()[0], -1.0, 1.0)
    return action, z_opt[0]


@jax.jit
def _jit_compute_reachability(agent, obs, z_intent, b_goal):
    """Computes F(s, z)^T B(w)."""
    obs_b = obs[None, :] if obs.ndim == 1 else obs
    z_b = z_intent[None, :] if z_intent.ndim == 1 else z_intent
    bg_b = b_goal[None, :] if b_goal.ndim == 1 else b_goal
    z_n = agent.normalize_z(z_b)
    f_rep = agent.network.select("forward_repr")(obs_b, z_n, goal_encoded=True)
    f_mean = jnp.mean(f_rep, axis=0)
    return jnp.sum(f_mean * bg_b)


# ==============================================================================
# Controller Implementations
# ==============================================================================

class BaseLowLevelController:
    def __init__(self, name="BaseController"):
        self.name = name

    def reset(self, obs, waypoints):
        pass

    def get_action_and_telemetry(self, obs, active_wp, step=0):
        raise NotImplementedError


class BaselineHighActorController(BaseLowLevelController):
    """Mode 1: Standard high_actor -> low_actor."""
    def __init__(self, agent, ds_obs, ds_coords, kdtree):
        super().__init__(name="baseline_high_actor")
        self.agent = agent
        self.ds_obs = ds_obs
        self.ds_coords = ds_coords
        self.kdtree = kdtree

    def get_action_and_telemetry(self, obs, active_wp_xy, step=0):
        # Encode active waypoint
        wp_idx = self.kdtree.query(active_wp_xy)[1]
        wp_state = self.ds_obs[wp_idx : wp_idx + 1]
        b_wp = self.agent.normalize_z(self.agent.network.select("backward_repr")(jnp.asarray(wp_state)))[0]

        obs_jnp = jnp.asarray(obs)
        action, z_subgoal = _jit_baseline_step(self.agent, obs_jnp, b_wp)
        reach = float(_jit_compute_reachability(self.agent, obs_jnp, z_subgoal, b_wp))
        return np.asarray(action), np.asarray(z_subgoal), b_wp, reach


class FBGradientOptController(BaseLowLevelController):
    """Mode 2: Latent optimization via projected gradient ascent on F(s, z)^T B(w)."""
    def __init__(self, agent, ds_obs, ds_coords, kdtree, n_steps=25, lr=0.15):
        super().__init__(name="fb_latent_optimization")
        self.agent = agent
        self.ds_obs = ds_obs
        self.ds_coords = ds_coords
        self.kdtree = kdtree
        self.n_steps = n_steps
        self.lr = lr

    def get_action_and_telemetry(self, obs, active_wp_xy, step=0):
        wp_idx = self.kdtree.query(active_wp_xy)[1]
        wp_state = self.ds_obs[wp_idx : wp_idx + 1]
        b_wp = self.agent.normalize_z(self.agent.network.select("backward_repr")(jnp.asarray(wp_state)))[0]

        obs_jnp = jnp.asarray(obs)
        # Initialize with high_actor or direct b_wp
        high_dist = self.agent.network.select("high_actor")(obs_jnp[None, :], b_wp[None, :], goal_encoded=True, temperature=0.0)
        z_init = self.agent.normalize_z(high_dist.mode())[0]

        action, z_opt = _jit_opt_z_step(self.agent, obs_jnp, b_wp, z_init, n_steps=self.n_steps, lr=self.lr)
        reach = float(_jit_compute_reachability(self.agent, obs_jnp, z_opt, b_wp))
        return np.asarray(action), np.asarray(z_opt), b_wp, reach


class HeadingAlignmentController(BaseLowLevelController):
    """Mode 3: Pure heading direction alignment projected into FB space."""
    def __init__(self, agent, ds_obs, ds_coords, kdtree, lookahead=2.0):
        super().__init__(name="heading_direction_alignment")
        self.agent = agent
        self.ds_obs = ds_obs
        self.ds_coords = ds_coords
        self.kdtree = kdtree
        self.lookahead = lookahead

    def get_action_and_telemetry(self, obs, active_wp_xy, step=0):
        cur_pos = obs[:2]
        vec = np.asarray(active_wp_xy) - np.asarray(cur_pos)
        dist = np.linalg.norm(vec)
        if dist < 1e-4:
            cand_xy = cur_pos
        else:
            cand_xy = cur_pos + (vec / dist) * min(dist, self.lookahead)

        cand_idx = self.kdtree.query(cand_xy)[1]
        cand_state = self.ds_obs[cand_idx : cand_idx + 1]
        z_heading = self.agent.normalize_z(self.agent.network.select("backward_repr")(jnp.asarray(cand_state)))[0]

        # Target backward repr for reachability logging
        wp_idx = self.kdtree.query(active_wp_xy)[1]
        b_wp = self.agent.normalize_z(self.agent.network.select("backward_repr")(jnp.asarray(self.ds_obs[wp_idx : wp_idx + 1])))[0]

        obs_jnp = jnp.asarray(obs)
        action, z_norm = _jit_direct_actor_step(self.agent, obs_jnp, z_heading)
        reach = float(_jit_compute_reachability(self.agent, obs_jnp, z_norm, b_wp))
        return np.asarray(action), np.asarray(z_norm), b_wp, reach


class HeuristicJointPDController(BaseLowLevelController):
    """
    Mode 4: Naive coordinate / Joint PD controller.
    Attempts to drive torso towards waypoint by applying heading-proportional joint torques.
    Demonstrates physical impossibility of walking without learned gait coordination.
    """
    def __init__(self, kp=1.5, kd=0.2):
        super().__init__(name="heuristic_joint_controller")
        self.kp = kp
        self.kd = kd

    def get_action_and_telemetry(self, obs, active_wp_xy, step=0):
        cur_pos = obs[:2]
        err = np.asarray(active_wp_xy) - cur_pos
        des_heading = np.arctan2(err[1], err[0])

        # Body orientation yaw from quaternion
        qw, qx, qy, qz = obs[3:7]
        yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
        heading_err = np.arctan2(np.sin(des_heading - yaw), np.cos(des_heading - yaw))

        # Joint angular velocities for D-gain
        qvel_joints = obs[21:29]

        # Naive joint torque command
        action = np.zeros(8, dtype=np.float32)
        # Steering torques on hip yaw joints (joints 0, 2, 4, 6)
        action[0::2] = np.clip(self.kp * heading_err - self.kd * qvel_joints[0::2], -1.0, 1.0)
        # Forward thrust torques on ankle pitch joints (joints 1, 3, 5, 7)
        action[1::2] = np.clip(0.8 - self.kd * qvel_joints[1::2], -1.0, 1.0)

        # Zero latent representation (no RL agent used)
        dummy_z = np.zeros(128, dtype=np.float32)
        return action, dummy_z, dummy_z, 0.0


# ==============================================================================
# Scenario Definitions
# ==============================================================================

SCENARIOS = {
    "straight_corridor": {
        "description": "8-meter straight hallway along bottom corridor",
        "start_xy": [0.0, 20.0],
        "target_xy": [8.0, 20.0],
        "waypoints": [[4.0, 20.0], [8.0, 20.0]],
        "max_steps": 140,
        "waypoint_radius": 1.2,
        "target_radius": 1.0,
    },
    "sharp_90deg_turn": {
        "description": "Sharp 90-degree right-angle turn around corner",
        "start_xy": [0.0, 0.0],
        "target_xy": [4.0, 8.0],
        "waypoints": [[4.0, 0.0], [4.0, 8.0]],
        "max_steps": 180,
        "waypoint_radius": 1.2,
        "target_radius": 1.0,
    },
    "s_curve_maneuver": {
        "description": "Multi-corner S-curve path with double 90-degree turns",
        "start_xy": [0.0, 0.0],
        "target_xy": [12.0, 4.0],
        "waypoints": [[4.0, 0.0], [4.0, 8.0], [8.0, 8.0], [12.0, 8.0], [12.0, 4.0]],
        "max_steps": 320,
        "waypoint_radius": 1.2,
        "target_radius": 1.0,
    },
}


# ==============================================================================
# Simulation & Telemetry Engine
# ==============================================================================

def run_single_simulation(
    controller,
    env,
    scenario_name,
    scenario_cfg,
    seed=0,
    dt=0.05,
):
    """
    Runs a single simulation episode with detailed metric calculation:
    - Step displacement & linear speed (m/s)
    - Fall / flip detection
    - Trajectory Jerk (m/s^3)
    - Angular Curvature (rad)
    - Latent Reachability
    """
    start_xy = list(scenario_cfg["start_xy"])
    target_xy = list(scenario_cfg["target_xy"])
    waypoints = [list(wp) for wp in scenario_cfg["waypoints"]]
    max_steps = int(scenario_cfg["max_steps"])
    wp_radius = float(scenario_cfg["waypoint_radius"])
    target_radius = float(scenario_cfg["target_radius"])

    # Reset environment
    env.action_space.seed(seed)
    np.random.seed(seed)
    env.reset(seed=seed)

    if hasattr(env.unwrapped, "set_goal"):
        env.unwrapped.set_goal(goal_xy=np.asarray(target_xy, dtype=np.float32))

    env.unwrapped.set_xy(np.asarray(start_xy, dtype=np.float32))
    obs = env.unwrapped.get_ob()

    controller.reset(obs, waypoints)

    wp_idx = 0
    records = []
    positions = [obs[:2].copy()]
    speeds = [0.0]
    heights = [obs[2]]
    flips = 0

    cur_pos = obs[:2]
    d_target = float(np.linalg.norm(cur_pos - np.asarray(target_xy)))

    records.append({
        "scenario": scenario_name,
        "mode": controller.name,
        "seed": seed,
        "step": 0,
        "x": float(cur_pos[0]),
        "y": float(cur_pos[1]),
        "z": float(obs[2]),
        "speed_mps": 0.0,
        "wp_idx": wp_idx,
        "active_wp_x": float(waypoints[wp_idx][0]),
        "active_wp_y": float(waypoints[wp_idx][1]),
        "dist_to_wp": float(np.linalg.norm(cur_pos - np.asarray(waypoints[wp_idx]))),
        "dist_to_target": d_target,
        "reachability": 0.0,
        "action_norm": 0.0,
        "is_flipped": False,
        "reached_goal": bool(d_target <= target_radius),
    })

    goal_ever_reached = bool(d_target <= target_radius)
    step = 0
    done = False

    while step < max_steps and not done:
        active_wp = waypoints[wp_idx]

        # Waypoint progress switching
        d_cur_wp = float(np.linalg.norm(cur_pos - np.asarray(active_wp)))
        if d_cur_wp <= wp_radius and wp_idx < len(waypoints) - 1:
            wp_idx += 1
            active_wp = waypoints[wp_idx]

        prev_pos = cur_pos.copy()

        # Controller action
        action, z_intent, b_goal, reach = controller.get_action_and_telemetry(obs, active_wp, step=step)

        # Environment step
        next_obs, reward, term, trunc, info = env.step(action)
        step += 1

        cur_pos = next_obs[:2]
        obs = next_obs
        positions.append(cur_pos.copy())
        heights.append(obs[2])

        disp = cur_pos - prev_pos
        step_speed = float(np.linalg.norm(disp)) / dt # m/s
        speeds.append(step_speed)

        # Fall / Flip detection: torso height < 0.28 or > 1.05 or pitch/roll excessive
        qx, qy = obs[4], obs[5]
        is_flip = bool(obs[2] < 0.28 or obs[2] > 1.05 or abs(qx) > 0.45 or abs(qy) > 0.45)
        if is_flip:
            flips += 1

        d_target = float(np.linalg.norm(cur_pos - np.asarray(target_xy)))
        reached_now = bool(d_target <= target_radius)
        if reached_now:
            goal_ever_reached = True

        if reached_now or term or trunc or step >= max_steps:
            done = True

        records.append({
            "scenario": scenario_name,
            "mode": controller.name,
            "seed": seed,
            "step": step,
            "x": float(cur_pos[0]),
            "y": float(cur_pos[1]),
            "z": float(obs[2]),
            "speed_mps": step_speed,
            "wp_idx": wp_idx,
            "active_wp_x": float(active_wp[0]),
            "active_wp_y": float(active_wp[1]),
            "dist_to_wp": float(np.linalg.norm(cur_pos - np.asarray(active_wp))),
            "dist_to_target": d_target,
            "reachability": float(reach),
            "action_norm": float(np.linalg.norm(action)),
            "is_flipped": is_flip,
            "reached_goal": reached_now,
        })

    df_telemetry = pd.DataFrame(records)
    pos_arr = np.asarray(positions) # (T+1, 2)

    # 1. Total Path Length
    if len(pos_arr) > 1:
        step_diffs = np.diff(pos_arr, axis=0)
        path_length = float(np.sum(np.linalg.norm(step_diffs, axis=1)))
    else:
        path_length = 0.0

    # 2. RMS Jerk: third derivative of position (m/s^3)
    if len(pos_arr) > 3:
        velocities = np.diff(pos_arr, axis=0) / dt # (T, 2)
        accelerations = np.diff(velocities, axis=0) / dt # (T-1, 2)
        jerks = np.diff(accelerations, axis=0) / dt # (T-2, 2)
        jerk_mags = np.linalg.norm(jerks, axis=1)
        rms_jerk = float(np.sqrt(np.mean(jerk_mags ** 2)))
    else:
        rms_jerk = 0.0

    # 3. Total Curvature (Sum of directional turning angles in radians)
    if len(pos_arr) > 2:
        v = np.diff(pos_arr, axis=0)
        v_norms = np.linalg.norm(v, axis=1)
        valid_mask = v_norms > 1e-3
        angles = np.arctan2(v[valid_mask, 1], v[valid_mask, 0])
        if len(angles) > 1:
            angle_diffs = np.diff(angles)
            # Wrap to [-pi, pi]
            angle_diffs = (angle_diffs + np.pi) % (2 * np.pi) - np.pi
            total_curvature = float(np.sum(np.abs(angle_diffs)))
        else:
            total_curvature = 0.0
    else:
        total_curvature = 0.0

    summary = {
        "scenario": scenario_name,
        "mode": controller.name,
        "seed": seed,
        "num_steps": step,
        "reached_goal": goal_ever_reached,
        "min_dist_to_target": float(df_telemetry["dist_to_target"].min()),
        "final_dist_to_target": float(df_telemetry["dist_to_target"].iloc[-1]),
        "mean_speed_mps": float(df_telemetry["speed_mps"].mean()),
        "max_speed_mps": float(df_telemetry["speed_mps"].max()),
        "flip_steps": int(flips),
        "ever_flipped": bool(flips > 0),
        "rms_jerk_mps3": rms_jerk,
        "total_curvature_rad": total_curvature,
        "path_length_m": path_length,
        "mean_reachability": float(df_telemetry["reachability"].mean()),
    }

    return summary, df_telemetry


# ==============================================================================
# Visualization Engine
# ==============================================================================

MODE_COLORS = {
    "baseline_high_actor": "#1976d2",          # Blue
    "fb_latent_optimization": "#388e3c",       # Green
    "heading_direction_alignment": "#7b1fa2",  # Purple
    "heuristic_joint_controller": "#d32f2f",   # Red
}

MODE_LABELS = {
    "baseline_high_actor": r"Baseline $\pi^h \to \pi^\ell$",
    "fb_latent_optimization": r"FB Latent Gradient Opt ($\nabla_z F^\top B$)",
    "heading_direction_alignment": r"Heading Direction Alignment",
    "heuristic_joint_controller": r"Naive Joint PD Controller",
}


def plot_scenario_comparison(scenario_name, scenario_cfg, mode_telemetries, output_path):
    """
    Creates a multi-panel publication plot comparing all 4 controllers on a scenario:
    - Main Panel: 2D Maze with full trajectories overlaid
    - Panel 2: Distance to Target vs Step
    - Panel 3: Linear Speed (m/s) vs Step
    - Panel 4: Torso Height z (Stability) vs Step
    """
    plt.close("all")
    fig = plt.figure(figsize=(18, 8.5), dpi=250)
    gs = GridSpec(3, 2, width_ratios=[1.25, 1.0], wspace=0.28, hspace=0.38)

    # --------------------------------------------------------------------------
    # Left Panel: 2D Maze + Trajectories
    # --------------------------------------------------------------------------
    ax_maze = fig.add_subplot(gs[:, 0])
    draw_maze(ax_maze, maze_type="medium")

    start_xy = scenario_cfg["start_xy"]
    target_xy = scenario_cfg["target_xy"]
    waypoints = scenario_cfg["waypoints"]

    # Target tolerance zone
    target_circle = patches.Circle(
        (target_xy[0], target_xy[1]),
        scenario_cfg["target_radius"],
        facecolor="#ffd54f",
        edgecolor="#ff8f00",
        linestyle="--",
        linewidth=1.4,
        alpha=0.35,
        zorder=2,
    )
    ax_maze.add_patch(target_circle)

    # Waypoints chain
    wps_arr = np.asarray([[start_xy[0], start_xy[1]]] + waypoints)
    ax_maze.plot(
        wps_arr[:, 0], wps_arr[:, 1],
        color="#757575", linestyle=":", linewidth=1.5, alpha=0.6, zorder=3, label="Waypoint Path"
    )
    ax_maze.scatter(
        wps_arr[1:-1, 0], wps_arr[1:-1, 1],
        color="#ab47bc", edgecolors="#4a148c", s=70, marker="D", zorder=4, label="Intermediate WPs"
    )

    # Plot trajectories for each mode (Seed 0)
    for mode_name, df_tel in mode_telemetries.items():
        color = MODE_COLORS.get(mode_name, "#424242")
        label = MODE_LABELS.get(mode_name, mode_name)
        xs, ys = df_tel["x"].values, df_tel["y"].values

        # Trajectory line
        ax_maze.plot(xs, ys, color=color, linewidth=2.0, alpha=0.85, zorder=5, label=label)
        # End position marker
        ax_maze.scatter(
            [xs[-1]], [ys[-1]],
            color=color, edgecolors="black", s=50, marker="X", zorder=6
        )

    # Start & Target
    ax_maze.scatter([start_xy[0]], [start_xy[1]], c="#2e7d32", s=150, marker="o", edgecolors="black", linewidths=1.5, zorder=7, label="Start $s_0$")
    ax_maze.scatter([target_xy[0]], [target_xy[1]], c="#fbc02d", s=220, marker="*", edgecolors="#d84315", linewidths=1.5, zorder=7, label="Target $g$")

    ax_maze.set_title(
        f"Trajectory Comparison: {scenario_name}\n({scenario_cfg['description']})",
        fontsize=12, fontweight="bold", pad=10
    )
    ax_maze.legend(loc="upper left", fontsize=8.5, framealpha=0.92)

    # --------------------------------------------------------------------------
    # Right Top Panel: Distance to Target vs Step
    # --------------------------------------------------------------------------
    ax_dist = fig.add_subplot(gs[0, 1])
    for mode_name, df_tel in mode_telemetries.items():
        color = MODE_COLORS.get(mode_name, "#424242")
        label = MODE_LABELS.get(mode_name, mode_name)
        ax_dist.plot(df_tel["step"], df_tel["dist_to_target"], color=color, linewidth=1.8, label=label)

    ax_dist.axhline(scenario_cfg["target_radius"], color="#ff8f00", linestyle="--", linewidth=1.2, label=f"Goal Radius ({scenario_cfg['target_radius']}m)")
    ax_dist.set_ylabel("Dist to Goal (m)", fontsize=9, fontweight="bold")
    ax_dist.grid(True, linestyle=":", alpha=0.5)
    ax_dist.set_title("Distance to Target vs Step", fontsize=10, fontweight="bold")
    ax_dist.tick_params(labelsize=8.5)
    ax_dist.legend(loc="upper right", fontsize=7.5, framealpha=0.9)

    # --------------------------------------------------------------------------
    # Right Middle Panel: Speed (m/s) vs Step
    # --------------------------------------------------------------------------
    ax_speed = fig.add_subplot(gs[1, 1])
    for mode_name, df_tel in mode_telemetries.items():
        color = MODE_COLORS.get(mode_name, "#424242")
        speeds = df_tel["speed_mps"].values
        rolling_speed = pd.Series(speeds).rolling(window=min(8, len(speeds)), min_periods=1).mean()
        ax_speed.plot(df_tel["step"], rolling_speed, color=color, linewidth=1.8, label=MODE_LABELS.get(mode_name, mode_name))

    ax_speed.set_ylabel("Speed (m/s)", fontsize=9, fontweight="bold")
    ax_speed.grid(True, linestyle=":", alpha=0.5)
    ax_speed.set_title("Linear Velocity vs Step (Rolling Mean w=8)", fontsize=10, fontweight="bold")
    ax_speed.tick_params(labelsize=8.5)

    # --------------------------------------------------------------------------
    # Right Bottom Panel: Torso Height z (Stability) vs Step
    # --------------------------------------------------------------------------
    ax_height = fig.add_subplot(gs[2, 1])
    for mode_name, df_tel in mode_telemetries.items():
        color = MODE_COLORS.get(mode_name, "#424242")
        ax_height.plot(df_tel["step"], df_tel["z"], color=color, linewidth=1.6, label=MODE_LABELS.get(mode_name, mode_name))

    ax_height.axhspan(0.28, 0.95, color="#81c784", alpha=0.15, label="Stable Torso Band")
    ax_height.axhline(0.28, color="#e53935", linestyle=":", linewidth=1.0)
    ax_height.set_xlabel("Step $t$", fontsize=9, fontweight="bold")
    ax_height.set_ylabel("Torso Height $z$ (m)", fontsize=9, fontweight="bold")
    ax_height.grid(True, linestyle=":", alpha=0.5)
    ax_height.set_title("Dynamic Balance & Stability (Torso Height vs Step)", fontsize=10, fontweight="bold")
    ax_height.tick_params(labelsize=8.5)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=250)
    plt.close(fig)


def plot_summary_bar_charts(df_summary, output_path):
    """
    Generates 4-panel statistical comparison bar charts across all scenarios and modes:
    - Success Rate (%)
    - Mean Linear Speed (m/s)
    - Trajectory Jerk (m/s^3)
    - Total Curvature (rad)
    """
    plt.close("all")
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), dpi=250)
    plt.subplots_adjust(wspace=0.25, hspace=0.35)

    scenarios = list(SCENARIOS.keys())
    modes = list(MODE_COLORS.keys())
    n_scenarios = len(scenarios)
    n_modes = len(modes)

    x = np.arange(n_scenarios)
    width = 0.18

    # 1. Success Rate (%)
    ax = axes[0, 0]
    for i, mode in enumerate(modes):
        sub = df_summary[df_summary["mode"] == mode]
        vals = []
        for sc in scenarios:
            sc_sub = sub[sub["scenario"] == sc]
            sr = (sc_sub["reached_goal"].mean() * 100.0) if len(sc_sub) > 0 else 0.0
            vals.append(sr)
        rects = ax.bar(x + i * width, vals, width, label=MODE_LABELS[mode], color=MODE_COLORS[mode], alpha=0.85, edgecolor="black", linewidth=0.8)
        for r in rects:
            h = r.get_height()
            ax.annotate(f"{h:.0f}%", xy=(r.get_x() + r.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

    ax.set_xticks(x + width * (n_modes - 1) / 2)
    ax.set_xticklabels([s.replace("_", " ").title() for s in scenarios], fontsize=9, fontweight="bold")
    ax.set_ylabel("Success Rate (%)", fontsize=10, fontweight="bold")
    ax.set_title("Waypoint Reach Success Rate (%)", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 115)
    ax.grid(True, linestyle=":", alpha=0.5, axis="y")
    ax.legend(loc="upper right", fontsize=8)

    # 2. Mean Linear Speed (m/s)
    ax = axes[0, 1]
    for i, mode in enumerate(modes):
        sub = df_summary[df_summary["mode"] == mode]
        means = []
        stds = []
        for sc in scenarios:
            sc_sub = sub[sub["scenario"] == sc]
            means.append(sc_sub["mean_speed_mps"].mean() if len(sc_sub) > 0 else 0.0)
            stds.append(sc_sub["mean_speed_mps"].std() if len(sc_sub) > 1 else 0.0)
        rects = ax.bar(x + i * width, means, width, yerr=stds, capsize=3, label=MODE_LABELS[mode], color=MODE_COLORS[mode], alpha=0.85, edgecolor="black", linewidth=0.8)
        for r in rects:
            h = r.get_height()
            ax.annotate(f"{h:.2f}", xy=(r.get_x() + r.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

    ax.set_xticks(x + width * (n_modes - 1) / 2)
    ax.set_xticklabels([s.replace("_", " ").title() for s in scenarios], fontsize=9, fontweight="bold")
    ax.set_ylabel("Speed (m/s)", fontsize=10, fontweight="bold")
    ax.set_title("Mean Linear Locomotion Speed (m/s)", fontsize=11, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.5, axis="y")

    # 3. RMS Jerk (m/s^3) [Smoothness]
    ax = axes[1, 0]
    for i, mode in enumerate(modes):
        sub = df_summary[df_summary["mode"] == mode]
        means = []
        for sc in scenarios:
            sc_sub = sub[sub["scenario"] == sc]
            means.append(sc_sub["rms_jerk_mps3"].mean() if len(sc_sub) > 0 else 0.0)
        rects = ax.bar(x + i * width, means, width, label=MODE_LABELS[mode], color=MODE_COLORS[mode], alpha=0.85, edgecolor="black", linewidth=0.8)

    ax.set_xticks(x + width * (n_modes - 1) / 2)
    ax.set_xticklabels([s.replace("_", " ").title() for s in scenarios], fontsize=9, fontweight="bold")
    ax.set_ylabel(r"RMS Jerk ($\mathrm{m/s^3}$)", fontsize=10, fontweight="bold")
    ax.set_title(r"Trajectory Smoothness / Jerk Metric ($\downarrow$ lower is smoother)", fontsize=11, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.5, axis="y")

    # 4. Total Curvature (rad) [Directness]
    ax = axes[1, 1]
    for i, mode in enumerate(modes):
        sub = df_summary[df_summary["mode"] == mode]
        means = []
        for sc in scenarios:
            sc_sub = sub[sub["scenario"] == sc]
            means.append(sc_sub["total_curvature_rad"].mean() if len(sc_sub) > 0 else 0.0)
        rects = ax.bar(x + i * width, means, width, label=MODE_LABELS[mode], color=MODE_COLORS[mode], alpha=0.85, edgecolor="black", linewidth=0.8)

    ax.set_xticks(x + width * (n_modes - 1) / 2)
    ax.set_xticklabels([s.replace("_", " ").title() for s in scenarios], fontsize=9, fontweight="bold")
    ax.set_ylabel("Total Curvature (rad)", fontsize=10, fontweight="bold")
    ax.set_title(r"Trajectory Total Angular Curvature ($\downarrow$ more direct)", fontsize=11, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.5, axis="y")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=250)
    plt.close(fig)


# ==============================================================================
# Main Benchmark Execution Engine
# ==============================================================================

def main():
    print("=" * 80)
    print("=== Low-Level Control & Straight-Line Waypoint Execution Benchmark ===")
    print("=" * 80)

    checkpoint_dir = "fb-test"
    split = "medium"
    seeds = [0, 1, 2, 3, 4]
    output_dir = "results/experiments/lowlevel_comparison"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading pretrained FB-RL agent from {checkpoint_dir}/{split}...")
    agent, env, train_ds, val_ds, config = load_pretrained_agent(checkpoint_dir, split, seed=0)
    ds_obs = train_ds["observations"]
    ds_coords = ds_obs[:, :2]

    print(f"Building KDTree for {len(ds_coords)} offline states...")
    t0 = time.perf_counter()
    kdtree = KDTree(ds_coords)
    print(f"KDTree built in {(time.perf_counter() - t0):.2f}s.")

    # Initialize controllers
    controllers = {
        "baseline_high_actor": BaselineHighActorController(agent, ds_obs, ds_coords, kdtree),
        "fb_latent_optimization": FBGradientOptController(agent, ds_obs, ds_coords, kdtree, n_steps=25, lr=0.15),
        "heading_direction_alignment": HeadingAlignmentController(agent, ds_obs, ds_coords, kdtree, lookahead=2.0),
        "heuristic_joint_controller": HeuristicJointPDController(kp=1.5, kd=0.2),
    }

    all_summaries = []
    telemetry_by_scenario = {sc: {} for sc in SCENARIOS}

    total_runs = len(SCENARIOS) * len(controllers) * len(seeds)
    run_idx = 0

    print(f"\nStarting benchmark across {len(SCENARIOS)} scenarios, {len(controllers)} modes, {len(seeds)} seeds (Total: {total_runs} episodes)...")
    print("-" * 80)

    for sc_name, sc_cfg in SCENARIOS.items():
        print(f"\n>>> Scenario: [{sc_name}] ({sc_cfg['description']})")

        for mode_name, ctrl in controllers.items():
            for seed in seeds:
                run_idx += 1
                t_start = time.perf_counter()

                summary, df_tel = run_single_simulation(
                    controller=ctrl,
                    env=env,
                    scenario_name=sc_name,
                    scenario_cfg=sc_cfg,
                    seed=seed,
                )
                elapsed = time.perf_counter() - t_start

                all_summaries.append(summary)

                # Save seed 0 telemetry for detailed scenario plotting
                if seed == 0:
                    telemetry_by_scenario[sc_name][mode_name] = df_tel

                # Save individual telemetry file
                tel_path = os.path.join(output_dir, f"telemetry_{sc_name}_{mode_name}_seed{seed}.csv")
                df_tel.to_csv(tel_path, index=False)

                status_str = "SUCCESS" if summary["reached_goal"] else "FAIL"
                print(
                    f"[{run_idx:02d}/{total_runs:02d}] {sc_name:18s} | {mode_name:28s} | Seed {seed} -> "
                    f"{status_str:7s} in {summary['num_steps']:3d} steps | Speed: {summary['mean_speed_mps']:.2f} m/s | "
                    f"Flips: {summary['flip_steps']:2d} | Jerk: {summary['rms_jerk_mps3']:.1f} | Final dist: {summary['final_dist_to_target']:.2f}m ({elapsed:.2f}s)"
                )

    # Save summary dataframe
    df_summary = pd.DataFrame(all_summaries)
    summary_csv = os.path.join(output_dir, "summary_lowlevel_control.csv")
    df_summary.to_csv(summary_csv, index=False)
    print(f"\nSummary CSV saved to {summary_csv}")

    # Generate scenario comparison plots
    print("\nGenerating high-resolution scenario comparison plots...")
    for sc_name, sc_cfg in SCENARIOS.items():
        plot_path = os.path.join(output_dir, f"comparison_{sc_name}.png")
        plot_scenario_comparison(sc_name, sc_cfg, telemetry_by_scenario[sc_name], plot_path)
        print(f" -> Saved {plot_path}")

    # Generate aggregate bar charts
    bar_chart_path = os.path.join(output_dir, "aggregate_metrics_comparison.png")
    plot_summary_bar_charts(df_summary, bar_chart_path)
    print(f" -> Saved {bar_chart_path}")

    # Print markdown table grouped by scenario and mode
    print("\n" + "=" * 90)
    print("=== AGGREGATE BENCHMARK RESULTS (Mean +/- Std over 5 seeds) ===")
    print("=" * 90)

    agg_rows = []
    for sc in SCENARIOS:
        for mode in controllers:
            sub = df_summary[(df_summary["scenario"] == sc) & (df_summary["mode"] == mode)]
            agg_rows.append({
                "Scenario": sc,
                "Mode": mode,
                "Success (%)": f"{sub['reached_goal'].mean() * 100.0:.1f}%",
                "Speed (m/s)": f"{sub['mean_speed_mps'].mean():.2f} +/- {sub['mean_speed_mps'].std():.2f}",
                "Steps": f"{sub['num_steps'].mean():.1f}",
                "Flips": f"{sub['flip_steps'].mean():.1f}",
                "RMS Jerk": f"{sub['rms_jerk_mps3'].mean():.1f}",
                "Curvature (rad)": f"{sub['total_curvature_rad'].mean():.2f}",
                "Final Dist (m)": f"{sub['final_dist_to_target'].mean():.2f}",
            })

    df_agg = pd.DataFrame(agg_rows)
    print(df_agg.to_string(index=False))
    print("=" * 90)

    # Also save aggregated table as JSON and LaTeX snippet
    agg_json = os.path.join(output_dir, "aggregated_table.json")
    df_agg.to_json(agg_json, orient="records", indent=2)

    latex_table_path = os.path.join(output_dir, "table_lowlevel_comparison.tex")
    with open(latex_table_path, "w") as f:
        f.write("% Low-Level Control Strategy Comparison on AntMaze\n")
        f.write("\\begin{table*}[t]\n\\centering\\small\n")
        f.write("\\caption{Comparison of Low-Level Execution Strategies on Ant Navigation Tasks.}\n")
        f.write("\\label{tab:lowlevel_comparison}\n")
        f.write("\\begin{tabular}{llccccccc}\n\\toprule\n")
        f.write("\\textbf{Scenario} & \\textbf{Control Mode} & \\textbf{Success (\\%)} & \\textbf{Speed (m/s)} & \\textbf{Steps} & \\textbf{Flips} & \\textbf{Jerk (m/s$^3$)} & \\textbf{Curvature (rad)} & \\textbf{Final Dist (m)} \\\\\n\\midrule\n")
        prev_sc = None
        for r in agg_rows:
            if prev_sc is not None and r["Scenario"] != prev_sc:
                f.write("\\midrule\n")
            prev_sc = r["Scenario"]
            mode_tex = r["Mode"].replace("_", "\\_")
            sc_tex = r["Scenario"].replace("_", "\\_")
            f.write(f"{sc_tex} & {mode_tex} & {r['Success (%)']} & {r['Speed (m/s)']} & {r['Steps']} & {r['Flips']} & {r['RMS Jerk']} & {r['Curvature (rad)']} & {r['Final Dist (m)']} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table*}\n")

    print(f"LaTeX table exported to {latex_table_path}")


if __name__ == "__main__":
    main()
