import os
import sys
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
from scipy.spatial import KDTree
import jax
import jax.numpy as jnp

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.agent_loader import load_pretrained_agent
from scripts.visualize_trajectories import draw_maze

SCENARIOS = {
    "straight_corridor": {
        "start_xy": [0.0, 0.0], "target_xy": [12.0, 0.0], "waypoints": [[4.0, 0.0], [8.0, 0.0], [12.0, 0.0]],
        "max_steps": 350, "waypoint_radius": 1.2, "target_radius": 1.0, "description": "Long straight hallway navigation",
    },
    "sharp_90deg_turn": {
        "start_xy": [0.0, 0.0], "target_xy": [8.0, 8.0], "waypoints": [[8.0, 0.0], [8.0, 8.0]],
        "max_steps": 450, "waypoint_radius": 1.2, "target_radius": 1.0, "description": "Right angle corner turn",
    },
    "s_curve_maneuver": {
        "start_xy": [0.0, 0.0], "target_xy": [16.0, 8.0], "waypoints": [[4.0, 0.0], [8.0, 4.0], [12.0, 8.0], [16.0, 8.0]],
        "max_steps": 550, "waypoint_radius": 1.2, "target_radius": 1.0, "description": "Multi-waypoint S-curve",
    },
}

MODE_COLORS = {"baseline_high_actor": "#1976d2", "fb_latent_optimization": "#d32f2f", "heading_direction_alignment": "#388e3c", "heuristic_joint_controller": "#f57c00"}
MODE_LABELS = {"baseline_high_actor": "Baseline High-Actor", "fb_latent_optimization": "FB Latent Optimization", "heading_direction_alignment": "Heading Dir Alignment", "heuristic_joint_controller": "Naive Coordinate PD"}


class BaseLowLevelController:
    def __init__(self, name):
        self.name = name
    def reset(self, obs, waypoints):
        pass
    def get_action_and_telemetry(self, obs, active_wp, step=0):
        raise NotImplementedError


class BaselineHighActorController(BaseLowLevelController):
    def __init__(self, agent, dataset_obs, dataset_coords, kdtree):
        super().__init__("baseline_high_actor")
        self.agent, self.dataset_obs, self.dataset_coords, self.kdtree = agent, dataset_obs, dataset_coords, kdtree

    def get_action_and_telemetry(self, obs, active_wp, step=0):
        _, idx = self.kdtree.query(active_wp, k=1)
        w_state = self.dataset_obs[idx:idx+1]
        b_goal = np.asarray(self.agent.normalize_z(self.agent.network.select("backward_repr")(jnp.asarray(w_state)))[0])
        high_dist = self.agent.network.select("high_actor")(obs[None, :], b_goal[None, :], goal_encoded=True, temperature=0.0)
        z_cmd = np.asarray(self.agent.normalize_z(high_dist.mode())[0])
        low_dist = self.agent.network.select("actor")(obs[None, :], z_cmd[None, :], goal_encoded=True, temperature=0.0)
        f_repr = np.asarray(self.agent.network.select("forward_repr")(obs[None, :], z_cmd[None, :], goal_encoded=True)[0])
        return np.clip(np.asarray(low_dist.mode()[0]), -1.0, 1.0), z_cmd, b_goal, float(np.sum(f_repr * b_goal))


class FBGradientOptController(BaseLowLevelController):
    def __init__(self, agent, dataset_obs, dataset_coords, kdtree, n_steps=20, lr=0.15):
        super().__init__("fb_latent_optimization")
        self.agent, self.dataset_obs, self.dataset_coords, self.kdtree = agent, dataset_obs, dataset_coords, kdtree
        self.n_steps, self.lr = n_steps, lr

        @jax.jit
        def _opt_z(s, b_target, z_init):
            def loss_fn(z):
                z_norm = z / (jnp.linalg.norm(z) + 1e-8)
                f = agent.network.select("forward_repr")(s[None, :], z_norm[None, :], goal_encoded=True)[0]
                if f.ndim == 2:
                    f = jnp.mean(f, axis=0)
                return -jnp.sum(f * b_target)
            grad_fn = jax.grad(loss_fn)
            z = z_init
            for _ in range(n_steps):
                z = z - lr * grad_fn(z)
                z = z / (jnp.linalg.norm(z) + 1e-8)
            return z
        self._opt_z_fn = _opt_z

    def get_action_and_telemetry(self, obs, active_wp, step=0):
        _, idx = self.kdtree.query(active_wp, k=1)
        b_goal = np.asarray(self.agent.normalize_z(self.agent.network.select("backward_repr")(jnp.asarray(self.dataset_obs[idx:idx+1])))[0])
        high_dist = self.agent.network.select("high_actor")(obs[None, :], b_goal[None, :], goal_encoded=True, temperature=0.0)
        z_init = np.asarray(self.agent.normalize_z(high_dist.mode())[0])
        z_opt = np.asarray(self._opt_z_fn(jnp.asarray(obs), jnp.asarray(b_goal), jnp.asarray(z_init)))
        low_dist = self.agent.network.select("actor")(obs[None, :], z_opt[None, :], goal_encoded=True, temperature=0.0)
        f_repr = np.asarray(self.agent.network.select("forward_repr")(obs[None, :], z_opt[None, :], goal_encoded=True)[0])
        return np.clip(np.asarray(low_dist.mode()[0]), -1.0, 1.0), z_opt, b_goal, float(np.sum(f_repr * b_goal))


class HeadingAlignmentController(BaseLowLevelController):
    def __init__(self, agent, dataset_obs, dataset_coords, kdtree, lookahead=2.0):
        super().__init__("heading_direction_alignment")
        self.agent, self.dataset_obs, self.dataset_coords, self.kdtree = agent, dataset_obs, dataset_coords, kdtree
        self.lookahead = lookahead

    def get_action_and_telemetry(self, obs, active_wp, step=0):
        cur_pos = obs[:2]
        v_des = np.asarray(active_wp) - cur_pos
        v_norm = np.linalg.norm(v_des)
        unit_v = (v_des / v_norm) if v_norm > 1e-4 else np.array([1.0, 0.0])
        virtual_target = cur_pos + unit_v * self.lookahead
        _, idx = self.kdtree.query(virtual_target, k=1)
        z_cmd = np.asarray(self.agent.normalize_z(self.agent.network.select("backward_repr")(jnp.asarray(self.dataset_obs[idx:idx+1])))[0])
        low_dist = self.agent.network.select("actor")(obs[None, :], z_cmd[None, :], goal_encoded=True, temperature=0.0)
        return np.clip(np.asarray(low_dist.mode()[0]), -1.0, 1.0), z_cmd, z_cmd, 1.0


class HeuristicJointPDController(BaseLowLevelController):
    def __init__(self, kp=1.5, kd=0.2):
        super().__init__("heuristic_joint_controller")
        self.kp, self.kd = kp, kd

    def get_action_and_telemetry(self, obs, active_wp, step=0):
        v_des = np.asarray(active_wp) - obs[:2]
        desired_yaw = np.arctan2(v_des[1], v_des[0])
        yaw_err = (desired_yaw - obs[3] + np.pi) % (2 * np.pi) - np.pi
        v_fwd = np.clip(np.linalg.norm(v_des), 0.0, 1.0)
        action = np.zeros(8, dtype=np.float32)
        action[0::2] = np.clip(self.kp * v_fwd - self.kd * obs[13::2][:4], -1.0, 1.0)
        action[1::2] = np.clip(self.kp * yaw_err - self.kd * obs[14::2][:4], -1.0, 1.0)
        dummy_z = np.zeros(128, dtype=np.float32)
        return np.clip(action, -1.0, 1.0), dummy_z, dummy_z, 0.0


def _step_sim(controller, env, obs, cur_pos, waypoints, wp_idx, wp_radius, step, dt):
    active_wp = waypoints[wp_idx]
    if float(np.linalg.norm(cur_pos - np.asarray(active_wp))) <= wp_radius and wp_idx < len(waypoints) - 1:
        wp_idx += 1
        active_wp = waypoints[wp_idx]
    action, z_i, b_g, reach = controller.get_action_and_telemetry(obs, active_wp, step=step)
    next_obs, reward, term, trunc, _ = env.step(action)
    next_pos = next_obs[:2]
    spd = float(np.linalg.norm(next_pos - cur_pos)) / dt
    is_flip = bool(next_obs[2] < 0.28 or next_obs[2] > 1.05 or abs(next_obs[4]) > 0.45 or abs(next_obs[5]) > 0.45)
    return next_obs, next_pos, wp_idx, spd, is_flip, term or trunc, action, z_i, b_g, reach


def _calc_jerk_and_curv(positions, speeds, dt):
    pos_arr = np.asarray(positions)
    if len(pos_arr) >= 4:
        v = np.diff(pos_arr, axis=0) / dt
        a = np.diff(v, axis=0) / dt
        j = np.diff(a, axis=0) / dt
        rms_jerk = float(np.sqrt(np.mean(np.sum(j**2, axis=-1))))
    else:
        rms_jerk = 0.0

    curv = 0.0
    for k in range(len(pos_arr) - 2):
        v1, v2 = pos_arr[k+1] - pos_arr[k], pos_arr[k+2] - pos_arr[k+1]
        l1, l2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if l1 > 1e-4 and l2 > 1e-4:
            curv += float(np.arccos(np.clip(np.dot(v1, v2) / (l1 * l2), -1.0, 1.0)))
    return rms_jerk, curv


def run_single_simulation(controller, env, scenario_name, scenario_cfg, seed=0, dt=0.05):
    start_xy, target_xy, waypoints = list(scenario_cfg["start_xy"]), list(scenario_cfg["target_xy"]), [list(w) for w in scenario_cfg["waypoints"]]
    max_steps, wp_radius, target_radius = int(scenario_cfg["max_steps"]), float(scenario_cfg["waypoint_radius"]), float(scenario_cfg["target_radius"])

    env.reset(seed=seed)
    if hasattr(env.unwrapped, "set_goal"):
        env.unwrapped.set_goal(goal_xy=np.asarray(target_xy, dtype=np.float32))
    env.unwrapped.set_xy(np.asarray(start_xy, dtype=np.float32))
    obs = env.unwrapped.get_ob()

    controller.reset(obs, waypoints)
    cur_pos, wp_idx, step, done, flips = obs[:2], 0, 0, False, 0
    positions, speeds, heights = [cur_pos.copy()], [0.0], [obs[2]]
    records = [{"scenario": scenario_name, "mode": controller.name, "seed": seed, "step": 0, "x": float(cur_pos[0]), "y": float(cur_pos[1]), "z": float(obs[2]), "speed_mps": 0.0, "dist_to_target": float(np.linalg.norm(cur_pos - np.asarray(target_xy))), "reached_goal": bool(np.linalg.norm(cur_pos - np.asarray(target_xy)) <= target_radius)}]

    while step < max_steps and not done:
        obs, next_pos, wp_idx, spd, is_flip, done_env, act, _, _, _ = _step_sim(controller, env, obs, cur_pos, waypoints, wp_idx, wp_radius, step, dt)
        step += 1
        cur_pos = next_pos
        positions.append(cur_pos.copy())
        speeds.append(spd)
        heights.append(obs[2])
        if is_flip:
            flips += 1
        d_target = float(np.linalg.norm(cur_pos - np.asarray(target_xy)))
        reached = bool(d_target <= target_radius)
        records.append({"scenario": scenario_name, "mode": controller.name, "seed": seed, "step": step, "x": float(cur_pos[0]), "y": float(cur_pos[1]), "z": float(obs[2]), "speed_mps": spd, "dist_to_target": d_target, "reached_goal": reached})
        if reached or done_env:
            done = True

    rms_jerk, total_curv = _calc_jerk_and_curv(positions, speeds, dt)
    final_d = float(np.linalg.norm(cur_pos - np.asarray(target_xy)))
    summary = {"scenario": scenario_name, "mode": controller.name, "seed": seed, "reached_goal": bool(final_d <= target_radius), "num_steps": step, "final_dist_to_target": final_d, "mean_speed_mps": float(np.mean(speeds[1:]) if len(speeds) > 1 else 0.0), "flip_steps": flips, "rms_jerk_mps3": rms_jerk, "total_curvature_rad": total_curv}
    return summary, pd.DataFrame(records)


def _plot_left_maze_panel(ax_maze, scenario_name, scenario_cfg, mode_telemetries):
    draw_maze(ax_maze, maze_type="medium")
    start_xy, target_xy = scenario_cfg["start_xy"], scenario_cfg["target_xy"]
    ax_maze.add_patch(patches.Circle((target_xy[0], target_xy[1]), scenario_cfg["target_radius"], facecolor="#ffd54f", edgecolor="#ff8f00", linestyle="--", linewidth=1.4, alpha=0.35, zorder=2))
    wps_arr = np.asarray([[start_xy[0], start_xy[1]]] + scenario_cfg["waypoints"])
    ax_maze.plot(wps_arr[:, 0], wps_arr[:, 1], color="#757575", linestyle=":", linewidth=1.5, alpha=0.6, zorder=3, label="Waypoint Path")
    ax_maze.scatter(wps_arr[1:-1, 0], wps_arr[1:-1, 1], color="#ab47bc", edgecolors="#4a148c", s=70, marker="D", zorder=4, label="Intermediate WPs")

    for mode_name, df_tel in mode_telemetries.items():
        c, lbl = MODE_COLORS.get(mode_name, "#424242"), MODE_LABELS.get(mode_name, mode_name)
        ax_maze.plot(df_tel["x"].values, df_tel["y"].values, color=c, linewidth=2.0, alpha=0.85, zorder=5, label=lbl)
        ax_maze.scatter([df_tel["x"].iloc[-1]], [df_tel["y"].iloc[-1]], color=c, edgecolors="black", s=50, marker="X", zorder=6)

    ax_maze.scatter([start_xy[0]], [start_xy[1]], c="#2e7d32", s=150, marker="o", edgecolors="black", linewidths=1.5, zorder=7, label="Start")
    ax_maze.scatter([target_xy[0]], [target_xy[1]], c="#fbc02d", s=220, marker="*", edgecolors="#d84315", linewidths=1.5, zorder=7, label="Target")
    ax_maze.set_title(f"Trajectory Comparison: {scenario_name}\n({scenario_cfg['description']})", fontsize=12, fontweight="bold", pad=10)
    ax_maze.legend(loc="upper left", fontsize=8.5, framealpha=0.92)


def _plot_right_metric_panels(fig, gs, scenario_cfg, mode_telemetries):
    ax_dist = fig.add_subplot(gs[0, 1])
    for mode_name, df_tel in mode_telemetries.items():
        ax_dist.plot(df_tel["step"], df_tel["dist_to_target"], color=MODE_COLORS.get(mode_name, "#424242"), linewidth=1.8, label=MODE_LABELS.get(mode_name, mode_name))
    ax_dist.axhline(scenario_cfg["target_radius"], color="#ff8f00", linestyle="--", linewidth=1.2)
    ax_dist.set_ylabel("Dist to Goal (m)", fontsize=9, fontweight="bold")
    ax_dist.grid(True, linestyle=":", alpha=0.5)

    ax_speed = fig.add_subplot(gs[1, 1], sharex=ax_dist)
    for mode_name, df_tel in mode_telemetries.items():
        s = pd.Series(df_tel["speed_mps"].values).rolling(window=min(8, len(df_tel)), min_periods=1).mean()
        ax_speed.plot(df_tel["step"], s, color=MODE_COLORS.get(mode_name, "#424242"), linewidth=1.8)
    ax_speed.set_ylabel("Speed (m/s)", fontsize=9, fontweight="bold")
    ax_speed.grid(True, linestyle=":", alpha=0.5)

    ax_height = fig.add_subplot(gs[2, 1], sharex=ax_dist)
    for mode_name, df_tel in mode_telemetries.items():
        ax_height.plot(df_tel["step"], df_tel["z"], color=MODE_COLORS.get(mode_name, "#424242"), linewidth=1.6)
    ax_height.axhspan(0.28, 0.95, color="#81c784", alpha=0.15)
    ax_height.set_xlabel("Step $t$", fontsize=9, fontweight="bold")
    ax_height.set_ylabel("Torso Height (m)", fontsize=9, fontweight="bold")
    ax_height.grid(True, linestyle=":", alpha=0.5)


def plot_scenario_comparison(scenario_name, scenario_cfg, mode_telemetries, output_path):
    fig = plt.figure(figsize=(18, 8.5), dpi=250)
    gs = GridSpec(3, 2, width_ratios=[1.25, 1.0], wspace=0.28, hspace=0.38)
    _plot_left_maze_panel(fig.add_subplot(gs[:, 0]), scenario_name, scenario_cfg, mode_telemetries)
    _plot_right_metric_panels(fig, gs, scenario_cfg, mode_telemetries)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=250)
    plt.close(fig)


def _plot_metric_bar(ax, df_summary, metric, ylabel, title, is_pct=False):
    scenarios, modes, width = list(SCENARIOS.keys()), list(MODE_COLORS.keys()), 0.18
    x = np.arange(len(scenarios))
    for i, m in enumerate(modes):
        sub = df_summary[df_summary["mode"] == m]
        vals = [((sub[sub["scenario"] == sc][metric].mean() * (100.0 if is_pct else 1.0)) if len(sub[sub["scenario"] == sc]) > 0 else 0.0) for sc in scenarios]
        rects = ax.bar(x + i * width, vals, width, label=MODE_LABELS[m], color=MODE_COLORS[m], alpha=0.85, edgecolor="black", linewidth=0.8)
        for r in rects:
            ax.annotate(f"{r.get_height():.1f}{'%' if is_pct else ''}", xy=(r.get_x() + r.get_width() / 2, r.get_height()), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=7.5, fontweight="bold")
    ax.set_xticks(x + width * (len(modes) - 1) / 2)
    ax.set_xticklabels([s.replace("_", " ").title() for s in scenarios], fontsize=9, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=10, fontweight="bold")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.5, axis="y")


def plot_summary_bar_charts(df_summary, output_path):
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), dpi=250)
    plt.subplots_adjust(wspace=0.25, hspace=0.35)
    _plot_metric_bar(axes[0, 0], df_summary, "reached_goal", "Success Rate (%)", "Waypoint Reach Success Rate (%)", is_pct=True)
    axes[0, 0].legend(loc="upper right", fontsize=8)
    _plot_metric_bar(axes[0, 1], df_summary, "mean_speed_mps", "Speed (m/s)", "Mean Locomotion Speed (m/s)")
    _plot_metric_bar(axes[1, 0], df_summary, "rms_jerk_mps3", "RMS Jerk (m/s^3)", "Trajectory Smoothness / Jerk (lower is better)")
    _plot_metric_bar(axes[1, 1], df_summary, "total_curvature_rad", "Total Curvature (rad)", "Trajectory Angular Curvature (lower is better)")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=250)
    plt.close(fig)


def main():
    output_dir = "results/experiments/lowlevel_comparison"
    os.makedirs(output_dir, exist_ok=True)
    agent, env, train_ds, _, _ = load_pretrained_agent("fb-test", "medium", seed=0)
    kdtree = KDTree(train_ds["observations"][:, :2])
    controllers = {
        "baseline_high_actor": BaselineHighActorController(agent, train_ds["observations"], train_ds["observations"][:, :2], kdtree),
        "fb_latent_optimization": FBGradientOptController(agent, train_ds["observations"], train_ds["observations"][:, :2], kdtree, n_steps=25, lr=0.15),
        "heading_direction_alignment": HeadingAlignmentController(agent, train_ds["observations"], train_ds["observations"][:, :2], kdtree, lookahead=2.0),
        "heuristic_joint_controller": HeuristicJointPDController(kp=1.5, kd=0.2),
    }

    all_sums, tel_by_sc = [], {sc: {} for sc in SCENARIOS}
    for sc_name, sc_cfg in SCENARIOS.items():
        for m_name, ctrl in controllers.items():
            for s in [0, 1, 2, 3, 4]:
                summary, df_tel = run_single_simulation(ctrl, env, sc_name, sc_cfg, seed=s)
                all_sums.append(summary)
                if s == 0:
                    tel_by_sc[sc_name][m_name] = df_tel
                df_tel.to_csv(os.path.join(output_dir, f"telemetry_{sc_name}_{m_name}_seed{s}.csv"), index=False)

    df_summary = pd.DataFrame(all_sums)
    df_summary.to_csv(os.path.join(output_dir, "summary_lowlevel_control.csv"), index=False)
    for sc_name, sc_cfg in SCENARIOS.items():
        plot_scenario_comparison(sc_name, sc_cfg, tel_by_sc[sc_name], os.path.join(output_dir, f"comparison_{sc_name}.png"))
    plot_summary_bar_charts(df_summary, os.path.join(output_dir, "aggregate_metrics_comparison.png"))


if __name__ == "__main__":
    main()
