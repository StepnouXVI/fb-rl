import os
import sys
import math
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
from tqdm import tqdm
import jax
import jax.numpy as jnp
import hydra
from omegaconf import DictConfig, OmegaConf

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.agent_loader import load_pretrained_agent
from src.planners import BufferGraphPlanner, _jit_baseline_step
from scripts.visualize_trajectories import draw_maze, normalize_maze_type


def find_nearest_observation(target_xy, dataset_observations):
    coords = dataset_observations[:, :2]
    dists = np.linalg.norm(coords - np.asarray(target_xy), axis=-1)
    best_idx = int(np.argmin(dists))
    return best_idx, float(dists[best_idx])


def encode_coord_to_z(agent, dataset_observations, target_xy):
    best_idx, _ = find_nearest_observation(target_xy, dataset_observations)
    best_state = dataset_observations[best_idx : best_idx + 1]
    z = np.asarray(agent.normalize_z(agent.network.select("backward_repr")(jnp.asarray(best_state)))[0])
    return z, dataset_observations[best_idx]


def _init_mode_state(agent, train_obs, scenario_cfg, mode, target_z, target_xy, obs, cached_planner):
    if mode == "planner":
        p_cfg = scenario_cfg.get("planner_cfg", {})
        p_lm, p_rad, p_cut, p_look = int(p_cfg.get("n_landmarks", 1000)), float(p_cfg.get("max_edge_radius", 3.5)), float(p_cfg.get("reachability_cutoff", 35.0)), float(p_cfg.get("lookahead_dist", 2.6))
        if cached_planner and cached_planner.n_landmarks == p_lm and cached_planner.max_edge_radius == p_rad:
            planner = cached_planner
        else:
            planner = BufferGraphPlanner(agent, train_obs, n_landmarks=p_lm, max_edge_radius=p_rad, reachability_cutoff=p_cut, lookahead_dist=p_look)
        planner.reset(obs, target_z)
        return [c.tolist() if isinstance(c, np.ndarray) else list(c) for c in planner.waypoint_coords], planner, [], [], None
    if mode == "direct_latent":
        return [target_xy], None, [], [], None
    if mode == "custom_waypoints":
        cw = [list(pt) for pt in scenario_cfg.get("custom_waypoints", [])] or [target_xy]
        if np.linalg.norm(np.asarray(cw[-1]) - np.asarray(target_xy)) > 0.5:
            cw.append(target_xy)
        return cw, None, cw, [encode_coord_to_z(agent, train_obs, pt)[0] for pt in cw], None
    if mode == "custom_z":
        raw_z = scenario_cfg.get("custom_z", None)
        return [target_xy], None, [], [], np.asarray(agent.normalize_z(jnp.asarray(raw_z))) if raw_z is not None else target_z
    raise ValueError(f"Unknown intention mode: {mode}")


def _get_step_intention(obs, mode, planner, target_z, target_xy, cw_list, cw_latents, current_wp_idx, z_vector, cur_pos, wp_thresh, step):
    if mode == "planner":
        return planner.get_subgoal_latent(obs, target_z, step=step), planner.get_subgoal_info().get("subgoal_xy") or target_xy, current_wp_idx
    if mode == "direct_latent":
        return target_z, target_xy, current_wp_idx
    if mode == "custom_waypoints":
        if current_wp_idx < len(cw_list) - 1 and float(np.linalg.norm(cur_pos - np.asarray(cw_list[current_wp_idx]))) < wp_thresh:
            current_wp_idx += 1
        return cw_latents[current_wp_idx], cw_list[current_wp_idx], current_wp_idx
    return z_vector, target_xy, current_wp_idx


def _run_scenario_loop(env, agent, obs, mode, planner, target_z, target_xy, cw_list, cw_latents, z_vec, wp_thresh, max_steps, target_radius, terminate_on_goal, eval_temp, seed, records):
    cur_pos, cur_wp_idx, step, done = np.asarray(obs[:2]), 0, 0, False
    while step < max_steps and not done:
        z_t, sg_xy, cur_wp_idx = _get_step_intention(obs, mode, planner, target_z, target_xy, cw_list, cw_latents, cur_wp_idx, z_vec, cur_pos, wp_thresh, step)
        d_sg, d_goal = float(np.linalg.norm(cur_pos - np.asarray(sg_xy))), float(np.linalg.norm(cur_pos - np.asarray(target_xy)))
        reached = bool(d_goal <= target_radius)

        seed_k = jax.random.PRNGKey(seed * 10000 + step) if eval_temp > 0 else None
        act, _ = _jit_baseline_step(agent, jnp.asarray(obs), jnp.asarray(z_t), seed=seed_k, temperature=eval_temp)
        action_arr = np.asarray(act)

        obs, reward, term, trunc, _ = env.step(action_arr)
        next_pos = np.asarray(obs[:2])
        v_vec = next_pos - cur_pos
        speed = float(np.linalg.norm(v_vec))

        sg_dir = np.asarray(sg_xy) - cur_pos
        sg_norm = np.linalg.norm(sg_dir)
        cos_align = float(np.dot(v_vec, sg_dir) / (speed * sg_norm)) if (speed > 1e-4 and sg_norm > 1e-4) else 1.0

        records.append({"step": step + 1, "x": float(next_pos[0]), "y": float(next_pos[1]), "speed": speed, "subgoal_x": float(sg_xy[0]), "subgoal_y": float(sg_xy[1]), "dist_to_subgoal": float(np.linalg.norm(next_pos - np.asarray(sg_xy))), "dist_to_goal": float(np.linalg.norm(next_pos - np.asarray(target_xy))), "alignment_cos": cos_align, "reached_goal": reached})
        cur_pos = next_pos
        step += 1
        if (terminate_on_goal and reached) or term or trunc:
            done = True


def run_scenario(agent, env, train_ds, config, scenario_cfg, eval_temperature=0.0, target_radius=1.0, terminate_on_goal=True, seed=0, cached_planner=None):
    start_xy, target_xy = list(scenario_cfg.get("start_xy", [0.0, 0.0])), list(scenario_cfg.get("target_xy", [0.0, 0.0]))
    mode, max_steps, wp_thresh = str(scenario_cfg.get("mode", "planner")).lower(), int(scenario_cfg.get("max_steps", 500)), float(scenario_cfg.get("waypoint_threshold", 1.8))

    env.reset(seed=seed)
    if hasattr(env.unwrapped, "set_goal"):
        env.unwrapped.set_goal(goal_xy=np.asarray(target_xy, dtype=np.float32))
    env.unwrapped.set_xy(np.asarray(start_xy, dtype=np.float32))
    obs = env.unwrapped.get_ob()

    target_z, _ = encode_coord_to_z(agent, train_ds["observations"], target_xy)
    waypoints, planner, cw_list, cw_latents, z_vec = _init_mode_state(agent, train_ds["observations"], scenario_cfg, mode, target_z, target_xy, obs, cached_planner)

    records = [{"step": 0, "x": float(start_xy[0]), "y": float(start_xy[1]), "speed": 0.0, "subgoal_x": float(waypoints[0][0]), "subgoal_y": float(waypoints[0][1]), "dist_to_subgoal": float(np.linalg.norm(np.asarray(start_xy) - np.asarray(waypoints[0]))), "dist_to_goal": float(np.linalg.norm(np.asarray(start_xy) - np.asarray(target_xy))), "alignment_cos": 1.0, "reached_goal": False}]
    _run_scenario_loop(env, agent, obs, mode, planner, target_z, target_xy, cw_list, cw_latents, z_vec, wp_thresh, max_steps, target_radius, terminate_on_goal, eval_temperature, seed, records)

    df_tel = pd.DataFrame(records)
    final_dist = float(df_tel["dist_to_goal"].iloc[-1])
    reached = bool(final_dist <= target_radius or df_tel["reached_goal"].any())
    summary = {"name": scenario_cfg.get("name", "scenario"), "mode": mode, "start_x": start_xy[0], "start_y": start_xy[1], "target_x": target_xy[0], "target_y": target_xy[1], "steps_taken": len(df_tel) - 1, "reached_goal": reached, "final_dist_to_goal": final_dist, "mean_speed": float(df_tel["speed"].mean()), "mean_alignment": float(df_tel["alignment_cos"].mean())}
    return summary, df_tel, waypoints


def _plot_maze_panel(ax_maze, summary, df_tel, waypoints, maze_type, target_radius):
    draw_maze(ax_maze, maze_type=maze_type)
    ax_maze.add_patch(patches.Circle((summary["target_x"], summary["target_y"]), target_radius, facecolor="#ffd54f", edgecolor="#ff8f00", linestyle="--", linewidth=1.2, alpha=0.35, zorder=2))
    if waypoints and len(waypoints) > 1:
        w = np.asarray(waypoints)
        ax_maze.plot(w[:, 0], w[:, 1], color="#8e24aa", linestyle="--", linewidth=1.5, alpha=0.65, zorder=3, label="Waypoints")
        ax_maze.scatter(w[:, 0], w[:, 1], color="#ab47bc", edgecolors="#4a148c", s=45, marker="D", zorder=4)

    xs, ys, steps = df_tel["x"].values, df_tel["y"].values, df_tel["step"].values
    if len(xs) > 1:
        ax_maze.plot(xs, ys, color="#424242", linewidth=1.2, alpha=0.4, zorder=4)
        ax_maze.scatter(xs, ys, c=steps, cmap="plasma", s=18, alpha=0.85, zorder=5, label="Trajectory")

    stride = max(1, len(df_tel) // 18)
    for idx in range(0, len(df_tel), stride):
        r = df_tel.iloc[idx]
        dx, dy = r["subgoal_x"] - r["x"], r["subgoal_y"] - r["y"]
        d = math.hypot(dx, dy)
        if d > 0.4:
            ax_maze.annotate("", xy=(r["x"] + (dx / d) * min(d, 1.8), r["y"] + (dy / d) * min(d, 1.8)), xytext=(r["x"], r["y"]), arrowprops=dict(arrowstyle="->,head_width=0.25,head_length=0.35", color="#00897b", lw=1.3, alpha=0.75), zorder=6)

    ax_maze.scatter([summary["start_x"]], [summary["start_y"]], c="#2e7d32", s=140, marker="o", edgecolors="black", linewidths=1.5, zorder=7, label="Start")
    ax_maze.scatter([summary["target_x"]], [summary["target_y"]], c="#fbc02d", s=220, marker="*", edgecolors="#d84315", linewidths=1.5, zorder=7, label="Target")
    ax_maze.set_title(f"Scenario: {summary['name']} [{summary['mode'].upper()}]\nStatus: {'SUCCESS' if summary['reached_goal'] else 'FAILED'} (Steps: {summary['steps_taken']}, Final: {summary['final_dist_to_goal']:.2f}m)", fontsize=10.5, fontweight="bold", color="#1b5e20" if summary['reached_goal'] else "#b71c1c")
    ax_maze.legend(loc="upper right", fontsize=8, framealpha=0.9)


def _plot_metrics_panel(fig, gs, df_tel, target_radius):
    steps = df_tel["step"].values
    ax_dist = fig.add_subplot(gs[0, 1])
    ax_dist.plot(steps, df_tel["dist_to_goal"], color="#d32f2f", linewidth=2.0, label="Dist to Target")
    ax_dist.plot(steps, df_tel["dist_to_subgoal"], color="#0288d1", linewidth=1.5, linestyle="--", label="Dist to Subgoal")
    ax_dist.axhline(target_radius, color="#388e3c", linestyle=":", label="Goal Radius")
    ax_dist.set_ylabel("Distance (m)", fontsize=9)
    ax_dist.legend(loc="upper right", fontsize=7.5)
    ax_dist.grid(True, linestyle="--", alpha=0.5)

    ax_spd = fig.add_subplot(gs[1, 1], sharex=ax_dist)
    ax_spd.plot(steps, df_tel["speed"], color="#388e3c", linewidth=1.8)
    ax_spd.set_ylabel("Speed (m/step)", fontsize=9)
    ax_spd.grid(True, linestyle="--", alpha=0.5)

    ax_align = fig.add_subplot(gs[2, 1], sharex=ax_dist)
    ax_align.plot(steps, df_tel["alignment_cos"], color="#7b1fa2", linewidth=1.6)
    ax_align.set_ylabel("Cos Alignment", fontsize=9)
    ax_align.set_xlabel("Step", fontsize=9)
    ax_align.set_ylim(-1.05, 1.05)
    ax_align.grid(True, linestyle="--", alpha=0.5)


def plot_scenario_results(summary, df_tel, waypoints, output_path, maze_type="medium", target_radius=1.0):
    fig = plt.figure(figsize=(16, 7.5), dpi=200)
    gs = GridSpec(3, 2, width_ratios=[1.15, 1.0], wspace=0.28, hspace=0.38)
    _plot_maze_panel(fig.add_subplot(gs[:, 0]), summary, df_tel, waypoints, maze_type, target_radius)
    _plot_metrics_panel(fig, gs, df_tel, target_radius)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


@hydra.main(version_base=None, config_path="../configs", config_name="experiment")
def main(cfg: DictConfig):
    out_dir = os.path.join(cfg.experiment.output_dir, "experiments")
    os.makedirs(out_dir, exist_ok=True)
    agent, env, train_ds, _, fb_cfg = load_pretrained_agent(cfg.experiment.checkpoint_dir, cfg.env.split, seed=cfg.experiment.seed)

    scenarios = [s for s in cfg.scenarios if cfg.experiment.scenario_name in ("all", s.get("name"))]
    summaries = []
    cached_p = None

    for sc in tqdm(scenarios, desc="Executing Intention Scenarios"):
        s_sum, df_tel, wps = run_scenario(agent, env, train_ds, fb_cfg, sc, cfg.experiment.eval_temperature, cfg.experiment.target_radius, cfg.experiment.terminate_on_goal, cfg.experiment.seed, cached_p)
        summaries.append(s_sum)
        df_tel.to_csv(os.path.join(out_dir, f"telemetry_{s_sum['name']}_{s_sum['mode']}.csv"), index=False)
        plot_scenario_results(s_sum, df_tel, wps, os.path.join(out_dir, f"scenario_{s_sum['name']}_{s_sum['mode']}.png"), cfg.env.split, cfg.experiment.target_radius)

    pd.DataFrame(summaries).to_csv(os.path.join(out_dir, "summary_experiments.csv"), index=False)


if __name__ == "__main__":
    main()
