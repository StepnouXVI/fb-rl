import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from collections import defaultdict

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

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
        [1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1],
        [1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1],
        [1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1],
        [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1],
        [1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    ]),
    "giant": np.array([
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
        [1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1],
        [1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
        [1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1],
        [1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1],
        [1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1],
        [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1],
        [1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1],
        [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
        [1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    ]),
    "teleport": np.array([
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
        [1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1],
        [1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1],
        [1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1],
        [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1],
        [1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    ]),
}

TELEPORT_INFO = {
    "teleport": {
        "teleport_in_xys": [(-2.0, 6.0), (10.0, 10.0)],
        "teleport_out_xys": [(14.0, 2.0), (6.0, 14.0), (-2.0, 18.0)],
        "teleport_radius": 1.0,
    }
}

TASK_COORDS = {
    "medium": {
        1: {"init": (0.0, 0.0), "goal": (20.0, 0.0), "name": "Task 1"},
        2: {"init": (0.0, 0.0), "goal": (20.0, 20.0), "name": "Task 2"},
        3: {"init": (0.0, 0.0), "goal": (0.0, 20.0), "name": "Task 3"},
        4: {"init": (20.0, 0.0), "goal": (0.0, 20.0), "name": "Task 4"},
        5: {"init": (20.0, 20.0), "goal": (0.0, 0.0), "name": "Task 5"},
    },
    "large": {
        1: {"init": (0.0, 0.0), "goal": (32.0, 0.0), "name": "Task 1"},
        2: {"init": (0.0, 0.0), "goal": (32.0, 24.0), "name": "Task 2"},
        3: {"init": (0.0, 0.0), "goal": (0.0, 24.0), "name": "Task 3"},
        4: {"init": (32.0, 0.0), "goal": (0.0, 24.0), "name": "Task 4"},
        5: {"init": (32.0, 24.0), "goal": (0.0, 0.0), "name": "Task 5"},
    },
    "giant": {
        1: {"init": (0.0, 0.0), "goal": (48.0, 0.0), "name": "Task 1"},
        2: {"init": (0.0, 0.0), "goal": (48.0, 36.0), "name": "Task 2"},
        3: {"init": (0.0, 0.0), "goal": (0.0, 36.0), "name": "Task 3"},
        4: {"init": (48.0, 0.0), "goal": (0.0, 36.0), "name": "Task 4"},
        5: {"init": (48.0, 36.0), "goal": (0.0, 0.0), "name": "Task 5"},
    },
    "teleport": {
        1: {"init": (0.0, 0.0), "goal": (32.0, 0.0), "name": "Task 1"},
        2: {"init": (0.0, 0.0), "goal": (32.0, 24.0), "name": "Task 2"},
        3: {"init": (0.0, 0.0), "goal": (0.0, 24.0), "name": "Task 3"},
        4: {"init": (32.0, 0.0), "goal": (0.0, 24.0), "name": "Task 4"},
        5: {"init": (32.0, 24.0), "goal": (0.0, 0.0), "name": "Task 5"},
    },
}


def normalize_maze_type(maze_name):
    low = str(maze_name).lower()
    for k in ["giant", "large", "teleport", "medium"]:
        if k in low:
            return k
    return "medium"


def sanitize_method_name(name):
    n = str(name).strip().lower()
    for p in ["1. ", "2. ", "3. ", "4. ", "5. ", "6. "]:
        if n.startswith(p):
            n = n[len(p):]
    return n.replace(" + ", "_").replace(" ", "_").replace("[", "").replace("]", "").replace("(", "").replace(")", "").replace("-", "_")


def auto_detect_maze_type(df_traj):
    max_c = max(df_traj["x"].abs().max(), df_traj["y"].abs().max())
    if max_c > 35.0:
        return "giant"
    if max_c > 22.0:
        pts = df_traj[["x", "y"]].values
        if len(pts) > 1 and np.max(np.linalg.norm(np.diff(pts, axis=0), axis=1)) > 8.0:
            return "teleport"
        return "large"
    return "medium"


def get_task_info(task_id, maze_type="medium", traj_df=None):
    norm_type = normalize_maze_type(maze_type)
    if norm_type in TASK_COORDS and task_id in TASK_COORDS[norm_type]:
        return TASK_COORDS[norm_type][task_id]
    if traj_df is not None and not traj_df.empty:
        t_sub = traj_df[traj_df["task_id"] == task_id]
        if not t_sub.empty:
            ep0 = t_sub[t_sub["episode"] == t_sub["episode"].iloc[0]]
            return {"init": (float(ep0["x"].iloc[0]), float(ep0["y"].iloc[0])), "goal": (float(ep0["x"].iloc[-1]), float(ep0["y"].iloc[-1])), "name": f"Task {task_id}"}
    return {"init": (0.0, 0.0), "goal": (20.0, 20.0), "name": f"Task {task_id}"}


def _draw_walls(ax, grid, unit_size, offset_x, offset_y):
    h, w = grid.shape
    for i in range(h):
        for j in range(w):
            if grid[i, j] == 1:
                x, y = j * unit_size - offset_x - unit_size / 2.0, i * unit_size - offset_y - unit_size / 2.0
                ax.add_patch(patches.Rectangle((x, y), unit_size, unit_size, linewidth=1, edgecolor="#2b2b2b", facecolor="#d0d0d0", zorder=1))


def _draw_teleport_portals(ax, t_info, show_links):
    in_xys, out_xys, rad = t_info.get("teleport_in_xys", []), t_info.get("teleport_out_xys", []), float(t_info.get("teleport_radius", 1.0))
    if show_links and in_xys and out_xys:
        for in_xy in in_xys:
            for out_xy in out_xys:
                ax.add_patch(patches.FancyArrowPatch(in_xy, out_xy, connectionstyle="arc3,rad=0.15", color="#ab47bc", linestyle="--", linewidth=1.1, alpha=0.35, arrowstyle="->,head_width=3,head_length=5", zorder=2))
    for idx, (px, py) in enumerate(in_xys):
        ax.add_patch(patches.Circle((px, py), rad * 1.5, fill=True, facecolor="#00e5ff", alpha=0.22, edgecolor="#00bcd4", linestyle="--", linewidth=1.5, zorder=2))
        ax.add_patch(patches.Circle((px, py), rad * 0.75, fill=True, facecolor="#00e5ff", edgecolor="#006064", linewidth=1.8, alpha=0.85, zorder=3))
        ax.text(px, py, f"{idx+1}$", ha="center", va="center", fontsize=7.5, fontweight="bold", color="#00363a", zorder=4)
    for idx, (px, py) in enumerate(out_xys):
        ax.add_patch(patches.Circle((px, py), rad * 1.2, fill=True, facecolor="#ff9100", alpha=0.22, edgecolor="#ff6d00", linestyle=":", linewidth=1.5, zorder=2))
        ax.add_patch(patches.Circle((px, py), rad * 0.75, fill=True, facecolor="#ff9100", edgecolor="#bf360c", linewidth=1.8, alpha=0.85, zorder=3))
        ax.text(px, py, f"{idx+1}$", ha="center", va="center", fontsize=7.5, fontweight="bold", color="#3e2723", zorder=4)


def draw_maze(ax, maze_type="medium", unit_size=4.0, offset_x=4.0, offset_y=4.0, custom_grid=None, show_portals=True, show_portal_links=True):
    grid = np.asarray(custom_grid) if custom_grid is not None else MAZE_LAYOUTS.get(normalize_maze_type(maze_type), MAZE_LAYOUTS["medium"])
    _draw_walls(ax, grid, unit_size, offset_x, offset_y)
    t_info = TELEPORT_INFO.get(normalize_maze_type(maze_type), None)
    if show_portals and t_info is not None:
        _draw_teleport_portals(ax, t_info, show_portal_links)

    h, w = grid.shape
    x_min, x_max = -offset_x - unit_size / 2.0, (w - 1) * unit_size - offset_x + unit_size / 2.0
    y_min, y_max = -offset_y - unit_size / 2.0, (h - 1) * unit_size - offset_y + unit_size / 2.0
    ax.set_xlim(x_min - 0.5, x_max + 0.5)
    ax.set_ylim(y_min - 0.5, y_max + 0.5)
    ax.set_aspect("equal")
    ax.set_facecolor("#f9f9f9")
    ax.grid(True, linestyle=":", alpha=0.4, color="#999999")


def plot_path_with_teleports(ax, xs, ys, jump_threshold=5.0, color="#333333", alpha=0.35, linewidth=1.2, zorder=2, label=None):
    if len(xs) == 0:
        return
    pts = np.column_stack([xs, ys])
    if len(pts) <= 1:
        ax.plot(xs, ys, color=color, alpha=alpha, linewidth=linewidth, zorder=zorder, label=label)
        return
    dists = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    jump_idxs = np.where(dists > jump_threshold)[0]
    starts, ends = [0] + list(jump_idxs + 1), list(jump_idxs + 1) + [len(pts)]
    for idx, (s, e) in enumerate(zip(starts, ends)):
        if e > s:
            ax.plot(xs[s:e], ys[s:e], color=color, alpha=alpha, linewidth=linewidth, zorder=zorder, label=(label if idx == 0 else None))
    for j in jump_idxs:
        ax.add_patch(patches.FancyArrowPatch(pts[j], pts[j+1], connectionstyle="arc3,rad=0.2", color="#d81b60", linestyle=":", linewidth=1.8, alpha=0.75, arrowstyle="->,head_width=3.5,head_length=5.5", zorder=zorder+1))


def extract_executed_waypoints(sg_df):
    if sg_df is None or sg_df.empty:
        return []
    if "planned_waypoints" in sg_df.columns:
        for val in sg_df["planned_waypoints"].dropna():
            try:
                pts = json.loads(val) if isinstance(val, str) else val
                if isinstance(pts, list) and len(pts) >= 2:
                    return [[float(p[0]), float(p[1])] for p in pts]
            except Exception:
                pass
    visited = []
    for _, row in sg_df.dropna(subset=["subgoal_x", "subgoal_y"]).iterrows():
        pt = [float(row["subgoal_x"]), float(row["subgoal_y"])]
        if not visited or np.linalg.norm(np.array(pt) - np.array(visited[-1])) > 0.8:
            visited.append(pt)
    return visited


def _render_markers_and_decorations(ax, init_xy, goal_xy, executed_wps, is_baseline, final_dist, title, steps_len, reached=False):
    ax.scatter(init_xy[0], init_xy[1], marker="o", color="#2e7d32", s=140, edgecolors="black", linewidth=1.5, zorder=6, label="Start (s0)")
    ax.scatter(goal_xy[0], goal_xy[1], marker="*", color="#ffd700", s=280, edgecolors="#b78103", linewidth=1.5, zorder=6, label="Final Goal (g)")
    if executed_wps and not is_baseline:
        w_arr = np.array(executed_wps)
        ax.plot(w_arr[:, 0], w_arr[:, 1], "r--", linewidth=1.8, alpha=0.75, zorder=4, label="Dijkstra Waypoints")
        ax.scatter(w_arr[:, 0], w_arr[:, 1], marker="D", color="#e53935", s=50, edgecolors="black", zorder=5)

    status_str, status_col = ("SUCCESS", "#1b5e20") if reached else ("FAILED", "#b71c1c")
    ax.set_title(f"{title}\nStatus: {status_str} | Final Dist: {final_dist:.2f}m | Steps: {steps_len}", fontsize=9.5, fontweight="bold", color=status_col)
    ax.legend(loc="upper right", fontsize=7.5, framealpha=0.9)


def plot_trajectory_panel(ax, traj_df, sg_df=None, title="", is_baseline=False, maze_type="medium", unit_size=4.0, show_portals=True, show_portal_links=True):
    draw_maze(ax, maze_type=maze_type, unit_size=unit_size, show_portals=show_portals, show_portal_links=show_portal_links)
    if traj_df is None or traj_df.empty:
        ax.set_title(f"{title}\n(No data)", fontsize=11, fontweight="bold")
        return False

    task_info = get_task_info(int(traj_df["task_id"].iloc[0]), maze_type=maze_type, traj_df=traj_df)
    xs, ys, steps = traj_df["x"].values, traj_df["y"].values, traj_df["step"].values
    ax.scatter(xs, ys, c=steps, cmap="plasma", s=14, alpha=0.85, zorder=3, label="Ant Path (t=0..T)")
    plot_path_with_teleports(ax, xs, ys, jump_threshold=5.0, color="#333333", alpha=0.35, linewidth=1.2, zorder=2)

    init_xy, goal_xy = task_info["init"], task_info["goal"]
    final_dist = float(np.linalg.norm(np.array([xs[-1], ys[-1]]) - np.array(goal_xy)))
    reached = bool(float(traj_df["reward"].max()) > 0.0) if ("reward" in traj_df.columns and float(traj_df["reward"].max()) > 0.0) else (final_dist < 1.5)
    _render_markers_and_decorations(ax, init_xy, goal_xy, extract_executed_waypoints(sg_df), is_baseline, final_dist, title, len(steps), reached=reached)
    return reached


def plot_landmarks_panel(ax, lm_df, sg_df=None, task_id=1, maze_type="medium", unit_size=4.0, show_portals=True, show_portal_links=True):
    draw_maze(ax, maze_type=maze_type, unit_size=unit_size, show_portals=show_portals, show_portal_links=show_portal_links)
    if lm_df is not None and not lm_df.empty:
        ax.scatter(lm_df["x"], lm_df["y"], c="#1976d2", s=15, alpha=0.55, zorder=3, label=f"Landmarks (N={len(lm_df)})")
    task_info = get_task_info(task_id, maze_type=maze_type)
    ax.scatter(task_info["init"][0], task_info["init"][1], marker="o", color="#2e7d32", s=140, edgecolors="black", linewidth=1.5, zorder=6, label="Start")
    ax.scatter(task_info["goal"][0], task_info["goal"][1], marker="*", color="#ffd700", s=280, edgecolors="#b78103", linewidth=1.5, zorder=6, label="Final Goal")
    ax.set_title(f"Reachability Graph Landmarks & Route\nTask {task_id} ({maze_type.capitalize()})", fontsize=9.5, fontweight="bold")
    ax.legend(loc="upper right", fontsize=7.5, framealpha=0.9)


def plot_multi_episode_panel(ax, traj_df, sg_df=None, title="", is_baseline=False, maze_type="medium", unit_size=4.0, show_portals=True, show_portal_links=True):
    draw_maze(ax, maze_type=maze_type, unit_size=unit_size, show_portals=show_portals, show_portal_links=show_portal_links)
    if traj_df is None or traj_df.empty:
        return
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    for ep_idx, ep in enumerate(sorted(traj_df["episode"].unique())):
        ep_traj = traj_df[traj_df["episode"] == ep]
        plot_path_with_teleports(ax, ep_traj["x"].values, ep_traj["y"].values, color=colors[ep_idx % 10], alpha=0.75, linewidth=1.5, zorder=3, label=f"Ep {ep}")
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.legend(loc="upper right", fontsize=7, framealpha=0.85)


def plot_overview_grid(df_traj, df_sg=None, seed=0, maze_type="medium", unit_size=4.0, save_path=None, show_portals=True, show_portal_links=True):
    tasks, methods = sorted(df_traj["task_id"].unique()), sorted(df_traj["method"].unique())
    fig, axes = plt.subplots(len(tasks), len(methods), figsize=(len(methods) * 4.5, len(tasks) * 4.0), dpi=120)
    if len(tasks) == 1 and len(methods) == 1:
        axes = np.array([[axes]])
    elif len(tasks) == 1 or len(methods) == 1:
        axes = np.atleast_2d(axes)

    for r, t in enumerate(tasks):
        for c, m in enumerate(methods):
            sub_traj = df_traj[(df_traj["method"] == m) & (df_traj["seed"] == seed) & (df_traj["task_id"] == t)]
            plot_multi_episode_panel(axes[r, c], sub_traj, title=f"{m} | Task {t}", is_baseline="baseline" in m.lower(), maze_type=maze_type, unit_size=unit_size, show_portals=show_portals, show_portal_links=show_portal_links)
    plt.tight_layout()
    out = save_path or f"results/plots/{maze_type}/overview_{maze_type}_seed{seed}.png"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, bbox_inches="tight")
    plt.close()


def render_3panel_comparison(df_traj, df_sg=None, df_lm=None, seed=0, task_id=1, episode=0, maze_type="medium", unit_size=4.0, show_portals=True, show_portal_links=True, target_method="Buffer Graph Dijkstra (Branch 2)", baseline_method="Single-Intention Baseline", output_dir="results/plots", save_path=None, organize_by_outcome=True):
    norm_type = normalize_maze_type(maze_type)
    grid = MAZE_LAYOUTS.get(norm_type, MAZE_LAYOUTS["medium"])
    fig_w, fig_h = max(6.5, 6.5 * (grid.shape[1] / max(1, grid.shape[0]))), 6.5
    fig, axes = plt.subplots(1, 3, figsize=(fig_w * 2.6, fig_h), dpi=150)

    sub_traj_b = df_traj[(df_traj["method"] == baseline_method) & (df_traj["seed"] == seed) & (df_traj["task_id"] == task_id) & (df_traj["episode"] == episode)]
    sub_sg_b = df_sg[(df_sg["method"] == baseline_method) & (df_sg["seed"] == seed) & (df_sg["task_id"] == task_id) & (df_sg["episode"] == episode)] if df_sg is not None else None
    plot_trajectory_panel(axes[0], sub_traj_b, sub_sg_b, title=f"{baseline_method} (Seed {seed})", is_baseline=True, maze_type=norm_type, unit_size=unit_size, show_portals=show_portals, show_portal_links=show_portal_links)

    sub_traj_d = df_traj[(df_traj["method"] == target_method) & (df_traj["seed"] == seed) & (df_traj["task_id"] == task_id) & (df_traj["episode"] == episode)]
    sub_sg_d = df_sg[(df_sg["method"] == target_method) & (df_sg["seed"] == seed) & (df_sg["task_id"] == task_id) & (df_sg["episode"] == episode)] if df_sg is not None else None
    d_success = plot_trajectory_panel(axes[1], sub_traj_d, sub_sg_d, title=f"{target_method} (Seed {seed})", is_baseline=False, maze_type=norm_type, unit_size=unit_size, show_portals=show_portals, show_portal_links=show_portal_links)

    plot_landmarks_panel(axes[2], df_lm, sub_sg_d, task_id=task_id, maze_type=norm_type, unit_size=unit_size, show_portals=show_portals, show_portal_links=show_portal_links)
    outcome = "success" if (d_success if not sub_traj_d.empty else (float(sub_traj_b["reward"].max()) > 0.0 if not sub_traj_b.empty else False)) else "failed"

    out_file = save_path or os.path.join(output_dir, norm_type, outcome if organize_by_outcome else "", f"compare_3panel_{norm_type}_seed{seed}_task{task_id}_ep{episode}.png")
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_file, bbox_inches="tight")
    plt.close()
    return out_file, outcome


def export_all_comparisons(df_traj, df_sg=None, df_lm=None, maze_type="medium", unit_size=4.0, show_portals=True, show_portal_links=True, output_dir="results/plots", seed_filter=None, task_filter=None):
    norm_type = normalize_maze_type(maze_type)
    episodes_meta = df_traj[["seed", "task_id", "episode"]].drop_duplicates().sort_values(["seed", "task_id", "episode"])
    if seed_filter is not None:
        episodes_meta = episodes_meta[episodes_meta["seed"] == seed_filter]
    if task_filter is not None:
        episodes_meta = episodes_meta[episodes_meta["task_id"] == task_filter]

    counts = {"success": 0, "failed": 0}
    for _, row in episodes_meta.iterrows():
        _, outcome = render_3panel_comparison(df_traj, df_sg, df_lm, int(row["seed"]), int(row["task_id"]), int(row["episode"]), norm_type, unit_size, show_portals, show_portal_links, output_dir=output_dir, organize_by_outcome=True)
        counts[outcome] += 1
    print(f"Exported {counts['success']} SUCCESS, {counts['failed']} FAILED comparisons to {output_dir}/{norm_type}/")
    return counts


def export_all_methods_trajectories(df_traj, df_sg=None, df_lm=None, maze_type="large", unit_size=4.0, show_portals=True, show_portal_links=True, base_output_dir="results", seed_filter=None, task_filter=None):
    norm_type = normalize_maze_type(maze_type)
    grid = MAZE_LAYOUTS.get(norm_type, MAZE_LAYOUTS["medium"])
    fig_w, fig_h = max(6.5, 6.5 * (grid.shape[1] / max(1, grid.shape[0]))), 6.5
    counts = defaultdict(lambda: {"success": 0, "failed": 0})

    for m in df_traj["method"].unique():
        clean_name = sanitize_method_name(m)
        sub_df = df_traj[df_traj["method"] == m]
        episodes = sub_df[["seed", "task_id", "episode"]].drop_duplicates().sort_values(["seed", "task_id", "episode"])
        if seed_filter is not None:
            episodes = episodes[episodes["seed"] == seed_filter]
        if task_filter is not None:
            episodes = episodes[episodes["task_id"] == task_filter]

        for _, row in episodes.iterrows():
            s, t, ep = int(row["seed"]), int(row["task_id"]), int(row["episode"])
            cur_traj = sub_df[(sub_df["seed"] == s) & (sub_df["task_id"] == t) & (sub_df["episode"] == ep)]
            cur_sg = df_sg[(df_sg["method"] == m) & (df_sg["seed"] == s) & (df_sg["task_id"] == t) & (df_sg["episode"] == ep)] if df_sg is not None else None

            fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=150)
            reached = plot_trajectory_panel(ax, cur_traj, cur_sg, title=f"{m} | Task {t} | Ep {ep} (Seed {s})", is_baseline="baseline" in clean_name, maze_type=norm_type, unit_size=unit_size, show_portals=show_portals, show_portal_links=show_portal_links)
            outcome = "success" if reached else "failed"
            save_file = os.path.join(base_output_dir, clean_name, norm_type, outcome, f"{clean_name}_{norm_type}_seed{s}_task{t}_ep{ep}.png")
            os.makedirs(os.path.dirname(save_file), exist_ok=True)
            plt.tight_layout()
            plt.savefig(save_file, bbox_inches="tight")
            plt.close()
            counts[clean_name][outcome] += 1
    return counts


def _find_csv(filepath):
    if not filepath:
        return None
    for p in [filepath, filepath + ".gz", os.path.join("results", "data", os.path.basename(filepath)), os.path.join("results", "data", os.path.basename(filepath) + ".gz"), os.path.join("results", os.path.basename(filepath)), os.path.join("results", os.path.basename(filepath) + ".gz")]:
        if os.path.exists(p):
            return p
    return None


def main():
    parser = argparse.ArgumentParser(description="Visualize AntMaze Navigation Trajectories")
    parser.add_argument("--traj_file", type=str, default="results/data/trajectories.csv")
    parser.add_argument("--sg_file", type=str, default="results/data/subgoals.csv")
    parser.add_argument("--lm_file", type=str, default="results/benchmarks/landmarks.csv")
    parser.add_argument("--output_dir", type=str, default="results/plots")
    parser.add_argument("--maze_type", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--task", type=int, default=1)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--export_comparisons", action="store_true")
    args = parser.parse_args()

    traj_path = _find_csv(args.traj_file)
    if not traj_path:
        print(f"File {args.traj_file} not found.")
        return

    df_traj = pd.read_csv(traj_path)
    sg_p = _find_csv(args.sg_file)
    df_sg = pd.read_csv(sg_p) if sg_p else None
    lm_p = _find_csv(args.lm_file)
    df_lm = pd.read_csv(lm_p) if lm_p else None
    maze_type = auto_detect_maze_type(df_traj) if args.maze_type.lower() == "auto" else normalize_maze_type(args.maze_type)

    if args.export_comparisons:
        export_all_comparisons(df_traj, df_sg, df_lm, maze_type=maze_type, output_dir=args.output_dir)
    elif args.compare:
        render_3panel_comparison(df_traj, df_sg, df_lm, seed=args.seed, task_id=args.task, episode=args.episode, maze_type=maze_type, output_dir=args.output_dir)
    else:
        export_all_methods_trajectories(df_traj, df_sg, df_lm, maze_type=maze_type, base_output_dir="results")


if __name__ == "__main__":
    main()
