import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# AntMaze medium maze layout definition
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
}

TASK_COORDS = {
    1: {"init": (0.0, 0.0), "goal": (20.0, 20.0), "name": "Task 1 (SW -> NE)"},
    2: {"init": (0.0, 20.0), "goal": (20.0, 0.0), "name": "Task 2 (NW -> SE)"},
    3: {"init": (8.0, 16.0), "goal": (4.0, 12.0), "name": "Task 3 (Center Navigate)"},
    4: {"init": (16.0, 20.0), "goal": (0.0, 20.0), "name": "Task 4 (E -> W Corridor)"},
    5: {"init": (20.0, 4.0), "goal": (0.0, 0.0), "name": "Task 5 (SE -> SW Corner)"},
}

def draw_maze(ax, maze_type="medium", unit_size=4.0):
    grid = MAZE_LAYOUTS.get(maze_type, MAZE_LAYOUTS["medium"])
    h, w = grid.shape
    for i in range(h):
        for j in range(w):
            if grid[i, j] == 1:
                x = (j - 1) * unit_size - unit_size / 2.0
                y = (i - 1) * unit_size - unit_size / 2.0
                rect = patches.Rectangle(
                    (x, y), unit_size, unit_size, linewidth=1, edgecolor="#2b2b2b", facecolor="#d0d0d0", zorder=1
                )
                ax.add_patch(rect)
    ax.set_xlim(-unit_size, (w - 1) * unit_size)
    ax.set_ylim(-unit_size, (h - 1) * unit_size)
    ax.set_aspect("equal")
    ax.set_facecolor("#f9f9f9")
    ax.grid(True, linestyle=":", alpha=0.4, color="#999999")

def extract_executed_waypoints(sg_df):
    if sg_df is None or sg_df.empty:
        return []
    sg_valid = sg_df.dropna(subset=["subgoal_x", "subgoal_y"])
    visited = []
    for _, row in sg_valid.iterrows():
        pt = [float(row["subgoal_x"]), float(row["subgoal_y"])]
        if len(visited) == 0 or np.linalg.norm(np.array(pt) - np.array(visited[-1])) > 0.8:
            visited.append(pt)
    return visited

def plot_trajectory_panel(ax, traj_df, sg_df=None, title="", is_baseline=False):
    draw_maze(ax)

    if traj_df.empty:
        ax.set_title(f"{title}\n(No data)", fontsize=11, fontweight="bold")
        return

    task_id = int(traj_df["task_id"].iloc[0])
    task_info = TASK_COORDS.get(task_id, {"init": (traj_df["x"].iloc[0], traj_df["y"].iloc[0]), "goal": (traj_df["x"].iloc[-1], traj_df["y"].iloc[-1])})

    xs = traj_df["x"].values
    ys = traj_df["y"].values
    steps = traj_df["step"].values

    # Plot ant path with color gradient
    scatter = ax.scatter(
        xs, ys, c=steps, cmap="plasma", s=14, alpha=0.85, zorder=3, label="Ant Path ($t=0..T$)"
    )
    ax.plot(xs, ys, color="#333333", alpha=0.35, linewidth=1.2, zorder=2)

    # Plot Start and Goal
    init_xy = task_info["init"]
    goal_xy = task_info["goal"]
    ax.scatter([init_xy[0]], [init_xy[1]], c="#2ca02c", s=130, marker="o", edgecolors="black", linewidths=1.5, zorder=6, label="Start ($s_0$)")
    ax.scatter([goal_xy[0]], [goal_xy[1]], c="#d62728", s=180, marker="*", edgecolors="black", linewidths=1.5, zorder=6, label="Goal ($g$)")

    # Plot Subgoals
    if sg_df is not None and not sg_df.empty:
        if is_baseline:
            sg_valid = sg_df.dropna(subset=["subgoal_x", "subgoal_y"])
            sg_sub = sg_valid.iloc[::max(1, len(sg_valid)//25)]
            if not sg_sub.empty:
                ax.scatter(sg_sub["subgoal_x"].values, sg_sub["subgoal_y"].values, c="#9467bd", s=40, marker="o", alpha=0.6, edgecolors="#4a148c", linewidths=1.0, zorder=4, label="High-Actor Intention")
                ax.plot(sg_sub["subgoal_x"].values, sg_sub["subgoal_y"].values, color="#9467bd", linestyle=":", alpha=0.4, linewidth=1.0, zorder=3)
        else:
            wps = extract_executed_waypoints(sg_df)
            if len(wps) > 0:
                wps_arr = np.array(wps)
                ax.plot(wps_arr[:, 0], wps_arr[:, 1], color="#1f77b4", linestyle="--", linewidth=1.8, alpha=0.8, zorder=4)
                ax.scatter(wps_arr[:, 0], wps_arr[:, 1], c="#1f77b4", s=110, marker="D", edgecolors="black", linewidths=1.5, zorder=5, label=f"Dijkstra Waypoints ({len(wps)})")

    reached = float(traj_df["reward"].max()) > 0.0 if "reward" in traj_df else False
    status_text = "SUCCESS" if reached else "FAILED"
    status_color = "#2ca02c" if reached else "#d62728"

    ax.set_title(f"{title}\nStatus: {status_text} ({len(traj_df)} steps)", fontsize=11, fontweight="bold", color=status_color)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)


def plot_landmarks_panel(ax, landmarks_df, sg_df=None, task_id=1):
    draw_maze(ax)
    task_info = TASK_COORDS.get(task_id, {"init": (0.0, 0.0), "goal": (20.0, 20.0)})

    if landmarks_df is not None and not landmarks_df.empty:
        ax.scatter(
            landmarks_df["x"].values,
            landmarks_df["y"].values,
            c="#7f7f7f",
            s=25,
            alpha=0.55,
            edgecolors="#333333",
            linewidths=0.5,
            zorder=3,
            label=f"Buffer Landmarks ($N={len(landmarks_df)}$)",
        )

    # Highlight complete executed Dijkstra path
    wps = extract_executed_waypoints(sg_df)
    if len(wps) > 0:
        wps_arr = np.array(wps)
        ax.plot(wps_arr[:, 0], wps_arr[:, 1], color="#ff7f0e", linestyle="-", linewidth=2.5, zorder=4, label=f"Shortest Dijkstra Path ({len(wps)} wps)")
        ax.scatter(wps_arr[:, 0], wps_arr[:, 1], c="#ff7f0e", s=120, marker="D", edgecolors="black", linewidths=1.5, zorder=5, label="Executed Waypoints")

    init_xy = task_info["init"]
    goal_xy = task_info["goal"]
    ax.scatter([init_xy[0]], [init_xy[1]], c="#2ca02c", s=130, marker="o", edgecolors="black", linewidths=1.5, zorder=6, label="Start ($s_0$)")
    ax.scatter([goal_xy[0]], [goal_xy[1]], c="#d62728", s=180, marker="*", edgecolors="black", linewidths=1.5, zorder=6, label="Goal ($g$)")

    ax.set_title(f"Buffer Landmarks & Graph Topology\nTask {task_id} Navigation Graph", fontsize=11, fontweight="bold", color="#1f77b4")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)


def main():
    parser = argparse.ArgumentParser(description="Visualize AntMaze Navigation Trajectories & Subgoals")
    parser.add_argument("--traj_file", type=str, default="results/trajectories.csv", help="Path to trajectories CSV")
    parser.add_argument("--sg_file", type=str, default="results/subgoals.csv", help="Path to subgoals CSV")
    parser.add_argument("--lm_file", type=str, default="results/landmarks.csv", help="Path to landmarks CSV")
    parser.add_argument("--method", type=str, default=None, help="Filter by method name")
    parser.add_argument("--seed", type=int, default=0, help="Filter by random seed")
    parser.add_argument("--task", type=int, default=1, help="Filter by task ID (1..5)")
    parser.add_argument("--episode", type=int, default=0, help="Filter by episode index (0..14)")
    parser.add_argument("--compare", action="store_true", help="Plot 3-panel comparison across Baseline, Dijkstra, and Landmarks")
    parser.add_argument("--save_path", type=str, default=None, help="Custom output image path")

    args = parser.parse_args()

    if not os.path.exists(args.traj_file):
        print(f"Trajectories file {args.traj_file} not found. Please run scripts/run_benchmark.py first.")
        return

    df_traj = pd.read_csv(args.traj_file)
    df_sg = pd.read_csv(args.sg_file) if os.path.exists(args.sg_file) else None
    df_lm = pd.read_csv(args.lm_file) if os.path.exists(args.lm_file) else None

    os.makedirs("results/plots", exist_ok=True)

    if args.compare:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=150)

        # 1. Baseline Panel
        b_name = "Single-Intention Baseline"
        sub_traj_b = df_traj[(df_traj["method"] == b_name) & (df_traj["seed"] == args.seed) & (df_traj["task_id"] == args.task) & (df_traj["episode"] == args.episode)]
        sub_sg_b = df_sg[(df_sg["method"] == b_name) & (df_sg["seed"] == args.seed) & (df_sg["task_id"] == args.task) & (df_sg["episode"] == args.episode)] if df_sg is not None else None
        plot_trajectory_panel(axes[0], sub_traj_b, sub_sg_b, title=f"{b_name} (Seed {args.seed})", is_baseline=True)

        # 2. Dijkstra Panel
        d_name = "Buffer Graph Dijkstra (Branch 2)"
        sub_traj_d = df_traj[(df_traj["method"] == d_name) & (df_traj["seed"] == args.seed) & (df_traj["task_id"] == args.task) & (df_traj["episode"] == args.episode)]
        sub_sg_d = df_sg[(df_sg["method"] == d_name) & (df_sg["seed"] == args.seed) & (df_sg["task_id"] == args.task) & (df_sg["episode"] == args.episode)] if df_sg is not None else None
        plot_trajectory_panel(axes[1], sub_traj_d, sub_sg_d, title=f"{d_name} (Seed {args.seed})", is_baseline=False)

        # 3. Landmarks Panel
        plot_landmarks_panel(axes[2], df_lm, sub_sg_d, task_id=args.task)

        save_file = args.save_path or f"results/plots/compare_3panel_seed{args.seed}_task{args.task}_ep{args.episode}.png"
        plt.tight_layout()
        plt.savefig(save_file, bbox_inches="tight")
        plt.close()
        print(f"Saved 3-panel comparison plot to {save_file}")

    else:
        target_method = args.method or df_traj["method"].iloc[0]
        is_b = "Baseline" in target_method
        sub_traj = df_traj[(df_traj["method"] == target_method) & (df_traj["seed"] == args.seed) & (df_traj["task_id"] == args.task) & (df_traj["episode"] == args.episode)]
        sub_sg = df_sg[(df_sg["method"] == target_method) & (df_sg["seed"] == args.seed) & (df_sg["task_id"] == args.task) & (df_sg["episode"] == args.episode)] if df_sg is not None else None

        fig, ax = plt.subplots(figsize=(7, 7), dpi=150)
        plot_trajectory_panel(ax, sub_traj, sub_sg, title=f"{target_method} | Task {args.task} | Ep {args.episode} (Seed {args.seed})", is_baseline=is_b)

        clean_name = target_method.replace(" ", "_").replace("(", "").replace(")", "")
        save_file = args.save_path or f"results/plots/{clean_name}_seed{args.seed}_task{args.task}_ep{args.episode}.png"
        plt.tight_layout()
        plt.savefig(save_file, bbox_inches="tight")
        plt.close()
        print(f"Saved trajectory plot to {save_file}")

if __name__ == "__main__":
    main()
