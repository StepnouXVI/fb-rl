import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D

# ==============================================================================
# Maze Layout Definitions (OGBench AntMaze environments & custom extensions)
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

TELEPORT_INFO = {
    "teleport": {
        "teleport_in_ijs": [(4, 6), (5, 1)],
        "teleport_in_xys": [(20.0, 12.0), (0.0, 16.0)],
        "teleport_out_ijs": [(1, 7), (6, 1), (6, 10)],
        "teleport_out_xys": [(24.0, 0.0), (0.0, 20.0), (36.0, 20.0)],
        "teleport_radius": 1.0,
    }
}

TASK_COORDS = {
    "medium": {
        1: {"init": (0.0, 0.0), "goal": (20.0, 20.0), "name": "Task 1 (SW -> NE)"},
        2: {"init": (0.0, 20.0), "goal": (20.0, 0.0), "name": "Task 2 (NW -> SE)"},
        3: {"init": (8.0, 16.0), "goal": (4.0, 12.0), "name": "Task 3 (Center Navigate)"},
        4: {"init": (16.0, 20.0), "goal": (0.0, 20.0), "name": "Task 4 (E -> W Corridor)"},
        5: {"init": (20.0, 4.0), "goal": (0.0, 0.0), "name": "Task 5 (SE -> SW Corner)"},
    },
    "large": {
        1: {"init": (0.0, 0.0), "goal": (36.0, 24.0), "name": "Task 1 (SW -> NE Corner)"},
        2: {"init": (12.0, 16.0), "goal": (0.0, 24.0), "name": "Task 2 (Center -> NW Corner)"},
        3: {"init": (12.0, 24.0), "goal": (36.0, 0.0), "name": "Task 3 (NW Center -> SE Corner)"},
        4: {"init": (28.0, 8.0), "goal": (12.0, 16.0), "name": "Task 4 (East -> Center)"},
        5: {"init": (0.0, 0.0), "goal": (12.0, 16.0), "name": "Task 5 (SW -> Center)"},
    },
    "giant": {
        1: {"init": (0.0, 0.0), "goal": (52.0, 36.0), "name": "Task 1 (SW -> NE Corner)"},
        2: {"init": (52.0, 0.0), "goal": (0.0, 36.0), "name": "Task 2 (SE -> NW Corner)"},
        3: {"init": (52.0, 28.0), "goal": (0.0, 0.0), "name": "Task 3 (East -> SW Corner)"},
        4: {"init": (8.0, 28.0), "goal": (44.0, 16.0), "name": "Task 4 (West -> East)"},
        5: {"init": (32.0, 16.0), "goal": (28.0, 8.0), "name": "Task 5 (Center -> South-Center)"},
    },
    "teleport": {
        1: {"init": (36.0, 0.0), "goal": (0.0, 24.0), "name": "Task 1 (East -> NW via Portal)"},
        2: {"init": (0.0, 0.0), "goal": (36.0, 24.0), "name": "Task 2 (SW -> NE via Portal)"},
        3: {"init": (20.0, 16.0), "goal": (36.0, 24.0), "name": "Task 3 (Center -> NE via Portal)"},
        4: {"init": (0.0, 24.0), "goal": (36.0, 24.0), "name": "Task 4 (NW -> NE via Portal)"},
        5: {"init": (20.0, 16.0), "goal": (0.0, 24.0), "name": "Task 5 (Center -> NW via Portal)"},
    },
}


def normalize_maze_type(maze_type):
    """Converts environment names or aliases into a canonical maze_type string."""
    if maze_type is None:
        return "medium"
    s = str(maze_type).lower().strip()
    s = s.replace("antmaze-", "").replace("pointmaze-", "").replace("humanoidmaze-", "")
    s = s.replace("-navigate-v0", "").replace("-singletask-v0", "").replace("-v0", "")
    return s if s in MAZE_LAYOUTS else ("teleport" if "teleport" in s else ("giant" if "giant" in s else ("large" if "large" in s else "medium")))


def auto_detect_maze_type(df_traj):
    """Infers maze type from trajectory coordinate bounds, metadata columns, and teleport jumps."""
    if df_traj is None or df_traj.empty:
        return "medium"

    for col in ["split", "env", "env_name", "maze_type"]:
        if col in df_traj.columns and not df_traj[col].isna().all():
            val = str(df_traj[col].iloc[0])
            norm = normalize_maze_type(val)
            if norm in MAZE_LAYOUTS:
                return norm

    max_x = df_traj["x"].max()
    max_y = df_traj["y"].max()
    max_coord = max(max_x, max_y)

    has_teleport_jumps = False
    for _, group in df_traj.groupby(["method", "seed", "task_id", "episode"]):
        if len(group) > 1:
            if "step" in group.columns:
                group = group.sort_values("step")
            pts = group[["x", "y"]].values
            dists = np.linalg.norm(np.diff(pts, axis=0), axis=1)
            if np.any(dists > 6.0):
                has_teleport_jumps = True
                break

    if max_coord > 38.0:
        return "giant"
    elif max_coord <= 22.0:
        return "medium"
    elif has_teleport_jumps:
        return "teleport"
    else:
        return "large"


def get_task_info(task_id, maze_type="medium", traj_df=None):
    """Retrieves standard task metadata (init, goal, label) for a given maze type."""
    norm_type = normalize_maze_type(maze_type)
    tasks_for_maze = TASK_COORDS.get(norm_type, TASK_COORDS["medium"])

    if task_id in tasks_for_maze:
        return tasks_for_maze[task_id]

    if traj_df is not None and not traj_df.empty:
        init_xy = (float(traj_df["x"].iloc[0]), float(traj_df["y"].iloc[0]))
        goal_xy = (float(traj_df["x"].iloc[-1]), float(traj_df["y"].iloc[-1]))
        return {"init": init_xy, "goal": goal_xy, "name": f"Task {task_id}"}

    return {"init": (0.0, 0.0), "goal": (20.0, 20.0), "name": f"Task {task_id}"}


def draw_maze(
    ax,
    maze_type="medium",
    unit_size=4.0,
    offset_x=4.0,
    offset_y=4.0,
    custom_grid=None,
    show_portals=True,
    show_portal_links=True,
):
    """
    Renders 2D maze geometry, walls, coordinate bounds, and teleport portals.
    Supports any grid dimensions (H x W), custom arrays, and OGBench teleportation specs.
    """
    if custom_grid is not None:
        grid = np.asarray(custom_grid)
    else:
        norm_type = normalize_maze_type(maze_type)
        grid = MAZE_LAYOUTS.get(norm_type, MAZE_LAYOUTS["medium"])

    h, w = grid.shape
    for i in range(h):
        for j in range(w):
            if grid[i, j] == 1:
                x = j * unit_size - offset_x - unit_size / 2.0
                y = i * unit_size - offset_y - unit_size / 2.0
                rect = patches.Rectangle(
                    (x, y), unit_size, unit_size, linewidth=1, edgecolor="#2b2b2b", facecolor="#d0d0d0", zorder=1
                )
                ax.add_patch(rect)

    norm_type = normalize_maze_type(maze_type)
    t_info = TELEPORT_INFO.get(norm_type, None)
    if show_portals and t_info is not None:
        in_xys = t_info.get("teleport_in_xys", [])
        out_xys = t_info.get("teleport_out_xys", [])
        radius = float(t_info.get("teleport_radius", 1.0))

        # Render subtle curved directed links connecting portal IN to OUT
        if show_portal_links and in_xys and out_xys:
            for in_xy in in_xys:
                for out_xy in out_xys:
                    link = patches.FancyArrowPatch(
                        in_xy,
                        out_xy,
                        connectionstyle="arc3,rad=0.15",
                        color="#ab47bc",
                        linestyle="--",
                        linewidth=1.1,
                        alpha=0.35,
                        arrowstyle="->,head_width=3,head_length=5",
                        zorder=2,
                    )
                    ax.add_patch(link)

        # Inbound Portals (Cyan glow, dashed trigger ring)
        for idx, (px, py) in enumerate(in_xys):
            trigger_circle = patches.Circle(
                (px, py),
                radius * 1.5,
                fill=True,
                facecolor="#00e5ff",
                alpha=0.22,
                edgecolor="#00bcd4",
                linestyle="--",
                linewidth=1.5,
                zorder=2,
            )
            ax.add_patch(trigger_circle)
            core_circle = patches.Circle(
                (px, py),
                radius * 0.75,
                fill=True,
                facecolor="#00e5ff",
                edgecolor="#006064",
                linewidth=1.8,
                alpha=0.85,
                zorder=3,
            )
            ax.add_patch(core_circle)
            ax.text(
                px,
                py,
                f"{idx+1}$",
                ha="center",
                va="center",
                fontsize=7.5,
                fontweight="bold",
                color="#00363a",
                zorder=4,
            )

        # Outbound Portals (Amber glow, destination ring)
        for idx, (px, py) in enumerate(out_xys):
            dest_circle = patches.Circle(
                (px, py),
                radius * 1.2,
                fill=True,
                facecolor="#ff9100",
                alpha=0.22,
                edgecolor="#ff6d00",
                linestyle=":",
                linewidth=1.5,
                zorder=2,
            )
            ax.add_patch(dest_circle)
            core_circle = patches.Circle(
                (px, py),
                radius * 0.75,
                fill=True,
                facecolor="#ff9100",
                edgecolor="#bf360c",
                linewidth=1.8,
                alpha=0.85,
                zorder=3,
            )
            ax.add_patch(core_circle)
            ax.text(
                px,
                py,
                f"{idx+1}$",
                ha="center",
                va="center",
                fontsize=7.5,
                fontweight="bold",
                color="#3e2723",
                zorder=4,
            )

    x_min = -offset_x - unit_size / 2.0
    x_max = (w - 1) * unit_size - offset_x + unit_size / 2.0
    y_min = -offset_y - unit_size / 2.0
    y_max = (h - 1) * unit_size - offset_y + unit_size / 2.0

    ax.set_xlim(x_min - 0.5, x_max + 0.5)
    ax.set_ylim(y_min - 0.5, y_max + 0.5)
    ax.set_aspect("equal")
    ax.set_facecolor("#f9f9f9")
    ax.grid(True, linestyle=":", alpha=0.4, color="#999999")


def plot_path_with_teleports(
    ax, xs, ys, jump_threshold=5.0, color="#333333", alpha=0.35, linewidth=1.2, zorder=2, label=None
):
    """
    Plots trajectory lines, breaking cleanly at teleport jumps to avoid drawing
    lines through walls, and highlighting teleport jumps with styled curved dashed arrows.
    """
    if len(xs) == 0:
        return
    pts = np.column_stack([xs, ys])
    if len(pts) <= 1:
        ax.plot(xs, ys, color=color, alpha=alpha, linewidth=linewidth, zorder=zorder, label=label)
        return

    dists = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    jump_indices = np.where(dists > jump_threshold)[0]

    seg_starts = [0] + list(jump_indices + 1)
    seg_ends = list(jump_indices + 1) + [len(pts)]

    has_labeled = False
    for start, end in zip(seg_starts, seg_ends):
        if end > start:
            lbl = label if (not has_labeled and label is not None) else None
            ax.plot(xs[start:end], ys[start:end], color=color, alpha=alpha, linewidth=linewidth, zorder=zorder, label=lbl)
            has_labeled = True

    # Render teleport jump connections
    for jump_idx in jump_indices:
        p_from = pts[jump_idx]
        p_to = pts[jump_idx + 1]
        arrow = patches.FancyArrowPatch(
            p_from,
            p_to,
            connectionstyle="arc3,rad=0.2",
            color="#d81b60",
            linestyle=":",
            linewidth=1.8,
            alpha=0.75,
            arrowstyle="->,head_width=3.5,head_length=5.5",
            zorder=zorder + 1,
        )
        ax.add_patch(arrow)


def extract_executed_waypoints(sg_df):
    """Extracts sequentially visited Dijkstra waypoints or full planned chain from subgoal history."""
    if sg_df is None or sg_df.empty:
        return []

    # Prioritize exact planned Dijkstra sequence if logged in planned_waypoints column
    if "planned_waypoints" in sg_df.columns:
        pw_series = sg_df["planned_waypoints"].dropna()
        if not pw_series.empty:
            for val in pw_series:
                try:
                    pts = json.loads(val) if isinstance(val, str) else val
                    if isinstance(pts, list) and len(pts) >= 2:
                        return [[float(p[0]), float(p[1])] for p in pts]
                except Exception:
                    pass

    sg_valid = sg_df.dropna(subset=["subgoal_x", "subgoal_y"])
    visited = []
    for _, row in sg_valid.iterrows():
        pt = [float(row["subgoal_x"]), float(row["subgoal_y"])]
        if len(visited) == 0 or np.linalg.norm(np.array(pt) - np.array(visited[-1])) > 0.8:
            visited.append(pt)
    return visited


def _add_portal_legend_handles(ax, maze_type, show_portals=True, show_portal_links=True, fontsize=8):
    """Appends portal proxy artists to the axis legend if portals exist on the map."""
    norm_type = normalize_maze_type(maze_type)
    t_info = TELEPORT_INFO.get(norm_type, None)
    if show_portals and t_info is not None:
        handles, labels = ax.get_legend_handles_labels()
        handles.append(
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#00e5ff", markeredgecolor="#006064", markersize=8, label="Portal IN")
        )
        handles.append(
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#ff9100", markeredgecolor="#bf360c", markersize=8, label="Portal OUT")
        )
        if show_portal_links:
            handles.append(Line2D([0], [0], color="#ab47bc", linestyle="--", linewidth=1.2, label="Portal Route"))
        ax.legend(handles=handles, loc="upper left", fontsize=fontsize, framealpha=0.9)


def plot_trajectory_panel(
    ax, traj_df, sg_df=None, title="", is_baseline=False, maze_type="medium", unit_size=4.0, show_portals=True, show_portal_links=True
):
    """Renders a single trajectory panel with ant positions, start/goal markers, and subgoals."""
    draw_maze(
        ax,
        maze_type=maze_type,
        unit_size=unit_size,
        show_portals=show_portals,
        show_portal_links=show_portal_links,
    )

    if traj_df is None or traj_df.empty:
        ax.set_title(f"{title}\n(No data)", fontsize=11, fontweight="bold")
        return False

    task_id = int(traj_df["task_id"].iloc[0])
    task_info = get_task_info(task_id, maze_type=maze_type, traj_df=traj_df)

    xs = traj_df["x"].values
    ys = traj_df["y"].values
    steps = traj_df["step"].values

    # Plot ant path with color gradient and teleport breaks
    ax.scatter(xs, ys, c=steps, cmap="plasma", s=14, alpha=0.85, zorder=3, label="Ant Path (t=0..T)")
    plot_path_with_teleports(ax, xs, ys, jump_threshold=5.0, color="#333333", alpha=0.35, linewidth=1.2, zorder=2)

    # Plot Start and Goal
    init_xy = task_info["init"]
    goal_xy = task_info["goal"]
    ax.scatter([init_xy[0]], [init_xy[1]], c="#2ca02c", s=130, marker="o", edgecolors="black", linewidths=1.5, zorder=6, label="Start (s0)")
    ax.scatter([goal_xy[0]], [goal_xy[1]], c="#d62728", s=180, marker="*", edgecolors="black", linewidths=1.5, zorder=6, label="Goal (g)")

    # Plot Subgoals
    if sg_df is not None and not sg_df.empty:
        if is_baseline:
            sg_valid = sg_df.dropna(subset=["subgoal_x", "subgoal_y"])
            sg_sub = sg_valid.iloc[::max(1, len(sg_valid) // 25)]
            if not sg_sub.empty:
                ax.scatter(
                    sg_sub["subgoal_x"].values,
                    sg_sub["subgoal_y"].values,
                    c="#9467bd",
                    s=40,
                    marker="o",
                    alpha=0.6,
                    edgecolors="#4a148c",
                    linewidths=1.0,
                    zorder=4,
                    label="High-Actor Intention",
                )
                ax.plot(
                    sg_sub["subgoal_x"].values,
                    sg_sub["subgoal_y"].values,
                    color="#9467bd",
                    linestyle=":",
                    alpha=0.4,
                    linewidth=1.0,
                    zorder=3,
                )
        else:
            wps = extract_executed_waypoints(sg_df)
            if len(wps) > 0:
                wps_arr = np.array(wps)
                plot_path_with_teleports(ax, wps_arr[:, 0], wps_arr[:, 1], jump_threshold=5.0, color="#1f77b4", alpha=0.8, linewidth=1.8, zorder=4)
                ax.scatter(
                    wps_arr[:, 0],
                    wps_arr[:, 1],
                    c="#1f77b4",
                    s=110,
                    marker="D",
                    edgecolors="black",
                    linewidths=1.5,
                    zorder=5,
                    label=f"Dijkstra Waypoints ({len(wps)})",
                )

    reached = float(traj_df["reward"].max()) > 0.0 if "reward" in traj_df else False
    status_text = "SUCCESS" if reached else "FAILED"
    status_color = "#2ca02c" if reached else "#d62728"

    ax.set_title(f"{title}\nStatus: {status_text} ({len(traj_df)} steps)", fontsize=11, fontweight="bold", color=status_color)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    _add_portal_legend_handles(ax, maze_type, show_portals=show_portals, show_portal_links=show_portal_links, fontsize=8)
    return reached


def plot_landmarks_panel(
    ax, landmarks_df, sg_df=None, task_id=1, maze_type="medium", unit_size=4.0, show_portals=True, show_portal_links=True
):
    """Renders sampled buffer landmarks and graph topology for a task."""
    draw_maze(
        ax,
        maze_type=maze_type,
        unit_size=unit_size,
        show_portals=show_portals,
        show_portal_links=show_portal_links,
    )
    task_info = get_task_info(task_id, maze_type=maze_type)

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
            label=f"Buffer Landmarks (N={len(landmarks_df)})",
        )

    wps = extract_executed_waypoints(sg_df)
    if len(wps) > 0:
        wps_arr = np.array(wps)
        plot_path_with_teleports(ax, wps_arr[:, 0], wps_arr[:, 1], jump_threshold=5.0, color="#ff7f0e", alpha=0.9, linewidth=2.5, zorder=4)
        ax.scatter(
            wps_arr[:, 0],
            wps_arr[:, 1],
            c="#ff7f0e",
            s=120,
            marker="D",
            edgecolors="black",
            linewidths=1.5,
            zorder=5,
            label=f"Executed Waypoints ({len(wps)})",
        )

    init_xy = task_info["init"]
    goal_xy = task_info["goal"]
    ax.scatter([init_xy[0]], [init_xy[1]], c="#2ca02c", s=130, marker="o", edgecolors="black", linewidths=1.5, zorder=6, label="Start (s0)")
    ax.scatter([goal_xy[0]], [goal_xy[1]], c="#d62728", s=180, marker="*", edgecolors="black", linewidths=1.5, zorder=6, label="Goal (g)")

    ax.set_title(
        f"Buffer Landmarks & Graph Topology\nTask {task_id} Navigation Graph ({maze_type.capitalize()})",
        fontsize=11,
        fontweight="bold",
        color="#1f77b4",
    )
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    _add_portal_legend_handles(ax, maze_type, show_portals=show_portals, show_portal_links=show_portal_links, fontsize=8)


def plot_multi_episode_panel(
    ax, traj_df, sg_df=None, title="", is_baseline=False, maze_type="medium", unit_size=4.0, show_portals=True, show_portal_links=True
):
    """Overlays multiple evaluation episodes for a given task and seed."""
    draw_maze(
        ax,
        maze_type=maze_type,
        unit_size=unit_size,
        show_portals=show_portals,
        show_portal_links=show_portal_links,
    )
    if traj_df is None or traj_df.empty:
        ax.set_title(f"{title}\n(No data)", fontsize=11, fontweight="bold")
        return

    task_id = int(traj_df["task_id"].iloc[0])
    task_info = get_task_info(task_id, maze_type=maze_type, traj_df=traj_df)

    episodes = traj_df["episode"].unique()
    success_count = 0
    total_steps = []

    for ep in episodes:
        ep_df = traj_df[traj_df["episode"] == ep].sort_values("step")
        if ep_df.empty:
            continue
        xs = ep_df["x"].values
        ys = ep_df["y"].values
        reached = float(ep_df["reward"].max()) > 0.0
        if reached:
            success_count += 1
        total_steps.append(len(ep_df))

        color = "#1f77b4" if reached else "#d62728"
        alpha = 0.55 if len(episodes) > 1 else 0.85
        plot_path_with_teleports(ax, xs, ys, jump_threshold=5.0, color=color, alpha=alpha, linewidth=1.5, zorder=3)

    init_xy = task_info["init"]
    goal_xy = task_info["goal"]
    ax.scatter([init_xy[0]], [init_xy[1]], c="#2ca02c", s=130, marker="o", edgecolors="black", linewidths=1.5, zorder=6, label="Start (s0)")
    ax.scatter([goal_xy[0]], [goal_xy[1]], c="#d62728", s=180, marker="*", edgecolors="black", linewidths=1.5, zorder=6, label="Goal (g)")

    if sg_df is not None and not sg_df.empty and not is_baseline:
        wps = extract_executed_waypoints(sg_df)
        if len(wps) > 0:
            wps_arr = np.array(wps)
            plot_path_with_teleports(ax, wps_arr[:, 0], wps_arr[:, 1], jump_threshold=5.0, color="#ff7f0e", alpha=0.8, linewidth=2.0, zorder=4)
            ax.scatter(
                wps_arr[:, 0],
                wps_arr[:, 1],
                c="#ff7f0e",
                s=90,
                marker="D",
                edgecolors="black",
                linewidths=1.2,
                zorder=5,
                label=f"Waypoints ({len(wps)})",
            )

    succ_rate = (success_count / max(1, len(episodes))) * 100.0
    mean_len = np.mean(total_steps) if total_steps else 0.0
    status_color = "#2ca02c" if succ_rate >= 70.0 else ("#ff7f0e" if succ_rate >= 40.0 else "#d62728")

    ax.set_title(
        f"{title}\nSuccess: {succ_rate:.1f}% ({success_count}/{len(episodes)} eps) | Mean Steps: {mean_len:.0f}",
        fontsize=10,
        fontweight="bold",
        color=status_color,
    )
    ax.legend(loc="upper left", fontsize=7, framealpha=0.9)
    _add_portal_legend_handles(ax, maze_type, show_portals=show_portals, show_portal_links=show_portal_links, fontsize=7)


def plot_overview_grid(
    df_traj,
    df_sg=None,
    seed=0,
    maze_type="medium",
    unit_size=4.0,
    save_path=None,
    show_portals=True,
    show_portal_links=True,
):
    """Renders a complete grid overview covering all tasks and methods for the given seed."""
    norm_type = normalize_maze_type(maze_type)
    methods = df_traj["method"].unique()
    tasks = sorted(df_traj["task_id"].unique())
    n_methods = len(methods)
    n_tasks = len(tasks)

    grid = MAZE_LAYOUTS.get(norm_type, MAZE_LAYOUTS["medium"])
    h, w = grid.shape
    aspect = w / max(1, h)
    col_w = max(4.5, 4.5 * aspect)
    row_h = 4.5

    fig, axes = plt.subplots(n_methods, n_tasks, figsize=(col_w * n_tasks, row_h * n_methods), dpi=150)
    if n_methods == 1 and n_tasks == 1:
        axes = np.array([[axes]])
    elif n_methods == 1:
        axes = axes[None, :]
    elif n_tasks == 1:
        axes = axes[:, None]

    for m_idx, method in enumerate(methods):
        is_b = "Baseline" in method
        for t_idx, task_id in enumerate(tasks):
            ax = axes[m_idx, t_idx]
            sub_traj = df_traj[(df_traj["method"] == method) & (df_traj["seed"] == seed) & (df_traj["task_id"] == task_id)]
            sub_sg = (
                df_sg[(df_sg["method"] == method) & (df_sg["seed"] == seed) & (df_sg["task_id"] == task_id)]
                if df_sg is not None
                else None
            )
            plot_multi_episode_panel(
                ax,
                sub_traj,
                sub_sg,
                title=f"{method} | Task {task_id}",
                is_baseline=is_b,
                maze_type=norm_type,
                unit_size=unit_size,
                show_portals=show_portals,
                show_portal_links=show_portal_links,
            )

    plt.suptitle(f"Overview of All Trajectories on {norm_type.capitalize()} Maze (Seed {seed})", fontsize=14, fontweight="bold", y=1.002)
    plt.tight_layout()
    out_file = save_path or f"results/plots/{norm_type}/overview_{norm_type}_seed{seed}.png"
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    plt.savefig(out_file, bbox_inches="tight")
    plt.close()
    print(f"Saved complete multi-trajectory overview grid to {out_file}")


def render_3panel_comparison(
    df_traj,
    df_sg=None,
    df_lm=None,
    seed=0,
    task_id=1,
    episode=0,
    maze_type="medium",
    unit_size=4.0,
    show_portals=True,
    show_portal_links=True,
    target_method="Buffer Graph Dijkstra (Branch 2)",
    baseline_method="Single-Intention Baseline",
    output_dir="results/plots",
    save_path=None,
    organize_by_outcome=True,
):
    """
    Renders a 3-panel comparison (Baseline, Planner, Landmarks) and organizes
    the resulting image into success/ or failed/ subfolders based on trial outcome.
    """
    norm_type = normalize_maze_type(maze_type)
    grid = MAZE_LAYOUTS.get(norm_type, MAZE_LAYOUTS["medium"])
    h, w = grid.shape
    aspect = w / max(1, h)
    fig_w = max(6.5, 6.5 * aspect)
    fig_h = 6.5

    fig, axes = plt.subplots(1, 3, figsize=(fig_w * 2.6, fig_h), dpi=150)

    # 1. Baseline Panel
    b_name = baseline_method
    sub_traj_b = df_traj[
        (df_traj["method"] == b_name)
        & (df_traj["seed"] == seed)
        & (df_traj["task_id"] == task_id)
        & (df_traj["episode"] == episode)
    ]
    sub_sg_b = (
        df_sg[
            (df_sg["method"] == b_name)
            & (df_sg["seed"] == seed)
            & (df_sg["task_id"] == task_id)
            & (df_sg["episode"] == episode)
        ]
        if df_sg is not None
        else None
    )
    plot_trajectory_panel(
        axes[0],
        sub_traj_b,
        sub_sg_b,
        title=f"{b_name} (Seed {seed})",
        is_baseline=True,
        maze_type=norm_type,
        unit_size=unit_size,
        show_portals=show_portals,
        show_portal_links=show_portal_links,
    )

    # 2. Target Planner Panel
    d_name = target_method
    sub_traj_d = df_traj[
        (df_traj["method"] == d_name)
        & (df_traj["seed"] == seed)
        & (df_traj["task_id"] == task_id)
        & (df_traj["episode"] == episode)
    ]
    sub_sg_d = (
        df_sg[
            (df_sg["method"] == d_name)
            & (df_sg["seed"] == seed)
            & (df_sg["task_id"] == task_id)
            & (df_sg["episode"] == episode)
        ]
        if df_sg is not None
        else None
    )
    d_success = plot_trajectory_panel(
        axes[1],
        sub_traj_d,
        sub_sg_d,
        title=f"{d_name} (Seed {seed})",
        is_baseline=False,
        maze_type=norm_type,
        unit_size=unit_size,
        show_portals=show_portals,
        show_portal_links=show_portal_links,
    )

    # 3. Landmarks Panel
    plot_landmarks_panel(
        axes[2],
        df_lm,
        sub_sg_d,
        task_id=task_id,
        maze_type=norm_type,
        unit_size=unit_size,
        show_portals=show_portals,
        show_portal_links=show_portal_links,
    )

    # Determine outcome based on evaluated planner (or baseline fallback)
    if not sub_traj_d.empty:
        outcome = "success" if d_success else "failed"
    elif not sub_traj_b.empty:
        outcome = "success" if float(sub_traj_b["reward"].max()) > 0.0 else "failed"
    else:
        outcome = "failed"

    if save_path:
        out_file = save_path
    elif organize_by_outcome:
        out_dir = os.path.join(output_dir, norm_type, outcome)
        os.makedirs(out_dir, exist_ok=True)
        out_file = os.path.join(out_dir, f"compare_3panel_{norm_type}_seed{seed}_task{task_id}_ep{episode}.png")
    else:
        out_dir = os.path.join(output_dir, norm_type)
        os.makedirs(out_dir, exist_ok=True)
        out_file = os.path.join(out_dir, f"compare_3panel_{norm_type}_seed{seed}_task{task_id}_ep{episode}.png")

    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_file, bbox_inches="tight")
    plt.close()
    return out_file, outcome


def export_all_comparisons(
    df_traj,
    df_sg=None,
    df_lm=None,
    maze_type="medium",
    unit_size=4.0,
    show_portals=True,
    show_portal_links=True,
    output_dir="results/plots",
    seed_filter=None,
    task_filter=None,
):
    """
    Iterates over all episodes in the trajectories dataset, renders 3-panel comparisons,
    and automatically classifies each into success/ or failed/ subfolders under the maze category.
    """
    norm_type = normalize_maze_type(maze_type)
    episodes_meta = df_traj[["seed", "task_id", "episode"]].drop_duplicates().sort_values(["seed", "task_id", "episode"])

    if seed_filter is not None:
        episodes_meta = episodes_meta[episodes_meta["seed"] == seed_filter]
    if task_filter is not None:
        episodes_meta = episodes_meta[episodes_meta["task_id"] == task_filter]

    total = len(episodes_meta)
    print(f"Exporting {total} 3-panel comparisons organized into '{output_dir}/{norm_type}/{{success,failed}}/'...")

    counts = {"success": 0, "failed": 0}
    for _, row in episodes_meta.iterrows():
        s = int(row["seed"])
        t = int(row["task_id"])
        ep = int(row["episode"])
        out_file, outcome = render_3panel_comparison(
            df_traj,
            df_sg=df_sg,
            df_lm=df_lm,
            seed=s,
            task_id=t,
            episode=ep,
            maze_type=norm_type,
            unit_size=unit_size,
            show_portals=show_portals,
            show_portal_links=show_portal_links,
            output_dir=output_dir,
            organize_by_outcome=True,
        )
        counts[outcome] += 1

    print(f"Export completed: {counts['success']} SUCCESS, {counts['failed']} FAILED plots saved to {output_dir}/{norm_type}/")
    return counts


def main():
    parser = argparse.ArgumentParser(description="Visualize AntMaze Navigation Trajectories, Subgoals & Portals")
    parser.add_argument("--traj_file", type=str, default="results/trajectories.csv", help="Path to trajectories CSV")
    parser.add_argument("--sg_file", type=str, default="results/subgoals.csv", help="Path to subgoals CSV")
    parser.add_argument("--lm_file", type=str, default="results/landmarks.csv", help="Path to landmarks CSV")
    parser.add_argument("--output_dir", type=str, default="results/plots", help="Base directory to save plots into")
    parser.add_argument("--maze_type", "--maze", "--split", dest="maze_type", type=str, default="auto", help="Maze type (auto, medium, large, giant, teleport)")
    parser.add_argument("--unit_size", type=float, default=4.0, help="Maze grid cell size (default: 4.0)")
    parser.add_argument("--custom_layout", type=str, default=None, help="Path to custom JSON or text layout file")
    parser.add_argument("--no_portals", action="store_true", help="Disable rendering teleport portals")
    parser.add_argument("--no_portal_links", action="store_true", help="Disable rendering curved teleport link arrows")
    parser.add_argument("--method", type=str, default=None, help="Filter by method name")
    parser.add_argument("--seed", type=int, default=0, help="Filter by random seed")
    parser.add_argument("--task", type=int, default=1, help="Filter by task ID (1..5)")
    parser.add_argument("--episode", type=int, default=0, help="Filter by episode index (0..14)")
    parser.add_argument("--compare", action="store_true", help="Plot 3-panel comparison across Baseline, Dijkstra, and Landmarks")
    parser.add_argument("--export_comparisons", "--batch_compare", action="store_true", help="Export all 3-panel comparisons classified into success/ and failed/ folders")
    parser.add_argument("--all_episodes", action="store_true", help="Overlay all episodes for the specified task/seed")
    parser.add_argument("--overview", action="store_true", help="Generate an overview grid of ALL tasks and methods for the given seed")
    parser.add_argument("--save_path", type=str, default=None, help="Custom output image path")

    args = parser.parse_args()

    if not os.path.exists(args.traj_file):
        print(f"Trajectories file {args.traj_file} not found. Please run scripts/run_benchmark.py first.")
        return

    df_traj = pd.read_csv(args.traj_file)
    df_sg = pd.read_csv(args.sg_file) if os.path.exists(args.sg_file) else None
    df_lm = pd.read_csv(args.lm_file) if os.path.exists(args.lm_file) else None

    # Load custom layout if provided
    if args.custom_layout is not None and os.path.exists(args.custom_layout):
        try:
            if args.custom_layout.endswith(".json"):
                with open(args.custom_layout, "r") as f:
                    c_grid = np.array(json.load(f))
            else:
                c_grid = np.loadtxt(args.custom_layout, dtype=int)
            MAZE_LAYOUTS["custom"] = c_grid
            maze_type = "custom"
        except Exception as e:
            print(f"Failed to load custom layout from {args.custom_layout}: {e}")
            maze_type = "medium"
    elif args.maze_type.lower() == "auto":
        maze_type = auto_detect_maze_type(df_traj)
        print(f"Auto-detected maze environment type: '{maze_type}'")
    else:
        maze_type = normalize_maze_type(args.maze_type)

    show_portals = not args.no_portals
    show_portal_links = not args.no_portal_links
    base_out_dir = os.path.join(args.output_dir, maze_type)
    os.makedirs(base_out_dir, exist_ok=True)

    grid = MAZE_LAYOUTS.get(maze_type, MAZE_LAYOUTS["medium"])
    h, w = grid.shape
    aspect = w / max(1, h)
    fig_w = max(6.5, 6.5 * aspect)
    fig_h = 6.5

    if args.export_comparisons:
        export_all_comparisons(
            df_traj,
            df_sg=df_sg,
            df_lm=df_lm,
            maze_type=maze_type,
            unit_size=args.unit_size,
            show_portals=show_portals,
            show_portal_links=show_portal_links,
            output_dir=args.output_dir,
            seed_filter=args.seed if "--seed" in os.sys.argv else None,
            task_filter=args.task if "--task" in os.sys.argv else None,
        )
    elif args.overview:
        save_file = args.save_path or os.path.join(base_out_dir, f"overview_{maze_type}_seed{args.seed}.png")
        plot_overview_grid(
            df_traj,
            df_sg,
            seed=args.seed,
            maze_type=maze_type,
            unit_size=args.unit_size,
            save_path=save_file,
            show_portals=show_portals,
            show_portal_links=show_portal_links,
        )
    elif args.all_episodes:
        target_method = args.method or df_traj["method"].iloc[0]
        is_b = "Baseline" in target_method
        sub_traj = df_traj[(df_traj["method"] == target_method) & (df_traj["seed"] == args.seed) & (df_traj["task_id"] == args.task)]
        sub_sg = (
            df_sg[(df_sg["method"] == target_method) & (df_sg["seed"] == args.seed) & (df_sg["task_id"] == args.task)]
            if df_sg is not None
            else None
        )

        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=150)
        plot_multi_episode_panel(
            ax,
            sub_traj,
            sub_sg,
            title=f"{target_method} | Task {args.task} (All Episodes, Seed {args.seed})",
            is_baseline=is_b,
            maze_type=maze_type,
            unit_size=args.unit_size,
            show_portals=show_portals,
            show_portal_links=show_portal_links,
        )

        clean_name = target_method.replace(" ", "_").replace("(", "").replace(")", "")
        save_file = args.save_path or os.path.join(base_out_dir, f"{clean_name}_{maze_type}_seed{args.seed}_task{args.task}_all_episodes.png")
        os.makedirs(os.path.dirname(save_file), exist_ok=True)
        plt.tight_layout()
        plt.savefig(save_file, bbox_inches="tight")
        plt.close()
        print(f"Saved all-episodes trajectory overlay to {save_file}")
    elif args.compare:
        out_file, outcome = render_3panel_comparison(
            df_traj,
            df_sg=df_sg,
            df_lm=df_lm,
            seed=args.seed,
            task_id=args.task,
            episode=args.episode,
            maze_type=maze_type,
            unit_size=args.unit_size,
            show_portals=show_portals,
            show_portal_links=show_portal_links,
            output_dir=args.output_dir,
            save_path=args.save_path,
            organize_by_outcome=True,
        )
        print(f"Saved 3-panel comparison plot ({outcome.upper()}) to {out_file}")
    else:
        target_method = args.method or df_traj["method"].iloc[0]
        is_b = "Baseline" in target_method
        sub_traj = df_traj[(df_traj["method"] == target_method) & (df_traj["seed"] == args.seed) & (df_traj["task_id"] == args.task) & (df_traj["episode"] == args.episode)]
        sub_sg = (
            df_sg[(df_sg["method"] == target_method) & (df_sg["seed"] == args.seed) & (df_sg["task_id"] == args.task) & (df_sg["episode"] == args.episode)]
            if df_sg is not None
            else None
        )

        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=150)
        reached = plot_trajectory_panel(
            ax,
            sub_traj,
            sub_sg,
            title=f"{target_method} | Task {args.task} | Ep {args.episode} (Seed {args.seed})",
            is_baseline=is_b,
            maze_type=maze_type,
            unit_size=args.unit_size,
            show_portals=show_portals,
            show_portal_links=show_portal_links,
        )
        outcome = "success" if reached else "failed"

        clean_name = target_method.replace(" ", "_").replace("(", "").replace(")", "")
        if args.save_path:
            save_file = args.save_path
        else:
            out_folder = os.path.join(base_out_dir, outcome)
            os.makedirs(out_folder, exist_ok=True)
            save_file = os.path.join(out_folder, f"{clean_name}_{maze_type}_seed{args.seed}_task{args.task}_ep{args.episode}.png")

        os.makedirs(os.path.dirname(save_file), exist_ok=True)
        plt.tight_layout()
        plt.savefig(save_file, bbox_inches="tight")
        plt.close()
        print(f"Saved trajectory plot ({outcome.upper()}) to {save_file}")


if __name__ == "__main__":
    main()
