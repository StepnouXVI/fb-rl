import marimo

__generated_with = "0.24.2"
app = marimo.App(width="full", app_title="FB-RL AntMaze Benchmark Analytics")


@app.cell
def __():
    """Import core data manipulation and visualization libraries."""
    import json
    import os
    import sqlite3
    from typing import Any, Dict, List, Optional, Tuple
    import marimo as mo
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    import plotly.io as pio

    return (
        Any,
        Dict,
        List,
        Optional,
        Tuple,
        go,
        json,
        mo,
        np,
        os,
        pd,
        pio,
        sqlite3,
    )


@app.cell
def __(np):
    """Define maze grid matrices, DB paths, and visual color tokens."""
    maze_layouts = {
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
    }
    db_path = "results/data/telemetry.db"
    method_colors = [
        "#4C72B0",
        "#DD8452",
        "#55A868",
        "#C44E52",
        "#8172D5",
        "#CCB974",
    ]
    return db_path, maze_layouts, method_colors


@app.cell
def __(os, pd, sqlite3):
    """Database query function for loading benchmark episodes."""
    def load_split_episodes(db_file: str, split: str) -> pd.DataFrame:
        """Load episodes for specified split joined with run records."""
        if not os.path.exists(db_file):
            return pd.DataFrame()
        conn = sqlite3.connect(db_file)
        query = """
            SELECT r.method, r.split, e.episode_id, e.seed, e.task_id, e.episode_idx,
                   e.is_success, e.total_steps, e.mean_speed, e.self_intersections,
                   e.mean_cross_track_error, e.max_cross_track_error, e.mean_latency_ms
            FROM episodes e
            JOIN runs r ON e.run_id = r.run_id
            WHERE r.split = ?
            ORDER BY r.method, e.seed, e.task_id, e.episode_idx
        """
        df = pd.read_sql_query(query, conn, params=(split,))
        conn.close()
        if df.empty:
            return df
        counts = df["method"].value_counts()
        valid_methods = counts[counts >= 50].index.tolist()
        return df[df["method"].isin(valid_methods)].reset_index(drop=True)

    return load_split_episodes,


@app.cell
def __(Any, Dict, json, os, sqlite3):
    """Database query function for fetching individual episode telemetry."""
    def load_episode_details(db_file: str, episode_id: str) -> Dict[str, Any]:
        """Fetch planned path coordinates and step records for an episode."""
        if not os.path.exists(db_file):
            return {"path_coords": None, "steps": []}
        conn = sqlite3.connect(db_file)
        cur = conn.cursor()
        cur.execute(
            "SELECT path_coords_json FROM planned_paths WHERE episode_id = ? ORDER BY replan_step LIMIT 1",
            (episode_id,),
        )
        p_row = cur.fetchone()
        path_coords = json.loads(p_row[0]) if (p_row and p_row[0]) else None
        cur.execute(
            "SELECT x, y, speed, action_norm, attention_targets, attention_weights FROM steps WHERE episode_id = ? ORDER BY step_idx",
            (episode_id,),
        )
        s_rows = cur.fetchall()
        cur.close()
        conn.close()
        keys = ["x", "y", "speed", "action_norm", "attention_targets", "attention_weights"]
        steps = [dict(zip(keys, r)) for r in s_rows]
        return {"path_coords": path_coords, "steps": steps}

    return load_episode_details,


@app.cell
def __(Any, Dict, Tuple, np, pd):
    """Benchmark metric computation and aggregation."""
    def compute_benchmark_metrics(
        df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        """Aggregate per-method benchmark statistics and task breakdowns."""
        methods = sorted(df["method"].unique())
        main_rows, task_rows, bundle = [], [], {}
        for m in methods:
            g = df[df["method"] == m]
            seeds = sorted(g["seed"].unique())
            seed_srs = [float(g[g["seed"] == s]["is_success"].mean() * 100.0) for s in seeds]
            sr_mean = float(np.mean(seed_srs)) if seed_srs else 0.0
            sr_std = float(np.std(seed_srs)) if seed_srs else 0.0
            task_srs = [float(g[g["task_id"] == t]["is_success"].mean() * 100.0) for t in range(1, 6)]
            steps_mean = float(g["total_steps"].mean())
            spd_mean = float(g["mean_speed"].mean())
            lat_mean = float(g["mean_latency_ms"].mean())
            loops_mean = float(g["self_intersections"].mean())
            cte_vals = g[g["mean_cross_track_error"] > 0]["mean_cross_track_error"]
            cte_mean = float(cte_vals.mean()) if len(cte_vals) > 0 else 0.0
            main_rows.append({
                "Method": m,
                "Success Rate (%)": f"{sr_mean:.1f} ± {sr_std:.1f}",
                "Mean Steps": f"{steps_mean:.1f}",
                "Mean Speed (m/s)": f"{spd_mean:.2f}",
                "Latency (ms/step)": f"{lat_mean:.2f}",
                "Self-Intersections (Loops)": f"{loops_mean:.1f}",
                "Mean CTE (m)": f"{cte_mean:.2f}" if cte_mean > 0.0 else "—",
            })
            task_rows.append({
                "Method": m,
                "Task 1 (%)": f"{task_srs[0]:.1f}",
                "Task 2 (%)": f"{task_srs[1]:.1f}",
                "Task 3 (%)": f"{task_srs[2]:.1f}",
                "Task 4 (%)": f"{task_srs[3]:.1f}",
                "Task 5 (%)": f"{task_srs[4]:.1f}",
                "Overall SR (%)": f"{sr_mean:.1f}",
            })
            bundle[m] = {
                "sr_mean": sr_mean,
                "sr_std": sr_std,
                "lat_mean": lat_mean,
                "steps_mean": steps_mean,
                "spd_mean": spd_mean,
                "loops_mean": loops_mean,
                "cte_mean": cte_mean,
                "task_srs": task_srs,
            }
        return pd.DataFrame(main_rows), pd.DataFrame(task_rows), bundle

    return compute_benchmark_metrics,


@app.cell
def __(pd):
    """Episode filtering function for inspector queries."""
    def filter_episodes(
        df: pd.DataFrame, method: str, status: str, task: str, seed: str
    ) -> pd.DataFrame:
        """Filter episodes by method, outcome status, task id, and seed."""
        if df.empty:
            return df
        sub = df[df["method"] == method].copy() if method != "All" else df.copy()
        if status == "Success Only":
            sub = sub[sub["is_success"] == 1]
        elif status == "Failed Only":
            sub = sub[sub["is_success"] == 0]
        if task != "All":
            sub = sub[sub["task_id"] == int(task)]
        if seed != "All":
            sub = sub[sub["seed"] == int(seed)]
        return sub

    return filter_episodes,


@app.cell
def __(Any, Dict, List, Tuple):
    """Radar chart normalization computation."""
    def compute_radar_matrix(
        methods: List[str], bundle: Dict[str, Any]
    ) -> Tuple[List[str], Dict[str, List[float]]]:
        """Normalize 5 performance metrics into 0-100 scale for radar profile."""
        categories = [
            "Success Rate (%)",
            "Low Latency Score",
            "Step Efficiency (1/steps)",
            "Clean Trajectory (1/loops)",
            "Speed",
        ]
        min_lat = min(bundle[m]["lat_mean"] for m in methods)
        min_steps = min(bundle[m]["steps_mean"] for m in methods)
        max_spd = max(bundle[m]["spd_mean"] for m in methods)
        all_loops = [bundle[m]["loops_mean"] for m in methods]
        min_loops, max_loops = min(all_loops), max(all_loops)
        matrix = {}
        for m in methods:
            b = bundle[m]
            sr_score = min(100.0, max(0.0, b["sr_mean"]))
            lat_score = 100.0 * (min_lat / max(b["lat_mean"], 1e-6))
            step_score = 100.0 * (min_steps / max(b["steps_mean"], 1e-6))
            clean_score = (
                (20.0 + 80.0 * (max_loops - b["loops_mean"]) / (max_loops - min_loops))
                if max_loops > min_loops
                else 100.0
            )
            spd_score = 100.0 * (b["spd_mean"] / max(max_spd, 1e-6))
            matrix[m] = [sr_score, lat_score, step_score, clean_score, spd_score]
        return categories, matrix

    return compute_radar_matrix,


@app.cell
def __(Dict, List, go):
    """Radar chart figure builder."""
    def build_radar_figure(
        methods: List[str],
        categories: List[str],
        values_matrix: Dict[str, List[float]],
        colors: List[str],
    ) -> go.Figure:
        """Construct radar chart comparing methods across 5 performance axes."""
        fig = go.Figure()
        for i, m in enumerate(methods):
            vals = list(values_matrix[m]) + [values_matrix[m][0]]
            cats = list(categories) + [categories[0]]
            fig.add_trace(
                go.Scatterpolar(
                    r=vals,
                    theta=cats,
                    name=m,
                    fill="toself",
                    line=dict(color=colors[i % len(colors)], width=2),
                    opacity=0.4,
                )
            )
        fig.update_layout(
            title=dict(text="Radar Profile (5-Axis Architecture Performance)", font=dict(size=14, family="sans-serif"), x=0.0, y=0.98),
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 100], gridcolor="#E2E8F0", linecolor="#CBD5E0"),
                angularaxis=dict(gridcolor="#E2E8F0", linecolor="#CBD5E0"),
                bgcolor="#F8FAFC",
            ),
            paper_bgcolor="#FFFFFF",
            template="seaborn",
            width=680,
            height=520,
            legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="right", x=1.0),
            margin=dict(l=40, r=40, t=80, b=40),
        )
        return fig

    return build_radar_figure,


@app.cell
def __(Any, Dict, List, go):
    """Pareto frontier chart builder."""
    def build_pareto_figure(
        methods: List[str], bundle: Dict[str, Any], colors: List[str]
    ) -> go.Figure:
        """Construct Pareto frontier chart of latency versus success rate."""
        fig = go.Figure()
        for i, m in enumerate(methods):
            b = bundle[m]
            fig.add_trace(
                go.Scatter(
                    x=[b["lat_mean"]],
                    y=[b["sr_mean"]],
                    mode="markers",
                    name=m,
                    error_y=dict(type="data", array=[b["sr_std"]], visible=True),
                    marker=dict(size=15, color=colors[i % len(colors)], line=dict(color="#1A202C", width=1.5)),
                    text=[m],
                    customdata=[b["sr_std"]],
                    hovertemplate="<b>%{text}</b><br>Success Rate: %{y:.1f}% ± %{customdata:.1f}%<br>Latency: %{x:.2f} ms<extra></extra>",
                )
            )
        srs = [bundle[m]["sr_mean"] for m in methods]
        fig.update_layout(
            title=dict(text="Pareto Trade-Off: Latency vs Success Rate", font=dict(size=14, family="sans-serif"), x=0.0, y=0.98),
            xaxis=dict(title="Latency (ms/step)", gridcolor="#E2E8F0", linecolor="#CBD5E0"),
            yaxis=dict(title="Success Rate (%)", range=[max(0.0, min(srs) - 10.0), min(100.0, max(srs) + 10.0)], gridcolor="#E2E8F0", linecolor="#CBD5E0"),
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#F8FAFC",
            template="seaborn",
            width=680,
            height=520,
            legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="right", x=1.0),
            margin=dict(l=55, r=40, t=80, b=50),
        )
        return fig

    return build_pareto_figure,


@app.cell
def __(Any, Dict, List, Tuple, np):
    """Maze geometry calculation and bounding box extraction."""
    def get_maze_wall_shapes(
        split: str, maze_layouts: Dict[str, np.ndarray]
    ) -> Tuple[List[Dict[str, Any]], Tuple[float, float, float, float]]:
        """Generate Plotly layout shapes for AntMaze walls and compute view bounds."""
        grid = maze_layouts.get(split, maze_layouts["medium"])
        h, w = grid.shape
        unit_size, offset_x, offset_y = 4.0, 4.0, 4.0
        shapes = []
        for i in range(h):
            for j in range(w):
                if grid[i, j] == 1:
                    x0 = j * unit_size - offset_x - unit_size / 2.0
                    x1 = x0 + unit_size
                    y0 = i * unit_size - offset_y - unit_size / 2.0
                    y1 = y0 + unit_size
                    shapes.append(
                        dict(
                            type="rect",
                            x0=x0,
                            y0=y0,
                            x1=x1,
                            y1=y1,
                            fillcolor="#2D3748",
                            line=dict(color="#1A202C", width=1),
                            layer="below",
                        )
                    )
        x_min = -offset_x - unit_size / 2.0 - 1.0
        x_max = (w - 1) * unit_size - offset_x + unit_size / 2.0 + 1.0
        y_min = -offset_y - unit_size / 2.0 - 1.0
        y_max = (h - 1) * unit_size - offset_y + unit_size / 2.0 + 1.0
        return shapes, (x_min, x_max, y_min, y_max)

    return get_maze_wall_shapes,


@app.cell
def __(Any, Dict, List, go, json):
    """Helper for adding attention token overlays to trajectory plots."""
    def append_attention_trace(
        fig: go.Figure, traj_steps: List[Dict[str, Any]]
    ) -> None:
        """Add attention tokens with weights to Plotly figure if present."""
        if not traj_steps:
            return
        mid_idx = len(traj_steps) // 2
        raw_tgt = traj_steps[mid_idx].get("attention_targets")
        raw_w = traj_steps[mid_idx].get("attention_weights")
        if not raw_tgt or not raw_w:
            return
        try:
            tgts = json.loads(raw_tgt) if isinstance(raw_tgt, str) else raw_tgt
            wts = json.loads(raw_w) if isinstance(raw_w, str) else raw_w
            if tgts and wts:
                fig.add_trace(go.Scatter(
                    x=[t[0] for t in tgts], y=[t[1] for t in tgts], mode="markers", name="Attention Targets",
                    marker=dict(size=[max(6, int(w * 25)) for w in wts], color="#3182CE", opacity=0.7),
                ))
        except Exception:
            pass

    return append_attention_trace,


@app.cell
def __(go):
    """Base traces for trajectory figure with Dijkstra path, endpoints, and ant."""
    def add_trajectory_base_traces(
        fig: go.Figure, px: list, py: list, tx: list, ty: list, is_succ: bool, ant_col: str,
    ) -> None:
        """Populate base Scatter traces for Dijkstra path, start, goal, and initial ant position."""
        fig.add_trace(go.Scatter(
            x=px, y=py, mode="lines+markers", name="Dijkstra Path",
            line=dict(color="#DD8452", width=2, dash="dash"), marker=dict(size=4, color="#DD8452"),
        ))
        fig.add_trace(go.Scatter(
            x=[tx[0]], y=[ty[0]], mode="markers+text", name="Start",
            marker=dict(size=13, color="#2CA02C", symbol="circle"), text=["Start"], textposition="top center",
        ))
        sym, lbl = ("star", "Goal Reached") if is_succ else ("x", "Final Position")
        fig.add_trace(go.Scatter(
            x=[tx[-1]], y=[ty[-1]], mode="markers+text", name=lbl,
            marker=dict(size=14, color=ant_col, symbol=sym), text=[lbl], textposition="top center",
        ))
        fig.add_trace(go.Scatter(
            x=[tx[0]], y=[ty[0]], mode="lines", name="Ant Path", line=dict(color=ant_col, width=3.5),
        ))
        fig.add_trace(go.Scatter(
            x=[tx[0]], y=[ty[0]], mode="markers", name="Ant Head",
            marker=dict(size=14, color=ant_col, symbol="circle", line=dict(color="#1A202C", width=2)),
        ))

    return add_trajectory_base_traces,


@app.cell
def __(Any, Dict, List, Tuple):
    """Animation controls and timeline generator for Plotly playback."""
    def make_animation_menus(
        indices: List[int],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Generate Play/Pause buttons and step scrub slider for Plotly animation."""
        steps = [
            dict(
                args=[[str(i)], dict(frame=dict(duration=0, redraw=False), mode="immediate", transition=dict(duration=0))],
                label=str(i), method="animate",
            )
            for i in indices
        ]
        menu = [
            dict(
                type="buttons", direction="left", x=0.0, y=1.07, xanchor="left", yanchor="bottom",
                buttons=[
                    dict(label="▶ Play", method="animate", args=[None, dict(frame=dict(duration=50, redraw=True), fromcurrent=True, mode="immediate")]),
                    dict(label="⏸ Pause", method="animate", args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")]),
                ],
            )
        ]
        sliders = [
            dict(
                active=0, y=-0.08, x=0.0, len=1.0,
                currentvalue=dict(font=dict(size=11), prefix="Step: ", visible=True, xanchor="right"),
                steps=steps,
            )
        ]
        return menu, sliders

    return make_animation_menus,


@app.cell
def __(Any, go):
    """Build frames for Plotly trajectory animation."""
    def create_trajectory_frames(
        tx: list, ty: list, indices: list, ant_color: str,
    ) -> list:
        """Generate animation frames updating trajectory line and Ant head marker."""
        return [
            go.Frame(
                data=[
                    go.Scatter(x=tx[: idx + 1], y=ty[: idx + 1], mode="lines", line=dict(color=ant_color, width=3.5)),
                    go.Scatter(x=[tx[idx]], y=[ty[idx]], mode="markers", marker=dict(size=14, color=ant_color, symbol="circle", line=dict(color="#1A202C", width=2))),
                ],
                name=str(idx), traces=[3, 4],
            )
            for idx in indices
        ]

    return create_trajectory_frames,


@app.cell
def __(
    Any,
    Dict,
    List,
    Optional,
    add_trajectory_base_traces,
    append_attention_trace,
    create_trajectory_frames,
    get_maze_wall_shapes,
    go,
    make_animation_menus,
    np,
):
    """Animated trajectory figure builder with Play/Pause controls and slider scrubber."""
    def build_animated_trajectory_figure(
        split: str,
        method_name: str,
        path_coords: Optional[List[List[float]]],
        traj_steps: List[Dict[str, Any]],
        is_success: bool,
        maze_layouts: Dict[str, np.ndarray],
    ) -> go.Figure:
        """Construct animated Plotly figure with Play/Pause and scrubber slider."""
        shapes, (x_min, x_max, y_min, y_max) = get_maze_wall_shapes(split, maze_layouts)
        fig = go.Figure()
        if not traj_steps:
            return fig
        tx, ty = [s["x"] for s in traj_steps], [s["y"] for s in traj_steps]
        ant_color = "#2CA02C" if is_success else "#D62728"
        px = [p[0] for p in path_coords] if path_coords else []
        py = [p[1] for p in path_coords] if path_coords else []
        add_trajectory_base_traces(fig, px, py, tx, ty, is_success, ant_color)
        append_attention_trace(fig, traj_steps)
        total = len(traj_steps)
        step_size = max(1, total // 50)
        indices = list(range(0, total, step_size))
        if indices[-1] != total - 1:
            indices.append(total - 1)
        frames = create_trajectory_frames(tx, ty, indices, ant_color)
        menu, sliders = make_animation_menus(indices)
        status_str = "SUCCESS" if is_success else "FAILED"
        fig.update_layout(
            title=dict(text=f"Trajectory: {method_name} [{status_str}]", font=dict(size=14, family="sans-serif"), x=0.22, y=1.07, xanchor="left"),
            shapes=shapes, plot_bgcolor="#F7FAFC", paper_bgcolor="#FFFFFF",
            xaxis=dict(title="X (meters)", range=[x_min, x_max], gridcolor="#E2E8F0", zeroline=False),
            yaxis=dict(title="Y (meters)", range=[y_min, y_max], scaleanchor="x", scaleratio=1, gridcolor="#E2E8F0", zeroline=False),
            template="seaborn", width=740, height=640, updatemenus=menu, sliders=sliders,
            legend=dict(orientation="h", yanchor="bottom", y=1.07, xanchor="right", x=1.0),
            margin=dict(l=40, r=40, t=80, b=60),
        )
        fig.frames = frames
        return fig

    return build_animated_trajectory_figure,


@app.cell
def __(Any, Dict, List, go, json, np):
    """Attention distribution or locomotion dynamics figure builder."""
    def build_attention_distribution_figure(
        traj_steps: List[Dict[str, Any]], method_name: str
    ) -> go.Figure:
        """Construct attention distribution bar chart or locomotion dynamics line chart."""
        fig = go.Figure()
        if not traj_steps:
            return fig
        raw_wts = [s.get("attention_weights") for s in traj_steps if s.get("attention_weights")]
        parsed_wts = []
        for rw in raw_wts:
            try:
                pw = json.loads(rw) if isinstance(rw, str) else rw
                if pw and len(pw) > 0:
                    parsed_wts.append([float(w) for w in pw])
            except Exception:
                pass
        if parsed_wts:
            mean_w = np.mean(parsed_wts, axis=0) * 100.0
            n_tokens = len(mean_w)
            labels = [f"WP {i + 1}" for i in range(n_tokens)]
            if n_tokens > 0:
                labels[-1] = "Goal (z)"
            fig.add_trace(go.Bar(
                x=labels, y=mean_w,
                text=[f"{v:.1f}%" for v in mean_w], textposition="outside",
                marker=dict(color="#3182CE", line=dict(color="#1A365D", width=1.5)),
                name="Attention Weight",
            ))
            y_top = min(100.0, max(mean_w) * 1.25 + 5.0) if len(mean_w) > 0 else 100.0
            fig.update_layout(
                title=dict(text=f"Waypoint Attention: {method_name}", font=dict(size=14, family="sans-serif"), x=0.0, y=0.98),
                xaxis=dict(title="Waypoint Token Sequence", gridcolor="#E2E8F0"),
                yaxis=dict(title="Attention Weight (%)", range=[0, y_top], gridcolor="#E2E8F0"),
                template="seaborn", width=540, height=640,
                margin=dict(l=50, r=40, t=80, b=60),
            )
        else:
            steps_idx = list(range(len(traj_steps)))
            speeds = [float(s.get("speed", 0.0)) for s in traj_steps]
            norms = [float(s.get("action_norm", 0.0)) for s in traj_steps]
            fig.add_trace(go.Scatter(x=steps_idx, y=speeds, mode="lines", name="Speed (m/s)", line=dict(color="#55A868", width=2)))
            fig.add_trace(go.Scatter(x=steps_idx, y=norms, mode="lines", name="Action Norm", line=dict(color="#DD8452", width=1.5, dash="dot"), yaxis="y2"))
            fig.update_layout(
                title=dict(text=f"Control Dynamics: {method_name}", font=dict(size=14, family="sans-serif"), x=0.0, y=0.98),
                xaxis=dict(title="Step Index", gridcolor="#E2E8F0"),
                yaxis=dict(title="Speed (m/s)", gridcolor="#E2E8F0"),
                yaxis2=dict(title="Action Norm", overlaying="y", side="right", showgrid=False),
                template="seaborn", width=540, height=640,
                legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="right", x=1.0),
                margin=dict(l=50, r=50, t=80, b=60),
            )
        return fig

    return build_attention_distribution_figure,


@app.cell
def __(mo):
    """Render top anti-slop application header."""
    app_header = mo.Html("""
    <div style="padding: 16px 0 20px 0; border-bottom: 1px solid #E2E8F0; margin-bottom: 20px;">
        <div style="display: flex; align-items: baseline; justify-content: space-between;">
            <div>
                <h1 style="margin: 0; font-size: 24px; font-weight: 700; color: #1A202C; letter-spacing: -0.02em;">
                    FB-RL AntMaze Benchmark Analytics
                </h1>
                <p style="margin: 4px 0 0 0; font-size: 13px; color: #718096;">
                    Relational Telemetry & Empirical Performance Evaluation
                </p>
            </div>
            <div style="font-size: 12px; color: #4A5568; background: #EDF2F7; padding: 4px 10px; border-radius: 4px; font-weight: 500;">
                telemetry.db connected
            </div>
        </div>
    </div>
    """)
    return app_header,


@app.cell
def __(mo):
    """Tab 1 split selector control."""
    tab1_split = mo.ui.radio(
        options=["medium", "large"],
        value="medium",
        label="Maze Split",
        inline=True,
    )
    return tab1_split,


@app.cell
def __(
    build_pareto_figure,
    build_radar_figure,
    compute_benchmark_metrics,
    compute_radar_matrix,
    db_path,
    load_split_episodes,
    method_colors,
    mo,
    tab1_split,
):
    """Build Tab 1 summary metrics tables, Pareto chart, and radar profile."""
    df_tab1 = load_split_episodes(db_path, tab1_split.value)
    if df_tab1.empty:
        tab1_content = mo.md(f"No benchmark telemetry records found for split '{tab1_split.value}'.")
    else:
        main_df, task_df, bundle = compute_benchmark_metrics(df_tab1)
        methods = list(bundle.keys())
        table_main = mo.ui.table(main_df, selection=None, pagination=False, label="Main Benchmark Metrics")
        table_task = mo.ui.table(task_df, selection=None, pagination=False, label="Task Breakdown Success Rates (%)")
        pareto_fig = build_pareto_figure(methods, bundle, method_colors)
        categories, radar_matrix = compute_radar_matrix(methods, bundle)
        radar_fig = build_radar_figure(methods, categories, radar_matrix, method_colors)
        tab1_content = mo.vstack([
            mo.hstack([tab1_split], justify="start"),
            mo.md("#### Overall Benchmark Metrics"),
            table_main,
            mo.md("#### Task-by-Task Success Rates"),
            table_task,
            mo.md("#### Performance Visualizations"),
            mo.hstack([pareto_fig, radar_fig], justify="start", gap=2),
        ], gap=1.5)
    return (
        bundle,
        categories,
        df_tab1,
        main_df,
        methods,
        pareto_fig,
        radar_fig,
        radar_matrix,
        tab1_content,
        table_main,
        table_task,
    )


@app.cell
def __(mo):
    """Tab 2 filter controls."""
    tab2_split = mo.ui.radio(
        options=["medium", "large"],
        value="medium",
        label="Maze Split",
        inline=True,
    )
    tab2_method = mo.ui.dropdown(
        options=[
            "1. Dijkstra + Sequence Attention",
            "2. Single-Intention Baseline",
            "3. Dijkstra + Single Waypoint Translator",
            "4. Dijkstra Teacher (high_actor)",
            "5. Direct Intention Planner [O(1)]",
        ],
        value="1. Dijkstra + Sequence Attention",
        label="Method",
    )
    tab2_status = mo.ui.dropdown(
        options=["All", "Success Only", "Failed Only"],
        value="All",
        label="Status",
    )
    tab2_task = mo.ui.dropdown(
        options=["All", "1", "2", "3", "4", "5"],
        value="All",
        label="Task ID",
    )
    tab2_seed = mo.ui.dropdown(
        options=["All"] + [str(i) for i in range(1, 11)],
        value="All",
        label="Seed",
    )
    return (
        tab2_method,
        tab2_seed,
        tab2_split,
        tab2_status,
        tab2_task,
    )


@app.cell
def __(Any, Dict, Tuple, pd):
    """Format episode rows for display table and selector options."""
    def format_episodes_data(
        filtered: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, Dict[str, str]]:
        """Convert filtered episodes into table rows and selector choices."""
        table_data, ep_options = [], {}
        if filtered.empty:
            return pd.DataFrame(), {}
        for _, row in filtered.iterrows():
            badge = "SUCCESS" if row["is_success"] else "FAILED"
            ep_id_str = str(row["episode_id"])
            table_data.append({
                "Episode ID": ep_id_str[:32] + "...",
                "Status": badge,
                "Task": int(row["task_id"]),
                "Seed": int(row["seed"]),
                "Steps": int(row["total_steps"]),
                "Speed (m/s)": f"{float(row['mean_speed']):.2f}",
                "Loops": int(row["self_intersections"]),
                "CTE (m)": f"{float(row['mean_cross_track_error']):.2f}" if row["mean_cross_track_error"] > 0 else "—",
                "Latency (ms)": f"{float(row['mean_latency_ms']):.2f}",
                "_full_id": ep_id_str,
            })
            label = (
                f"[{badge}] Task {int(row['task_id'])} | Seed {int(row['seed'])} | Steps: {int(row['total_steps'])} | {ep_id_str[:16]}..."
            )
            ep_options[label] = ep_id_str
        return pd.DataFrame(table_data), ep_options

    def style_episode_cell(row_id: str, col_name: str, val: Any) -> Dict[str, Any]:
        """Apply color-coded status styling for table cells."""
        if col_name == "Status":
            if val == "SUCCESS":
                return {"backgroundColor": "#2CA02C20", "color": "#2CA02C", "fontWeight": "600", "textAlign": "center"}
            if val == "FAILED":
                return {"backgroundColor": "#D6272820", "color": "#D62728", "fontWeight": "600", "textAlign": "center"}
        return {}

    return format_episodes_data, style_episode_cell


@app.cell
def __(
    db_path,
    filter_episodes,
    format_episodes_data,
    load_split_episodes,
    mo,
    style_episode_cell,
    tab2_method,
    tab2_seed,
    tab2_split,
    tab2_status,
    tab2_task,
):
    """Tab 2 episode filtering and table generation."""
    df_tab2 = load_split_episodes(db_path, tab2_split.value)
    filtered_episodes = filter_episodes(
        df_tab2, tab2_method.value, tab2_status.value, tab2_task.value, tab2_seed.value
    )
    ep_df, ep_options = format_episodes_data(filtered_episodes)
    display_df = ep_df.drop(columns=["_full_id"]) if "_full_id" in ep_df.columns else ep_df
    ep_table = mo.ui.table(
        display_df,
        selection="single",
        page_size=10,
        style_cell=style_episode_cell,
        label=f"Filtered Episodes ({len(filtered_episodes)} total)",
    )
    default_label = list(ep_options.keys())[0] if ep_options else None
    ep_selector = mo.ui.dropdown(
        options=ep_options,
        value=default_label,
        label="Select Episode to Inspect",
    )
    return (
        default_label,
        df_tab2,
        display_df,
        ep_df,
        ep_options,
        ep_selector,
        ep_table,
        filtered_episodes,
    )


@app.cell
def __(pd):
    """Build HTML metric stat cards for selected episode."""
    def render_metric_cards_html(ep_row: pd.Series) -> str:
        """Generate clean HTML cards for episode outcome and summary metrics."""
        is_succ = bool(ep_row["is_success"])
        st_color = "#2CA02C" if is_succ else "#D62728"
        st_txt = "SUCCESS" if is_succ else "FAILED"
        cte_val = float(ep_row["mean_cross_track_error"])
        cte_str = f"{cte_val:.2f} m" if cte_val > 0 else "—"
        return f"""
        <div style="display: flex; gap: 12px; margin: 12px 0 16px 0; flex-wrap: wrap;">
            <div style="flex: 1; min-width: 110px; background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 6px; padding: 10px 14px;">
                <div style="font-size: 11px; text-transform: uppercase; color: #718096; font-weight: 600;">Status</div>
                <div style="font-size: 18px; font-weight: 700; color: {st_color};">{st_txt}</div>
            </div>
            <div style="flex: 1; min-width: 110px; background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 6px; padding: 10px 14px;">
                <div style="font-size: 11px; text-transform: uppercase; color: #718096; font-weight: 600;">Steps</div>
                <div style="font-size: 18px; font-weight: 700; color: #1A202C;">{int(ep_row['total_steps'])}</div>
            </div>
            <div style="flex: 1; min-width: 110px; background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 6px; padding: 10px 14px;">
                <div style="font-size: 11px; text-transform: uppercase; color: #718096; font-weight: 600;">Speed</div>
                <div style="font-size: 18px; font-weight: 700; color: #1A202C;">{float(ep_row['mean_speed']):.2f} m/s</div>
            </div>
            <div style="flex: 1; min-width: 110px; background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 6px; padding: 10px 14px;">
                <div style="font-size: 11px; text-transform: uppercase; color: #718096; font-weight: 600;">Self-Intersections</div>
                <div style="font-size: 18px; font-weight: 700; color: #1A202C;">{int(ep_row['self_intersections'])}</div>
            </div>
            <div style="flex: 1; min-width: 110px; background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 6px; padding: 10px 14px;">
                <div style="font-size: 11px; text-transform: uppercase; color: #718096; font-weight: 600;">Mean CTE</div>
                <div style="font-size: 18px; font-weight: 700; color: #1A202C;">{cte_str}</div>
            </div>
            <div style="flex: 1; min-width: 110px; background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 6px; padding: 10px 14px;">
                <div style="font-size: 11px; text-transform: uppercase; color: #718096; font-weight: 600;">Latency</div>
                <div style="font-size: 18px; font-weight: 700; color: #1A202C;">{float(ep_row['mean_latency_ms']):.2f} ms</div>
            </div>
        </div>
        """

    return render_metric_cards_html,


@app.cell
def __(
    db_path,
    ep_df,
    ep_selector,
    ep_table,
    filtered_episodes,
    load_episode_details,
):
    """Resolve selected episode and fetch step telemetry."""
    active_ep_id = ep_selector.value
    if ep_table.value is not None and len(ep_table.value) > 0:
        val = ep_table.value
        sel_dict = (
            val.iloc[0].to_dict()
            if hasattr(val, "iloc")
            else (val[0] if isinstance(val, list) and isinstance(val[0], dict) else (val if isinstance(val, dict) else {}))
        )
        target_id = sel_dict.get("Episode ID")
        if target_id is not None:
            match_rows = ep_df[ep_df["Episode ID"] == target_id]
            if not match_rows.empty and "_full_id" in match_rows.columns:
                active_ep_id = match_rows.iloc[0]["_full_id"]
        elif hasattr(val, "index") and len(val.index) > 0 and 0 <= val.index[0] < len(ep_df):
            active_ep_id = ep_df.iloc[val.index[0]]["_full_id"]

    matching = (
        filtered_episodes[filtered_episodes["episode_id"] == active_ep_id]
        if active_ep_id
        else filtered_episodes
    )
    if filtered_episodes.empty:
        ep_row, details, is_succ = None, {"path_coords": None, "steps": []}, False
    else:
        ep_row = matching.iloc[0] if not matching.empty else filtered_episodes.iloc[0]
        active_ep_id = str(ep_row["episode_id"])
        details = load_episode_details(db_path, active_ep_id)
        is_succ = bool(ep_row["is_success"])
    return active_ep_id, details, ep_row, is_succ


@app.cell
def __(
    build_animated_trajectory_figure,
    build_attention_distribution_figure,
    details,
    ep_row,
    ep_selector,
    ep_table,
    filtered_episodes,
    is_succ,
    maze_layouts,
    mo,
    pio,
    render_metric_cards_html,
    tab2_method,
    tab2_seed,
    tab2_split,
    tab2_status,
    tab2_task,
):
    """Assemble inspector filter controls, metric cards, and figures."""
    filter_bar = mo.hstack(
        [tab2_split, tab2_method, tab2_status, tab2_task, tab2_seed],
        justify="start", gap=1.5,
    )
    if filtered_episodes.empty or ep_row is None:
        inspector_content = mo.vstack([
            filter_bar,
            mo.md("No episodes match the selected filter criteria."),
        ])
    else:
        cards = render_metric_cards_html(ep_row)
        anim_fig = build_animated_trajectory_figure(
            tab2_split.value, tab2_method.value, details["path_coords"], details["steps"], is_succ, maze_layouts,
        )
        attn_fig = build_attention_distribution_figure(
            details["steps"], tab2_method.value,
        )
        anim_html = mo.Html(pio.to_html(anim_fig, include_plotlyjs="cdn", full_html=False))
        attn_html = mo.Html(pio.to_html(attn_fig, include_plotlyjs=False, full_html=False))
        inspector_content = mo.vstack([
            filter_bar,
            ep_table,
            mo.hstack([ep_selector], justify="start"),
            mo.Html(cards),
            mo.md("#### Trajectory Playback & Telemetry Analysis"),
            mo.hstack([anim_html, attn_html], justify="start", gap=2),
        ])
    return filter_bar, inspector_content


@app.cell
def __(app_header, inspector_content, mo, tab1_content):
    """Main application tabs view."""
    main_tabs = mo.ui.tabs({
        "Summary & Benchmark Metrics": tab1_content,
        "Episode & Trajectory Inspector": inspector_content,
    })
    app_layout = mo.vstack([
        app_header,
        main_tabs,
    ])
    app_layout
    return app_layout, main_tabs


if __name__ == "__main__":
    app.run()
