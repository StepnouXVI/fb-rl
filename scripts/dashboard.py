import json
import os
import sqlite3
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.telemetry.plotting import (
    build_pareto_figure,
    build_radar_figure,
    build_task_breakdown_figure,
    build_trajectory_figure,
)


@st.cache_data
def load_split_episodes(db_path: str, split: str) -> pd.DataFrame:
    """Query all episodes for specified split joined with run configuration."""
    conn = sqlite3.connect(db_path)
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
    return df


@st.cache_data
def load_episode_details(db_path: str, episode_id: str) -> Dict[str, Any]:
    """Fetch planned waypoints and downsampled step telemetry for an episode."""
    conn = sqlite3.connect(db_path)
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


def execute_sql_query(db_path: str, query: str) -> Tuple[Optional[pd.DataFrame], float, Optional[str]]:
    """Execute arbitrary SQL query against telemetry database and return metrics."""
    t0 = time.time()
    try:
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query(query, conn)
        conn.close()
        elapsed = (time.time() - t0) * 1000.0
        return df, elapsed, None
    except Exception as exc:
        elapsed = (time.time() - t0) * 1000.0
        return None, elapsed, str(exc)


def compute_benchmark_metrics(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Aggregate per-method benchmark statistics including seed std and task breakdowns."""
    methods = sorted(df["method"].unique())
    rows = []
    bundle = {}
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
        fps = 1000.0 / lat_mean if lat_mean > 0.0 else 0.0
        loops_mean = float(g["self_intersections"].mean())
        cte_vals = g[g["mean_cross_track_error"] > 0]["mean_cross_track_error"]
        cte_mean = float(cte_vals.mean()) if len(cte_vals) > 0 else 0.0
        rows.append({
            "Method": m,
            "Success Rate (%)": f"{sr_mean:.1f} ± {sr_std:.1f}",
            "Task 1 (%)": f"{task_srs[0]:.1f}",
            "Task 2 (%)": f"{task_srs[1]:.1f}",
            "Task 3 (%)": f"{task_srs[2]:.1f}",
            "Task 4 (%)": f"{task_srs[3]:.1f}",
            "Task 5 (%)": f"{task_srs[4]:.1f}",
            "Mean Steps": f"{steps_mean:.1f}",
            "Mean Speed (m/s)": f"{spd_mean:.2f}",
            "Latency & FPS": f"{lat_mean:.2f} ms ({fps:.0f} FPS)",
            "Self-Intersections": f"{loops_mean:.1f}",
            "Mean CTE (m)": f"{cte_mean:.2f}" if cte_mean > 0.0 else "—",
        })
        bundle[m] = {
            "sr_mean": sr_mean, "sr_std": sr_std, "lat_mean": lat_mean,
            "steps_mean": steps_mean, "loops_mean": loops_mean,
            "cte_mean": cte_mean, "task_srs": task_srs,
        }
    return pd.DataFrame(rows), bundle


def compute_radar_matrix(methods: List[str], bundle: Dict[str, Any]) -> Tuple[List[str], Dict[str, List[float]]]:
    """Normalize multi-axis metrics into 0-100 scale for radar architecture profile."""
    categories = ["Success Rate (%)", "Latency Score", "Step Efficiency", "Clean Path", "CTE Precision"]
    min_lat = min(bundle[m]["lat_mean"] for m in methods)
    min_steps = min(bundle[m]["steps_mean"] for m in methods)
    all_loops = [bundle[m]["loops_mean"] for m in methods]
    min_loops, max_loops = min(all_loops), max(all_loops)
    valid_ctes = [bundle[m]["cte_mean"] for m in methods if bundle[m]["cte_mean"] > 0]
    min_cte = min(valid_ctes) if valid_ctes else 0.0
    max_cte = max(valid_ctes) if valid_ctes else 0.0
    values_matrix = {}
    for m in methods:
        b = bundle[m]
        sr_score = min(100.0, max(0.0, b["sr_mean"]))
        lat_score = 100.0 * (min_lat / max(b["lat_mean"], 1e-6))
        steps_score = 100.0 * (min_steps / max(b["steps_mean"], 1e-6))
        if max_loops > min_loops:
            clean_score = 20.0 + 80.0 * (max_loops - b["loops_mean"]) / (max_loops - min_loops)
        else:
            clean_score = 100.0
        if b["cte_mean"] > 0.0 and max_cte > min_cte:
            cte_score = 20.0 + 80.0 * (max_cte - b["cte_mean"]) / (max_cte - min_cte)
        elif b["cte_mean"] > 0.0:
            cte_score = 100.0
        else:
            cte_score = 20.0
        values_matrix[m] = [sr_score, lat_score, steps_score, clean_score, cte_score]
    return categories, values_matrix


def render_overview_tab(df: pd.DataFrame) -> None:
    """Render comprehensive summary table, Pareto frontier, radar profile, and task breakdown."""
    st.markdown("### Benchmark Overview & Evaluation Metrics")
    table_df, bundle = compute_benchmark_metrics(df)
    st.dataframe(table_df, use_container_width=True, hide_index=True)
    methods = list(bundle.keys())
    latencies = [bundle[m]["lat_mean"] for m in methods]
    success_rates = [bundle[m]["sr_mean"] for m in methods]
    stds = [bundle[m]["sr_std"] for m in methods]
    task_matrix = {m: bundle[m]["task_srs"] for m in methods}
    categories, radar_matrix = compute_radar_matrix(methods, bundle)
    col1, col2 = st.columns(2)
    with col1:
        fig_pareto = build_pareto_figure(methods, latencies, success_rates, stds)
        st.plotly_chart(fig_pareto, use_container_width=True)
    with col2:
        fig_radar = build_radar_figure(methods, categories, radar_matrix)
        st.plotly_chart(fig_radar, use_container_width=True)
    fig_tasks = build_task_breakdown_figure(methods, task_matrix)
    st.plotly_chart(fig_tasks, use_container_width=True)


def filter_episodes(df: pd.DataFrame, method: str, seed: Any, task: Any, outcome: str) -> pd.DataFrame:
    """Filter episodes dataframe by method, seed, task, and outcome status."""
    sub = df[df["method"] == method].copy()
    if seed != "All":
        sub = sub[sub["seed"] == int(seed)]
    if task != "All":
        sub = sub[sub["task_id"] == int(task)]
    if outcome == "Successful Only":
        sub = sub[sub["is_success"] == 1]
    elif outcome == "Failed Only":
        sub = sub[sub["is_success"] == 0]
    return sub


def render_episode_metrics(ep_row: pd.Series) -> None:
    """Render 6-column metric display for selected episode telemetry record."""
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    outcome_str = "Success" if int(ep_row["is_success"]) == 1 else "Failed"
    c1.metric("Outcome", outcome_str)
    c2.metric("Total Steps", int(ep_row["total_steps"]))
    c3.metric("Mean Speed", f"{float(ep_row['mean_speed']):.2f} m/s")
    c4.metric("Self-Intersections", int(ep_row["self_intersections"]))
    cte = float(ep_row["mean_cross_track_error"])
    c5.metric("Mean CTE", f"{cte:.2f} m" if cte > 0.0 else "N/A")
    c6.metric("Step Latency", f"{float(ep_row['mean_latency_ms']):.2f} ms")


def render_inspector_tab(db_path: str, df: pd.DataFrame) -> None:
    """Render trajectory inspector with multi-parameter episode filtering and visualizer."""
    st.markdown("### Trajectory & Episode Inspector")
    methods = sorted(df["method"].unique())
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        sel_method = st.selectbox("Method", options=methods, index=0)
    with c2:
        seeds = sorted(df[df["method"] == sel_method]["seed"].unique())
        sel_seed = st.selectbox("Seed", options=["All"] + list(seeds))
    with c3:
        tasks = sorted(df[df["method"] == sel_method]["task_id"].unique())
        sel_task = st.selectbox("Task", options=["All"] + list(tasks))
    with c4:
        sel_outcome = st.selectbox("Outcome", options=["All", "Successful Only", "Failed Only"])
    filtered = filter_episodes(df, sel_method, sel_seed, sel_task, sel_outcome)
    if filtered.empty:
        st.info("No episodes match the selected filter criteria.")
        return
    st.caption(f"Showing {len(filtered)} matching episodes")
    display_cols = [
        "episode_id", "seed", "task_id", "is_success", "total_steps",
        "mean_speed", "self_intersections", "mean_cross_track_error",
    ]
    st.dataframe(filtered[display_cols], use_container_width=True, hide_index=True)
    ep_list = filtered["episode_id"].tolist()
    sel_ep = st.selectbox(
        "Select Episode to Inspect",
        options=ep_list,
        format_func=lambda eid: f"{eid[:32]}... | Seed {filtered[filtered['episode_id'] == eid]['seed'].values[0]} | Task {filtered[filtered['episode_id'] == eid]['task_id'].values[0]} | {'Success' if filtered[filtered['episode_id'] == eid]['is_success'].values[0] else 'Failed'}",
    )
    if sel_ep:
        ep_row = filtered[filtered["episode_id"] == sel_ep].iloc[0]
        render_episode_metrics(ep_row)
        details = load_episode_details(db_path, sel_ep)
        fig = build_trajectory_figure(
            method_name=sel_method,
            path_coords=details["path_coords"],
            traj_steps=details["steps"],
            is_success=bool(ep_row["is_success"]),
        )
        st.plotly_chart(fig, use_container_width=True)


def render_sql_console_tab(db_path: str) -> None:
    """Render interactive SQL query console with schema suggestions and execution timing."""
    st.markdown("### SQL Query Console")
    st.caption("Execute custom SQL queries directly against telemetry.db")
    default_query = """SELECT r.method, count(e.episode_id) as episodes,
       round(avg(e.is_success) * 100, 1) as success_rate_pct,
       round(avg(e.total_steps), 1) as avg_steps,
       round(avg(e.mean_speed), 3) as avg_speed_mps,
       round(avg(e.mean_latency_ms), 2) as avg_latency_ms
FROM runs r
JOIN episodes e ON r.run_id = e.run_id
GROUP BY r.method
ORDER BY success_rate_pct DESC"""
    query = st.text_area("SQL Statement", value=default_query, height=130)
    if st.button("Run Query") or query:
        df, elapsed_ms, err = execute_sql_query(db_path, query)
        if err:
            st.error(f"SQL Execution Error: {err}")
        elif df is not None:
            st.success(f"Execution completed in {elapsed_ms:.1f} ms — returned {len(df)} rows")
            st.dataframe(df, use_container_width=True)


def render_sidebar(default_db: str) -> Tuple[str, str]:
    """Render sidebar controls for dataset split and database file configuration."""
    st.sidebar.title("Configuration")
    split = st.sidebar.selectbox("Maze Split", options=["medium", "large"], index=0)
    db_path = st.sidebar.text_input("Database Path", value=default_db)
    return db_path, split


def main() -> None:
    """Main application entry point orchestrating benchmark tabs and telemetry data."""
    st.set_page_config(page_title="AntMaze Benchmark Telemetry", layout="wide")
    st.title("AntMaze Benchmark Telemetry & Analysis")
    default_db = "results/data/telemetry.db"
    db_path, split = render_sidebar(default_db)
    if not os.path.exists(db_path):
        st.error(f"Telemetry database not found at '{db_path}'. Check path configuration.")
        return
    df = load_split_episodes(db_path, split)
    if df.empty:
        st.warning(f"No benchmark telemetry records found for split '{split}' in {db_path}.")
        return
    tab1, tab2, tab3 = st.tabs([
        "Benchmark Overview & Metrics",
        "Trajectory & Episode Inspector",
        "SQL Query Console",
    ])
    with tab1:
        render_overview_tab(df)
    with tab2:
        render_inspector_tab(db_path, df)
    with tab3:
        render_sql_console_tab(db_path)


if __name__ == "__main__":
    main()
