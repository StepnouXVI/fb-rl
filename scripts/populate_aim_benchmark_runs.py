"""Populate individual benchmark episode runs with metrics and trajectory maps in Aim."""

import json
import os
import sqlite3
import sys
import time
from typing import Any, Dict, List, Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from aim import Figure, Repo, Run
from src.telemetry.plotting import build_trajectory_figure


METHODS = [
    ("1. Dijkstra + Sequence Attention", "dijkstra_+_sequence_attention"),
    ("2. Single-Intention Baseline", "single-intention_baseline"),
    ("3. Dijkstra + Single Waypoint Translator", "dijkstra_+_single_waypoint_translator"),
    ("4. Dijkstra Teacher (high_actor)", "dijkstra_teacher_high_actor"),
    ("5. Direct Intention Planner [O(1)]", "direct_intention_planner_o1"),
]


def _clean_unwanted_experiments(repo: Repo) -> None:
    """Remove summary and outdated experiment traces from Aim repository."""
    unwanted = [
        "benchmark_medium_summary",
        "benchmark_large_summary",
        "benchmark_medium",
        "benchmark_large",
    ]
    for exp_name in unwanted:
        try:
            repo.delete_experiment(exp_name)
            print(f"Removed outdated experiment: {exp_name}")
        except Exception:
            pass


def _fetch_method_episodes(
    cur: sqlite3.Cursor, split: str, method: str
) -> List[Dict[str, Any]]:
    """Retrieve 500 deterministic benchmark episodes for split and method."""
    run_filter = "r.run_id LIKE '9b7b9518-%'" if split == "medium" else "r.run_id LIKE '9014a9a5-%'"
    query = f"""
        SELECT e.episode_id, e.seed, e.task_id, e.episode_idx, e.is_success,
               e.total_steps, e.mean_speed, e.mean_cross_track_error,
               e.max_cross_track_error, e.self_intersections, e.mean_latency_ms
        FROM episodes e
        JOIN runs r ON e.run_id = r.run_id
        WHERE r.split = ? AND r.method = ? AND {run_filter}
        ORDER BY e.seed, e.task_id, e.episode_idx
    """
    cur.execute(query, (split, method))
    rows = cur.fetchall()
    keys = [
        "episode_id", "seed", "task_id", "episode_idx", "is_success",
        "total_steps", "mean_speed", "mean_cross_track_error",
        "max_cross_track_error", "self_intersections", "mean_latency_ms"
    ]
    return [dict(zip(keys, r)) for r in rows]


def _extract_trajectory_data(
    cur: sqlite3.Cursor, episode_id: str
) -> Dict[str, Any]:
    """Extract downsampled step coordinates and planned path from SQLite."""
    cur.execute(
        "SELECT x, y FROM steps WHERE episode_id = ? ORDER BY step_idx",
        (episode_id,)
    )
    step_rows = cur.fetchall()
    all_steps = [{"x": r[0], "y": r[1]} for r in step_rows]
    stride = max(1, len(all_steps) // 50)
    downsampled = all_steps[::stride]
    if all_steps and (not downsampled or downsampled[-1] != all_steps[-1]):
        downsampled.append(all_steps[-1])

    cur.execute(
        "SELECT path_coords_json FROM planned_paths WHERE episode_id = ? ORDER BY replan_step LIMIT 1",
        (episode_id,)
    )
    path_row = cur.fetchone()
    path_coords = json.loads(path_row[0]) if (path_row and path_row[0]) else None
    return {"path_coords": path_coords, "traj_steps": downsampled}


def _log_single_episode(
    repo: Repo,
    exp_name: str,
    split: str,
    method: str,
    method_slug: str,
    ep: Dict[str, Any],
    cur: sqlite3.Cursor,
) -> None:
    """Create Aim run for episode, recording scalar metrics and trajectory map."""
    run_name = f"s{ep['seed']:02d}_t{ep['task_id']}_ep{ep['episode_idx']:02d}"
    run = Run(
        repo=repo,
        experiment=exp_name,
        system_tracking_interval=None,
        capture_terminal_logs=False,
    )
    run.name = run_name
    run["hparams"] = {
        "split": split,
        "method": method,
        "method_slug": method_slug,
        "seed": ep["seed"],
        "task_id": ep["task_id"],
        "episode_idx": ep["episode_idx"],
        "is_success": bool(ep["is_success"]),
        "total_steps": int(ep["total_steps"]),
    }
    for tag in [split, method_slug, f"seed_{ep['seed']}", f"task_{ep['task_id']}"]:
        run.add_tag(tag)

    run.track(float(ep["is_success"]), name="is_success")
    run.track(int(ep["total_steps"]), name="total_steps")
    run.track(float(ep["mean_speed"]), name="mean_speed")
    run.track(float(ep["self_intersections"]), name="self_intersections")
    run.track(float(ep["mean_cross_track_error"]), name="mean_cross_track_error")
    run.track(float(ep["max_cross_track_error"]), name="max_cross_track_error")
    run.track(float(ep["mean_latency_ms"]), name="mean_latency_ms")

    traj_data = _extract_trajectory_data(cur, ep["episode_id"])
    if traj_data["traj_steps"]:
        fig = build_trajectory_figure(
            method,
            path_coords=traj_data["path_coords"],
            traj_steps=traj_data["traj_steps"],
            is_success=bool(ep["is_success"]),
        )
        run.track(Figure(fig), name="trajectory_map")

    run.close()


def populate_all_benchmark_runs(db_path: str, aim_repo: str) -> None:
    """Ingest 500 runs per method across Medium and Large mazes into Aim."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    repo = Repo(aim_repo, init=True)
    _clean_unwanted_experiments(repo)

    total_episodes_logged = 0
    t_start = time.time()

    for split in ["medium", "large"]:
        print(f"=== Starting Ingestion for {split.upper()} MAZE ===")
        for method_name, method_slug in METHODS:
            exp_name = f"benchmark_{split}_{method_slug}"
            episodes = _fetch_method_episodes(cur, split, method_name)
            print(f"  [{split}] {method_name}: {len(episodes)} episodes -> {exp_name}")
            t_m0 = time.time()
            for idx, ep in enumerate(episodes):
                _log_single_episode(repo, exp_name, split, method_name, method_slug, ep, cur)
                total_episodes_logged += 1
                if (idx + 1) % 100 == 0:
                    dt = time.time() - t_m0
                    print(f"    Processed {idx + 1}/500 episodes ({dt:.1f}s, {dt/(idx+1)*1000:.1f}ms/ep)")
            dt_m = time.time() - t_m0
            print(f"  Finished {method_name} in {dt_m:.1f}s")

    conn.close()
    elapsed = time.time() - t_start
    print(f"Successfully populated {total_episodes_logged} runs across 10 experiments in {elapsed:.1f}s!")


if __name__ == "__main__":
    database_file = sys.argv[1] if len(sys.argv) > 1 else "results/data/telemetry.db"
    aim_directory = sys.argv[2] if len(sys.argv) > 2 else "results/aim"
    populate_all_benchmark_runs(database_file, aim_directory)
