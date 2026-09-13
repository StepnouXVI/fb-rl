"""Export structured benchmark telemetry, metrics, and episode trajectories from SQLite."""

import json
import os
import sqlite3
import sys
from typing import Any, Dict, List


def _query_method_episodes(
    cur: sqlite3.Cursor, split: str, method: str
) -> List[Dict[str, Any]]:
    """Fetch episode records for given method and split."""
    query = """
        SELECT e.episode_id, e.seed, e.task_id, e.episode_idx, e.is_success,
               e.mean_speed, e.mean_cross_track_error, e.max_cross_track_error,
               e.self_intersections, e.mean_latency_ms, e.total_steps
        FROM episodes e
        JOIN runs r ON e.run_id = r.run_id
        WHERE r.split = ? AND r.method = ?
        ORDER BY e.seed, e.task_id, e.episode_idx
    """
    cur.execute(query, (split, method))
    rows = cur.fetchall()
    keys = [
        "episode_id", "seed", "task_id", "episode_idx", "is_success",
        "mean_speed", "mean_cross_track_error", "max_cross_track_error",
        "self_intersections", "mean_latency_ms", "total_steps"
    ]
    return [dict(zip(keys, r)) for r in rows]


def _extract_episode_trajectory(
    cur: sqlite3.Cursor, episode_id: str
) -> Dict[str, Any]:
    """Extract downsampled step coordinates and planned path for an episode."""
    cur.execute(
        "SELECT x, y, speed, action_norm, attention_targets, attention_weights FROM steps WHERE episode_id = ? ORDER BY step_idx",
        (episode_id,)
    )
    step_rows = cur.fetchall()
    step_keys = ["x", "y", "speed", "action_norm", "attention_targets", "attention_weights"]
    all_steps = [dict(zip(step_keys, r)) for r in step_rows]
    stride = max(1, len(all_steps) // 120)
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


def _aggregate_method_metrics(episodes: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute overall and per-task statistical aggregates across episodes."""
    if not episodes:
        return {}
    import numpy as np

    seeds = sorted(list(set(e["seed"] for e in episodes)))
    seed_sr = []
    for s in seeds:
        s_eps = [e for e in episodes if e["seed"] == s]
        seed_sr.append(float(np.mean([e["is_success"] for e in s_eps])) * 100.0)

    task_stats = {}
    for t in range(1, 6):
        t_eps = [e for e in episodes if e["task_id"] == t]
        if t_eps:
            task_stats[t] = {
                "success_rate": float(np.mean([e["is_success"] for e in t_eps])) * 100.0,
                "mean_speed": float(np.mean([e["mean_speed"] for e in t_eps])),
                "mean_cte": float(np.mean([e["mean_cross_track_error"] for e in t_eps])),
                "self_intersections": float(np.mean([e["self_intersections"] for e in t_eps])),
                "latency_ms": float(np.mean([e["mean_latency_ms"] for e in t_eps])),
            }

    return {
        "seeds": seeds,
        "n_seeds": len(seeds),
        "total_episodes": len(episodes),
        "overall_success_rate": float(np.mean(seed_sr)),
        "success_rate_std": float(np.std(seed_sr)),
        "mean_speed": float(np.mean([e["mean_speed"] for e in episodes])),
        "mean_cross_track_error": float(np.mean([e["mean_cross_track_error"] for e in episodes])),
        "max_cross_track_error": float(np.max([e["max_cross_track_error"] for e in episodes])) if episodes else 0.0,
        "self_intersections": float(np.mean([e["self_intersections"] for e in episodes])),
        "mean_latency_ms": float(np.mean([e["mean_latency_ms"] for e in episodes])),
        "task_breakdown": task_stats,
    }


def export_bundle(db_path: str, out_path: str) -> None:
    """Extract complete benchmark summary bundle from SQLite database."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    methods = [
        "1. Dijkstra + Sequence Attention",
        "2. Single-Intention Baseline",
        "3. Dijkstra + Single Waypoint Translator",
        "4. Dijkstra Teacher (high_actor)",
        "5. Direct Intention Planner [O(1)]",
    ]
    bundle = {}
    for split in ["medium", "large"]:
        bundle[split] = {}
        for m in methods:
            eps = _query_method_episodes(cur, split, m)
            if not eps:
                continue
            metrics = _aggregate_method_metrics(eps)
            succ_eps = [e for e in eps if e["is_success"] == 1]
            fail_eps = [e for e in eps if e["is_success"] == 0]
            succ_traj = _extract_episode_trajectory(cur, succ_eps[0]["episode_id"]) if succ_eps else None
            fail_traj = _extract_episode_trajectory(cur, fail_eps[0]["episode_id"]) if fail_eps else None
            bundle[split][m] = {
                "metrics": metrics,
                "succ_episode": succ_traj,
                "fail_episode": fail_traj,
            }
    conn.close()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(bundle, f)
    print(f"Exported telemetry bundle to {out_path} ({os.path.getsize(out_path)} bytes)")


if __name__ == "__main__":
    db_p = sys.argv[1] if len(sys.argv) > 1 else "results/data/telemetry.db"
    out_p = sys.argv[2] if len(sys.argv) > 2 else "results/benchmarks/benchmark_telemetry_bundle.json"
    export_bundle(db_p, out_p)
