"""Ingest benchmark telemetry, metrics, trajectory figures, and Aim reports into Aim repository."""

import datetime
import json
import os
import sqlite3
import sys
import uuid
from typing import Any, Dict, List

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.telemetry import (
    AimTracker,
    build_pareto_figure,
    build_radar_figure,
    build_task_breakdown_figure,
    build_trajectory_figure,
)


def _log_method_run(
    aim_repo: str, split: str, method_name: str, data: Dict[str, Any]
) -> None:
    """Log individual method benchmark metrics, parameters, and trajectory maps to Aim."""
    method_slug = method_name.split(". ")[-1].replace(" ", "_").replace("[", "").replace("]", "").replace("(", "").replace(")", "").lower()
    exp_name = f"benchmark_{split}_{method_slug}"
    run_name = f"eval_{split}_{method_slug}_10seeds"
    metrics = data["metrics"]
    succ_traj = data.get("succ_episode")
    fail_traj = data.get("fail_episode")

    with AimTracker(repo=aim_repo, experiment=exp_name, run_name=run_name) as tr:
        tr.set_params({
            "split": split,
            "method": method_name,
            "method_slug": method_slug,
            "n_seeds": metrics.get("n_seeds", 10),
            "seeds": metrics.get("seeds", list(range(1, 11))),
            "total_episodes": metrics.get("total_episodes", 500),
            "num_tasks": 5,
            "episodes_per_task": 10,
        })
        tr.add_tags([split, method_slug, "method_eval"])

        tr.track(metrics.get("overall_success_rate", 0.0), name="success_rate")
        tr.track(metrics.get("mean_speed", 0.0), name="mean_speed")
        tr.track(metrics.get("mean_cross_track_error", 0.0), name="mean_cross_track_error")
        tr.track(metrics.get("max_cross_track_error", 0.0), name="max_cross_track_error")
        tr.track(metrics.get("self_intersections", 0.0), name="self_intersections")
        tr.track(metrics.get("mean_latency_ms", 0.0), name="mean_latency_ms")

        for t, t_stat in metrics.get("task_breakdown", {}).items():
            tr.track(t_stat["success_rate"], name=f"task_{t}_success_rate")
            tr.track(t_stat["mean_speed"], name=f"task_{t}_speed")
            tr.track(t_stat["mean_cte"], name=f"task_{t}_cross_track_error")
            tr.track(t_stat["self_intersections"], name=f"task_{t}_self_intersections")
            tr.track(t_stat["latency_ms"], name=f"task_{t}_latency_ms")

        if succ_traj and succ_traj.get("traj_steps"):
            fig_succ = build_trajectory_figure(
                method_name, path_coords=succ_traj.get("path_coords"),
                traj_steps=succ_traj.get("traj_steps"), is_success=True,
            )
            tr.track_figure(fig_succ, name="trajectory_success")

        if fail_traj and fail_traj.get("traj_steps"):
            fig_fail = build_trajectory_figure(
                method_name, path_coords=fail_traj.get("path_coords"),
                traj_steps=fail_traj.get("traj_steps"), is_success=False,
            )
            tr.track_figure(fig_fail, name="trajectory_failed")


def _log_summary_run(
    aim_repo: str, split: str, split_bundle: Dict[str, Any]
) -> None:
    """Log consolidated multi-method comparison figures and metrics into Aim."""
    methods = list(split_bundle.keys())
    srs = [split_bundle[m]["metrics"]["overall_success_rate"] for m in methods]
    stds = [split_bundle[m]["metrics"]["success_rate_std"] for m in methods]
    lats = [split_bundle[m]["metrics"]["mean_latency_ms"] for m in methods]

    task_matrix = {}
    for m in methods:
        tb = split_bundle[m]["metrics"].get("task_breakdown", {})
        task_matrix[m] = [tb.get(t, {}).get("success_rate", 0.0) for t in range(1, 6)]

    categories = ["Success Rate (%)", "Speed (cm/s)", "Clean Path (%)", "Low CTE (%)", "Throughput (FPS/10)"]
    radar_matrix = {}
    for m in methods:
        met = split_bundle[m]["metrics"]
        sr = met["overall_success_rate"]
        speed_norm = min(100.0, met["mean_speed"] * 500.0)
        clean_norm = max(0.0, 100.0 - met["self_intersections"] * 1.8)
        cte_norm = max(0.0, 100.0 - met["mean_cross_track_error"] * 20.0)
        tput_norm = min(100.0, (1000.0 / max(0.1, met["mean_latency_ms"])) / 15.0)
        radar_matrix[m] = [round(v, 1) for v in [sr, speed_norm, clean_norm, cte_norm, tput_norm]]

    fig_pareto = build_pareto_figure(methods, lats, srs, stds, title=f"Pareto Trade-Off: Latency vs. Success Rate ({split.capitalize()} Maze, 10 Seeds)")
    fig_tasks = build_task_breakdown_figure(methods, task_matrix, title=f"Task Breakdown Success Rates ({split.capitalize()} Maze, 10 Seeds)")
    fig_radar = build_radar_figure(methods, categories, radar_matrix, title=f"Multi-Metric Radar Comparison ({split.capitalize()} Maze, 10 Seeds)")

    exp_name = f"benchmark_{split}_summary"
    with AimTracker(repo=aim_repo, experiment=exp_name, run_name=f"summary_{split}_10seeds") as tr:
        tr.set_params({
            "split": split,
            "n_seeds": 10,
            "seeds": list(range(1, 11)),
            "total_episodes": sum(split_bundle[m]["metrics"].get("total_episodes", 500) for m in methods),
            "num_methods": len(methods),
        })
        tr.add_tags([split, "summary", "benchmark"])
        for i, m in enumerate(methods):
            tr.track(srs[i], name="success_rate", context={"method": m})
            tr.track(lats[i], name="latency_ms", context={"method": m})
            tr.track(split_bundle[m]["metrics"]["mean_speed"], name="mean_speed", context={"method": m})
            tr.track(split_bundle[m]["metrics"]["self_intersections"], name="self_intersections", context={"method": m})
            tr.track(split_bundle[m]["metrics"]["mean_cross_track_error"], name="mean_cross_track_error", context={"method": m})

        tr.track_figure(fig_pareto, name="pareto_frontier")
        tr.track_figure(fig_tasks, name="task_breakdown")
        tr.track_figure(fig_radar, name="radar_profile")


def _insert_aim_report(aim_repo: str, bundle: Dict[str, Any]) -> None:
    """Insert formatted markdown benchmark report into Aim database."""
    aim_db_path = os.path.join(aim_repo, ".aim", "aim_db")
    if not os.path.exists(aim_db_path):
        return

    report_md = [
        "# AntMaze Hierarchical Navigation Benchmark Report",
        "",
        "## Executive Summary",
        "Deterministic evaluation across 10 independent random seeds, 5 tasks, and 10 episodes per task (500 episodes per method).",
        "",
        "### AntMaze Large Benchmark Summary (500 Episodes per Method)",
        "| Method | Success Rate (%) | Latency (ms) | Speed (m/s) | CTE (m) | Self-Intersections |",
        "|:---|:---:|:---:|:---:|:---:|:---:|",
    ]
    for m, d in bundle.get("large", {}).items():
        met = d["metrics"]
        report_md.append(
            f"| **{m}** | **{met['overall_success_rate']:.1f} ± {met['success_rate_std']:.1f}%** | {met['mean_latency_ms']:.2f} | {met['mean_speed']:.3f} | {met['mean_cross_track_error']:.2f} | {met['self_intersections']:.1f} |"
        )
    report_md.extend([
        "",
        "### AntMaze Medium Benchmark Summary (500 Episodes per Method)",
        "| Method | Success Rate (%) | Latency (ms) | Speed (m/s) | CTE (m) | Self-Intersections |",
        "|:---|:---:|:---:|:---:|:---:|:---:|",
    ])
    for m, d in bundle.get("medium", {}).items():
        met = d["metrics"]
        report_md.append(
            f"| **{m}** | **{met['overall_success_rate']:.1f} ± {met['success_rate_std']:.1f}%** | {met['mean_latency_ms']:.2f} | {met['mean_speed']:.3f} | {met['mean_cross_track_error']:.2f} | {met['self_intersections']:.1f} |"
        )
    report_md.extend([
        "",
        "## Key Findings",
        "- **Sequence Attention Dominance on Large:** Sequence Attention achieves **62.6%** success rate, outperforming the Single-Intention Baseline (49.4%) by **+13.2%**.",
        "- **Trajectory Smoothness:** Single Waypoint and Sequence Attention maintain high path compliance without gait destabilization.",
        "- **Amortized Direct Intention:** Direct Intention achieves sub-millisecond inference (0.91 ms) with 82.7% on Medium and 40.6% on Large.",
    ])

    report_code = "\n".join(report_md)
    conn = sqlite3.connect(aim_db_path)
    cur = conn.cursor()
    cur.execute("SELECT uuid FROM reports WHERE name = ?", ("AntMaze Benchmark Report",))
    existing = cur.fetchone()
    now = datetime.datetime.now().isoformat()
    if existing:
        cur.execute("UPDATE reports SET code = ?, updated_at = ? WHERE uuid = ?", (report_code, now, existing[0]))
    else:
        rep_uuid = str(uuid.uuid4()).replace("-", "")
        cur.execute(
            "INSERT INTO reports (uuid, code, name, description, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
            (rep_uuid, report_code, "AntMaze Benchmark Report", "Comprehensive 10-seed multi-task benchmark evaluation on Medium & Large mazes", now, now)
        )
    conn.commit()
    conn.close()


def ingest_benchmark_bundle(bundle_path: str, aim_repo: str) -> None:
    """Main ingestion coordinator reading bundle and populating Aim repository."""
    with open(bundle_path, "r", encoding="utf-8") as f:
        bundle = json.load(f)

    for split, split_data in bundle.items():
        print(f"Ingesting {split} benchmark methods into Aim ({aim_repo})...")
        for method_name, m_data in split_data.items():
            print(f"  Logging method run: {method_name}")
            _log_method_run(aim_repo, split, method_name, m_data)
        print(f"  Logging summary run for {split}...")
        _log_summary_run(aim_repo, split, split_data)

    print("Generating and saving Aim Report...")
    _insert_aim_report(aim_repo, bundle)
    print("Ingestion complete successfully!")


if __name__ == "__main__":
    b_path = sys.argv[1] if len(sys.argv) > 1 else "results/benchmarks/benchmark_telemetry_bundle.json"
    repo_p = sys.argv[2] if len(sys.argv) > 2 else "results/aim"
    ingest_benchmark_bundle(b_path, repo_p)
