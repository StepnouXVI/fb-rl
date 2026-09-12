"""Statistical metrics, trajectory analytics, and database aggregation utilities."""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from scipy import stats

from src.telemetry.db import TelemetryDatabase
from src.telemetry.metrics import (
    compute_cross_track_errors,
    count_spatial_self_intersections,
)


def bootstrap_ci(
    data: Union[List[float], np.ndarray],
    num_bootstraps: int = 2000,
    ci: float = 95.0,
) -> Tuple[float, float]:
    """Calculate non-parametric bootstrap confidence interval."""
    arr = np.asarray(data, dtype=np.float64)
    if len(arr) == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(42)
    boot_means = [
        float(np.mean(rng.choice(arr, size=len(arr), replace=True)))
        for _ in range(num_bootstraps)
    ]
    lower = float(np.percentile(boot_means, (100.0 - ci) / 2.0))
    upper = float(np.percentile(boot_means, 100.0 - (100.0 - ci) / 2.0))
    return lower, upper


def _calculate_metric_stats(values: List[float]) -> Dict[str, float]:
    """Calculate mean, standard deviation, and bootstrap confidence interval."""
    val_arr = np.asarray(values, dtype=np.float64) if values else np.array([0.0])
    mean_val = float(np.mean(val_arr))
    std_val = float(np.std(val_arr))
    ci_low, ci_high = bootstrap_ci(val_arr)
    return {
        "mean": mean_val,
        "std": std_val,
        "ci_lower": ci_low,
        "ci_upper": ci_high,
    }


def aggregate_runs(
    results_dict: Dict[str, List[Dict[str, Any]]],
    baseline_name: str = "Single-Intention Baseline",
) -> Dict[str, Dict[str, Any]]:
    """Aggregate per-run metrics dictionaries and compute statistical significance."""
    summary: Dict[str, Dict[str, Any]] = {}
    baseline_succ = None
    if baseline_name in results_dict:
        baseline_succ = [r["success_rate"] for r in results_dict[baseline_name]]

    for method, runs in results_dict.items():
        succ = [r["success_rate"] for r in runs]
        lens = [r["mean_length"] for r in runs]
        lats = [r["latency_ms"] for r in runs]

        p_val = None
        if baseline_succ is not None and method != baseline_name and len(succ) > 1:
            try:
                _, p_val = stats.ttest_ind(succ, baseline_succ, equal_var=False)
                p_val = float(p_val)
            except Exception:
                p_val = None

        summary[method] = {
            "success_rate": _calculate_metric_stats(succ),
            "mean_length": _calculate_metric_stats(lens),
            "latency_ms": _calculate_metric_stats(lats),
            "p_value_vs_baseline": p_val,
            "n_seeds": len(runs),
        }
    return summary


def aggregate_db_runs(
    db_or_path: Union[str, TelemetryDatabase],
    experiment_id: Optional[str] = None,
    baseline_name: str = "Single-Intention Baseline",
) -> Dict[str, Dict[str, Any]]:
    """Aggregate run metrics directly from SQLite TelemetryDatabase."""
    db = (
        TelemetryDatabase(db_or_path)
        if isinstance(db_or_path, str)
        else db_or_path
    )
    cur = db.conn.cursor()
    query = """
        SELECT r.method, r.seed,
               AVG(e.is_success) * 100.0 as succ_rate,
               AVG(e.total_steps) as mean_steps,
               AVG(e.mean_latency_ms) as mean_lat,
               AVG(e.mean_cross_track_error) as mean_cte,
               AVG(e.self_intersections) as mean_intersections
        FROM episodes e
        JOIN runs r ON e.run_id = r.run_id
    """
    params: List[Any] = []
    if experiment_id is not None:
        query += " WHERE r.experiment_id = ?"
        params.append(experiment_id)
    query += " GROUP BY r.method, r.seed"
    cur.execute(query, params)
    rows = cur.fetchall()

    run_grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        m = row["method"]
        if m not in run_grouped:
            run_grouped[m] = []
        run_grouped[m].append({
            "success_rate": float(row["succ_rate"]),
            "mean_length": float(row["mean_steps"]),
            "latency_ms": float(row["mean_lat"]),
            "mean_cte": float(row["mean_cte"]),
            "mean_intersections": float(row["mean_intersections"]),
        })
    cur.close()
    if isinstance(db_or_path, str):
        db.close()

    summary = aggregate_runs(run_grouped, baseline_name=baseline_name)
    for m, runs in run_grouped.items():
        if m in summary:
            summary[m]["cross_track_error"] = _calculate_metric_stats(
                [r["mean_cte"] for r in runs]
            )
            summary[m]["self_intersections"] = _calculate_metric_stats(
                [r["mean_intersections"] for r in runs]
            )
    return summary


def export_latex_table(summary: Dict[str, Dict[str, Any]]) -> str:
    """Generate LaTeX tabular markup comparing evaluated planning methods."""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"\textbf{Hierarchical Planning Method} & \textbf{Success Rate (\%)} & \textbf{Avg Steps} & \textbf{Latency (ms)} & \textbf{p-value} \\",
        r"\midrule",
    ]
    for method, m in summary.items():
        succ_str = (
            f"{m['success_rate']['mean']:.1f} $\\pm$ {m['success_rate']['std']:.1f}"
        )
        step_str = (
            f"{m['mean_length']['mean']:.1f} $\\pm$ {m['mean_length']['std']:.1f}"
        )
        lat_str = f"{m['latency_ms']['mean']:.2f}"
        p_val = m.get("p_value_vs_baseline")
        p_str = f"{p_val:.2e}" if p_val is not None else "--"
        lines.append(f"{method} & {succ_str} & {step_str} & {lat_str} & {p_str} \\\\")
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Multi-Subgoal Planning Benchmark on OGBench AntMaze.}",
        r"\end{table}",
    ])
    return "\n".join(lines)
