import json
import numpy as np
import pandas as pd
from scipy import stats

def bootstrap_ci(data, num_bootstraps=2000, ci=95):
    if len(data) == 0:
        return 0.0, 0.0
    boot_means = [np.mean(np.random.choice(data, size=len(data), replace=True)) for _ in range(num_bootstraps)]
    lower = np.percentile(boot_means, (100 - ci) / 2)
    upper = np.percentile(boot_means, 100 - (100 - ci) / 2)
    return float(lower), float(upper)


def aggregate_runs(results_dict, baseline_name="Single-Intention Baseline"):
    summary = {}
    baseline_successes = None
    if baseline_name in results_dict:
        baseline_successes = [r["success_rate"] for r in results_dict[baseline_name]]

    for method, runs in results_dict.items():
        succ = [r["success_rate"] for r in runs]
        lens = [r["mean_length"] for r in runs]
        lats = [r["latency_ms"] for r in runs]

        succ_low, succ_high = bootstrap_ci(succ)
        len_low, len_high = bootstrap_ci(lens)
        lat_low, lat_high = bootstrap_ci(lats)

        p_val = None
        if baseline_successes is not None and method != baseline_name and len(succ) > 1:
            try:
                _, p_val = stats.ttest_ind(succ, baseline_successes, equal_var=False)
                p_val = float(p_val)
            except Exception:
                p_val = None

        summary[method] = {
            "success_rate": {"mean": float(np.mean(succ)), "std": float(np.std(succ)), "ci_lower": succ_low, "ci_upper": succ_high},
            "mean_length": {"mean": float(np.mean(lens)), "std": float(np.std(lens)), "ci_lower": len_low, "ci_upper": len_high},
            "latency_ms": {"mean": float(np.mean(lats)), "std": float(np.std(lats)), "ci_lower": lat_low, "ci_upper": lat_high},
            "p_value_vs_baseline": p_val,
            "n_seeds": len(runs),
        }
    return summary


def export_latex_table(summary):
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
        succ_str = f"{m['success_rate']['mean']:.1f} $\\pm$ {m['success_rate']['std']:.1f}"
        step_str = f"{m['mean_length']['mean']:.1f} $\\pm$ {m['mean_length']['std']:.1f}"
        lat_str = f"{m['latency_ms']['mean']:.2f}"
        p_str = f"{m['p_value_vs_baseline']:.2e}" if m["p_value_vs_baseline"] is not None else "--"
        lines.append(f"{method} & {succ_str} & {step_str} & {lat_str} & {p_str} \\\\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\caption{Multi-Subgoal Planning Benchmark on OGBench AntMaze.}", r"\end{table}"])
    return "\n".join(lines)
