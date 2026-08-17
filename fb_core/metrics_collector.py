"""
MetricsCollector: Statistical processing, Bootstrap CI computation, and CSV/JSON/LaTeX exporter.
"""

import json
from typing import Dict, List, Any
import numpy as np
import scipy.stats as stats


class MetricsCollector:
    """
    Standardized metrics aggregation and export for multi-seed RL benchmarks.
    # ponytail: Native scipy and numpy stats with clean LaTeX table output.
    """

    @staticmethod
    def compute_bootstrap_ci(data: List[float], n_bootstraps: int = 1000, ci: float = 0.95) -> Dict[str, float]:
        """Compute bootstrap mean and confidence interval."""
        arr = np.asarray(data, dtype=np.float64)
        if len(arr) == 0:
            return {"mean": 0.0, "std": 0.0, "ci_lower": 0.0, "ci_upper": 0.0}
        
        mean_val = float(np.mean(arr))
        std_val = float(np.std(arr))
        
        if len(arr) < 2:
            return {"mean": mean_val, "std": std_val, "ci_lower": mean_val, "ci_upper": mean_val}
            
        boot_means = []
        rng = np.random.default_rng(42)
        for _ in range(n_bootstraps):
            sample = rng.choice(arr, size=len(arr), replace=True)
            boot_means.append(np.mean(sample))
            
        lower_pct = ((1.0 - ci) / 2.0) * 100
        upper_pct = (1.0 - (1.0 - ci) / 2.0) * 100
        ci_lower = float(np.percentile(boot_means, lower_pct))
        ci_upper = float(np.percentile(boot_means, upper_pct))
        
        return {
            "mean": mean_val,
            "std": std_val,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
        }

    @classmethod
    def aggregate_runs(
        cls,
        results_by_method: Dict[str, List[Dict[str, Any]]],
        baseline_name: str = "Baseline",
    ) -> Dict[str, Any]:
        """
        Aggregate results across seeds for multiple methods.
        Each method has a list of seed results containing 'success_rate', 'mean_steps', 'mean_latency_ms'.
        """
        summary = {}
        baseline_successes = None
        
        if baseline_name in results_by_method:
            baseline_successes = [r["success_rate"] for r in results_by_method[baseline_name]]

        for method, runs in results_by_method.items():
            successes = [r.get("success_rate", 0.0) for r in runs]
            steps = [r.get("mean_steps", 0.0) for r in runs]
            latencies = [r.get("mean_latency_ms", 0.0) for r in runs]
            
            p_val = None
            if baseline_successes is not None and method != baseline_name and len(successes) > 1:
                t_stat, p_val = stats.ttest_ind(successes, baseline_successes, equal_var=False)
                p_val = float(p_val)

            summary[method] = {
                "success_rate": cls.compute_bootstrap_ci(successes),
                "steps_to_goal": cls.compute_bootstrap_ci(steps),
                "latency_ms": cls.compute_bootstrap_ci(latencies),
                "p_value_vs_baseline": p_val,
                "n_seeds": len(runs),
            }
            
        return summary

    @staticmethod
    def export_summary_to_json(summary: Dict[str, Any], filepath: str) -> None:
        """Export summary to JSON file."""
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    @staticmethod
    def generate_latex_table(summary: Dict[str, Any]) -> str:
        """Generate LaTeX booktabs tabular fragment for report/main.tex."""
        lines = [
            r"\begin{table}[h]",
            r"\centering",
            r"\caption{Сравнение методов планирования на \texttt{antmaze-medium-navigate-v0} (5 сидов).}",
            r"\label{tab:planning_comparison}",
            r"\begin{tabular}{lccc}",
            r"\toprule",
            r"\textbf{Метод} & \textbf{Success Rate (\%)} & \textbf{Шагов до цели} & \textbf{Задержка (мс/шаг)} \\",
            r"\midrule",
        ]
        for method, stats_dict in summary.items():
            sr = stats_dict["success_rate"]
            st = stats_dict["steps_to_goal"]
            lat = stats_dict["latency_ms"]
            sr_str = f"{sr['mean']:.1f} $\\pm$ {sr['std']:.1f}"
            st_str = f"{st['mean']:.0f} $\\pm$ {st['std']:.0f}"
            lat_str = f"{lat['mean']:.2f}"
            
            # Bold if top performer
            lines.append(f"{method} & {sr_str} & {st_str} & {lat_str} \\\\")
            
        lines.extend([
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ])
        return "\n".join(lines)
