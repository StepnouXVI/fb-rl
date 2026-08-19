import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# ponytail: Minimal publication plot generator with Matplotlib PNGs and Plotly HTML
def generate_plots(results_dir="results", output_dir="results/plots"):
    os.makedirs(output_dir, exist_ok=True)
    metrics_path = os.path.join(results_dir, "summary_metrics.json")
    if not os.path.exists(metrics_path):
        print(f"Metrics file {metrics_path} not found.")
        return

    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    methods = list(metrics.keys())
    success_means = [metrics[m]["success_rate"]["mean"] for m in methods]
    success_stds = [metrics[m]["success_rate"]["std"] for m in methods]
    latencies = [metrics[m]["latency_ms"]["mean"] for m in methods]
    steps_means = [metrics[m]["mean_length"]["mean"] for m in methods]

    # 1. Success Rate Comparison Bar Chart
    plt.figure(figsize=(9, 5), dpi=300)
    bars = plt.bar(methods, success_means, yerr=success_stds, capsize=6, color=["#4A90E2", "#50E3C2", "#F5A623", "#9013FE"], alpha=0.85)
    plt.ylabel("Success Rate (%)", fontsize=12, fontweight="bold")
    plt.title("Zero-Shot Multi-Subgoal FB Planning on AntMaze-Medium", fontsize=14, fontweight="bold")
    plt.ylim(0, 100)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    for bar, mean in zip(bars, success_means):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, f"{mean:.1f}%", ha="center", fontweight="bold")
    plt.xticks(rotation=15, ha="right", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "success_rate_comparison.png"))
    plt.close()

    # 2. Latency vs Success Rate Pareto Plot (Matplotlib)
    plt.figure(figsize=(8, 5), dpi=300)
    colors = ["#4A90E2", "#50E3C2", "#F5A623", "#9013FE"]
    for i, method in enumerate(methods):
        plt.scatter(latencies[i], success_means[i], color=colors[i], s=160, label=method, edgecolors="black", zorder=3)
        plt.annotate(method, (latencies[i], success_means[i] + 2), fontsize=9, fontweight="bold", ha="center")
    plt.xlabel("Inference Latency (ms / step)", fontsize=12, fontweight="bold")
    plt.ylabel("Success Rate (%)", fontsize=12, fontweight="bold")
    plt.title("Latency vs. Success Rate Trade-off", fontsize=14, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pareto_latency_accuracy.png"))
    plt.close()

    # 3. Interactive Plotly HTML
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=latencies,
        y=success_means,
        mode="markers+text",
        text=methods,
        textposition="top center",
        marker=dict(size=14, color=colors, line=dict(width=1, color="black")),
    ))
    fig.update_layout(
        title="Accuracy vs. Latency Pareto Trade-off",
        xaxis_title="Inference Latency (ms / step)",
        yaxis_title="Success Rate (%)",
        template="plotly_white",
        font=dict(size=12),
    )
    fig.write_html(os.path.join(output_dir, "pareto_latency_accuracy.html"))

    print(f"Publication-quality plots successfully saved to {output_dir}/")

if __name__ == "__main__":
    generate_plots()
