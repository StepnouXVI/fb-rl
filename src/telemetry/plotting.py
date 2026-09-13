"""Interactive Plotly figure builders for RL trajectory visualization and benchmark comparisons."""

import json
from typing import Any, Dict, List, Optional
import plotly.graph_objects as go


def _append_attention_tokens(fig: go.Figure, step_dict: Dict[str, Any]) -> None:
    """Extract and append attention tokens with softmax weights to Plotly figure."""
    raw_targets = step_dict.get("attention_targets")
    raw_weights = step_dict.get("attention_weights")
    if not raw_targets:
        return
    try:
        targets = json.loads(raw_targets) if isinstance(raw_targets, str) else raw_targets
        weights = json.loads(raw_weights) if isinstance(raw_weights, str) else raw_weights
        if targets and weights:
            at_x = [t[0] for t in targets]
            at_y = [t[1] for t in targets]
            at_sz = [max(8, int(w * 35)) for w in weights]
            txt = [f"Weight: {w:.3f}" for w in weights]
            fig.add_trace(go.Scatter(
                x=at_x, y=at_y, mode="markers+text", name="Transformer Attention Tokens",
                marker=dict(size=at_sz, color=weights, colorscale="Viridis", showscale=True, colorbar=dict(title="Attention")),
                text=txt, textposition="bottom right",
            ))
    except Exception:
        pass


def build_trajectory_figure(
    method_name: str,
    path_coords: Optional[List[List[float]]] = None,
    traj_steps: Optional[List[Dict[str, Any]]] = None,
) -> go.Figure:
    """Generate interactive Plotly figure with planned path, trajectory, and attention targets."""
    fig = go.Figure()
    if path_coords:
        px = [p[0] for p in path_coords]
        py = [p[1] for p in path_coords]
        fig.add_trace(go.Scatter(
            x=px, y=py, mode="lines+markers", name="Dijkstra Planned Path",
            line=dict(color="#FF9800", width=2, dash="dash"),
            marker=dict(size=5, color="#FF9800"),
        ))
    if traj_steps:
        tx = [s["x"] for s in traj_steps]
        ty = [s["y"] for s in traj_steps]
        fig.add_trace(go.Scatter(
            x=tx, y=ty, mode="lines", name="Ant Trajectory",
            line=dict(color="#00E5FF", width=3),
        ))
        fig.add_trace(go.Scatter(
            x=[tx[0]], y=[ty[0]], mode="markers+text", name="Start (s0)",
            marker=dict(size=12, color="#00FF00", symbol="circle"),
            text=["Start"], textposition="top center",
        ))
        fig.add_trace(go.Scatter(
            x=[tx[-1]], y=[ty[-1]], mode="markers+text", name="Final Position",
            marker=dict(size=12, color="#FF1744", symbol="star"),
            text=["Goal Reach"], textposition="top center",
        ))
        mid_idx = len(traj_steps) // 3
        _append_attention_tokens(fig, traj_steps[mid_idx])

    fig.update_layout(
        title=f"Trajectory & Navigation Map: {method_name}",
        xaxis=dict(title="X Coordinate (meters)", gridcolor="#333333"),
        yaxis=dict(title="Y Coordinate (meters)", gridcolor="#333333", scaleanchor="x", scaleratio=1),
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig


def build_pareto_figure(
    methods: List[str],
    latencies: List[float],
    success_rates: List[float],
    stds: List[float],
) -> go.Figure:
    """Construct interactive Pareto frontier chart of latency versus success rate."""
    colors = ["#00E5FF", "#FF9800", "#76FF03", "#E040FB", "#FFD600"]
    fig = go.Figure()
    for i, m in enumerate(methods):
        fig.add_trace(go.Scatter(
            x=[latencies[i]], y=[success_rates[i]],
            mode="markers+text", name=m,
            error_y=dict(type="data", array=[stds[i]], visible=True),
            marker=dict(size=18, color=colors[i % len(colors)], line=dict(color="#FFFFFF", width=2)),
            text=[f"{m}<br>{success_rates[i]:.1f}% | {latencies[i]:.2f}ms"],
            textposition="top center",
        ))
    fig.update_layout(
        title="Pareto Trade-Off: Inference Latency (ms) vs. Success Rate (%)",
        xaxis=dict(title="Latency per Step (ms)", gridcolor="#333333"),
        yaxis=dict(title="Overall Success Rate (%)", range=[60, 100], gridcolor="#333333"),
        template="plotly_dark",
        margin=dict(l=50, r=50, t=60, b=50),
    )
    return fig


def build_task_breakdown_figure(
    methods: List[str],
    task_matrix: Dict[str, List[float]],
) -> go.Figure:
    """Construct grouped bar chart comparing performance across 5 maze tasks."""
    fig = go.Figure()
    for m in methods:
        fig.add_trace(go.Bar(
            name=m, x=[f"Task {t}" for t in range(1, len(task_matrix[m]) + 1)], y=task_matrix[m],
        ))
    fig.update_layout(
        barmode="group",
        title="Task-by-Task Success Rate (%) Comparison",
        xaxis=dict(title="Maze Task ID", gridcolor="#333333"),
        yaxis=dict(title="Success Rate (%)", range=[0, 105], gridcolor="#333333"),
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig
