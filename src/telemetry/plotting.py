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
                x=at_x, y=at_y, mode="markers+text", name="Attention Tokens",
                marker=dict(size=at_sz, color=weights, colorscale="Viridis", showscale=True, colorbar=dict(title="Attention")),
                text=txt, textposition="bottom right",
            ))
    except Exception:
        pass


def build_trajectory_figure(
    method_name: str,
    path_coords: Optional[List[List[float]]] = None,
    traj_steps: Optional[List[Dict[str, Any]]] = None,
    is_success: Optional[bool] = None,
) -> go.Figure:
    """Generate interactive Plotly figure with planned path, trajectory, and attention targets."""
    fig = go.Figure()
    if path_coords:
        px = [p[0] for p in path_coords]
        py = [p[1] for p in path_coords]
        fig.add_trace(go.Scatter(
            x=px, y=py, mode="lines+markers", name="Dijkstra Path",
            line=dict(color="#DD8452", width=2, dash="dash"),
            marker=dict(size=5, color="#DD8452"),
        ))
    if traj_steps:
        tx = [s["x"] for s in traj_steps]
        ty = [s["y"] for s in traj_steps]
        ant_color = "#55A868" if is_success is True else ("#C44E52" if is_success is False else "#4C72B0")
        fig.add_trace(go.Scatter(
            x=tx, y=ty, mode="lines", name="Ant Trajectory",
            line=dict(color=ant_color, width=3),
        ))
        fig.add_trace(go.Scatter(
            x=[tx[0]], y=[ty[0]], mode="markers+text", name="Start (s0)",
            marker=dict(size=12, color="#2CA02C", symbol="circle"),
            text=["Start"], textposition="top center",
        ))
        fin_sym = "star" if is_success is True else "x"
        fin_name = "Goal Reached" if is_success is True else "End Position"
        fig.add_trace(go.Scatter(
            x=[tx[-1]], y=[ty[-1]], mode="markers+text", name=fin_name,
            marker=dict(size=14, color="#D62728", symbol=fin_sym),
            text=[fin_name], textposition="top center",
        ))
        mid_idx = len(traj_steps) // 3
        _append_attention_tokens(fig, traj_steps[mid_idx])

    status_str = f" [{'SUCCESS' if is_success else 'FAILED'}]" if is_success is not None else ""
    fig.update_layout(
        title=f"Trajectory: {method_name}{status_str}",
        xaxis=dict(title="X Coordinate (meters)", gridcolor="#EAEAF2"),
        yaxis=dict(title="Y Coordinate (meters)", gridcolor="#EAEAF2", scaleanchor="x", scaleratio=1),
        template="seaborn",
        width=800,
        height=600,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig


def build_pareto_figure(
    methods: List[str],
    latencies: List[float],
    success_rates: List[float],
    stds: List[float],
    title: str = "Pareto Trade-Off: Latency vs. Success Rate",
) -> go.Figure:
    """Construct interactive Pareto frontier chart of latency versus success rate."""
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172D5", "#CCB974"]
    fig = go.Figure()
    for i, m in enumerate(methods):
        fig.add_trace(go.Scatter(
            x=[latencies[i]], y=[success_rates[i]],
            mode="markers", name=m,
            error_y=dict(type="data", array=[stds[i]], visible=True),
            marker=dict(size=16, color=colors[i % len(colors)], line=dict(color="#333333", width=1.5)),
            text=[m],
            customdata=[stds[i]],
            hovertemplate="<b>%{text}</b><br>Success Rate: %{y:.1f}% ± %{customdata:.1f}%<br>Latency: %{x:.2f} ms<extra></extra>",
        ))
    y_min = max(0.0, min(success_rates) - 10.0)
    y_max = min(100.0, max(success_rates) + 10.0)
    fig.update_layout(
        title=title,
        xaxis=dict(title="Inference Latency per Step (ms)", gridcolor="#EAEAF2"),
        yaxis=dict(title="Overall Success Rate (%)", range=[y_min, y_max], gridcolor="#EAEAF2"),
        template="seaborn",
        width=800,
        height=600,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=50, t=60, b=50),
    )
    return fig


def build_task_breakdown_figure(
    methods: List[str],
    task_matrix: Dict[str, List[float]],
    title: str = "Task-by-Task Success Rate (%) Comparison",
) -> go.Figure:
    """Construct grouped bar chart comparing performance across maze tasks."""
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172D5", "#CCB974"]
    fig = go.Figure()
    for i, m in enumerate(methods):
        fig.add_trace(go.Bar(
            name=m, x=[f"Task {t}" for t in range(1, len(task_matrix[m]) + 1)],
            y=task_matrix[m], marker_color=colors[i % len(colors)],
        ))
    fig.update_layout(
        barmode="group",
        title=title,
        xaxis=dict(title="Maze Task ID", gridcolor="#EAEAF2"),
        yaxis=dict(title="Success Rate (%)", range=[0, 105], gridcolor="#EAEAF2"),
        template="seaborn",
        width=800,
        height=600,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig


def build_radar_figure(
    methods: List[str],
    categories: List[str],
    values_matrix: Dict[str, List[float]],
    title: str = "Multi-Metric Architecture Profile Comparison",
) -> go.Figure:
    """Construct radar chart comparing methods across multiple performance axes."""
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172D5", "#CCB974"]
    fig = go.Figure()
    for i, m in enumerate(methods):
        vals = list(values_matrix[m])
        vals.append(vals[0])
        cats = list(categories)
        cats.append(cats[0])
        fig.add_trace(go.Scatterpolar(
            r=vals, theta=cats, name=m,
            fill="toself",
            line=dict(color=colors[i % len(colors)], width=2),
            opacity=0.6,
        ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        title=title,
        template="seaborn",
        width=800,
        height=600,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig
