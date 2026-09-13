"""Telemetry and metrics storage interfaces for FB-RL."""

from src.telemetry.db import TelemetryDatabase
from src.telemetry.metrics import (
    compute_cross_track_errors,
    count_spatial_self_intersections,
)
from src.telemetry.plotting import (
    build_pareto_figure,
    build_radar_figure,
    build_task_breakdown_figure,
    build_trajectory_figure,
)
from src.telemetry.profiler import ExecutionProfiler

__all__ = [
    "ExecutionProfiler",
    "TelemetryDatabase",
    "build_pareto_figure",
    "build_radar_figure",
    "build_task_breakdown_figure",
    "build_trajectory_figure",
    "compute_cross_track_errors",
    "count_spatial_self_intersections",
]
