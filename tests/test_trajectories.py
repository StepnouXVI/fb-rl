import pytest
import numpy as np


def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])


def segments_intersect(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)


def count_self_intersections(trajectory_xy, min_step_dist=0.35):
    """
    Computes the exact number of self-intersections in a 2D trajectory.
    Filters micro-jitter from physics simulation steps (ant leg gait oscillations).
    """
    if len(trajectory_xy) < 4:
        return 0

    pts = [trajectory_xy[0]]
    for p in trajectory_xy[1:]:
        if np.linalg.norm(p - pts[-1]) >= min_step_dist:
            pts.append(p)
    pts = np.asarray(pts)
    n = len(pts)
    intersections = 0
    for i in range(n - 1):
        for j in range(i + 2, n - 1):
            if segments_intersect(pts[i], pts[i + 1], pts[j], pts[j + 1]):
                intersections += 1
    return intersections


def test_straight_line_zero_intersections():
    straight = np.array([[0.0, float(i)] for i in range(20)])
    assert count_self_intersections(straight) == 0


def test_l_shaped_turn_zero_intersections():
    l_path = np.array([[0.0, float(i)] for i in range(10)] + [[float(i), 10.0] for i in range(1, 10)])
    assert count_self_intersections(l_path) == 0


def test_figure_8_intersection():
    fig8 = np.array([
        [0.0, 0.0],
        [2.0, 2.0],
        [2.0, 0.0],
        [0.0, 2.0],
    ])
    assert count_self_intersections(fig8, min_step_dist=0.1) == 1


def test_loop_circle_intersection():
    t = np.linspace(0, 2.5 * np.pi, 50)
    circle = np.stack([np.cos(t), np.sin(t)], axis=-1)
    assert count_self_intersections(circle, min_step_dist=0.1) >= 1
