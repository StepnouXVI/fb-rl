import numpy as np


def _vectorized_point_to_segments(point, seg_a, seg_b):
    """Vectorized distance from a single 2D point to multiple line segments."""
    v = seg_b - seg_a
    v_sq = np.sum(v * v, axis=-1)
    u = point - seg_a
    uv = np.sum(u * v, axis=-1)
    t = np.where(v_sq < 1e-12, 0.0, np.clip(uv / np.maximum(v_sq, 1e-12), 0.0, 1.0))
    proj = seg_a + t[:, None] * v
    return np.linalg.norm(point - proj, axis=-1)


def compute_cross_track_errors(traj_xy, path_xy, window_size: int = 4):
    """Vectorized cross-track error to polyline path segments with a local anti-wall-penetration window."""
    traj = np.asarray(traj_xy, dtype=np.float64)
    path = np.asarray(path_xy, dtype=np.float64)
    if len(traj) == 0:
        return np.empty((0,), dtype=np.float64)
    if len(path) < 2:
        if len(path) == 1:
            return np.linalg.norm(traj - path[0], axis=-1)
        return np.zeros(len(traj), dtype=np.float64)

    seg_a = path[:-1]
    seg_b = path[1:]
    num_segments = len(seg_a)
    errors = np.empty(len(traj), dtype=np.float64)
    curr_seg = 0

    for i, point in enumerate(traj):
        start = max(0, curr_seg - 1)
        end = min(num_segments, curr_seg + window_size)
        if end <= start:
            end = min(num_segments, start + 1)
        dists = _vectorized_point_to_segments(point, seg_a[start:end], seg_b[start:end])
        best_offset = int(np.argmin(dists))
        errors[i] = dists[best_offset]
        curr_seg = start + best_offset

    return errors


def _ccw(a, b, c):
    """Determine counter-clockwise orientation of three 2D points."""
    return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])


def _segments_intersect(a, b, c, d):
    """Check if line segment ab intersects line segment cd using CCW predicate."""
    return (_ccw(a, c, d) != _ccw(b, c, d)) and (_ccw(a, b, c) != _ccw(a, b, d))


def _aabb_overlap(b1, b2):
    """Determine if two 2D axis-aligned bounding boxes overlap."""
    return not (b1[1] < b2[0] or b1[0] > b2[1] or b1[3] < b2[2] or b1[2] > b2[3])


def _filter_trajectory(traj, min_step_dist):
    """Filter out consecutive points closer than min_step_dist to eliminate micro-jitter."""
    if len(traj) == 0:
        return np.empty((0, 2), dtype=np.float64)
    pts = [traj[0]]
    for p in traj[1:]:
        if np.linalg.norm(p - pts[-1]) >= min_step_dist:
            pts.append(p)
    return np.asarray(pts, dtype=np.float64)


def _compute_segment_aabbs(pts):
    """Compute axis-aligned bounding boxes for each segment in trajectory."""
    boxes = []
    for k in range(len(pts) - 1):
        p1, p2 = pts[k], pts[k + 1]
        boxes.append((
            min(p1[0], p2[0]), max(p1[0], p2[0]),
            min(p1[1], p2[1]), max(p1[1], p2[1])
        ))
    return boxes


def count_spatial_self_intersections(traj_xy, min_step_dist: float = 0.35):
    """Count self-intersections of 2D trajectory using AABB pruning and CCW predicate."""
    traj = np.asarray(traj_xy, dtype=np.float64)
    pts = _filter_trajectory(traj, min_step_dist)
    n = len(pts)
    if n < 4:
        return 0

    boxes = _compute_segment_aabbs(pts)
    intersections = 0
    for i in range(n - 1):
        box_i = boxes[i]
        p_i, p_ip1 = pts[i], pts[i + 1]
        for j in range(i + 2, n - 1):
            if not _aabb_overlap(box_i, boxes[j]):
                continue
            if _segments_intersect(p_i, p_ip1, pts[j], pts[j + 1]):
                intersections += 1

    return intersections
