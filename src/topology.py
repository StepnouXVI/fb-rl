"""Topological graph construction, shortest path search, and trajectory geometry utilities."""

from typing import Any, List, Optional, Tuple
import jax
import jax.numpy as jnp
import numpy as np
from scipy.sparse.csgraph import dijkstra


@jax.jit
def jit_batch_reach(agent: Any, states: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
    """Compute inner product between forward representations and target vectors."""
    f = agent.network.select("forward_repr")(states, targets, goal_encoded=True)
    if f.ndim == 3:
        f = jnp.mean(f, axis=0)
    return jnp.sum(f * targets, axis=-1)


@jax.jit
def jit_decode_latent_to_coords(latent: jnp.ndarray, landmark_latents: jnp.ndarray) -> jnp.ndarray:
    """Find landmark index with highest cosine similarity to the latent vector."""
    z = latent / (jnp.linalg.norm(latent, axis=-1, keepdims=True) + 1e-8)
    return jnp.argmax(jnp.matmul(landmark_latents, z.T))


def find_connected_landmarks(
    landmark_coords: np.ndarray,
    landmark_latents: jnp.ndarray,
    all_dist: np.ndarray,
    obs_xy: np.ndarray,
    goal_z: np.ndarray,
) -> Tuple[int, int]:
    """Find start and goal landmark indices with valid finite shortest path between them."""
    sims = np.asarray(jnp.matmul(landmark_latents, (goal_z / (np.linalg.norm(goal_z) + 1e-8)).T))
    top_goals = np.argsort(-sims)[:15]
    top_starts = np.argsort(np.linalg.norm(landmark_coords - obs_xy, axis=-1))[:20]
    for g in top_goals:
        for s in top_starts:
            if np.isfinite(all_dist[s, g]):
                return int(s), int(g)
    return int(top_starts[0]), int(top_goals[0])


def backtrack_dijkstra_path(
    all_pred: np.ndarray, start_idx: int, goal_idx: int, max_nodes: int
) -> List[int]:
    """Reconstruct sequence of node indices from Dijkstra predecessor matrix."""
    path, curr = [], goal_idx
    while curr != -9999 and curr != start_idx and len(path) <= max_nodes:
        path.append(curr)
        curr = all_pred[start_idx, curr]
    if curr == start_idx:
        path.append(start_idx)
        path.reverse()
        return path
    return [start_idx, goal_idx]


def extract_corner_waypoints(
    path_coords: List[np.ndarray], path_latents: List[Any], lookahead_dist: float
) -> Tuple[List[np.ndarray], List[Any], List[int]]:
    """Select salient corner and distance-spaced waypoints along polyline path."""
    if len(path_coords) <= 2:
        return list(path_coords), list(path_latents), list(range(len(path_coords)))
    filtered = [0]
    accum = 0.0
    for k in range(1, len(path_coords) - 1):
        v1 = path_coords[k] - path_coords[k - 1]
        v2 = path_coords[k + 1] - path_coords[k]
        l1, l2 = float(np.linalg.norm(v1)), float(np.linalg.norm(v2))
        accum += l1
        is_turn = bool(l1 > 0.1 and l2 > 0.1 and (np.dot(v1, v2) / (l1 * l2)) < 0.7)
        if accum >= lookahead_dist or is_turn:
            filtered.append(k)
            accum = 0.0
    if filtered[-1] != len(path_coords) - 1:
        filtered.append(len(path_coords) - 1)
    return [path_coords[k] for k in filtered], [path_latents[k] for k in filtered], filtered


def track_local_path_index(
    obs_xy: np.ndarray, path_coords: List[np.ndarray], current_idx: int, max_window: int = 7
) -> Tuple[int, float]:
    """Identify closest path point within a bounded search window around current pointer."""
    search_start = max(0, current_idx - 2)
    search_end = min(len(path_coords), current_idx + max_window)
    dists = [float(np.linalg.norm(obs_xy - path_coords[k])) for k in range(search_start, search_end)]
    best_offset = int(np.argmin(dists))
    return search_start + best_offset, dists[best_offset]


def project_agent_on_path(
    obs_xy: np.ndarray, path_coords: List[np.ndarray], current_seg: int, window: int, n_pts: int
) -> Tuple[int, float, np.ndarray]:
    """Project agent coordinates onto local segment window along polyline."""
    best_dist = float("inf")
    best_seg = min(max(0, current_seg), n_pts - 2)
    best_t = 0.0
    best_proj = path_coords[best_seg]

    search_start = max(0, current_seg - 2)
    search_end = min(n_pts - 1, current_seg + window)

    for k in range(search_start, search_end):
        a_pt = np.array(path_coords[k])
        b_pt = np.array(path_coords[k + 1])
        v = b_pt - a_pt
        v_norm_sq = float(np.dot(v, v))
        t = 0.0 if v_norm_sq < 1e-8 else float(np.clip(np.dot(obs_xy - a_pt, v) / v_norm_sq, 0.0, 1.0))
        proj = a_pt + t * v
        d = float(np.linalg.norm(obs_xy - proj))
        if d < best_dist:
            best_dist, best_seg, best_t, best_proj = d, k, t, proj
    return best_seg, best_t, best_proj


def march_along_path(
    path_coords: List[np.ndarray],
    best_seg: int,
    best_t: float,
    best_proj: np.ndarray,
    lookahead_dist: float,
    n_pts: int,
) -> np.ndarray:
    """March forward by lookahead_dist along polyline segments."""
    rem = float(lookahead_dist)
    curr_k = best_seg
    a_pt = np.array(path_coords[curr_k])
    b_pt = np.array(path_coords[curr_k + 1])
    v = b_pt - a_pt
    v_len = float(np.linalg.norm(v))
    rem_in_seg = (1.0 - best_t) * v_len

    if rem <= rem_in_seg:
        return best_proj + (rem / max(v_len, 1e-6)) * v

    rem -= rem_in_seg
    curr_k += 1
    while curr_k < n_pts - 1:
        a_pt = np.array(path_coords[curr_k])
        b_pt = np.array(path_coords[curr_k + 1])
        v = b_pt - a_pt
        v_len = float(np.linalg.norm(v))
        if rem <= v_len:
            return a_pt + (rem / max(v_len, 1e-6)) * v
        rem -= v_len
        curr_k += 1
    return np.array(path_coords[-1])


def compute_continuous_lookahead(
    obs_xy: np.ndarray,
    path_coords: List[np.ndarray],
    lookahead_dist: float = 2.6,
    current_seg: int = 0,
    window: int = 7,
    path_dists: Optional[List[float]] = None,
) -> Tuple[np.ndarray, int, float]:
    """Compute exact continuous sliding lookahead coordinate along polyline path."""
    n_pts = len(path_coords)
    if n_pts <= 1:
        return np.array(path_coords[0]), 0, 0.0

    best_seg, best_t, best_proj = project_agent_on_path(
        obs_xy, path_coords, current_seg, window, n_pts
    )
    if path_dists is not None and best_seg < len(path_dists):
        base_s = path_dists[best_seg]
        next_pt = path_coords[min(best_seg + 1, n_pts - 1)]
        seg_len = float(np.linalg.norm(next_pt - path_coords[best_seg]))
        s_agent = base_s + best_t * seg_len
    else:
        s_agent = float(best_seg)

    c_t = march_along_path(
        path_coords, best_seg, best_t, best_proj, lookahead_dist, n_pts
    )
    return c_t, best_seg, s_agent


def extract_curvature_angles(coords: List[np.ndarray]) -> List[float]:
    """Compute cosine of turning angles between consecutive waypoint direction vectors."""
    k_len = len(coords)
    curvs = []
    for k in range(k_len):
        if k + 2 < k_len:
            v1 = np.asarray(coords[k + 1]) - np.asarray(coords[k])
            v2 = np.asarray(coords[k + 2]) - np.asarray(coords[k + 1])
            l1, l2 = float(np.linalg.norm(v1)), float(np.linalg.norm(v2))
            cos_val = float(np.clip(np.dot(v1, v2) / (l1 * l2), -1.0, 1.0)) if (l1 > 1e-4 and l2 > 1e-4) else 1.0
            curvs.append(cos_val)
        else:
            curvs.append(1.0)
    return curvs


def check_stuck_state(
    pos_history: List[np.ndarray], obs_xy: np.ndarray, dist_to_final: float, stuck_count: int
) -> Tuple[bool, int]:
    """Monitor moving window of agent positions and detect progress stagnation."""
    pos_history.append(obs_xy.copy())
    if len(pos_history) > 40:
        pos_history.pop(0)
    is_stuck = False
    if len(pos_history) >= 40 and dist_to_final > 1.8:
        if float(np.linalg.norm(obs_xy - pos_history[0])) < 0.4:
            is_stuck = True
            stuck_count += 1
        else:
            stuck_count = max(0, stuck_count - 1)
    return is_stuck, stuck_count


class DijkstraGraph:
    """Reachability landmark graph with Dijkstra shortest path computation."""

    def __init__(
        self,
        agent: Any,
        dataset_states: np.ndarray,
        n_landmarks: int = 1000,
        max_edge_radius: float = 3.5,
        reachability_cutoff: float = 35.0,
        lookahead_dist: float = 2.6,
    ) -> None:
        """Construct landmark graph and compute all-pairs shortest paths."""
        self.agent = agent
        self.n_landmarks = min(n_landmarks, len(dataset_states))
        self.max_edge_radius = max_edge_radius
        self.reachability_cutoff = reachability_cutoff
        self.lookahead_dist = lookahead_dist

        idxs = np.random.default_rng(42).choice(
            len(dataset_states), size=self.n_landmarks, replace=False
        )
        self.dataset_states = dataset_states[idxs]
        self.landmarks = jnp.asarray(self.dataset_states)
        self.landmark_coords = np.asarray(self.dataset_states[:, :2])
        self.landmark_latents = jnp.asarray(
            agent.normalize_z(agent.network.select("backward_repr")(self.landmarks))
        )
        self.cost_matrix, self.reach_matrix = self._build_matrices()
        self.all_dist, self.all_pred = dijkstra(
            self.cost_matrix, directed=True, return_predecessors=True
        )

    def _build_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        """Compute pairwise reachability values and build filtered cost matrix."""
        n = len(self.landmarks)
        s_rep = jnp.repeat(self.landmarks, n, axis=0)
        z_tile = jnp.tile(self.landmark_latents, (n, 1))
        reach_flat = [
            np.asarray(jit_batch_reach(self.agent, s_rep[i : i + 45000], z_tile[i : i + 45000]))
            for i in range(0, len(s_rep), 45000)
        ]
        reach_m = np.concatenate(reach_flat, axis=0).reshape((n, n))
        dists_e = np.linalg.norm(
            self.landmark_coords[:, None, :] - self.landmark_coords[None, :, :], axis=-1
        )
        max_diag = float(np.max(np.diag(reach_m))) if np.max(np.diag(reach_m)) > 0 else 1.0
        cost_m = np.maximum(0.0, -np.log(np.clip(reach_m / max_diag, 1e-6, 1.0)))
        cost_m[dists_e > self.max_edge_radius] = np.inf
        cost_m[reach_m < self.reachability_cutoff] = np.inf
        np.fill_diagonal(cost_m, 0.0)
        return cost_m, reach_m

    def plan_path(
        self, obs_xy: np.ndarray, goal_z: np.ndarray
    ) -> Tuple[List[np.ndarray], List[Any], List[np.ndarray]]:
        """Find landmark path connecting current position to target latent."""
        start_idx, goal_idx = find_connected_landmarks(
            self.landmark_coords, self.landmark_latents, self.all_dist, obs_xy, goal_z
        )
        node_indices = backtrack_dijkstra_path(
            self.all_pred, start_idx, goal_idx, self.n_landmarks
        )
        coords = [self.landmark_coords[i] for i in node_indices]
        latents = [self.landmark_latents[i] for i in node_indices]
        states = [self.dataset_states[i] for i in node_indices]
        return coords, latents, states

    def get_subgoal_latent(
        self, obs: np.ndarray, goal_z: np.ndarray, step: int = 0
    ) -> np.ndarray:
        """Query teacher intention for current state toward goal."""
        subgoal_z, _ = self.get_subgoal_and_coord(obs, goal_z)
        return subgoal_z

    def get_subgoal_and_coord(
        self, obs: np.ndarray, goal_z: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Query teacher intention and target coordinate for current state toward goal."""
        coords, latents, _ = self.plan_path(obs[:2], goal_z)
        if not coords:
            return np.asarray(goal_z), np.asarray(obs[:2])
        accum, target_idx = 0.0, 0
        while target_idx < len(coords) - 1 and accum < self.lookahead_dist:
            accum += float(np.linalg.norm(coords[target_idx + 1] - coords[target_idx]))
            target_idx += 1
        dist_final = float(np.linalg.norm(obs[:2] - coords[-1]))
        is_goal = bool(target_idx >= len(coords) - 1 or dist_final <= 2.2)
        target_latent = goal_z if is_goal else latents[target_idx]
        target_coord = coords[-1] if is_goal else coords[target_idx]
        high_dist = self.agent.network.select("high_actor")(
            obs[None, :], target_latent[None, :], goal_encoded=True, temperature=0.0
        )
        subgoal_z = np.asarray(self.agent.normalize_z(high_dist.mode())[0])
        return subgoal_z, np.asarray(target_coord)

