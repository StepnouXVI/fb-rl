"""
BufferGraphPlanner: Topological Graph Search on Replay Buffer Landmarks.
Builds a reachability graph using FB representations and finds global shortest path via Dijkstra.
"""

from typing import Optional, Dict, Any, List
import numpy as np
from scipy.sparse.csgraph import dijkstra
from fb_core.base_planner import BaseHierarchicalPlanner
from fb_core.fb_model_wrapper import FBModelWrapper
from fb_core.dataset_sampler import DatasetSampler
from fb_core.math_utils import normalize_latent


class BufferGraphPlanner(BaseHierarchicalPlanner):
    """
    Branch 2: Topological Graph Search on Replay Buffer Landmarks.
    # ponytail: Use scipy.sparse.csgraph.dijkstra for fast C-optimized graph search.
    """

    def __init__(
        self,
        fb_model: FBModelWrapper,
        dataset_sampler: DatasetSampler,
        name: str = "Buffer Graph Dijkstra (Branch 2)",
        n_landmarks: int = 150,
        reachability_cutoff: float = 0.05,
        switch_distance: float = 1.0,
        replan_interval: int = 100,
        latent_dim: int = 128,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(name=name, config=config)
        self.fb_model = fb_model
        self.dataset_sampler = dataset_sampler
        self.n_landmarks = n_landmarks
        self.reachability_cutoff = reachability_cutoff
        self.switch_distance = switch_distance
        self.replan_interval = replan_interval
        self.latent_dim = latent_dim

        self.landmarks: Optional[np.ndarray] = None
        self.adj_matrix: Optional[np.ndarray] = None
        self.path_waypoints: List[np.ndarray] = []
        self.current_waypoint_idx: int = 0

    def build_graph(self) -> None:
        """Construct the topological graph over FPS landmarks."""
        if self.landmarks is not None:
            return

        # 1. Select landmarks
        self.landmarks = self.dataset_sampler.select_fps_landmarks(
            n_landmarks=self.n_landmarks,
            fb_model=self.fb_model,
            use_spatial_xy_only=True,
        )
        n = len(self.landmarks)

        # 2. Vectorized pairwise reachability matrix
        reach_matrix = self.fb_model.compute_batch_reachability(self.landmarks, self.landmarks)
        
        # 3. Cost matrix: c(i, j) = max(0.0, -log(reach_matrix[i, j] / max_diag))
        # ponytail: Normalize by maximum diagonal reachability so all edge costs are non-negative.
        max_diag = np.max(np.diag(reach_matrix)) if np.max(np.diag(reach_matrix)) > 0 else 1.0
        normalized_reach = np.clip(reach_matrix / max_diag, a_min=1e-6, a_max=1.0)
        cost_matrix = -np.log(normalized_reach)
        cost_matrix = np.maximum(0.0, cost_matrix)
        
        # Cut off edges with reachability below threshold or excessive cost
        cutoff_mask = reach_matrix < self.reachability_cutoff
        np.fill_diagonal(cutoff_mask, False)
        cost_matrix[cutoff_mask] = np.inf
        np.fill_diagonal(cost_matrix, 0.0)

        self.adj_matrix = cost_matrix

    def _find_shortest_path(self, start_obs: np.ndarray, goal: np.ndarray) -> List[np.ndarray]:
        """Insert start and goal into graph and compute shortest path with Dijkstra."""
        self.build_graph()
        assert self.landmarks is not None and self.adj_matrix is not None

        # Find closest landmark to start and goal
        start_dists = np.linalg.norm(self.landmarks[:, :2] - start_obs[:2], axis=-1)
        goal_dists = np.linalg.norm(self.landmarks[:, :2] - goal[:2], axis=-1)

        start_idx = int(np.argmin(start_dists))
        goal_idx = int(np.argmin(goal_dists))

        if start_idx == goal_idx:
            return [goal]

        # Run Dijkstra
        dist_matrix, predecessors = dijkstra(
            csgraph=self.adj_matrix,
            directed=True,
            indices=start_idx,
            return_predecessors=True,
        )

        # Reconstruct path from predecessors
        path_indices = []
        curr = goal_idx
        while curr != -9999 and curr != start_idx:
            path_indices.append(curr)
            curr = predecessors[curr]
            if len(path_indices) > len(self.landmarks):  # Cycle protection
                break

        if curr == start_idx:
            path_indices.append(start_idx)
            path_indices.reverse()
            waypoints = [self.landmarks[idx] for idx in path_indices]
            waypoints.append(goal)
            return waypoints

        # Fallback if disconnected: direct to closest landmark then goal
        return [self.landmarks[start_idx], self.landmarks[goal_idx], goal]

    def reset(self, initial_obs: np.ndarray, goal: np.ndarray) -> None:
        super().reset(initial_obs, goal)
        self.path_waypoints = self._find_shortest_path(initial_obs, goal)
        self.current_waypoint_idx = 0

    def get_intention(self, obs: np.ndarray, goal: np.ndarray, step: int = 0) -> np.ndarray:
        if not self.path_waypoints or self.current_goal is None or not np.array_equal(goal, self.current_goal):
            self.reset(obs, goal)

        if step > 0 and step % self.replan_interval == 0:
            self.reset(obs, goal)

        curr_waypoint = self.path_waypoints[self.current_waypoint_idx]
        dir_vec = curr_waypoint[:2] - obs[:2]
        dist = float(np.linalg.norm(dir_vec))

        if dist < self.switch_distance and self.current_waypoint_idx < len(self.path_waypoints) - 1:
            self.current_waypoint_idx += 1
            curr_waypoint = self.path_waypoints[self.current_waypoint_idx]
            dir_vec = curr_waypoint[:2] - obs[:2]
            dist = float(np.linalg.norm(dir_vec))

        z = np.zeros(self.latent_dim, dtype=np.float32)
        if dist > 1e-4:
            z[:2] = dir_vec / dist
        else:
            z[:2] = dir_vec
        return normalize_latent(z, latent_dim=self.latent_dim)
