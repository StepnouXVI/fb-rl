"""
RecursiveBisectionPlanner: Divide-and-Conquer Multi-Subgoal Planner.
Recursively decomposes the trajectory s -> g by finding the highest log-successor density midpoints.
"""

from typing import Optional, Dict, Any, List
import numpy as np
from fb_core.base_planner import BaseHierarchicalPlanner
from fb_core.fb_model_wrapper import FBModelWrapper
from fb_core.dataset_sampler import DatasetSampler
from fb_core.math_utils import normalize_latent, log_successor_bisection_score


class RecursiveBisectionPlanner(BaseHierarchicalPlanner):
    """
    Branch 1: Recursive Bisection Planner (Divide & Conquer).
    # ponytail: Vectorized numpy argmax over sampled buffer candidates.
    """

    def __init__(
        self,
        fb_model: FBModelWrapper,
        dataset_sampler: DatasetSampler,
        name: str = "Recursive Bisection (Branch 1)",
        max_depth: int = 2,
        n_candidates: int = 300,
        switch_distance: float = 1.0,
        replan_interval: int = 50,
        latent_dim: int = 128,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(name=name, config=config)
        self.fb_model = fb_model
        self.dataset_sampler = dataset_sampler
        self.max_depth = max_depth
        self.n_candidates = n_candidates
        self.switch_distance = switch_distance
        self.replan_interval = replan_interval
        self.latent_dim = latent_dim

        self.subgoals: List[np.ndarray] = []
        self.current_subgoal_idx: int = 0
        self.candidate_states: Optional[np.ndarray] = None
        self.candidate_b_embeds: Optional[np.ndarray] = None

    def _prepare_candidates(self) -> None:
        """Sample and cache candidate subgoals from replay buffer."""
        if self.candidate_states is None:
            self.candidate_states = self.dataset_sampler.sample_candidates(self.n_candidates)
            self.candidate_b_embeds = self.fb_model.encode_backward(self.candidate_states)

    def _find_best_midpoint(self, s: np.ndarray, g: np.ndarray) -> np.ndarray:
        """
        Find w* = argmax_{w in D} [log M(s -> w) + log M(w -> g)]
        """
        self._prepare_candidates()
        assert self.candidate_states is not None and self.candidate_b_embeds is not None

        b_g = self.fb_model.encode_backward(g)
        n = len(self.candidate_states)

        # 1. Forward reachability from s to each candidate w
        s_expanded = np.tile(s[None, :], (n, 1))
        f_sw = self.fb_model.evaluate_forward(s_expanded, self.candidate_b_embeds)

        # 2. Forward reachability from each candidate w to goal g
        b_g_expanded = np.tile(b_g[None, :], (n, 1))
        f_wg = self.fb_model.evaluate_forward(self.candidate_states, b_g_expanded)

        # 3. Compute joint log-bisection score
        scores = log_successor_bisection_score(
            f_sw=f_sw,
            b_w=self.candidate_b_embeds,
            f_wg=f_wg,
            b_g=b_g_expanded,
        )

        best_idx = int(np.argmax(scores))
        return self.candidate_states[best_idx]

    def _recursive_plan(self, s: np.ndarray, g: np.ndarray, depth: int) -> List[np.ndarray]:
        """Recursive subdivision of the trajectory."""
        if depth >= self.max_depth:
            return []

        midpoint = self._find_best_midpoint(s, g)
        
        # Recurse on left interval [s, midpoint] and right interval [midpoint, g]
        left_subgoals = self._recursive_plan(s, midpoint, depth + 1)
        right_subgoals = self._recursive_plan(midpoint, g, depth + 1)

        return left_subgoals + [midpoint] + right_subgoals

    def reset(self, initial_obs: np.ndarray, goal: np.ndarray) -> None:
        super().reset(initial_obs, goal)
        self.subgoals = self._recursive_plan(initial_obs, goal, depth=0)
        self.subgoals.append(goal)
        self.current_subgoal_idx = 0

    def get_intention(self, obs: np.ndarray, goal: np.ndarray, step: int = 0) -> np.ndarray:
        if not self.subgoals or self.current_goal is None or not np.array_equal(goal, self.current_goal):
            self.reset(obs, goal)

        # Periodic replan or check if stuck
        if step > 0 and step % self.replan_interval == 0:
            self.reset(obs, goal)

        # Advance subgoal if close enough
        curr_target = self.subgoals[self.current_subgoal_idx]
        dir_vec = curr_target[:2] - obs[:2]
        dist = float(np.linalg.norm(dir_vec))
        
        if dist < self.switch_distance and self.current_subgoal_idx < len(self.subgoals) - 1:
            self.current_subgoal_idx += 1
            curr_target = self.subgoals[self.current_subgoal_idx]
            dir_vec = curr_target[:2] - obs[:2]
            dist = float(np.linalg.norm(dir_vec))

        z = np.zeros(self.latent_dim, dtype=np.float32)
        if dist > 1e-4:
            z[:2] = dir_vec / dist
        else:
            z[:2] = dir_vec
        return normalize_latent(z, latent_dim=self.latent_dim)
