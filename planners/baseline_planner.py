"""
BaselinePlanner: Single-Intention High-Level Controller.
Chooses a single intention direction z ~ B(g) or via the baseline high-level policy.
"""

from typing import Optional, Dict, Any
import numpy as np
from fb_core.base_planner import BaseHierarchicalPlanner
from fb_core.fb_model_wrapper import FBModelWrapper
from fb_core.math_utils import normalize_latent


class SingleIntentionPlanner(BaseHierarchicalPlanner):
    """
    Baseline Single-Intention High-Level Controller.
    # ponytail: Straightforward goal encoding without multi-subgoal chaining.
    """

    def __init__(
        self,
        fb_model: FBModelWrapper,
        name: str = "Single-Intention Baseline",
        latent_dim: int = 128,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(name=name, config=config)
        self.fb_model = fb_model
        self.latent_dim = latent_dim
        self.cached_intention: Optional[np.ndarray] = None

    def reset(self, initial_obs: np.ndarray, goal: np.ndarray) -> None:
        super().reset(initial_obs, goal)
        # Compute single target intention toward global goal
        b_goal = self.fb_model.encode_backward(goal)
        self.cached_intention = normalize_latent(b_goal, latent_dim=self.latent_dim)

    def get_intention(self, obs: np.ndarray, goal: np.ndarray, step: int = 0) -> np.ndarray:
        if self.cached_intention is None or self.current_goal is None or not np.array_equal(goal, self.current_goal):
            self.reset(obs, goal)
        # Direction from current observation to goal
        dir_vec = goal[:2] - obs[:2]
        dist = np.linalg.norm(dir_vec)
        z = np.zeros(self.latent_dim, dtype=np.float32)
        if dist > 1e-4:
            z[:2] = dir_vec / dist
        else:
            z[:2] = dir_vec
        return normalize_latent(z, latent_dim=self.latent_dim)
