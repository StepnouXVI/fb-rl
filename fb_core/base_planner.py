"""
Base Abstract Class for Hierarchical Planners in FB-RL.
Enforces the DRY principle across all branches (Baseline, Recursive Bisection, Graph Dijkstra, Distilled MLP).
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import numpy as np


class BaseHierarchicalPlanner(ABC):
    """
    Abstract interface for high-level controllers / planners.
    # ponytail: Keep interface lean - only reset, get_intention, and name properties.
    """

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.config = config or {}
        self.current_goal: Optional[np.ndarray] = None
        self.step_count: int = 0

    @abstractmethod
    def reset(self, initial_obs: np.ndarray, goal: np.ndarray) -> None:
        """Reset the planner state at the beginning of an episode."""
        self.current_goal = goal
        self.step_count = 0

    @abstractmethod
    def get_intention(self, obs: np.ndarray, goal: np.ndarray, step: int = 0) -> np.ndarray:
        """
        Given current observation s and global goal g, return latent intention z for low-level policy.
        z must be a numpy vector of shape (latent_dim,) normalized such that ||z|| = sqrt(latent_dim).
        """
        pass

    def on_step_end(self, obs: np.ndarray, reward: float, done: bool, info: Dict[str, Any]) -> None:
        """Optional hook called after environment step for online adaptation/replanning."""
        self.step_count += 1

    def get_diagnostics(self) -> Dict[str, Any]:
        """Return diagnostic metrics (number of replans, current subgoal index, etc.)."""
        return {"step_count": self.step_count}
