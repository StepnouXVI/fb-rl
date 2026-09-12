"""Context structures for staged reinforcement learning agent pipelines."""

from dataclasses import dataclass, field
from typing import Any, List, Optional
import numpy as np


@dataclass
class BaseMazeContext:
    """Base context holding state, action, commands, and telemetry for maze agents."""

    obs: Optional[np.ndarray] = None
    goal_latent: Optional[np.ndarray] = None
    step: int = 0
    seed: Any = None
    temperature: float = 0.0
    action: Optional[np.ndarray] = None
    z_cmd: Optional[np.ndarray] = None
    reward: float = 0.0
    done: bool = False
    latency_ms: float = 0.0
    db: Any = None
    episode_id: Optional[str] = None

    @property
    def pos_xy(self) -> Optional[np.ndarray]:
        """Extract 2D planar position (x, y) from the current observation."""
        if self.obs is not None and len(self.obs) >= 2:
            return np.asarray(self.obs[:2], dtype=np.float32)
        return None


@dataclass
class TopologicalPathContext(BaseMazeContext):
    """Context extended with topological path planning states and waypoints."""

    path_coords: List[np.ndarray] = field(default_factory=list)
    path_latents: List[np.ndarray] = field(default_factory=list)
    waypoint_coords: List[np.ndarray] = field(default_factory=list)
    waypoint_indices: List[int] = field(default_factory=list)
    current_path_idx: int = 0
    replan_triggered: bool = False
    lookahead_xy: Optional[np.ndarray] = None
    dist_to_lookahead: Optional[float] = None
    path_states: List[np.ndarray] = field(default_factory=list)
    path_dists: List[float] = field(default_factory=list)
    target_latent: Optional[np.ndarray] = None
    is_direct_goal: bool = False
    stuck_count: int = 0


@dataclass
class SequenceAttentionContext(TopologicalPathContext):
    """Context extended with sequence attention targets, weights, and tensors."""

    attention_targets: List[np.ndarray] = field(default_factory=list)
    attention_latents: List[np.ndarray] = field(default_factory=list)
    attention_weights: List[float] = field(default_factory=list)
    pad_seq: Optional[np.ndarray] = None
    seq_mask: Optional[np.ndarray] = None
    curv_arr: Optional[np.ndarray] = None
