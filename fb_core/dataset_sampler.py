"""
DatasetSampler: Offline dataset loader and Farthest Point Sampling (FPS) landmark selector.
"""

from typing import Optional
import numpy as np
from fb_core.fb_model_wrapper import FBModelWrapper


class DatasetSampler:
    """
    Manages offline trajectory datasets and samples topological landmarks.
    # ponytail: Efficient vectorized numpy FPS without heavy graph dependencies.
    """

    def __init__(self, dataset_states: np.ndarray, seed: int = 42):
        self.states = np.asarray(dataset_states, dtype=np.float32)
        self.num_states = len(self.states)
        self.rng = np.random.default_rng(seed)

    @classmethod
    def from_ogbench(cls, env_name: str = "antmaze-medium-navigate-v0", max_samples: int = 50000, seed: int = 42):
        """Load dataset from OGBench environment."""
        try:
            import ogbench
            # Strip ogbench- prefix if present
            clean_env_name = env_name.replace("ogbench-", "")
            env, train_dataset, _ = ogbench.make_env_and_datasets(clean_env_name)
            obs = train_dataset["observations"]
            if len(obs) > max_samples:
                indices = np.random.choice(len(obs), max_samples, replace=False)
                obs = obs[indices]
            return cls(obs, seed=seed)
        except Exception as e:
            print(f"[Warning] Could not load OGBench dataset directly ({e}), creating synthetic grid dataset for testing.")
            # Create synthetic maze grid states (x, y, vx, vy, angles...)
            x = np.linspace(-10, 10, int(np.sqrt(max_samples)))
            y = np.linspace(-10, 10, int(np.sqrt(max_samples)))
            xx, yy = np.meshgrid(x, y)
            grid_xy = np.stack([xx.flatten(), yy.flatten()], axis=-1)
            # Pad with 27 additional state dimensions (antmaze has 29 state dims)
            pad = np.zeros((len(grid_xy), 27), dtype=np.float32)
            mock_obs = np.concatenate([grid_xy, pad], axis=-1)
            return cls(mock_obs, seed=seed)

    def sample_candidates(self, n_candidates: int) -> np.ndarray:
        """Sample random candidates from replay buffer."""
        n = min(n_candidates, self.num_states)
        indices = self.rng.choice(self.num_states, n, replace=False)
        return self.states[indices]

    def select_fps_landmarks(
        self,
        n_landmarks: int,
        fb_model: Optional[FBModelWrapper] = None,
        use_spatial_xy_only: bool = True,
    ) -> np.ndarray:
        """
        Farthest Point Sampling (FPS) algorithm to pick uniformly spaced landmarks across the maze.
        If fb_model is provided and use_spatial_xy_only is False, performs FPS in B(s) embedding space.
        """
        n_landmarks = min(n_landmarks, self.num_states)
        if n_landmarks <= 0:
            return np.empty((0, self.states.shape[1]), dtype=np.float32)

        # Feature representation for distance calculation
        if use_spatial_xy_only or fb_model is None:
            features = self.states[:, :2]  # Spatial 2D coordinates in maze
        else:
            features = fb_model.encode_backward(self.states)

        selected_indices = [0]
        # Distances of all points to the nearest selected landmark
        dists = np.linalg.norm(features - features[0:1], axis=-1)

        for _ in range(1, n_landmarks):
            # Select point with maximum distance to existing landmarks
            next_idx = int(np.argmax(dists))
            selected_indices.append(next_idx)
            # Update minimum distances
            new_dists = np.linalg.norm(features - features[next_idx:next_idx+1], axis=-1)
            dists = np.minimum(dists, new_dists)

        return self.states[selected_indices]
