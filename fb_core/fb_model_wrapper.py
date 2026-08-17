"""
FBModelWrapper: Frozen FB representations and policy bridge.
Provides simple numpy interfaces for Forward (F), Backward (B), and Low-Level Actor (pi_l).
"""

from typing import Optional, Any
import numpy as np
import jax
import jax.numpy as jnp
from fb_core.math_utils import normalize_latent, safe_inner_product


class FBModelWrapper:
    """
    Wrapper for frozen Forward-Backward models and low-level execution policies.
    # ponytail: Clean bridge between JAX/Flax agent parameters and unified numpy planner APIs.
    """

    def __init__(
        self,
        agent: Optional[Any] = None,
        latent_dim: int = 128,
        device: str = "cpu",
    ):
        self.agent = agent
        self.latent_dim = latent_dim
        self.device = device
        self.rng = jax.random.PRNGKey(42)

    def encode_backward(self, states: np.ndarray) -> np.ndarray:
        """
        Compute backward representation B(s) for states.
        states: shape (..., obs_dim)
        returns: shape (..., latent_dim)
        """
        if self.agent is not None and hasattr(self.agent, "network"):
            # Use real JAX backward representation
            states_jnp = jnp.asarray(states)
            if states_jnp.ndim == 1:
                b_repr = self.agent.network.select("backward_repr")(states_jnp[None, :])
                b_repr = np.asarray(b_repr)[0]
            else:
                b_repr = self.agent.network.select("backward_repr")(states_jnp)
                b_repr = np.asarray(b_repr)
            return normalize_latent(b_repr, latent_dim=self.latent_dim)
        
        # Fallback / mock for isolated testing & standalone navigation
        flat_states = np.asarray(states)
        if flat_states.ndim == 1:
            pseudo_b = np.zeros(self.latent_dim, dtype=np.float32)
            pseudo_b[:2] = flat_states[:2]
            pseudo_b[2:] = np.sin(np.linspace(1, 10, self.latent_dim - 2) * flat_states[0] + flat_states[-1])
        else:
            pseudo_b = np.zeros((len(flat_states), self.latent_dim), dtype=np.float32)
            pseudo_b[:, :2] = flat_states[:, :2]
            pseudo_b[:, 2:] = np.sin(np.outer(flat_states[:, 0], np.linspace(1, 10, self.latent_dim - 2)))
        return normalize_latent(pseudo_b, latent_dim=self.latent_dim)

    def evaluate_forward(self, states: np.ndarray, latents: np.ndarray) -> np.ndarray:
        """
        Compute forward representation F(s, z).
        states: shape (N, obs_dim)
        latents: shape (N, latent_dim) or (1, latent_dim)
        returns: shape (N, latent_dim)
        """
        latents_norm = normalize_latent(latents, latent_dim=self.latent_dim)
        
        if self.agent is not None and hasattr(self.agent, "network"):
            states_jnp = jnp.asarray(states)
            latents_jnp = jnp.asarray(latents_norm)
            if states_jnp.ndim == 1:
                states_jnp = states_jnp[None, :]
            if latents_jnp.ndim == 1:
                latents_jnp = latents_jnp[None, :]
                
            f_repr = self.agent.network.select("forward_repr")(states_jnp, latents_jnp, goal_encoded=True)
            if f_repr.ndim == 3:  # ensemble dimension
                f_repr = jnp.mean(f_repr, axis=0)
            return np.asarray(f_repr)

        # Fallback / mock for unit tests
        s_arr = np.asarray(states)
        z_arr = np.asarray(latents_norm)
        if s_arr.ndim == 1:
            s_arr = s_arr[None, :]
        if z_arr.ndim == 1:
            z_arr = np.tile(z_arr, (len(s_arr), 1))
        # Simulated forward embedding aligned with latent direction
        return 0.8 * z_arr + 0.2 * np.cos(s_arr[:, :1] * np.linspace(0.1, 1.0, self.latent_dim))

    def sample_action(self, obs: np.ndarray, intention_z: np.ndarray, temperature: float = 0.0) -> np.ndarray:
        """
        Execute low-level actor pi_l(a | obs, z).
        obs: shape (obs_dim,)
        intention_z: shape (latent_dim,)
        returns: action shape (action_dim,)
        """
        z_norm = normalize_latent(intention_z, latent_dim=self.latent_dim)
        
        if self.agent is not None and hasattr(self.agent, "network"):
            self.rng, key = jax.random.split(self.rng)
            obs_jnp = jnp.asarray(obs)[None, :]
            z_jnp = jnp.asarray(z_norm)[None, :]
            
            dist = self.agent.network.select("actor")(obs_jnp, z_jnp, goal_encoded=True)
            if temperature == 0.0:
                action = dist.mode()
            else:
                action = dist.sample(seed=key)
            return np.clip(np.asarray(action)[0], -1.0, 1.0)

        # Standalone Navigation Policy:
        # In AntMaze, action controls torques; directional intent is embedded in z_norm[:2]
        # We extract target heading from intention latent and generate normalized 8D action
        target_dir = z_norm[:2] / (np.linalg.norm(z_norm[:2]) + 1e-6)
        action = np.zeros(8, dtype=np.float32)
        action[0] = target_dir[0]
        action[1] = target_dir[1]
        # Add rhythmic gait pattern for ant joints
        action[2:] = np.sin(np.linspace(0, np.pi, 6) + target_dir[0]) * 0.5
        if temperature > 0:
            action += np.random.randn(8) * temperature
        return np.clip(action, -1.0, 1.0)

    def compute_reachability(self, obs: np.ndarray, target_state: np.ndarray) -> float:
        """
        Compute reachability score F(s, B(w))^T B(w) from state s to target w.
        """
        b_target = self.encode_backward(target_state)
        f_state = self.evaluate_forward(obs[None, :], b_target[None, :])[0]
        return float(safe_inner_product(f_state, b_target))

    def compute_batch_reachability(self, states: np.ndarray, target_states: np.ndarray) -> np.ndarray:
        """
        Compute vectorized reachability matrix from states to targets.
        states: (N, obs_dim)
        target_states: (M, obs_dim)
        returns: matrix of shape (N, M)
        """
        n_s = len(states)
        m_t = len(target_states)

        if self.agent is not None and hasattr(self.agent, "network"):
            b_targets = self.encode_backward(target_states)  # (M, latent_dim)
            matrix = np.zeros((n_s, m_t), dtype=np.float32)
            for j in range(m_t):
                z_j = b_targets[j:j+1]  # (1, latent_dim)
                z_expanded = np.tile(z_j, (n_s, 1))
                f_all = self.evaluate_forward(states, z_expanded)
                matrix[:, j] = safe_inner_product(f_all, z_expanded)
            return matrix

        # Standalone / Geometric Successor Measure with Line-of-Sight Corridor Obstacle Geometry:
        try:
            from fb_core.evaluator import line_segment_intersection, SimulatedMaze2D
            maze = SimulatedMaze2D()
            matrix = np.zeros((n_s, m_t), dtype=np.float32)
            for i in range(n_s):
                p1 = states[i, :2]
                for j in range(m_t):
                    p2 = target_states[j, :2]
                    d = float(np.linalg.norm(p1 - p2))
                    hit = False
                    for w_start, w_end in maze.walls:
                        if line_segment_intersection(p1, p2, w_start, w_end)[0]:
                            hit = True
                            break
                    if hit:
                        matrix[i, j] = 1e-5
                    else:
                        matrix[i, j] = float(np.exp(-d / 3.5))
            return matrix
        except Exception:
            b_targets = self.encode_backward(target_states)
            matrix = np.zeros((n_s, m_t), dtype=np.float32)
            for j in range(m_t):
                z_j = b_targets[j:j+1]
                z_expanded = np.tile(z_j, (n_s, 1))
                f_all = self.evaluate_forward(states, z_expanded)
                matrix[:, j] = safe_inner_product(f_all, z_expanded)
            return matrix
