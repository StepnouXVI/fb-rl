import os
import functools
import numpy as np
import jax
import jax.numpy as jnp
from scipy.sparse.csgraph import dijkstra
import torch
import torch.nn as nn
from src.models import build_student_model

# ponytail: Pure JIT-Compiled Primitives for Ultra-Fast Inference (<0.05ms/step)

@functools.partial(jax.jit, static_argnames=("temperature",))
def _jit_baseline_step(agent, obs, goal_latent, seed=None, temperature=0.0):
    """Fused high_actor + actor execution in a single XLA kernel."""
    obs_b = obs[None, :] if obs.ndim == 1 else obs
    z_b = goal_latent[None, :] if goal_latent.ndim == 1 else goal_latent

    high_dist = agent.network.select("high_actor")(obs_b, z_b, goal_encoded=True, temperature=0.0)
    subgoal_z = agent.normalize_z(high_dist.mode())

    low_dist = agent.network.select("actor")(obs_b, subgoal_z, goal_encoded=True, temperature=temperature)
    if temperature == 0.0 or seed is None:
        action = low_dist.mode()
    else:
        action = low_dist.sample(seed=seed)

    return jnp.clip(action[0], -1.0, 1.0), subgoal_z[0]



@jax.jit
def _jit_batch_reach(agent, states, targets):
    """Jitted batch reachability metric."""
    f = agent.network.select("forward_repr")(states, targets, goal_encoded=True)
    if f.ndim == 3:
        f = jnp.mean(f, axis=0)
    return jnp.sum(f * targets, axis=-1)


@jax.jit
def _jit_actor_from_latent(agent, obs, latent):
    """Jitted low-actor execution given a latent intention."""
    obs_b = obs[None, :] if obs.ndim == 1 else obs
    z_b = latent[None, :] if latent.ndim == 1 else latent
    high_dist = agent.network.select("high_actor")(obs_b, z_b, goal_encoded=True, temperature=0.0)
    subgoal_z = agent.normalize_z(high_dist.mode())
    low_dist = agent.network.select("actor")(obs_b, subgoal_z, goal_encoded=True, temperature=0.0)
    return jnp.clip(low_dist.mode()[0], -1.0, 1.0), subgoal_z[0]


@jax.jit
def _jit_decode_latent_to_coords(latent, landmark_latents):
    """Finds the nearest landmark index in latent space."""
    z = latent / jnp.linalg.norm(latent, axis=-1, keepdims=True)
    sims = jnp.matmul(landmark_latents, z.T)
    return jnp.argmax(sims)


# ==========================================
# Pure FB Planner Classes (Zero Maze Map Access)
# ==========================================

class BasePlanner:
    def __init__(self, agent, name="BasePlanner"):
        self.agent = agent
        self.name = name
        self.last_subgoal_info = {"subgoal_xy": None, "waypoints_xy": [], "is_direct_goal": True}

    def reset(self, obs, goal_latent):
        self.last_subgoal_info = {"subgoal_xy": None, "waypoints_xy": [], "is_direct_goal": True}

    def sample_action(self, obs, goal_latent, step=0, seed=None, temperature=0.0):
        action, _ = _jit_baseline_step(
            self.agent, jnp.asarray(obs), jnp.asarray(goal_latent), seed=seed, temperature=temperature
        )
        return np.asarray(action)

    def get_subgoal_info(self):
        return self.last_subgoal_info


class BaselinePlanner(BasePlanner):
    def __init__(self, agent, dataset_states=None, use_high_actor=True, name="Single-Intention Baseline"):
        super().__init__(agent, name=name)
        self.use_high_actor = use_high_actor

        if dataset_states is not None:
            n_samples = min(500, len(dataset_states))
            rng = np.random.default_rng(42)
            idxs = rng.choice(len(dataset_states), size=n_samples, replace=False)
            self.ref_coords = np.asarray(dataset_states[idxs][:, :2])
            self.ref_latents = jnp.asarray(
                agent.normalize_z(agent.network.select("backward_repr")(jnp.asarray(dataset_states[idxs])))
            )
        else:
            self.ref_coords = None
            self.ref_latents = None

    def reset(self, obs, goal_latent):
        self.last_subgoal_info = {"subgoal_xy": None, "waypoints_xy": [], "is_direct_goal": False}

    def sample_action(self, obs, goal_latent, step=0, seed=None, temperature=0.0):
        obs_jnp = jnp.asarray(obs)
        goal_jnp = jnp.asarray(goal_latent)
        if self.use_high_actor:
            action, high_z = _jit_baseline_step(self.agent, obs_jnp, goal_jnp, seed=seed, temperature=temperature)

            if self.ref_latents is not None:
                best_idx = int(_jit_decode_latent_to_coords(high_z, self.ref_latents))
                decoded_xy = self.ref_coords[best_idx].tolist()
            else:
                decoded_xy = None

            self.last_subgoal_info = {
                "subgoal_xy": decoded_xy,
                "waypoints_xy": [decoded_xy] if decoded_xy else [],
                "is_direct_goal": False,
            }
        else:
            norm_z = self.agent.normalize_z(goal_jnp)
            low_dist = self.agent.network.select("actor")(
                obs_jnp[None, :], norm_z[None, :], goal_encoded=True, temperature=temperature
            )
            action = jnp.clip(low_dist.mode()[0], -1.0, 1.0)
            self.last_subgoal_info = {"subgoal_xy": None, "waypoints_xy": [], "is_direct_goal": True}

        return np.asarray(action)


class BufferGraphPlanner(BasePlanner):
    """
    Offline RL planning using purely the Forward-Backward reachability model.
    Zero access to environment maze_map or wall geometries.
    """
    def __init__(self, agent, dataset_states, n_landmarks=1000, max_edge_radius=3.5, reachability_cutoff=35.0, lookahead_dist=2.6, name="buffer_graph"):
        super().__init__(agent, name=name)
        self.n_landmarks = min(n_landmarks, len(dataset_states))
        self.max_edge_radius = max_edge_radius
        self.reachability_cutoff = reachability_cutoff
        self.lookahead_dist = lookahead_dist

        # 1. Sample landmarks uniformly from offline dataset
        rng = np.random.default_rng(42)
        idxs = rng.choice(len(dataset_states), size=self.n_landmarks, replace=False)
        self.landmarks = jnp.asarray(dataset_states[idxs])
        self.landmark_coords = np.asarray(dataset_states[idxs][:, :2])

        # 2. Encode all landmark backward representations
        self.landmark_latents = jnp.asarray(
            agent.normalize_z(agent.network.select("backward_repr")(self.landmarks))
        )

        # 3. Build reachability graph and compute shortest path predecessors
        self.cost_matrix, self.reach_matrix = self._build_graph()
        self.all_dist, self.all_pred = dijkstra(
            self.cost_matrix, directed=True, return_predecessors=True
        )

        self.path_coords = []
        self.path_latents = []
        self.waypoint_coords = []
        self.waypoints = []
        self.goal_z = None
        self.current_path_idx = 0
        self.pos_history = []

    def _build_graph(self):
        n = len(self.landmarks)
        s_rep = jnp.repeat(self.landmarks, n, axis=0)
        z_tile = jnp.tile(self.landmark_latents, (n, 1))

        reach_flat = []
        for i in range(0, len(s_rep), 45000):
            sb = s_rep[i : i + 45000]
            zb = z_tile[i : i + 45000]
            reach_flat.append(np.asarray(_jit_batch_reach(self.agent, sb, zb)))
        reach_matrix = np.concatenate(reach_flat, axis=0).reshape((n, n))

        dists_euclid = np.linalg.norm(
            self.landmark_coords[:, None, :] - self.landmark_coords[None, :, :], axis=-1
        )

        max_diag = float(np.max(np.diag(reach_matrix))) if np.max(np.diag(reach_matrix)) > 0 else 1.0
        normalized = np.clip(reach_matrix / max_diag, 1e-6, 1.0)
        cost_matrix = np.maximum(0.0, -np.log(normalized))

        # Disallow edges that span across walls
        cost_matrix[dists_euclid > self.max_edge_radius] = np.inf
        cost_matrix[reach_matrix < self.reachability_cutoff] = np.inf
        np.fill_diagonal(cost_matrix, 0.0)

        return cost_matrix, reach_matrix

    def reset(self, obs, goal_z):
        obs_xy = np.asarray(obs[:2])
        start_idx = int(np.argmin(np.linalg.norm(self.landmark_coords - obs_xy, axis=-1)))

        # Find closest landmark to goal_z
        goal_z_norm = goal_z / np.linalg.norm(goal_z)
        sims = np.asarray(jnp.matmul(self.landmark_latents, goal_z_norm.T))
        goal_idx = int(np.argmax(sims))

        # Reconstruct path from predecessors
        path = []
        curr = goal_idx
        while curr != -9999 and curr != start_idx:
            path.append(curr)
            curr = self.all_pred[start_idx, curr]
            if len(path) > self.n_landmarks:
                break

        if curr == start_idx:
            path.append(start_idx)
            path.reverse()
        else:
            path = [start_idx, goal_idx]

        self.path_coords = [self.landmark_coords[i] for i in path]
        self.path_latents = [self.landmark_latents[i] for i in path]
        self.goal_z = goal_z
        self.current_path_idx = 0
        self.pos_history = []

        # Extract downsampled key waypoints for logging/plotting
        if len(self.path_coords) > 2:
            filtered_indices = [0]
            accum = 0.0
            for k in range(1, len(self.path_coords) - 1):
                p_prev = self.path_coords[k - 1]
                p_curr = self.path_coords[k]
                p_next = self.path_coords[k + 1]
                v1 = p_curr - p_prev
                v2 = p_next - p_curr
                l1 = np.linalg.norm(v1)
                l2 = np.linalg.norm(v2)
                accum += l1
                is_corner = False
                if l1 > 0.1 and l2 > 0.1:
                    cos_theta = np.dot(v1, v2) / (l1 * l2)
                    if cos_theta < 0.7:  # Turn angle > 45 deg
                        is_corner = True

                if accum >= self.lookahead_dist or is_corner:
                    filtered_indices.append(k)
                    accum = 0.0

            if filtered_indices[-1] != len(self.path_coords) - 1:
                filtered_indices.append(len(self.path_coords) - 1)

            self.waypoint_coords = [self.path_coords[k] for k in filtered_indices]
            self.waypoints = [self.path_latents[k] for k in filtered_indices]
        else:
            self.waypoint_coords = list(self.path_coords)
            self.waypoints = list(self.path_latents)

        curr_c = self.waypoint_coords[0] if self.waypoint_coords else None
        self.last_subgoal_info = {
            "subgoal_xy": [float(curr_c[0]), float(curr_c[1])] if curr_c is not None else None,
            "waypoints_xy": [[float(c[0]), float(c[1])] for c in self.waypoint_coords],
            "is_direct_goal": False,
        }

    def sample_action(self, obs, goal_z, step=0, seed=None, temperature=0.0):
        if not self.path_coords:
            self.reset(obs, goal_z)

        obs_xy = np.asarray(obs[:2])

        # 1. Advance along path (search forward window)
        search_end = min(len(self.path_coords), self.current_path_idx + 12)
        window_dists = [np.linalg.norm(obs_xy - self.path_coords[k]) for k in range(self.current_path_idx, search_end)]
        best_offset = int(np.argmin(window_dists))
        self.current_path_idx += best_offset

        # 2. Look ahead by `lookahead_dist` along the topological path
        accum = 0.0
        target_idx = self.current_path_idx
        while target_idx < len(self.path_coords) - 1 and accum < self.lookahead_dist:
            accum += np.linalg.norm(self.path_coords[target_idx + 1] - self.path_coords[target_idx])
            target_idx += 1

        dist_to_final = float(np.linalg.norm(obs_xy - self.path_coords[-1]))

        # 3. Select target latent (direct goal handover when near terminal zone)
        if target_idx >= len(self.path_coords) - 1 or dist_to_final <= 2.2:
            target_latent = self.goal_z
            target_xy = self.path_coords[-1]
            is_goal = True
        else:
            target_latent = self.path_latents[target_idx]
            target_xy = self.path_coords[target_idx]
            is_goal = False

        # 4. Stuck detection
        self.pos_history.append(obs_xy.copy())
        if len(self.pos_history) > 40:
            self.pos_history.pop(0)

        is_stuck = False
        if len(self.pos_history) >= 40 and dist_to_final > 2.0:
            if float(np.linalg.norm(obs_xy - self.pos_history[0])) < 0.4:
                is_stuck = True

        eval_temp = 0.2 if is_stuck else temperature
        seed_k = jax.random.PRNGKey(step) if eval_temp > 0 else seed

        action, subgoal_z = _jit_baseline_step(self.agent, jnp.asarray(obs), target_latent, seed=seed_k, temperature=eval_temp)

        self.last_subgoal_info = {
            "subgoal_xy": [float(target_xy[0]), float(target_xy[1])],
            "waypoints_xy": [[float(c[0]), float(c[1])] for c in self.waypoint_coords],
            "is_direct_goal": is_goal,
        }
        return np.asarray(action)

    def get_subgoal_latent(self, obs, goal_z, step=0):
        """Returns the optimal target subgoal latent for the current state and goal."""
        if not self.path_coords:
            self.reset(obs, goal_z)

        obs_xy = np.asarray(obs[:2])
        search_end = min(len(self.path_coords), self.current_path_idx + 12)
        window_dists = [np.linalg.norm(obs_xy - self.path_coords[k]) for k in range(self.current_path_idx, search_end)]
        best_offset = int(np.argmin(window_dists))
        self.current_path_idx += best_offset

        accum = 0.0
        target_idx = self.current_path_idx
        while target_idx < len(self.path_coords) - 1 and accum < self.lookahead_dist:
            accum += np.linalg.norm(self.path_coords[target_idx + 1] - self.path_coords[target_idx])
            target_idx += 1

        dist_to_final = float(np.linalg.norm(obs_xy - self.path_coords[-1]))
        if target_idx >= len(self.path_coords) - 1 or dist_to_final <= 2.2:
            return np.asarray(self.goal_z)
        return np.asarray(self.path_latents[target_idx])


class RecursiveBisectionPlanner(BasePlanner):
    def __init__(self, agent, dataset_states, max_depth=2, n_candidates=200, hit_threshold=35.0, name="Recursive Bisection"):
        super().__init__(agent, name=name)
        self.max_depth = max_depth
        self.n_candidates = min(n_candidates, len(dataset_states))
        self.hit_threshold = hit_threshold

        rng = np.random.default_rng(42)
        idxs = rng.choice(len(dataset_states), size=self.n_candidates, replace=False)
        self.candidate_states = jnp.asarray(dataset_states[idxs])
        self.candidate_coords = np.asarray(dataset_states[idxs][:, :2])
        self.candidate_latents = jnp.asarray(
            agent.normalize_z(agent.network.select("backward_repr")(self.candidate_states))
        )
        self.cached_latent = None
        self.cached_coord = None
        self.steps_on_latent = 0

    def _find_midpoint_fast(self, obs, goal_latent):
        obs_exp = jnp.broadcast_to(obs[None, :], (self.candidate_states.shape[0], obs.shape[0]))
        r_sw = _jit_batch_reach(self.agent, obs_exp, self.candidate_latents)

        g_exp = jnp.broadcast_to(goal_latent[None, :], (self.candidate_states.shape[0], goal_latent.shape[0]))
        r_wg = _jit_batch_reach(self.agent, self.candidate_states, g_exp)

        scores = jnp.log(jnp.maximum(1e-4, r_sw)) + jnp.log(jnp.maximum(1e-4, r_wg))
        best_idx = int(jnp.argmax(scores))
        self.cached_coord = self.candidate_coords[best_idx].tolist()
        return self.candidate_latents[best_idx]

    def reset(self, obs, goal_latent):
        self.cached_latent = self._find_midpoint_fast(jnp.asarray(obs), jnp.asarray(goal_latent))
        self.steps_on_latent = 0
        self.last_subgoal_info = {
            "subgoal_xy": self.cached_coord,
            "waypoints_xy": [self.cached_coord] if self.cached_coord else [],
            "is_direct_goal": False,
        }

    def sample_action(self, obs, goal_latent, step=0, seed=None, temperature=0.0):
        obs_jnp = jnp.asarray(obs)
        goal_jnp = jnp.asarray(goal_latent)

        if self.cached_latent is None:
            self.reset(obs, goal_latent)

        r_curr = float(_jit_batch_reach(self.agent, obs_jnp[None, :], self.cached_latent[None, :])[0])
        self.steps_on_latent += 1
        if r_curr >= self.hit_threshold or self.steps_on_latent > 60:
            self.reset(obs, goal_latent)

        action, _ = _jit_actor_from_latent(self.agent, obs_jnp, self.cached_latent)
        self.last_subgoal_info = {
            "subgoal_xy": self.cached_coord,
            "waypoints_xy": [self.cached_coord] if self.cached_coord else [],
            "is_direct_goal": False,
        }
        return np.asarray(action)


class DistilledMLPPlanner(BasePlanner):
    def __init__(
        self,
        agent,
        model_type="dense_eca",
        checkpoint_path=None,
        hidden_dim=256,
        n_layers=3,
        num_heads=4,
        device="cpu",
        name="Distilled Latent Policy",
    ):
        super().__init__(agent, name=name)
        obs_dim = 29
        latent_dim = agent.config["latent_dim"]
        self.device = torch.device("cpu")
        torch.set_num_threads(1)

        self.model = build_student_model(
            model_type=model_type,
            obs_dim=obs_dim,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            num_heads=num_heads,
        )
        if checkpoint_path and os.path.exists(checkpoint_path):
            self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def reset(self, obs, goal_latent):
        self.last_subgoal_info = {"subgoal_xy": None, "waypoints_xy": [], "is_direct_goal": False}

    def sample_action(self, obs, goal_latent, step=0, seed=None, temperature=0.0):
        x = np.concatenate([obs, np.asarray(goal_latent)], axis=-1).astype(np.float32)
        with torch.no_grad():
            inp = torch.from_numpy(x).unsqueeze(0)
            pred_z = self.model(inp).squeeze(0).numpy()

        action, _ = _jit_actor_from_latent(self.agent, jnp.asarray(obs), jnp.asarray(pred_z))
        self.last_subgoal_info = {"subgoal_xy": None, "waypoints_xy": [], "is_direct_goal": False}
        return np.asarray(action)
