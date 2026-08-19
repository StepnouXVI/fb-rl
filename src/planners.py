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
# Optimized Planner Classes
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
    Obstacle-Free Topological Graph Planner with Raycast String-Pulling Shortcut Simplification,
    Dynamic Line-of-Sight Skipping, and Direct Low-Actor Goal Locking.
    """
    def __init__(
        self,
        agent,
        dataset_states,
        maze_map=None,
        n_landmarks=400,
        max_edge_radius=5.5,
        reachability_cutoff=20.0,
        wp_switch_dist=3.5,
        name="Buffer Graph Dijkstra",
    ):
        super().__init__(agent, name=name)
        self.n_landmarks = min(n_landmarks, len(dataset_states))
        self.max_edge_radius = max_edge_radius
        self.reachability_cutoff = reachability_cutoff
        self.wp_switch_dist = wp_switch_dist

        # 1. Parse maze walls for obstacle-free line-of-sight validation
        if maze_map is None:
            maze_map = np.array([
                [1, 1, 1, 1, 1, 1, 1, 1],
                [1, 0, 0, 1, 1, 0, 0, 1],
                [1, 0, 0, 1, 0, 0, 0, 1],
                [1, 1, 0, 0, 0, 1, 1, 1],
                [1, 0, 0, 1, 0, 0, 0, 1],
                [1, 0, 1, 0, 0, 1, 0, 1],
                [1, 0, 0, 0, 1, 0, 0, 1],
                [1, 1, 1, 1, 1, 1, 1, 1],
            ])
        self.wall_boxes = []
        for i in range(maze_map.shape[0]):
            for j in range(maze_map.shape[1]):
                if maze_map[i, j] == 1:
                    cx = (j - 1) * 4.0
                    cy = (i - 1) * 4.0
                    self.wall_boxes.append((cx - 2.0, cx + 2.0, cy - 2.0, cy + 2.0))

        # 2. Sample landmarks uniformly across dataset
        rng = np.random.default_rng(42)
        idxs = rng.choice(len(dataset_states), size=self.n_landmarks, replace=False)
        self.landmarks = jnp.asarray(dataset_states[idxs])
        self.landmark_coords = np.asarray(dataset_states[idxs][:, :2])
        self.landmark_latents = jnp.asarray(
            agent.normalize_z(agent.network.select("backward_repr")(self.landmarks))
        )

        # 3. Build connectivity graph with obstacle avoidance & Precompute APSP
        self.cost_matrix, self.max_diag = self._build_graph()
        self.all_dist, self.all_pred = dijkstra(
            self.cost_matrix, directed=True, return_predecessors=True
        )

        self.waypoints = []
        self.waypoint_coords = []
        self.current_idx = 0
        self.steps_on_wp = 0
        self.pos_history = []

    def has_los(self, p1, p2, margin=0.45):
        p1 = np.asarray(p1)
        p2 = np.asarray(p2)
        dist = np.linalg.norm(p2 - p1)
        n_samples = max(10, int(dist * 5))
        ts = np.linspace(0.0, 1.0, n_samples)
        xs = p1[0] + ts * (p2[0] - p1[0])
        ys = p1[1] + ts * (p2[1] - p1[1])
        for x, y in zip(xs, ys):
            for xmin, xmax, ymin, ymax in self.wall_boxes:
                if (xmin - margin) <= x <= (xmax + margin) and (ymin - margin) <= y <= (ymax + margin):
                    return False
        return True

    def _build_graph(self):
        n = len(self.landmarks)
        s_rep = jnp.repeat(self.landmarks, n, axis=0)
        z_tile = jnp.tile(self.landmark_latents, (n, 1))

        reach_flat = []
        for i in range(0, len(s_rep), 45000):
            sb = s_rep[i:i+45000]
            zb = z_tile[i:i+45000]
            reach_flat.append(np.asarray(_jit_batch_reach(self.agent, sb, zb)))
        reach_matrix = np.concatenate(reach_flat, axis=0).reshape((n, n))

        dists_euclid = np.linalg.norm(self.landmark_coords[:, None, :] - self.landmark_coords[None, :, :], axis=-1)
        max_diag = float(np.max(np.diag(reach_matrix))) if np.max(np.diag(reach_matrix)) > 0 else 1.0
        normalized = np.clip(reach_matrix / max_diag, 1e-6, 1.0)
        cost_matrix = np.maximum(0.0, -np.log(normalized))

        cost_matrix[dists_euclid > self.max_edge_radius] = np.inf
        cost_matrix[reach_matrix < self.reachability_cutoff] = np.inf

        for i in range(n):
            for j in range(n):
                if cost_matrix[i, j] < np.inf and i != j:
                    if not self.has_los(self.landmark_coords[i], self.landmark_coords[j], margin=0.45):
                        cost_matrix[i, j] = np.inf

        np.fill_diagonal(cost_matrix, 0.0)
        return cost_matrix, max_diag

    def _plan(self, obs, goal_z):
        obs_xy = np.asarray(obs[:2])
        start_idx = int(np.argmin(np.linalg.norm(self.landmark_coords - obs_xy, axis=-1)))

        goal_z_norm = goal_z / np.linalg.norm(goal_z)
        sims = np.asarray(jnp.matmul(self.landmark_latents, goal_z_norm.T))
        goal_idx = int(np.argmax(sims))

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

        # Explicit paired list: [(coord, latent), ...]
        pairs = []
        for i in path:
            pairs.append((self.landmark_coords[i], self.landmark_latents[i]))
        pairs.append((self.landmark_coords[goal_idx], goal_z))

        # Raycast String-Pulling Shortcut Algorithm
        smooth_pairs = [pairs[0]]
        curr_i = 0
        n = len(pairs)
        while curr_i < n - 1:
            furthest = curr_i + 1
            for k in range(n - 1, curr_i, -1):
                if self.has_los(pairs[curr_i][0], pairs[k][0], margin=0.45):
                    furthest = k
                    break
            curr_i = furthest
            smooth_pairs.append(pairs[curr_i])

        # If ant already has line-of-sight to a later waypoint in smooth_pairs, start directly from it
        start_wp_idx = 0
        for k in range(len(smooth_pairs) - 1, -1, -1):
            if self.has_los(obs_xy, smooth_pairs[k][0], margin=0.45):
                start_wp_idx = k
                break

        exec_pairs = smooth_pairs[start_wp_idx:]
        if len(exec_pairs) == 0 or exec_pairs[-1][1] is not goal_z:
            exec_pairs.append((pairs[-1][0], goal_z))

        self.waypoint_coords = [p[0].tolist() if isinstance(p[0], np.ndarray) else p[0] for p in exec_pairs]
        self.waypoints = [p[1] for p in exec_pairs]
        self.pos_history = []
        return self.waypoints

    def reset(self, obs, goal_z):
        self._plan(obs, goal_z)
        self.current_idx = 0
        self.steps_on_wp = 0
        self.pos_history = []
        self.last_subgoal_info = {
            "subgoal_xy": self.waypoint_coords[0] if self.waypoint_coords else None,
            "waypoints_xy": self.waypoint_coords,
            "is_direct_goal": False,
        }

    def sample_action(self, obs, goal_z, step=0, seed=None, temperature=0.0):
        if not self.waypoints or len(self.waypoint_coords) == 0:
            self.reset(obs, goal_z)

        # 1. Dynamic Line-of-Sight Shortcut: skip ahead if later waypoint is directly visible
        for k in range(len(self.waypoint_coords) - 1, self.current_idx, -1):
            if self.has_los(obs[:2], self.waypoint_coords[k], margin=0.45):
                if k > self.current_idx:
                    self.current_idx = k
                    self.steps_on_wp = 0
                break

        # 2. Progression-based waypoint advancement
        curr_coord = np.asarray(self.waypoint_coords[self.current_idx]) if self.current_idx < len(self.waypoint_coords) else None
        next_coord = np.asarray(self.waypoint_coords[self.current_idx + 1]) if self.current_idx < len(self.waypoint_coords) - 1 else None

        dist_curr = 999.0
        if curr_coord is not None:
            dist_curr = float(np.linalg.norm(obs[:2] - curr_coord))
            if next_coord is not None:
                dist_next = float(np.linalg.norm(obs[:2] - next_coord))
                edge_vec = next_coord - curr_coord
                proj = float(np.dot(obs[:2] - curr_coord, edge_vec))
                if dist_curr <= self.wp_switch_dist or dist_next < dist_curr or (proj > 0 and dist_curr < 5.0):
                    self.current_idx += 1
                    self.steps_on_wp = 0
            else:
                if dist_curr <= self.wp_switch_dist and self.current_idx < len(self.waypoints) - 1:
                    self.current_idx += 1
                    self.steps_on_wp = 0

        curr_wp = self.waypoints[self.current_idx]
        self.pos_history.append(obs[:2].copy())
        if len(self.pos_history) > 40:
            self.pos_history.pop(0)

        # Subtle unsticking jitter if stuck against wall for > 40 steps with displacement < 0.5m
        is_stuck = False
        if len(self.pos_history) >= 40:
            disp = float(np.linalg.norm(obs[:2] - self.pos_history[0]))
            if disp < 0.5:
                is_stuck = True

        eval_temp = 0.2 if is_stuck else 0.0
        seed_k = jax.random.PRNGKey(step) if eval_temp > 0 else None

        # When close to the final goal (dist <= 3.0m), use low actor directly on normalized goal latent
        # to prevent high-actor backward looping in terminal rooms
        if self.current_idx >= len(self.waypoints) - 1 and dist_curr <= 3.0:
            norm_z = self.agent.normalize_z(jnp.asarray(goal_z)[None, :])
            low_dist = self.agent.network.select("actor")(
                jnp.asarray(obs)[None, :], norm_z, goal_encoded=True, temperature=eval_temp
            )
            if eval_temp == 0.0 or seed_k is None:
                action = low_dist.mode()
            else:
                action = low_dist.sample(seed=seed_k)
            action = jnp.clip(action[0], -1.0, 1.0)
        else:
            action, _ = _jit_baseline_step(self.agent, jnp.asarray(obs), curr_wp, seed=seed_k, temperature=eval_temp)

        self.steps_on_wp += 1

        c = self.waypoint_coords[self.current_idx] if self.current_idx < len(self.waypoint_coords) else None
        self.last_subgoal_info = {
            "subgoal_xy": c,
            "waypoints_xy": self.waypoint_coords,
            "is_direct_goal": bool(self.current_idx >= len(self.waypoints) - 1),
        }
        return np.asarray(action)


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
