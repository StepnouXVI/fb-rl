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
    Pure Forward-Backward Graph Dijkstra Planner.
    Extracts topological connectivity, distances, and corridor transitions purely from
    the learned FB representations and replay buffer observations (100% Zero Map Knowledge).
    """
    def __init__(
        self,
        agent,
        dataset_states,
        n_landmarks=400,
        max_edge_radius=4.5,
        reachability_cutoff=20.0,
        wp_switch_dist=3.2,
        name="Buffer Graph Dijkstra",
        **kwargs,
    ):
        super().__init__(agent, name=name)
        self.n_landmarks = min(n_landmarks, len(dataset_states))
        self.max_edge_radius = max_edge_radius
        self.reachability_cutoff = reachability_cutoff
        self.wp_switch_dist = wp_switch_dist

        # 1. Sample landmarks uniformly from offline dataset
        rng = np.random.default_rng(42)
        idxs = rng.choice(len(dataset_states), size=self.n_landmarks, replace=False)
        self.landmarks = jnp.asarray(dataset_states[idxs])
        self.landmark_coords = np.asarray(dataset_states[idxs][:, :2])
        self.landmark_latents = jnp.asarray(
            agent.normalize_z(agent.network.select("backward_repr")(self.landmarks))
        )

        # 2. Build graph strictly from learned FB reachability
        self.cost_matrix, self.reach_matrix = self._build_graph()
        self.all_dist, self.all_pred = dijkstra(
            self.cost_matrix, directed=True, return_predecessors=True
        )

        self.waypoints = []
        self.waypoint_coords = []
        self.current_idx = 0
        self.steps_on_wp = 0
        self.pos_history = []

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

        # Topology filtering purely from local observation radius and FB reachability
        cost_matrix[dists_euclid > self.max_edge_radius] = np.inf
        cost_matrix[reach_matrix < self.reachability_cutoff] = np.inf
        np.fill_diagonal(cost_matrix, 0.0)
        return cost_matrix, reach_matrix

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

        coords = [self.landmark_coords[i] for i in path]

        # RDP trajectory simplification to extract corner turnpoints along corridors
        def rdp(pts, epsilon=0.9, max_len=4.5):
            if len(pts) <= 2:
                return pts
            dmax, index = 0.0, 0
            p1, p2 = pts[0], pts[-1]
            line_vec = p2 - p1
            line_len = np.linalg.norm(line_vec)
            if line_len < 1e-6:
                return [p1, p2]
            line_unit = line_vec / line_len

            for i in range(1, len(pts) - 1):
                v = pts[i] - p1
                proj = np.dot(v, line_unit)
                d = np.linalg.norm(v - proj * line_unit)
                if d > dmax:
                    index, dmax = i, d
            if dmax > epsilon or line_len > max_len:
                res1 = rdp(pts[:index+1], epsilon, max_len)
                res2 = rdp(pts[index:], epsilon, max_len)
                return res1[:-1] + res2
            else:
                return [p1, p2]

        simplified_coords = rdp(coords, epsilon=0.9, max_len=4.5)

        pairs = []
        for c in simplified_coords:
            idx = int(np.argmin(np.linalg.norm(self.landmark_coords - c, axis=-1)))
            pairs.append((self.landmark_coords[idx], self.landmark_latents[idx]))
        pairs.append((self.landmark_coords[goal_idx], goal_z))

        # Start from forward waypoint (skip backwards start hook)
        start_wp_idx = 0
        if len(pairs) > 1:
            v01 = pairs[1][0] - pairs[0][0]
            v_ant = pairs[0][0] - obs_xy
            if np.dot(v_ant, v01) < 0 or np.linalg.norm(obs_xy - pairs[1][0]) < np.linalg.norm(obs_xy - pairs[0][0]):
                start_wp_idx = 1

        exec_pairs = pairs[start_wp_idx:]
        self.waypoint_coords = [p[0].tolist() if isinstance(p[0], np.ndarray) else p[0] for p in exec_pairs]
        self.waypoints = [p[1] for p in exec_pairs]
        self.current_idx = 0
        self.steps_on_wp = 0
        self.pos_history = []
        return self.waypoints

    def reset(self, obs, goal_z):
        self._plan(obs, goal_z)
        self.last_subgoal_info = {
            "subgoal_xy": self.waypoint_coords[0] if self.waypoint_coords else None,
            "waypoints_xy": self.waypoint_coords,
            "is_direct_goal": False,
        }

    def sample_action(self, obs, goal_z, step=0, seed=None, temperature=0.0):
        if not self.waypoints or len(self.waypoint_coords) == 0:
            self.reset(obs, goal_z)

        curr_coord = np.asarray(self.waypoint_coords[self.current_idx]) if self.current_idx < len(self.waypoint_coords) else None
        next_coord = np.asarray(self.waypoint_coords[self.current_idx + 1]) if self.current_idx < len(self.waypoint_coords) - 1 else None

        dist_curr = 999.0
        if curr_coord is not None:
            dist_curr = float(np.linalg.norm(obs[:2] - curr_coord))
            if next_coord is not None:
                dist_next = float(np.linalg.norm(obs[:2] - next_coord))
                edge_vec = next_coord - curr_coord
                proj = float(np.dot(obs[:2] - curr_coord, edge_vec))
                if dist_curr <= self.wp_switch_dist or dist_next < dist_curr or (proj > 0 and dist_curr < 4.2):
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

        is_stuck = False
        if len(self.pos_history) >= 40:
            disp = float(np.linalg.norm(obs[:2] - self.pos_history[0]))
            if disp < 0.5:
                is_stuck = True

        eval_temp = 0.2 if is_stuck else 0.0
        seed_k = jax.random.PRNGKey(step) if eval_temp > 0 else None

        # Direct low-actor goal locking near terminal target to prevent room looping
        if self.current_idx >= len(self.waypoints) - 1 and dist_curr <= 3.0:
            norm_z = self.agent.normalize_z(jnp.asarray(goal_z)[None, :])
            low_dist = self.agent.network.select("actor")(
                jnp.asarray(obs)[None, :], norm_z, goal_encoded=True, temperature=eval_temp
            )
            action = jnp.clip(low_dist.mode()[0] if eval_temp == 0.0 or seed_k is None else low_dist.sample(seed=seed_k)[0], -1.0, 1.0)
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
