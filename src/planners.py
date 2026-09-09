import os
import functools
import pickle
import numpy as np
import jax
import jax.numpy as jnp
import flax
from scipy.sparse.csgraph import dijkstra
import torch
import torch.nn as nn
from src.models import build_student_model
from src.waypoint_translators import (
    FlaxSingleWaypointTranslator,
    FlaxSequenceWaypointAttentionTranslator,
    FlaxEnhancedSequenceWaypointAttentionTranslator,
    build_flax_translator,
)

@functools.partial(jax.jit, static_argnames=("temperature",))
def _jit_baseline_step(agent, obs, goal_latent, seed=None, temperature=0.0):
    obs_b = obs[None, :] if obs.ndim == 1 else obs
    z_b = goal_latent[None, :] if goal_latent.ndim == 1 else goal_latent
    high_dist = agent.network.select("high_actor")(obs_b, z_b, goal_encoded=True, temperature=0.0)
    subgoal_z = agent.normalize_z(high_dist.mode())
    low_dist = agent.network.select("actor")(obs_b, subgoal_z, goal_encoded=True, temperature=temperature)
    action = low_dist.mode() if (temperature == 0.0 or seed is None) else low_dist.sample(seed=seed)
    return jnp.clip(action[0], -1.0, 1.0), subgoal_z[0]


@jax.jit
def _jit_batch_reach(agent, states, targets):
    f = agent.network.select("forward_repr")(states, targets, goal_encoded=True)
    if f.ndim == 3:
        f = jnp.mean(f, axis=0)
    return jnp.sum(f * targets, axis=-1)


@jax.jit
def _jit_actor_from_latent(agent, obs, latent):
    obs_b, z_b = obs[None, :], latent[None, :]
    high_dist = agent.network.select("high_actor")(obs_b, z_b, goal_encoded=True, temperature=0.0)
    subgoal_z = agent.normalize_z(high_dist.mode())
    low_dist = agent.network.select("actor")(obs_b, subgoal_z, goal_encoded=True, temperature=0.0)
    return jnp.clip(low_dist.mode()[0], -1.0, 1.0), subgoal_z[0]


@jax.jit
def _jit_decode_latent_to_coords(latent, landmark_latents):
    z = latent / jnp.linalg.norm(latent, axis=-1, keepdims=True)
    return jnp.argmax(jnp.matmul(landmark_latents, z.T))


def _find_connected_landmarks(landmark_coords, landmark_latents, all_dist, obs_xy, goal_z):
    sims = np.asarray(jnp.matmul(landmark_latents, (goal_z / (np.linalg.norm(goal_z) + 1e-8)).T))
    top_goals = np.argsort(-sims)[:15]
    top_starts = np.argsort(np.linalg.norm(landmark_coords - obs_xy, axis=-1))[:20]
    for g in top_goals:
        for s in top_starts:
            if np.isfinite(all_dist[s, g]):
                return s, g
    return top_starts[0], top_goals[0]


def _backtrack_dijkstra_path(all_pred, start_idx, goal_idx, max_nodes):
    path, curr = [], goal_idx
    while curr != -9999 and curr != start_idx and len(path) <= max_nodes:
        path.append(curr)
        curr = all_pred[start_idx, curr]
    if curr == start_idx:
        path.append(start_idx)
        path.reverse()
        return path
    return [start_idx, goal_idx]


def _extract_corner_waypoints(path_coords, path_latents, lookahead_dist):
    if len(path_coords) <= 2:
        return list(path_coords), list(path_latents), list(range(len(path_coords)))
    filtered = [0]
    accum = 0.0
    for k in range(1, len(path_coords) - 1):
        v1, v2 = path_coords[k] - path_coords[k - 1], path_coords[k + 1] - path_coords[k]
        l1, l2 = np.linalg.norm(v1), np.linalg.norm(v2)
        accum += l1
        is_turn = (l1 > 0.1 and l2 > 0.1 and (np.dot(v1, v2) / (l1 * l2)) < 0.7)
        if accum >= lookahead_dist or is_turn:
            filtered.append(k)
            accum = 0.0
    if filtered[-1] != len(path_coords) - 1:
        filtered.append(len(path_coords) - 1)
    return [path_coords[k] for k in filtered], [path_latents[k] for k in filtered], filtered


def _track_local_path_index(obs_xy, path_coords, current_idx, max_window=7):
    search_start = max(0, current_idx - 2)
    search_end = min(len(path_coords), current_idx + max_window)
    dists = [np.linalg.norm(obs_xy - path_coords[k]) for k in range(search_start, search_end)]
    best_offset = int(np.argmin(dists))
    return search_start + best_offset, dists[best_offset]


def _compute_continuous_lookahead(obs_xy, path_coords, lookahead_dist=2.6, current_seg=0, window=7, path_dists=None):
    """
    Computes exact continuous sliding lookahead coordinate along the polyline path.
    Projects agent position onto path segments and marches forward by lookahead_dist.
    """
    N = len(path_coords)
    if N <= 1:
        return np.array(path_coords[0]), 0, 0.0

    best_dist = float("inf")
    best_seg = current_seg
    best_t = 0.0
    best_proj = path_coords[current_seg]

    search_start = max(0, current_seg - 2)
    search_end = min(N - 1, current_seg + window)

    for k in range(search_start, search_end):
        A = np.array(path_coords[k])
        B = np.array(path_coords[k + 1])
        v = B - A
        v_norm_sq = float(np.dot(v, v))
        if v_norm_sq < 1e-8:
            t = 0.0
        else:
            t = float(np.clip(np.dot(obs_xy - A, v) / v_norm_sq, 0.0, 1.0))
        proj = A + t * v
        d = float(np.linalg.norm(obs_xy - proj))
        if d < best_dist:
            best_dist = d
            best_seg = k
            best_t = t
            best_proj = proj

    if path_dists is not None and best_seg < len(path_dists):
        base_s = path_dists[best_seg]
        seg_len = float(np.linalg.norm(path_coords[min(best_seg + 1, N - 1)] - path_coords[best_seg]))
        s_agent = base_s + best_t * seg_len
    else:
        s_agent = float(best_seg)

    rem = float(lookahead_dist)
    curr_k = best_seg
    A = np.array(path_coords[curr_k])
    B = np.array(path_coords[curr_k + 1])
    v = B - A
    v_len = float(np.linalg.norm(v))
    rem_in_seg = (1.0 - best_t) * v_len

    if rem <= rem_in_seg:
        c_t = best_proj + (rem / max(v_len, 1e-6)) * v
        return c_t, best_seg, s_agent

    rem -= rem_in_seg
    curr_k += 1

    while curr_k < N - 1:
        A = np.array(path_coords[curr_k])
        B = np.array(path_coords[curr_k + 1])
        v = B - A
        v_len = float(np.linalg.norm(v))
        if rem <= v_len:
            c_t = A + (rem / max(v_len, 1e-6)) * v
            return c_t, best_seg, s_agent
        rem -= v_len
        curr_k += 1

    return np.array(path_coords[-1]), best_seg, s_agent


def _extract_curvature_angles(coords):
    K = len(coords)
    curvs = []
    for k in range(K):
        if k + 2 < K:
            v1, v2 = coords[k + 1] - coords[k], coords[k + 2] - coords[k + 1]
            l1, l2 = np.linalg.norm(v1), np.linalg.norm(v2)
            curvs.append(float(np.clip(np.dot(v1, v2) / (l1 * l2), -1.0, 1.0)) if (l1 > 1e-4 and l2 > 1e-4) else 1.0)
        else:
            curvs.append(1.0)
    return curvs


def _check_stuck_state(pos_history, obs_xy, dist_to_final, stuck_count):
    pos_history.append(obs_xy.copy())
    if len(pos_history) > 40:
        pos_history.pop(0)
    is_stuck = False
    if len(pos_history) >= 40 and dist_to_final > 1.8:
        if float(np.linalg.norm(obs_xy - pos_history[0])) < 0.4:
            is_stuck = True
            stuck_count += 1
        else:
            stuck_count = max(0, stuck_count - 1)
    return is_stuck, stuck_count


class BasePlanner:
    def __init__(self, agent, name="BasePlanner"):
        self.agent = agent
        self.name = name
        self.last_subgoal_info = {"subgoal_xy": None, "waypoints_xy": [], "is_direct_goal": True}

    def reset(self, obs, goal_latent):
        self.last_subgoal_info = {"subgoal_xy": None, "waypoints_xy": [], "is_direct_goal": True}

    def sample_action(self, obs, goal_latent, step=0, seed=None, temperature=0.0):
        action, _ = _jit_baseline_step(self.agent, jnp.asarray(obs), jnp.asarray(goal_latent), seed=seed, temperature=temperature)
        return np.asarray(action)

    def get_subgoal_info(self):
        return self.last_subgoal_info


class BaselinePlanner(BasePlanner):
    def __init__(self, agent, dataset_states=None, use_high_actor=True, name="Single-Intention Baseline"):
        super().__init__(agent, name=name)
        self.use_high_actor = use_high_actor
        if dataset_states is not None:
            n_samples = min(500, len(dataset_states))
            idxs = np.random.default_rng(42).choice(len(dataset_states), size=n_samples, replace=False)
            self.ref_coords = np.asarray(dataset_states[idxs][:, :2])
            self.ref_latents = jnp.asarray(agent.normalize_z(agent.network.select("backward_repr")(jnp.asarray(dataset_states[idxs]))))
        else:
            self.ref_coords, self.ref_latents = None, None

    def reset(self, obs, goal_latent):
        self.last_subgoal_info = {"subgoal_xy": None, "waypoints_xy": [], "is_direct_goal": False}

    def sample_action(self, obs, goal_latent, step=0, seed=None, temperature=0.0):
        obs_jnp, goal_jnp = jnp.asarray(obs), jnp.asarray(goal_latent)
        if self.use_high_actor:
            action, high_z = _jit_baseline_step(self.agent, obs_jnp, goal_jnp, seed=seed, temperature=temperature)
            decoded_xy = self.ref_coords[int(_jit_decode_latent_to_coords(high_z, self.ref_latents))].tolist() if self.ref_latents is not None else None
            self.last_subgoal_info = {"subgoal_xy": decoded_xy, "waypoints_xy": [decoded_xy] if decoded_xy else [], "is_direct_goal": False}
        else:
            low_dist = self.agent.network.select("actor")(obs_jnp[None, :], self.agent.normalize_z(goal_jnp)[None, :], goal_encoded=True, temperature=temperature)
            action = jnp.clip(low_dist.mode()[0], -1.0, 1.0)
            self.last_subgoal_info = {"subgoal_xy": None, "waypoints_xy": [], "is_direct_goal": True}
        return np.asarray(action)


class BufferGraphPlanner(BasePlanner):
    """Offline RL planning using purely the Forward-Backward reachability model."""
    def __init__(self, agent, dataset_states, n_landmarks=1000, max_edge_radius=3.5, reachability_cutoff=35.0, lookahead_dist=2.6, name="buffer_graph"):
        super().__init__(agent, name=name)
        self.n_landmarks = min(n_landmarks, len(dataset_states))
        self.max_edge_radius = max_edge_radius
        self.reachability_cutoff = reachability_cutoff
        self.lookahead_dist = lookahead_dist

        idxs = np.random.default_rng(42).choice(len(dataset_states), size=self.n_landmarks, replace=False)
        self.landmarks = jnp.asarray(dataset_states[idxs])
        self.landmark_coords = np.asarray(dataset_states[idxs][:, :2])
        self.landmark_latents = jnp.asarray(agent.normalize_z(agent.network.select("backward_repr")(self.landmarks)))

        self.cost_matrix, self.reach_matrix = self._build_graph()
        self.all_dist, self.all_pred = dijkstra(self.cost_matrix, directed=True, return_predecessors=True)
        self.path_coords, self.path_latents, self.waypoint_coords, self.waypoints = [], [], [], []
        self.goal_z, self.current_path_idx, self.pos_history, self.stuck_count = None, 0, [], 0

    def _build_graph(self):
        n = len(self.landmarks)
        s_rep, z_tile = jnp.repeat(self.landmarks, n, axis=0), jnp.tile(self.landmark_latents, (n, 1))
        reach_flat = [np.asarray(_jit_batch_reach(self.agent, s_rep[i:i+45000], z_tile[i:i+45000])) for i in range(0, len(s_rep), 45000)]
        reach_m = np.concatenate(reach_flat, axis=0).reshape((n, n))
        dists_e = np.linalg.norm(self.landmark_coords[:, None, :] - self.landmark_coords[None, :, :], axis=-1)
        max_diag = float(np.max(np.diag(reach_m))) if np.max(np.diag(reach_m)) > 0 else 1.0
        cost_m = np.maximum(0.0, -np.log(np.clip(reach_m / max_diag, 1e-6, 1.0)))
        cost_m[dists_e > self.max_edge_radius] = np.inf
        cost_m[reach_m < self.reachability_cutoff] = np.inf
        np.fill_diagonal(cost_m, 0.0)
        return cost_m, reach_m

    def reset(self, obs, goal_z):
        obs_xy = np.asarray(obs[:2])
        s_idx, g_idx = _find_connected_landmarks(self.landmark_coords, self.landmark_latents, self.all_dist, obs_xy, goal_z)
        path = _backtrack_dijkstra_path(self.all_pred, s_idx, g_idx, self.n_landmarks)

        self.path_indices = path
        self.path_coords = [self.landmark_coords[i] for i in path]
        self.path_latents = [self.landmark_latents[i] for i in path]
        self.path_states = [np.asarray(self.landmarks[i]) for i in path]
        self.goal_z, self.current_path_idx, self.pos_history, self.stuck_count = goal_z, 0, [], 0
        self.current_seg = 0
        self.waypoint_coords, self.waypoints, self.waypoint_indices = _extract_corner_waypoints(self.path_coords, self.path_latents, self.lookahead_dist)
        self.path_dists = [0.0]
        for i in range(len(self.path_coords) - 1):
            self.path_dists.append(self.path_dists[-1] + float(np.linalg.norm(self.path_coords[i + 1] - self.path_coords[i])))

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
        best_idx, min_dist = _track_local_path_index(obs_xy, self.path_coords, self.current_path_idx, max_window=7)
        if min_dist > 4.8 and len(self.path_coords) > 2:
            self.reset(obs, goal_z)
            best_idx = 0
        self.current_path_idx = best_idx

        accum, target_idx = 0.0, self.current_path_idx
        while target_idx < len(self.path_coords) - 1 and accum < self.lookahead_dist:
            accum += np.linalg.norm(self.path_coords[target_idx + 1] - self.path_coords[target_idx])
            target_idx += 1

        dist_to_final = float(np.linalg.norm(obs_xy - self.path_coords[-1]))
        target_latent = self.goal_z if (target_idx >= len(self.path_coords) - 1 or dist_to_final <= 2.2) else self.path_latents[target_idx]
        target_xy = self.path_coords[-1] if (target_idx >= len(self.path_coords) - 1 or dist_to_final <= 2.2) else self.path_coords[target_idx]
        is_goal = bool(target_idx >= len(self.path_coords) - 1 or dist_to_final <= 2.2)

        is_stuck, self.stuck_count = _check_stuck_state(self.pos_history, obs_xy, dist_to_final, self.stuck_count)
        if self.stuck_count > 45:
            self.reset(obs, goal_z)
            self.stuck_count = 0

        eval_temp = 0.25 if is_stuck else temperature
        seed_k = jax.random.PRNGKey(step) if eval_temp > 0 else seed
        action, _ = _jit_baseline_step(self.agent, jnp.asarray(obs), target_latent, seed=seed_k, temperature=eval_temp)
        self.last_subgoal_info = {
            "subgoal_xy": [float(target_xy[0]), float(target_xy[1])],
            "waypoints_xy": [[float(c[0]), float(c[1])] for c in self.waypoint_coords],
            "is_direct_goal": is_goal,
        }
        return np.asarray(action)

    def get_subgoal_latent(self, obs, goal_z, step=0):
        self.reset(obs, goal_z)
        self.current_path_idx, _ = _track_local_path_index(np.asarray(obs[:2]), self.path_coords, self.current_path_idx, max_window=7)
        accum, target_idx = 0.0, self.current_path_idx
        while target_idx < len(self.path_coords) - 1 and accum < self.lookahead_dist:
            accum += np.linalg.norm(self.path_coords[target_idx + 1] - self.path_coords[target_idx])
            target_idx += 1
        return np.asarray(self.goal_z) if (target_idx >= len(self.path_coords) - 1 or float(np.linalg.norm(np.asarray(obs[:2]) - self.path_coords[-1])) <= 2.2) else np.asarray(self.path_latents[target_idx])


class RecursiveBisectionPlanner(BasePlanner):
    def __init__(self, agent, dataset_states, max_depth=2, n_candidates=200, hit_threshold=35.0, name="Recursive Bisection"):
        super().__init__(agent, name=name)
        self.max_depth, self.hit_threshold = max_depth, hit_threshold
        self.n_candidates = min(n_candidates, len(dataset_states))
        idxs = np.random.default_rng(42).choice(len(dataset_states), size=self.n_candidates, replace=False)
        self.candidate_states = jnp.asarray(dataset_states[idxs])
        self.candidate_coords = np.asarray(dataset_states[idxs][:, :2])
        self.candidate_latents = jnp.asarray(agent.normalize_z(agent.network.select("backward_repr")(self.candidate_states)))
        self.cached_latent, self.cached_coord, self.steps_on_latent = None, None, 0

    def _find_midpoint(self, obs, goal_latent):
        r_sw = _jit_batch_reach(self.agent, jnp.broadcast_to(obs[None, :], (self.candidate_states.shape[0], obs.shape[0])), self.candidate_latents)
        r_wg = _jit_batch_reach(self.agent, self.candidate_states, jnp.broadcast_to(goal_latent[None, :], (self.candidate_states.shape[0], goal_latent.shape[0])))
        best_idx = int(jnp.argmax(jnp.log(jnp.maximum(1e-4, r_sw)) + jnp.log(jnp.maximum(1e-4, r_wg))))
        self.cached_coord = self.candidate_coords[best_idx].tolist()
        return self.candidate_latents[best_idx]

    def reset(self, obs, goal_latent):
        self.cached_latent = self._find_midpoint(jnp.asarray(obs), jnp.asarray(goal_latent))
        self.steps_on_latent = 0
        self.last_subgoal_info = {"subgoal_xy": self.cached_coord, "waypoints_xy": [self.cached_coord] if self.cached_coord else [], "is_direct_goal": False}

    def sample_action(self, obs, goal_latent, step=0, seed=None, temperature=0.0):
        if self.cached_latent is None:
            self.reset(obs, goal_latent)
        r_curr = float(_jit_batch_reach(self.agent, jnp.asarray(obs)[None, :], self.cached_latent[None, :])[0])
        self.steps_on_latent += 1
        if r_curr >= self.hit_threshold or self.steps_on_latent > 60:
            self.reset(obs, goal_latent)
        action, _ = _jit_actor_from_latent(self.agent, jnp.asarray(obs), self.cached_latent)
        self.last_subgoal_info = {"subgoal_xy": self.cached_coord, "waypoints_xy": [self.cached_coord] if self.cached_coord else [], "is_direct_goal": False}
        return np.asarray(action)


class DistilledMLPPlanner(BasePlanner):
    def __init__(self, agent, model_type="dense_eca", checkpoint_path=None, hidden_dim=256, n_layers=3, num_heads=4, device="cpu", name=None):
        super().__init__(agent, name=name or f"Distilled ({model_type})")
        self.device = torch.device(device)
        self.model = build_student_model(model_type, 29, agent.config["latent_dim"], hidden_dim, n_layers, num_heads)
        if checkpoint_path and os.path.exists(checkpoint_path):
            self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.model.to(self.device).eval()
        self.pos_history = []

    def reset(self, obs, goal_latent):
        self.pos_history = []
        self.last_subgoal_info = {"subgoal_xy": None, "waypoints_xy": [], "is_direct_goal": False}

    def sample_action(self, obs, goal_latent, step=0, seed=None, temperature=0.0):
        obs_xy = np.asarray(obs[:2])
        self.pos_history.append(obs_xy.copy())
        if len(self.pos_history) > 40:
            self.pos_history.pop(0)
        is_stuck = bool(len(self.pos_history) >= 40 and float(np.linalg.norm(obs_xy - self.pos_history[0])) < 0.4)
        eval_temp = 0.2 if is_stuck else temperature
        seed_k = jax.random.PRNGKey(step) if eval_temp > 0 else seed

        inp = torch.from_numpy(np.concatenate([obs, np.asarray(goal_latent)], axis=-1).astype(np.float32)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            pred_z = self.model(inp).squeeze(0).cpu().numpy()
        action, _ = _jit_baseline_step(self.agent, jnp.asarray(obs), jnp.asarray(pred_z), seed=seed_k, temperature=eval_temp)
        self.last_subgoal_info = {"subgoal_xy": None, "waypoints_xy": [], "is_direct_goal": False}
        return np.asarray(action)


class SequenceWaypointAttentionPlanner(BufferGraphPlanner):
    """Standard Sequence-Aware Attention Translator."""
    def __init__(self, agent, dataset_observations, checkpoint_path, n_landmarks=1000, max_edge_radius=3.5, reachability_cutoff=35.0, lookahead_dist=2.6, max_seq_len=16, hidden_dim=256, num_heads=4, n_layers=2, name="Dijkstra + Sequence Attention"):
        super().__init__(agent, dataset_observations, n_landmarks, max_edge_radius, reachability_cutoff, lookahead_dist, name)
        with open(checkpoint_path, "rb") as f:
            data = pickle.load(f)
            self.seq_params = flax.core.freeze(data["params"])
            cfg = data.get("config", {})
            h_dim, n_heads, n_lay, m_len = cfg.get("hidden_dim", hidden_dim), cfg.get("num_heads", num_heads), cfg.get("n_layers", n_layers), cfg.get("max_seq_len", max_seq_len)
        self.max_seq_len = m_len
        self.seq_translator_def = FlaxSequenceWaypointAttentionTranslator(latent_dim=agent.config["latent_dim"], hidden_dim=h_dim, num_heads=n_heads, max_seq_len=m_len, n_layers=n_lay)

        @functools.partial(jax.jit, static_argnames=("temp",))
        def _fused_step(obs_jnp, seq_jnp, mask_jnp, params, seed_k=None, temp=0.0):
            z_cmd = self.seq_translator_def.apply({"params": params}, obs_jnp[None, :], seq_jnp[None, :, :], mask_jnp[None, :])[0]
            act_dist = agent.network.select("actor")(obs_jnp[None, :], z_cmd[None, :], goal_encoded=True, temperature=temp)
            a = act_dist.mode() if (temp == 0.0 or seed_k is None) else act_dist.sample(seed=seed_k)
            return jnp.clip(a[0], -1.0, 1.0), z_cmd
        self._fused_seq_step = _fused_step

    def sample_action(self, obs, goal_z, step=0, seed=None, temperature=0.0):
        if not self.path_coords:
            self.reset(obs, goal_z)
        self.current_path_idx, _ = _track_local_path_index(np.asarray(obs[:2]), self.path_coords, self.current_path_idx, max_window=7)
        accum, target_idx = 0.0, self.current_path_idx
        while target_idx < len(self.path_coords) - 1 and accum < self.lookahead_dist:
            accum += np.linalg.norm(self.path_coords[target_idx + 1] - self.path_coords[target_idx])
            target_idx += 1

        dist_final = float(np.linalg.norm(np.asarray(obs[:2]) - self.path_coords[-1]))
        future_latents = [goal_z] if dist_final < 1.8 else (self.path_latents[target_idx:] + [goal_z])
        K = min(len(future_latents), self.max_seq_len)
        pad_seq = np.zeros((self.max_seq_len, self.agent.config["latent_dim"]), dtype=np.float32)
        seq_mask = np.zeros(self.max_seq_len, dtype=bool)
        for i in range(K):
            pad_seq[i], seq_mask[i] = future_latents[i], True

        curr_c = self.path_coords[-1] if dist_final < 1.8 else self.path_coords[target_idx]
        self.last_subgoal_info = {"subgoal_xy": [float(curr_c[0]), float(curr_c[1])], "waypoints_xy": [[float(c[0]), float(c[1])] for c in self.waypoint_coords], "is_direct_goal": bool(dist_final < 1.8)}
        action, _ = self._fused_seq_step(jnp.asarray(obs), jnp.asarray(pad_seq), jnp.asarray(seq_mask), self.seq_params, jax.random.PRNGKey(step) if temperature > 0 else seed, temperature)
        return np.asarray(action)


# ==============================================================================
# Distilled JAX & Waypoint Translators
# ==============================================================================

class DistilledJAXPlanner(BasePlanner):
    def __init__(self, agent, model_type="gated_attn", checkpoint_path=None, dataset_states=None, hidden_dim=256, n_layers=3, name=None):
        name = name or f"Distilled JAX ({model_type})"
        super().__init__(agent, name=name)
        self.student_def = build_flax_translator(model_type=model_type, latent_dim=agent.config["latent_dim"], hidden_dim=hidden_dim, n_layers=n_layers)
        if checkpoint_path and os.path.exists(checkpoint_path):
            with open(checkpoint_path, "rb") as f:
                self.params = flax.core.freeze(pickle.load(f)["params"])
        else:
            self.params = self.student_def.init(jax.random.PRNGKey(0), jnp.zeros((1, 29)), jnp.zeros((1, agent.config["latent_dim"])))["params"]

        if dataset_states is not None:
            n_samples = min(500, len(dataset_states))
            idxs = np.random.default_rng(42).choice(len(dataset_states), size=n_samples, replace=False)
            self.ref_coords = np.asarray(dataset_states[idxs][:, :2])
            self.ref_latents = jnp.asarray(agent.normalize_z(agent.network.select("backward_repr")(jnp.asarray(dataset_states[idxs]))))
        else:
            self.ref_coords, self.ref_latents = None, None

        self.pos_history = []
        @functools.partial(jax.jit, static_argnames=("temp",))
        def _fused_step(obs_jnp, goal_jnp, params, seed_k=None, temp=0.0):
            z_cmd = self.student_def.apply({"params": params}, obs_jnp[None, :], goal_jnp[None, :])[0]
            act_dist = agent.network.select("actor")(obs_jnp[None, :], z_cmd[None, :], goal_encoded=True, temperature=temp)
            a = act_dist.mode() if (temp == 0.0 or seed_k is None) else act_dist.sample(seed=seed_k)
            return jnp.clip(a[0], -1.0, 1.0), z_cmd
        self._fused_step = _fused_step

    def reset(self, obs=None, goal_latent=None):
        self.pos_history = []
        self.last_subgoal_info = {"subgoal_xy": None, "waypoints_xy": [], "is_direct_goal": False}

    def sample_action(self, obs, goal_latent, step=0, seed=None, temperature=0.0):
        obs_xy = np.asarray(obs[:2])
        self.pos_history.append(obs_xy.copy())
        if len(self.pos_history) > 40:
            self.pos_history.pop(0)
        is_stuck = bool(len(self.pos_history) >= 40 and float(np.linalg.norm(obs_xy - self.pos_history[0])) < 0.4)
        eval_temp = 0.2 if is_stuck else temperature
        seed_k = jax.random.PRNGKey(step) if eval_temp > 0 else seed
        action, z_cmd = self._fused_step(jnp.asarray(obs), jnp.asarray(goal_latent), self.params, seed_k, eval_temp)
        decoded_xy = self.ref_coords[int(_jit_decode_latent_to_coords(z_cmd, self.ref_latents))].tolist() if self.ref_latents is not None else None
        self.last_subgoal_info = {"subgoal_xy": decoded_xy, "waypoints_xy": [decoded_xy] if decoded_xy else [], "is_direct_goal": False}
        return np.asarray(action)


class WaypointTranslatorPlanner(BufferGraphPlanner):
    """Combines topological Dijkstra planning with learned FlaxSingleWaypointTranslator."""
    def __init__(self, agent, dataset_observations, checkpoint_path, n_landmarks=1000, max_edge_radius=3.5, reachability_cutoff=35.0, lookahead_dist=2.6, hidden_dim=256, n_layers=3, name="Dijkstra + Waypoint Translator"):
        super().__init__(agent, dataset_observations, n_landmarks, max_edge_radius, reachability_cutoff, lookahead_dist, name)
        with open(checkpoint_path, "rb") as f:
            data = pickle.load(f)
            self.translator_params = flax.core.freeze(data["params"])
            cfg = data.get("config", {})
            h_dim = cfg.get("hidden_dim", hidden_dim)
            n_lay = cfg.get("n_layers", n_layers)

        self.translator_def = FlaxSingleWaypointTranslator(latent_dim=agent.config["latent_dim"], hidden_dim=h_dim, n_layers=n_lay)
        @functools.partial(jax.jit, static_argnames=("temp",))
        def _fused_step(obs_jnp, w1_jnp, params, seed_k=None, temp=0.0):
            z_cmd = self.translator_def.apply({"params": params}, obs_jnp[None, :], w1_jnp[None, :])[0]
            act_dist = agent.network.select("actor")(obs_jnp[None, :], z_cmd[None, :], goal_encoded=True, temperature=temp)
            a = act_dist.mode() if (temp == 0.0 or seed_k is None) else act_dist.sample(seed=seed_k)
            return jnp.clip(a[0], -1.0, 1.0), z_cmd
        self._fused_step = _fused_step

    def sample_action(self, obs, goal_z, step=0, seed=None, temperature=0.0):
        if not self.path_coords:
            self.reset(obs, goal_z)
        self.current_path_idx, _ = _track_local_path_index(np.asarray(obs[:2]), self.path_coords, self.current_path_idx, max_window=12)
        accum, target_idx = 0.0, self.current_path_idx
        while target_idx < len(self.path_coords) - 1 and accum < self.lookahead_dist:
            accum += np.linalg.norm(self.path_coords[target_idx + 1] - self.path_coords[target_idx])
            target_idx += 1

        dist_final = float(np.linalg.norm(np.asarray(obs[:2]) - self.path_coords[-1]))
        target_latent = goal_z if dist_final < 1.8 else self.path_latents[target_idx]
        curr_c = self.path_coords[-1] if dist_final < 1.8 else self.path_coords[target_idx]
        self.last_subgoal_info = {"subgoal_xy": [float(curr_c[0]), float(curr_c[1])], "waypoints_xy": [[float(c[0]), float(c[1])] for c in self.waypoint_coords], "is_direct_goal": bool(dist_final < 1.8)}
        seed_k = jax.random.PRNGKey(step) if temperature > 0 else seed
        action, _ = self._fused_step(jnp.asarray(obs), jnp.asarray(target_latent), self.translator_params, seed_k, temperature)
        return np.asarray(action)


class EnhancedSequenceWaypointAttentionPlanner(BufferGraphPlanner):
    """Dijkstra Planner paired with Enhanced ALiBi Sequence Attention Transformer."""
    def __init__(self, agent, dataset_observations, checkpoint_path, n_landmarks=1000, max_edge_radius=3.5, reachability_cutoff=35.0, lookahead_dist=2.6, max_seq_len=16, hidden_dim=384, num_heads=6, n_layers=3, dropout_rate=0.2, alibi_slope=0.4, local_window_size=6, name="Dijkstra + Enhanced Sequence Attention Translator"):
        super().__init__(agent, dataset_observations, n_landmarks, max_edge_radius, reachability_cutoff, lookahead_dist, name)
        with open(checkpoint_path, "rb") as f:
            data = pickle.load(f)
            self.seq_params = flax.core.freeze(data["params"])
            cfg = data.get("config", {})
            h_dim, n_heads, n_lay, m_len = cfg.get("hidden_dim", hidden_dim), cfg.get("num_heads", num_heads), cfg.get("n_layers", n_layers), cfg.get("max_seq_len", max_seq_len)
            d_rate, a_slope, w_size = cfg.get("dropout_rate", dropout_rate), cfg.get("alibi_slope", alibi_slope), cfg.get("local_window_size", local_window_size)

        self.max_seq_len = m_len
        self.seq_translator_def = FlaxEnhancedSequenceWaypointAttentionTranslator(latent_dim=agent.config["latent_dim"], hidden_dim=h_dim, num_heads=n_heads, max_seq_len=m_len, n_layers=n_lay, dropout_rate=d_rate, alibi_slope=a_slope, local_window_size=w_size)

        @functools.partial(jax.jit, static_argnames=("temp",))
        def _fused_step(obs_jnp, seq_jnp, mask_jnp, curv_jnp, params, seed_k=None, temp=0.0):
            z_cmd = self.seq_translator_def.apply({"params": params}, obs_jnp[None, :], seq_jnp[None, :, :], mask_jnp[None, :], curv_jnp[None, :, :], deterministic=True)[0]
            act_dist = agent.network.select("actor")(obs_jnp[None, :], z_cmd[None, :], goal_encoded=True, temperature=temp)
            a = act_dist.mode() if (temp == 0.0 or seed_k is None) else act_dist.sample(seed=seed_k)
            return jnp.clip(a[0], -1.0, 1.0), z_cmd
        self._fused_enhanced_seq_step = _fused_step

    def _prepare_sequence_inputs(self, obs, obs_xy, goal_z):
        best_idx, min_dist = _track_local_path_index(obs_xy, self.path_coords, self.current_path_idx, max_window=5)
        if min_dist > 4.8 and len(self.path_coords) > 2:
            self.reset(obs, goal_z)
            best_idx = 0
        self.current_path_idx = best_idx

        accum, target_idx = 0.0, self.current_path_idx
        N_pts = len(self.path_coords)
        while target_idx < N_pts - 1 and accum < self.lookahead_dist:
            accum += np.linalg.norm(self.path_coords[target_idx + 1] - self.path_coords[target_idx])
            target_idx += 1

        dist_final = float(np.linalg.norm(obs_xy - self.path_coords[-1]))
        if target_idx >= N_pts - 1 or dist_final <= 2.2:
            future_latents, future_coords, is_direct = [goal_z], [self.path_coords[-1]], True
            curr_c = self.path_coords[-1]
        else:
            future_latents = self.path_latents[target_idx:] + [goal_z]
            future_coords = self.path_coords[target_idx:]
            curr_c = self.path_coords[target_idx]
            is_direct = False

        curvs = _extract_curvature_angles(future_coords)
        K = min(len(future_latents), self.max_seq_len)
        pad_seq = np.zeros((self.max_seq_len, self.agent.config["latent_dim"]), dtype=np.float32)
        seq_mask = np.zeros(self.max_seq_len, dtype=bool)
        curv_arr = np.ones((self.max_seq_len, 1), dtype=np.float32)
        for i in range(K):
            pad_seq[i], seq_mask[i] = future_latents[i], True
            if i < len(curvs):
                curv_arr[i, 0] = curvs[i]

        return pad_seq, seq_mask, curv_arr, curr_c, is_direct, dist_final

    def sample_action(self, obs, goal_z, step=0, seed=None, temperature=0.0):
        if not self.path_coords:
            self.reset(obs, goal_z)
        obs_xy = np.asarray(obs[:2])
        pad_seq, seq_mask, curv_arr, curr_c_coarse, is_direct, dist_final = self._prepare_sequence_inputs(obs, obs_xy, goal_z)

        # Calculate exact continuous sliding lookahead point along the path
        curr_c, self.current_seg, s_agent = _compute_continuous_lookahead(
            obs_xy, self.path_coords, self.lookahead_dist, getattr(self, "current_seg", 0), path_dists=getattr(self, "path_dists", None)
        )
        if is_direct:
            curr_c = np.array(self.path_coords[-1])

        is_stuck, self.stuck_count = _check_stuck_state(self.pos_history, obs_xy, dist_final, self.stuck_count)
        if self.stuck_count > 45:
            self.reset(obs, goal_z)
            self.stuck_count = 0

        # Build attention targets: sliding lookahead point c_t first, then strictly downstream corner waypoints and goal
        wp_indices = getattr(self, "waypoint_indices", list(range(len(self.waypoint_coords))))
        path_dists = getattr(self, "path_dists", None)
        downstream_wps = []
        for w, k in zip(self.waypoint_coords, wp_indices):
            s_wp = path_dists[k] if (path_dists is not None and k < len(path_dists)) else float(k)
            if s_wp > s_agent + 0.8 and np.linalg.norm(np.array(w) - curr_c) > 0.8:
                downstream_wps.append(w)

        if is_direct:
            attn_targets = [curr_c]
        else:
            attn_targets = [curr_c] + downstream_wps[:3]
            if len(attn_targets) > 0 and np.linalg.norm(np.array(attn_targets[-1]) - np.array(self.path_coords[-1])) > 0.5:
                attn_targets.append(self.path_coords[-1])

        slopes = np.exp(-0.4 * np.arange(len(attn_targets)))
        attn_weights = (slopes / np.sum(slopes)).tolist()

        eval_temp = 0.25 if is_stuck else temperature
        seed_k = jax.random.PRNGKey(step) if eval_temp > 0 else seed
        self.last_subgoal_info = {
            "subgoal_xy": [float(curr_c[0]), float(curr_c[1])],
            "lookahead_xy": [float(curr_c[0]), float(curr_c[1])],
            "attention_targets": [[float(t[0]), float(t[1])] for t in attn_targets],
            "attention_weights": [float(w) for w in attn_weights],
            "waypoints_xy": [[float(c[0]), float(c[1])] for c in self.waypoint_coords],
            "is_direct_goal": is_direct,
            "stuck_count": int(self.stuck_count),
        }
        action, _ = self._fused_enhanced_seq_step(jnp.asarray(obs), jnp.asarray(pad_seq), jnp.asarray(seq_mask), jnp.asarray(curv_arr), self.seq_params, seed_k, eval_temp)
        return np.asarray(action)
