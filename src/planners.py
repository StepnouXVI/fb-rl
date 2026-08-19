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
def _jit_graph_step(agent, obs, goal_z, curr_wp):
    """
    Fused graph planner step:
    Computes reachability to goal & curr_wp, plus candidate actions in a single batch-2 pass.
    """
    obs_2 = jnp.stack([obs, obs], axis=0)
    targets = jnp.stack([goal_z, curr_wp], axis=0)

    # 1. Forward reachability for [goal, curr_wp]
    f = agent.network.select("forward_repr")(obs_2, targets, goal_encoded=True)
    if f.ndim == 3:
        f = jnp.mean(f, axis=0)
    reaches = jnp.sum(f * targets, axis=-1)  # shape (2,) -> [reach_goal, reach_wp]

    # 2. High actor local steering
    high_dist = agent.network.select("high_actor")(obs_2, targets, goal_encoded=True, temperature=0.0)
    subgoals = agent.normalize_z(high_dist.mode())  # shape (2, latent_dim)

    # 3. Low actor action execution
    low_dist = agent.network.select("actor")(obs_2, subgoals, goal_encoded=True, temperature=0.0)
    actions = jnp.clip(low_dist.mode(), -1.0, 1.0)  # shape (2, act_dim)

    return reaches, actions, subgoals


@jax.jit
def _jit_plan_query(agent, obs, goal_z, landmark_states, landmark_latents):
    """Vectorized start and goal node query in 1 jitted pass."""
    obs_exp = jnp.broadcast_to(obs[None, :], (landmark_states.shape[0], obs.shape[0]))
    f_start = agent.network.select("forward_repr")(obs_exp, landmark_latents, goal_encoded=True)
    if f_start.ndim == 3:
        f_start = jnp.mean(f_start, axis=0)
    r_start = jnp.sum(f_start * landmark_latents, axis=-1)
    start_idx = jnp.argmax(r_start)

    z_exp = jnp.broadcast_to(goal_z[None, :], (landmark_states.shape[0], goal_z.shape[0]))
    f_goal = agent.network.select("forward_repr")(landmark_states, z_exp, goal_encoded=True)
    if f_goal.ndim == 3:
        f_goal = jnp.mean(f_goal, axis=0)
    r_goal = jnp.sum(f_goal * z_exp, axis=-1)
    goal_idx = jnp.argmax(r_goal)

    return start_idx, goal_idx


@jax.jit
def _jit_actor_from_latent(agent, obs, latent):
    """Jitted low-actor execution given a latent intention."""
    obs_b = obs[None, :] if obs.ndim == 1 else obs
    z_b = latent[None, :] if latent.ndim == 1 else latent
    high_dist = agent.network.select("high_actor")(obs_b, z_b, goal_encoded=True, temperature=0.0)
    subgoal_z = agent.normalize_z(high_dist.mode())
    low_dist = agent.network.select("actor")(obs_b, subgoal_z, goal_encoded=True, temperature=0.0)
    return jnp.clip(low_dist.mode()[0], -1.0, 1.0), subgoal_z[0]


# ==========================================
# Optimized Planner Classes
# ==========================================

class BasePlanner:
    def __init__(self, agent, name="BasePlanner"):
        self.agent = agent
        self.name = name

    def reset(self, obs, goal_latent):
        pass

    def sample_action(self, obs, goal_latent, step=0, seed=None, temperature=0.0):
        action, _ = _jit_baseline_step(
            self.agent, jnp.asarray(obs), jnp.asarray(goal_latent), seed=seed, temperature=temperature
        )
        return np.asarray(action)


class BaselinePlanner(BasePlanner):
    def __init__(self, agent, use_high_actor=True, name="Single-Intention Baseline"):
        super().__init__(agent, name=name)
        self.use_high_actor = use_high_actor

    def sample_action(self, obs, goal_latent, step=0, seed=None, temperature=0.0):
        obs_jnp = jnp.asarray(obs)
        goal_jnp = jnp.asarray(goal_latent)
        if self.use_high_actor:
            action, _ = _jit_baseline_step(self.agent, obs_jnp, goal_jnp, seed=seed, temperature=temperature)
        else:
            norm_z = self.agent.normalize_z(goal_jnp)
            low_dist = self.agent.network.select("actor")(
                obs_jnp[None, :], norm_z[None, :], goal_encoded=True, temperature=temperature
            )
            action = jnp.clip(low_dist.mode()[0], -1.0, 1.0)
        return np.asarray(action)


class BufferGraphPlanner(BasePlanner):
    """
    Topological Graph Shortest-Path Planner with Precomputed APSP and Fused Step JIT.
    Pure Forward-Backward Reachability + Dijkstra + High-Actor Local Steering.
    # ponytail: Precomputed All-Pairs Shortest Paths (APSP) + JIT Fused Execution.
    """
    def __init__(
        self,
        agent,
        dataset_states,
        n_landmarks=300,
        reachability_cutoff=20.0,
        hit_threshold=35.0,
        name="Buffer Graph Dijkstra",
    ):
        super().__init__(agent, name=name)
        self.n_landmarks = min(n_landmarks, len(dataset_states))
        self.reachability_cutoff = reachability_cutoff
        self.hit_threshold = hit_threshold

        # 1. Sample landmarks uniformly
        rng = np.random.default_rng(42)
        idxs = rng.choice(len(dataset_states), size=self.n_landmarks, replace=False)
        self.landmarks = jnp.asarray(dataset_states[idxs])
        self.landmark_latents = jnp.asarray(
            agent.normalize_z(agent.network.select("backward_repr")(self.landmarks))
        )

        # 2. Build graph & Precompute All-Pairs Shortest Paths (APSP)
        self.cost_matrix, self.max_diag = self._build_graph()
        self.all_dist, self.all_pred = dijkstra(
            self.cost_matrix, directed=True, return_predecessors=True
        )

        self.waypoints = []
        self.current_idx = 0
        self.steps_on_wp = 0

    def _build_graph(self):
        n = len(self.landmarks)
        s_rep = jnp.repeat(self.landmarks, n, axis=0)
        z_tile = jnp.tile(self.landmark_latents, (n, 1))

        # Vectorized jitted evaluation of all pairs
        reach_flat = []
        for i in range(0, len(s_rep), 45000):
            sb = s_rep[i:i+45000]
            zb = z_tile[i:i+45000]
            reach_flat.append(np.asarray(_jit_batch_reach(self.agent, sb, zb)))
        reach_matrix = np.concatenate(reach_flat, axis=0).reshape((n, n))

        max_diag = float(np.max(np.diag(reach_matrix))) if np.max(np.diag(reach_matrix)) > 0 else 1.0
        normalized = np.clip(reach_matrix / max_diag, 1e-6, 1.0)
        cost_matrix = np.maximum(0.0, -np.log(normalized))
        cost_matrix[reach_matrix < self.reachability_cutoff] = np.inf
        np.fill_diagonal(cost_matrix, 0.0)
        return cost_matrix, max_diag

    def _plan(self, obs, goal_z):
        start_idx, goal_idx = _jit_plan_query(
            self.agent, jnp.asarray(obs), jnp.asarray(goal_z), self.landmarks, self.landmark_latents
        )
        start_idx, goal_idx = int(start_idx), int(goal_idx)

        # O(1) Precomputed Dijkstra lookup
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
            if len(path) > 1:
                path = path[1:]
            return [self.landmark_latents[i] for i in path] + [goal_z]

        return [self.landmark_latents[goal_idx], goal_z]

    def reset(self, obs, goal_z):
        self.waypoints = self._plan(obs, goal_z)
        self.current_idx = 0
        self.steps_on_wp = 0

    def sample_action(self, obs, goal_z, step=0, seed=None, temperature=0.0):
        obs_jnp = jnp.asarray(obs)
        goal_jnp = jnp.asarray(goal_z)

        if not self.waypoints:
            self.reset(obs, goal_z)

        curr_wp = self.waypoints[self.current_idx]

        # Single fused JIT step execution
        reaches, actions, _ = _jit_graph_step(self.agent, obs_jnp, goal_jnp, curr_wp)

        reach_goal = float(reaches[0])
        reach_wp = float(reaches[1])

        # 1. Short-circuit: Direct visibility to global goal
        if reach_goal >= self.hit_threshold * 0.7:
            return np.asarray(actions[0])

        self.steps_on_wp += 1

        # 2. Hitting time waypoint switching
        if reach_wp >= self.hit_threshold and self.current_idx < len(self.waypoints) - 1:
            self.current_idx += 1
            self.steps_on_wp = 0
        # 3. Fallback: Replanning if stuck
        elif self.steps_on_wp > 80 and self.current_idx < len(self.waypoints) - 1:
            self.reset(obs, goal_z)

        return np.asarray(actions[1])


class RecursiveBisectionPlanner(BasePlanner):
    def __init__(self, agent, dataset_states, max_depth=2, n_candidates=200, hit_threshold=35.0, name="Recursive Bisection"):
        super().__init__(agent, name=name)
        self.max_depth = max_depth
        self.n_candidates = min(n_candidates, len(dataset_states))
        self.hit_threshold = hit_threshold

        rng = np.random.default_rng(42)
        idxs = rng.choice(len(dataset_states), size=self.n_candidates, replace=False)
        self.candidate_states = jnp.asarray(dataset_states[idxs])
        self.candidate_latents = jnp.asarray(
            agent.normalize_z(agent.network.select("backward_repr")(self.candidate_states))
        )
        self.cached_latent = None
        self.steps_on_latent = 0

    def _find_midpoint_fast(self, obs, goal_latent):
        obs_exp = jnp.broadcast_to(obs[None, :], (self.candidate_states.shape[0], obs.shape[0]))
        r_sw = _jit_batch_reach(self.agent, obs_exp, self.candidate_latents)

        g_exp = jnp.broadcast_to(goal_latent[None, :], (self.candidate_states.shape[0], goal_latent.shape[0]))
        r_wg = _jit_batch_reach(self.agent, self.candidate_states, g_exp)

        scores = jnp.log(jnp.maximum(1e-4, r_sw)) + jnp.log(jnp.maximum(1e-4, r_wg))
        best_idx = int(jnp.argmax(scores))
        return self.candidate_latents[best_idx]

    def reset(self, obs, goal_latent):
        self.cached_latent = self._find_midpoint_fast(jnp.asarray(obs), jnp.asarray(goal_latent))
        self.steps_on_latent = 0

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

    def sample_action(self, obs, goal_latent, step=0, seed=None, temperature=0.0):
        x = np.concatenate([obs, np.asarray(goal_latent)], axis=-1).astype(np.float32)
        with torch.no_grad():
            inp = torch.from_numpy(x).unsqueeze(0)
            pred_z = self.model(inp).squeeze(0).numpy()

        action, _ = _jit_actor_from_latent(self.agent, jnp.asarray(obs), jnp.asarray(pred_z))
        return np.asarray(action)
