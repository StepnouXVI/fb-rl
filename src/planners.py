import os
import numpy as np
import jax, jax.numpy as jnp
from scipy.sparse.csgraph import dijkstra
import torch
import torch.nn as nn
from src.models import build_student_model

# ponytail: Common interface for all high-level controllers
class BasePlanner:
    def __init__(self, agent, name="BasePlanner"):
        self.agent = agent
        self.name = name
        self.rng = jax.random.PRNGKey(0)

    def reset(self, obs, goal_latent):
        pass

    def get_subgoal_latent(self, obs, goal_latent, step=0):
        return goal_latent

    def sample_action(self, obs, goal_latent, step=0, seed=None, temperature=0.0):
        # ponytail: Pass high-level intention to low-level frozen JAX actor
        subgoal_z = self.get_subgoal_latent(obs, goal_latent, step=step)
        subgoal_z = self.agent.normalize_z(subgoal_z)

        obs_jnp = jnp.asarray(obs)[None, :] if obs.ndim == 1 else jnp.asarray(obs)
        z_jnp = jnp.asarray(subgoal_z)[None, :] if subgoal_z.ndim == 1 else jnp.asarray(subgoal_z)

        low_dist = self.agent.network.select("actor")(obs_jnp, z_jnp, goal_encoded=True, temperature=temperature)
        if temperature == 0.0 or seed is None:
            action = low_dist.mode()
        else:
            action = low_dist.sample(seed=seed)
        return np.clip(np.asarray(action)[0], -1.0, 1.0)


class BaselinePlanner(BasePlanner):
    def __init__(self, agent, use_high_actor=True, name="Single-Intention Baseline"):
        super().__init__(agent, name=name)
        self.use_high_actor = use_high_actor

    def get_subgoal_latent(self, obs, goal_latent, step=0):
        if not self.use_high_actor:
            return self.agent.normalize_z(goal_latent)

        obs_jnp = jnp.asarray(obs)[None, :] if obs.ndim == 1 else jnp.asarray(obs)
        z_jnp = jnp.asarray(goal_latent)[None, :] if goal_latent.ndim == 1 else jnp.asarray(goal_latent)

        high_dist = self.agent.network.select("high_actor")(obs_jnp, z_jnp, goal_encoded=True, temperature=0.0)
        subgoal_z = high_dist.mode()
        return np.asarray(self.agent.normalize_z(subgoal_z))[0]


class RecursiveBisectionPlanner(BasePlanner):
    def __init__(self, agent, dataset_states, max_depth=2, n_candidates=200, hit_threshold=35.0, name="Recursive Bisection"):
        super().__init__(agent, name=name)
        self.max_depth = max_depth
        self.n_candidates = min(n_candidates, len(dataset_states))
        self.hit_threshold = hit_threshold

        rng = np.random.default_rng(42)
        idxs = rng.choice(len(dataset_states), size=self.n_candidates, replace=False)
        self.candidate_states = dataset_states[idxs]
        self.candidate_latents = np.asarray(
            agent.normalize_z(agent.network.select("backward_repr")(self.candidate_states))
        )
        self.cached_latent = None
        self.steps_on_latent = 0

    def _eval_reachability(self, s, target_latents):
        s_expanded = jnp.tile(jnp.asarray(s)[None, :], (len(target_latents), 1))
        f_repr = self.agent.network.select("forward_repr")(s_expanded, target_latents, goal_encoded=True)
        if f_repr.ndim == 3:
            f_repr = jnp.mean(f_repr, axis=0)
        return jnp.sum(f_repr * target_latents, axis=-1)

    def _find_midpoint(self, s, goal_latent, depth=0):
        if depth >= self.max_depth:
            return goal_latent

        r_sw = np.maximum(1e-4, np.asarray(self._eval_reachability(s, self.candidate_latents)))
        z_g_expanded = jnp.tile(jnp.asarray(goal_latent)[None, :], (len(self.candidate_states), 1))
        f_wg = self.agent.network.select("forward_repr")(self.candidate_states, z_g_expanded, goal_encoded=True)
        if f_wg.ndim == 3:
            f_wg = jnp.mean(f_wg, axis=0)
        r_wg = np.maximum(1e-4, np.asarray(jnp.sum(f_wg * z_g_expanded, axis=-1)))

        scores = np.log(r_sw) + np.log(r_wg)
        best_idx = int(np.argmax(scores))
        best_w_latent = self.candidate_latents[best_idx]

        if depth + 1 < self.max_depth:
            return self._find_midpoint(s, best_w_latent, depth=depth + 1)
        return best_w_latent

    def reset(self, obs, goal_latent):
        self.cached_latent = self._find_midpoint(obs, goal_latent)
        self.steps_on_latent = 0

    def get_subgoal_latent(self, obs, goal_latent, step=0):
        if self.cached_latent is None:
            self.reset(obs, goal_latent)
            return self.cached_latent

        r_curr = float(np.asarray(self._eval_reachability(obs, self.cached_latent[None, :]))[0])
        self.steps_on_latent += 1
        if r_curr >= self.hit_threshold or self.steps_on_latent > 60:
            self.reset(obs, goal_latent)
        
        # Steer high_actor towards current midpoint
        obs_jnp = jnp.asarray(obs)[None, :]
        z_jnp = jnp.asarray(self.cached_latent)[None, :]
        high_dist = self.agent.network.select("high_actor")(obs_jnp, z_jnp, goal_encoded=True, temperature=0.0)
        return np.asarray(self.agent.normalize_z(high_dist.mode()))[0]


class BufferGraphPlanner(BasePlanner):
    """
    Topological Graph Shortest-Path Planner over Replay Buffer.
    Pure Forward-Backward Reachability + Dijkstra + High-Actor Local Steering.
    # ponytail: Vectorized batch Dijkstra with hitting-time waypoint switching.
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

        # 1. Sample landmarks uniformly from replay buffer
        rng = np.random.default_rng(42)
        idxs = rng.choice(len(dataset_states), size=self.n_landmarks, replace=False)
        self.landmarks = dataset_states[idxs]
        self.landmark_latents = np.asarray(
            agent.normalize_z(agent.network.select("backward_repr")(self.landmarks))
        )

        # 2. Vectorized batch NxN reachability matrix
        self.cost_matrix, self.max_diag = self._build_graph()
        self.waypoints = []
        self.current_idx = 0
        self.steps_on_wp = 0

    def _batch_reach(self, states, targets):
        f = self.agent.network.select("forward_repr")(states, targets, goal_encoded=True)
        if f.ndim == 3:
            f = jnp.mean(f, axis=0)
        return jnp.sum(f * targets, axis=-1)

    def _build_graph(self):
        n = len(self.landmarks)
        s_rep = np.repeat(self.landmarks, n, axis=0)
        z_tile = np.tile(self.landmark_latents, (n, 1))
        reach_flat = []
        for i in range(0, len(s_rep), 20000):
            sb = jnp.asarray(s_rep[i:i+20000])
            zb = jnp.asarray(z_tile[i:i+20000])
            reach_flat.append(np.asarray(self._batch_reach(sb, zb)))
        reach_matrix = np.concatenate(reach_flat, axis=0).reshape((n, n))

        max_diag = float(np.max(np.diag(reach_matrix))) if np.max(np.diag(reach_matrix)) > 0 else 1.0
        normalized = np.clip(reach_matrix / max_diag, 1e-6, 1.0)
        cost_matrix = np.maximum(0.0, -np.log(normalized))
        cost_matrix[reach_matrix < self.reachability_cutoff] = np.inf
        np.fill_diagonal(cost_matrix, 0.0)
        return cost_matrix, max_diag

    def _plan(self, obs, goal_z):
        n = len(self.landmarks)
        # 1. Start node: reachability from obs to landmarks (pure FB metric)
        obs_tile = jnp.tile(jnp.asarray(obs)[None, :], (n, 1))
        r_start = np.asarray(self._batch_reach(obs_tile, self.landmark_latents))
        start_idx = int(np.argmax(r_start))

        # 2. Goal node: reachability from landmarks to goal_z (pure FB metric)
        z_g_tile = jnp.tile(jnp.asarray(goal_z)[None, :], (n, 1))
        r_goal = np.asarray(self._batch_reach(self.landmarks, z_g_tile))
        goal_idx = int(np.argmax(r_goal))

        dist_mat, pred = dijkstra(self.cost_matrix, directed=True, indices=start_idx, return_predecessors=True)
        path = []
        curr = goal_idx
        while curr != -9999 and curr != start_idx:
            path.append(curr)
            curr = pred[curr]
            if len(path) > n:
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

    def get_subgoal_latent(self, obs, goal_z, step=0):
            obs_jnp = jnp.asarray(obs)[None, :]
            z_goal_jnp = jnp.asarray(goal_z)[None, :]

            # 1. Short-circuit: Прямая видимость до глобальной цели
            reach_goal = float(np.asarray(self._batch_reach(obs_jnp, z_goal_jnp))[0])
            if reach_goal >= self.hit_threshold * 0.7: 
                # Напрямую подруливаем к финишу
                high_dist = self.agent.network.select("high_actor")(obs_jnp, z_goal_jnp, goal_encoded=True, temperature=0.0)
                return np.asarray(self.agent.normalize_z(high_dist.mode()))[0]

            if not self.waypoints:
                self.reset(obs, goal_z)
                return self.waypoints[0]

            curr_wp = self.waypoints[self.current_idx]
            z_jnp = jnp.asarray(curr_wp)[None, :]
            reach = float(np.asarray(self._batch_reach(obs_jnp, z_jnp))[0])

            self.steps_on_wp += 1
            
            # 2. Hitting time: переключение на следующий узел графа
            if reach >= self.hit_threshold and self.current_idx < len(self.waypoints) - 1:
                self.current_idx += 1
                self.steps_on_wp = 0
                curr_wp = self.waypoints[self.current_idx]
                z_jnp = jnp.asarray(curr_wp)[None, :] # Обновляем вектор для подруливания
                
            # 3. Fallback: если муравей застрял, перестраиваем весь маршрут
            elif self.steps_on_wp > 80 and self.current_idx < len(self.waypoints) - 1:
                self.reset(obs, goal_z)
                curr_wp = self.waypoints[self.current_idx]
                z_jnp = jnp.asarray(curr_wp)[None, :]

            # 4. ЛОКАЛЬНОЕ ПОДРУЛИВАНИЕ (Исправление ошибки):
            # Пропускаем статичный узел графа через high_actor для проекции на знакомое многообразие
            high_dist = self.agent.network.select("high_actor")(obs_jnp, z_jnp, goal_encoded=True, temperature=0.0)
            return np.asarray(self.agent.normalize_z(high_dist.mode()))[0]


class DistilledMLPPlanner(BasePlanner):
    def __init__(
        self,
        agent,
        model_type="dense_eca",
        checkpoint_path=None,
        hidden_dim=256,
        n_layers=3,
        num_heads=4,
        device="auto",
        name="Distilled Latent Policy",
    ):
        super().__init__(agent, name=name)
        obs_dim = 29
        latent_dim = agent.config["latent_dim"]
        if device == "auto":
            self.device = torch.device(
                "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
            )
        else:
            self.device = torch.device(device)

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

    def get_subgoal_latent(self, obs, goal_latent, step=0):
        x = np.concatenate([obs, np.asarray(goal_latent)], axis=-1).astype(np.float32)
        with torch.no_grad():
            inp = torch.from_numpy(x).unsqueeze(0).to(self.device)
            pred_z = self.model(inp).squeeze(0).cpu().numpy()
            
        # Local high_actor steering
        obs_jnp = jnp.asarray(obs)[None, :]
        z_jnp = jnp.asarray(pred_z)[None, :]
        high_dist = self.agent.network.select("high_actor")(obs_jnp, z_jnp, goal_encoded=True, temperature=0.0)
        return np.asarray(self.agent.normalize_z(high_dist.mode()))[0]
