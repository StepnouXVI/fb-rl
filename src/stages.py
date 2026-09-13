"""Pipeline stages for modular staged agents in maze navigation tasks."""

import functools
import os
import pickle
from typing import Any, List, Optional
import flax
import jax
import jax.numpy as jnp
import numpy as np
from scipy.sparse.csgraph import dijkstra

from src.contexts import (
    BaseMazeContext,
    SequenceAttentionContext,
    TopologicalPathContext,
)
from src.topology import (
    backtrack_dijkstra_path,
    check_stuck_state,
    compute_continuous_lookahead,
    extract_corner_waypoints,
    extract_curvature_angles,
    find_connected_landmarks,
    jit_batch_reach,
    jit_decode_latent_to_coords,
    track_local_path_index,
)
from src.networks import (
    SequenceAttentionNetwork,
    SingleWaypointNetwork,
    build_direct_translator,
)


class PipelineStage:
    """Base interface for agent pipeline stages."""

    def reset(self, ctx: BaseMazeContext) -> None:
        """Reset internal stage state for a new episode."""
        pass

    def __call__(self, ctx: BaseMazeContext) -> None:
        """Process context and mutate its fields."""
        raise NotImplementedError


class DijkstraPathBuilder(PipelineStage):
    """Constructs offline Dijkstra paths and tracks agent progress with replanning."""

    def __init__(
        self,
        agent: Any,
        dataset_states: np.ndarray,
        n_landmarks: int = 1000,
        max_edge_radius: float = 3.5,
        reachability_cutoff: float = 35.0,
        lookahead_dist: float = 2.6,
    ) -> None:
        """Initialize Dijkstra landmark graph and distance matrices."""
        self.agent = agent
        self.n_landmarks = min(n_landmarks, len(dataset_states))
        self.max_edge_radius = max_edge_radius
        self.reachability_cutoff = reachability_cutoff
        self.lookahead_dist = lookahead_dist

        idxs = np.random.default_rng(42).choice(
            len(dataset_states), size=self.n_landmarks, replace=False
        )
        self.landmarks = jnp.asarray(dataset_states[idxs])
        self.landmark_coords = np.asarray(dataset_states[idxs][:, :2])
        self.landmark_latents = jnp.asarray(
            agent.normalize_z(
                agent.network.select("backward_repr")(self.landmarks)
            )
        )
        self.cost_matrix, self.reach_matrix = self._build_graph()
        self.all_dist, self.all_pred = dijkstra(
            self.cost_matrix, directed=True, return_predecessors=True
        )
        self.pos_history: List[np.ndarray] = []
        self.stuck_count: int = 0

    def _build_graph(self) -> Any:
        """Build cost and reachability matrices between landmarks."""
        n = len(self.landmarks)
        s_rep = jnp.repeat(self.landmarks, n, axis=0)
        z_tile = jnp.tile(self.landmark_latents, (n, 1))
        reach_flat = [
            np.asarray(
                jit_batch_reach(self.agent, s_rep[i : i + 45000], z_tile[i : i + 45000])
            )
            for i in range(0, len(s_rep), 45000)
        ]
        reach_m = np.concatenate(reach_flat, axis=0).reshape((n, n))
        dists_e = np.linalg.norm(
            self.landmark_coords[:, None, :] - self.landmark_coords[None, :, :], axis=-1
        )
        max_diag = float(np.max(np.diag(reach_m))) if np.max(np.diag(reach_m)) > 0 else 1.0
        cost_m = np.maximum(0.0, -np.log(np.clip(reach_m / max_diag, 1e-6, 1.0)))
        cost_m[dists_e > self.max_edge_radius] = np.inf
        cost_m[reach_m < self.reachability_cutoff] = np.inf
        np.fill_diagonal(cost_m, 0.0)
        return cost_m, reach_m

    def _update_lookahead(self, ctx: TopologicalPathContext) -> None:
        """Calculate lookahead waypoint index and target latent vector."""
        accum, target_idx = 0.0, ctx.current_path_idx
        while target_idx < len(ctx.path_coords) - 1 and accum < self.lookahead_dist:
            accum += float(
                np.linalg.norm(ctx.path_coords[target_idx + 1] - ctx.path_coords[target_idx])
            )
            target_idx += 1
        dist_final = float(np.linalg.norm(ctx.pos_xy - ctx.path_coords[-1]))
        is_goal = bool(target_idx >= len(ctx.path_coords) - 1 or dist_final <= 2.2)
        ctx.target_latent = ctx.goal_latent if is_goal else ctx.path_latents[target_idx]
        ctx.lookahead_xy = ctx.path_coords[-1] if is_goal else ctx.path_coords[target_idx]
        ctx.dist_to_lookahead = float(np.linalg.norm(ctx.pos_xy - ctx.lookahead_xy))
        ctx.stuck_count = self.stuck_count
        ctx.is_direct_goal = is_goal

    def reset(self, ctx: BaseMazeContext) -> None:
        """Find landmarks, backtrack Dijkstra path, and reset progress tracking."""
        if not isinstance(ctx, TopologicalPathContext):
            return
        if ctx.pos_xy is None or ctx.goal_latent is None:
            return
        s_idx, g_idx = find_connected_landmarks(
            self.landmark_coords, self.landmark_latents, self.all_dist, ctx.pos_xy, ctx.goal_latent
        )
        path = backtrack_dijkstra_path(self.all_pred, s_idx, g_idx, self.n_landmarks)
        ctx.path_coords = [self.landmark_coords[i] for i in path]
        ctx.path_latents = [self.landmark_latents[i] for i in path]
        ctx.path_states = [np.asarray(self.landmarks[i]) for i in path]
        ctx.waypoint_coords, _, filtered = extract_corner_waypoints(
            ctx.path_coords, ctx.path_latents, self.lookahead_dist
        )
        ctx.waypoint_indices = filtered
        ctx.path_dists = [0.0]
        for i in range(len(ctx.path_coords) - 1):
            seg_len = float(np.linalg.norm(ctx.path_coords[i + 1] - ctx.path_coords[i]))
            ctx.path_dists.append(ctx.path_dists[-1] + seg_len)
        self.pos_history = []
        self.stuck_count = 0
        ctx.current_path_idx = 0
        ctx.replan_triggered = True
        self._update_lookahead(ctx)

    def __call__(self, ctx: BaseMazeContext) -> None:
        """Track agent position along the path and trigger replan when deviated or stuck."""
        if not isinstance(ctx, TopologicalPathContext):
            return
        if not ctx.path_coords:
            self.reset(ctx)
            return
        best_idx, min_dist = track_local_path_index(
            ctx.pos_xy, ctx.path_coords, ctx.current_path_idx, max_window=7
        )
        ctx.replan_triggered = False
        if min_dist > 4.8:
            self.reset(ctx)
            ctx.replan_triggered = True
            return
        ctx.current_path_idx = best_idx
        dist_final = float(np.linalg.norm(ctx.pos_xy - ctx.path_coords[-1]))
        is_stuck, self.stuck_count = check_stuck_state(
            self.pos_history, ctx.pos_xy, dist_final, self.stuck_count
        )
        if self.stuck_count > 45:
            self.reset(ctx)
            self.stuck_count = 0
            ctx.replan_triggered = True
            return
        self._update_lookahead(ctx)


class PathDatabaseLogger(PipelineStage):
    """Writes planned topological paths to SQLite database upon replan triggers."""

    def __init__(self, db_getter: Optional[Any] = None) -> None:
        """Initialize with optional database getter."""
        self.db_getter = db_getter
        self._has_logged_initial: bool = False

    def _resolve_db(self, ctx: BaseMazeContext) -> Optional[Any]:
        """Resolve database from context or db_getter."""
        if hasattr(ctx, "db") and ctx.db is not None:
            return ctx.db
        if callable(self.db_getter):
            return self.db_getter()
        return self.db_getter

    def reset(self, ctx: BaseMazeContext) -> None:
        """Reset logged state and attempt to log initial path."""
        self._has_logged_initial = False
        if self._maybe_log_path(ctx, is_initial=True):
            self._has_logged_initial = True

    def __call__(self, ctx: BaseMazeContext) -> None:
        """Log planned path if initial was missed or replan triggered."""
        if not self._has_logged_initial:
            if self._maybe_log_path(ctx, is_initial=True):
                self._has_logged_initial = True
                return
        if getattr(ctx, "replan_triggered", False):
            self._maybe_log_path(ctx, is_initial=False)

    def _maybe_log_path(self, ctx: BaseMazeContext, is_initial: bool = False) -> bool:
        """Write path coordinates and waypoints into SQLite if available."""
        if not isinstance(ctx, TopologicalPathContext):
            return False
        db = self._resolve_db(ctx)
        ep_id = getattr(ctx, "episode_id", None)
        if db is None or ep_id is None:
            return False
        wps = [
            [float(c[0]), float(c[1])]
            for c in getattr(ctx, "waypoint_coords", [])
        ]
        coords = [
            [float(c[0]), float(c[1])]
            for c in getattr(ctx, "path_coords", [])
        ]
        if coords or wps:
            db.insert_planned_path(
                episode_id=ep_id,
                replan_step=int(getattr(ctx, "step", 0)),
                is_initial=is_initial,
                waypoints=wps,
                path_coords=coords,
            )
            return True
        return False


class StepDatabaseLogger(PipelineStage):
    """Writes step telemetry from context directly to SQLite database."""

    def __init__(self, db_getter: Optional[Any] = None) -> None:
        """Initialize with optional database getter."""
        self.db_getter = db_getter

    def _resolve_db(self, ctx: BaseMazeContext) -> Optional[Any]:
        """Resolve database instance from context or getter."""
        if hasattr(ctx, "db") and ctx.db is not None:
            return ctx.db
        if callable(self.db_getter):
            return self.db_getter()
        return self.db_getter

    def __call__(self, ctx: BaseMazeContext) -> None:
        """Extract step telemetry from context and insert into database."""
        db = self._resolve_db(ctx)
        ep_id = getattr(ctx, "episode_id", None)
        if db is None or ep_id is None:
            return
        obs = ctx.obs
        pos_xy = ctx.pos_xy
        x = float(pos_xy[0]) if pos_xy is not None else 0.0
        y = float(pos_xy[1]) if pos_xy is not None else 0.0
        z = float(obs[2]) if obs is not None and len(obs) > 2 else 0.0
        vx = float(obs[15]) if obs is not None and len(obs) > 17 else 0.0
        vy = float(obs[16]) if obs is not None and len(obs) > 17 else 0.0
        speed = float(np.hypot(vx, vy))
        action = ctx.action
        act_norm = float(np.linalg.norm(action)) if action is not None else 0.0
        torques = [float(a) for a in action] if action is not None else []
        lh_xy = getattr(ctx, "lookahead_xy", None)
        lh_x = float(lh_xy[0]) if lh_xy is not None else None
        lh_y = float(lh_xy[1]) if lh_xy is not None else None
        dist_lh = (
            float(np.hypot(x - lh_x, y - lh_y))
            if (lh_x is not None and lh_y is not None)
            else None
        )
        targets = getattr(ctx, "attention_targets", [])
        weights = getattr(ctx, "attention_weights", [])
        targets_json = [[float(t[0]), float(t[1])] for t in targets] if targets else []
        weights_json = [float(w) for w in weights] if weights else []
        step_idx = int(getattr(ctx, "step", 0))
        step_data = {
            "step_id": f"{ep_id}_{step_idx}",
            "episode_id": ep_id,
            "step_idx": step_idx,
            "x": x,
            "y": y,
            "z": z,
            "vx": vx,
            "vy": vy,
            "speed": speed,
            "action_norm": act_norm,
            "action_torques": torques,
            "lookahead_x": lh_x,
            "lookahead_y": lh_y,
            "dist_to_lookahead": dist_lh,
            "attention_targets": targets_json,
            "attention_weights": weights_json,
            "is_direct_goal": int(bool(getattr(ctx, "is_direct_goal", False))),
            "reward": float(getattr(ctx, "reward", 0.0)),
            "done": int(bool(getattr(ctx, "done", False))),
            "latency_ms": float(getattr(ctx, "latency_ms", 0.0)),
        }
        db.insert_steps_batch([step_data], episode_id=ep_id)


class SubgoalSelector(PipelineStage):
    """Selects continuous lookahead point, downstream landmarks, and goal target for attention."""

    def __init__(
        self,
        lookahead_dist: float = 2.6,
        num_landmarks: int = 4,
    ) -> None:
        """Configure continuous lookahead distance and landmark selection count."""
        self.lookahead_dist = lookahead_dist
        self.num_landmarks = num_landmarks
        self.current_seg: int = 0

    def reset(self, ctx: BaseMazeContext) -> None:
        """Reset segment tracking pointer."""
        self.current_seg = 0

    def __call__(self, ctx: BaseMazeContext) -> None:
        """Select lookahead coordinate, salient forward landmarks, and final goal target."""
        if not isinstance(ctx, TopologicalPathContext) or not ctx.path_coords:
            return
        if ctx.replan_triggered:
            self.current_seg = 0
        curr_c, self.current_seg, s_agent = compute_continuous_lookahead(
            ctx.pos_xy,
            ctx.path_coords,
            self.lookahead_dist,
            self.current_seg,
            path_dists=getattr(ctx, "path_dists", None),
        )
        dist_final = float(np.linalg.norm(ctx.pos_xy - ctx.path_coords[-1]))
        is_direct = bool(dist_final < 1.8)
        if is_direct:
            curr_c = np.array(ctx.path_coords[-1])
        ctx.lookahead_xy = curr_c
        ctx.dist_to_lookahead = float(np.linalg.norm(ctx.pos_xy - curr_c))

        path_dists = getattr(ctx, "path_dists", None)
        wp_indices = getattr(ctx, "waypoint_indices", None)
        downstream_wps = []
        downstream_latents = []
        for k, w in enumerate(ctx.waypoint_coords):
            s_wp = path_dists[wp_indices[k]] if (path_dists and wp_indices and k < len(wp_indices)) else float(k)
            if s_wp > s_agent + 0.8 and np.linalg.norm(np.array(w) - curr_c) > 0.8:
                downstream_wps.append(w)
                if wp_indices and k < len(wp_indices) and wp_indices[k] < len(ctx.path_latents):
                    downstream_latents.append(ctx.path_latents[wp_indices[k]])
                elif k < len(ctx.path_latents):
                    downstream_latents.append(ctx.path_latents[k])
                else:
                    downstream_latents.append(ctx.goal_latent)

        if is_direct:
            targets = [curr_c]
            latents = [ctx.goal_latent]
        else:
            lh_latent = ctx.target_latent if ctx.target_latent is not None else ctx.goal_latent
            targets = [curr_c] + downstream_wps[: self.num_landmarks]
            latents = [lh_latent] + downstream_latents[: self.num_landmarks]
            if len(targets) > 0 and np.linalg.norm(np.array(targets[-1]) - np.array(ctx.path_coords[-1])) > 0.5:
                targets.append(ctx.path_coords[-1])
                latents.append(ctx.goal_latent)

        ctx.attention_targets = targets
        if hasattr(ctx, "attention_latents"):
            ctx.attention_latents = latents


class SequenceAttentionTranslator(PipelineStage):
    """Runs inference for sequence attention transformer model."""

    def __init__(
        self,
        model: Optional[SequenceAttentionNetwork] = None,
        params: Optional[Any] = None,
        checkpoint_path: Optional[str] = None,
        latent_dim: int = 128,
        hidden_dim: int = 384,
        num_heads: int = 6,
        max_seq_len: int = 16,
        n_layers: int = 3,
        dropout_rate: float = 0.2,
        alibi_slope: float = 0.4,
    ) -> None:
        """Initialize or load weights for the sequence attention translator."""
        ckpt_params, cfg = self._load_checkpoint(checkpoint_path)
        params = params or ckpt_params
        h_dim = cfg.get("hidden_dim", hidden_dim)
        n_heads = cfg.get("num_heads", num_heads)
        n_lay = cfg.get("n_layers", n_layers)
        m_len = cfg.get("max_seq_len", max_seq_len)
        d_rate = cfg.get("dropout_rate", dropout_rate)
        a_slope = cfg.get("alibi_slope", alibi_slope)

        self.model = model or SequenceAttentionNetwork(
            latent_dim=latent_dim,
            hidden_dim=h_dim,
            num_heads=n_heads,
            max_seq_len=m_len,
            n_layers=n_lay,
            dropout_rate=d_rate,
            alibi_slope=a_slope,
        )
        self.params = self._resolve_params(params, m_len, latent_dim)

        @jax.jit
        def _infer_jit(obs_b, seq_b, mask_b, curv_b, p):
            return self.model.apply(
                {"params": p},
                obs_b,
                seq_b,
                mask_b,
                curv_b,
                deterministic=True,
                return_attention=True,
            )

        self._infer_jit = _infer_jit

    def _load_checkpoint(self, checkpoint_path: Optional[str]) -> Any:
        """Load parameters and config dictionary from disk."""
        if checkpoint_path and os.path.exists(checkpoint_path):
            with open(checkpoint_path, "rb") as f:
                data = pickle.load(f)
                return flax.core.freeze(data["params"]), data.get("config", {})
        return None, {}

    def _resolve_params(
        self, params: Optional[Any], max_seq_len: int, latent_dim: int
    ) -> Any:
        """Return frozen params or initialize random weights."""
        if params is not None:
            return flax.core.freeze(params)
        dummy_s = jnp.zeros((1, 29))
        dummy_seq = jnp.zeros((1, max_seq_len, latent_dim))
        dummy_c = jnp.ones((1, max_seq_len, 1), dtype=jnp.float32)
        init_rng = jax.random.PRNGKey(0)
        p = self.model.init(
            {"params": init_rng, "dropout": init_rng},
            dummy_s,
            dummy_seq,
            None,
            dummy_c,
        )["params"]
        return flax.core.freeze(p)

    def __call__(self, ctx: BaseMazeContext) -> None:
        """Execute forward pass of attention transformer, assigning z_cmd and real attention weights."""
        if not isinstance(ctx, SequenceAttentionContext):
            return
        latents = getattr(ctx, "attention_latents", None)
        targets = getattr(ctx, "attention_targets", None)
        if latents and len(latents) > 0:
            wp_seq = np.asarray(latents, dtype=np.float32)[None, :, :]
            curvs = extract_curvature_angles(targets) if targets else []
            curv_arr = np.asarray(curvs, dtype=np.float32)[None, :, None] if curvs else None
            z_cmd, attn_weights = self._infer_jit(
                jnp.asarray(ctx.obs)[None, :],
                jnp.asarray(wp_seq),
                None,
                jnp.asarray(curv_arr) if curv_arr is not None else None,
                self.params,
            )
            ctx.z_cmd = np.asarray(z_cmd[0])
            ctx.attention_weights = np.asarray(attn_weights[0]).tolist()
        elif ctx.pad_seq is not None:
            z_cmd, attn_weights = self._infer_jit(
                jnp.asarray(ctx.obs)[None, :],
                jnp.asarray(ctx.pad_seq)[None, :, :],
                jnp.asarray(ctx.seq_mask)[None, :] if ctx.seq_mask is not None else None,
                jnp.asarray(ctx.curv_arr)[None, :, :] if ctx.curv_arr is not None else None,
                self.params,
            )
            ctx.z_cmd = np.asarray(z_cmd[0])
            ctx.attention_weights = np.asarray(attn_weights[0]).tolist()


class SingleWaypointTranslator(PipelineStage):
    """Inference for single waypoint neural translator."""

    def __init__(
        self,
        model: Optional[SingleWaypointNetwork] = None,
        params: Optional[Any] = None,
        checkpoint_path: Optional[str] = None,
        latent_dim: int = 128,
        hidden_dim: int = 256,
        n_layers: int = 3,
    ) -> None:
        """Initialize or load weights for the single waypoint translator."""
        if checkpoint_path and os.path.exists(checkpoint_path):
            with open(checkpoint_path, "rb") as f:
                data = pickle.load(f)
                params = flax.core.freeze(data["params"])
                cfg = data.get("config", {})
                hidden_dim = cfg.get("hidden_dim", hidden_dim)
                n_layers = cfg.get("n_layers", n_layers)

        self.model = model or SingleWaypointNetwork(
            latent_dim=latent_dim, hidden_dim=hidden_dim, n_layers=n_layers
        )
        if params is None:
            init_rng = jax.random.PRNGKey(0)
            params = self.model.init(
                init_rng, jnp.zeros((1, 29)), jnp.zeros((1, latent_dim))
            )["params"]
        self.params = flax.core.freeze(params)

        @jax.jit
        def _infer_jit(obs_b, wp_b, p):
            return self.model.apply({"params": p}, obs_b, wp_b)[0]

        self._infer_jit = _infer_jit

    def __call__(self, ctx: BaseMazeContext) -> None:
        """Execute single-waypoint translator to generate command intention."""
        target_z = getattr(ctx, "target_latent", None)
        if target_z is None:
            target_z = ctx.goal_latent
        z_cmd = self._infer_jit(
            jnp.asarray(ctx.obs)[None, :], jnp.asarray(target_z)[None, :], self.params
        )
        ctx.z_cmd = np.asarray(z_cmd)


class DirectIntentionTranslator(PipelineStage):
    """Direct amortized intention planner mapping state and goal directly to z_cmd."""

    def __init__(
        self,
        model: Optional[Any] = None,
        params: Optional[Any] = None,
        checkpoint_path: Optional[str] = None,
        dataset_states: Optional[np.ndarray] = None,
        agent: Optional[Any] = None,
        model_type: str = "gated_attn",
        latent_dim: int = 128,
        hidden_dim: int = 256,
        n_layers: int = 3,
    ) -> None:
        """Initialize direct intention model or load parameters from checkpoint."""
        if checkpoint_path and os.path.exists(checkpoint_path):
            with open(checkpoint_path, "rb") as f:
                params = flax.core.freeze(pickle.load(f)["params"])

        self.model = model or build_direct_translator(
            model_type=model_type,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
        )
        if params is None:
            init_rng = jax.random.PRNGKey(0)
            params = self.model.init(
                init_rng, jnp.zeros((1, 29)), jnp.zeros((1, latent_dim))
            )["params"]
        self.params = flax.core.freeze(params)

        if dataset_states is not None and agent is not None:
            n_samples = min(500, len(dataset_states))
            idxs = np.random.default_rng(42).choice(len(dataset_states), size=n_samples, replace=False)
            self.ref_coords = np.asarray(dataset_states[idxs][:, :2])
            self.ref_latents = jnp.asarray(
                agent.normalize_z(agent.network.select("backward_repr")(jnp.asarray(dataset_states[idxs])))
            )
        else:
            self.ref_coords, self.ref_latents = None, None

        @jax.jit
        def _infer_jit(obs_b, goal_b, p):
            return self.model.apply({"params": p}, obs_b, goal_b)[0]

        self._infer_jit = _infer_jit

    def __call__(self, ctx: BaseMazeContext) -> None:
        """Directly map current observation and goal latent to intention command."""
        z_cmd = self._infer_jit(
            jnp.asarray(ctx.obs)[None, :],
            jnp.asarray(ctx.goal_latent)[None, :],
            self.params,
        )
        ctx.z_cmd = np.asarray(z_cmd)
        if self.ref_latents is not None:
            decoded_idx = int(jit_decode_latent_to_coords(z_cmd, self.ref_latents))
            ctx.lookahead_xy = self.ref_coords[decoded_idx]


class HighLevelActor(PipelineStage):
    """High-level actor stage extracting subgoal latents from offline representations."""

    def __init__(self, agent: Any, dataset_states: Optional[np.ndarray] = None) -> None:
        """Store agent reference and optional reference landmarks for subgoal decoding."""
        self.agent = agent
        if dataset_states is not None:
            n_samples = min(500, len(dataset_states))
            idxs = np.random.default_rng(42).choice(
                len(dataset_states), size=n_samples, replace=False
            )
            self.ref_coords = np.asarray(dataset_states[idxs][:, :2])
            self.ref_latents = jnp.asarray(
                agent.normalize_z(
                    agent.network.select("backward_repr")(jnp.asarray(dataset_states[idxs]))
                )
            )
        else:
            self.ref_coords, self.ref_latents = None, None

        @jax.jit
        def _infer_high_jit(obs_b, tgt_b):
            high_dist = agent.network.select("high_actor")(
                obs_b, tgt_b, goal_encoded=True, temperature=0.0
            )
            return agent.normalize_z(high_dist.mode())[0]

        self._infer_high_jit = _infer_high_jit

    def __call__(self, ctx: BaseMazeContext) -> None:
        """Query high actor for intermediate subgoal intention."""
        target_z = getattr(ctx, "target_latent", None)
        if target_z is None:
            target_z = ctx.goal_latent
        subgoal_z = self._infer_high_jit(
            jnp.asarray(ctx.obs)[None, :], jnp.asarray(target_z)[None, :]
        )
        ctx.z_cmd = np.asarray(subgoal_z)
        if self.ref_latents is not None:
            decoded_idx = int(
                jit_decode_latent_to_coords(subgoal_z, self.ref_latents)
            )
            ctx.lookahead_xy = np.asarray(self.ref_coords[decoded_idx])


class LowLevelActor(PipelineStage):
    """Low-level actor policy executing actions conditioned on command intention."""

    def __init__(self, agent: Any) -> None:
        """Store agent reference and compile actor step."""
        self.agent = agent

        @functools.partial(jax.jit, static_argnames=("temp",))
        def _infer_act_jit(obs_b, z_b, seed_k=None, temp=0.0):
            act_dist = agent.network.select("actor")(
                obs_b, z_b, goal_encoded=True, temperature=temp
            )
            act = (
                act_dist.mode()
                if (temp == 0.0 or seed_k is None)
                else act_dist.sample(seed=seed_k)
            )
            return jnp.clip(act[0], -1.0, 1.0)

        self._infer_act_jit = _infer_act_jit

    def __call__(self, ctx: BaseMazeContext) -> None:
        """Sample action from low-level policy given state and commanded latent."""
        z = ctx.z_cmd if ctx.z_cmd is not None else ctx.goal_latent
        temp = ctx.temperature
        seed_k = (
            jax.random.PRNGKey(ctx.step)
            if (temp > 0.0 and ctx.seed is None)
            else ctx.seed
        )
        act = self._infer_act_jit(
            jnp.asarray(ctx.obs)[None, :],
            jnp.asarray(z)[None, :],
            seed_k=seed_k,
            temp=temp,
        )
        ctx.action = np.asarray(act)
