"""Deterministic evaluation engine with SQLite telemetry logging."""

from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple
import jax
import numpy as np
from tqdm import tqdm

from utils.datasets import Dataset, HGCDataset
from utils.env_utils import relabel_dataset
from utils.evaluation import flatten

from src.stages import StepDatabaseLogger
from src.telemetry.db import TelemetryDatabase
from src.telemetry.metrics import count_spatial_self_intersections
from src.telemetry.profiler import ExecutionProfiler


def _init_episode_record(
    db: Optional[TelemetryDatabase],
    ep_id: str,
    run_id: str,
    seed: int,
    task_id: int,
    ep: int,
    obs: np.ndarray,
    goal_xy: Optional[Tuple[float, float]] = None,
) -> None:
    """Initialize episode header row to satisfy foreign key constraints for steps."""
    if db is None:
        return
    gx = float(goal_xy[0]) if goal_xy is not None else None
    gy = float(goal_xy[1]) if goal_xy is not None else None
    db.insert_episode(
        episode_id=ep_id,
        run_id=run_id,
        seed=seed,
        task_id=task_id,
        episode_idx=ep,
        start_x=float(obs[0]),
        start_y=float(obs[1]),
        goal_x=gx,
        goal_y=gy,
        is_success=False,
        total_steps=0,
        total_reward=0.0,
        mean_speed=0.0,
        self_intersections=0,
        mean_cross_track_error=0.0,
        max_cross_track_error=0.0,
        mean_latency_ms=0.0,
    )


class ZeroShotEvaluator:
    """Deterministic evaluation engine with full trajectory and database logging."""

    def __init__(
        self,
        env: Any,
        agent: Any,
        dataset_dict: Dict[str, Any],
        config: Any,
        env_name: str = "ogbench-antmaze-medium-navigate-v0",
        max_episode_steps: Optional[int] = None,
        db: Optional[TelemetryDatabase] = None,
        run_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize evaluation environment, agent, dataset, and storage backends."""
        self.env = env
        self.agent = agent
        self.dataset_dict = dataset_dict
        self.config = config
        self.env_name = env_name
        self.max_episode_steps = max_episode_steps
        self.db = db
        self.run_id = run_id or "default_run"
        self.step_logger = (
            StepDatabaseLogger(db_getter=lambda: self.db)
            if db is not None
            else None
        )
        self.inferred_latents_cache: Dict[Tuple[int, int], np.ndarray] = {}

    def get_inferred_latent(self, task_id: int, seed: int = 0) -> np.ndarray:
        """Infer goal latent representation for specified task seed with caching."""
        cache_key = (task_id, seed)
        if cache_key in self.inferred_latents_cache:
            return self.inferred_latents_cache[cache_key]

        task_seed = seed * 100000 + task_id * 1000
        self.env.action_space.seed(task_seed)
        np.random.seed(task_seed)
        self.env.reset(seed=task_seed, options=dict(task_id=task_id))

        zero_shot_ds = HGCDataset(
            Dataset.create(**relabel_dataset(self.env_name, self.env, self.dataset_dict)),
            self.config,
        )
        rng = np.random.default_rng(task_seed)
        n_samples = min(50000, zero_shot_ds.size)
        idxs = rng.choice(zero_shot_ds.size, size=n_samples, replace=False)
        batch = zero_shot_ds.sample(
            n_samples, idxs=idxs, relabeling=False, augmentation=False
        )
        inferred = np.asarray(self.agent.infer_latent(batch))
        self.inferred_latents_cache[cache_key] = inferred
        return inferred

    def _execute_episode_loop(
        self,
        planner: Any,
        inferred_latent: np.ndarray,
        ep_seed: int,
        max_steps: Optional[int],
        eval_temp: float,
        obs: np.ndarray,
    ) -> Tuple[Any, np.ndarray, float]:
        """Run interaction loop for an episode and record profiling telemetry."""
        done, step, traj = False, 0, [obs[:2].copy()]
        seed_key = jax.random.PRNGKey(ep_seed)
        profiler = ExecutionProfiler()
        info: Dict[str, Any] = {}

        while not done:
            with profiler:
                action = planner.sample_action(
                    obs, inferred_latent, step=step, seed=seed_key, temperature=eval_temp
                )
                profiler.sync(action)

            obs, reward, term, trunc, info = self.env.step(action)
            step += 1
            if max_steps is not None and step >= max_steps:
                trunc = True
            done = bool(term or trunc)
            traj.append(obs[:2].copy())

            if self.step_logger is not None:
                planner.ctx.reward = float(reward)
                planner.ctx.done = done
                planner.ctx.latency_ms = float(profiler.last_elapsed_ms)
                self.step_logger(planner.ctx)

        return info, np.asarray(traj), profiler.mean_ms

    def _rollout_episode(
        self,
        planner: Any,
        task_id: int,
        ep: int,
        seed: int,
        max_steps: Optional[int],
        eval_temp: float,
        inferred_latent: np.ndarray,
        run_id: Optional[str] = None,
    ) -> Tuple[Any, np.ndarray, float]:
        """Reset environment and execute rollout with database persistence."""
        active_run = run_id or self.run_id
        ep_id = f"{active_run}_{task_id}_{seed}_{ep}"
        ep_seed = seed * 100000 + task_id * 1000 + ep
        self.env.action_space.seed(ep_seed)
        np.random.seed(ep_seed)
        obs, _ = self.env.reset(seed=ep_seed, options=dict(task_id=task_id))
        task_infos = getattr(self.env.unwrapped, "task_infos", getattr(self.env, "task_infos", []))
        goal_xy = None
        if task_infos and 0 <= task_id - 1 < len(task_infos):
            t_info = task_infos[task_id - 1]
            if "goal_xy" in t_info:
                goal_xy = (float(t_info["goal_xy"][0]), float(t_info["goal_xy"][1]))
            elif "target_xy" in t_info:
                goal_xy = (float(t_info["target_xy"][0]), float(t_info["target_xy"][1]))
        if goal_xy is None and hasattr(self.env, "target_xy"):
            goal_xy = (float(self.env.target_xy[0]), float(self.env.target_xy[1]))

        _init_episode_record(self.db, ep_id, active_run, seed, task_id, ep, obs, goal_xy=goal_xy)
        planner.reset(obs, inferred_latent, db=self.db, episode_id=ep_id)
        if self.db is not None:
            planner.ctx.db = self.db
            planner.ctx.episode_id = ep_id

        info, traj, mean_lat = self._execute_episode_loop(
            planner, inferred_latent, ep_seed, max_steps, eval_temp, obs
        )

        if self.db is not None:
            final_xy = traj[-1] if len(traj) > 0 else obs[:2]
            ep_metrics = self.db.finalize_episode(ep_id, final_obs_xy=final_xy)
            if ep_metrics:
                mean_lat = float(ep_metrics.get("mean_latency_ms", mean_lat))

        return info, traj, mean_lat

    def evaluate_task(
        self,
        planner: Any,
        task_id: int,
        num_episodes: int = 15,
        eval_temperature: float = 0.0,
        seed: int = 0,
        max_episode_steps: Optional[int] = None,
        run_id: Optional[str] = None,
    ) -> Tuple[Dict[str, float], List[np.ndarray], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Evaluate a single task across multiple episodes and aggregate statistics."""
        max_steps = max_episode_steps or self.max_episode_steps
        latent = self.get_inferred_latent(task_id, seed=seed)
        stats, trajs, latencies = defaultdict(list), [], []

        for ep in range(num_episodes):
            info, traj, lat = self._rollout_episode(
                planner, task_id, ep, seed, max_steps, eval_temperature, latent,
                run_id=run_id
            )
            latencies.append(lat)
            trajs.append(traj)
            for k, v in flatten(info).items():
                stats[k].append(v)
            stats["self_intersections"].append(count_spatial_self_intersections(traj))

        mean_stats = {k: float(np.mean(v)) for k, v in stats.items()}
        mean_stats["latency_ms"] = float(np.mean(latencies)) if latencies else 0.0
        return mean_stats, trajs

    def evaluate_all_tasks(
        self,
        planner: Any,
        num_episodes: int = 15,
        eval_temperature: float = 0.0,
        seed: int = 0,
        max_episode_steps: Optional[int] = None,
        run_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Evaluate agent across all maze tasks and summarize performance."""
        task_infos = getattr(self.env.unwrapped, "task_infos", getattr(self.env, "task_infos", []))
        num_tasks = len(task_infos)
        metrics, all_trajs = defaultdict(list), []

        pbar = tqdm(
            range(1, num_tasks + 1),
            desc=f"Evaluating {planner.name} (Seed {seed})",
            leave=False,
        )
        for t_id in pbar:
            task_stats, task_trajs = self.evaluate_task(
                planner, t_id, num_episodes, eval_temperature, seed,
                max_episode_steps, run_id=run_id
            )
            for k, v in task_stats.items():
                metrics[k].append(v)
            all_trajs.extend(task_trajs)
            pbar.set_postfix({"success": f"{np.mean(metrics.get('success', [0])):.2%}"})

        return {
            "success_rate": float(np.mean(metrics["success"]) * 100.0) if "success" in metrics else 0.0,
            "mean_length": float(np.mean(metrics["episode.length"])) if "episode.length" in metrics else 0.0,
            "latency_ms": float(np.mean(metrics["latency_ms"])) if "latency_ms" in metrics else 0.0,
            "self_intersections": float(np.mean(metrics["self_intersections"])) if "self_intersections" in metrics else 0.0,
            "task_successes": [float(s) for s in metrics["success"]],
            "trajectories": all_trajs,
            "trajectory_records": [],
            "subgoal_records": [],
        }
