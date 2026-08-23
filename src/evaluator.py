import time
import json
from collections import defaultdict
import numpy as np
import jax
from tqdm import tqdm
from utils.env_utils import relabel_dataset
from utils.datasets import Dataset, HGCDataset
from utils.evaluation import supply_rng, flatten


def _append_telemetry(traj_records, sg_records, planner_name, seed, task_id, ep, step, x, y, reward, done, sg_xy, is_direct, wps_json):
    traj_records.append({
        "method": planner_name, "seed": int(seed), "task_id": int(task_id),
        "episode": int(ep), "step": int(step), "x": float(x), "y": float(y),
        "reward": float(reward), "done": bool(done),
    })
    sg_records.append({
        "method": planner_name, "seed": int(seed), "task_id": int(task_id),
        "episode": int(ep), "step": int(step),
        "subgoal_x": float(sg_xy[0]) if sg_xy is not None else None,
        "subgoal_y": float(sg_xy[1]) if sg_xy is not None else None,
        "is_direct_goal": bool(is_direct), "planned_waypoints": wps_json,
    })


class ZeroShotEvaluator:
    """Deterministic evaluation engine with full trajectory & waypoint logging."""
    def __init__(self, env, agent, dataset_dict, config, env_name="ogbench-antmaze-medium-navigate-v0", max_episode_steps=None):
        self.env = env
        self.agent = agent
        self.dataset_dict = dataset_dict
        self.config = config
        self.env_name = env_name
        self.max_episode_steps = max_episode_steps
        self.inferred_latents_cache = {}

    def get_inferred_latent(self, task_id, seed=0):
        cache_key = (task_id, seed)
        if cache_key in self.inferred_latents_cache:
            return self.inferred_latents_cache[cache_key]

        task_seed = seed * 100000 + task_id * 1000
        self.env.action_space.seed(task_seed)
        np.random.seed(task_seed)
        self.env.reset(seed=task_seed, options=dict(task_id=task_id))

        zero_shot_ds = HGCDataset(Dataset.create(**relabel_dataset(self.env_name, self.env, self.dataset_dict)), self.config)
        rng = np.random.default_rng(task_seed)
        n_samples = min(50000, zero_shot_ds.size)
        idxs = rng.choice(zero_shot_ds.size, size=n_samples, replace=False)
        batch = zero_shot_ds.sample(n_samples, idxs=idxs, relabeling=False, augmentation=False)
        inferred = np.asarray(self.agent.infer_latent(batch))
        self.inferred_latents_cache[cache_key] = inferred
        return inferred

    def _rollout_episode(self, planner, task_id, ep, seed, max_steps, eval_temp, inferred_latent, traj_recs, sg_recs):
        ep_seed = seed * 100000 + task_id * 1000 + ep
        self.env.action_space.seed(ep_seed)
        np.random.seed(ep_seed)
        obs, info = self.env.reset(seed=ep_seed, options=dict(task_id=task_id))
        planner.reset(obs, inferred_latent)

        done, step, traj = False, 0, [obs[:2].copy()]
        seed_key = jax.random.PRNGKey(ep_seed)
        sg_info = planner.get_subgoal_info()
        wps_json = json.dumps(sg_info.get("waypoints_xy", []))
        _append_telemetry(traj_recs, sg_recs, planner.name, seed, task_id, ep, step, obs[0], obs[1], 0.0, False, sg_info.get("subgoal_xy"), sg_info.get("is_direct_goal", False), wps_json)

        t0 = time.perf_counter()
        while not done:
            action = planner.sample_action(obs, inferred_latent, step=step, seed=seed_key, temperature=eval_temp)
            sg_info = planner.get_subgoal_info()
            obs, reward, term, trunc, info = self.env.step(action)
            step += 1
            if max_steps is not None and step >= max_steps:
                trunc = True
            done = term or trunc
            traj.append(obs[:2].copy())
            _append_telemetry(traj_recs, sg_recs, planner.name, seed, task_id, ep, step, obs[0], obs[1], reward, done, sg_info.get("subgoal_xy"), sg_info.get("is_direct_goal", False), wps_json)

        latency = (time.perf_counter() - t0) * 1000.0 / max(step, 1)
        return info, np.asarray(traj), latency

    def evaluate_task(self, planner, task_id, num_episodes=15, eval_temperature=0.0, seed=0, max_episode_steps=None):
        max_steps = max_episode_steps or self.max_episode_steps
        latent = self.get_inferred_latent(task_id, seed=seed)
        stats, trajs, latencies, traj_recs, sg_recs = defaultdict(list), [], [], [], []

        for ep in range(num_episodes):
            info, traj, lat = self._rollout_episode(planner, task_id, ep, seed, max_steps, eval_temperature, latent, traj_recs, sg_recs)
            latencies.append(lat)
            trajs.append(traj)
            for k, v in flatten(info).items():
                stats[k].append(v)
            from tests.test_trajectories import count_self_intersections
            stats["self_intersections"].append(count_self_intersections(traj))

        mean_stats = {k: float(np.mean(v)) for k, v in stats.items()}
        mean_stats["latency_ms"] = float(np.mean(latencies)) if latencies else 0.0
        return mean_stats, trajs, traj_recs, sg_recs

    def evaluate_all_tasks(self, planner, num_episodes=15, eval_temperature=0.0, seed=0, max_episode_steps=None):
        num_tasks = len(self.env.unwrapped.task_infos if hasattr(self.env.unwrapped, "task_infos") else self.env.task_infos)
        metrics, all_trajs, traj_recs, sg_recs = defaultdict(list), [], [], []

        pbar = tqdm(range(1, num_tasks + 1), desc=f"Evaluating {planner.name} (Seed {seed})", leave=False)
        for t_id in pbar:
            task_stats, task_trajs, t_recs, s_recs = self.evaluate_task(planner, t_id, num_episodes, eval_temperature, seed, max_episode_steps)
            for k, v in task_stats.items():
                metrics[k].append(v)
            all_trajs.extend(task_trajs)
            traj_recs.extend(t_recs)
            sg_recs.extend(s_recs)
            pbar.set_postfix({"success": f"{np.mean(metrics.get('success', [0])):.2%}"})

        return {
            "success_rate": float(np.mean(metrics["success"]) * 100.0) if "success" in metrics else 0.0,
            "mean_length": float(np.mean(metrics["episode.length"])) if "episode.length" in metrics else 0.0,
            "latency_ms": float(np.mean(metrics["latency_ms"])) if "latency_ms" in metrics else 0.0,
            "self_intersections": float(np.mean(metrics["self_intersections"])) if "self_intersections" in metrics else 0.0,
            "task_successes": [float(s) for s in metrics["success"]],
            "trajectories": all_trajs,
            "trajectory_records": traj_recs,
            "subgoal_records": sg_recs,
        }
