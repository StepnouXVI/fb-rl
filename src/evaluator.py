import time
import json
from collections import defaultdict
import numpy as np
import jax
from tqdm import tqdm
from utils.env_utils import relabel_dataset
from utils.datasets import Dataset, HGCDataset
from utils.evaluation import supply_rng, flatten

# ponytail: High-performance deterministic evaluation engine with full trajectory & waypoint logging
class ZeroShotEvaluator:
    def __init__(self, env, agent, dataset_dict, config, env_name="ogbench-antmaze-medium-navigate-v0"):
        self.env = env
        self.agent = agent
        self.dataset_dict = dataset_dict
        self.config = config
        self.env_name = env_name
        self.inferred_latents_cache = {}

    def get_inferred_latent(self, task_id, seed=0):
        cache_key = (task_id, seed)
        if cache_key in self.inferred_latents_cache:
            return self.inferred_latents_cache[cache_key]

        task_seed = seed * 100000 + task_id * 1000
        self.env.action_space.seed(task_seed)
        np.random.seed(task_seed)
        self.env.reset(seed=task_seed, options=dict(task_id=task_id))

        zero_shot_ds = relabel_dataset(self.env_name, self.env, self.dataset_dict)
        zero_shot_ds = HGCDataset(Dataset.create(**zero_shot_ds), self.config)

        rng = np.random.default_rng(task_seed)
        n_samples = min(50000, zero_shot_ds.size)
        idxs = rng.choice(zero_shot_ds.size, size=n_samples, replace=False)
        zero_shot_batch = zero_shot_ds.sample(n_samples, idxs=idxs, relabeling=False, augmentation=False)
        inferred_latent = np.asarray(self.agent.infer_latent(zero_shot_batch))

        self.inferred_latents_cache[cache_key] = inferred_latent
        return inferred_latent

    def evaluate_task(self, planner, task_id, num_episodes=15, eval_temperature=0.0, seed=0):
        inferred_latent = self.get_inferred_latent(task_id, seed=seed)

        stats = defaultdict(list)
        trajectories = []
        latencies = []
        trajectory_records = []
        subgoal_records = []

        for ep in range(num_episodes):
            ep_seed = seed * 100000 + task_id * 1000 + ep
            self.env.action_space.seed(ep_seed)
            np.random.seed(ep_seed)
            obs, info = self.env.reset(seed=ep_seed, options=dict(task_id=task_id))
            planner.reset(obs, inferred_latent)

            done = False
            step = 0
            traj = [obs[:2].copy()]
            seed_key = jax.random.PRNGKey(ep_seed)

            # Record initial step
            sg_info = planner.get_subgoal_info()
            sg_xy = sg_info.get("subgoal_xy")
            planned_wps = sg_info.get("waypoints_xy", [])
            wps_json = json.dumps(planned_wps)

            trajectory_records.append({
                "method": planner.name,
                "seed": int(seed),
                "task_id": int(task_id),
                "episode": int(ep),
                "step": int(step),
                "x": float(obs[0]),
                "y": float(obs[1]),
                "reward": 0.0,
                "done": False,
            })
            subgoal_records.append({
                "method": planner.name,
                "seed": int(seed),
                "task_id": int(task_id),
                "episode": int(ep),
                "step": int(step),
                "subgoal_x": float(sg_xy[0]) if sg_xy is not None else None,
                "subgoal_y": float(sg_xy[1]) if sg_xy is not None else None,
                "is_direct_goal": bool(sg_info.get("is_direct_goal", False)),
                "planned_waypoints": wps_json,
            })

            t0 = time.perf_counter()
            while not done:
                action = planner.sample_action(
                    obs, inferred_latent, step=step, seed=seed_key, temperature=eval_temperature
                )
                sg_info = planner.get_subgoal_info()
                sg_xy = sg_info.get("subgoal_xy")

                next_obs, reward, terminated, truncated, info = self.env.step(action)
                step += 1
                done = terminated or truncated
                traj.append(next_obs[:2].copy())
                obs = next_obs

                trajectory_records.append({
                    "method": planner.name,
                    "seed": int(seed),
                    "task_id": int(task_id),
                    "episode": int(ep),
                    "step": int(step),
                    "x": float(next_obs[0]),
                    "y": float(next_obs[1]),
                    "reward": float(reward),
                    "done": bool(done),
                })
                subgoal_records.append({
                    "method": planner.name,
                    "seed": int(seed),
                    "task_id": int(task_id),
                    "episode": int(ep),
                    "step": int(step),
                    "subgoal_x": float(sg_xy[0]) if sg_xy is not None else None,
                    "subgoal_y": float(sg_xy[1]) if sg_xy is not None else None,
                    "is_direct_goal": bool(sg_info.get("is_direct_goal", False)),
                    "planned_waypoints": wps_json,
                })

            latencies.append((time.perf_counter() - t0) * 1000.0 / max(step, 1))

            for k, v in flatten(info).items():
                stats[k].append(v)
            trajectories.append(np.asarray(traj))

        mean_stats = {k: float(np.mean(v)) for k, v in stats.items()}
        mean_stats["latency_ms"] = float(np.mean(latencies)) if latencies else 0.0
        return mean_stats, trajectories, trajectory_records, subgoal_records

    def evaluate_all_tasks(self, planner, num_episodes=15, eval_temperature=0.0, seed=0):
        task_infos = self.env.unwrapped.task_infos if hasattr(self.env.unwrapped, "task_infos") else self.env.task_infos
        num_tasks = len(task_infos)

        all_metrics = defaultdict(list)
        all_trajs = []
        all_trajectory_records = []
        all_subgoal_records = []

        pbar = tqdm(range(1, num_tasks + 1), desc=f"Evaluating {planner.name} (Seed {seed})", leave=False)
        for task_id in pbar:
            task_stats, task_trajs, traj_recs, sg_recs = self.evaluate_task(
                planner,
                task_id=task_id,
                num_episodes=num_episodes,
                eval_temperature=eval_temperature,
                seed=seed,
            )
            for k, v in task_stats.items():
                all_metrics[k].append(v)
            all_trajs.extend(task_trajs)
            all_trajectory_records.extend(traj_recs)
            all_subgoal_records.extend(sg_recs)
            pbar.set_postfix({"success": f"{np.mean(all_metrics.get('success', [0])):.2%}"})

        summary = {
            "success_rate": float(np.mean(all_metrics["success"]) * 100.0) if "success" in all_metrics else 0.0,
            "mean_length": float(np.mean(all_metrics["episode.length"])) if "episode.length" in all_metrics else 0.0,
            "latency_ms": float(np.mean(all_metrics["latency_ms"])) if "latency_ms" in all_metrics else 0.0,
            "task_successes": [float(s) for s in all_metrics["success"]],
            "trajectories": all_trajs,
            "trajectory_records": all_trajectory_records,
            "subgoal_records": all_subgoal_records,
        }
        return summary
