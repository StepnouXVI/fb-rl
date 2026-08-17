"""
Evaluator: Unified multi-seed rollout engine for FB hierarchical planners.
"""

import time
from typing import Dict, Any
import numpy as np
from fb_core.base_planner import BaseHierarchicalPlanner
from fb_core.fb_model_wrapper import FBModelWrapper


class EpisodeEvaluator:
    """
    Evaluates hierarchical planners on OGBench navigation environments.
    # ponytail: Straightforward evaluation loop recording successes, lengths, latency, and 2D paths.
    """

    def __init__(
        self,
        env_name: str = "ogbench-antmaze-medium-navigate-v0",
        max_episode_steps: int = 600,
        goal_threshold: float = 0.5,
    ):
        self.env_name = env_name
        self.max_episode_steps = max_episode_steps
        self.goal_threshold = goal_threshold

    def evaluate_planner(
        self,
        planner: BaseHierarchicalPlanner,
        fb_model: FBModelWrapper,
        env: Any,
        num_episodes: int = 20,
        seed: int = 0,
    ) -> Dict[str, Any]:
        """
        Run evaluation for a specific seed across num_episodes.
        """
        episode_successes = []
        episode_steps = []
        step_latencies_ms = []
        trajectories = []

        for ep in range(num_episodes):
            np.random.seed(seed * 1000 + ep)
            
            # Reset environment
            if hasattr(env, "reset"):
                obs_out = env.reset()
                if isinstance(obs_out, tuple):
                    obs, info = obs_out
                else:
                    obs, info = obs_out, {}
            else:
                obs = np.zeros(29, dtype=np.float32)
                info = {}

            goal = info.get("goal", None)
            if goal is None:
                # Default goal for maze navigation from info or synthetic offset
                goal = obs.copy()
                goal[0] += 5.0
                goal[1] += 5.0

            planner.reset(obs, goal)
            traj_xy = [obs[:2].copy()]
            done = False
            step = 0
            ep_success = False

            while not done and step < self.max_episode_steps:
                # Time the planner decision step
                t0 = time.perf_counter()
                intention_z = planner.get_intention(obs, goal, step=step)
                t1 = time.perf_counter()
                step_latencies_ms.append((t1 - t0) * 1000.0)

                # Low-level actor executes action
                action = fb_model.sample_action(obs, intention_z, temperature=0.0)
                
                # Step environment
                if hasattr(env, "step"):
                    step_out = env.step(action)
                    if len(step_out) == 5:
                        obs, reward, terminated, truncated, info = step_out
                        done = terminated or truncated
                    else:
                        obs, reward, done, info = step_out
                else:
                    # Synthetic environment transition step
                    obs[:2] += action[:2] * 0.2
                    dist_to_goal = float(np.linalg.norm(obs[:2] - goal[:2]))
                    done = dist_to_goal < self.goal_threshold
                    info = {"success": done}

                traj_xy.append(obs[:2].copy())
                planner.on_step_end(obs, reward if 'reward' in locals() else 0.0, done, info)
                step += 1

                # Check goal completion
                if info.get("success", False) or np.linalg.norm(obs[:2] - goal[:2]) < self.goal_threshold:
                    ep_success = True
                    done = True

            episode_successes.append(1.0 if ep_success else 0.0)
            episode_steps.append(step)
            trajectories.append(np.asarray(traj_xy))

        success_rate = float(np.mean(episode_successes) * 100.0)
        mean_steps = float(np.mean(episode_steps))
        mean_lat = float(np.mean(step_latencies_ms)) if step_latencies_ms else 0.0

        return {
            "method_name": planner.name,
            "seed": seed,
            "num_episodes": num_episodes,
            "success_rate": success_rate,
            "mean_steps": mean_steps,
            "mean_latency_ms": mean_lat,
            "episode_successes": episode_successes,
            "episode_steps": episode_steps,
            "trajectories": trajectories,
        }
