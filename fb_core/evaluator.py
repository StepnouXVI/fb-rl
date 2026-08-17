"""
Evaluator: Unified multi-seed rollout engine for FB hierarchical planners.
Supports real Gymnasium/OGBench environments and high-fidelity simulated 2D Maze environments with wall collisions.
"""

import time
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from fb_core.base_planner import BaseHierarchicalPlanner
from fb_core.fb_model_wrapper import FBModelWrapper


def line_segment_intersection(
    p1: np.ndarray, p2: np.ndarray,
    p3: np.ndarray, p4: np.ndarray,
) -> Tuple[bool, float]:
    """
    Check if line segment p1->p2 intersects segment p3->p4.
    Returns (intersects: bool, t: float) where t is the intersection parameter on p1->p2.
    """
    d1 = p2 - p1
    d2 = p4 - p3
    det = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(det) < 1e-8:
        return False, 1.0

    dp = p3 - p1
    t = (dp[0] * d2[1] - dp[1] * d2[0]) / det
    u = (dp[0] * d1[1] - dp[1] * d1[0]) / det

    if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0:
        return True, float(t)
    return False, 1.0


class SimulatedMaze2D:
    """
    2D Maze Environment with wall collision and sliding physics.
    Matches AntMaze-Medium corridor topology.
    """

    def __init__(self):
        self.bounds = np.array([[-8.0, 8.0], [-8.0, 8.0]], dtype=np.float32)
        # Maze interior walls: list of [p_start, p_end]
        self.walls = [
            # Horizontal walls
            (np.array([-8.0, 0.0]), np.array([0.0, 0.0])),
            (np.array([-4.0, 4.0]), np.array([4.0, 4.0])),
            (np.array([0.0, -4.0]), np.array([8.0, -4.0])),
            # Vertical walls
            (np.array([0.0, -4.0]), np.array([0.0, 4.0])),
            (np.array([4.0, -8.0]), np.array([4.0, 0.0])),
            (np.array([-4.0, -8.0]), np.array([-4.0, -4.0])),
        ]
        # Benchmark rooms / waypoints for generating diverse navigation problems
        self.room_centers = [
            np.array([-6.0, -6.0]),
            np.array([-6.0, 2.0]),
            np.array([-6.0, 6.0]),
            np.array([-2.0, 2.0]),
            np.array([-2.0, 6.0]),
            np.array([2.0, 6.0]),
            np.array([2.0, 2.0]),
            np.array([2.0, -2.0]),
            np.array([6.0, -2.0]),
            np.array([6.0, 6.0]),
            np.array([6.0, -6.0]),
            np.array([2.0, -6.0]),
        ]

    def sample_episode_endpoints(self, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
        """Sample start and goal positions requiring multi-corridor traversal."""
        idx1, idx2 = rng.choice(len(self.room_centers), size=2, replace=False)
        start_pos = self.room_centers[idx1] + rng.uniform(-0.8, 0.8, size=2)
        goal_pos = self.room_centers[idx2] + rng.uniform(-0.8, 0.8, size=2)

        start_state = np.zeros(29, dtype=np.float32)
        start_state[:2] = start_pos
        start_state[2:4] = rng.uniform(-0.1, 0.1, size=2)  # initial velocity

        goal_state = np.zeros(29, dtype=np.float32)
        goal_state[:2] = goal_pos
        return start_state, goal_state

    def step(self, pos: np.ndarray, action: np.ndarray, dt: float = 0.25) -> np.ndarray:
        """
        Step physics: move along action with collision detection and sliding response.
        """
        move = action[:2] * dt
        new_pos = pos + move

        # Check wall collisions
        for w_start, w_end in self.walls:
            hit, t = line_segment_intersection(pos, new_pos, w_start, w_end)
            if hit:
                # Stop at contact point
                contact = pos + move * (t * 0.95)
                # Compute wall tangent
                wall_vec = w_end - w_start
                wall_len = np.linalg.norm(wall_vec)
                if wall_len > 1e-6:
                    wall_tangent = wall_vec / wall_len
                    # Project remaining velocity along wall tangent (sliding)
                    remaining = move * (1.0 - t)
                    slide = np.dot(remaining, wall_tangent) * wall_tangent * 0.5
                    new_pos = contact + slide
                else:
                    new_pos = contact
                pos = new_pos

        # Clamp to outer bounding box
        new_pos[0] = np.clip(new_pos[0], -7.5, 7.5)
        new_pos[1] = np.clip(new_pos[1], -7.5, 7.5)
        return new_pos


class EpisodeEvaluator:
    """
    Evaluates hierarchical planners on navigation environments.
    """

    def __init__(
        self,
        env_name: str = "ogbench-antmaze-medium-navigate-v0",
        max_episode_steps: int = 600,
        goal_threshold: float = 0.8,
    ):
        self.env_name = env_name
        self.max_episode_steps = max_episode_steps
        self.goal_threshold = goal_threshold
        self.sim_maze = SimulatedMaze2D()

    def evaluate_planner(
        self,
        planner: BaseHierarchicalPlanner,
        fb_model: FBModelWrapper,
        env: Any = None,
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
        rng = np.random.default_rng(seed * 1000 + 42)

        for ep in range(num_episodes):
            ep_rng = np.random.default_rng(seed * 1000 + ep)

            # Handle environment reset
            if env is not None and hasattr(env, "reset"):
                obs_out = env.reset()
                if isinstance(obs_out, tuple):
                    obs, info = obs_out
                else:
                    obs, info = obs_out, {}
                goal = info.get("goal", None)
                if goal is None:
                    goal = obs.copy()
                    goal[:2] += 5.0
            else:
                # Use high-fidelity simulated maze
                obs, goal = self.sim_maze.sample_episode_endpoints(ep_rng)
                info = {"goal": goal}

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
                if env is not None and hasattr(env, "step"):
                    step_out = env.step(action)
                    if len(step_out) == 5:
                        obs, reward, terminated, truncated, info = step_out
                        done = terminated or truncated
                    else:
                        obs, reward, done, info = step_out
                else:
                    # Simulated 2D Maze step with collisions
                    new_pos = self.sim_maze.step(obs[:2], action)
                    obs[:2] = new_pos
                    dist_to_goal = float(np.linalg.norm(obs[:2] - goal[:2]))
                    done = dist_to_goal < self.goal_threshold
                    info = {"success": done}

                traj_xy.append(obs[:2].copy())
                planner.on_step_end(obs, 0.0, done, info)
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
