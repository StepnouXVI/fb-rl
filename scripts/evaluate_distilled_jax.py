import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.agent_loader import load_pretrained_agent
from src.evaluator import ZeroShotEvaluator
from src.planners import DistilledJAXPlanner


def evaluate_distilled_jax(
    checkpoint_dir: str = "fb-test",
    split: str = "medium",
    model_type: str = "gated_attn",
    weights_path: str = None,
    num_tasks: int = 5,
    episodes_per_task: int = 10,
    seed: int = 0,
):
    if weights_path is None:
        weights_path = f"results/distilled_jax_{model_type}_{split}.pkl"

    print("=" * 80)
    print(f"=== Zero-Shot Benchmarking Distilled JAX Planner: {model_type} on {split} ===")
    print(f"=== Weights: {weights_path} | Tasks: 1-{num_tasks} | Episodes/task: {episodes_per_task} ===")
    print("=" * 80)

    agent, env, train_ds, val_ds, config = load_pretrained_agent(checkpoint_dir, split, seed=seed)

    planner = DistilledJAXPlanner(
        agent=agent,
        model_type=model_type,
        checkpoint_path=weights_path,
    )

    evaluator = ZeroShotEvaluator(
        env=env,
        agent=agent,
        dataset_dict=val_ds,
        config=config,
        env_name=f"ogbench-antmaze-{split}-navigate-v0",
    )

    task_results = []
    total_steps = []
    latencies = []

    for task_id in range(1, num_tasks + 1):
        task_stats, task_trajs, traj_recs, sg_recs = evaluator.evaluate_task(
            planner=planner,
            task_id=task_id,
            num_episodes=episodes_per_task,
            seed=seed,
        )

        sr = float(task_stats.get("success", 0.0)) * 100.0
        avg_steps = float(task_stats.get("episode_length", 0.0))
        avg_lat = float(task_stats.get("latency_ms", 0.0))

        print(f"Task {task_id:02d}: Success Rate = {sr:5.1f}% | Avg Steps = {avg_steps:5.1f} | Latency = {avg_lat:.2f} ms/step")

        task_results.append(sr)
        total_steps.append(avg_steps)
        latencies.append(avg_lat)

    overall_sr = np.mean(task_results)
    mean_steps = np.mean(total_steps)
    mean_latency = np.mean(latencies) * 1000.0

    print("\n" + "=" * 80)
    print(f"OVERALL EVALUATION RESULTS: {model_type.upper()} ({split})")
    print(f"  Success Rate:          {overall_sr:.1f}%")
    print(f"  Mean Episode Steps:    {mean_steps:.1f}")
    print(f"  Avg Inference Latency: {mean_latency:.3f} ms/step")
    print("=" * 80)

    return {
        "model_type": model_type,
        "split": split,
        "success_rate": overall_sr,
        "mean_steps": mean_steps,
        "latency_ms": mean_latency,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="medium")
    parser.add_argument("--model_type", type=str, default="gated_attn")
    parser.add_argument("--weights_path", type=str, default=None)
    parser.add_argument("--num_tasks", type=int, default=5)
    parser.add_argument("--episodes_per_task", type=int, default=10)
    args = parser.parse_args()

    evaluate_distilled_jax(
        split=args.split,
        model_type=args.model_type,
        weights_path=args.weights_path,
        num_tasks=args.num_tasks,
        episodes_per_task=args.episodes_per_task,
    )
