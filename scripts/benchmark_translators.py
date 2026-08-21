import os
import sys
import time
import json
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict

# Root path setup
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.agent_loader import load_pretrained_agent
from src.evaluator import ZeroShotEvaluator
from src.planners import (
    BaselinePlanner,
    BufferGraphPlanner,
    WaypointTranslatorPlanner,
    SequenceWaypointAttentionPlanner,
    DistilledJAXPlanner,
)


def run_comprehensive_benchmark(
    checkpoint_dir: str = "fb-test",
    split: str = "medium",
    num_tasks: int = 5,
    episodes_per_task: int = 15,
    seed: int = 42,
    output_dir: str = "results/benchmarks",
):
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 90)
    print(f"=== Comprehensive Benchmark: Single vs Sequence vs Distillation on {split.upper()} ===")
    print(f"=== Tasks: 1-{num_tasks} | Episodes per Task: {episodes_per_task} | Seed: {seed} ===")
    print("=" * 90)

    # 1. Load agent and environment
    agent, env, train_ds, val_ds, config = load_pretrained_agent(checkpoint_dir, split, seed=seed)
    train_obs = train_ds["observations"]

    # 2. Setup Evaluator
    evaluator = ZeroShotEvaluator(
        env=env,
        agent=agent,
        dataset_dict=val_ds,
        config=config,
        env_name=f"ogbench-antmaze-{split}-navigate-v0",
    )

    # 3. Instantiate Candidate Planners
    single_wp_ckpt = os.path.join(PROJECT_ROOT, "outputs", "checkpoints", f"best_single_wp_{split}.pkl")
    seq_attn_ckpt = os.path.join(PROJECT_ROOT, "outputs", "checkpoints", f"best_sequence_attn_{split}.pkl")
    distill_ckpt = os.path.join(PROJECT_ROOT, "results", f"distilled_jax_gated_attn_{split}.pkl")

    # Fallback to local root checkpoints if outputs/checkpoints is different
    if not os.path.exists(single_wp_ckpt):
        single_wp_ckpt = f"checkpoint_single_wp_{split}.pkl"
    if not os.path.exists(seq_attn_ckpt):
        seq_attn_ckpt = f"checkpoint_sequence_attn_{split}.pkl"

    planners = []

    # 1. Single-Intention Baseline
    planners.append(
        BaselinePlanner(agent, dataset_states=train_obs, name="1. Single-Intention Baseline")
    )

    # 2. Dijkstra Teacher (with standard high_actor)
    planners.append(
        BufferGraphPlanner(
            agent,
            train_obs,
            n_landmarks=1000,
            max_edge_radius=3.5,
            reachability_cutoff=35.0,
            lookahead_dist=2.6,
            name="2. Dijkstra Teacher (high_actor)",
        )
    )

    # 3. Dijkstra + Single Waypoint Translator
    if os.path.exists(single_wp_ckpt):
        planners.append(
            WaypointTranslatorPlanner(
                agent=agent,
                dataset_observations=train_obs,
                checkpoint_path=single_wp_ckpt,
                hidden_dim=384,
                n_layers=4,
                lookahead_dist=2.6,
                name="3. Dijkstra + Single WP Translator",
            )
        )
    else:
        print(f"[Warning] single_wp checkpoint not found at {single_wp_ckpt}")

    # 4. Dijkstra + Sequence-Aware Attention Translator
    if os.path.exists(seq_attn_ckpt):
        planners.append(
            SequenceWaypointAttentionPlanner(
                agent=agent,
                dataset_observations=train_obs,
                checkpoint_path=seq_attn_ckpt,
                hidden_dim=384,
                num_heads=6,
                n_layers=4,
                max_seq_len=16,
                lookahead_dist=2.6,
                name="4. Dijkstra + Sequence Attention Translator",
            )
        )
    else:
        print(f"[Warning] seq_attn checkpoint not found at {seq_attn_ckpt}")

    # 5. Distilled JAX Planner (Amortized O(1))
    if os.path.exists(distill_ckpt):
        planners.append(
            DistilledJAXPlanner(
                agent=agent,
                model_type="gated_attn",
                checkpoint_path=distill_ckpt,
                name="5. Distilled JAX GatedAttn [O(1)]",
            )
        )

    # 4. Run Evaluation across all Planners
    all_summary_rows = []
    task_breakdown = defaultdict(dict)

    for planner in planners:
        print(f"\nEvaluating: >>> {planner.name} <<<")
        planner_sr_list = []
        planner_steps_list = []
        planner_lat_list = []

        for task_id in range(1, num_tasks + 1):
            stats, _, _, _ = evaluator.evaluate_task(
                planner=planner,
                task_id=task_id,
                num_episodes=episodes_per_task,
                seed=seed,
            )

            sr = float(stats.get("success", 0.0)) * 100.0
            steps = float(stats.get("episode_length", 0.0))
            lat = float(stats.get("latency_ms", 0.0))

            planner_sr_list.append(sr)
            planner_steps_list.append(steps)
            planner_lat_list.append(lat)

            task_breakdown[planner.name][f"Task_{task_id:02d}"] = sr
            print(f"  Task {task_id:02d}: Success = {sr:5.1f}% | Steps = {steps:5.1f} | Latency = {lat:.2f} ms/step")

        overall_sr = float(np.mean(planner_sr_list))
        overall_steps = float(np.mean(planner_steps_list))
        overall_lat = float(np.mean(planner_lat_list))

        summary_entry = {
            "Method": planner.name,
            "Overall Success (%)": round(overall_sr, 1),
            "Mean Steps": round(overall_steps, 1),
            "Latency (ms/step)": round(overall_lat, 2),
        }
        for task_id in range(1, num_tasks + 1):
            summary_entry[f"Task {task_id:02d} (%)"] = task_breakdown[planner.name][f"Task_{task_id:02d}"]

        all_summary_rows.append(summary_entry)
        print(f"  -> OVERALL {planner.name}: Success = {overall_sr:.1f}%, Latency = {overall_lat:.2f} ms")

    # 5. Save Summary DataFrame
    df_summary = pd.DataFrame(all_summary_rows)
    csv_path = os.path.join(output_dir, f"benchmark_summary_{split}.csv")
    md_path = os.path.join(output_dir, f"benchmark_summary_{split}.md")

    df_summary.to_csv(csv_path, index=False)
    with open(md_path, "w") as f:
        f.write(df_summary.to_markdown(index=False))

    print("\n" + "=" * 90)
    print("=== FINAL BENCHMARK SUMMARY TABLE ===")
    print("=" * 90)
    print(df_summary.to_markdown(index=False))
    print(f"\nSaved summary to {csv_path} and {md_path}")

    return df_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="medium")
    parser.add_argument("--num_tasks", type=int, default=5)
    parser.add_argument("--episodes_per_task", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="results/benchmarks")
    args = parser.parse_args()

    run_comprehensive_benchmark(
        split=args.split,
        num_tasks=args.num_tasks,
        episodes_per_task=args.episodes_per_task,
        seed=args.seed,
        output_dir=args.output_dir,
    )
