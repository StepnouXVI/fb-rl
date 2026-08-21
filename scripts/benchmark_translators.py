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
    EnhancedSequenceWaypointAttentionPlanner,
    DistilledJAXPlanner,
)


def run_comprehensive_benchmark(
    checkpoint_dir: str = "fb-test",
    split: str = "medium",
    num_tasks: int = 5,
    episodes_per_task: int = 10,
    seeds: list = None,
    output_dir: str = "results/benchmarks",
):
    if seeds is None:
        seeds = list(range(1, 11))

    os.makedirs(output_dir, exist_ok=True)

    print("=" * 90)
    print(f"=== Comprehensive Benchmark: Single vs Sequence vs Distillation on {split.upper()} ===")
    print(f"=== Tasks: 1-{num_tasks} | Episodes per Task: {episodes_per_task} | Seeds: {seeds} ===")
    print("=" * 90)

    # 1. Load agent and environment
    agent, env, train_ds, val_ds, config = load_pretrained_agent(checkpoint_dir, split, seed=seeds[0])
    train_obs = train_ds["observations"]

    # 2. Setup Evaluator with full train_ds for accurate zero-shot goal inference
    evaluator = ZeroShotEvaluator(
        env=env,
        agent=agent,
        dataset_dict=train_ds,
        config=config,
        env_name=f"ogbench-antmaze-{split}-navigate-v0",
    )

    # 3. Instantiate Candidate Planners
    single_wp_ckpt = os.path.join(PROJECT_ROOT, "outputs", "checkpoints", f"best_single_wp_{split}.pkl")
    seq_attn_ckpt = os.path.join(PROJECT_ROOT, "outputs", "checkpoints", f"best_sequence_attn_{split}.pkl")
    enhanced_seq_ckpt = os.path.join(PROJECT_ROOT, "outputs", "checkpoints", f"best_enhanced_sequence_attn_{split}.pkl")
    distill_ckpt = os.path.join(PROJECT_ROOT, "results", f"distilled_jax_gated_attn_{split}.pkl")

    # Fallback to local root checkpoints if outputs/checkpoints is different
    if not os.path.exists(single_wp_ckpt):
        single_wp_ckpt = f"checkpoint_single_wp_{split}.pkl"
    if not os.path.exists(seq_attn_ckpt):
        seq_attn_ckpt = f"checkpoint_sequence_attn_{split}.pkl"
    if not os.path.exists(enhanced_seq_ckpt):
        enhanced_seq_ckpt = f"checkpoint_enhanced_seq_attn_{split}.pkl"

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

    # 5. Dijkstra + Enhanced Sequence-Aware Attention Translator
    if os.path.exists(enhanced_seq_ckpt):
        planners.append(
            EnhancedSequenceWaypointAttentionPlanner(
                agent=agent,
                dataset_observations=train_obs,
                checkpoint_path=enhanced_seq_ckpt,
                hidden_dim=384,
                num_heads=6,
                n_layers=4,
                max_seq_len=16,
                lookahead_dist=2.6,
                name="5. Dijkstra + Enhanced Sequence Attention",
            )
        )
    else:
        print(f"[Warning] enhanced_seq_attn checkpoint not found at {enhanced_seq_ckpt}")

    # 6. Distilled JAX Planner (Amortized O(1))
    if os.path.exists(distill_ckpt):
        planners.append(
            DistilledJAXPlanner(
                agent=agent,
                model_type="gated_attn",
                checkpoint_path=distill_ckpt,
                name="6. Distilled JAX GatedAttn [O(1)]",
            )
        )

    # 4. Run Evaluation across all Planners
    # 4. Run Evaluation across all Planners over all seeds
    all_summary_rows = []

    for planner in planners:
        print(f"\nEvaluating: >>> {planner.name} <<< across {len(seeds)} seeds: {seeds}")
        seed_task_sr = defaultdict(list)
        seed_overall_sr = []
        seed_latencies = []
        seed_steps = []

        for s in seeds:
            for task_id in range(1, num_tasks + 1):
                stats, _, _, _ = evaluator.evaluate_task(
                    planner=planner,
                    task_id=task_id,
                    num_episodes=episodes_per_task,
                    seed=s,
                )

                sr = float(stats.get("success", 0.0)) * 100.0
                steps = float(stats.get("episode_length", 0.0))
                lat = float(stats.get("latency_ms", 0.0))

                seed_task_sr[task_id].append(sr)
                seed_latencies.append(lat)
                seed_steps.append(steps)

            task_mean_for_seed = np.mean([seed_task_sr[t][-1] for t in range(1, num_tasks + 1)])
            seed_overall_sr.append(task_mean_for_seed)
            print(f"  Seed {s:02d}: Overall = {task_mean_for_seed:5.1f}% | " + " | ".join([f"T{t}:{seed_task_sr[t][-1]:.0f}%" for t in range(1, num_tasks + 1)]))

        overall_mean_sr = float(np.mean(seed_overall_sr))
        overall_std_sr = float(np.std(seed_overall_sr))
        overall_lat = float(np.mean(seed_latencies))
        overall_steps = float(np.mean(seed_steps))

        summary_entry = {
            "Method": planner.name,
            "Success Rate (%)": f"{overall_mean_sr:.1f} ± {overall_std_sr:.1f}",
            "Latency (ms)": f"{overall_lat:.2f}",
        }
        for task_id in range(1, num_tasks + 1):
            summary_entry[f"Task {task_id:02d} (%)"] = f"{np.mean(seed_task_sr[task_id]):.1f}"

        all_summary_rows.append(summary_entry)
        print(f"  -> FINAL {planner.name}: {overall_mean_sr:.1f}% ± {overall_std_sr:.1f}% | Latency = {overall_lat:.2f} ms")

    # 5. Save Summary DataFrame
    df_summary = pd.DataFrame(all_summary_rows)
    csv_path = os.path.join(output_dir, f"benchmark_summary_10seeds_{split}.csv")
    md_path = os.path.join(output_dir, f"benchmark_summary_10seeds_{split}.md")

    df_summary.to_csv(csv_path, index=False)
    with open(md_path, "w") as f:
        f.write(df_summary.to_markdown(index=False))

    print("\n" + "=" * 90)
    print("=== FINAL 10-SEED BENCHMARK SUMMARY TABLE ===")
    print("=" * 90)
    print(df_summary.to_markdown(index=False))
    print(f"\nSaved summary to {csv_path} and {md_path}")

    return df_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="medium")
    parser.add_argument("--num_tasks", type=int, default=5)
    parser.add_argument("--episodes_per_task", type=int, default=10)
    parser.add_argument("--start_seed", type=int, default=1)
    parser.add_argument("--end_seed", type=int, default=10)
    parser.add_argument("--output_dir", type=str, default="results/benchmarks")
    args = parser.parse_args()

    seeds = list(range(args.start_seed, args.end_seed + 1))
    run_comprehensive_benchmark(
        split=args.split,
        num_tasks=args.num_tasks,
        episodes_per_task=args.episodes_per_task,
        seeds=seeds,
        output_dir=args.output_dir,
    )
