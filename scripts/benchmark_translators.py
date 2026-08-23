#!/usr/bin/env python3
import os
import sys
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.agent_loader import load_pretrained_agent
from src.evaluator import ZeroShotEvaluator
from src.planners import (
    BaselinePlanner,
    BufferGraphPlanner,
    WaypointTranslatorPlanner,
    EnhancedSequenceWaypointAttentionPlanner,
    DistilledJAXPlanner,
)


def _find_checkpoint(name, dirs):
    for d in dirs:
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    return None


def _init_candidate_planners(agent, train_obs, split, n_landmarks):
    dirs = [os.path.join(PROJECT_ROOT, "outputs", "checkpoints"), os.path.join(PROJECT_ROOT, "results", "checkpoints"), PROJECT_ROOT]
    planners = [
        BaselinePlanner(agent, dataset_states=train_obs, name="1. Single-Intention Baseline"),
        BufferGraphPlanner(agent, train_obs, n_landmarks=n_landmarks, name="2. Dijkstra Teacher (high_actor)"),
    ]
    single_wp_ckpt = _find_checkpoint(f"best_single_wp_{split}.pkl", dirs) or _find_checkpoint(f"checkpoint_single_wp_{split}.pkl", dirs)
    if single_wp_ckpt:
        planners.append(WaypointTranslatorPlanner(agent, train_obs, single_wp_ckpt, n_landmarks, hidden_dim=384, n_layers=4, name="3. Dijkstra + Single WP Translator"))

    enhanced_ckpt = _find_checkpoint(f"best_enhanced_sequence_attn_{split}.pkl", dirs) or _find_checkpoint(f"checkpoint_enhanced_seq_attn_{split}.pkl", dirs)
    if enhanced_ckpt:
        planners.append(EnhancedSequenceWaypointAttentionPlanner(agent, train_obs, enhanced_ckpt, n_landmarks, hidden_dim=384, num_heads=6, n_layers=4, name="5. Dijkstra + Enhanced Sequence Attention"))

    jax_ckpt = _find_checkpoint(f"distilled_jax_gated_attn_{split}.pkl", dirs)
    if jax_ckpt:
        planners.append(DistilledJAXPlanner(agent, checkpoint_path=jax_ckpt, model_type="gated_attn", name="6. Distilled JAX GatedAttn [O(1)]"))
    return planners


def _eval_single_planner(evaluator, planner, seeds, num_tasks, ep_per_task):
    print(f"\nEvaluating: >>> {planner.name} <<< across {len(seeds)} seeds: {seeds}")
    seed_task_sr, seed_overall_sr, seed_latencies = defaultdict(list), [], []
    for s in seeds:
        for t_id in range(1, num_tasks + 1):
            stats, _, _, _ = evaluator.evaluate_task(planner, t_id, ep_per_task, seed=s)
            seed_task_sr[t_id].append(float(stats.get("success", 0.0)) * 100.0)
            seed_latencies.append(float(stats.get("latency_ms", 0.0)))
        task_mean = np.mean([seed_task_sr[t][-1] for t in range(1, num_tasks + 1)])
        seed_overall_sr.append(task_mean)
        print(f"  Seed {s:02d}: Overall = {task_mean:5.1f}% | " + " | ".join([f"T{t}:{seed_task_sr[t][-1]:.0f}%" for t in range(1, num_tasks + 1)]))

    entry = {"Method": planner.name, "Success Rate (%)": f"{np.mean(seed_overall_sr):.1f} ± {np.std(seed_overall_sr):.1f}", "Latency (ms)": f"{np.mean(seed_latencies):.2f}"}
    for t_id in range(1, num_tasks + 1):
        entry[f"Task {t_id:02d} (%)"] = f"{np.mean(seed_task_sr[t_id]):.1f}"
    return entry


def _save_summary_tables(rows, output_dir, split):
    df = pd.DataFrame(rows)
    os.makedirs(output_dir, exist_ok=True)
    csv_p, md_p = os.path.join(output_dir, f"benchmark_summary_10seeds_{split}.csv"), os.path.join(output_dir, f"benchmark_summary_10seeds_{split}.md")
    df.to_csv(csv_p, index=False)
    try:
        md_text = df.to_markdown(index=False)
    except Exception:
        md_text = df.to_string()
    with open(md_p, "w") as f:
        f.write(md_text)
    print("\n" + "=" * 90 + "\n=== FINAL BENCHMARK SUMMARY TABLE ===\n" + "=" * 90)
    print(md_text)
    print(f"\nSaved summary to {csv_p} and {md_p}")
    return df


def run_comprehensive_benchmark(checkpoint_dir="fb-test", split="medium", num_tasks=5, episodes_per_task=10, seeds=None, output_dir="results/benchmarks"):
    seeds = seeds or list(range(1, 11))
    max_steps = 1500 if split in ["large", "giant"] else 1000
    agent, env, train_ds, _, cfg = load_pretrained_agent(checkpoint_dir, split, seed=seeds[0], max_episode_steps=max_steps)
    evaluator = ZeroShotEvaluator(env, agent, train_ds, cfg, env_name=f"ogbench-antmaze-{split}-navigate-v0", max_episode_steps=max_steps)
    planners = _init_candidate_planners(agent, train_ds["observations"], split, 2000 if split == "large" else 1000)
    rows = [_eval_single_planner(evaluator, p, seeds, num_tasks, episodes_per_task) for p in planners]
    return _save_summary_tables(rows, output_dir, split)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="medium")
    parser.add_argument("--num_tasks", type=int, default=5)
    parser.add_argument("--episodes_per_task", type=int, default=10)
    parser.add_argument("--start_seed", type=int, default=1)
    parser.add_argument("--end_seed", type=int, default=10)
    parser.add_argument("--output_dir", type=str, default="results/benchmarks")
    args = parser.parse_args()
    run_comprehensive_benchmark(split=args.split, num_tasks=args.num_tasks, episodes_per_task=args.episodes_per_task, seeds=list(range(args.start_seed, args.end_seed + 1)), output_dir=args.output_dir)


if __name__ == "__main__":
    main()
