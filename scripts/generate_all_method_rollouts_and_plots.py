#!/usr/bin/env python3
import os
import sys
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm

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
from scripts.visualize_trajectories import export_all_methods_trajectories, sanitize_method_name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="large", help="Maze split (medium, large, giant)")
    parser.add_argument("--checkpoint_dir", type=str, default="fb-test")
    parser.add_argument("--num_tasks", type=int, default=5)
    parser.add_argument("--episodes_per_task", type=int, default=3)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--output_base_dir", type=str, default="results")
    args = parser.parse_args()

    split = args.split
    print("=" * 80)
    print(f"=== Generating & Exporting Trajectories for ALL Methods on {split.upper()} ===")
    print(f"=== Tasks: 1-{args.num_tasks} | Episodes: {args.episodes_per_task} | Seeds: {args.seeds} ===")
    print("=" * 80)

    # 1. Load Agent & Environment
    max_episode_steps = 1500 if split in ["large", "giant"] else 1000
    agent, env, train_ds, _, config = load_pretrained_agent(
        args.checkpoint_dir, split, seed=args.seeds[0], max_episode_steps=max_episode_steps
    )
    train_obs = train_ds["observations"]

    evaluator = ZeroShotEvaluator(
        env=env,
        agent=agent,
        dataset_dict=train_ds,
        config=config,
        env_name=f"ogbench-antmaze-{split}-navigate-v0",
        max_episode_steps=max_episode_steps,
    )

    # 2. Checkpoints
    single_wp_ckpt = os.path.join(PROJECT_ROOT, "outputs", "checkpoints", f"best_single_wp_{split}.pkl")
    enhanced_seq_ckpt = os.path.join(PROJECT_ROOT, "outputs", "checkpoints", f"best_enhanced_sequence_attn_{split}.pkl")
    distill_ckpt = os.path.join(PROJECT_ROOT, "results", f"distilled_jax_gated_attn_{split}.pkl")

    # Fallback checks if naming convention differs
    if not os.path.exists(single_wp_ckpt):
        single_wp_ckpt = os.path.join(PROJECT_ROOT, f"checkpoint_single_wp_{split}.pkl")
    if not os.path.exists(enhanced_seq_ckpt):
        alt_ckpt = os.path.join(PROJECT_ROOT, "outputs", "checkpoints", f"best_enhanced_seq_attn_{split}.pkl")
        if os.path.exists(alt_ckpt):
            enhanced_seq_ckpt = alt_ckpt
        else:
            enhanced_seq_ckpt = os.path.join(PROJECT_ROOT, f"checkpoint_enhanced_seq_attn_{split}.pkl")

    n_landmarks = 2000 if split == "large" else 1000

    planners = []

    # 1. Dijkstra + Enhanced Sequence Attention (Best Method)
    if os.path.exists(enhanced_seq_ckpt):
        planners.append(
            EnhancedSequenceWaypointAttentionPlanner(
                agent=agent,
                dataset_observations=train_obs,
                checkpoint_path=enhanced_seq_ckpt,
                n_landmarks=n_landmarks,
                hidden_dim=384,
                num_heads=6,
                n_layers=4,
                max_seq_len=16,
                lookahead_dist=2.6,
                name="Enhanced Sequence Attention",
            )
        )

    # 2. Single-Intention Baseline
    planners.append(
        BaselinePlanner(agent, dataset_states=train_obs, name="Single-Intention Baseline")
    )

    # 3. Dijkstra Teacher (high_actor)
    planners.append(
        BufferGraphPlanner(
            agent,
            train_obs,
            n_landmarks=n_landmarks,
            max_edge_radius=3.5,
            reachability_cutoff=35.0,
            lookahead_dist=2.6,
            name="Dijkstra Teacher",
        )
    )

    # 4. Dijkstra + Single WP Translator
    if os.path.exists(single_wp_ckpt):
        planners.append(
            WaypointTranslatorPlanner(
                agent=agent,
                dataset_observations=train_obs,
                checkpoint_path=single_wp_ckpt,
                n_landmarks=n_landmarks,
                hidden_dim=384,
                n_layers=4,
                lookahead_dist=2.6,
                name="Single WP Translator",
            )
        )

    # 5. Distilled JAX GatedAttn
    if os.path.exists(distill_ckpt):
        planners.append(
            DistilledJAXPlanner(
                agent=agent,
                checkpoint_path=distill_ckpt,
                model_type="gated_attn",
                name="Distilled JAX GatedAttn",
            )
        )

    all_trajectories = []
    all_subgoals = []

    for planner in planners:
        print(f"\n--- Running Rollouts for: [{planner.name}] ---")
        for s in args.seeds:
            for t_id in range(1, args.num_tasks + 1):
                stats, _, traj_records, sg_records = evaluator.evaluate_task(
                    planner=planner,
                    task_id=t_id,
                    num_episodes=args.episodes_per_task,
                    seed=s,
                )
                all_trajectories.extend(traj_records)
                all_subgoals.extend(sg_records)
                sr = stats.get("success", 0.0)
                print(f"  [{planner.name}] Seed {s}, Task {t_id}: Success = {sr*100.0:.1f}%")

    df_traj = pd.DataFrame(all_trajectories)
    df_sg = pd.DataFrame(all_subgoals)

    traj_csv_path = os.path.join(args.output_base_dir, f"trajectories_{split}.csv")
    sg_csv_path = os.path.join(args.output_base_dir, f"subgoals_{split}.csv")

    df_traj.to_csv(traj_csv_path, index=False)
    df_sg.to_csv(sg_csv_path, index=False)
    print(f"\nSaved {len(df_traj)} trajectory steps to {traj_csv_path}")

    # 3. Export all individual trajectory plots organized by results/{method}/{split}/{success,failed}/
    print("\n--- Exporting Trajectory Images into results/{method_name}/{split}/{failed,success}/ ---")
    summary_counts = export_all_methods_trajectories(
        df_traj=df_traj,
        df_sg=df_sg,
        maze_type=split,
        base_output_dir=args.output_base_dir,
    )

    print("\n" + "=" * 80)
    print(f"=== Trajectory Rendering Summary on {split.upper()} ===")
    print("=" * 80)
    for method_name, counts in summary_counts.items():
        print(f"  • {method_name:32s} -> {counts['success']} SUCCESS, {counts['failed']} FAILED")
    print(f"\nAll plots successfully saved under '{args.output_base_dir}/<method_name>/{split}/'!")


if __name__ == "__main__":
    main()
