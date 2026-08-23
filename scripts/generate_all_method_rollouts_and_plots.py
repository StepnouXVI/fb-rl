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
from scripts.visualize_trajectories import export_all_methods_trajectories


def _find_checkpoint(filename, fallback_dirs):
    for d in fallback_dirs:
        path = os.path.join(d, filename)
        if os.path.exists(path):
            return path
    return None


def _init_planners(agent, train_obs, split, n_landmarks):
    ckpt_dirs = [
        os.path.join(PROJECT_ROOT, "outputs", "checkpoints"),
        os.path.join(PROJECT_ROOT, "results", "checkpoints"),
        PROJECT_ROOT,
    ]
    planners = []

    # 1. Enhanced Sequence Attention
    seq_ckpt = _find_checkpoint(f"best_enhanced_sequence_attn_{split}.pkl", ckpt_dirs) or _find_checkpoint(f"checkpoint_enhanced_seq_attn_{split}.pkl", ckpt_dirs)
    if seq_ckpt:
        planners.append(EnhancedSequenceWaypointAttentionPlanner(agent, train_obs, seq_ckpt, n_landmarks, name="Enhanced Sequence Attention"))

    # 2. Single-Intention Baseline & Dijkstra Teacher
    planners.append(BaselinePlanner(agent, train_obs, name="Single-Intention Baseline"))
    planners.append(BufferGraphPlanner(agent, train_obs, n_landmarks, name="Dijkstra Teacher"))

    # 3. Single WP Translator
    wp_ckpt = _find_checkpoint(f"best_single_wp_{split}.pkl", ckpt_dirs) or _find_checkpoint(f"checkpoint_single_wp_{split}.pkl", ckpt_dirs)
    if wp_ckpt:
        planners.append(WaypointTranslatorPlanner(agent, train_obs, wp_ckpt, n_landmarks, name="Single WP Translator"))

    # 4. Distilled JAX GatedAttn
    jax_ckpt = _find_checkpoint(f"distilled_jax_gated_attn_{split}.pkl", ckpt_dirs)
    if jax_ckpt:
        planners.append(DistilledJAXPlanner(agent, checkpoint_path=jax_ckpt, model_type="gated_attn", name="Distilled JAX GatedAttn"))

    return planners


def _collect_rollouts(evaluator, planners, seeds, num_tasks, ep_per_task):
    all_trajs, all_sgs = [], []
    for planner in planners:
        print(f"\n--- Running Rollouts for: [{planner.name}] ---")
        for s in seeds:
            for t_id in range(1, num_tasks + 1):
                stats, _, traj_recs, sg_recs = evaluator.evaluate_task(planner, t_id, ep_per_task, seed=s)
                all_trajs.extend(traj_recs)
                all_sgs.extend(sg_recs)
                print(f"  [{planner.name}] Seed {s}, Task {t_id}: Success = {stats.get('success', 0.0)*100.0:.1f}%")
    return pd.DataFrame(all_trajs), pd.DataFrame(all_sgs)


def _export_results(df_traj, df_sg, split, base_out):
    data_dir = os.path.join(base_out, "data")
    os.makedirs(data_dir, exist_ok=True)
    traj_p, sg_p = os.path.join(data_dir, f"trajectories_{split}.csv"), os.path.join(data_dir, f"subgoals_{split}.csv")
    df_traj.to_csv(traj_p, index=False)
    df_sg.to_csv(sg_p, index=False)
    print(f"\nSaved telemetry datasets to {traj_p} & {sg_p}")

    print(f"\n--- Exporting Trajectory Images to {base_out}/<method>/{split}/ ---")
    summary = export_all_methods_trajectories(df_traj, df_sg, maze_type=split, base_output_dir=base_out)
    for m, c in summary.items():
        print(f"  • {m:32s} -> {c['success']} SUCCESS, {c['failed']} FAILED")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="large")
    parser.add_argument("--checkpoint_dir", type=str, default="fb-test")
    parser.add_argument("--num_tasks", type=int, default=5)
    parser.add_argument("--episodes_per_task", type=int, default=2)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--output_base_dir", type=str, default="results")
    args = parser.parse_args()

    max_steps = 1500 if args.split in ["large", "giant"] else 1000
    agent, env, train_ds, _, cfg = load_pretrained_agent(args.checkpoint_dir, args.split, seed=args.seeds[0], max_episode_steps=max_steps)
    evaluator = ZeroShotEvaluator(env, agent, train_ds, cfg, env_name=f"ogbench-antmaze-{args.split}-navigate-v0", max_episode_steps=max_steps)
    n_landmarks = 2000 if args.split == "large" else 1000

    planners = _init_planners(agent, train_ds["observations"], args.split, n_landmarks)
    df_traj, df_sg = _collect_rollouts(evaluator, planners, args.seeds, args.num_tasks, args.episodes_per_task)
    _export_results(df_traj, df_sg, args.split, args.output_base_dir)


if __name__ == "__main__":
    main()
