#!/usr/bin/env bash
#!/usr/bin/env python3
import os
import sys
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict
from omegaconf import OmegaConf

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.agent_loader import load_pretrained_agent
from src.evaluator import ZeroShotEvaluator
from src.planners import (
    BaselinePlanner,
    RecursiveBisectionPlanner,
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


def _load_env_config(split):
    cfg_file = os.path.join(PROJECT_ROOT, "configs", "env", f"antmaze_{split}.yaml")
    if os.path.exists(cfg_file):
        try:
            return OmegaConf.load(cfg_file)
        except Exception:
            pass
    return OmegaConf.create({
        "name": f"antmaze-{split}-navigate-v0", "split": split,
        "max_episode_steps": 1500 if split in ["large", "giant"] else 1000,
        "planner": {
            "n_landmarks": 2000 if split == "large" else (3000 if split == "giant" else 1000),
            "lookahead_dist": 2.6, "max_edge_radius": 3.5, "reachability_cutoff": 35.0,
            "recursive_bisection": {"max_depth": 2 if split == "medium" else 3, "n_candidates": 200 if split == "medium" else 400, "hit_threshold": 35.0},
            "single_wp": {"hidden_dim": 384, "n_layers": 4},
            "enhanced_seq_attn": {"hidden_dim": 384, "num_heads": 6, "n_layers": 4},
        }
    })


def _init_candidate_planners(agent, train_obs, split, env_cfg=None):
    dirs = [os.path.join(PROJECT_ROOT, "results", "checkpoints"), os.path.join(PROJECT_ROOT, "outputs", "checkpoints"), PROJECT_ROOT]
    p_cfg = env_cfg.planner if env_cfg and hasattr(env_cfg, "planner") else {}
    n_landmarks = int(p_cfg.get("n_landmarks", 2000 if split == "large" else 1000))
    lookahead = float(p_cfg.get("lookahead_dist", 2.6))
    max_radius = float(p_cfg.get("max_edge_radius", 3.5))
    reach_cutoff = float(p_cfg.get("reachability_cutoff", 35.0))

    rb_cfg = p_cfg.get("recursive_bisection", {})
    rb_depth = int(rb_cfg.get("max_depth", 2 if split == "medium" else 3))
    rb_cand = int(rb_cfg.get("n_candidates", 200 if split == "medium" else 400))
    rb_hit = float(rb_cfg.get("hit_threshold", 35.0))

    sw_cfg = p_cfg.get("single_wp", {})
    esa_cfg = p_cfg.get("enhanced_seq_attn", {})

    enhanced_ckpt = _find_checkpoint(f"best_enhanced_sequence_attn_{split}.pkl", dirs) or _find_checkpoint(f"best_enhanced_seq_attn_{split}.pkl", dirs)
    single_wp_ckpt = _find_checkpoint(f"best_single_wp_{split}.pkl", dirs)
    jax_ckpt = _find_checkpoint(f"distilled_jax_gated_attn_{split}.pkl", dirs)
    if not enhanced_ckpt or not single_wp_ckpt or not jax_ckpt:
        raise FileNotFoundError(f"Missing required checkpoints in {dirs} for split {split}!")

    return [
        EnhancedSequenceWaypointAttentionPlanner(agent, train_obs, enhanced_ckpt, n_landmarks, hidden_dim=int(esa_cfg.get("hidden_dim", 384)), num_heads=int(esa_cfg.get("num_heads", 6)), n_layers=int(esa_cfg.get("n_layers", 4)), max_edge_radius=max_radius, reachability_cutoff=reach_cutoff, lookahead_dist=lookahead, name="1. Dijkstra + Enhanced Sequence Attention"),
        BaselinePlanner(agent, dataset_states=train_obs, name="2. Single-Intention Baseline"),
        RecursiveBisectionPlanner(agent, dataset_states=train_obs, max_depth=rb_depth, n_candidates=rb_cand, hit_threshold=rb_hit, name="3. Recursive Bisection Planner"),
        WaypointTranslatorPlanner(agent, train_obs, single_wp_ckpt, n_landmarks, hidden_dim=int(sw_cfg.get("hidden_dim", 384)), n_layers=int(sw_cfg.get("n_layers", 4)), max_edge_radius=max_radius, reachability_cutoff=reach_cutoff, lookahead_dist=lookahead, name="4. Dijkstra + Single WP Translator"),
        BufferGraphPlanner(agent, train_obs, n_landmarks=n_landmarks, max_edge_radius=max_radius, reachability_cutoff=reach_cutoff, lookahead_dist=lookahead, name="5. Dijkstra Teacher (high_actor)"),
        DistilledJAXPlanner(agent, checkpoint_path=jax_ckpt, model_type="gated_attn", name="6. Distilled JAX GatedAttn [O(1)]"),
    ]


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


def _save_summary_tables(rows, output_dir, split, n_seeds=10):
    df = pd.DataFrame(rows)
    os.makedirs(output_dir, exist_ok=True)
    csv_p, md_p = os.path.join(output_dir, f"benchmark_summary_{n_seeds}seeds_{split}.csv"), os.path.join(output_dir, f"benchmark_summary_{n_seeds}seeds_{split}.md")
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


def run_comprehensive_benchmark(checkpoint_dir="fb-test", split="medium", num_tasks=5, episodes_per_task=10, seeds=None, output_dir="results/benchmarks", **overrides):
    seeds = seeds or list(range(1, 11))
    env_cfg = _load_env_config(split)
    max_steps = overrides.get("max_episode_steps") or getattr(env_cfg, "max_episode_steps", 1500 if split in ["large", "giant"] else 1000)
    agent, env, train_ds, _, cfg = load_pretrained_agent(checkpoint_dir, split, seed=seeds[0], max_episode_steps=max_steps)
    evaluator = ZeroShotEvaluator(env, agent, train_ds, cfg, env_name=f"ogbench-antmaze-{split}-navigate-v0", max_episode_steps=max_steps)
    planners = _init_candidate_planners(agent, train_ds["observations"], split, env_cfg)
    rows = [_eval_single_planner(evaluator, p, seeds, num_tasks, episodes_per_task) for p in planners]
    return _save_summary_tables(rows, output_dir, split, len(seeds))


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
