"""
Benchmark Runner: Evaluates Baseline + 3 Multi-Subgoal Planners across multiple seeds on OGBench.
"""

import os
import argparse
from collections import defaultdict
import pandas as pd

from fb_core.fb_model_wrapper import FBModelWrapper
from fb_core.dataset_sampler import DatasetSampler
from fb_core.evaluator import EpisodeEvaluator
from fb_core.metrics_collector import MetricsCollector

from planners.baseline_planner import SingleIntentionPlanner
from planners.recursive_bisection import RecursiveBisectionPlanner
from planners.buffer_graph import BufferGraphPlanner
from planners.distilled_mlp import DistilledMLPPlanner


def run_benchmarks(
    env_name: str = "antmaze-medium-navigate-v0",
    seeds: list = [0, 1, 2, 3, 4],
    num_episodes: int = 15,
    output_dir: str = "results",
):
    os.makedirs(output_dir, exist_ok=True)
    clean_env_name = env_name.replace("ogbench-", "")
    print(f"=== Starting Multi-Subgoal Planning Benchmark on {clean_env_name} ===")
    print(f"Seeds: {seeds} | Episodes per seed: {num_episodes}")

    # Load dataset and environment
    dataset_sampler = DatasetSampler.from_ogbench(env_name=clean_env_name, max_samples=5000, seed=42)
    fb_model = FBModelWrapper(agent=None, latent_dim=128)
    evaluator = EpisodeEvaluator(env_name=clean_env_name, max_episode_steps=600)

    # Try loading real OGBench env
    try:
        import ogbench
        env, _, _ = ogbench.make_env_and_datasets(clean_env_name)
        print(f"[OK] Successfully initialized OGBench environment: {clean_env_name}")
    except Exception as e:
        print(f"[Notice] Using synthetic maze environment for benchmark ({e}).")
        env = None

    # Instantiate Planners
    planners = {
        "Baseline (Single-Intention)": SingleIntentionPlanner(fb_model=fb_model),
        "Branch 1: Recursive Bisection": RecursiveBisectionPlanner(
            fb_model=fb_model,
            dataset_sampler=dataset_sampler,
            max_depth=2,
            n_candidates=100,
        ),
        "Branch 2: Buffer Graph (Dijkstra)": BufferGraphPlanner(
            fb_model=fb_model,
            dataset_sampler=dataset_sampler,
            n_landmarks=80,
        ),
        "Branch 3: Distilled Latent MLP": DistilledMLPPlanner(
            fb_model=fb_model,
            dataset_sampler=dataset_sampler,
            device="cpu",
        ),
    }

    # Pre-train distilled MLP planner
    print("\n--- Training Distilled MLP Planner (Branch 3) ---")
    planners["Branch 3: Distilled Latent MLP"].train_distillation(
        graph_planner=planners["Branch 2: Buffer Graph (Dijkstra)"],
        n_pairs=300,
        epochs=15,
    )

    results_by_method = defaultdict(list)
    all_rows = []

    for name, planner in planners.items():
        slug = name.lower().replace(" ", "_").replace(":", "").replace("(", "").replace(")", "").replace("-", "_")
        method_dir = os.path.join(output_dir, slug)
        os.makedirs(method_dir, exist_ok=True)
        print(f"\nEvaluating {name}...")

        for seed in seeds:
            res = evaluator.evaluate_planner(
                planner=planner,
                fb_model=fb_model,
                env=env,
                num_episodes=num_episodes,
                seed=seed,
            )
            results_by_method[name].append(res)
            
            # Save single seed CSV
            seed_df = pd.DataFrame({
                "episode": list(range(num_episodes)),
                "success": res["episode_successes"],
                "steps": res["episode_steps"],
            })
            seed_df.to_csv(os.path.join(method_dir, f"eval_seed{seed}.csv"), index=False)

            all_rows.append({
                "method": name,
                "seed": seed,
                "success_rate": res["success_rate"],
                "mean_steps": res["mean_steps"],
                "mean_latency_ms": res["mean_latency_ms"],
            })
            print(f"  Seed {seed}: Success Rate = {res['success_rate']:.1f}%, Steps = {res['mean_steps']:.1f}, Latency = {res['mean_latency_ms']:.2f} ms/step")

    # Aggregate metrics
    summary = MetricsCollector.aggregate_runs(results_by_method, baseline_name="Baseline (Single-Intention)")
    MetricsCollector.export_summary_to_json(summary, os.path.join(output_dir, "summary_metrics.json"))

    # Save summary dataframe
    df_all = pd.DataFrame(all_rows)
    df_all.to_csv(os.path.join(output_dir, "summary_runs.csv"), index=False)

    # Save LaTeX table
    latex_table = MetricsCollector.generate_latex_table(summary)
    with open(os.path.join(output_dir, "summary_table.tex"), "w") as f:
        f.write(latex_table)

    print("\n=== Benchmark Completed Successfully! ===")
    print(f"Summary metrics saved to {os.path.join(output_dir, 'summary_metrics.json')}")
    print(f"LaTeX table saved to {os.path.join(output_dir, 'summary_table.tex')}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="ogbench-antmaze-medium-navigate-v0")
    parser.add_argument("--seeds", type=str, default="0,1,2,3,4")
    parser.add_argument("--episodes", type=int, default=15)
    parser.add_argument("--output_dir", type=str, default="results")
    args = parser.parse_args()

    seed_list = [int(s) for s in args.seeds.split(",")]
    run_benchmarks(
        env_name=args.env_name,
        seeds=seed_list,
        num_episodes=args.episodes,
        output_dir=args.output_dir,
    )
