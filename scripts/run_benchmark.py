import os, sys, json, hydra
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
from tqdm import tqdm
from omegaconf import DictConfig
from src.agent_loader import load_pretrained_agent
from src.planners import (
    BaselinePlanner,
    RecursiveBisectionPlanner,
    BufferGraphPlanner,
    DistilledMLPPlanner,
)
from src.evaluator import ZeroShotEvaluator
from src.metrics import aggregate_runs, export_latex_table

# ponytail: Top-level worker function for parallel multi-process evaluation
def _evaluate_worker(args):
    method_name, planner_type, seed, checkpoint_dir, split, env_name, num_episodes, eval_temperature, device = args
    agent, env, train_ds, _, fb_cfg = load_pretrained_agent(checkpoint_dir, split, seed=seed)
    evaluator = ZeroShotEvaluator(env, agent, train_ds, fb_cfg, env_name=env_name)

    if planner_type == "baseline":
        planner = BaselinePlanner(agent, use_high_actor=True, name=method_name)
    elif planner_type == "recursive_bisection":
        planner = RecursiveBisectionPlanner(agent, train_ds["observations"], max_depth=2, n_candidates=200, hit_threshold=50.0, name=method_name)
    elif planner_type == "buffer_graph":
        planner = BufferGraphPlanner(agent, train_ds["observations"], n_landmarks=300, reachability_cutoff=20.0, hit_threshold=50.0, name=method_name)
    elif planner_type == "distilled_mlp":
        distilled_ckpt = os.path.join("results", f"distilled_mlp_{split}.pt")
        planner = DistilledMLPPlanner(
            agent,
            checkpoint_path=distilled_ckpt if os.path.exists(distilled_ckpt) else None,
            device=device,
            name=method_name,
        )
    else:
        raise ValueError(f"Unknown planner type: {planner_type}")

    np.random.seed(seed)
    summary = evaluator.evaluate_all_tasks(
        planner, num_episodes=num_episodes, eval_temperature=eval_temperature
    )
    return method_name, seed, summary


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    print(f"=== Multi-Subgoal FB Planning Parallel Benchmark on {cfg.env.name} ===")
    print(f"Seeds: {list(cfg.eval.seeds)} | Episodes per task: {cfg.eval.num_episodes} | Workers: {cfg.eval.n_workers}")
    os.makedirs(cfg.eval.output_dir, exist_ok=True)

    planner_map = {
        "Single-Intention Baseline": "baseline",
        "Recursive Bisection (Branch 1)": "recursive_bisection",
        "Buffer Graph Dijkstra (Branch 2)": "buffer_graph",
        "Distilled Latent Policy (Branch 3)": "distilled_mlp",
    }

    if cfg.planner.type == "baseline_vs_dijkstra":
        target_planners = {
            "Single-Intention Baseline": "baseline",
            "Buffer Graph Dijkstra (Branch 2)": "buffer_graph",
        }
    elif cfg.planner.type != "all" and cfg.planner.name in planner_map:
        target_planners = {cfg.planner.name: planner_map[cfg.planner.name]}
    else:
        target_planners = planner_map

    # Prepare job queue
    jobs = []
    for method_name, planner_type in target_planners.items():
        for seed in cfg.eval.seeds:
            jobs.append((
                method_name,
                planner_type,
                int(seed),
                str(cfg.eval.checkpoint_dir),
                str(cfg.env.split),
                str(cfg.env.name),
                int(cfg.eval.num_episodes),
                float(cfg.eval.eval_temperature),
                str(cfg.eval.device),
            ))

    print(f"Total parallel jobs to execute: {len(jobs)}")
    results_by_method = defaultdict(list)
    all_rows = []

    n_workers = min(int(cfg.eval.n_workers), len(jobs))
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_evaluate_worker, job): job for job in jobs}
        pbar = tqdm(as_completed(futures), total=len(jobs), desc="Parallel Benchmark Progress")
        for fut in pbar:
            method_name, seed, summary = fut.result()
            results_by_method[method_name].append(summary)
            all_rows.append({
                "method": method_name,
                "seed": seed,
                "success_rate": summary["success_rate"],
                "mean_length": summary["mean_length"],
                "latency_ms": summary["latency_ms"],
            })
            pbar.write(f"[{method_name}] Seed {seed}: Success = {summary['success_rate']:.1f}%, Steps = {summary['mean_length']:.1f}, Latency = {summary['latency_ms']:.2f} ms")

    # Aggregate statistics
    aggregated = aggregate_runs(results_by_method, baseline_name="Single-Intention Baseline")

    metrics_path = os.path.join(cfg.eval.output_dir, "summary_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(aggregated, f, indent=2)

    df_runs = pd.DataFrame(all_rows)
    df_runs.to_csv(os.path.join(cfg.eval.output_dir, "summary_runs.csv"), index=False)

    latex_str = export_latex_table(aggregated)
    with open(os.path.join(cfg.eval.output_dir, "summary_table.tex"), "w") as f:
        f.write(latex_str)

    print("\n=== Parallel Benchmark Completed Successfully! ===")
    print(f"Summary metrics saved to {metrics_path}")
    print(f"Summary table saved to {os.path.join(cfg.eval.output_dir, 'summary_table.tex')}")

if __name__ == "__main__":
    main()
