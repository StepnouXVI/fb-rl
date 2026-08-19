import os, sys, json, hydra
# Set single-thread flags before loading numerical libraries to avoid thread thrashing
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import multiprocessing as mp
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
        planner = BaselinePlanner(agent, dataset_states=train_ds["observations"], use_high_actor=True, name=method_name)
    elif planner_type == "recursive_bisection":
        planner = RecursiveBisectionPlanner(agent, train_ds["observations"], max_depth=2, n_candidates=200, hit_threshold=35.0, name=method_name)
    elif planner_type == "buffer_graph":
        planner = BufferGraphPlanner(
            agent,
            dataset_states=train_ds["observations"],
            name=method_name,
        )
    elif planner_type == "distilled_mlp":
        distilled_ckpt = os.path.join("results", f"distilled_mlp_{split}.pt")
        planner = DistilledMLPPlanner(
            agent,
            checkpoint_path=distilled_ckpt if os.path.exists(distilled_ckpt) else None,
            device="cpu",
            name=method_name,
        )
    else:
        raise ValueError(f"Unknown planner type: {planner_type}")

    summary = evaluator.evaluate_all_tasks(
        planner, num_episodes=num_episodes, eval_temperature=eval_temperature, seed=seed
    )
    return method_name, seed, summary


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    print(f"=== Ultra-Fast Multi-Subgoal FB Planning Parallel Benchmark on {cfg.env.name} ===")
    print(f"Seeds: {list(cfg.eval.seeds)} | Episodes per task: {cfg.eval.num_episodes} | Workers: {cfg.eval.n_workers}")
    os.makedirs(cfg.eval.output_dir, exist_ok=True)

    # Save sampled landmark coordinates for visualization
    _, _, sample_ds, _, _ = load_pretrained_agent(str(cfg.eval.checkpoint_dir), str(cfg.env.split), seed=0)
    rng = np.random.default_rng(42)
    lm_idxs = rng.choice(len(sample_ds["observations"]), size=1000, replace=False)
    df_landmarks = pd.DataFrame({
        "landmark_id": range(1000),
        "x": sample_ds["observations"][lm_idxs, 0],
        "y": sample_ds["observations"][lm_idxs, 1],
    })
    df_landmarks.to_csv(os.path.join(cfg.eval.output_dir, "landmarks.csv"), index=False)

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
    all_trajectories = []
    all_subgoals = []

    ctx = mp.get_context("spawn")
    n_workers = min(int(cfg.eval.n_workers), len(jobs))
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as executor:
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
                "self_intersections": summary.get("self_intersections", 0.0),
            })
            if "trajectory_records" in summary:
                all_trajectories.extend(summary["trajectory_records"])
            if "subgoal_records" in summary:
                all_subgoals.extend(summary["subgoal_records"])

            pbar.write(f"[{method_name}] Seed {seed}: Success = {summary['success_rate']:.1f}%, Steps = {summary['mean_length']:.1f}, Loops/Crossings = {summary.get('self_intersections', 0.0):.1f}, Latency = {summary['latency_ms']:.2f} ms")

    # Aggregate statistics
    aggregated = aggregate_runs(results_by_method, baseline_name="Single-Intention Baseline")

    metrics_path = os.path.join(cfg.eval.output_dir, "summary_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(aggregated, f, indent=2)

    df_runs = pd.DataFrame(all_rows)
    df_runs.to_csv(os.path.join(cfg.eval.output_dir, "summary_runs.csv"), index=False)

    if all_trajectories:
        df_traj = pd.DataFrame(all_trajectories)
        df_traj.to_csv(os.path.join(cfg.eval.output_dir, "trajectories.csv"), index=False)
        print(f"Saved {len(df_traj)} trajectory steps to {os.path.join(cfg.eval.output_dir, 'trajectories.csv')}")

    if all_subgoals:
        df_sg = pd.DataFrame(all_subgoals)
        df_sg.to_csv(os.path.join(cfg.eval.output_dir, "subgoals.csv"), index=False)
        print(f"Saved {len(df_sg)} subgoal records to {os.path.join(cfg.eval.output_dir, 'subgoals.csv')}")

    latex_str = export_latex_table(aggregated)
    with open(os.path.join(cfg.eval.output_dir, "summary_table.tex"), "w") as f:
        f.write(latex_str)

    print("\n=== Parallel Benchmark Completed Successfully! ===")
    print(f"Summary metrics saved to {metrics_path}")
    print(f"Summary table saved to {os.path.join(cfg.eval.output_dir, 'summary_table.tex')}")

if __name__ == "__main__":
    main()
