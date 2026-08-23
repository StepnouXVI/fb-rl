import os
import sys
import json
import multiprocessing as mp
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.agent_loader import load_pretrained_agent
from src.planners import (
    BaselinePlanner,
    RecursiveBisectionPlanner,
    BufferGraphPlanner,
    DistilledMLPPlanner,
)
from src.evaluator import ZeroShotEvaluator
from src.metrics import aggregate_runs, export_latex_table


def _build_planner_for_worker(p_type, agent, train_obs, split, name, cfg):
    if p_type == "baseline":
        return BaselinePlanner(agent, dataset_states=train_obs, use_high_actor=cfg.get("use_high_actor", True), name=name)
    if p_type == "recursive_bisection":
        return RecursiveBisectionPlanner(agent, train_obs, max_depth=cfg.get("max_depth", 2), n_candidates=cfg.get("n_candidates", 200), hit_threshold=cfg.get("hit_threshold", 35.0), name=name)
    if p_type == "buffer_graph":
        return BufferGraphPlanner(agent, train_obs, n_landmarks=cfg.get("n_landmarks", 1000), max_edge_radius=cfg.get("max_edge_radius", 3.5), reachability_cutoff=cfg.get("reachability_cutoff", 35.0), lookahead_dist=cfg.get("lookahead_dist", 2.6), name=name)
    if p_type == "distilled_mlp":
        ckpt = os.path.join("results", "checkpoints", f"distilled_mlp_{split}.pt")
        if not os.path.exists(ckpt):
            ckpt = os.path.join("results", f"distilled_mlp_{split}.pt")
        return DistilledMLPPlanner(agent, checkpoint_path=ckpt if os.path.exists(ckpt) else None, device="cpu", name=name)
    raise ValueError(f"Unknown planner type: {p_type}")


def _evaluate_worker(args):
    name, p_type, seed, ckpt_dir, split, env_name, n_eps, temp, dev, max_steps, cfg = args
    agent, env, train_ds, _, fb_cfg = load_pretrained_agent(ckpt_dir, split, seed=seed, max_episode_steps=max_steps)
    evaluator = ZeroShotEvaluator(env, agent, train_ds, fb_cfg, env_name=env_name, max_episode_steps=max_steps)
    planner = _build_planner_for_worker(p_type, agent, train_ds["observations"], split, name, cfg)
    summary = evaluator.evaluate_all_tasks(planner, num_episodes=n_eps, eval_temperature=temp, seed=seed, max_episode_steps=max_steps)
    return name, seed, summary


def _save_landmark_reference(ckpt_dir, split, output_dir, n_landmarks):
    _, _, sample_ds, _, _ = load_pretrained_agent(ckpt_dir, split, seed=0)
    idxs = np.random.default_rng(42).choice(len(sample_ds["observations"]), size=min(n_landmarks, len(sample_ds["observations"])), replace=False)
    bench_dir = os.path.join(output_dir, "benchmarks") if not output_dir.endswith("benchmarks") else output_dir
    os.makedirs(bench_dir, exist_ok=True)
    pd.DataFrame({"landmark_id": range(len(idxs)), "x": sample_ds["observations"][idxs, 0], "y": sample_ds["observations"][idxs, 1]}).to_csv(os.path.join(bench_dir, "landmarks.csv"), index=False)


def _gather_results(jobs, n_workers):
    results_by_method, all_rows, all_trajs, all_sgs = defaultdict(list), [], [], []
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=min(n_workers, len(jobs)), mp_context=ctx) as executor:
        futures = {executor.submit(_evaluate_worker, j): j for j in jobs}
        for fut in tqdm(as_completed(futures), total=len(jobs), desc="Benchmark Progress"):
            name, seed, summ = fut.result()
            results_by_method[name].append(summ)
            all_rows.append({"method": name, "seed": seed, "success_rate": summ["success_rate"], "mean_length": summ["mean_length"], "latency_ms": summ["latency_ms"], "self_intersections": summ.get("self_intersections", 0.0)})
            if "trajectory_records" in summ:
                all_trajs.extend(summ["trajectory_records"])
            if "subgoal_records" in summ:
                all_sgs.extend(summ["subgoal_records"])
    return results_by_method, all_rows, all_trajs, all_sgs


def _save_benchmark_artifacts(aggregated, all_rows, all_trajs, all_sgs, output_dir):
    bench_dir = os.path.join(output_dir, "benchmarks") if not output_dir.endswith("benchmarks") else output_dir
    data_dir = os.path.join(output_dir, "data") if not output_dir.endswith("benchmarks") else os.path.join(os.path.dirname(output_dir), "data")
    os.makedirs(bench_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    with open(os.path.join(bench_dir, "summary_metrics.json"), "w") as f:
        json.dump(aggregated, f, indent=2)
    pd.DataFrame(all_rows).to_csv(os.path.join(bench_dir, "summary_runs.csv"), index=False)
    if all_trajs:
        pd.DataFrame(all_trajs).to_csv(os.path.join(data_dir, "trajectories.csv"), index=False)
    if all_sgs:
        pd.DataFrame(all_sgs).to_csv(os.path.join(data_dir, "subgoals.csv"), index=False)
    with open(os.path.join(bench_dir, "summary_table.tex"), "w") as f:
        f.write(export_latex_table(aggregated))


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    p_cfg = OmegaConf.to_container(cfg.planner, resolve=True) if hasattr(cfg, "planner") else {}
    _save_landmark_reference(str(cfg.eval.checkpoint_dir), str(cfg.env.split), str(cfg.eval.output_dir), int(p_cfg.get("n_landmarks", 1000)))

    pmap = {"Single-Intention Baseline": "baseline", "Recursive Bisection (Branch 1)": "recursive_bisection", "Buffer Graph Dijkstra (Branch 2)": "buffer_graph", "Distilled Latent Policy (Branch 3)": "distilled_mlp"}
    targets = {"Single-Intention Baseline": "baseline", "Buffer Graph Dijkstra (Branch 2)": "buffer_graph"} if cfg.planner.type == "baseline_vs_dijkstra" else ({cfg.planner.name: pmap[cfg.planner.name]} if cfg.planner.type != "all" and cfg.planner.name in pmap else pmap)
    max_steps = int(cfg.env.max_episode_steps) if hasattr(cfg.env, "max_episode_steps") and cfg.env.max_episode_steps is not None else None

    jobs = [(m, p_type, int(s), str(cfg.eval.checkpoint_dir), str(cfg.env.split), str(cfg.env.name), int(cfg.eval.num_episodes), float(cfg.eval.eval_temperature), str(cfg.eval.device), max_steps, p_cfg) for m, p_type in targets.items() for s in cfg.eval.seeds]
    results_by_method, all_rows, all_trajs, all_sgs = _gather_results(jobs, int(cfg.eval.n_workers))
    _save_benchmark_artifacts(aggregate_runs(results_by_method, baseline_name="Single-Intention Baseline"), all_rows, all_trajs, all_sgs, str(cfg.eval.output_dir))


if __name__ == "__main__":
    main()
