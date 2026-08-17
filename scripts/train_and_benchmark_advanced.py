"""
Train and Benchmark Script for Advanced Neural Architectures.
Generates 3,000+ Dijkstra optimal transitions, trains ResNet-ECA, Gated Cross-Attention, Dense-ECA,
and evaluates them across 10 random seeds on antmaze-medium-navigate-v0.
"""

import os
import time
import json
import numpy as np
import pandas as pd
import torch

from fb_core.fb_model_wrapper import FBModelWrapper
from fb_core.dataset_sampler import DatasetSampler
from fb_core.evaluator import EpisodeEvaluator
from fb_core.metrics_collector import MetricsCollector

from planners.baseline_planner import SingleIntentionPlanner
from planners.recursive_bisection import RecursiveBisectionPlanner
from planners.buffer_graph import BufferGraphPlanner
from planners.advanced_distilled_models import AdvancedDistilledPlanner


def generate_dijkstra_dataset(
    graph_planner: BufferGraphPlanner,
    dataset_sampler: DatasetSampler,
    fb_model: FBModelWrapper,
    n_pairs: int = 3500,
    seed: int = 42,
) -> dict:
    """Sample state-goal pairs and compute Dijkstra teacher optimal waypoints."""
    print(f"\n[Dataset Generation] Sampling {n_pairs} state-goal pairs with Dijkstra teacher (N={graph_planner.n_landmarks} landmarks)...")
    np.random.seed(seed)
    states = dataset_sampler.sample_candidates(n_pairs * 2)
    start_states = states[:n_pairs]
    goal_states = states[n_pairs:]

    inputs_obs = []
    inputs_zg = []
    targets_zw = []

    t0 = time.perf_counter()
    for i in range(n_pairs):
        s = start_states[i]
        g = goal_states[i]
        z_g = fb_model.encode_backward(g)

        waypoints = graph_planner._find_shortest_path(s, g)
        target_waypoint = waypoints[1] if len(waypoints) > 1 else g
        z_w = fb_model.encode_backward(target_waypoint)

        inputs_obs.append(s)
        inputs_zg.append(z_g)
        targets_zw.append(z_w)

        if (i + 1) % 1000 == 0 or (i + 1) == n_pairs:
            print(f"  Generated {i + 1}/{n_pairs} pairs ({time.perf_counter() - t0:.1f}s)...")

    return {
        "obs": np.asarray(inputs_obs, dtype=np.float32),
        "z_goal": np.asarray(inputs_zg, dtype=np.float32),
        "z_waypoint": np.asarray(targets_zw, dtype=np.float32),
    }


def run_comprehensive_experiment(
    env_name: str = "antmaze-medium-navigate-v0",
    n_pairs: int = 3500,
    val_ratio: float = 0.2,
    epochs: int = 30,
    eval_seeds: list = list(range(10)),
    num_episodes: int = 15,
    output_dir: str = "results/advanced_benchmarks",
):
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    os.makedirs(output_dir, exist_ok=True)
    clean_env_name = env_name.replace("ogbench-", "")
    print(f"================================================================================")
    print(f"Starting Advanced Neural Distillation & Benchmark Experiment")
    print(f"Environment: {clean_env_name}")
    print(f"Distillation Dataset: {n_pairs} pairs (Train: {int(n_pairs*(1-val_ratio))}, Val: {int(n_pairs*val_ratio)}) | Epochs: {epochs}")
    print(f"Evaluation Seeds: {eval_seeds} ({len(eval_seeds)} seeds, {num_episodes} episodes/seed)")
    print(f"================================================================================")

    # Initialize environment, model and sampler
    dataset_sampler = DatasetSampler.from_ogbench(env_name=clean_env_name, max_samples=10000, seed=42)
    fb_model = FBModelWrapper(agent=None, latent_dim=128)
    evaluator = EpisodeEvaluator(env_name=clean_env_name, max_episode_steps=600)

    try:
        import ogbench
        env, _, _ = ogbench.make_env_and_datasets(clean_env_name)
    except Exception:
        env = None

    # Teacher graph planner
    teacher_graph = BufferGraphPlanner(
        fb_model=fb_model,
        dataset_sampler=dataset_sampler,
        n_landmarks=200,
    )

    # Generate or load dataset
    dataset_cache_path = os.path.join(output_dir, f"distill_dataset_{n_pairs}pairs.npz")
    if os.path.exists(dataset_cache_path):
        print(f"[Cache] Loading pre-generated dataset from {dataset_cache_path}")
        npz = np.load(dataset_cache_path)
        distill_dataset = {
            "obs": npz["obs"],
            "z_goal": npz["z_goal"],
            "z_waypoint": npz["z_waypoint"],
        }
    else:
        distill_dataset = generate_dijkstra_dataset(
            graph_planner=teacher_graph,
            dataset_sampler=dataset_sampler,
            fb_model=fb_model,
            n_pairs=n_pairs,
            seed=42,
        )
        np.savez_compressed(
            dataset_cache_path,
            obs=distill_dataset["obs"],
            z_goal=distill_dataset["z_goal"],
            z_waypoint=distill_dataset["z_waypoint"],
        )
        print(f"[Saved] Dataset cached to {dataset_cache_path}")

    # Define all models to compare
    models_to_train = {
        "Standard-MLP": {
            "arch": "standard_mlp",
            "hidden_dim": 256,
            "num_blocks": 2,
            "desc": "2-Layer Baseline MLP (256 units, LayerNorm, GELU)",
        },
        "ResNet-ECA": {
            "arch": "resnet_eca",
            "hidden_dim": 256,
            "num_blocks": 4,
            "desc": "4-Block Deep ResNet with 1D Efficient Channel Attention (ECA-1D)",
        },
        "Gated-CrossAttn": {
            "arch": "gated_cross_attn",
            "hidden_dim": 256,
            "num_blocks": 3,
            "desc": "3-Layer Gated Cross-Attention (State queries x Goal keys/values) with SwiGLU",
        },
        "Dense-ECA": {
            "arch": "dense_eca",
            "hidden_dim": 256,
            "num_blocks": 4,
            "desc": "Densely Connected ResNet with ECA-1D (Growth Rate = 64)",
        },
    }

    trained_planners = {}
    training_results = {}

    for model_name, cfg in models_to_train.items():
        print(f"\n--------------------------------------------------------------------------------")
        print(f"Training: {model_name} ({cfg['desc']})")
        print(f"--------------------------------------------------------------------------------")

        planner = AdvancedDistilledPlanner(
            fb_model=fb_model,
            dataset_sampler=dataset_sampler,
            architecture_type=cfg["arch"],
            name=model_name,
            obs_dim=29,
            latent_dim=128,
            hidden_dim=cfg["hidden_dim"],
            num_blocks=cfg["num_blocks"],
            device="cpu",
        )

        t_start = time.perf_counter()
        train_stats = planner.train_distillation(
            dataset=distill_dataset,
            val_ratio=val_ratio,
            epochs=epochs,
            batch_size=64,
            lr=1e-3,
            weight_decay=1e-4,
        )
        t_duration = time.perf_counter() - t_start

        param_count = planner.count_parameters()
        flops = planner.estimate_flops()

        train_stats["param_count"] = param_count
        train_stats["flops"] = flops
        train_stats["training_time_sec"] = t_duration
        train_stats["history"] = planner.training_history

        training_results[model_name] = train_stats
        trained_planners[model_name] = planner

        print(f"  Completed in {t_duration:.1f}s | Params: {param_count:,} | FLOPs: {flops:,}")
        print(f"  Final Train Loss: {train_stats['final_train_loss']:.4f} | Val Loss: {train_stats['final_val_loss']:.4f} | Val Cosine Sim: {train_stats['final_val_cosine_sim']:.4f} | Val MSE: {train_stats['final_val_mse']:.4f}")

    # Add Non-distilled Baselines for Full Benchmark Comparison
    all_eval_planners = {
        "Single-Intention Baseline": SingleIntentionPlanner(fb_model=fb_model),
        "Branch 1: Bisection": RecursiveBisectionPlanner(
            fb_model=fb_model,
            dataset_sampler=dataset_sampler,
            max_depth=2,
            n_candidates=100,
        ),
        "Branch 2: Buffer Graph (Dijkstra Teacher)": teacher_graph,
        **trained_planners,
    }

    # Run Rollouts across 10 seeds
    print(f"\n================================================================================")
    print(f"Executing Multi-Seed Rollout Benchmark (10 Seeds x {num_episodes} Episodes)")
    print(f"================================================================================")

    results_by_method = {}
    detailed_runs = []

    for name, planner in all_eval_planners.items():
        print(f"\nEvaluating Planner: {name}...")
        method_runs = []
        seed_successes, seed_steps, seed_latencies = [], [], []

        for seed in eval_seeds:
            res = evaluator.evaluate_planner(
                planner=planner,
                fb_model=fb_model,
                env=env,
                num_episodes=num_episodes,
                seed=seed,
            )
            method_runs.append(res)
            seed_successes.append(res["success_rate"])
            seed_steps.append(res["mean_steps"])
            seed_latencies.append(res["mean_latency_ms"])

            detailed_runs.append({
                "method": name,
                "seed": seed,
                "success_rate": res["success_rate"],
                "mean_steps": res["mean_steps"],
                "mean_latency_ms": res["mean_latency_ms"],
            })

            print(f"  Seed {seed:2d}: Success = {res['success_rate']:5.1f}% | Steps = {res['mean_steps']:5.1f} | Latency = {res['mean_latency_ms']:.3f} ms")

        results_by_method[name] = method_runs
        sr_ci = MetricsCollector.compute_bootstrap_ci(seed_successes)
        st_ci = MetricsCollector.compute_bootstrap_ci(seed_steps)
        lat_ci = MetricsCollector.compute_bootstrap_ci(seed_latencies)
        print(f"  --> {name} Aggregate: Success = {sr_ci['mean']:.1f} +/- {sr_ci['std']:.1f}% | Steps = {st_ci['mean']:.1f} +/- {st_ci['std']:.1f} | Latency = {lat_ci['mean']:.3f} ms")

    # Aggregate full summary
    summary = MetricsCollector.aggregate_runs(results_by_method, baseline_name="Single-Intention Baseline")

    # Attach training metadata for neural models
    for model_name, t_res in training_results.items():
        if model_name in summary:
            summary[model_name]["training_metrics"] = {
                "params": t_res["param_count"],
                "flops": t_res["flops"],
                "train_loss": t_res["final_train_loss"],
                "val_loss": t_res["final_val_loss"],
                "val_cosine_sim": t_res["final_val_cosine_sim"],
                "val_mse": t_res["final_val_mse"],
                "training_time_sec": t_res["training_time_sec"],
            }

    # Save outputs
    with open(os.path.join(output_dir, "advanced_summary_metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(output_dir, "training_histories.json"), "w") as f:
        # serialize training histories
        hist_serializable = {
            m: {
                "train_loss": tr["history"]["train_loss"],
                "val_loss": tr["history"]["val_loss"],
                "val_cosine_sim": tr["history"]["val_cosine_sim"],
                "val_mse": tr["history"]["val_mse"],
                "learning_rates": tr["history"]["learning_rates"],
            }
            for m, tr in training_results.items()
        }
        json.dump(hist_serializable, f, indent=2)

    df_runs = pd.DataFrame(detailed_runs)
    df_runs.to_csv(os.path.join(output_dir, "all_eval_runs.csv"), index=False)

    # Also update results/summary_metrics.json for root scripts
    os.makedirs("results", exist_ok=True)
    with open("results/summary_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n================================================================================")
    print(f"Experiment completed successfully!")
    print(f"Results saved to {output_dir}")
    print(f"================================================================================")
    return summary, training_results


if __name__ == "__main__":
    run_comprehensive_experiment(
        env_name="ogbench-antmaze-medium-navigate-v0",
        n_pairs=3500,
        val_ratio=0.2,
        epochs=30,
        eval_seeds=list(range(10)),
        num_episodes=15,
        output_dir="results/advanced_benchmarks",
    )
