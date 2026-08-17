"""
Architecture Search & Ablation Study for Distilled Latent Multi-Subgoal Controller.
Compares 5 neural network variants across depth, width, residual connections, and parameter scale.
"""

import os
import json
import time
import numpy as np
import torch
import pandas as pd

from fb_core.fb_model_wrapper import FBModelWrapper
from fb_core.dataset_sampler import DatasetSampler
from fb_core.evaluator import EpisodeEvaluator
from fb_core.metrics_collector import MetricsCollector

from planners.buffer_graph import BufferGraphPlanner
from planners.distilled_mlp import DistilledMLPPlanner


def run_architecture_search(
    env_name: str = "antmaze-medium-navigate-v0",
    n_pairs: int = 1500,
    val_ratio: float = 0.2,
    epochs: int = 25,
    eval_seeds: list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    num_episodes: int = 10,
    output_dir: str = "results/architecture_ablation",
):
    os.makedirs(output_dir, exist_ok=True)
    clean_env_name = env_name.replace("ogbench-", "")
    print(f"=== Starting Neural Architecture Search on {clean_env_name} ===")
    print(f"Distillation pairs: {n_pairs} (Train: {int(n_pairs*(1-val_ratio))}, Val: {int(n_pairs*val_ratio)})")
    print(f"Evaluation: {len(eval_seeds)} seeds ({eval_seeds}), {num_episodes} episodes/seed")

    dataset_sampler = DatasetSampler.from_ogbench(env_name=clean_env_name, max_samples=10000, seed=42)
    fb_model = FBModelWrapper(agent=None, latent_dim=128)
    evaluator = EpisodeEvaluator(env_name=clean_env_name, max_episode_steps=600)

    try:
        import ogbench
        env, _, _ = ogbench.make_env_and_datasets(clean_env_name)
    except Exception:
        env = None

    # Teacher graph planner
    graph_planner = BufferGraphPlanner(
        fb_model=fb_model,
        dataset_sampler=dataset_sampler,
        n_landmarks=200,
    )

    # 5 Architecture Variants
    architectures = {
        "Tiny-MLP (2x128)": {"hidden_dims": [128], "use_residual": False},
        "Base-MLP (3x256)": {"hidden_dims": [256, 256], "use_residual": False},
        "Deep-MLP (4x256)": {"hidden_dims": [256, 256, 256], "use_residual": False},
        "Wide-MLP (3x512)": {"hidden_dims": [512, 512], "use_residual": False},
        "ResNet-MLP (4x256-Skip)": {"hidden_dims": [256, 256, 256], "use_residual": True},
    }

    ablation_summary = {}
    all_runs = []

    for arch_name, arch_cfg in architectures.items():
        print(f"\n==================================================")
        print(f"Training & Evaluating Architecture: {arch_name}")
        print(f"Config: {arch_cfg}")
        print(f"==================================================")

        planner = DistilledMLPPlanner(
            fb_model=fb_model,
            dataset_sampler=dataset_sampler,
            name=arch_name,
            obs_dim=29,
            latent_dim=128,
            hidden_dims=arch_cfg["hidden_dims"],
            use_residual=arch_cfg["use_residual"],
            device="cpu",
        )

        param_count = sum(p.numel() for p in planner.model.parameters())

        # Train
        t_train_0 = time.perf_counter()
        train_res = planner.train_distillation(
            graph_planner=graph_planner,
            n_pairs=n_pairs,
            val_ratio=val_ratio,
            epochs=epochs,
            batch_size=64,
            lr=1e-3,
        )
        t_train = time.perf_counter() - t_train_0

        print(f"  Training finished in {t_train:.1f}s | Train Loss: {train_res['final_train_loss']:.4f} | Val Loss: {train_res['final_val_loss']:.4f} | Val Cosine: {train_res['final_val_cosine_sim']:.4f} | Params: {param_count:,}")

        # Multi-seed rollout evaluation
        seed_successes, seed_steps, seed_latencies = [], [], []
        for seed in eval_seeds:
            res = evaluator.evaluate_planner(
                planner=planner,
                fb_model=fb_model,
                env=env,
                num_episodes=num_episodes,
                seed=seed,
            )
            seed_successes.append(res["success_rate"])
            seed_steps.append(res["mean_steps"])
            seed_latencies.append(res["mean_latency_ms"])

            all_runs.append({
                "architecture": arch_name,
                "seed": seed,
                "success_rate": res["success_rate"],
                "mean_steps": res["mean_steps"],
                "mean_latency_ms": res["mean_latency_ms"],
                "val_cosine_sim": train_res["final_val_cosine_sim"],
                "val_loss": train_res["final_val_loss"],
                "param_count": param_count,
            })

        ablation_summary[arch_name] = {
            "parameters": param_count,
            "training_time_sec": float(t_train),
            "final_train_loss": float(train_res["final_train_loss"]),
            "final_val_loss": float(train_res["final_val_loss"]),
            "final_val_cosine_sim": float(train_res["final_val_cosine_sim"]),
            "final_val_mse": float(train_res["final_val_mse"]),
            "success_rate": MetricsCollector.compute_bootstrap_ci(seed_successes),
            "steps_to_goal": MetricsCollector.compute_bootstrap_ci(seed_steps),
            "latency_ms": MetricsCollector.compute_bootstrap_ci(seed_latencies),
        }

    # Export results
    with open(os.path.join(output_dir, "ablation_summary.json"), "w") as f:
        json.dump(ablation_summary, f, indent=2)

    df_runs = pd.DataFrame(all_runs)
    df_runs.to_csv(os.path.join(output_dir, "ablation_runs.csv"), index=False)

    print("\n=== Architecture Ablation Study Completed! ===")
    print(f"Results saved to {os.path.join(output_dir, 'ablation_summary.json')}")
    return ablation_summary


if __name__ == "__main__":
    run_architecture_search(n_pairs=1000, epochs=20, eval_seeds=list(range(10)), num_episodes=10)
