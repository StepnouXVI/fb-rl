import os
import sys
import time
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

# Ensure top-level directory is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent_loader import load_pretrained_agent
from src.planners import (
    BaselinePlanner,
    BufferGraphPlanner,
    DistilledMLPPlanner,
)
from src.evaluator import ZeroShotEvaluator
from src.models import build_student_model
from src.metrics import aggregate_runs, bootstrap_ci


def compute_bootstrap_ci(data, num_bootstraps=2000, ci=95):
    if len(data) == 0:
        return {"mean": 0.0, "std": 0.0, "ci_lower": 0.0, "ci_upper": 0.0}
    data = np.asarray(data)
    boot_means = [np.mean(np.random.choice(data, size=len(data), replace=True)) for _ in range(num_bootstraps)]
    lower = float(np.percentile(boot_means, (100 - ci) / 2))
    upper = float(np.percentile(boot_means, 100 - (100 - ci) / 2))
    return {
        "mean": float(np.mean(data)),
        "std": float(np.std(data)),
        "ci_lower": lower,
        "ci_upper": upper,
    }


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def estimate_flops(model: nn.Module, in_dim: int = 157) -> int:
    """FLOPs estimation for single forward pass."""
    total_flops = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            total_flops += 2 * module.in_features * module.out_features
        elif isinstance(module, nn.Conv1d):
            total_flops += 2 * module.in_channels * module.out_channels * module.kernel_size[0]
        elif isinstance(module, nn.MultiheadAttention):
            d = module.embed_dim
            total_flops += 4 * (2 * d * d) + 2 * d
    return total_flops


def generate_or_load_dataset(
    teacher: BufferGraphPlanner,
    agent,
    train_ds,
    n_pairs: int = 8000,
    split: str = "medium",
    output_dir: str = "results",
    seed: int = 42,
):
    os.makedirs(output_dir, exist_ok=True)
    cache_path = os.path.join(output_dir, f"distill_dataset_{split}_{n_pairs}pairs.npz")
    if os.path.exists(cache_path):
        print(f"[Dataset] Loading cached dataset from {cache_path}")
        data = np.load(cache_path)
        return data["states"], data["goal_latents"], data["target_latents"]

    print(f"[Dataset] Generating {n_pairs} demonstration pairs from BufferGraphPlanner (Teacher)...")
    rng = np.random.default_rng(seed)
    s_idxs = rng.choice(len(train_ds["observations"]), size=n_pairs)
    g_idxs = rng.choice(len(train_ds["observations"]), size=n_pairs)

    states = train_ds["observations"][s_idxs]
    goal_latents = np.asarray(
        agent.normalize_z(agent.network.select("backward_repr")(train_ds["observations"][g_idxs]))
    )

    target_latents = []
    t0 = time.perf_counter()
    for i in tqdm(range(n_pairs), desc="Generating targets"):
        subgoal_z = teacher.get_subgoal_latent(states[i], goal_latents[i], step=0)
        target_latents.append(subgoal_z)
    t1 = time.perf_counter()
    print(f"[Dataset] Generation completed in {t1 - t0:.2f} s ({n_pairs / (t1 - t0):.1f} pairs/sec)")

    target_latents = np.asarray(target_latents, dtype=np.float32)
    np.savez_compressed(
        cache_path,
        states=states,
        goal_latents=goal_latents,
        target_latents=target_latents,
    )
    print(f"[Dataset] Saved to {cache_path}")
    return states, goal_latents, target_latents


def train_single_architecture(
    model_type: str,
    states: np.ndarray,
    goal_latents: np.ndarray,
    target_latents: np.ndarray,
    split: str = "medium",
    val_ratio: float = 0.2,
    epochs: int = 30,
    batch_size: int = 128,
    lr: float = 1e-3,
    hidden_dim: int = 256,
    n_layers: int = 3,
    num_heads: int = 4,
    lambda_mse: float = 0.1,
    output_dir: str = "results",
    device_str: str = "auto",
):
    save_path = os.path.join(output_dir, f"distilled_{model_type}_{split}.pt")
    obs_dim = states.shape[-1]
    latent_dim = goal_latents.shape[-1]

    dummy_model = build_student_model(
        model_type=model_type,
        obs_dim=obs_dim,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        num_heads=num_heads,
    )
    params = count_parameters(dummy_model)
    flops = estimate_flops(dummy_model, in_dim=obs_dim + latent_dim)

    # If already trained and exists in training_metrics_all.json, return
    metrics_path = os.path.join(output_dir, "training_metrics_all.json")
    if os.path.exists(metrics_path) and os.path.exists(save_path):
        with open(metrics_path, "r") as f:
            cached_metrics = json.load(f)
        if model_type in cached_metrics:
            print(f"[Cached Model] Found trained {model_type} at {save_path}")
            return cached_metrics[model_type]

    print(f"\n=======================================================")
    print(f"Training Architecture: {model_type.upper()}")
    print(f"=======================================================")

    if device_str == "auto":
        device = torch.device(
            "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
        )
    else:
        device = torch.device(device_str)

    n_samples = len(states)
    n_val = int(n_samples * val_ratio)
    n_train = n_samples - n_val

    # Deterministic split
    rng = np.random.default_rng(42)
    perm = rng.permutation(n_samples)
    train_idx = perm[:n_train]
    val_idx = perm[n_train:]

    X = np.concatenate([states, goal_latents], axis=-1).astype(np.float32)
    Y = target_latents.astype(np.float32)

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X[train_idx]), torch.from_numpy(Y[train_idx])),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X[val_idx]), torch.from_numpy(Y[val_idx])),
        batch_size=batch_size,
        shuffle=False,
    )

    model = build_student_model(
        model_type=model_type,
        obs_dim=obs_dim,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        num_heads=num_heads,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_cos_loss": [],
        "val_cos_sim": [],
        "val_mse": [],
    }

    t_start = time.perf_counter()
    pbar = tqdm(range(1, epochs + 1), desc=f"Training {model_type}")
    for epoch in pbar:
        model.train()
        train_loss_acc = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            pred = model(bx)
            cos_loss = 1.0 - F.cosine_similarity(pred, by, dim=-1).mean()
            mse_loss = F.mse_loss(pred, by)
            loss = cos_loss + lambda_mse * mse_loss
            loss.backward()
            optimizer.step()
            train_loss_acc += loss.item() * len(bx)
        scheduler.step()

        train_loss = train_loss_acc / n_train

        # Validation
        model.eval()
        val_loss_acc, val_cos_loss_acc, val_cos_sim_acc, val_mse_acc = 0.0, 0.0, 0.0, 0.0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                pred = model(bx)
                cos_sim = F.cosine_similarity(pred, by, dim=-1).mean().item()
                cos_loss = 1.0 - cos_sim
                mse_loss = F.mse_loss(pred, by).item()
                loss = cos_loss + lambda_mse * mse_loss

                val_loss_acc += loss * len(bx)
                val_cos_loss_acc += cos_loss * len(bx)
                val_cos_sim_acc += cos_sim * len(bx)
                val_mse_acc += mse_loss * len(bx)

        val_loss = val_loss_acc / n_val
        val_cos_loss = val_cos_loss_acc / n_val
        val_cos_sim = val_cos_sim_acc / n_val
        val_mse = val_mse_acc / n_val

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_cos_loss"].append(val_cos_loss)
        history["val_cos_sim"].append(val_cos_sim)
        history["val_mse"].append(val_mse)

        pbar.set_postfix({
            "TrLoss": f"{train_loss:.4f}",
            "ValLoss": f"{val_loss:.4f}",
            "CosSim": f"{val_cos_sim:.4f}",
            "MSE": f"{val_mse:.4f}",
        })

    train_time = time.perf_counter() - t_start

    # Save checkpoint
    torch.save(model.state_dict(), save_path)
    print(f"[Model Saved] {save_path}")

    return {
        "model_type": model_type,
        "parameters": params,
        "flops": flops,
        "checkpoint_path": save_path,
        "training_time_sec": train_time,
        "final_train_loss": history["train_loss"][-1],
        "final_val_loss": history["val_loss"][-1],
        "final_val_cos_loss": history["val_cos_loss"][-1],
        "final_val_cos_sim": history["val_cos_sim"][-1],
        "final_val_mse": history["val_mse"][-1],
        "history": history,
    }


def run_full_distillation_and_benchmark(
    checkpoint_dir: str = "fb-test",
    split: str = "medium",
    n_pairs: int = 8000,
    epochs: int = 30,
    eval_seeds: list = [0, 1, 2, 3, 4],
    num_episodes: int = 10,
    output_dir: str = "results",
):
    os.makedirs(output_dir, exist_ok=True)
    print("================================================================================")
    print(f"Starting Comprehensive Amortized Latent Planning Distillation & Benchmark")
    print(f"Environment: ogbench-antmaze-{split}-navigate-v0")
    print(f"Demonstrations: {n_pairs} pairs | Epochs: {epochs} | Seeds: {eval_seeds} ({num_episodes} eps/task)")
    print("================================================================================")

    # 1. Load Pretrained Agent and Environment
    agent, env, train_ds, val_ds, config = load_pretrained_agent(checkpoint_dir, split, seed=0)

    # 2. Build Teacher Dijkstra Planner
    print("\n[Teacher] Building Buffer Graph Planner (1000 landmarks)...")
    teacher = BufferGraphPlanner(
        agent,
        train_ds["observations"],
        n_landmarks=1000,
        max_edge_radius=3.5,
        reachability_cutoff=35.0,
        lookahead_dist=2.6,
    )

    # 3. Generate Demonstration Dataset
    states, goal_latents, target_latents = generate_or_load_dataset(
        teacher=teacher,
        agent=agent,
        train_ds=train_ds,
        n_pairs=n_pairs,
        split=split,
        output_dir=output_dir,
    )

    # 4. List of Architectures to Distill
    architectures = [
        "mlp",
        "dense_eca",
        "gated_attn",
        "resnet_eca",
        "film_resnet",
        "transformer",
    ]

    training_results = {}
    for arch in architectures:
        res = train_single_architecture(
            model_type=arch,
            states=states,
            goal_latents=goal_latents,
            target_latents=target_latents,
            split=split,
            val_ratio=0.2,
            epochs=epochs,
            output_dir=output_dir,
        )
        training_results[arch] = res

    # Save training histories
    with open(os.path.join(output_dir, "training_metrics_all.json"), "w") as f:
        json.dump(training_results, f, indent=2)

    # 5. Build Planners for Evaluation
    eval_planners = {
        "Single-Intention Baseline": BaselinePlanner(
            agent,
            dataset_states=train_ds["observations"],
            use_high_actor=True,
            name="Single-Intention Baseline",
        ),
        "Buffer Graph (Dijkstra Teacher)": teacher,
    }

    arch_display_names = {
        "mlp": "Distilled StandardMLP",
        "dense_eca": "Distilled DenseECANetwork",
        "gated_attn": "Distilled GatedCrossAttention",
        "resnet_eca": "Distilled ResNetECANetwork",
        "film_resnet": "Distilled FiLMResNetNetwork",
        "transformer": "Distilled TransformerEncoder",
    }

    for arch in architectures:
        ckpt_path = training_results[arch]["checkpoint_path"]
        eval_planners[arch_display_names[arch]] = DistilledMLPPlanner(
            agent=agent,
            model_type=arch,
            checkpoint_path=ckpt_path,
            device="cpu",
            name=arch_display_names[arch],
        )

    # 6. Run Multi-Seed Rollout Evaluation
    print("\n================================================================================")
    print(f"Executing Multi-Seed Rollout Benchmark ({len(eval_seeds)} Seeds x {num_episodes} Episodes x 5 Tasks)")
    print("================================================================================")

    evaluator = ZeroShotEvaluator(
        env=env,
        agent=agent,
        dataset_dict=train_ds,
        config=config,
        env_name=f"ogbench-antmaze-{split}-navigate-v0",
    )

    benchmark_records = []
    summary_by_method = {}

    for method_name, planner in eval_planners.items():
        print(f"\n---> Evaluating {method_name}...")
        method_seed_results = []
        seed_successes, seed_steps, seed_latencies, seed_crossings = [], [], [], []

        for seed in eval_seeds:
            summary = evaluator.evaluate_all_tasks(
                planner=planner,
                num_episodes=num_episodes,
                eval_temperature=0.0,
                seed=seed,
            )
            method_seed_results.append(summary)
            seed_successes.append(summary["success_rate"])
            seed_steps.append(summary["mean_length"])
            seed_latencies.append(summary["latency_ms"])
            seed_crossings.append(summary["self_intersections"])

            benchmark_records.append({
                "method": method_name,
                "seed": seed,
                "success_rate": summary["success_rate"],
                "mean_steps": summary["mean_length"],
                "latency_ms": summary["latency_ms"],
                "self_intersections": summary["self_intersections"],
            })

            print(f"   [Seed {seed}] Success: {summary['success_rate']:5.1f}% | Steps: {summary['mean_length']:5.1f} | Latency: {summary['latency_ms']:.3f} ms | Loops: {summary['self_intersections']:.2f}")

        # Compute aggregate bootstrap CIs
        sr_ci = compute_bootstrap_ci(seed_successes)
        st_ci = compute_bootstrap_ci(seed_steps)
        lat_ci = compute_bootstrap_ci(seed_latencies)
        cross_ci = compute_bootstrap_ci(seed_crossings)

        print(f"   ==> {method_name} Summary: Success = {sr_ci['mean']:.1f} ± {sr_ci['std']:.1f}% | Steps = {st_ci['mean']:.1f} ± {st_ci['std']:.1f} | Latency = {lat_ci['mean']:.3f} ms")

        summary_by_method[method_name] = {
            "success_rate": sr_ci,
            "mean_steps": st_ci,
            "latency_ms": lat_ci,
            "self_intersections": cross_ci,
        }

    # 7. Merge Training & Evaluation Metrics into Comprehensive Comparison Table
    table_rows = []
    for method_name in eval_planners.keys():
        eval_stats = summary_by_method[method_name]
        
        # Find corresponding architecture if distilled
        matched_arch = None
        for arch, dname in arch_display_names.items():
            if dname == method_name:
                matched_arch = arch
                break

        if matched_arch:
            t_res = training_results[matched_arch]
            param_cnt = t_res["parameters"]
            val_cos_loss = t_res["final_val_cos_loss"]
            val_cos_sim = t_res["final_val_cos_sim"]
            val_mse = t_res["final_val_mse"]
            val_total_loss = t_res["final_val_loss"]
        else:
            param_cnt = 0
            val_cos_loss = 0.0
            val_cos_sim = 1.0 if "Teacher" in method_name else 0.0
            val_mse = 0.0
            val_total_loss = 0.0

        table_rows.append({
            "Architecture / Planner": method_name,
            "Parameters": param_cnt,
            "Val Cosine Sim": val_cos_sim,
            "Val Cosine Loss": val_cos_loss,
            "Val MSE": val_mse,
            "Success Rate (%)": f"{eval_stats['success_rate']['mean']:.1f} ± {eval_stats['success_rate']['std']:.1f}",
            "Success Mean (%)": eval_stats['success_rate']['mean'],
            "Success Std (%)": eval_stats['success_rate']['std'],
            "Mean Episode Steps": f"{eval_stats['mean_steps']['mean']:.1f} ± {eval_stats['mean_steps']['std']:.1f}",
            "Mean Steps (val)": eval_stats['mean_steps']['mean'],
            "Inference Latency (ms/step)": f"{eval_stats['latency_ms']['mean']:.3f} ± {eval_stats['latency_ms']['std']:.3f}",
            "Latency Mean (ms)": eval_stats['latency_ms']['mean'],
            "Loops / Crossings": f"{eval_stats['self_intersections']['mean']:.2f} ± {eval_stats['self_intersections']['std']:.2f}",
        })

    df_table = pd.DataFrame(table_rows)
    df_table.to_csv(os.path.join(output_dir, "distillation_benchmark_table.csv"), index=False)

    df_records = pd.DataFrame(benchmark_records)
    df_records.to_csv(os.path.join(output_dir, "distillation_benchmark_runs.csv"), index=False)

    # Generate Markdown Table
    md_table = "| Architecture / Planner | Params | Val Cosine Sim | Val MSE | Success Rate (%) | Mean Steps | Latency (ms/step) | Loops/Episode |\n"
    md_table += "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n"
    for r in table_rows:
        param_str = f"{r['Parameters']:,}" if r['Parameters'] > 0 else "-"
        cos_str = f"{r['Val Cosine Sim']:.4f}" if r['Val Cosine Sim'] > 0 else "-"
        mse_str = f"{r['Val MSE']:.4f}" if r['Val MSE'] > 0 else "-"
        md_table += f"| **{r['Architecture / Planner']}** | {param_str} | {cos_str} | {mse_str} | **{r['Success Rate (%)']}** | {r['Mean Episode Steps']} | {r['Inference Latency (ms/step)']} | {r['Loops / Crossings']} |\n"

    with open(os.path.join(output_dir, "distillation_benchmark_table.md"), "w") as f:
        f.write(md_table)

    final_payload = {
        "summary_by_method": summary_by_method,
        "training_results": training_results,
        "table_rows": table_rows,
        "markdown_table": md_table,
    }

    with open(os.path.join(output_dir, "distillation_benchmark_results.json"), "w") as f:
        json.dump(final_payload, f, indent=2)

    print("\n================================================================================")
    print("Benchmark Completed Successfully!")
    print("================================================================================")
    print(md_table)
    return final_payload


if __name__ == "__main__":
    run_full_distillation_and_benchmark()
