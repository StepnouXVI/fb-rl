#!/usr/bin/env python3
import os
import sys
import time
import json
import pickle
import argparse
from typing import Dict, Any, Tuple
import numpy as np
from tqdm import tqdm
import jax
import jax.numpy as jnp
import flax
import optax

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.agent_loader import load_pretrained_agent
from src.planners import BufferGraphPlanner
from src.waypoint_translators import build_flax_translator, make_train_step, make_eval_step


def _extract_dijkstra_targets(teacher, noisy_states, goal_latents, agent):
    z_targets = []
    for i in tqdm(range(len(noisy_states)), desc="Dijkstra Waypoints"):
        z_targets.append(teacher.get_subgoal_latent(noisy_states[i], goal_latents[i], step=0))
    z_targets = np.asarray(z_targets, dtype=np.float32)
    act_dist = agent.network.select("actor")(noisy_states, z_targets, goal_encoded=True, temperature=0.0)
    return z_targets, np.asarray(act_dist.mode())


def generate_or_load_golden_dataset(agent, train_obs, split="medium", n_pairs=10000, noise_sigma=0.05, save_path=None, seed=42):
    if save_path and os.path.exists(save_path):
        print(f"Loading cached golden dataset from {save_path}...")
        d = np.load(save_path)
        return {"states": d["states"], "goal_zs": d["goal_zs"], "z_targets": d["z_targets"], "a_targets": d["a_targets"]}

    n_landmarks = 2000 if split == "large" else 1000
    teacher = BufferGraphPlanner(agent, train_obs, n_landmarks=n_landmarks)
    rng = np.random.default_rng(seed)
    s_idxs, g_idxs = rng.choice(len(train_obs), size=n_pairs), rng.choice(len(train_obs), size=n_pairs)
    noisy_states = (train_obs[s_idxs] + rng.normal(0.0, noise_sigma, size=train_obs[s_idxs].shape)).astype(np.float32)
    goal_latents = np.asarray(agent.normalize_z(agent.network.select("backward_repr")(train_obs[g_idxs])))
    z_targets, a_targets = _extract_dijkstra_targets(teacher, noisy_states, goal_latents, agent)

    dataset = {"states": noisy_states, "goal_zs": goal_latents, "z_targets": z_targets, "a_targets": a_targets}
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        np.savez_compressed(save_path, **dataset)
        print(f"Saved dataset to {save_path}")
    return dataset


def _run_epoch_batches(step_fn, params, opt_state, train_data, batch_size, lambdas, rng, n_train):
    rng, perm_rng = jax.random.split(rng)
    perms = jax.random.permutation(perm_rng, n_train)
    num_batches = n_train // batch_size
    metrics_list = []
    for b in range(num_batches):
        idx = perms[b * batch_size : (b + 1) * batch_size]
        batch = {k: train_data[k][idx] for k in ["state", "goal_z", "z_target", "a_target"]}
        params, opt_state, m = step_fn(params, opt_state, batch, lambdas)
        metrics_list.append({k: float(v) for k, v in m.items()})
    return params, opt_state, rng, {k: float(np.mean([m[k] for m in metrics_list])) for k in metrics_list[0]}


def _save_checkpoint(params, model_type, split, lambdas, history, output_dir, use_mlflow):
    ckpt_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    save_path = os.path.join(ckpt_dir, f"distilled_jax_{model_type}_{split}.pkl")
    with open(save_path, "wb") as f:
        pickle.dump({"params": flax.core.unfreeze(params), "model_type": model_type, "config": lambdas}, f)
    print(f"\nModel saved to {save_path}")

    bench_dir = os.path.join(output_dir, "benchmarks")
    os.makedirs(bench_dir, exist_ok=True)
    with open(os.path.join(bench_dir, f"history_jax_{model_type}_{split}.json"), "w") as f:
        json.dump(history, f, indent=2)

    if use_mlflow:
        try:
            import mlflow
            mlflow.log_artifact(save_path)
            mlflow.end_run()
        except Exception:
            pass


def train_jax_distillation(checkpoint_dir="fb-test", split="medium", model_type="gated_attn", hidden_dim=256, n_layers=3, n_pairs=15000, noise_sigma=0.05, epochs=40, batch_size=256, lr=3e-4, l_mse=0.1, l_cos=1.0, l_action=0.5, l_reach=0.02, l_goal=0.05, use_mlflow=True, output_dir="results", seed=0):
    agent, _, train_ds, _, _ = load_pretrained_agent(checkpoint_dir, split, seed=seed)
    cache_p = os.path.join(output_dir, "datasets", f"golden_dataset_{split}_{n_pairs}p_s{int(noise_sigma*100)}.npz")
    dataset = generate_or_load_golden_dataset(agent, train_ds["observations"], split, n_pairs, noise_sigma, cache_p, seed)

    N = len(dataset["states"])
    n_train = int(0.9 * N)
    train_data = {"state": jnp.asarray(dataset["states"][:n_train]), "goal_z": jnp.asarray(dataset["goal_zs"][:n_train]), "z_target": jnp.asarray(dataset["z_targets"][:n_train]), "a_target": jnp.asarray(dataset["a_targets"][:n_train])}
    val_data = {"state": jnp.asarray(dataset["states"][n_train:]), "goal_z": jnp.asarray(dataset["goal_zs"][n_train:]), "z_target": jnp.asarray(dataset["z_targets"][n_train:]), "a_target": jnp.asarray(dataset["a_targets"][n_train:])}

    student_def = build_flax_translator(model_type, dataset["goal_zs"].shape[-1], hidden_dim, n_layers)
    rng = jax.random.PRNGKey(seed)
    params = student_def.init(rng, jnp.zeros((1, dataset["states"].shape[-1])), jnp.zeros((1, dataset["goal_zs"].shape[-1])))["params"]
    optimizer = optax.adamw(learning_rate=lr, weight_decay=1e-4)
    opt_state = optimizer.init(params)

    lambdas = {"l_mse": l_mse, "l_cos": l_cos, "l_action": l_action, "l_reach": l_reach, "l_goal": l_goal}
    step_fn, eval_fn = make_train_step(student_def, agent, optimizer), make_eval_step(student_def, agent)
    history = []

    for ep in range(1, epochs + 1):
        params, opt_state, rng, tr_m = _run_epoch_batches(step_fn, params, opt_state, train_data, batch_size, lambdas, rng, n_train)
        val_m = {k: float(v) for k, v in eval_fn(params, val_data).items()}
        history.append({"epoch": ep, **tr_m, **val_m})
        if ep % 5 == 0 or ep == epochs:
            print(f"Epoch {ep:3d}/{epochs} | Loss: {tr_m['loss']:.4f} | Val Cos: {val_m['val_cos_sim']:.4f} | Val ActMSE: {val_m['val_action_mse']:.4f}")

    _save_checkpoint(params, model_type, split, lambdas, history, output_dir, use_mlflow)
    return params, history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="medium")
    parser.add_argument("--model_type", type=str, default="gated_attn")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--n_pairs", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--no_mlflow", action="store_true")
    args = parser.parse_args()
    train_jax_distillation(split=args.split, model_type=args.model_type, epochs=args.epochs, batch_size=args.batch_size, n_pairs=args.n_pairs, lr=args.lr, use_mlflow=not args.no_mlflow)


if __name__ == "__main__":
    main()
