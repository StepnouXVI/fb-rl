"""Train amortized direct intention distillation policies from Dijkstra demonstrations."""

import argparse
import json
import os
import pickle
import sys
import time
from typing import Any, Dict, Optional, Tuple
import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.agent_loader import load_pretrained_agent
from src.networks import build_direct_translator
from src.telemetry import AimTracker
from src.telemetry.db import TelemetryDatabase
from src.topology import DijkstraGraph
from src.training import (
    make_direct_intention_eval_step,
    make_direct_intention_train_step,
)


def _extract_dijkstra_targets(
    teacher: DijkstraGraph, noisy_states: np.ndarray, goal_latents: np.ndarray, agent: Any
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate intention and action targets using Dijkstra graph teacher."""
    z_targets = []
    for i in tqdm(range(len(noisy_states)), desc="Dijkstra Waypoints"):
        z_targets.append(teacher.get_subgoal_latent(noisy_states[i], goal_latents[i], step=0))
    z_targets_arr = np.asarray(z_targets, dtype=np.float32)
    act_dist = agent.network.select("actor")(
        noisy_states, z_targets_arr, goal_encoded=True, temperature=0.0
    )
    return z_targets_arr, np.asarray(act_dist.mode())


def generate_or_load_golden_dataset(
    agent: Any,
    train_obs: np.ndarray,
    split: str = "medium",
    n_pairs: int = 10000,
    noise_sigma: float = 0.05,
    save_path: Optional[str] = None,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """Load or generate dataset of noisy state-goal pairs and teacher intentions."""
    if save_path and os.path.exists(save_path):
        print(f"Loading cached golden dataset from {save_path}...")
        d = np.load(save_path)
        return {
            "states": d["states"],
            "goal_zs": d["goal_zs"],
            "z_targets": d["z_targets"],
            "a_targets": d["a_targets"],
        }

    n_landmarks = 2000 if split == "large" else 1000
    teacher = DijkstraGraph(agent, train_obs, n_landmarks=n_landmarks)
    rng = np.random.default_rng(seed)
    s_idxs = rng.choice(len(train_obs), size=n_pairs)
    g_idxs = rng.choice(len(train_obs), size=n_pairs)
    noisy_states = (
        train_obs[s_idxs] + rng.normal(0.0, noise_sigma, size=train_obs[s_idxs].shape)
    ).astype(np.float32)
    goal_latents = np.asarray(
        agent.normalize_z(agent.network.select("backward_repr")(train_obs[g_idxs]))
    )
    z_targets, a_targets = _extract_dijkstra_targets(teacher, noisy_states, goal_latents, agent)

    dataset = {
        "states": noisy_states,
        "goal_zs": goal_latents,
        "z_targets": z_targets,
        "a_targets": a_targets,
    }
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        np.savez_compressed(save_path, **dataset)
        print(f"Saved dataset to {save_path}")
    return dataset


def _run_epoch_batches(
    step_fn: Any,
    params: Any,
    opt_state: Any,
    train_data: Dict[str, jnp.ndarray],
    batch_size: int,
    lambdas: Dict[str, float],
    rng: Any,
    n_train: int,
) -> Tuple[Any, Any, Any, Dict[str, float]]:
    """Execute training epoch over minibatches."""
    rng, perm_rng = jax.random.split(rng)
    perms = jax.random.permutation(perm_rng, n_train)
    actual_bs = min(batch_size, n_train)
    num_batches = max(1, n_train // actual_bs)
    metrics_list = []
    for b in range(num_batches):
        idx = perms[b * actual_bs : (b + 1) * actual_bs if b < num_batches - 1 else n_train]
        if len(idx) == 0:
            continue
        batch = {k: train_data[k][idx] for k in ["state", "goal_z", "z_target", "a_target"]}
        params, opt_state, m = step_fn(params, opt_state, batch, lambdas)
        metrics_list.append({k: float(v) for k, v in m.items()})
    mean_metrics = {k: float(np.mean([m[k] for m in metrics_list])) for k in metrics_list[0]}
    return params, opt_state, rng, mean_metrics


def _save_checkpoint(
    params: Any,
    model_type: str,
    split: str,
    lambdas: Dict[str, float],
    history: list,
    output_dir: str,
) -> str:
    """Save model weights and training history to disk and canonical checkpoint dir."""
    ckpt_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    payload = {"params": flax.core.unfreeze(params), "model_type": model_type, "config": lambdas}
    save_path = os.path.join(ckpt_dir, f"distilled_{model_type}_{split}.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(payload, f)
    print(f"\nModel saved to {save_path}")

    canonical_dir = os.path.join(PROJECT_ROOT, "outputs", "checkpoints")
    os.makedirs(canonical_dir, exist_ok=True)
    canonical_path = os.path.join(canonical_dir, f"distilled_{model_type}_{split}.pkl")
    with open(canonical_path, "wb") as f:
        pickle.dump(payload, f)

    bench_dir = os.path.join(output_dir, "benchmarks")
    os.makedirs(bench_dir, exist_ok=True)
    with open(os.path.join(bench_dir, f"history_{model_type}_{split}.json"), "w") as f:
        json.dump(history, f, indent=2)

    return canonical_path


def _prepare_train_val_split(
    dataset: Dict[str, np.ndarray]
) -> Tuple[Dict[str, jnp.ndarray], Dict[str, jnp.ndarray], int]:
    """Partition raw numpy dataset into JAX train and validation arrays."""
    n_total = len(dataset["states"])
    n_train = int(0.9 * n_total)
    train_data = {
        "state": jnp.asarray(dataset["states"][:n_train]),
        "goal_z": jnp.asarray(dataset["goal_zs"][:n_train]),
        "z_target": jnp.asarray(dataset["z_targets"][:n_train]),
        "a_target": jnp.asarray(dataset["a_targets"][:n_train]),
    }
    val_data = {
        "state": jnp.asarray(dataset["states"][n_train:]),
        "goal_z": jnp.asarray(dataset["goal_zs"][n_train:]),
        "z_target": jnp.asarray(dataset["z_targets"][n_train:]),
        "a_target": jnp.asarray(dataset["a_targets"][n_train:]),
    }
    return train_data, val_data, n_train


def _load_distillation_data(
    checkpoint_dir: str, split: str, seed: int, output_dir: str, n_pairs: int, noise_sigma: float
) -> Tuple[Any, Dict[str, jnp.ndarray], Dict[str, jnp.ndarray], int]:
    """Load agent and partitioned offline demonstration dataset."""
    agent, _, train_ds, _, _ = load_pretrained_agent(checkpoint_dir, split, seed=seed)
    cache_p = os.path.join(
        output_dir, "datasets", f"golden_dataset_{split}_{n_pairs}p_s{int(noise_sigma * 100)}.npz"
    )
    dataset = generate_or_load_golden_dataset(
        agent, train_ds["observations"], split, n_pairs, noise_sigma, cache_p, seed
    )
    train_data, val_data, n_train = _prepare_train_val_split(dataset)
    return agent, train_data, val_data, n_train


def _init_distillation_models(
    model_type: str,
    hidden_dim: int,
    n_layers: int,
    lr: float,
    seed: int,
    state_dim: int,
    goal_dim: int,
    agent: Any,
) -> Tuple[Any, Any, Any, Any]:
    """Initialize student network, optimizer state, and training step functions."""
    student_def = build_direct_translator(model_type, goal_dim, hidden_dim, n_layers)
    rng = jax.random.PRNGKey(seed)
    params = student_def.init(
        rng, jnp.zeros((1, state_dim)), jnp.zeros((1, goal_dim))
    )["params"]
    optimizer = optax.adamw(learning_rate=lr, weight_decay=1e-4)
    opt_state = optimizer.init(params)
    step_fn = make_direct_intention_train_step(student_def, agent, optimizer)
    eval_fn = make_direct_intention_eval_step(student_def, agent)
    return params, opt_state, step_fn, eval_fn


def _execute_training_loop(
    step_fn: Any,
    eval_fn: Any,
    params: Any,
    opt_state: Any,
    train_data: Dict[str, jnp.ndarray],
    val_data: Dict[str, jnp.ndarray],
    batch_size: int,
    lambdas: Dict[str, float],
    rng: Any,
    n_train: int,
    epochs: int,
    db: TelemetryDatabase,
    train_run_id: str,
    lr: float,
    save_info: Tuple[str, str, str],
    aim_tracker: Optional[AimTracker] = None,
) -> Tuple[Any, list, Optional[str]]:
    """Execute iterative optimization epochs with SQLite and Aim logging."""
    history = []
    best_ckpt = None
    for ep in range(1, epochs + 1):
        t0 = time.time()
        params, opt_state, rng, tr_m = _run_epoch_batches(
            step_fn, params, opt_state, train_data, batch_size, lambdas, rng, n_train
        )
        val_m = {k: float(v) for k, v in eval_fn(params, val_data).items()}
        ep_time = time.time() - t0
        history.append({"epoch": ep, **tr_m, **val_m})
        ckpt_path = None
        if ep == epochs:
            model_type, split, out_dir = save_info
            ckpt_path = _save_checkpoint(params, model_type, split, lambdas, history, out_dir)
            best_ckpt = ckpt_path
        combined_m = {**tr_m, **val_m}
        db.insert_training_epoch(
            train_run_id, ep, combined_m,
            checkpoint_path=ckpt_path, epoch_time_s=ep_time, learning_rate=lr,
        )
        if aim_tracker is not None:
            for k, v in combined_m.items():
                aim_tracker.track(v, name=k, epoch=ep, step=ep)
            aim_tracker.track(lr, name="learning_rate", epoch=ep, step=ep)
            aim_tracker.track(ep_time, name="epoch_time_s", epoch=ep, step=ep)
            if ckpt_path:
                aim_tracker.set_params({"best_checkpoint_path": ckpt_path})
        if ep % 5 == 0 or ep == epochs:
            print(
                f"Epoch {ep:3d}/{epochs} | Loss: {tr_m['loss']:.4f} | "
                f"Val Cos: {val_m['val_cos_sim']:.4f} | Val ActMSE: {val_m['val_action_mse']:.4f}"
            )
    return params, history, best_ckpt


def train_distillation(
    checkpoint_dir: str = "fb-test",
    split: str = "medium",
    model_type: str = "gated_attn",
    hidden_dim: int = 256,
    n_layers: int = 3,
    n_pairs: int = 15000,
    noise_sigma: float = 0.05,
    epochs: int = 40,
    batch_size: int = 256,
    lr: float = 3e-4,
    l_mse: float = 0.1,
    l_cos: float = 1.0,
    l_action: float = 0.5,
    l_reach: float = 0.02,
    l_goal: float = 0.05,
    output_dir: str = "results",
    seed: int = 0,
    db_path: str = "results/telemetry.db",
    aim_repo: Optional[str] = None,
) -> Tuple[Any, list]:
    """Train direct intention translator network on offline demonstrations."""
    agent, train_data, val_data, n_train = _load_distillation_data(
        checkpoint_dir, split, seed, output_dir, n_pairs, noise_sigma
    )
    params, opt_state, step_fn, eval_fn = _init_distillation_models(
        model_type, hidden_dim, n_layers, lr, seed,
        train_data["state"].shape[-1], train_data["goal_z"].shape[-1], agent,
    )
    lambdas = {
        "l_mse": l_mse, "l_cos": l_cos, "l_action": l_action, "l_reach": l_reach, "l_goal": l_goal
    }
    db_file = db_path if os.path.isabs(db_path) else os.path.join(PROJECT_ROOT, db_path)
    os.makedirs(os.path.dirname(os.path.abspath(db_file)), exist_ok=True)
    db = TelemetryDatabase(db_file)
    repo = aim_repo or os.path.join(PROJECT_ROOT, "results", "aim")
    try:
        run_name = f"distillation_{model_type}_{split}_s{seed}"
        config_dict = {
            "checkpoint_dir": checkpoint_dir, "split": split, "model_type": model_type,
            "hidden_dim": hidden_dim, "n_layers": n_layers, "n_pairs": n_pairs,
            "noise_sigma": noise_sigma, "epochs": epochs, "batch_size": batch_size,
            "lr": lr, "seed": seed, **lambdas,
        }
        train_run_id = db.create_training_run(
            run_name=run_name, model_type=model_type, split=split, seed=seed, config=config_dict,
        )
        save_info = (model_type, split, output_dir)
        with AimTracker(repo=repo, experiment=f"train_{split}", run_name=run_name) as aim_tr:
            aim_tr.set_params(config_dict)
            aim_tr.add_tags([model_type, split])
            params, history, best_ckpt = _execute_training_loop(
                step_fn, eval_fn, params, opt_state, train_data, val_data, batch_size,
                lambdas, jax.random.PRNGKey(seed), n_train, epochs, db, train_run_id, lr, save_info, aim_tr,
            )
        db.finish_training_run(train_run_id, best_checkpoint_path=best_ckpt, status="completed")
    finally:
        db.close()
    return params, history


def main() -> None:
    """Parse CLI options and start direct intention distillation training."""
    parser = argparse.ArgumentParser(description="Train direct intention distillation policy.")
    parser.add_argument("--split", type=str, default="medium")
    parser.add_argument("--model_type", type=str, default="gated_attn")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--n_pairs", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--db_path", type=str, default="results/telemetry.db")
    parser.add_argument("--no_mlflow", action="store_true")
    args = parser.parse_args()
    train_distillation(
        split=args.split,
        model_type=args.model_type,
        epochs=args.epochs,
        batch_size=args.batch_size,
        n_pairs=args.n_pairs,
        lr=args.lr,
        db_path=args.db_path,
    )


if __name__ == "__main__":
    main()
