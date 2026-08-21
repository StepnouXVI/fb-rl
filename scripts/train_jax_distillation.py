import os
import sys
import time
import json
import argparse
from typing import Dict, Any, Tuple
import numpy as np
from tqdm import tqdm
import jax
import jax.numpy as jnp
import flax
import optax

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.agent_loader import load_pretrained_agent
from src.planners import BufferGraphPlanner
from src.jax_distillation import build_flax_translator, make_train_step, make_eval_step


def generate_or_load_golden_dataset(
    agent,
    train_obs: np.ndarray,
    split: str = "medium",
    n_pairs: int = 10000,
    noise_sigma: float = 0.05,
    save_path: str = None,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """
    Generates or loads offline golden demonstration dataset:
    [state (with noise), goal_z] -> [z_target from high_actor(w_1), a_target from low_actor(z_target)].
    """
    if save_path and os.path.exists(save_path):
        print(f"Loading cached golden dataset from {save_path}...")
        data = np.load(save_path)
        return {
            "states": data["states"],
            "goal_zs": data["goal_zs"],
            "z_targets": data["z_targets"],
            "a_targets": data["a_targets"],
        }

    n_landmarks = 2000 if split == "large" else 1000
    print(f"Constructing Dijkstra teacher graph with {n_landmarks} landmarks on {len(train_obs)} observations...")
    teacher = BufferGraphPlanner(
        agent,
        train_obs,
        n_landmarks=n_landmarks,
        max_edge_radius=3.5,
        reachability_cutoff=35.0,
        lookahead_dist=2.6,
    )

    print(f"Generating {n_pairs} golden demonstration pairs (noise_sigma={noise_sigma})...")
    rng = np.random.default_rng(seed)
    s_idxs = rng.choice(len(train_obs), size=n_pairs)
    g_idxs = rng.choice(len(train_obs), size=n_pairs)

    raw_states = train_obs[s_idxs].copy()
    raw_goals = train_obs[g_idxs].copy()

    # Apply Gaussian noise to states (specifically positions & velocities)
    noise = rng.normal(0.0, noise_sigma, size=raw_states.shape).astype(np.float32)
    noisy_states = (raw_states + noise).astype(np.float32)

    # Encode global goals B(g_j)
    goal_latents = np.asarray(
        agent.normalize_z(agent.network.select("backward_repr")(raw_goals))
    )

    z_targets = []
    a_targets = []

    # Batch compute next waypoints via Dijkstra and pass through frozen high_actor
    for i in tqdm(range(n_pairs), desc="Dijkstra Waypoints & High-Actor Targets"):
        subgoal_z = teacher.get_subgoal_latent(noisy_states[i], goal_latents[i], step=0)
        z_targets.append(subgoal_z)

    z_targets = np.asarray(z_targets, dtype=np.float32)

    # Compute reference actions a_target = low_actor(s_i, z_target)
    print("Computing reference actions through frozen low-level actor...")
    act_dist = agent.network.select("actor")(noisy_states, z_targets, goal_encoded=True, temperature=0.0)
    a_targets = np.asarray(act_dist.mode())

    dataset = {
        "states": noisy_states,
        "goal_zs": goal_latents,
        "z_targets": z_targets,
        "a_targets": a_targets,
    }

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        np.savez_compressed(
            save_path,
            states=noisy_states,
            goal_zs=goal_latents,
            z_targets=z_targets,
            a_targets=a_targets,
        )
        print(f"Golden dataset saved to {save_path}")

    return dataset


def train_jax_distillation(
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
    use_mlflow: bool = True,
    output_dir: str = "results",
    seed: int = 0,
):
    print("=" * 75)
    print(f"=== Differentiable JAX Latent Distillation: {model_type} on {split} ===")
    print(f"=== Loss: BC(cos+mse) + ActionAlign({l_action}) + Reach({l_reach}) + GoalAlign({l_goal}) ===")
    print("=" * 75)

    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, f"train_jax_distill_{model_type}_{split}.log")

    if use_mlflow:
        try:
            import mlflow
            mlflow.set_experiment(f"jax_latent_distillation_{split}")
            mlflow.start_run(run_name=f"jax_{model_type}_{split}")
            mlflow.log_params({
                "model_type": model_type,
                "split": split,
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
                "n_pairs": n_pairs,
                "noise_sigma": noise_sigma,
                "l_action": l_action,
                "l_reach": l_reach,
                "l_goal": l_goal,
            })
        except Exception:
            pass

    print(f"Loading pretrained FB agent on {split}...")
    agent, env, train_ds, _, _ = load_pretrained_agent(checkpoint_dir, split, seed=seed)

    # 1. Dataset
    data_cache_path = os.path.join(output_dir, f"golden_dataset_{split}_{n_pairs}p_s{int(noise_sigma*100)}.npz")
    dataset = generate_or_load_golden_dataset(
        agent=agent,
        train_obs=train_ds["observations"],
        split=split,
        n_pairs=n_pairs,
        noise_sigma=noise_sigma,
        save_path=data_cache_path,
        seed=seed,
    )

    N = len(dataset["states"])
    n_train = int(0.9 * N)

    train_data = {k: jnp.asarray(v[:n_train]) for k, v in dataset.items()}
    val_data = {
        "state": jnp.asarray(dataset["states"][n_train:]),
        "goal_z": jnp.asarray(dataset["goal_zs"][n_train:]),
        "z_target": jnp.asarray(dataset["z_targets"][n_train:]),
        "a_target": jnp.asarray(dataset["a_targets"][n_train:]),
    }

    # 2. Model & Optimizer
    obs_dim = dataset["states"].shape[-1]
    latent_dim = dataset["goal_zs"].shape[-1]

    student_def = build_flax_translator(
        model_type=model_type,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
    )

    rng = jax.random.PRNGKey(seed)
    dummy_s = jnp.zeros((1, obs_dim))
    dummy_g = jnp.zeros((1, latent_dim))
    init_params = student_def.init(rng, dummy_s, dummy_g)["params"]

    param_count = sum(x.size for x in jax.tree_util.tree_leaves(init_params))
    print(f"Student Model: {model_type} | Parameters: {param_count:,}")

    optimizer = optax.adamw(learning_rate=lr, weight_decay=1e-4)
    opt_state = optimizer.init(init_params)

    lambdas = {
        "l_mse": l_mse,
        "l_cos": l_cos,
        "l_action": l_action,
        "l_reach": l_reach,
        "l_goal": l_goal,
    }

    step_fn = make_train_step(student_def, agent, optimizer)
    eval_fn = make_eval_step(student_def, agent)

    # 3. Training Loop
    params = init_params
    num_batches = n_train // batch_size
    history = []

    print(f"\nStarting JAX Training ({epochs} epochs, {num_batches} batches/epoch)...")
    with open(log_file, "w") as lf:
        lf.write(f"Epoch,Loss,Loss_BC,Loss_Action,Loss_Reach,Loss_Goal,Cos_Sim,Val_Cos_Sim,Val_Action_MSE\n")

        for epoch in range(1, epochs + 1):
            t0 = time.perf_counter()
            rng, perm_rng = jax.random.split(rng)
            perms = jax.random.permutation(perm_rng, n_train)

            epoch_metrics = []
            for b_idx in range(num_batches):
                batch_indices = perms[b_idx * batch_size : (b_idx + 1) * batch_size]
                batch = {
                    "state": train_data["states"][batch_indices],
                    "goal_z": train_data["goal_zs"][batch_indices],
                    "z_target": train_data["z_targets"][batch_indices],
                    "a_target": train_data["a_targets"][batch_indices],
                }

                params, opt_state, metrics = step_fn(params, opt_state, batch, lambdas)
                epoch_metrics.append({k: float(v) for k, v in metrics.items()})

            val_metrics = eval_fn(params, val_data)
            val_metrics = {k: float(v) for k, v in val_metrics.items()}

            avg_train = {k: np.mean([m[k] for m in epoch_metrics]) for k in epoch_metrics[0].keys()}
            elapsed = time.perf_counter() - t0

            log_line = (
                f"Epoch {epoch:02d}/{epochs:02d} [{elapsed:.1f}s] | "
                f"Loss: {avg_train['loss']:.4f} (BC: {avg_train['loss_bc']:.4f}, "
                f"Act: {avg_train['loss_action']:.4f}, Reach: {avg_train['loss_reach']:.4f}, Goal: {avg_train['loss_goal']:.4f}) | "
                f"CosSim: {avg_train['cos_sim']:.4f} | "
                f"Val CosSim: {val_metrics['val_cos_sim']:.4f} | Val ActMSE: {val_metrics['val_action_mse']:.4f}"
            )
            print(log_line)

            csv_row = (
                f"{epoch},{avg_train['loss']:.5f},{avg_train['loss_bc']:.5f},"
                f"{avg_train['loss_action']:.5f},{avg_train['loss_reach']:.5f},{avg_train['loss_goal']:.5f},"
                f"{avg_train['cos_sim']:.5f},{val_metrics['val_cos_sim']:.5f},{val_metrics['val_action_mse']:.5f}\n"
            )
            lf.write(csv_row)
            lf.flush()

            if use_mlflow:
                try:
                    import mlflow
                    mlflow.log_metrics({
                        "train_loss": avg_train["loss"],
                        "train_loss_bc": avg_train["loss_bc"],
                        "train_loss_action": avg_train["loss_action"],
                        "train_loss_reach": avg_train["loss_reach"],
                        "train_loss_goal": avg_train["loss_goal"],
                        "train_cos_sim": avg_train["cos_sim"],
                        "val_cos_sim": val_metrics["val_cos_sim"],
                        "val_action_mse": val_metrics["val_action_mse"],
                    }, step=epoch)
                except Exception:
                    pass

            history.append({
                "epoch": epoch,
                "train": avg_train,
                "val": val_metrics,
                "elapsed": elapsed,
            })

    # Save trained Flax parameters
    save_path = os.path.join(output_dir, f"distilled_jax_{model_type}_{split}.pkl")
    with open(save_path, "wb") as f:
        import pickle
        pickle.dump({"params": flax.core.unfreeze(params), "model_type": model_type, "config": lambdas}, f)
    print(f"\nDistilled JAX model successfully saved to {save_path}")

    if use_mlflow:
        try:
            import mlflow
            mlflow.log_artifact(save_path)
            mlflow.end_run()
        except Exception:
            pass

    # Save history json
    history_file = os.path.join(output_dir, f"history_jax_{model_type}_{split}.json")
    with open(history_file, "w") as f:
        json.dump(history, f, indent=2)

    return params, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="medium")
    parser.add_argument("--model_type", type=str, default="gated_attn")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--n_pairs", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--l_action", type=float, default=0.5)
    parser.add_argument("--l_reach", type=float, default=0.02)
    parser.add_argument("--l_goal", type=float, default=0.05)
    parser.add_argument("--no_mlflow", action="store_true")
    args = parser.parse_args()

    train_jax_distillation(
        split=args.split,
        model_type=args.model_type,
        epochs=args.epochs,
        batch_size=args.batch_size,
        n_pairs=args.n_pairs,
        lr=args.lr,
        l_action=args.l_action,
        l_reach=args.l_reach,
        l_goal=args.l_goal,
        use_mlflow=not args.no_mlflow,
    )
