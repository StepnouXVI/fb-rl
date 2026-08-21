import os
import sys
import time
import json
import pickle
from typing import Dict, Any, Tuple, Sequence
import numpy as np
from tqdm import tqdm
import jax
import jax.numpy as jnp
import flax
import optax
import hydra
from omegaconf import DictConfig, OmegaConf

# Root path setup
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.agent_loader import load_pretrained_agent
from src.planners import BufferGraphPlanner
from src.waypoint_translators import (
    FlaxSingleWaypointTranslator,
    FlaxSequenceWaypointAttentionTranslator,
    make_single_wp_train_step,
    make_sequence_wp_train_step,
)


def generate_waypoint_sequences_dataset(
    agent,
    train_obs: np.ndarray,
    n_pairs: int = 30000,
    noise_sigma: float = 0.05,
    max_seq_len: int = 16,
    lookahead_dist: float = 2.6,
    cache_path: str = None,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    if cache_path and os.path.exists(cache_path):
        print(f"Loading cached waypoint sequences dataset from {cache_path}...")
        data = np.load(cache_path)
        return {
            "states": data["states"],
            "wp_seqs": data["wp_seqs"],
            "seq_masks": data["seq_masks"],
            "w1_zs": data["w1_zs"],
            "final_goals_z": data["final_goals_z"],
            "z_targets": data["z_targets"],
            "a_targets": data["a_targets"],
        }

    print(f"Constructing Dijkstra teacher graph on {len(train_obs)} observations...")
    teacher = BufferGraphPlanner(
        agent,
        train_obs,
        n_landmarks=1000,
        max_edge_radius=3.5,
        reachability_cutoff=35.0,
        lookahead_dist=lookahead_dist,
    )

    print(f"Generating {n_pairs} multi-hop waypoint sequences (max_seq_len={max_seq_len}, noise={noise_sigma})...")
    rng = np.random.default_rng(seed)
    s_idxs = rng.choice(len(train_obs), size=n_pairs)
    g_idxs = rng.choice(len(train_obs), size=n_pairs)

    raw_states = train_obs[s_idxs].copy()
    raw_goals = train_obs[g_idxs].copy()

    # Apply Gaussian noise to states for recovery from drift
    noise = rng.normal(0.0, noise_sigma, size=raw_states.shape).astype(np.float32)
    noisy_states = (raw_states + noise).astype(np.float32)

    final_goals_z = np.asarray(
        agent.normalize_z(agent.network.select("backward_repr")(raw_goals))
    )

    latent_dim = agent.config["latent_dim"]
    wp_seqs = np.zeros((n_pairs, max_seq_len, latent_dim), dtype=np.float32)
    seq_masks = np.zeros((n_pairs, max_seq_len), dtype=bool)
    w1_zs = np.zeros((n_pairs, latent_dim), dtype=np.float32)
    z_targets = np.zeros((n_pairs, latent_dim), dtype=np.float32)

    for i in tqdm(range(n_pairs), desc="Dijkstra Multi-Hop Sequence Extraction"):
        teacher.reset(noisy_states[i], final_goals_z[i])
        subgoal_z = teacher.get_subgoal_latent(noisy_states[i], final_goals_z[i], step=0)
        z_targets[i] = subgoal_z

        remaining_latents = teacher.path_latents[teacher.current_path_idx :]
        if not remaining_latents:
            remaining_latents = [final_goals_z[i]]

        w1_zs[i] = remaining_latents[0]

        K = min(len(remaining_latents), max_seq_len)
        for k in range(K):
            wp_seqs[i, k] = remaining_latents[k]
            seq_masks[i, k] = True

    print("Computing target actions through frozen low-level actor...")
    act_dist = agent.network.select("actor")(noisy_states, z_targets, goal_encoded=True, temperature=0.0)
    a_targets = np.asarray(act_dist.mode())

    dataset = {
        "states": noisy_states,
        "wp_seqs": wp_seqs,
        "seq_masks": seq_masks,
        "w1_zs": w1_zs,
        "final_goals_z": final_goals_z,
        "z_targets": z_targets,
        "a_targets": a_targets,
    }

    if cache_path:
        os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
        np.savez_compressed(cache_path, **dataset)
        print(f"Dataset cached to {cache_path}")

    return dataset


@hydra.main(config_path="../configs", config_name="train_translator", version_base="1.3")
def main(cfg: DictConfig):
    print("=" * 80)
    print(f"=== Training Waypoint Translator: Mode = {cfg.mode.upper()} on {cfg.split.upper()} ===")
    print("=" * 80)
    print(OmegaConf.to_yaml(cfg))

    run_dir = os.getcwd()
    print(f"Hydra Output Directory: {run_dir}")

    # 1. MLflow tracking
    mlflow_active = False
    if cfg.use_mlflow:
        try:
            import mlflow
            mlflow.set_experiment(f"waypoint_translators_{cfg.split}")
            mlflow.start_run(run_name=f"{cfg.mode}_{cfg.split}")
            mlflow.log_params(OmegaConf.to_container(cfg, resolve=True))
            mlflow_active = True
            print("MLflow tracking initialized.")
        except Exception as e:
            print(f"MLflow skipped: {e}")

    # 2. Load agent & Dataset
    agent, env, train_ds, _, _ = load_pretrained_agent(cfg.checkpoint_dir, cfg.split, seed=cfg.seed)

    # Use a shared cache directory for datasets to save time across runs
    cache_dir = os.path.join(PROJECT_ROOT, "outputs", "cache")
    cache_path = os.path.join(cache_dir, f"wp_dataset_{cfg.split}_{cfg.n_pairs}p_k{cfg.max_seq_len}.npz")

    dataset = generate_waypoint_sequences_dataset(
        agent=agent,
        train_obs=train_ds["observations"],
        n_pairs=cfg.n_pairs,
        noise_sigma=cfg.noise_sigma,
        max_seq_len=cfg.max_seq_len,
        lookahead_dist=cfg.lookahead_dist,
        cache_path=cache_path,
        seed=cfg.seed,
    )

    N = len(dataset["states"])
    n_train = int(0.9 * N)
    n_val = N - n_train

    train_data = {k: jnp.asarray(v[:n_train]) for k, v in dataset.items()}
    val_data = {k: jnp.asarray(v[n_train:]) for k, v in dataset.items()}

    # 3. Model Definition & Optimizer
    latent_dim = agent.config["latent_dim"]
    obs_dim = dataset["states"].shape[-1]

    total_steps = (n_train // cfg.batch_size) * cfg.epochs
    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=1e-5,
        peak_value=cfg.lr,
        warmup_steps=int(0.05 * total_steps),
        decay_steps=total_steps,
        end_value=1e-6,
    )

    if cfg.mode == "single_wp":
        model_def = FlaxSingleWaypointTranslator(
            obs_dim=obs_dim,
            latent_dim=latent_dim,
            hidden_dim=cfg.hidden_dim,
            n_layers=cfg.n_layers,
        )
        rng = jax.random.PRNGKey(cfg.seed)
        dummy_s = jnp.zeros((1, obs_dim))
        dummy_w = jnp.zeros((1, latent_dim))
        params = model_def.init(rng, dummy_s, dummy_w)["params"]

        def frozen_actor_fn(s, z, goal_encoded=True, temperature=0.0):
            return agent.network.select("actor")(s, z, goal_encoded=goal_encoded, temperature=temperature)

        def frozen_f_fn(s, z, goal_encoded=True):
            return agent.network.select("forward_repr")(s, z, goal_encoded=goal_encoded)

        def frozen_b_fn(g):
            return agent.network.select("backward_repr")(g)

        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(learning_rate=lr_schedule, weight_decay=1e-4),
        )
        opt_state = optimizer.init(params)
        step_fn = make_single_wp_train_step(model_def.apply, frozen_actor_fn, frozen_f_fn, frozen_b_fn, optimizer)

        @jax.jit
        def eval_fn(p, val_batch):
            pred_z = model_def.apply({"params": p}, val_batch["states"], val_batch["w1_zs"])
            cos_sim = jnp.sum(pred_z * val_batch["z_targets"], axis=-1) / (
                jnp.linalg.norm(pred_z, axis=-1) * jnp.linalg.norm(val_batch["z_targets"], axis=-1) + 1e-8
            )
            act_dist = frozen_actor_fn(val_batch["states"], pred_z, goal_encoded=True, temperature=0.0)
            pred_a = act_dist.mode()
            act_mse = jnp.mean((pred_a - val_batch["a_targets"]) ** 2)
            return {"val_cos_sim": jnp.mean(cos_sim), "val_action_mse": act_mse}

    elif cfg.mode == "sequence_attn":
        model_def = FlaxSequenceWaypointAttentionTranslator(
            obs_dim=obs_dim,
            latent_dim=latent_dim,
            hidden_dim=cfg.hidden_dim,
            num_heads=cfg.num_heads,
            max_seq_len=cfg.max_seq_len,
            n_layers=cfg.n_layers,
        )
        rng = jax.random.PRNGKey(cfg.seed)
        dummy_s = jnp.zeros((1, obs_dim))
        dummy_seq = jnp.zeros((1, cfg.max_seq_len, latent_dim))
        dummy_mask = jnp.ones((1, cfg.max_seq_len), dtype=bool)
        params = model_def.init(rng, dummy_s, dummy_seq, dummy_mask)["params"]

        def frozen_actor_fn(s, z, goal_encoded=True, temperature=0.0):
            return agent.network.select("actor")(s, z, goal_encoded=goal_encoded, temperature=temperature)

        def frozen_f_fn(s, z, goal_encoded=True):
            return agent.network.select("forward_repr")(s, z, goal_encoded=goal_encoded)

        def frozen_b_fn(g):
            return agent.network.select("backward_repr")(g)

        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(learning_rate=lr_schedule, weight_decay=1e-4),
        )
        opt_state = optimizer.init(params)
        step_fn = make_sequence_wp_train_step(model_def.apply, frozen_actor_fn, frozen_f_fn, frozen_b_fn, optimizer)

        @jax.jit
        def eval_fn(p, val_batch):
            pred_z = model_def.apply({"params": p}, val_batch["states"], val_batch["wp_seqs"], val_batch["seq_masks"])
            cos_sim = jnp.sum(pred_z * val_batch["z_targets"], axis=-1) / (
                jnp.linalg.norm(pred_z, axis=-1) * jnp.linalg.norm(val_batch["z_targets"], axis=-1) + 1e-8
            )
            act_dist = frozen_actor_fn(val_batch["states"], pred_z, goal_encoded=True, temperature=0.0)
            pred_a = act_dist.mode()
            act_mse = jnp.mean((pred_a - val_batch["a_targets"]) ** 2)
            return {"val_cos_sim": jnp.mean(cos_sim), "val_action_mse": act_mse}

    else:
        raise ValueError(f"Unknown mode: {cfg.mode}")

    # 4. Training Loop
    lambdas = {
        "l_cos": cfg.l_cos,
        "l_mse": cfg.l_mse,
        "l_action": cfg.l_action,
        "l_reach": cfg.l_reach,
        "l_goal": cfg.l_goal,
    }

    num_batches = n_train // cfg.batch_size
    history = []
    log_file = os.path.join(run_dir, f"training_{cfg.mode}_{cfg.split}.log")

    best_val_mse = float("inf")
    best_params = params

    print(f"\nStarting {cfg.mode.upper()} Training on RTX 4070 ({cfg.epochs} epochs, {num_batches} batches/epoch)...")
    with open(log_file, "w") as lf:
        lf.write("Epoch,Loss,Loss_BC,Loss_Action,Loss_Reach,Loss_Goal,CosSim,Val_CosSim,Val_ActMSE\n")

        for epoch in range(1, cfg.epochs + 1):
            t0 = time.perf_counter()
            rng, perm_rng = jax.random.split(rng)
            perms = jax.random.permutation(perm_rng, n_train)

            epoch_metrics = []
            for b_idx in range(num_batches):
                batch_indices = perms[b_idx * cfg.batch_size : (b_idx + 1) * cfg.batch_size]
                if cfg.mode == "single_wp":
                    batch = {
                        "state": train_data["states"][batch_indices],
                        "w1_z": train_data["w1_zs"][batch_indices],
                        "final_goal_z": train_data["final_goals_z"][batch_indices],
                        "z_target": train_data["z_targets"][batch_indices],
                        "a_target": train_data["a_targets"][batch_indices],
                    }
                else:
                    batch = {
                        "state": train_data["states"][batch_indices],
                        "wp_seq": train_data["wp_seqs"][batch_indices],
                        "seq_mask": train_data["seq_masks"][batch_indices],
                        "final_goal_z": train_data["final_goals_z"][batch_indices],
                        "z_target": train_data["z_targets"][batch_indices],
                        "a_target": train_data["a_targets"][batch_indices],
                    }

                params, opt_state, metrics = step_fn(params, opt_state, batch, lambdas)
                epoch_metrics.append({k: float(v) for k, v in metrics.items()})

            val_metrics = eval_fn(params, val_data)
            val_metrics = {k: float(v) for k, v in val_metrics.items()}
            avg_train = {k: np.mean([m[k] for m in epoch_metrics]) for k in epoch_metrics[0].keys()}
            elapsed = time.perf_counter() - t0

            if val_metrics["val_action_mse"] < best_val_mse:
                best_val_mse = val_metrics["val_action_mse"]
                best_params = params

            log_line = (
                f"Epoch {epoch:04d}/{cfg.epochs:04d} [{elapsed:.2f}s] | "
                f"Loss: {avg_train['loss']:.4f} (BC: {avg_train['loss_bc']:.4f}, Act: {avg_train['loss_action']:.4f}, "
                f"Reach: {avg_train['loss_reach']:.4f}, Goal: {avg_train['loss_goal']:.4f}) | "
                f"Train CosSim: {avg_train['cos_sim']:.4f} | "
                f"Val CosSim: {val_metrics['val_cos_sim']:.4f} | Val ActMSE: {val_metrics['val_action_mse']:.4f} (Best: {best_val_mse:.4f})"
            )
            print(log_line)

            csv_row = (
                f"{epoch},{avg_train['loss']:.5f},{avg_train['loss_bc']:.5f},"
                f"{avg_train['loss_action']:.5f},{avg_train['loss_reach']:.5f},{avg_train['loss_goal']:.5f},"
                f"{avg_train['cos_sim']:.5f},{val_metrics['val_cos_sim']:.5f},{val_metrics['val_action_mse']:.5f}\n"
            )
            lf.write(csv_row)
            lf.flush()

            if mlflow_active:
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

    # Save model checkpoint in the Hydra run folder
    save_path = os.path.join(run_dir, f"checkpoint_{cfg.mode}_{cfg.split}.pkl")
    with open(save_path, "wb") as f:
        pickle.dump({
            "params": flax.core.unfreeze(best_params),
            "mode": cfg.mode,
            "split": cfg.split,
            "config": OmegaConf.to_container(cfg, resolve=True),
        }, f)
    print(f"\nModel checkpoint successfully saved to {save_path}")

    # Also save a canonical symlink/copy in outputs/checkpoints/ for evaluation scripts
    canonical_dir = os.path.join(PROJECT_ROOT, "outputs", "checkpoints")
    os.makedirs(canonical_dir, exist_ok=True)
    canonical_path = os.path.join(canonical_dir, f"best_{cfg.mode}_{cfg.split}.pkl")
    with open(canonical_path, "wb") as f:
        pickle.dump({
            "params": flax.core.unfreeze(best_params),
            "mode": cfg.mode,
            "split": cfg.split,
            "config": OmegaConf.to_container(cfg, resolve=True),
        }, f)
    print(f"Canonical checkpoint updated at {canonical_path}")

    if mlflow_active:
        try:
            import mlflow
            mlflow.log_artifact(save_path)
            mlflow.end_run()
        except Exception:
            pass


if __name__ == "__main__":
    main()
