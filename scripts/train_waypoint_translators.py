"""Train single-waypoint and sequence-attention neural intention translators."""

import os
import pickle
import sys
import time
from typing import Any, Dict, Optional, Tuple
import flax
import hydra
import jax
import jax.numpy as jnp
import numpy as np
import optax
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.agent_loader import load_pretrained_agent
from src.networks import SequenceAttentionNetwork, SingleWaypointNetwork
from src.telemetry.db import TelemetryDatabase
from src.topology import DijkstraGraph
from src.training import (
    make_sequence_attention_train_step,
    make_single_waypoint_train_step,
)


def _extract_sequence_item(
    teacher: DijkstraGraph,
    raw_start: np.ndarray,
    final_goal_z: np.ndarray,
    agent: Any,
    noise_sigma: float,
    max_seq_len: int,
    lookahead_dist: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract waypoint sequence, curvature annotations, and target intentions."""
    coords, latents, states = teacher.plan_path(raw_start[:2], final_goal_z)
    path_len = len(coords)
    t = int(rng.integers(0, max(path_len, 1)))
    base_state = states[t] if t < len(states) else raw_start
    s_t = (base_state + rng.normal(0.0, noise_sigma, size=base_state.shape)).astype(np.float32)

    dist_to_final = float(np.linalg.norm(s_t[:2] - coords[-1]))
    accum, target_idx = 0.0, t
    while target_idx < path_len - 1 and accum < lookahead_dist:
        accum += float(np.linalg.norm(coords[target_idx + 1] - coords[target_idx]))
        target_idx += 1

    is_end = bool(target_idx >= path_len - 1 or dist_to_final <= 2.2)
    target_latent = final_goal_z if is_end else latents[target_idx]
    future_latents = [final_goal_z] if is_end else (latents[target_idx:] + [final_goal_z])
    future_coords = [coords[-1]] if is_end else (coords[target_idx:] + [coords[-1]])

    high_dist = agent.network.select("high_actor")(
        s_t[None, :], target_latent[None, :], goal_encoded=True, temperature=0.0
    )
    z_target = np.asarray(agent.normalize_z(high_dist.mode())[0])

    k_len = min(len(future_latents), max_seq_len)
    wp_seq = np.zeros((max_seq_len, len(final_goal_z)), dtype=np.float32)
    seq_mask = np.zeros(max_seq_len, dtype=bool)
    curvs = np.ones((max_seq_len, 1), dtype=np.float32)
    for k in range(k_len):
        wp_seq[k] = future_latents[k]
        seq_mask[k] = True
        if k + 2 < len(future_coords):
            v1 = future_coords[k + 1] - future_coords[k]
            v2 = future_coords[k + 2] - future_coords[k + 1]
            l1, l2 = float(np.linalg.norm(v1)), float(np.linalg.norm(v2))
            cos_v = float(np.clip(np.dot(v1, v2) / (l1 * l2), -1.0, 1.0)) if (l1 > 1e-4 and l2 > 1e-4) else 1.0
            curvs[k, 0] = cos_v
    return s_t, wp_seq, seq_mask, curvs, target_latent, z_target


def generate_waypoint_sequences_dataset(
    agent: Any,
    train_obs: np.ndarray,
    n_pairs: int = 60000,
    noise_sigma: float = 0.05,
    max_seq_len: int = 16,
    lookahead_dist: float = 2.6,
    split: str = "medium",
    cache_path: Optional[str] = None,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """Build or load training dataset containing waypoint sequences and intentions."""
    if cache_path and os.path.exists(cache_path):
        print(f"Loading cached dataset from {cache_path}...")
        d = np.load(cache_path)
        curvs = d["curvatures"] if "curvatures" in d else np.ones((len(d["states"]), max_seq_len, 1), dtype=np.float32)
        return {
            k: d[k] for k in ["states", "wp_seqs", "seq_masks", "w1_zs", "final_goals_z", "z_targets", "a_targets"]
        } | {"curvatures": curvs}

    n_landmarks = 2000 if split == "large" else 1000
    teacher = DijkstraGraph(agent, train_obs, n_landmarks=n_landmarks, lookahead_dist=lookahead_dist)
    rng = np.random.default_rng(seed)
    s_idxs = rng.choice(len(train_obs), size=n_pairs)
    g_idxs = rng.choice(len(train_obs), size=n_pairs)
    final_goals_z = np.asarray(
        agent.normalize_z(agent.network.select("backward_repr")(train_obs[g_idxs]))
    )

    states = np.zeros((n_pairs, train_obs.shape[-1]), np.float32)
    wp_seqs = np.zeros((n_pairs, max_seq_len, agent.config["latent_dim"]), np.float32)
    seq_masks = np.zeros((n_pairs, max_seq_len), bool)
    curvs = np.ones((n_pairs, max_seq_len, 1), np.float32)
    w1_zs = np.zeros((n_pairs, agent.config["latent_dim"]), np.float32)
    z_targets = np.zeros((n_pairs, agent.config["latent_dim"]), np.float32)

    for i in tqdm(range(n_pairs), desc="Dijkstra Sequences"):
        s_t, w_s, s_m, c_s, w1, zt = _extract_sequence_item(
            teacher, train_obs[s_idxs[i]], final_goals_z[i], agent, noise_sigma, max_seq_len, lookahead_dist, rng
        )
        states[i], wp_seqs[i], seq_masks[i], curvs[i], w1_zs[i], z_targets[i] = s_t, w_s, s_m, c_s, w1, zt

    a_targets = np.asarray(
        agent.network.select("actor")(states, z_targets, goal_encoded=True, temperature=0.0).mode()
    )
    dataset = {
        "states": states, "wp_seqs": wp_seqs, "seq_masks": seq_masks, "curvatures": curvs,
        "w1_zs": w1_zs, "final_goals_z": final_goals_z, "z_targets": z_targets, "a_targets": a_targets,
    }
    if cache_path:
        os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
        np.savez_compressed(cache_path, **dataset)
    return dataset


def _setup_model_and_step_fn(
    cfg: Any, obs_dim: int, latent_dim: int, frozen_fns: Dict[str, Any], optimizer: Any
) -> Tuple[Any, Any, Any, Any]:
    """Initialize neural translator module, parameters, and compiled train step."""
    if cfg.mode in ["single_wp", "single_waypoint"]:
        m = SingleWaypointNetwork(latent_dim=latent_dim, hidden_dim=cfg.hidden_dim, n_layers=cfg.n_layers)
        params = m.init(jax.random.PRNGKey(cfg.seed), jnp.zeros((1, obs_dim)), jnp.zeros((1, latent_dim)))["params"]
        step_fn = make_single_waypoint_train_step(
            m.apply, frozen_fns["actor"], frozen_fns["f"], frozen_fns["b"], optimizer
        )
    elif cfg.mode in ["sequence_attention", "enhanced_seq_attn", "sequence_attn", "enhanced_sequence_attn"]:
        m = SequenceAttentionNetwork(
            latent_dim=latent_dim,
            hidden_dim=cfg.hidden_dim,
            num_heads=cfg.num_heads,
            max_seq_len=cfg.max_seq_len,
            n_layers=cfg.n_layers,
            dropout_rate=float(cfg.get("dropout_rate", 0.2)),
            alibi_slope=float(cfg.get("alibi_slope", 0.4)),
        )
        rng = jax.random.PRNGKey(cfg.seed)
        params = m.init(
            {"params": rng, "dropout": rng},
            jnp.zeros((1, obs_dim)),
            jnp.zeros((1, cfg.max_seq_len, latent_dim)),
            jnp.ones((1, cfg.max_seq_len), bool),
            jnp.ones((1, cfg.max_seq_len, 1)),
            deterministic=False,
        )["params"]
        step_fn = make_sequence_attention_train_step(
            m.apply, frozen_fns["actor"], frozen_fns["f"], frozen_fns["b"], optimizer
        )
    else:
        raise ValueError(f"Unknown mode: {cfg.mode}")
    return m, params, optimizer.init(params), step_fn


def _run_training_epoch(
    mode: str,
    step_fn: Any,
    params: Any,
    opt_state: Any,
    train_data: Dict[str, jnp.ndarray],
    batch_size: int,
    lambdas: Dict[str, float],
    rng: Any,
    n_train: int,
) -> Tuple[Any, Any, Any, Dict[str, float]]:
    """Iterate over minibatches for one training epoch."""
    rng, perm_rng = jax.random.split(rng)
    perms = jax.random.permutation(perm_rng, n_train)
    actual_bs = min(batch_size, n_train)
    num_batches = max(1, n_train // actual_bs)
    ep_metrics = []
    for b in range(num_batches):
        idx = perms[b * actual_bs : (b + 1) * actual_bs if b < num_batches - 1 else n_train]
        if len(idx) == 0:
            continue
        if mode in ["single_wp", "single_waypoint"]:
            batch = {
                "state": train_data["states"][idx], "w1_z": train_data["w1_zs"][idx],
                "final_goal_z": train_data["final_goals_z"][idx],
                "z_target": train_data["z_targets"][idx], "a_target": train_data["a_targets"][idx],
            }
            params, opt_state, m = step_fn(params, opt_state, batch, lambdas)
        else:
            batch = {
                "state": train_data["states"][idx], "wp_seq": train_data["wp_seqs"][idx],
                "seq_mask": train_data["seq_masks"][idx], "curvatures": train_data["curvatures"][idx],
                "final_goal_z": train_data["final_goals_z"][idx],
                "z_target": train_data["z_targets"][idx], "a_target": train_data["a_targets"][idx],
            }
            params, opt_state, m, rng = step_fn(params, opt_state, batch, lambdas, rng)
        ep_metrics.append({k: float(v) for k, v in m.items()})
    return params, opt_state, rng, {k: float(np.mean([m[k] for m in ep_metrics])) for k in ep_metrics[0]}


def _save_translator_checkpoints(
    best_params: Any, cfg: Any, run_dir: str, save_path: str
) -> str:
    """Write model parameters to checkpoint files in local hydra and canonical dirs."""
    payload = {
        "params": flax.core.unfreeze(best_params),
        "mode": cfg.mode,
        "split": cfg.split,
        "config": OmegaConf.to_container(cfg, resolve=True),
    }
    with open(save_path, "wb") as f:
        pickle.dump(payload, f)
    canonical_dir = os.path.join(PROJECT_ROOT, "outputs", "checkpoints")
    os.makedirs(canonical_dir, exist_ok=True)
    if cfg.mode in ["sequence_attention", "sequence_attn"]:
        canonical_path = os.path.join(canonical_dir, f"best_sequence_attention_{cfg.split}.pkl")
    elif cfg.mode in ["enhanced_seq_attn", "enhanced_sequence_attn"]:
        canonical_path = os.path.join(canonical_dir, f"best_enhanced_sequence_attn_{cfg.split}.pkl")
    elif cfg.mode in ["single_wp", "single_waypoint"]:
        canonical_path = os.path.join(canonical_dir, f"best_single_waypoint_{cfg.split}.pkl")
    else:
        canonical_path = os.path.join(canonical_dir, f"best_{cfg.mode}_{cfg.split}.pkl")
    with open(canonical_path, "wb") as f:
        pickle.dump(payload, f)
    return canonical_path


def _prepare_train_data(
    cfg: Any, agent: Any, train_ds: Any
) -> Tuple[Dict[str, jnp.ndarray], int]:
    """Prepare and slice waypoint sequence dataset into training arrays."""
    cache_path = os.path.join(
        PROJECT_ROOT, "results", "datasets", f"wp_dataset_{cfg.split}_{cfg.n_pairs}p_k{cfg.max_seq_len}.npz"
    )
    dataset = generate_waypoint_sequences_dataset(
        agent, train_ds["observations"], cfg.n_pairs, cfg.noise_sigma,
        cfg.max_seq_len, cfg.lookahead_dist, cfg.split, cache_path, cfg.seed,
    )
    n_train = int(0.9 * len(dataset["states"]))
    return {k: jnp.asarray(v[:n_train]) for k, v in dataset.items()}, n_train


def _execute_translator_epochs(
    cfg: Any,
    step_fn: Any,
    params: Any,
    opt_state: Any,
    train_data: Dict[str, jnp.ndarray],
    lr_sched: Any,
    lambdas: Dict[str, float],
    rng: Any,
    n_train: int,
    db: TelemetryDatabase,
    train_run_id: str,
) -> Tuple[Any, Optional[str]]:
    """Execute training epochs, record telemetry to SQLite, and persist checkpoints."""
    best_ckpt = None
    save_path = os.path.join(os.getcwd(), f"best_{cfg.mode}_{cfg.split}.pkl")
    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        params, opt_state, rng, tr_m = _run_training_epoch(
            cfg.mode, step_fn, params, opt_state, train_data, cfg.batch_size, lambdas, rng, n_train
        )
        ep_time = time.time() - t0
        step_idx = (epoch - 1) * max(1, n_train // cfg.batch_size)
        curr_lr = float(lr_sched(step_idx))
        ckpt_path = None
        if epoch == cfg.epochs:
            ckpt_path = _save_translator_checkpoints(params, cfg, os.getcwd(), save_path)
            best_ckpt = ckpt_path
        db.insert_training_epoch(
            train_run_id, epoch, tr_m,
            checkpoint_path=ckpt_path, epoch_time_s=ep_time, learning_rate=curr_lr,
        )
        if epoch % 50 == 0 or epoch == cfg.epochs:
            print(f"Epoch {epoch:4d}/{cfg.epochs} | Loss: {tr_m['loss']:.4f} | CosSim: {tr_m['cos_sim']:.4f}")
    return params, best_ckpt


@hydra.main(config_path="../configs", config_name="train_translator", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Hydra entry point for training waypoint translator networks."""
    agent, _, train_ds, _, _ = load_pretrained_agent(cfg.checkpoint_dir, cfg.split, seed=cfg.seed)
    train_data, n_train = _prepare_train_data(cfg, agent, train_ds)
    total_steps = max(1, (n_train // cfg.batch_size) * cfg.epochs)
    warmup_steps = min(total_steps - 1, max(0, int(0.05 * total_steps)))
    lr_sched = optax.warmup_cosine_decay_schedule(1e-5, cfg.lr, warmup_steps, total_steps, 1e-6)
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0), optax.adamw(learning_rate=lr_sched, weight_decay=1e-4)
    )
    frozen_fns = {
        "actor": agent.network.select("actor"),
        "f": agent.network.select("forward_repr"),
        "b": agent.network.select("backward_repr"),
    }
    _, params, opt_state, step_fn = _setup_model_and_step_fn(
        cfg, train_data["states"].shape[-1], agent.config["latent_dim"], frozen_fns, optimizer
    )
    lambdas = {
        "l_cos": cfg.l_cos, "l_mse": cfg.l_mse, "l_action": cfg.l_action,
        "l_reach": cfg.l_reach, "l_goal": cfg.l_goal, "l_aux": float(cfg.get("l_aux", 0.2)),
    }
    db_path = getattr(cfg, "db_path", "results/telemetry.db")
    db_file = db_path if os.path.isabs(db_path) else os.path.join(PROJECT_ROOT, db_path)
    os.makedirs(os.path.dirname(os.path.abspath(db_file)), exist_ok=True)
    db = TelemetryDatabase(db_file)
    run_name = f"translator_{cfg.mode}_{cfg.split}_s{cfg.seed}"
    try:
        train_run_id = db.create_training_run(
            run_name=run_name, model_type=cfg.mode, split=cfg.split,
            seed=int(cfg.seed), config=OmegaConf.to_container(cfg, resolve=True),
        )
        _, best_ckpt = _execute_translator_epochs(
            cfg, step_fn, params, opt_state, train_data, lr_sched,
            lambdas, jax.random.PRNGKey(cfg.seed), n_train, db, train_run_id,
        )
        db.finish_training_run(train_run_id, best_checkpoint_path=best_ckpt, status="completed")
    finally:
        db.close()


if __name__ == "__main__":
    main()
