#!/usr/bin/env python3
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

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.agent_loader import load_pretrained_agent
from src.planners import BufferGraphPlanner
from src.waypoint_translators import (
    FlaxSingleWaypointTranslator,
    FlaxSequenceWaypointAttentionTranslator,
    FlaxEnhancedSequenceWaypointAttentionTranslator,
    make_single_wp_train_step,
    make_sequence_wp_train_step,
    make_enhanced_sequence_wp_train_step,
)


def _extract_sequence_item(teacher, raw_start, final_goal_z, agent, noise_sigma, max_seq_len, lookahead_dist, rng):
    teacher.reset(raw_start, final_goal_z)
    path_len = len(teacher.path_coords)
    t = int(rng.integers(0, max(path_len, 1)))
    base_state = teacher.path_states[t] if t < len(teacher.path_states) else raw_start
    s_t = (base_state + rng.normal(0.0, noise_sigma, size=base_state.shape)).astype(np.float32)

    dist_to_final = float(np.linalg.norm(s_t[:2] - teacher.path_coords[-1]))
    accum, target_idx = 0.0, t
    while target_idx < path_len - 1 and accum < lookahead_dist:
        accum += np.linalg.norm(teacher.path_coords[target_idx + 1] - teacher.path_coords[target_idx])
        target_idx += 1

    is_end = (target_idx >= path_len - 1 or dist_to_final <= 2.2)
    target_latent = final_goal_z if is_end else teacher.path_latents[target_idx]
    future_latents = [final_goal_z] if is_end else (teacher.path_latents[target_idx:] + [final_goal_z])
    future_coords = [teacher.path_coords[-1]] if is_end else (teacher.path_coords[target_idx:] + [teacher.path_coords[-1]])

    high_dist = agent.network.select("high_actor")(s_t[None, :], target_latent[None, :], goal_encoded=True, temperature=0.0)
    z_target = np.asarray(agent.normalize_z(high_dist.mode())[0])

    K = min(len(future_latents), max_seq_len)
    wp_seq, seq_mask, curvs = np.zeros((max_seq_len, len(final_goal_z)), dtype=np.float32), np.zeros(max_seq_len, dtype=bool), np.ones((max_seq_len, 1), dtype=np.float32)
    for k in range(K):
        wp_seq[k], seq_mask[k] = future_latents[k], True
        if k + 2 < len(future_coords):
            v1, v2 = future_coords[k + 1] - future_coords[k], future_coords[k + 2] - future_coords[k + 1]
            l1, l2 = np.linalg.norm(v1), np.linalg.norm(v2)
            curvs[k, 0] = float(np.clip(np.dot(v1, v2) / (l1 * l2), -1.0, 1.0)) if (l1 > 1e-4 and l2 > 1e-4) else 1.0
    return s_t, wp_seq, seq_mask, curvs, target_latent, z_target


def generate_waypoint_sequences_dataset(agent, train_obs, n_pairs=60000, noise_sigma=0.05, max_seq_len=16, lookahead_dist=2.6, split="medium", cache_path=None, seed=42):
    if cache_path and os.path.exists(cache_path):
        print(f"Loading cached dataset from {cache_path}...")
        d = np.load(cache_path)
        curvs = d["curvatures"] if "curvatures" in d else np.ones((len(d["states"]), max_seq_len, 1), dtype=np.float32)
        return {k: d[k] for k in ["states", "wp_seqs", "seq_masks", "w1_zs", "final_goals_z", "z_targets", "a_targets"]} | {"curvatures": curvs}

    teacher = BufferGraphPlanner(agent, train_obs, n_landmarks=(2000 if split == "large" else 1000), lookahead_dist=lookahead_dist)
    rng = np.random.default_rng(seed)
    s_idxs, g_idxs = rng.choice(len(train_obs), size=n_pairs), rng.choice(len(train_obs), size=n_pairs)
    final_goals_z = np.asarray(agent.normalize_z(agent.network.select("backward_repr")(train_obs[g_idxs])))

    states, wp_seqs, seq_masks, curvs, w1_zs, z_targets = (
        np.zeros((n_pairs, train_obs.shape[-1]), np.float32), np.zeros((n_pairs, max_seq_len, agent.config["latent_dim"]), np.float32),
        np.zeros((n_pairs, max_seq_len), bool), np.ones((n_pairs, max_seq_len, 1), np.float32),
        np.zeros((n_pairs, agent.config["latent_dim"]), np.float32), np.zeros((n_pairs, agent.config["latent_dim"]), np.float32)
    )

    for i in tqdm(range(n_pairs), desc="Dijkstra Multi-Stage Sequences"):
        s_t, w_s, s_m, c_s, w1, zt = _extract_sequence_item(teacher, train_obs[s_idxs[i]], final_goals_z[i], agent, noise_sigma, max_seq_len, lookahead_dist, rng)
        states[i], wp_seqs[i], seq_masks[i], curvs[i], w1_zs[i], z_targets[i] = s_t, w_s, s_m, c_s, w1, zt

    a_targets = np.asarray(agent.network.select("actor")(states, z_targets, goal_encoded=True, temperature=0.0).mode())
    dataset = {"states": states, "wp_seqs": wp_seqs, "seq_masks": seq_masks, "curvatures": curvs, "w1_zs": w1_zs, "final_goals_z": final_goals_z, "z_targets": z_targets, "a_targets": a_targets}
    if cache_path:
        os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
        np.savez_compressed(cache_path, **dataset)
    return dataset


def _setup_model_and_step_fn(cfg, obs_dim, latent_dim, frozen_fns, optimizer):
    is_enhanced = cfg.mode in ["enhanced_seq_attn", "enhanced_sequence_attn"]
    if cfg.mode == "single_wp":
        m = FlaxSingleWaypointTranslator(obs_dim=obs_dim, latent_dim=latent_dim, hidden_dim=cfg.hidden_dim, n_layers=cfg.n_layers)
        params = m.init(jax.random.PRNGKey(cfg.seed), jnp.zeros((1, obs_dim)), jnp.zeros((1, latent_dim)))["params"]
        step_fn = make_single_wp_train_step(m.apply, frozen_fns["actor"], frozen_fns["f"], frozen_fns["b"], optimizer)
    elif cfg.mode == "sequence_attn":
        m = FlaxSequenceWaypointAttentionTranslator(obs_dim=obs_dim, latent_dim=latent_dim, hidden_dim=cfg.hidden_dim, num_heads=cfg.num_heads, max_seq_len=cfg.max_seq_len, n_layers=cfg.n_layers)
        params = m.init(jax.random.PRNGKey(cfg.seed), jnp.zeros((1, obs_dim)), jnp.zeros((1, cfg.max_seq_len, latent_dim)), jnp.ones((1, cfg.max_seq_len), bool))["params"]
        step_fn = make_sequence_wp_train_step(m.apply, frozen_fns["actor"], frozen_fns["f"], frozen_fns["b"], optimizer)
    elif is_enhanced:
        m = FlaxEnhancedSequenceWaypointAttentionTranslator(obs_dim=obs_dim, latent_dim=latent_dim, hidden_dim=cfg.hidden_dim, num_heads=cfg.num_heads, max_seq_len=cfg.max_seq_len, n_layers=cfg.n_layers, dropout_rate=float(cfg.get("dropout_rate", 0.2)), alibi_slope=float(cfg.get("alibi_slope", 0.4)), local_window_size=int(cfg.get("local_window_size", 6)))
        rng = jax.random.PRNGKey(cfg.seed)
        params = m.init({"params": rng, "dropout": rng}, jnp.zeros((1, obs_dim)), jnp.zeros((1, cfg.max_seq_len, latent_dim)), jnp.ones((1, cfg.max_seq_len), bool), jnp.ones((1, cfg.max_seq_len, 1)), deterministic=False)["params"]
        step_fn = make_enhanced_sequence_wp_train_step(m.apply, frozen_fns["actor"], frozen_fns["f"], frozen_fns["b"], optimizer)
    else:
        raise ValueError(f"Unknown mode: {cfg.mode}")
    return m, params, optimizer.init(params), step_fn


def _run_training_epoch(mode, step_fn, params, opt_state, train_data, batch_size, lambdas, rng, n_train):
    rng, perm_rng = jax.random.split(rng)
    perms = jax.random.permutation(perm_rng, n_train)
    num_batches, ep_metrics = n_train // batch_size, []
    for b in range(num_batches):
        idx = perms[b * batch_size : (b + 1) * batch_size]
        if mode == "single_wp":
            batch = {"state": train_data["states"][idx], "w1_z": train_data["w1_zs"][idx], "final_goal_z": train_data["final_goals_z"][idx], "z_target": train_data["z_targets"][idx], "a_target": train_data["a_targets"][idx]}
            params, opt_state, m = step_fn(params, opt_state, batch, lambdas)
        elif mode == "sequence_attn":
            batch = {"state": train_data["states"][idx], "wp_seq": train_data["wp_seqs"][idx], "seq_mask": train_data["seq_masks"][idx], "final_goal_z": train_data["final_goals_z"][idx], "z_target": train_data["z_targets"][idx], "a_target": train_data["a_targets"][idx]}
            params, opt_state, m = step_fn(params, opt_state, batch, lambdas)
        else:
            batch = {"state": train_data["states"][idx], "wp_seq": train_data["wp_seqs"][idx], "seq_mask": train_data["seq_masks"][idx], "curvatures": train_data["curvatures"][idx], "final_goal_z": train_data["final_goals_z"][idx], "z_target": train_data["z_targets"][idx], "a_target": train_data["a_targets"][idx]}
            params, opt_state, m, rng = step_fn(params, opt_state, batch, lambdas, rng)
        ep_metrics.append({k: float(v) for k, v in m.items()})
    return params, opt_state, rng, {k: float(np.mean([m[k] for m in ep_metrics])) for k in ep_metrics[0]}


def _save_translator_checkpoints(best_params, cfg, run_dir, save_path, mlflow_active):
    payload = {"params": flax.core.unfreeze(best_params), "mode": cfg.mode, "split": cfg.split, "config": OmegaConf.to_container(cfg, resolve=True)}
    with open(save_path, "wb") as f:
        pickle.dump(payload, f)
    canonical_dir = os.path.join(PROJECT_ROOT, "outputs", "checkpoints")
    os.makedirs(canonical_dir, exist_ok=True)
    with open(os.path.join(canonical_dir, f"best_{cfg.mode}_{cfg.split}.pkl"), "wb") as f:
        pickle.dump(payload, f)
    if cfg.mode == "enhanced_seq_attn":
        with open(os.path.join(canonical_dir, f"best_enhanced_sequence_attn_{cfg.split}.pkl"), "wb") as f:
            pickle.dump(payload, f)
    if mlflow_active:
        try:
            import mlflow
            mlflow.log_artifact(save_path)
            mlflow.end_run()
        except Exception:
            pass


@hydra.main(config_path="../configs", config_name="train_translator", version_base="1.3")
def main(cfg: DictConfig):
    agent, env, train_ds, _, _ = load_pretrained_agent(cfg.checkpoint_dir, cfg.split, seed=cfg.seed)
    cache_path = os.path.join(PROJECT_ROOT, "results", "datasets", f"wp_dataset_{cfg.split}_{cfg.n_pairs}p_k{cfg.max_seq_len}.npz")
    dataset = generate_waypoint_sequences_dataset(agent, train_ds["observations"], cfg.n_pairs, cfg.noise_sigma, cfg.max_seq_len, cfg.lookahead_dist, cfg.split, cache_path, cfg.seed)

    N = len(dataset["states"])
    n_train = int(0.9 * N)
    train_data = {k: jnp.asarray(v[:n_train]) for k, v in dataset.items()}
    val_data = {k: jnp.asarray(v[n_train:]) for k, v in dataset.items()}

    lr_schedule = optax.warmup_cosine_decay_schedule(1e-5, cfg.lr, int(0.05 * (n_train // cfg.batch_size) * cfg.epochs), (n_train // cfg.batch_size) * cfg.epochs, 1e-6)
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(learning_rate=lr_schedule, weight_decay=1e-4))
    frozen_fns = {"actor": agent.network.select("actor"), "f": agent.network.select("forward_repr"), "b": agent.network.select("backward_repr")}

    model_def, params, opt_state, step_fn = _setup_model_and_step_fn(cfg, dataset["states"].shape[-1], agent.config["latent_dim"], frozen_fns, optimizer)
    lambdas = {"l_cos": cfg.l_cos, "l_mse": cfg.l_mse, "l_action": cfg.l_action, "l_reach": cfg.l_reach, "l_goal": cfg.l_goal, "l_aux": float(cfg.get("l_aux", 0.2))}

    rng = jax.random.PRNGKey(cfg.seed)
    for epoch in range(1, cfg.epochs + 1):
        params, opt_state, rng, tr_m = _run_training_epoch(cfg.mode, step_fn, params, opt_state, train_data, cfg.batch_size, lambdas, rng, n_train)
        if epoch % 50 == 0 or epoch == cfg.epochs:
            print(f"Epoch {epoch:4d}/{cfg.epochs} | Loss: {tr_m['loss']:.4f} | CosSim: {tr_m['cos_sim']:.4f}")

    _save_translator_checkpoints(params, cfg, os.getcwd(), os.path.join(os.getcwd(), f"checkpoint_{cfg.mode}_{cfg.split}.pkl"), False)


if __name__ == "__main__":
    main()
