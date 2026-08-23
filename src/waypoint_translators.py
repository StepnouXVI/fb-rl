import os
import functools
from typing import Dict, Any, Tuple, Sequence, Optional
import numpy as np
import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
import optax


# ==============================================================================
# 1. Single-Waypoint & Gated Modulation Translators
# ==============================================================================

class FlaxSingleWaypointTranslator(nn.Module):
    """
    Direct translator from immediate next waypoint B(w_1) and state s_t to low-level intention z_cmd.
    """
    obs_dim: int = 29
    latent_dim: int = 128
    hidden_dim: int = 256
    n_layers: int = 3

    @nn.compact
    def __call__(self, state: jnp.ndarray, waypoint_z: jnp.ndarray) -> jnp.ndarray:
        h_s = nn.gelu(nn.LayerNorm(name="s_ln_0")(nn.Dense(self.hidden_dim, name="s_enc_0")(state)))
        h_s = nn.LayerNorm(name="s_ln_1")(nn.Dense(self.hidden_dim, name="s_enc_1")(h_s))

        h_w = nn.gelu(nn.LayerNorm(name="w_ln_0")(nn.Dense(self.hidden_dim, name="w_enc_0")(waypoint_z)))
        h_w = nn.Dense(self.hidden_dim, name="w_enc_1")(h_w)

        gate = nn.sigmoid(nn.Dense(self.hidden_dim, name="gate_fc")(jnp.concatenate([h_s, h_w], axis=-1)))
        h_fused = gate * h_w + (1.0 - gate) * h_s

        for i in range(self.n_layers - 1):
            h_res = nn.gelu(nn.LayerNorm(name=f"refine_ln_{i}")(nn.Dense(self.hidden_dim, name=f"refine_fc_{i}")(h_fused)))
            h_fused = h_fused + h_res

        out = nn.Dense(self.latent_dim, name="out_head")(h_fused)
        return out / (jnp.linalg.norm(out, axis=-1, keepdims=True) + 1e-8) * jnp.sqrt(self.latent_dim)


class FlaxGatedAttentionTranslator(nn.Module):
    """
    Gated State-Goal Modulation Network in Flax Linen for O(1) Amortized Planning.
    """
    latent_dim: int = 128
    hidden_dim: int = 256
    n_layers: int = 3

    @nn.compact
    def __call__(self, state: jnp.ndarray, goal_latent: jnp.ndarray) -> jnp.ndarray:
        h_s = state
        for _ in range(self.n_layers):
            h_s = nn.gelu(nn.LayerNorm()(nn.Dense(self.hidden_dim)(h_s)))

        h_g = goal_latent
        for _ in range(self.n_layers - 1):
            h_g = nn.gelu(nn.LayerNorm()(nn.Dense(self.hidden_dim)(h_g)))
        gate_g = nn.sigmoid(nn.Dense(self.hidden_dim)(h_g))

        h_fused = h_s * gate_g
        out = nn.gelu(nn.LayerNorm()(nn.Dense(self.hidden_dim)(h_fused)))
        out = nn.Dense(self.latent_dim)(out)
        return out / (jnp.linalg.norm(out, axis=-1, keepdims=True) + 1e-8) * jnp.sqrt(self.latent_dim)


class FlaxDenseECATranslator(nn.Module):
    """
    Dense feature concatenation network with 1D Efficient Channel Attention in Flax.
    """
    latent_dim: int = 128
    hidden_dim: int = 256
    n_layers: int = 3

    @nn.compact
    def __call__(self, state: jnp.ndarray, goal_latent: jnp.ndarray) -> jnp.ndarray:
        x = jnp.concatenate([state, goal_latent], axis=-1)
        h0 = nn.gelu(nn.LayerNorm()(nn.Dense(self.hidden_dim)(x)))

        features = [h0]
        for _ in range(self.n_layers - 1):
            cat_feat = jnp.concatenate(features, axis=-1)
            h_next = nn.gelu(nn.LayerNorm()(nn.Dense(self.hidden_dim)(cat_feat)))

            w_eca = self.param(f"eca_w_{len(features)}", nn.initializers.ones, (3,))
            h_pad = jnp.pad(h_next, ((0, 0), (1, 1)), mode="edge")
            att = nn.sigmoid(w_eca[0] * h_pad[:, :-2] + w_eca[1] * h_pad[:, 1:-1] + w_eca[2] * h_pad[:, 2:])
            features.append(h_next * att)

        all_cat = jnp.concatenate(features, axis=-1)
        out = nn.Dense(self.latent_dim)(all_cat)
        return out / (jnp.linalg.norm(out, axis=-1, keepdims=True) + 1e-8) * jnp.sqrt(self.latent_dim)


def build_flax_translator(model_type: str = "gated_attn", latent_dim: int = 128, hidden_dim: int = 256, n_layers: int = 3):
    if model_type in ("gated_attn", "gated", "cross_gated"):
        return FlaxGatedAttentionTranslator(latent_dim=latent_dim, hidden_dim=hidden_dim, n_layers=n_layers)
    elif model_type in ("dense_eca", "dense"):
        return FlaxDenseECATranslator(latent_dim=latent_dim, hidden_dim=hidden_dim, n_layers=n_layers)
    return FlaxGatedAttentionTranslator(latent_dim=latent_dim, hidden_dim=hidden_dim, n_layers=n_layers)


# ==============================================================================
# 2. Sequence Attention & Transformer Encoders
# ==============================================================================

class PositionalEncoding(nn.Module):
    dim: int
    max_len: int = 32

    @nn.compact
    def __call__(self, seq_len: int) -> jnp.ndarray:
        pe = self.param("pe", nn.initializers.normal(stddev=0.02), (self.max_len, self.dim))
        return pe[:seq_len]


class FlaxSequenceWaypointAttentionTranslator(nn.Module):
    """
    Sequence-Aware High-Level Attention Translator.
    """
    obs_dim: int = 29
    latent_dim: int = 128
    hidden_dim: int = 256
    num_heads: int = 4
    max_seq_len: int = 16
    n_layers: int = 2

    @nn.compact
    def __call__(self, state: jnp.ndarray, waypoint_seq: jnp.ndarray, seq_mask: jnp.ndarray = None) -> jnp.ndarray:
        B, K, _ = waypoint_seq.shape
        w_proj = nn.Dense(self.hidden_dim, name="w_proj")(waypoint_seq)
        pos_enc = PositionalEncoding(dim=self.hidden_dim, max_len=self.max_seq_len)(K)
        h_w = w_proj + pos_enc[None, :, :]
        attn_mask = seq_mask[:, None, None, :] if seq_mask is not None else None

        for l in range(self.n_layers):
            attn_layer = nn.MultiHeadDotProductAttention(num_heads=self.num_heads, qkv_features=self.hidden_dim, name=f"self_attn_{l}")
            h_w = nn.LayerNorm(name=f"sa_ln_{l}")(h_w + attn_layer(h_w, h_w, mask=attn_mask))
            h_mlp = nn.Dense(self.hidden_dim, name=f"sa_mlp_1_{l}")(nn.gelu(nn.Dense(self.hidden_dim, name=f"sa_mlp_0_{l}")(h_w)))
            h_w = nn.LayerNorm(name=f"sa_ln_mlp_{l}")(h_w + h_mlp)

        s_q = nn.gelu(nn.LayerNorm(name="s_q_ln")(nn.Dense(self.hidden_dim, name="s_query_0")(state)))
        s_query = nn.Dense(self.hidden_dim, name="s_query_1")(s_q)[:, None, :]

        cross_attn = nn.MultiHeadDotProductAttention(num_heads=self.num_heads, qkv_features=self.hidden_dim, name="cross_attn")
        cross_out = cross_attn(s_query, h_w, mask=attn_mask).squeeze(1)

        s_rep = s_query.squeeze(1)
        gate = nn.sigmoid(nn.Dense(self.hidden_dim, name="fusion_gate")(jnp.concatenate([s_rep, cross_out], axis=-1)))
        fused = gate * cross_out + (1.0 - gate) * s_rep

        out = nn.Dense(self.latent_dim, name="out_head")(nn.gelu(nn.LayerNorm(name="head_ln")(nn.Dense(self.hidden_dim, name="head_fc")(fused))))
        return out / (jnp.linalg.norm(out, axis=-1, keepdims=True) + 1e-8) * jnp.sqrt(self.latent_dim)


def _compute_curvatures_helper(waypoint_seq: jnp.ndarray, curvatures: Optional[jnp.ndarray]) -> jnp.ndarray:
    if curvatures is not None:
        return curvatures[:, :, None] if curvatures.ndim == 2 else curvatures
    B = waypoint_seq.shape[0]
    v = waypoint_seq[:, 1:] - waypoint_seq[:, :-1]
    v_norm = jnp.linalg.norm(v, axis=-1, keepdims=True) + 1e-6
    v_u = v / v_norm
    cos_th = jnp.sum(v_u[:, :-1] * v_u[:, 1:], axis=-1, keepdims=True)
    pad = jnp.ones((B, 1, 1), dtype=waypoint_seq.dtype)
    return jnp.concatenate([pad, cos_th, pad], axis=1)


class FlaxEnhancedSequenceWaypointAttentionTranslator(nn.Module):
    """
    Enhanced Sequence-Aware Waypoint Attention Translator with ALiBi, Windowing, and Curvature.
    """
    obs_dim: int = 29
    latent_dim: int = 128
    hidden_dim: int = 384
    num_heads: int = 6
    max_seq_len: int = 16
    n_layers: int = 3
    dropout_rate: float = 0.2
    alibi_slope: float = 0.4
    local_window_size: int = 6

    def _build_active_mask(self, B: int, K: int, seq_mask: Optional[jnp.ndarray], deterministic: bool):
        if seq_mask is None:
            seq_mask = jnp.ones((B, K), dtype=bool)
        seq_len = jnp.sum(seq_mask.astype(jnp.int32), axis=-1, keepdims=True)
        k_idx = jnp.arange(K)[None, :]
        is_terminal = (k_idx == jnp.maximum(0, seq_len - 1))
        local_global_mask = seq_mask & ((k_idx < self.local_window_size) | is_terminal)

        if not deterministic and self.dropout_rate > 0.0:
            rnd = jax.random.uniform(self.make_rng("dropout"), (B, K))
            is_anchored = (k_idx == 0) | is_terminal
            return local_global_mask & ((rnd >= self.dropout_rate) | is_anchored), k_idx, is_terminal
        return local_global_mask, k_idx, is_terminal

    def _apply_alibi_cross_attn(self, s_q: jnp.ndarray, h_w: jnp.ndarray, active_mask: jnp.ndarray, k_idx: jnp.ndarray, is_terminal: jnp.ndarray, deterministic: bool):
        B, K, _ = h_w.shape
        head_dim = self.hidden_dim // self.num_heads
        q = nn.Dense(self.hidden_dim, name="cross_q")(s_q).reshape((B, 1, self.num_heads, head_dim))
        k = nn.Dense(self.hidden_dim, name="cross_k")(h_w).reshape((B, K, self.num_heads, head_dim))
        v = nn.Dense(self.hidden_dim, name="cross_v")(h_w).reshape((B, K, self.num_heads, head_dim))

        logits = jnp.einsum("bqhd,bkhd->bhqk", q, k) / jnp.sqrt(head_dim)
        decay = jnp.where(is_terminal & (k_idx >= self.local_window_size), float(self.local_window_size), k_idx.astype(jnp.float32))
        slopes = jnp.array([2.0 ** (-4.0 * h / max(self.num_heads - 1, 1)) for h in range(self.num_heads)])
        logits = logits + (-self.alibi_slope * slopes[None, :, None, None] * decay[:, None, None, :])
        logits = jnp.where(active_mask[:, None, None, :], logits, -1e9)

        weights = nn.Dropout(rate=self.dropout_rate)(jax.nn.softmax(logits, axis=-1), deterministic=deterministic)
        cross_out = jnp.einsum("bhqk,bkhd->bqhd", weights, v).reshape((B, 1, self.hidden_dim)).squeeze(1)
        return nn.Dense(self.hidden_dim, name="cross_out_proj")(cross_out)

    @nn.compact
    def __call__(self, state: jnp.ndarray, waypoint_seq: jnp.ndarray, seq_mask: jnp.ndarray = None, curvatures: jnp.ndarray = None, deterministic: bool = True) -> jnp.ndarray:
        B, K, D = waypoint_seq.shape
        curv_3d = _compute_curvatures_helper(waypoint_seq, curvatures)
        c_emb = nn.Dense(self.hidden_dim, name="curv_proj")(curv_3d)

        w_emb = nn.Dense(self.hidden_dim, name="w_proj")(waypoint_seq) + PositionalEncoding(dim=self.hidden_dim, max_len=self.max_seq_len)(K)[None, :K, :] + c_emb
        w_emb = nn.Dropout(rate=self.dropout_rate)(w_emb, deterministic=deterministic)

        active_mask, k_idx, is_terminal = self._build_active_mask(B, K, seq_mask, deterministic)
        sa_mask = active_mask[:, None, None, :]
        h_w = w_emb
        for l in range(self.n_layers):
            attn = nn.MultiHeadDotProductAttention(num_heads=self.num_heads, qkv_features=self.hidden_dim, dropout_rate=self.dropout_rate, deterministic=deterministic, name=f"self_attn_{l}")
            h_w = nn.LayerNorm(name=f"sa_ln_{l}")(h_w + attn(h_w, h_w, mask=sa_mask))
            h_mlp = nn.Dense(self.hidden_dim, name=f"sa_mlp_1_{l}")(nn.Dropout(rate=self.dropout_rate)(nn.gelu(nn.Dense(self.hidden_dim, name=f"sa_mlp_0_{l}")(h_w)), deterministic=deterministic))
            h_w = nn.LayerNorm(name=f"sa_ln_mlp_{l}")(h_w + h_mlp)

        s_q = nn.Dense(self.hidden_dim, name="s_query_1")(nn.gelu(nn.LayerNorm(name="s_q_ln")(nn.Dense(self.hidden_dim, name="s_query_0")(state))))[:, None, :]
        cross_out = self._apply_alibi_cross_attn(s_q, h_w, active_mask, k_idx, is_terminal, deterministic)

        s_rep = s_q.squeeze(1)
        gate = nn.sigmoid(nn.Dense(self.hidden_dim, name="fusion_gate")(jnp.concatenate([s_rep, cross_out], axis=-1)))
        fused = gate * cross_out + (1.0 - gate) * s_rep

        out = nn.Dense(self.latent_dim, name="out_head")(nn.gelu(nn.LayerNorm(name="head_ln")(nn.Dense(self.hidden_dim, name="head_fc")(fused))))
        return out / (jnp.linalg.norm(out, axis=-1, keepdims=True) + 1e-8) * jnp.sqrt(self.latent_dim)


# ==============================================================================
# 3. Differentiable Training & Evaluation Step Factories
# ==============================================================================

def _compute_core_losses(pred_z: jnp.ndarray, z_target: jnp.ndarray, a_target: jnp.ndarray, state: jnp.ndarray, frozen_actor_fn: Any, frozen_f_fn: Any):
    cos_sim = jnp.sum(pred_z * z_target, axis=-1) / ((jnp.linalg.norm(pred_z, axis=-1) * jnp.linalg.norm(z_target, axis=-1)) + 1e-8)
    loss_cos = jnp.mean(1.0 - cos_sim)
    loss_mse = jnp.mean(jnp.sum((pred_z - z_target) ** 2, axis=-1))
    
    pred_action = frozen_actor_fn(state, pred_z, goal_encoded=True, temperature=0.0).mode()
    loss_action = jnp.mean(jnp.sum((pred_action - a_target) ** 2, axis=-1))

    f_rep = frozen_f_fn(state, pred_z, goal_encoded=True)
    if f_rep.ndim == 3:
        f_rep = jnp.mean(f_rep, axis=0)
    loss_reach = -jnp.mean(jnp.sum(f_rep * pred_z, axis=-1))
    return cos_sim, loss_cos, loss_mse, loss_action, loss_reach


def make_single_wp_train_step(student_apply_fn, frozen_actor_fn, frozen_f_fn, frozen_b_fn, optimizer):
    @jax.jit
    def train_step(params, opt_state, batch, lambdas):
        def loss_fn(p):
            pred_z = student_apply_fn({"params": p}, batch["state"], batch["w1_z"])
            cos_sim, loss_cos, loss_mse, loss_action, loss_reach = _compute_core_losses(
                pred_z, batch["z_target"], batch["a_target"], batch["state"], frozen_actor_fn, frozen_f_fn
            )
            loss_bc = loss_cos + lambdas["l_mse"] * loss_mse
            loss_goal = -jnp.mean(jnp.sum(pred_z * batch["w1_z"], axis=-1))
            total_loss = lambdas["l_cos"] * loss_bc + lambdas["l_action"] * loss_action + lambdas["l_reach"] * loss_reach + lambdas["l_goal"] * loss_goal
            return total_loss, {"loss": total_loss, "loss_bc": loss_bc, "loss_action": loss_action, "loss_reach": loss_reach, "loss_goal": loss_goal, "cos_sim": jnp.mean(cos_sim)}

        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), new_opt_state, metrics
    return train_step


def make_sequence_wp_train_step(student_apply_fn, frozen_actor_fn, frozen_f_fn, frozen_b_fn, optimizer):
    @jax.jit
    def train_step(params, opt_state, batch, lambdas):
        def loss_fn(p):
            pred_z = student_apply_fn({"params": p}, batch["state"], batch["wp_seq"], batch["seq_mask"])
            cos_sim, loss_cos, loss_mse, loss_action, loss_reach = _compute_core_losses(
                pred_z, batch["z_target"], batch["a_target"], batch["state"], frozen_actor_fn, frozen_f_fn
            )
            loss_bc = loss_cos + lambdas["l_mse"] * loss_mse
            loss_goal = -jnp.mean(jnp.sum(pred_z * batch["wp_seq"][:, 0], axis=-1))
            total_loss = lambdas["l_cos"] * loss_bc + lambdas["l_action"] * loss_action + lambdas["l_reach"] * loss_reach + lambdas["l_goal"] * loss_goal
            return total_loss, {"loss": total_loss, "loss_bc": loss_bc, "loss_action": loss_action, "loss_reach": loss_reach, "loss_goal": loss_goal, "cos_sim": jnp.mean(cos_sim)}

        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), new_opt_state, metrics
    return train_step


def make_enhanced_sequence_wp_train_step(student_apply_fn, frozen_actor_fn, frozen_f_fn, frozen_b_fn, optimizer):
    @jax.jit
    def train_step(params, opt_state, batch, lambdas, rng):
        rng, drop_rng = jax.random.split(rng)
        def loss_fn(p):
            pred_z = student_apply_fn(
                {"params": p}, batch["state"], batch["wp_seq"], batch["seq_mask"],
                batch.get("curvatures", None), deterministic=False, rngs={"dropout": drop_rng}
            )
            cos_sim, loss_cos, loss_mse, loss_action, loss_reach = _compute_core_losses(
                pred_z, batch["z_target"], batch["a_target"], batch["state"], frozen_actor_fn, frozen_f_fn
            )
            loss_bc = loss_cos + lambdas["l_mse"] * loss_mse
            loss_goal = -jnp.mean(jnp.sum(pred_z * batch["wp_seq"][:, 0], axis=-1))
            loss_aux = loss_mse
            total_loss = (lambdas["l_cos"] * loss_bc + lambdas["l_action"] * loss_action + lambdas["l_reach"] * loss_reach + lambdas["l_goal"] * loss_goal + lambdas.get("l_aux", 0.2) * loss_aux)
            return total_loss, {"loss": total_loss, "loss_bc": loss_bc, "loss_action": loss_action, "loss_reach": loss_reach, "loss_goal": loss_goal, "loss_aux": loss_aux, "cos_sim": jnp.mean(cos_sim)}

        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), new_opt_state, metrics, rng
    return train_step


def make_jax_distill_train_step(student_def: nn.Module, agent: Any, optimizer: optax.GradientTransformation):
    """JIT-compiled training step for JAX Gated-Attention / Dense-ECA Amortized Distillation."""
    actor_fn = agent.network.select("actor")
    f_fn = agent.network.select("forward_repr")

    @jax.jit
    def step_fn(params, opt_state, batch, lambdas: Dict[str, float]):
        def loss_fn(p):
            z_cmd = student_def.apply({"params": p}, batch["state"], batch["goal_z"])
            cos_sim, loss_cos, loss_mse, loss_action, loss_reach = _compute_core_losses(
                z_cmd, batch["z_target"], batch["a_target"], batch["state"], actor_fn, f_fn
            )
            loss_bc = lambdas["l_mse"] * loss_mse + lambdas["l_cos"] * loss_cos
            goal_cos = jnp.sum(z_cmd * batch["goal_z"], axis=-1) / ((jnp.linalg.norm(z_cmd, axis=-1) * jnp.linalg.norm(batch["goal_z"], axis=-1)) + 1e-8)
            loss_goal = -jnp.mean(goal_cos)
            total_loss = loss_bc + lambdas["l_action"] * loss_action + lambdas["l_reach"] * loss_reach + lambdas["l_goal"] * loss_goal
            return total_loss, {"loss": total_loss, "loss_bc": loss_bc, "loss_action": loss_action, "loss_reach": loss_reach, "loss_goal": loss_goal, "cos_sim": jnp.mean(cos_sim)}

        grads, metrics = jax.grad(loss_fn, has_aux=True)(params)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), new_opt_state, metrics
    return step_fn


def make_jax_distill_eval_step(student_def: nn.Module, agent: Any):
    """JIT-compiled validation evaluation step for JAX Distillation."""
    actor_fn = agent.network.select("actor")

    @jax.jit
    def eval_fn(params, batch):
        z_cmd = student_def.apply({"params": params}, batch["state"], batch["goal_z"])
        mse_z = jnp.mean(jnp.sum((z_cmd - batch["z_target"]) ** 2, axis=-1))
        cos_sim = jnp.sum(z_cmd * batch["z_target"], axis=-1) / ((jnp.linalg.norm(z_cmd, axis=-1) * jnp.linalg.norm(batch["z_target"], axis=-1)) + 1e-8)
        pred_action = actor_fn(batch["state"], z_cmd, goal_encoded=True, temperature=0.0).mode()
        return {"val_mse": mse_z, "val_cos_sim": jnp.mean(cos_sim), "val_action_mse": jnp.mean(jnp.sum((pred_action - batch["a_target"]) ** 2, axis=-1))}
    return eval_fn


# Aliases for backwards compatibility
make_train_step = make_jax_distill_train_step
make_eval_step = make_jax_distill_eval_step
