import os
import functools
from typing import Dict, Any, Tuple, Sequence
import numpy as np
import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
import optax


class FlaxSingleWaypointTranslator(nn.Module):
    """
    Direct translator from immediate next waypoint B(w_1) and state s_t to low-level intention z_cmd.
    Bypasses high_actor with a specialized Gated Attention mapping.
    """
    obs_dim: int = 29
    latent_dim: int = 128
    hidden_dim: int = 256
    n_layers: int = 3

    @nn.compact
    def __call__(self, state: jnp.ndarray, waypoint_z: jnp.ndarray) -> jnp.ndarray:
        # 1. State feature extraction
        h_s = nn.Dense(self.hidden_dim, name="s_enc_0")(state)
        h_s = nn.LayerNorm(name="s_ln_0")(h_s)
        h_s = nn.gelu(h_s)
        h_s = nn.Dense(self.hidden_dim, name="s_enc_1")(h_s)
        h_s = nn.LayerNorm(name="s_ln_1")(h_s)

        # 2. Next Waypoint feature extraction
        h_w = nn.Dense(self.hidden_dim, name="w_enc_0")(waypoint_z)
        h_w = nn.LayerNorm(name="w_ln_0")(h_w)
        h_w = nn.gelu(h_w)
        h_w = nn.Dense(self.hidden_dim, name="w_enc_1")(h_w)

        # 3. Gating Mechanism
        gate = nn.sigmoid(nn.Dense(self.hidden_dim, name="gate_fc")(jnp.concatenate([h_s, h_w], axis=-1)))
        h_fused = gate * h_w + (1.0 - gate) * h_s

        # 4. Deep refinement
        for i in range(self.n_layers - 1):
            h_res = nn.Dense(self.hidden_dim, name=f"refine_fc_{i}")(h_fused)
            h_res = nn.LayerNorm(name=f"refine_ln_{i}")(h_res)
            h_res = nn.gelu(h_res)
            h_fused = h_fused + h_res

        # 5. Output Projection onto Sphere
        out = nn.Dense(self.latent_dim, name="out_head")(h_fused)
        z_norm = out / (jnp.linalg.norm(out, axis=-1, keepdims=True) + 1e-8) * jnp.sqrt(self.latent_dim)
        return z_norm


class PositionalEncoding(nn.Module):
    dim: int
    max_len: int = 32

    @nn.compact
    def __call__(self, seq_len: int) -> jnp.ndarray:
        pe = self.param("pe", nn.initializers.normal(stddev=0.02), (self.max_len, self.dim))
        return pe[:seq_len]


class FlaxSequenceWaypointAttentionTranslator(nn.Module):
    """
    Sequence-Aware High-Level Attention Translator:
    Reasons over the full multi-hop sequence of waypoints [B(w_1), B(w_2), ..., B(w_K)].
    Uses Self-Attention across waypoints and Cross-Attention with current agent state s_t.
    """
    obs_dim: int = 29
    latent_dim: int = 128
    hidden_dim: int = 256
    num_heads: int = 4
    max_seq_len: int = 16
    n_layers: int = 2

    @nn.compact
    def __call__(
        self,
        state: jnp.ndarray,
        waypoint_seq: jnp.ndarray,
        seq_mask: jnp.ndarray = None,
    ) -> jnp.ndarray:
        """
        state: (B, obs_dim)
        waypoint_seq: (B, K, latent_dim)
        seq_mask: (B, K) boolean mask where True = valid waypoint, False = padding
        """
        B, K, _ = waypoint_seq.shape

        # 1. Waypoint Projection + Positional Embeddings
        w_proj = nn.Dense(self.hidden_dim, name="w_proj")(waypoint_seq) # (B, K, H)
        pos_enc = PositionalEncoding(dim=self.hidden_dim, max_len=self.max_seq_len)(K) # (K, H)
        w_emb = w_proj + pos_enc[None, :, :] # (B, K, H)

        # Mask formatting for attention: (B, 1, 1, K) or (B, 1, K)
        attn_mask = None
        if seq_mask is not None:
            # (B, 1, 1, K) for multihead attention
            attn_mask = seq_mask[:, None, None, :]

        # 2. Waypoint Self-Attention (Contextualize entire trajectory topology)
        h_w = w_emb
        for l in range(self.n_layers):
            attn_layer = nn.MultiHeadDotProductAttention(
                num_heads=self.num_heads,
                qkv_features=self.hidden_dim,
                name=f"self_attn_{l}",
            )
            h_att = attn_layer(h_w, h_w, mask=attn_mask)
            h_w = nn.LayerNorm(name=f"sa_ln_{l}")(h_w + h_att)
            h_mlp = nn.Dense(self.hidden_dim, name=f"sa_mlp_0_{l}")(h_w)
            h_mlp = nn.gelu(h_mlp)
            h_mlp = nn.Dense(self.hidden_dim, name=f"sa_mlp_1_{l}")(h_mlp)
            h_w = nn.LayerNorm(name=f"sa_ln_mlp_{l}")(h_w + h_mlp)

        # 3. State Query Projection
        s_query = nn.Dense(self.hidden_dim, name="s_query_0")(state) # (B, H)
        s_query = nn.LayerNorm(name="s_q_ln")(s_query)
        s_query = nn.gelu(s_query)
        s_query = nn.Dense(self.hidden_dim, name="s_query_1")(s_query)[:, None, :] # (B, 1, H)

        # 4. Cross-Attention: State attends to the contextualized waypoint sequence
        cross_attn = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.hidden_dim,
            name="cross_attn",
        )
        # Query: state (B, 1, H), Key/Value: contextualized waypoints (B, K, H)
        cross_out = cross_attn(s_query, h_w, mask=attn_mask).squeeze(1) # (B, H)

        # 5. Gated Fusion between agent state and trajectory attention
        s_rep = s_query.squeeze(1)
        gate = nn.sigmoid(nn.Dense(self.hidden_dim, name="fusion_gate")(jnp.concatenate([s_rep, cross_out], axis=-1)))
        fused = gate * cross_out + (1.0 - gate) * s_rep

        # 6. Output Projection onto Sphere
        out = nn.Dense(self.hidden_dim, name="head_fc")(fused)
        out = nn.LayerNorm(name="head_ln")(out)
        out = nn.gelu(out)
        out = nn.Dense(self.latent_dim, name="out_head")(out)

        z_norm = out / (jnp.linalg.norm(out, axis=-1, keepdims=True) + 1e-8) * jnp.sqrt(self.latent_dim)
        return z_norm


def make_single_wp_train_step(student_apply_fn, frozen_actor_fn, frozen_f_fn, frozen_b_fn, optimizer):
    """
    Creates a JIT-compiled train step for Single Waypoint Translator.
    Loss: L_BC + L_Action + L_Reach + L_Goal
    """
    @jax.jit
    def train_step(params, opt_state, batch, lambdas):
        states = batch["state"]
        w1_z = batch["w1_z"]
        z_targets = batch["z_target"]
        a_targets = batch["a_target"]
        final_goals_z = batch["final_goal_z"]

        def loss_fn(p):
            pred_z = student_apply_fn({"params": p}, states, w1_z)

            # 1. Behavioral Cloning Loss (Cosine + MSE)
            cos_sim = jnp.sum(pred_z * z_targets, axis=-1) / (
                jnp.linalg.norm(pred_z, axis=-1) * jnp.linalg.norm(z_targets, axis=-1) + 1e-8
            )
            loss_bc = jnp.mean(1.0 - cos_sim) + lambdas["l_mse"] * jnp.mean((pred_z - z_targets) ** 2)

            # 2. Differentiable Action-Alignment Loss through low-level actor
            act_dist = frozen_actor_fn(states, pred_z, goal_encoded=True, temperature=0.0)
            pred_a = act_dist.mode()
            loss_action = jnp.mean((pred_a - a_targets) ** 2)

            # 3. Reachability Regularization: F(s, z)^T z
            f_rep = frozen_f_fn(states, pred_z, goal_encoded=True)
            if f_rep.ndim == 3:
                f_rep = jnp.mean(f_rep, axis=0)
            reach_val = jnp.sum(f_rep * pred_z, axis=-1)
            loss_reach = -jnp.mean(reach_val)

            # 4. Next Waypoint Alignment Loss
            wp_align = jnp.sum(pred_z * w1_z, axis=-1)
            loss_goal = -jnp.mean(wp_align)

            total_loss = (
                lambdas["l_cos"] * loss_bc
                + lambdas["l_action"] * loss_action
                + lambdas["l_reach"] * loss_reach
                + lambdas["l_goal"] * loss_goal
            )
            return total_loss, {
                "loss": total_loss,
                "loss_bc": loss_bc,
                "loss_action": loss_action,
                "loss_reach": loss_reach,
                "loss_goal": loss_goal,
                "cos_sim": jnp.mean(cos_sim),
            }

        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, metrics

    return train_step


def make_sequence_wp_train_step(student_apply_fn, frozen_actor_fn, frozen_f_fn, frozen_b_fn, optimizer):
    """
    Creates a JIT-compiled train step for Sequence-Aware Attention Translator.
    Loss: L_BC + L_Action + L_Reach + L_TrajectoryConsistency
    """
    @jax.jit
    def train_step(params, opt_state, batch, lambdas):
        states = batch["state"]
        wp_seq = batch["wp_seq"] # (B, K, D)
        seq_mask = batch["seq_mask"] # (B, K)
        z_targets = batch["z_target"]
        a_targets = batch["a_target"]
        w1_z = wp_seq[:, 0]

        def loss_fn(p):
            pred_z = student_apply_fn({"params": p}, states, wp_seq, seq_mask)

            # 1. Behavioral Cloning Loss
            cos_sim = jnp.sum(pred_z * z_targets, axis=-1) / (
                jnp.linalg.norm(pred_z, axis=-1) * jnp.linalg.norm(z_targets, axis=-1) + 1e-8
            )
            loss_bc = jnp.mean(1.0 - cos_sim) + lambdas["l_mse"] * jnp.mean((pred_z - z_targets) ** 2)

            # 2. Differentiable Action-Alignment Loss
            act_dist = frozen_actor_fn(states, pred_z, goal_encoded=True, temperature=0.0)
            pred_a = act_dist.mode()
            loss_action = jnp.mean((pred_a - a_targets) ** 2)

            # 3. Reachability Regularization
            f_rep = frozen_f_fn(states, pred_z, goal_encoded=True)
            if f_rep.ndim == 3:
                f_rep = jnp.mean(f_rep, axis=0)
            reach_val = jnp.sum(f_rep * pred_z, axis=-1)
            loss_reach = -jnp.mean(reach_val)

            # 4. Immediate Waypoint Alignment
            wp_align = jnp.sum(pred_z * w1_z, axis=-1)
            loss_goal = -jnp.mean(wp_align)

            total_loss = (
                lambdas["l_cos"] * loss_bc
                + lambdas["l_action"] * loss_action
                + lambdas["l_reach"] * loss_reach
                + lambdas["l_goal"] * loss_goal
            )
            return total_loss, {
                "loss": total_loss,
                "loss_bc": loss_bc,
                "loss_action": loss_action,
                "loss_reach": loss_reach,
                "loss_goal": loss_goal,
                "cos_sim": jnp.mean(cos_sim),
            }

        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, metrics

    return train_step
