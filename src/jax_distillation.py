import os
import functools
from typing import Dict, Any, Tuple
import numpy as np
import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
import optax
from omegaconf import DictConfig

# ==============================================================================
# Flax Neural Architectures for Latent Distillation
# ==============================================================================

class FlaxGatedAttentionTranslator(nn.Module):
    """
    Gated Cross-Attention / State-Goal Modulation Network in Flax Linen.
    h_s = MLP(state)
    gate_g = Sigmoid(MLP(goal_latent))
    h_fused = h_s * gate_g
    z_cmd = normalize_z(MLP(h_fused))
    """
    latent_dim: int = 128
    hidden_dim: int = 256
    n_layers: int = 3

    @nn.compact
    def __call__(self, state: jnp.ndarray, goal_latent: jnp.ndarray) -> jnp.ndarray:
        # 1. State Encoder
        h_s = state
        for _ in range(self.n_layers):
            h_s = nn.Dense(self.hidden_dim)(h_s)
            h_s = nn.LayerNorm()(h_s)
            h_s = nn.gelu(h_s)

        # 2. Goal Encoder (produces gating modulation signal in [0, 1])
        h_g = goal_latent
        for _ in range(self.n_layers - 1):
            h_g = nn.Dense(self.hidden_dim)(h_g)
            h_g = nn.LayerNorm()(h_g)
            h_g = nn.gelu(h_g)
        gate_g = nn.sigmoid(nn.Dense(self.hidden_dim)(h_g))

        # 3. Gated Fusion
        h_fused = h_s * gate_g

        # 4. Output Projection
        out = nn.Dense(self.hidden_dim)(h_fused)
        out = nn.LayerNorm()(out)
        out = nn.gelu(out)
        out = nn.Dense(self.latent_dim)(out)

        # Sphere Normalization matching FB representation radius
        z_norm = out / (jnp.linalg.norm(out, axis=-1, keepdims=True) + 1e-8) * jnp.sqrt(self.latent_dim)
        return z_norm


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
        h0 = nn.Dense(self.hidden_dim)(x)
        h0 = nn.LayerNorm()(h0)
        h0 = nn.gelu(h0)

        features = [h0]
        for _ in range(self.n_layers - 1):
            cat_feat = jnp.concatenate(features, axis=-1)
            h_next = nn.Dense(self.hidden_dim)(cat_feat)
            h_next = nn.LayerNorm()(h_next)
            h_next = nn.gelu(h_next)

            # 1D Channel Attention (ECA) via depthwise-like 1D conv
            att = nn.Conv(features=1, kernel_size=(3,), padding="SAME", use_bias=False)(h_next[:, :, None])
            att = nn.sigmoid(att[:, :, 0])
            h_next = h_next * att
            features.append(h_next)

        all_cat = jnp.concatenate(features, axis=-1)
        out = nn.Dense(self.latent_dim)(all_cat)
        z_norm = out / (jnp.linalg.norm(out, axis=-1, keepdims=True) + 1e-8) * jnp.sqrt(self.latent_dim)
        return z_norm


def build_flax_translator(model_type: str = "gated_attn", latent_dim: int = 128, hidden_dim: int = 256, n_layers: int = 3):
    if model_type in ("gated_attn", "gated", "cross_gated"):
        return FlaxGatedAttentionTranslator(latent_dim=latent_dim, hidden_dim=hidden_dim, n_layers=n_layers)
    elif model_type in ("dense_eca", "dense"):
        return FlaxDenseECATranslator(latent_dim=latent_dim, hidden_dim=hidden_dim, n_layers=n_layers)
    else:
        return FlaxGatedAttentionTranslator(latent_dim=latent_dim, hidden_dim=hidden_dim, n_layers=n_layers)


# ==============================================================================
# Differentiable Physics-Grounded Loss Step
# ==============================================================================

def make_train_step(student_def: nn.Module, agent: Any, optimizer: optax.GradientTransformation):
    """
    Creates a JIT-compiled training step with gradients backpropagating through
    frozen agent actor and forward representation.
    """
    @jax.jit
    def step_fn(params, opt_state, batch, lambdas: Dict[str, float]):
        def loss_fn(p):
            state = batch["state"]           # (B, obs_dim)
            goal_z = batch["goal_z"]         # (B, latent_dim)
            z_target = batch["z_target"]     # (B, latent_dim)
            a_target = batch["a_target"]     # (B, action_dim)

            # 1. Forward pass of student translator
            z_cmd = student_def.apply({"params": p}, state, goal_z)

            # 2. Behavioral Cloning Loss (MSE + Cosine)
            mse_z = jnp.mean(jnp.sum((z_cmd - z_target) ** 2, axis=-1))
            cos_sim = jnp.sum(z_cmd * z_target, axis=-1) / (
                (jnp.linalg.norm(z_cmd, axis=-1) * jnp.linalg.norm(z_target, axis=-1)) + 1e-8
            )
            loss_cos = jnp.mean(1.0 - cos_sim)
            loss_bc = lambdas["l_mse"] * mse_z + lambdas["l_cos"] * loss_cos

            # 3. Action-Alignment Loss (Differentiating through frozen low_level actor)
            pred_actor_dist = agent.network.select("actor")(state, z_cmd, goal_encoded=True, temperature=0.0)
            pred_action = pred_actor_dist.mode()
            loss_action = jnp.mean(jnp.sum((pred_action - a_target) ** 2, axis=-1))

            # 4. Reachability Loss (Maximizing F(s, z_cmd)^\top z_cmd)
            f_repr = agent.network.select("forward_repr")(state, z_cmd, goal_encoded=True)
            if f_repr.ndim == 3:
                f_repr = jnp.mean(f_repr, axis=0)
            reachability = jnp.sum(f_repr * z_cmd, axis=-1)
            loss_reach = -jnp.mean(reachability)

            # 5. Goal-Consistency Loss (Maximizing alignment with global goal B(g))
            goal_cos = jnp.sum(z_cmd * goal_z, axis=-1) / (
                (jnp.linalg.norm(z_cmd, axis=-1) * jnp.linalg.norm(goal_z, axis=-1)) + 1e-8
            )
            loss_goal = -jnp.mean(goal_cos)

            # Total combined objective
            total_loss = (
                loss_bc
                + lambdas["l_action"] * loss_action
                + lambdas["l_reach"] * loss_reach
                + lambdas["l_goal"] * loss_goal
            )

            metrics = {
                "loss": total_loss,
                "loss_bc": loss_bc,
                "loss_mse": mse_z,
                "loss_cos": loss_cos,
                "loss_action": loss_action,
                "loss_reach": loss_reach,
                "loss_goal": loss_goal,
                "cos_sim": jnp.mean(cos_sim),
                "goal_cos": jnp.mean(goal_cos),
                "action_mse": loss_action,
            }
            return total_loss, metrics

        grads, metrics = jax.grad(loss_fn, has_aux=True)(params)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, metrics

    return step_fn


def make_eval_step(student_def: nn.Module, agent: Any):
    """Creates a JIT-compiled validation evaluation step."""
    @jax.jit
    def eval_fn(params, batch):
        state = batch["state"]
        goal_z = batch["goal_z"]
        z_target = batch["z_target"]
        a_target = batch["a_target"]

        z_cmd = student_def.apply({"params": params}, state, goal_z)

        mse_z = jnp.mean(jnp.sum((z_cmd - z_target) ** 2, axis=-1))
        cos_sim = jnp.sum(z_cmd * z_target, axis=-1) / (
            (jnp.linalg.norm(z_cmd, axis=-1) * jnp.linalg.norm(z_target, axis=-1)) + 1e-8
        )

        pred_actor_dist = agent.network.select("actor")(state, z_cmd, goal_encoded=True, temperature=0.0)
        pred_action = pred_actor_dist.mode()
        action_mse = jnp.mean(jnp.sum((pred_action - a_target) ** 2, axis=-1))

        return {
            "val_mse": mse_z,
            "val_cos_sim": jnp.mean(cos_sim),
            "val_action_mse": action_mse,
        }

    return eval_fn
