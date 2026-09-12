"""Differentiable loss functions, JIT step factories, and validation routines for intention distillation."""

from typing import Any, Dict, Tuple
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax


def compute_core_losses(
    pred_z: jnp.ndarray,
    z_target: jnp.ndarray,
    a_target: jnp.ndarray,
    state: jnp.ndarray,
    frozen_actor_fn: Any,
    frozen_f_fn: Any,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute cosine similarity, MSE, low-level action error, and FB reachability loss."""
    cos_sim = jnp.sum(pred_z * z_target, axis=-1) / (
        (jnp.linalg.norm(pred_z, axis=-1) * jnp.linalg.norm(z_target, axis=-1)) + 1e-8
    )
    loss_cos = jnp.mean(1.0 - cos_sim)
    loss_mse = jnp.mean(jnp.sum((pred_z - z_target) ** 2, axis=-1))

    pred_action = frozen_actor_fn(state, pred_z, goal_encoded=True, temperature=0.0).mode()
    loss_action = jnp.mean(jnp.sum((pred_action - a_target) ** 2, axis=-1))

    f_rep = frozen_f_fn(state, pred_z, goal_encoded=True)
    if f_rep.ndim == 3:
        f_rep = jnp.mean(f_rep, axis=0)
    loss_reach = -jnp.mean(jnp.sum(f_rep * pred_z, axis=-1))
    return cos_sim, loss_cos, loss_mse, loss_action, loss_reach


def make_single_waypoint_train_step(
    student_apply_fn: Any,
    frozen_actor_fn: Any,
    frozen_f_fn: Any,
    frozen_b_fn: Any,
    optimizer: optax.GradientTransformation,
) -> Any:
    """Compile JIT training step for single-waypoint translator."""
    @jax.jit
    def train_step(params, opt_state, batch, lambdas):
        def loss_fn(p):
            pred_z = student_apply_fn({"params": p}, batch["state"], batch["w1_z"])
            cos_sim, loss_cos, loss_mse, loss_action, loss_reach = compute_core_losses(
                pred_z, batch["z_target"], batch["a_target"], batch["state"], frozen_actor_fn, frozen_f_fn
            )
            loss_bc = loss_cos + lambdas["l_mse"] * loss_mse
            loss_goal = -jnp.mean(jnp.sum(pred_z * batch["w1_z"], axis=-1))
            total_loss = (
                lambdas["l_cos"] * loss_bc
                + lambdas["l_action"] * loss_action
                + lambdas["l_reach"] * loss_reach
                + lambdas["l_goal"] * loss_goal
            )
            metrics = {
                "loss": total_loss,
                "loss_bc": loss_bc,
                "loss_action": loss_action,
                "loss_reach": loss_reach,
                "loss_goal": loss_goal,
                "cos_sim": jnp.mean(cos_sim),
            }
            return total_loss, metrics

        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), new_opt_state, metrics
    return train_step


def make_sequence_attention_train_step(
    student_apply_fn: Any,
    frozen_actor_fn: Any,
    frozen_f_fn: Any,
    frozen_b_fn: Any,
    optimizer: optax.GradientTransformation,
) -> Any:
    """Compile JIT training step for sequence-waypoint attention transformer."""
    @jax.jit
    def train_step(params, opt_state, batch, lambdas, rng):
        rng, drop_rng = jax.random.split(rng)
        def loss_fn(p):
            pred_z = student_apply_fn(
                {"params": p},
                batch["state"],
                batch["wp_seq"],
                batch["seq_mask"],
                batch.get("curvatures", None),
                deterministic=False,
                rngs={"dropout": drop_rng},
            )
            cos_sim, loss_cos, loss_mse, loss_action, loss_reach = compute_core_losses(
                pred_z, batch["z_target"], batch["a_target"], batch["state"], frozen_actor_fn, frozen_f_fn
            )
            loss_bc = loss_cos + lambdas["l_mse"] * loss_mse
            loss_goal = -jnp.mean(jnp.sum(pred_z * batch["wp_seq"][:, 0], axis=-1))
            loss_aux = loss_mse
            total_loss = (
                lambdas["l_cos"] * loss_bc
                + lambdas["l_action"] * loss_action
                + lambdas["l_reach"] * loss_reach
                + lambdas["l_goal"] * loss_goal
                + lambdas.get("l_aux", 0.2) * loss_aux
            )
            metrics = {
                "loss": total_loss,
                "loss_bc": loss_bc,
                "loss_action": loss_action,
                "loss_reach": loss_reach,
                "loss_goal": loss_goal,
                "loss_aux": loss_aux,
                "cos_sim": jnp.mean(cos_sim),
            }
            return total_loss, metrics

        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), new_opt_state, metrics, rng
    return train_step


def make_direct_intention_train_step(
    student_def: nn.Module, agent: Any, optimizer: optax.GradientTransformation
) -> Any:
    """Compile JIT training step for direct intention amortized policy distillation."""
    actor_fn = agent.network.select("actor")
    f_fn = agent.network.select("forward_repr")

    @jax.jit
    def step_fn(params, opt_state, batch, lambdas: Dict[str, float]):
        def loss_fn(p):
            z_cmd = student_def.apply({"params": p}, batch["state"], batch["goal_z"])
            cos_sim, loss_cos, loss_mse, loss_action, loss_reach = compute_core_losses(
                z_cmd, batch["z_target"], batch["a_target"], batch["state"], actor_fn, f_fn
            )
            loss_bc = lambdas["l_mse"] * loss_mse + lambdas["l_cos"] * loss_cos
            goal_cos = jnp.sum(z_cmd * batch["goal_z"], axis=-1) / (
                (jnp.linalg.norm(z_cmd, axis=-1) * jnp.linalg.norm(batch["goal_z"], axis=-1)) + 1e-8
            )
            loss_goal = -jnp.mean(goal_cos)
            total_loss = (
                loss_bc
                + lambdas["l_action"] * loss_action
                + lambdas["l_reach"] * loss_reach
                + lambdas["l_goal"] * loss_goal
            )
            metrics = {
                "loss": total_loss,
                "loss_bc": loss_bc,
                "loss_action": loss_action,
                "loss_reach": loss_reach,
                "loss_goal": loss_goal,
                "cos_sim": jnp.mean(cos_sim),
            }
            return total_loss, metrics

        grads, metrics = jax.grad(loss_fn, has_aux=True)(params)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), new_opt_state, metrics
    return step_fn


def make_direct_intention_eval_step(student_def: nn.Module, agent: Any) -> Any:
    """Compile JIT validation evaluation step for direct intention distillation."""
    actor_fn = agent.network.select("actor")

    @jax.jit
    def eval_fn(params, batch):
        z_cmd = student_def.apply({"params": params}, batch["state"], batch["goal_z"])
        mse_z = jnp.mean(jnp.sum((z_cmd - batch["z_target"]) ** 2, axis=-1))
        cos_sim = jnp.sum(z_cmd * batch["z_target"], axis=-1) / (
            (jnp.linalg.norm(z_cmd, axis=-1) * jnp.linalg.norm(batch["z_target"], axis=-1)) + 1e-8
        )
        pred_action = actor_fn(batch["state"], z_cmd, goal_encoded=True, temperature=0.0).mode()
        return {
            "val_mse": mse_z,
            "val_cos_sim": jnp.mean(cos_sim),
            "val_action_mse": jnp.mean(jnp.sum((pred_action - batch["a_target"]) ** 2, axis=-1)),
        }
    return eval_fn
