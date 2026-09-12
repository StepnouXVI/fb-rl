"""Unit tests for neural network modules and differentiable training steps."""

import numpy as np
import pytest
import jax
import jax.numpy as jnp
import optax

from src.networks import (
    SequenceAttentionNetwork,
    SingleWaypointNetwork,
    build_direct_translator,
)
from src.training import (
    make_sequence_attention_train_step,
    make_single_waypoint_train_step,
)


class MockDist:
    def __init__(self, val):
        self.val = val

    def mode(self):
        return self.val


def mock_actor(s, z, **kwargs):
    return MockDist(jnp.tanh(s[:, :8] + z[:, :8]))


def mock_f(s, z, **kwargs):
    return z


def mock_b(g):
    return g


def test_single_waypoint_network_forward():
    model = SingleWaypointNetwork(latent_dim=128, hidden_dim=64, n_layers=2)
    rng = jax.random.PRNGKey(0)
    dummy_s = jax.random.normal(rng, (4, 29))
    dummy_w = jax.random.normal(rng, (4, 128))

    params = model.init(rng, dummy_s, dummy_w)["params"]
    z_out = model.apply({"params": params}, dummy_s, dummy_w)

    assert z_out.shape == (4, 128)
    norms = jnp.linalg.norm(z_out, axis=-1)
    np.testing.assert_allclose(norms, np.sqrt(128), atol=1e-3)


def test_single_waypoint_train_step():
    model = SingleWaypointNetwork(latent_dim=128, hidden_dim=64, n_layers=1)
    rng = jax.random.PRNGKey(0)
    dummy_s = jnp.ones((4, 29))
    dummy_w = jnp.ones((4, 128))
    params = model.init(rng, dummy_s, dummy_w)["params"]

    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(params)
    step_fn = make_single_waypoint_train_step(model.apply, mock_actor, mock_f, mock_b, optimizer)

    batch = {
        "state": dummy_s,
        "w1_z": dummy_w,
        "z_target": jnp.ones((4, 128)) * 0.1,
        "a_target": jnp.zeros((4, 8)),
    }
    lambdas = {"l_cos": 1.0, "l_mse": 0.1, "l_action": 0.5, "l_reach": 0.02, "l_goal": 0.05}
    new_params, new_opt_state, metrics = step_fn(params, opt_state, batch, lambdas)

    assert metrics["loss"] > 0
    assert "cos_sim" in metrics


def test_sequence_attention_network_forward():
    model = SequenceAttentionNetwork(
        latent_dim=128,
        hidden_dim=64,
        num_heads=2,
        max_seq_len=8,
        n_layers=1,
    )
    rng = jax.random.PRNGKey(0)
    dummy_s = jax.random.normal(rng, (4, 29))
    dummy_seq = jax.random.normal(rng, (4, 8, 128))
    dummy_mask = jnp.ones((4, 8), dtype=bool)
    dummy_curv = jnp.ones((4, 8, 1), dtype=jnp.float32)

    params = model.init({"params": rng, "dropout": rng}, dummy_s, dummy_seq, dummy_mask, dummy_curv)["params"]
    z_out = model.apply({"params": params}, dummy_s, dummy_seq, dummy_mask, dummy_curv, deterministic=True)

    assert z_out.shape == (4, 128)
    norms = jnp.linalg.norm(z_out, axis=-1)
    np.testing.assert_allclose(norms, np.sqrt(128), atol=1e-3)


def test_sequence_attention_train_step():
    model = SequenceAttentionNetwork(
        latent_dim=128,
        hidden_dim=64,
        num_heads=2,
        max_seq_len=8,
        n_layers=1,
    )
    rng = jax.random.PRNGKey(0)
    dummy_s = jnp.ones((4, 29))
    dummy_seq = jnp.ones((4, 8, 128))
    dummy_mask = jnp.ones((4, 8), dtype=bool)
    dummy_curv = jnp.ones((4, 8, 1), dtype=jnp.float32)

    params = model.init({"params": rng, "dropout": rng}, dummy_s, dummy_seq, dummy_mask, dummy_curv)["params"]
    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(params)
    step_fn = make_sequence_attention_train_step(model.apply, mock_actor, mock_f, mock_b, optimizer)

    batch = {
        "state": dummy_s,
        "wp_seq": dummy_seq,
        "seq_mask": dummy_mask,
        "curvatures": dummy_curv,
        "z_target": jnp.ones((4, 128)) * 0.1,
        "a_target": jnp.zeros((4, 8)),
    }
    lambdas = {"l_cos": 1.0, "l_mse": 0.1, "l_action": 0.5, "l_reach": 0.02, "l_goal": 0.05, "l_aux": 0.2}
    new_params, new_opt_state, metrics, rng = step_fn(params, opt_state, batch, lambdas, rng)

    assert metrics["loss"] > 0
    assert "loss_aux" in metrics


def test_build_direct_translator():
    for m_type in ["mlp", "dense_eca", "gated_attn", "resnet_eca"]:
        net = build_direct_translator(model_type=m_type, latent_dim=128, hidden_dim=64, n_layers=2)
        rng = jax.random.PRNGKey(42)
        params = net.init(rng, jnp.zeros((2, 29)), jnp.zeros((2, 128)))["params"]
        out = net.apply({"params": params}, jnp.zeros((2, 29)), jnp.zeros((2, 128)))
        assert out.shape == (2, 128)
