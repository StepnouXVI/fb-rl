import os
import sys
import numpy as np
import pytest
import jax
import jax.numpy as jnp
import flax
import optax

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.waypoint_translators import (
    FlaxSingleWaypointTranslator,
    FlaxSequenceWaypointAttentionTranslator,
    FlaxEnhancedSequenceWaypointAttentionTranslator,
    make_single_wp_train_step,
    make_sequence_wp_train_step,
    make_enhanced_sequence_wp_train_step,
)


def test_single_waypoint_translator_forward():
    model = FlaxSingleWaypointTranslator(obs_dim=29, latent_dim=128, hidden_dim=64, n_layers=2)
    rng = jax.random.PRNGKey(0)
    dummy_s = jax.random.normal(rng, (4, 29))
    dummy_w = jax.random.normal(rng, (4, 128))

    params = model.init(rng, dummy_s, dummy_w)["params"]
    z_out = model.apply({"params": params}, dummy_s, dummy_w)

    assert z_out.shape == (4, 128)
    norms = jnp.linalg.norm(z_out, axis=-1)
    np.testing.assert_allclose(norms, np.sqrt(128), atol=1e-3)


def test_sequence_waypoint_attention_translator_forward():
    model = FlaxSequenceWaypointAttentionTranslator(
        obs_dim=29,
        latent_dim=128,
        hidden_dim=64,
        num_heads=2,
        max_seq_len=8,
        n_layers=1,
    )
    rng = jax.random.PRNGKey(0)
    dummy_s = jax.random.normal(rng, (4, 29))
    dummy_seq = jax.random.normal(rng, (4, 8, 128))
    dummy_mask = jnp.array([
        [True, True, True, True, False, False, False, False],
        [True, True, True, True, True, True, False, False],
        [True, True, False, False, False, False, False, False],
        [True, True, True, True, True, True, True, True],
    ])

    params = model.init(rng, dummy_s, dummy_seq, dummy_mask)["params"]
    z_out = model.apply({"params": params}, dummy_s, dummy_seq, dummy_mask)

    assert z_out.shape == (4, 128)
    norms = jnp.linalg.norm(z_out, axis=-1)
    np.testing.assert_allclose(norms, np.sqrt(128), atol=1e-3)


def test_sequence_wp_differentiable_train_step():
    model = FlaxSequenceWaypointAttentionTranslator(
        obs_dim=29,
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
    params = model.init(rng, dummy_s, dummy_seq, dummy_mask)["params"]

    # Mock frozen networks
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

    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(params)
    step_fn = make_sequence_wp_train_step(model.apply, mock_actor, mock_f, mock_b, optimizer)

    batch = {
        "state": dummy_s,
        "wp_seq": dummy_seq,
        "seq_mask": dummy_mask,
        "z_target": jnp.ones((4, 128)) * 0.1,
        "a_target": jnp.zeros((4, 8)),
    }
    lambdas = {"l_cos": 1.0, "l_mse": 0.1, "l_action": 0.5, "l_reach": 0.02, "l_goal": 0.05}

    new_params, new_opt_state, metrics = step_fn(params, opt_state, batch, lambdas)
    assert metrics["loss"] is not None
    assert "loss_action" in metrics
    assert "cos_sim" in metrics


def test_enhanced_sequence_waypoint_attention_translator_forward():
    model = FlaxEnhancedSequenceWaypointAttentionTranslator(
        obs_dim=29,
        latent_dim=128,
        hidden_dim=64,
        num_heads=2,
        max_seq_len=8,
        n_layers=1,
        dropout_rate=0.2,
        alibi_slope=0.4,
        local_window_size=4,
    )
    rng = jax.random.PRNGKey(0)
    rng, r1, r2, r3, r4 = jax.random.split(rng, 5)
    dummy_s = jax.random.normal(r1, (4, 29))
    dummy_seq = jax.random.normal(r2, (4, 8, 128))
    dummy_mask = jnp.array([
        [True, True, True, True, False, False, False, False],
        [True, True, True, True, True, True, False, False],
        [True, True, False, False, False, False, False, False],
        [True, True, True, True, True, True, True, True],
    ])
    dummy_curv = jax.random.uniform(r3, (4, 8, 1), minval=-1.0, maxval=1.0)

    params = model.init({"params": r4, "dropout": r4}, dummy_s, dummy_seq, dummy_mask, dummy_curv, deterministic=False)["params"]

    # Eval mode forward
    z_eval = model.apply({"params": params}, dummy_s, dummy_seq, dummy_mask, dummy_curv, deterministic=True)
    assert z_eval.shape == (4, 128)
    norms = jnp.linalg.norm(z_eval, axis=-1)
    np.testing.assert_allclose(norms, np.sqrt(128), atol=1e-3)

    # Train mode forward with dropout
    z_train = model.apply({"params": params}, dummy_s, dummy_seq, dummy_mask, dummy_curv, deterministic=False, rngs={"dropout": r1})
    assert z_train.shape == (4, 128)
    np.testing.assert_allclose(jnp.linalg.norm(z_train, axis=-1), np.sqrt(128), atol=1e-3)

    # Fallback without curvatures
    z_nocurv = model.apply({"params": params}, dummy_s, dummy_seq, dummy_mask, None, deterministic=True)
    assert z_nocurv.shape == (4, 128)
    np.testing.assert_allclose(jnp.linalg.norm(z_nocurv, axis=-1), np.sqrt(128), atol=1e-3)


class _MockDist:
    def __init__(self, val):
        self.val = val
    def mode(self):
        return self.val


def test_enhanced_sequence_wp_differentiable_train_step():
    model = FlaxEnhancedSequenceWaypointAttentionTranslator(
        obs_dim=29, latent_dim=128, hidden_dim=64, num_heads=2, max_seq_len=8, n_layers=1, dropout_rate=0.2, alibi_slope=0.4, local_window_size=4
    )
    rng, r1, r2 = jax.random.split(jax.random.PRNGKey(0), 3)
    dummy_s, dummy_seq = jnp.ones((4, 29)), jnp.ones((4, 8, 128))
    dummy_mask, dummy_curv = jnp.ones((4, 8), dtype=bool), jnp.ones((4, 8, 1), dtype=jnp.float32)
    params = model.init({"params": r1, "dropout": r1}, dummy_s, dummy_seq, dummy_mask, dummy_curv, deterministic=False)["params"]

    mock_actor = lambda s, z, **kw: _MockDist(jnp.tanh(s[:, :8] + z[:, :8]))
    mock_f, mock_b = (lambda s, z, **kw: z), (lambda g: g)

    optimizer = optax.adam(1e-3)
    step_fn = make_enhanced_sequence_wp_train_step(model.apply, mock_actor, mock_f, mock_b, optimizer)
    batch = {"state": dummy_s, "wp_seq": dummy_seq, "seq_mask": dummy_mask, "curvatures": dummy_curv, "z_target": jnp.ones((4, 128)) * 0.1, "a_target": jnp.zeros((4, 8))}
    lambdas = {"l_cos": 1.0, "l_mse": 0.1, "l_action": 0.5, "l_reach": 0.02, "l_goal": 0.05, "l_aux": 0.2}

    new_params, new_opt_state, metrics, new_rng = step_fn(params, optimizer.init(params), batch, lambdas, r2)
    assert metrics["loss"] is not None
    assert "loss_action" in metrics and "loss_aux" in metrics and "cos_sim" in metrics

