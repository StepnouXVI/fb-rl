"""Unit tests for direct intention distillation models and training steps."""

import numpy as np
import pytest
import jax
import jax.numpy as jnp
import optax

from src.agent import create_direct_intention_agent
from src.agent_loader import load_pretrained_agent
from src.networks import build_direct_translator
from src.training import make_direct_intention_train_step


@pytest.fixture(scope="module")
def setup_agent():
    agent, env, train_ds, val_ds, config = load_pretrained_agent(split="medium", seed=0)
    return agent, env, train_ds, val_ds, config


def test_gated_attention_translator_forward():
    model = build_direct_translator(model_type="gated_attn", latent_dim=128, hidden_dim=64, n_layers=2)
    rng = jax.random.PRNGKey(42)
    dummy_s = jnp.ones((4, 29))
    dummy_g = jnp.ones((4, 128))

    params = model.init(rng, dummy_s, dummy_g)["params"]
    z_out = model.apply({"params": params}, dummy_s, dummy_g)

    assert z_out.shape == (4, 128)
    expected_norm = np.sqrt(128)
    norms = jnp.linalg.norm(z_out, axis=-1)
    np.testing.assert_allclose(norms, expected_norm, atol=1e-3)


def test_dense_eca_translator_forward():
    model = build_direct_translator(model_type="dense_eca", latent_dim=128, hidden_dim=64, n_layers=2)
    rng = jax.random.PRNGKey(42)
    dummy_s = jnp.ones((4, 29))
    dummy_g = jnp.ones((4, 128))

    params = model.init(rng, dummy_s, dummy_g)["params"]
    z_out = model.apply({"params": params}, dummy_s, dummy_g)

    assert z_out.shape == (4, 128)
    expected_norm = np.sqrt(128)
    norms = jnp.linalg.norm(z_out, axis=-1)
    np.testing.assert_allclose(norms, expected_norm, atol=1e-3)


def test_differentiable_physics_training_step(setup_agent):
    agent, _, train_ds, _, _ = setup_agent
    model = build_direct_translator(model_type="gated_attn", latent_dim=128, hidden_dim=64, n_layers=2)

    rng = jax.random.PRNGKey(0)
    dummy_s = jnp.ones((2, 29))
    dummy_g = jnp.ones((2, 128))
    init_params = model.init(rng, dummy_s, dummy_g)["params"]

    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(init_params)
    step_fn = make_direct_intention_train_step(model, agent, optimizer)

    batch = {
        "state": jnp.asarray(train_ds["observations"][:4]),
        "goal_z": jnp.asarray(agent.normalize_z(agent.network.select("backward_repr")(train_ds["observations"][4:8]))),
        "z_target": jnp.asarray(agent.normalize_z(agent.network.select("backward_repr")(train_ds["observations"][8:12]))),
        "a_target": jnp.zeros((4, 8)),
    }
    lambdas = {
        "l_mse": 0.1,
        "l_cos": 1.0,
        "l_action": 0.5,
        "l_reach": 0.02,
        "l_goal": 0.05,
    }
    new_params, new_opt_state, metrics = step_fn(init_params, opt_state, batch, lambdas)

    assert new_params is not None
    assert "loss" in metrics
    assert "loss_action" in metrics
    assert "loss_reach" in metrics
    assert "loss_goal" in metrics
    assert not jnp.isnan(metrics["loss"])


def test_direct_intention_agent(setup_agent):
    agent, env, train_ds, _, _ = setup_agent
    staged_agent = create_direct_intention_agent(
        agent, model_type="gated_attn", hidden_dim=64, n_layers=2
    )
    obs = train_ds["observations"][0]
    goal_z = agent.normalize_z(agent.network.select("backward_repr")(train_ds["observations"][10:11]))[0]

    action = staged_agent.sample_action(obs, goal_z, step=0)
    assert action.shape == (8,)
    assert np.all(action >= -1.0) and np.all(action <= 1.0)
