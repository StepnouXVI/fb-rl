import pytest
import numpy as np
import src
from src.agent_loader import load_pretrained_agent
from src.planners import (
    BaselinePlanner,
    RecursiveBisectionPlanner,
    BufferGraphPlanner,
    DistilledMLPPlanner,
    EnhancedSequenceWaypointAttentionPlanner,
)

@pytest.fixture(scope="module")
def setup_agent():
    return load_pretrained_agent("fb-test", "medium")

def test_baseline_planner(setup_agent):
    agent, env, train_ds, _, _ = setup_agent
    planner = BaselinePlanner(agent)
    obs = train_ds["observations"][0]
    goal_z = np.asarray(agent.normalize_z(agent.network.select("backward_repr")(train_ds["observations"][10:11]))[0])
    action = planner.sample_action(obs, goal_z)
    assert action.shape == (8,)
    assert np.all(action >= -1.0) and np.all(action <= 1.0)

def test_recursive_bisection_planner(setup_agent):
    agent, env, train_ds, _, _ = setup_agent
    planner = RecursiveBisectionPlanner(agent, train_ds["observations"][:200], n_candidates=30)
    obs = train_ds["observations"][0]
    goal_z = np.asarray(agent.normalize_z(agent.network.select("backward_repr")(train_ds["observations"][10:11]))[0])
    action = planner.sample_action(obs, goal_z)
    assert action.shape == (8,)

def test_buffer_graph_planner(setup_agent):
    agent, env, train_ds, _, _ = setup_agent
    planner = BufferGraphPlanner(agent, train_ds["observations"][:200], n_landmarks=20)
    obs = train_ds["observations"][0]
    goal_z = np.asarray(agent.normalize_z(agent.network.select("backward_repr")(train_ds["observations"][10:11]))[0])
    action = planner.sample_action(obs, goal_z)
    assert action.shape == (8,)

def test_distilled_mlp_planner(setup_agent):
    agent, env, train_ds, _, _ = setup_agent
    planner = DistilledMLPPlanner(agent)
    obs = train_ds["observations"][0]
    goal_z = np.asarray(agent.normalize_z(agent.network.select("backward_repr")(train_ds["observations"][10:11]))[0])
    action = planner.sample_action(obs, goal_z)
    assert action.shape == (8,)

def test_enhanced_sequence_waypoint_planner(setup_agent, tmp_path):
    import pickle
    import jax
    import jax.numpy as jnp
    from src.waypoint_translators import FlaxEnhancedSequenceWaypointAttentionTranslator

    agent, env, train_ds, _, _ = setup_agent
    model = FlaxEnhancedSequenceWaypointAttentionTranslator(
        obs_dim=29, latent_dim=128, hidden_dim=64, num_heads=2, max_seq_len=8, n_layers=1
    )
    rng = jax.random.PRNGKey(0)
    dummy_s = jnp.zeros((1, 29))
    dummy_seq = jnp.zeros((1, 8, 128))
    dummy_mask = jnp.ones((1, 8), dtype=bool)
    dummy_curv = jnp.ones((1, 8, 1), dtype=jnp.float32)
    params = model.init({"params": rng, "dropout": rng}, dummy_s, dummy_seq, dummy_mask, dummy_curv)["params"]

    ckpt_path = tmp_path / "mock_enhanced_seq.pkl"
    with open(ckpt_path, "wb") as f:
        pickle.dump({
            "params": params,
            "config": {"hidden_dim": 64, "num_heads": 2, "max_seq_len": 8, "n_layers": 1}
        }, f)

    planner = EnhancedSequenceWaypointAttentionPlanner(
        agent=agent,
        dataset_observations=train_ds["observations"][:200],
        checkpoint_path=str(ckpt_path),
        n_landmarks=20,
        max_seq_len=8,
        hidden_dim=64,
        num_heads=2,
        n_layers=1,
    )
    obs = train_ds["observations"][0]
    goal_z = np.asarray(agent.normalize_z(agent.network.select("backward_repr")(train_ds["observations"][10:11]))[0])
    action = planner.sample_action(obs, goal_z)
    assert action.shape == (8,)
    assert np.all(action >= -1.0) and np.all(action <= 1.0)

