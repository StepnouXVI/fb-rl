import pytest
import numpy as np
import src
from src.agent_loader import load_pretrained_agent
from src.planners import BaselinePlanner, RecursiveBisectionPlanner, BufferGraphPlanner, DistilledMLPPlanner

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
