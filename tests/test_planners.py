"""
Unit tests for all 4 hierarchical planners (Baseline + 3 Ideas).
"""

import numpy as np
import pytest
from fb_core.fb_model_wrapper import FBModelWrapper
from fb_core.dataset_sampler import DatasetSampler
from planners.baseline_planner import SingleIntentionPlanner
from planners.recursive_bisection import RecursiveBisectionPlanner
from planners.buffer_graph import BufferGraphPlanner
from planners.distilled_mlp import DistilledMLPPlanner


@pytest.fixture
def setup_env_and_data():
    np.random.seed(42)
    states = np.random.randn(100, 29).astype(np.float32)
    sampler = DatasetSampler(states, seed=42)
    fb_model = FBModelWrapper(agent=None, latent_dim=128)
    obs = states[0]
    goal = states[-1]
    return sampler, fb_model, obs, goal


def test_baseline_planner(setup_env_and_data):
    sampler, fb_model, obs, goal = setup_env_and_data
    planner = SingleIntentionPlanner(fb_model=fb_model)
    planner.reset(obs, goal)
    
    z = planner.get_intention(obs, goal, step=0)
    assert z.shape == (128,)
    np.testing.assert_allclose(np.linalg.norm(z), np.sqrt(128), rtol=1e-4)


def test_recursive_bisection_planner(setup_env_and_data):
    sampler, fb_model, obs, goal = setup_env_and_data
    planner = RecursiveBisectionPlanner(
        fb_model=fb_model,
        dataset_sampler=sampler,
        max_depth=2,
        n_candidates=50,
    )
    planner.reset(obs, goal)
    assert len(planner.subgoals) >= 1
    
    z = planner.get_intention(obs, goal, step=0)
    assert z.shape == (128,)
    np.testing.assert_allclose(np.linalg.norm(z), np.sqrt(128), rtol=1e-4)


def test_buffer_graph_planner(setup_env_and_data):
    sampler, fb_model, obs, goal = setup_env_and_data
    planner = BufferGraphPlanner(
        fb_model=fb_model,
        dataset_sampler=sampler,
        n_landmarks=20,
    )
    planner.reset(obs, goal)
    assert len(planner.path_waypoints) >= 1
    
    z = planner.get_intention(obs, goal, step=0)
    assert z.shape == (128,)
    np.testing.assert_allclose(np.linalg.norm(z), np.sqrt(128), rtol=1e-4)


def test_distilled_mlp_planner(setup_env_and_data):
    sampler, fb_model, obs, goal = setup_env_and_data
    planner = DistilledMLPPlanner(
        fb_model=fb_model,
        dataset_sampler=sampler,
        obs_dim=29,
        latent_dim=128,
        device="cpu",
    )
    train_res = planner.train_distillation(n_pairs=30, epochs=3, batch_size=10)
    assert "final_loss" in train_res
    
    planner.reset(obs, goal)
    z = planner.get_intention(obs, goal, step=0)
    assert z.shape == (128,)
    np.testing.assert_allclose(np.linalg.norm(z), np.sqrt(128), rtol=1e-4)
