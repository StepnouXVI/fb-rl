"""
Unit tests for fb_core: FBModelWrapper, DatasetSampler, MetricsCollector, and Evaluator.
"""

import numpy as np
from fb_core.fb_model_wrapper import FBModelWrapper
from fb_core.dataset_sampler import DatasetSampler
from fb_core.metrics_collector import MetricsCollector
from fb_core.base_planner import BaseHierarchicalPlanner
from fb_core.evaluator import EpisodeEvaluator


class DummyPlanner(BaseHierarchicalPlanner):
    def reset(self, initial_obs: np.ndarray, goal: np.ndarray) -> None:
        super().reset(initial_obs, goal)

    def get_intention(self, obs: np.ndarray, goal: np.ndarray, step: int = 0) -> np.ndarray:
        # Returns simple pseudo-intention
        return np.ones(128, dtype=np.float32)


def test_fb_model_wrapper_methods():
    model = FBModelWrapper(agent=None, latent_dim=128)
    obs = np.random.randn(29).astype(np.float32)
    goal = np.random.randn(29).astype(np.float32)
    
    b_g = model.encode_backward(goal)
    assert b_g.shape == (128,)
    np.testing.assert_allclose(np.linalg.norm(b_g), np.sqrt(128), rtol=1e-4)
    
    f_s = model.evaluate_forward(obs, b_g)
    assert f_s.shape == (1, 128)
    
    action = model.sample_action(obs, b_g)
    assert action.shape == (8,)
    assert np.all(action >= -1.0) and np.all(action <= 1.0)


def test_dataset_sampler_fps():
    # 20 states in 2D space
    states = np.random.randn(50, 29).astype(np.float32)
    sampler = DatasetSampler(states, seed=42)
    
    candidates = sampler.sample_candidates(10)
    assert candidates.shape == (10, 29)
    
    landmarks = sampler.select_fps_landmarks(n_landmarks=5)
    assert landmarks.shape == (5, 29)


def test_metrics_collector_and_bootstrap():
    data = [80.0, 85.0, 75.0, 90.0, 80.0]
    ci = MetricsCollector.compute_bootstrap_ci(data)
    assert ci["mean"] == 82.0
    assert ci["ci_lower"] <= 82.0 <= ci["ci_upper"]
    
    summary = MetricsCollector.aggregate_runs({
        "Baseline": [{"success_rate": 40.0, "mean_steps": 500, "mean_latency_ms": 0.1}],
        "Method1": [{"success_rate": 80.0, "mean_steps": 300, "mean_latency_ms": 1.0}],
    })
    assert "Baseline" in summary
    assert "Method1" in summary
    latex_tab = MetricsCollector.generate_latex_table(summary)
    assert r"\begin{table}" in latex_tab


def test_evaluator_dummy():
    planner = DummyPlanner("DummyPlanner")
    fb_model = FBModelWrapper(agent=None, latent_dim=128)
    evaluator = EpisodeEvaluator(max_episode_steps=10)
    
    res = evaluator.evaluate_planner(planner, fb_model, env=None, num_episodes=2, seed=0)
    assert res["method_name"] == "DummyPlanner"
    assert len(res["episode_successes"]) == 2
