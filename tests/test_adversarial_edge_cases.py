"""
Adversarial Edge Cases & Stress Tests for FB-RL Hierarchical Planning.
Audited under Fable Judge and Ponytail Protocols.
"""

import numpy as np
import pytest
import torch
from fb_core.fb_model_wrapper import FBModelWrapper
from fb_core.dataset_sampler import DatasetSampler
from fb_core.math_utils import (
    normalize_latent,
    safe_inner_product,
    safe_log_dot,
    compute_quasi_distance,
    log_successor_bisection_score,
)
from fb_core.metrics_collector import MetricsCollector
from fb_core.evaluator import EpisodeEvaluator
from planners.baseline_planner import SingleIntentionPlanner
from planners.recursive_bisection import RecursiveBisectionPlanner
from planners.buffer_graph import BufferGraphPlanner
from planners.distilled_mlp import DistilledMLPPlanner


@pytest.fixture
def minimal_fb_setup():
    """Provides standard mock wrapper and dataset sampler for edge case testing."""
    np.random.seed(1337)
    states = np.random.randn(50, 29).astype(np.float32)
    sampler = DatasetSampler(states, seed=1337)
    fb_model = FBModelWrapper(agent=None, latent_dim=128)
    return sampler, fb_model


# ==============================================================================
# 1. Stress Test: Start State Already at Goal (s == g)
# ==============================================================================
def test_start_state_already_at_goal(minimal_fb_setup):
    """Verify all planners handle identical start and goal without division by zero or NaN."""
    sampler, fb_model = minimal_fb_setup
    s_and_g = np.ones(29, dtype=np.float32) * 3.5

    planners = [
        SingleIntentionPlanner(fb_model=fb_model),
        RecursiveBisectionPlanner(fb_model=fb_model, dataset_sampler=sampler, max_depth=2, n_candidates=20),
        BufferGraphPlanner(fb_model=fb_model, dataset_sampler=sampler, n_landmarks=15),
        DistilledMLPPlanner(fb_model=fb_model, dataset_sampler=sampler, obs_dim=29, latent_dim=128),
    ]

    for planner in planners:
        if isinstance(planner, DistilledMLPPlanner):
            planner.train_distillation(n_pairs=20, epochs=2, batch_size=10)

        planner.reset(s_and_g, s_and_g)
        z = planner.get_intention(s_and_g, s_and_g, step=0)

        assert z.shape == (128,), f"Failed shape for {planner.name}"
        assert not np.any(np.isnan(z)), f"NaN detected in intention for {planner.name}"
        assert not np.any(np.isinf(z)), f"Inf detected in intention for {planner.name}"

        # Action execution should also be safe and within [-1.0, 1.0]
        action = fb_model.sample_action(s_and_g, z)
        assert action.shape == (8,)
        assert not np.any(np.isnan(action))
        assert np.all(action >= -1.0) and np.all(action <= 1.0)


# ==============================================================================
# 2. Stress Test: Extremely Large Distances between s and g
# ==============================================================================
def test_extremely_large_distances(minimal_fb_setup):
    """Verify numerical stability with coordinates at extreme scales (+/- 1e7)."""
    sampler, fb_model = minimal_fb_setup
    s_extreme = np.full(29, -1e7, dtype=np.float32)
    g_extreme = np.full(29, 1e7, dtype=np.float32)

    planners = [
        SingleIntentionPlanner(fb_model=fb_model),
        RecursiveBisectionPlanner(fb_model=fb_model, dataset_sampler=sampler, max_depth=2, n_candidates=20),
        BufferGraphPlanner(fb_model=fb_model, dataset_sampler=sampler, n_landmarks=15),
        DistilledMLPPlanner(fb_model=fb_model, dataset_sampler=sampler, obs_dim=29, latent_dim=128),
    ]

    for planner in planners:
        if isinstance(planner, DistilledMLPPlanner):
            planner.train_distillation(n_pairs=20, epochs=2, batch_size=10)

        planner.reset(s_extreme, g_extreme)
        z = planner.get_intention(s_extreme, g_extreme, step=0)

        assert z.shape == (128,)
        assert not np.any(np.isnan(z)), f"NaN in extreme distance test for {planner.name}"
        assert not np.any(np.isinf(z)), f"Inf in extreme distance test for {planner.name}"
        np.testing.assert_allclose(np.linalg.norm(z), np.sqrt(128), rtol=1e-3)


# ==============================================================================
# 3. Stress Test: Single-Element Candidate Buffers (N = 1) and Edge Landmarking
# ==============================================================================
def test_single_element_buffer_stress():
    """Verify planners operate safely when dataset contains only N=1 sample."""
    single_state = np.ones((1, 29), dtype=np.float32)
    sampler_single = DatasetSampler(single_state, seed=42)
    fb_model = FBModelWrapper(agent=None, latent_dim=128)

    # 1. DatasetSampler edge checks
    candidates = sampler_single.sample_candidates(10)
    assert candidates.shape == (1, 29)

    landmarks = sampler_single.select_fps_landmarks(n_landmarks=5)
    assert landmarks.shape == (1, 29)

    # 2. RecursiveBisection with N=1 candidate
    rb_planner = RecursiveBisectionPlanner(
        fb_model=fb_model,
        dataset_sampler=sampler_single,
        max_depth=2,
        n_candidates=1,
    )
    s = np.zeros(29, dtype=np.float32)
    g = np.ones(29, dtype=np.float32) * 5.0

    rb_planner.reset(s, g)
    z_rb = rb_planner.get_intention(s, g, step=0)
    assert z_rb.shape == (128,)
    assert not np.any(np.isnan(z_rb))

    # 3. BufferGraph with N=1 landmark
    bg_planner = BufferGraphPlanner(
        fb_model=fb_model,
        dataset_sampler=sampler_single,
        n_landmarks=1,
    )
    bg_planner.reset(s, g)
    z_bg = bg_planner.get_intention(s, g, step=0)
    assert z_bg.shape == (128,)
    assert not np.any(np.isnan(z_bg))


# ==============================================================================
# 4. Stress Test: Degenerate / Zeroed Inputs & Numerical Safety
# ==============================================================================
def test_degenerate_zero_inputs():
    """Verify mathematical functions and wrappers on all-zero arrays."""
    fb_model = FBModelWrapper(agent=None, latent_dim=128)
    zeros_29 = np.zeros(29, dtype=np.float32)
    zeros_128 = np.zeros(128, dtype=np.float32)

    # Normalize zero latent
    z_norm = normalize_latent(zeros_128, latent_dim=128)
    assert not np.any(np.isnan(z_norm))
    assert not np.any(np.isinf(z_norm))

    # Safe inner product on zeros
    ip = safe_inner_product(zeros_128, zeros_128)
    assert ip == 0.0

    # Safe log dot on negative / zero values
    log_dot_zero = safe_log_dot(zeros_128, zeros_128, eps=1e-6)
    np.testing.assert_allclose(log_dot_zero, np.log(1e-6), rtol=1e-5)
    assert not np.isnan(log_dot_zero)

    # Quasi distance on zeros
    qd = compute_quasi_distance(zeros_128, zeros_128)
    assert not np.isnan(qd)

    # Bisection score on zeros
    bs = log_successor_bisection_score(zeros_128, zeros_128, zeros_128, zeros_128)
    assert not np.isnan(bs)

    # Action sampling with zero obs and zero intention
    action = fb_model.sample_action(zeros_29, zeros_128)
    assert action.shape == (8,)
    assert not np.any(np.isnan(action))


# ==============================================================================
# 5. Stress Test: Rapidly Switching / Oscillating Goals
# ==============================================================================
def test_rapidly_oscillating_goals(minimal_fb_setup):
    """Verify that planners adapt instantaneously when goals oscillate every step."""
    sampler, fb_model = minimal_fb_setup
    s = np.zeros(29, dtype=np.float32)
    g_north = np.zeros(29, dtype=np.float32)
    g_north[1] = 10.0  # (0, 10)
    g_east = np.zeros(29, dtype=np.float32)
    g_east[0] = 10.0   # (10, 0)

    planners = [
        SingleIntentionPlanner(fb_model=fb_model),
        RecursiveBisectionPlanner(fb_model=fb_model, dataset_sampler=sampler, max_depth=1, n_candidates=20),
        BufferGraphPlanner(fb_model=fb_model, dataset_sampler=sampler, n_landmarks=15),
        DistilledMLPPlanner(fb_model=fb_model, dataset_sampler=sampler, obs_dim=29, latent_dim=128),
    ]

    for planner in planners:
        if isinstance(planner, DistilledMLPPlanner):
            planner.train_distillation(n_pairs=20, epochs=2, batch_size=10)

        # Oscillate goals for 10 alternating steps
        intentions = []
        for step in range(10):
            current_target = g_north if (step % 2 == 0) else g_east
            z = planner.get_intention(s, current_target, step=step)
            assert not np.any(np.isnan(z)), f"NaN at step {step} for {planner.name}"
            intentions.append(z)

        # Verify intention vectors alternate (cosine similarity between alternating steps < 0.99)
        dot_01 = float(np.dot(intentions[0], intentions[1])) / 128.0
        assert dot_01 < 0.99, f"Planner {planner.name} failed to switch intention on oscillating goals"


# ==============================================================================
# 6. Stress Test: Disconnected Topological Graph (Graph Fallback Behavior)
# ==============================================================================
def test_disconnected_topological_graph():
    """Verify BufferGraphPlanner gracefully falls back when no connected graph path exists."""
    # Create two distant isolated clusters in 2D
    cluster_a = np.zeros((10, 29), dtype=np.float32)
    cluster_a[:, :2] = np.random.randn(10, 2) - 50.0  # far left
    cluster_b = np.zeros((10, 29), dtype=np.float32)
    cluster_b[:, :2] = np.random.randn(10, 2) + 50.0  # far right
    states = np.concatenate([cluster_a, cluster_b], axis=0)

    sampler = DatasetSampler(states, seed=42)
    fb_model = FBModelWrapper(agent=None, latent_dim=128)

    # Set very high cutoff so inter-cluster reachability is cut off
    planner = BufferGraphPlanner(
        fb_model=fb_model,
        dataset_sampler=sampler,
        n_landmarks=20,
        reachability_cutoff=1e9,  # Force disconnect
    )

    s = states[0]
    g = states[-1]
    planner.reset(s, g)

    # Path should safely fall back to start landmark -> goal landmark -> goal
    assert len(planner.path_waypoints) >= 2
    z = planner.get_intention(s, g, step=0)
    assert z.shape == (128,)
    assert not np.any(np.isnan(z))


# ==============================================================================
# 7. Stress Test: Metrics Collector Edge Cases (Zero Variance, Single Element)
# ==============================================================================
def test_metrics_collector_edge_cases():
    """Verify MetricsCollector doesn't crash on empty, single-element, or zero-variance inputs."""
    # Empty data
    ci_empty = MetricsCollector.compute_bootstrap_ci([])
    assert ci_empty["mean"] == 0.0
    assert ci_empty["ci_lower"] == 0.0

    # Single element
    ci_single = MetricsCollector.compute_bootstrap_ci([42.0])
    assert ci_single["mean"] == 42.0
    assert ci_single["ci_lower"] == 42.0
    assert ci_single["ci_upper"] == 42.0

    # Zero variance (all identical)
    ci_zeros = MetricsCollector.compute_bootstrap_ci([0.0, 0.0, 0.0, 0.0, 0.0])
    assert ci_zeros["mean"] == 0.0
    assert ci_zeros["std"] == 0.0
    assert ci_zeros["ci_lower"] == 0.0
    assert ci_zeros["ci_upper"] == 0.0


# ==============================================================================
# 8. Stress Test: Multi-Framework Normalization Edge Cases (PyTorch & JAX)
# ==============================================================================
def test_framework_tensor_normalization_edge_cases():
    """Verify PyTorch and JAX tensor normalization on extreme and zero vectors."""
    import jax.numpy as jnp

    # PyTorch zero vector & extreme vector
    t_zero = torch.zeros(128, dtype=torch.float32)
    t_zero_norm = normalize_latent(t_zero, latent_dim=128)
    assert not torch.isnan(t_zero_norm).any()
    assert not torch.isinf(t_zero_norm).any()

    t_huge = torch.full((128,), 1e9, dtype=torch.float32)
    t_huge_norm = normalize_latent(t_huge, latent_dim=128)
    assert not torch.isnan(t_huge_norm).any()
    assert abs(float(torch.linalg.norm(t_huge_norm)) - np.sqrt(128)) < 1e-3

    # JAX zero vector & extreme vector
    j_zero = jnp.zeros(128, dtype=jnp.float32)
    j_zero_norm = normalize_latent(j_zero, latent_dim=128)
    assert not jnp.any(jnp.isnan(j_zero_norm))

    j_huge = jnp.full((128,), 1e9, dtype=jnp.float32)
    j_huge_norm = normalize_latent(j_huge, latent_dim=128)
    assert not jnp.any(jnp.isnan(j_huge_norm))
    assert abs(float(jnp.linalg.norm(j_huge_norm)) - np.sqrt(128)) < 1e-3


# ==============================================================================
# 9. Stress Test: Evaluator on Zero-Distance & Truncated Episodes
# ==============================================================================
def test_evaluator_adversarial_episodes(minimal_fb_setup):
    """Verify EpisodeEvaluator handles immediate goal completion safely."""
    sampler, fb_model = minimal_fb_setup
    planner = SingleIntentionPlanner(fb_model=fb_model)
    evaluator = EpisodeEvaluator(max_episode_steps=5, goal_threshold=1.0)

    # Class with custom synthetic env where goal is already at agent position
    class ImmediateGoalEnv:
        def reset(self):
            return np.zeros(29, dtype=np.float32), {"goal": np.zeros(29, dtype=np.float32)}
        def step(self, action):
            return np.zeros(29, dtype=np.float32), 1.0, True, False, {"success": True}

    res = evaluator.evaluate_planner(planner, fb_model, env=ImmediateGoalEnv(), num_episodes=5, seed=42)
    assert res["success_rate"] == 100.0
    assert res["mean_steps"] == 1.0

