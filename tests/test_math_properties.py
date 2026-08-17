"""
Unit tests for mathematical properties, numerical stability, and geometry in FB representations.
Performs adversarial verification of mathematical theorems, quasi-metrics, and log-space compositions.
"""

import numpy as np
import torch
import jax.numpy as jnp
from scipy.sparse.csgraph import dijkstra

from fb_core.math_utils import (
    normalize_latent,
    safe_log_dot,
    compute_quasi_distance,
    log_successor_bisection_score,
)
from fb_core.fb_model_wrapper import FBModelWrapper
from fb_core.dataset_sampler import DatasetSampler
from planners.buffer_graph import BufferGraphPlanner


# ==============================================================================
# 1. Latent Normalization & Invariance Tests
# ==============================================================================

def test_latent_normalization_numpy():
    """Verify ||normalize_latent(z)||_2 == sqrt(latent_dim) across various tensor dimensions."""
    latent_dim = 128
    rng = np.random.default_rng(42)

    # 1D vector
    z_1d = rng.standard_normal(latent_dim).astype(np.float32)
    z_1d_norm = normalize_latent(z_1d, latent_dim=latent_dim)
    assert z_1d_norm.shape == (latent_dim,)
    np.testing.assert_allclose(np.linalg.norm(z_1d_norm), np.sqrt(latent_dim), rtol=1e-5)

    # 2D batch
    z_2d = rng.standard_normal((32, latent_dim)).astype(np.float32)
    z_2d_norm = normalize_latent(z_2d, latent_dim=latent_dim)
    assert z_2d_norm.shape == (32, latent_dim)
    norms_2d = np.linalg.norm(z_2d_norm, axis=-1)
    np.testing.assert_allclose(norms_2d, np.sqrt(latent_dim), rtol=1e-5)

    # 3D tensor
    z_3d = rng.standard_normal((4, 16, latent_dim)).astype(np.float32)
    z_3d_norm = normalize_latent(z_3d, latent_dim=latent_dim)
    assert z_3d_norm.shape == (4, 16, latent_dim)
    norms_3d = np.linalg.norm(z_3d_norm, axis=-1)
    np.testing.assert_allclose(norms_3d, np.sqrt(latent_dim), rtol=1e-5)


def test_latent_normalization_cross_backend_consistency():
    """Verify cross-framework consistency between NumPy, PyTorch, and JAX backends."""
    latent_dim = 64
    rng = np.random.default_rng(123)
    raw_np = rng.standard_normal((10, latent_dim)).astype(np.float32)

    raw_torch = torch.from_numpy(raw_np)
    raw_jax = jnp.asarray(raw_np)

    norm_np = normalize_latent(raw_np, latent_dim=latent_dim)
    norm_torch = normalize_latent(raw_torch, latent_dim=latent_dim).cpu().numpy()
    norm_jax = np.asarray(normalize_latent(raw_jax, latent_dim=latent_dim))

    np.testing.assert_allclose(norm_np, norm_torch, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(norm_np, norm_jax, rtol=1e-6, atol=1e-6)


def test_latent_normalization_scale_invariance():
    """Verify scale invariance: normalize_latent(alpha * z) == normalize_latent(z) for all alpha > 0."""
    latent_dim = 128
    rng = np.random.default_rng(42)
    z = rng.standard_normal(latent_dim).astype(np.float32)
    z_ref = normalize_latent(z, latent_dim=latent_dim)

    scales = [1e-4, 1e-2, 0.5, 2.0, 100.0, 1e4]
    for scale in scales:
        z_scaled = z * scale
        z_norm = normalize_latent(z_scaled, latent_dim=latent_dim)
        np.testing.assert_allclose(
            z_norm, z_ref, rtol=1e-4, atol=1e-4,
            err_msg=f"Scale invariance violated for scale factor {scale}",
        )


def test_latent_normalization_idempotence():
    """Verify idempotence: normalize_latent(normalize_latent(z)) == normalize_latent(z)."""
    latent_dim = 128
    rng = np.random.default_rng(7)
    z = rng.standard_normal((20, latent_dim)).astype(np.float32)

    z_norm_once = normalize_latent(z, latent_dim=latent_dim)
    z_norm_twice = normalize_latent(z_norm_once, latent_dim=latent_dim)

    np.testing.assert_allclose(z_norm_twice, z_norm_once, rtol=1e-6, atol=1e-6)


def test_latent_normalization_zero_vector():
    """Verify zero-vector edge case does not produce NaN or Inf (guarded by epsilon)."""
    latent_dim = 128
    z_zero = np.zeros(latent_dim, dtype=np.float32)
    z_norm = normalize_latent(z_zero, latent_dim=latent_dim)

    assert not np.any(np.isnan(z_norm)), "Zero vector normalization produced NaN"
    assert not np.any(np.isinf(z_norm)), "Zero vector normalization produced Inf"
    np.testing.assert_allclose(z_norm, 0.0, atol=1e-7)


# ==============================================================================
# 2. Numerical Stability of safe_log_dot & Safe Inner Product
# ==============================================================================

def test_safe_log_dot_near_zero_and_negative():
    """
    Adversarially test safe_log_dot across extreme values:
    Negative, zero, subnormal, near-zero epsilon boundary, and large positive values.
    """
    eps = 1e-6
    max_val = 1e6

    # Test cases: (F, B, expected_dot)
    # 1. Heavily negative
    f_neg = np.array([-100.0, 0.0], dtype=np.float32)
    b_neg = np.array([10.0, 0.0], dtype=np.float32)  # dot = -1000.0

    # 2. Exactly zero
    f_zero = np.array([0.0, 0.0], dtype=np.float32)
    b_zero = np.array([1.0, 1.0], dtype=np.float32)  # dot = 0.0

    # 3. Subnormal / near-zero below eps
    f_tiny = np.array([1e-12, 0.0], dtype=np.float32)
    b_tiny = np.array([1.0, 0.0], dtype=np.float32)  # dot = 1e-12 < eps

    # 4. Normal positive
    f_pos = np.array([2.0, 3.0], dtype=np.float32)
    b_pos = np.array([4.0, 1.0], dtype=np.float32)  # dot = 11.0

    # 5. Extreme large exceeding max_val
    f_huge = np.array([1e5, 1e5], dtype=np.float32)
    b_huge = np.array([1e5, 1e5], dtype=np.float32)  # dot = 2e10 > max_val

    f_batch = np.stack([f_neg, f_zero, f_tiny, f_pos, f_huge])
    b_batch = np.stack([b_neg, b_zero, b_tiny, b_pos, b_huge])

    res = safe_log_dot(f_batch, b_batch, eps=eps, max_val=max_val)

    # Assertions on sanity
    assert not np.any(np.isnan(res)), "safe_log_dot produced NaNs"
    assert not np.any(np.isinf(res)), "safe_log_dot produced Infs"

    # Exact value checks
    np.testing.assert_allclose(res[0], np.log(eps), rtol=1e-5, err_msg="Negative dot did not clamp to log(eps)")
    np.testing.assert_allclose(res[1], np.log(eps), rtol=1e-5, err_msg="Zero dot did not clamp to log(eps)")
    np.testing.assert_allclose(res[2], np.log(eps), rtol=1e-5, err_msg="Sub-eps dot did not clamp to log(eps)")
    np.testing.assert_allclose(res[3], np.log(11.0), rtol=1e-5, err_msg="Normal dot value incorrect")
    np.testing.assert_allclose(res[4], np.log(max_val), rtol=1e-5, err_msg="Huge dot did not clamp to log(max_val)")


def test_safe_log_dot_tensor_broadcasting():
    """Verify safe_log_dot handles multi-dimensional batches and broadcasting properly."""
    f = np.ones((5, 1, 10), dtype=np.float32)
    b = np.ones((1, 8, 10), dtype=np.float32)

    res = safe_log_dot(f, b)
    assert res.shape == (5, 8), f"Expected shape (5, 8), got {res.shape}"
    # dot product = 10 * (1*1) = 10.0
    np.testing.assert_allclose(res, np.log(10.0), rtol=1e-5)


# ==============================================================================
# 3. Directed Quasi-Metric Geometry & Triangle Inequality
# ==============================================================================

def test_directed_quasi_metric_asymmetry():
    """
    Verify directed quasi-distance is generally asymmetric:
    In non-reversible dynamical systems, c(s_i, s_j) != c(s_j, s_i).
    """
    # Downstream flow representation: s1 -> s2 has high reachability, s2 -> s1 has low reachability
    f_1_to_2 = np.array([2.0, 0.5])
    b_2 = np.array([2.0, 0.5])  # dot = 4.25 (high)

    f_2_to_1 = np.array([0.05, 0.01])
    b_1 = np.array([2.0, 0.5])  # dot = 0.105 (low)

    cost_1_to_2 = compute_quasi_distance(f_1_to_2, b_2)
    cost_2_to_1 = compute_quasi_distance(f_2_to_1, b_1)

    assert cost_1_to_2 < cost_2_to_1, "Downstream travel must be cheaper than upstream travel"
    assert not np.isclose(cost_1_to_2, cost_2_to_1), "Quasi-metric must capture directional asymmetry"


def test_quasi_distance_relaxed_triangle_inequality():
    """
    Verify the relaxed / sub-additive triangle inequality for Successor Measure quasi-distance:
    c(s_i, s_k) <= c(s_i, s_j) + c(s_j, s_k) + delta.

    Mathematical Justification:
    Under Markovian dynamics, M(s_i -> s_k) >= eta * M(s_i -> s_j) * M(s_j -> s_k) for some eta > 0.
    Taking -log: -log M(i->k) <= -log M(i->j) - log M(j->k) - log(eta).
    Thus c(i, k) <= c(i, j) + c(j, k) + delta, where delta = -log(eta) >= 0.
    """
    # Simulate a 3-step chain s_i -> s_j -> s_k with transition measures in (0, 1]
    m_ij = 0.6  # transition i -> j
    m_jk = 0.7  # transition j -> k
    
    # Due to intermediate state dispersion / bottleneck, direct composite measure
    # satisfies m_ik >= eta * (m_ij * m_jk) with eta > 0
    eta = 0.5
    m_ik = eta * m_ij * m_jk  # 0.21

    f_ij = np.array([m_ij, 0.0])
    b_j = np.array([1.0, 0.0])

    f_jk = np.array([m_jk, 0.0])
    b_k = np.array([1.0, 0.0])

    f_ik = np.array([m_ik, 0.0])

    c_ij = compute_quasi_distance(f_ij, b_j)
    c_jk = compute_quasi_distance(f_jk, b_k)
    c_ik = compute_quasi_distance(f_ik, b_k)

    delta = -np.log(eta)
    assert delta >= 0.0, "Delta slack parameter must be non-negative"
    assert c_ik <= c_ij + c_jk + delta + 1e-6, "Relaxed triangle inequality failed"


def test_buffer_graph_dijkstra_exact_triangle_inequality():
    """
    Verify that the shortest path metric d_G on the BufferGraphPlanner
    strictly satisfies the exact directed triangle inequality:
    d_G(u, w) <= d_G(u, v) + d_G(v, w) for all nodes u, v, w.
    """
    # Create synthetic spatial dataset
    np.random.seed(42)
    pts = np.random.uniform(-5, 5, size=(30, 29)).astype(np.float32)
    sampler = DatasetSampler(pts, seed=42)
    fb_model = FBModelWrapper(latent_dim=64)

    planner = BufferGraphPlanner(
        fb_model=fb_model,
        dataset_sampler=sampler,
        n_landmarks=20,
        reachability_cutoff=0.01,
    )
    planner.build_graph()
    adj = planner.adj_matrix
    assert adj is not None

    # Compute all-pairs shortest paths via Dijkstra
    dist_matrix = dijkstra(csgraph=adj, directed=True, return_predecessors=False)

    n_nodes = len(adj)
    # Check triangle inequality for all reachable triplets (u, v, w)
    violations = 0
    checked = 0
    for u in range(n_nodes):
        for v in range(n_nodes):
            for w in range(n_nodes):
                d_uv = dist_matrix[u, v]
                d_vw = dist_matrix[v, w]
                d_uw = dist_matrix[u, w]

                if np.isfinite(d_uv) and np.isfinite(d_vw) and np.isfinite(d_uw):
                    checked += 1
                    if d_uw > d_uv + d_vw + 1e-5:
                        violations += 1

    assert checked > 0, "No finite path triplets found in graph"
    assert violations == 0, f"Found {violations} triangle inequality violations in graph metric"


# ==============================================================================
# 4. Transition Probability Composition in Log Space
# ==============================================================================

def test_transition_probability_log_space_composition():
    """
    Demonstrate that multi-hop transition composition in log-space:
    log P(s_0 -> s_K) = sum_{k=0}^{K-1} log M(s_k -> s_{k+1})
    prevents catastrophic floating-point underflow that occurs in linear space product.
    """
    K = 100  # 100 subgoals
    # Each transition has small but valid probability M_k = 0.05
    m_k = 0.05
    f_k = np.full((K, 1), m_k, dtype=np.float32)
    b_k = np.ones((K, 1), dtype=np.float32)

    # Linear space product: 0.05^100 -> ~ 10^(-130) -> underflow to 0.0 in float32
    linear_product_float32 = np.prod(f_k[:, 0])
    assert linear_product_float32 == 0.0, "Float32 failed to demonstrate linear underflow as expected"

    # Log space sum: 100 * log(0.05) = 100 * (-2.9957) = -299.57 (exact and well-conditioned)
    log_terms = safe_log_dot(f_k, b_k)
    log_composition = np.sum(log_terms)

    assert not np.isnan(log_composition), "Log space composition produced NaN"
    assert not np.isinf(log_composition), "Log space composition produced Inf"
    np.testing.assert_allclose(log_composition, K * np.log(m_k), rtol=1e-5)


def test_recursive_bisection_midpoint_score_optimality():
    """
    Verify that log_successor_bisection_score correctly identifies the optimal midpoint
    and preserves strict monotonic order under varying bottleneck qualities.
    """
    # Candidate A: high quality bridge (s -> A: 0.8, A -> g: 0.7) => composite = 0.56
    score_A = log_successor_bisection_score(
        f_sw=np.array([0.8, 0.0]), b_w=np.array([1.0, 0.0]),
        f_wg=np.array([0.7, 0.0]), b_g=np.array([1.0, 0.0]),
    )

    # Candidate B: asymmetric bottleneck (s -> B: 0.9, B -> g: 0.1) => composite = 0.09
    score_B = log_successor_bisection_score(
        f_sw=np.array([0.9, 0.0]), b_w=np.array([1.0, 0.0]),
        f_wg=np.array([0.1, 0.0]), b_g=np.array([1.0, 0.0]),
    )

    # Candidate C: bad on both legs (s -> C: 0.05, C -> g: 0.05) => composite = 0.0025
    score_C = log_successor_bisection_score(
        f_sw=np.array([0.05, 0.0]), b_w=np.array([1.0, 0.0]),
        f_wg=np.array([0.05, 0.0]), b_g=np.array([1.0, 0.0]),
    )

    assert score_A > score_B > score_C, "Bisection ranking does not match true composite transition likelihood"
    np.testing.assert_allclose(score_A, np.log(0.8 * 0.7), rtol=1e-5)
    np.testing.assert_allclose(score_B, np.log(0.9 * 0.1), rtol=1e-5)
    np.testing.assert_allclose(score_C, np.log(0.05 * 0.05), rtol=1e-5)


# ==============================================================================
# 5. Distilled MLP Loss Mathematical Consistency
# ==============================================================================

def test_distilled_mlp_combined_loss_properties():
    """
    Verify mathematical properties of the distillation loss function:
    L(z, z*) = MSE(z, z*) + 0.5 * (1 - cos(z, z*)).
    
    1. Identity of indiscernibles: L(z, z) == 0.
    2. Non-negativity: L(z, z*) >= 0 for all z, z*.
    3. Strict positivity: L(z, z*) > 0 whenever z != z*.
    """
    cos_sim = torch.nn.CosineSimilarity(dim=-1)

    def compute_loss(pred_z: torch.Tensor, target_z: torch.Tensor) -> torch.Tensor:
        mse = torch.nn.functional.mse_loss(pred_z, target_z)
        cos_dist = torch.mean(1.0 - cos_sim(pred_z, target_z))
        return mse + 0.5 * cos_dist

    z_target = torch.randn(16, 128)
    z_target = z_target / torch.linalg.norm(z_target, dim=-1, keepdim=True) * np.sqrt(128)

    # 1. Identity: target vs target
    loss_zero = compute_loss(z_target, z_target)
    np.testing.assert_allclose(loss_zero.item(), 0.0, atol=1e-6)

    # 2. Opposite direction: worst case
    z_opposite = -z_target
    loss_opposite = compute_loss(z_opposite, z_target)
    assert loss_opposite.item() > 0.0
    # cos_sim is -1, so 1 - (-1) = 2; 0.5 * 2 = 1.0
    # mse is mean((-2*z)^2) = 4 * ||z||^2 / 128 = 4.0
    # total = 4.0 + 1.0 = 5.0
    np.testing.assert_allclose(loss_opposite.item(), 5.0, rtol=1e-4)

    # 3. Orthogonal direction
    z_ortho = torch.randn(16, 128)
    # Gram-Schmidt orthogonalize
    z_ortho = z_ortho - (torch.sum(z_ortho * z_target, dim=-1, keepdim=True) / torch.sum(z_target * z_target, dim=-1, keepdim=True)) * z_target
    z_ortho = z_ortho / torch.linalg.norm(z_ortho, dim=-1, keepdim=True) * np.sqrt(128)
    
    loss_ortho = compute_loss(z_ortho, z_target)
    # mse = mean(z_ortho^2 + z_target^2) = (128 + 128) / 128 = 2.0
    # cos_sim = 0, so 0.5 * (1 - 0) = 0.5
    # total = 2.0 + 0.5 = 2.5
    np.testing.assert_allclose(loss_ortho.item(), 2.5, rtol=1e-4)
    assert loss_opposite.item() > loss_ortho.item() > loss_zero.item()
