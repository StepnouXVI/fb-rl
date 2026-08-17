"""
Mathematical utility functions for Forward-Backward (FB) Representations.
Includes numerically safe operations, latent normalizations, and quasi-distance metrics.
"""

from typing import Union
import numpy as np
import jax.numpy as jnp
import torch


def normalize_latent(z: Union[np.ndarray, jnp.ndarray, torch.Tensor], latent_dim: int = 128, eps: float = 1e-8) -> Union[np.ndarray, jnp.ndarray, torch.Tensor]:
    """
    Normalize latent intention vector z such that ||z||_2 = sqrt(latent_dim).
    # ponytail: matches Touati & Ollivier (2021) and Stojanovic & Proutiere (2026) convention.
    """
    if isinstance(z, np.ndarray):
        norm = np.linalg.norm(z, axis=-1, keepdims=True)
        return (z / (norm + eps)) * np.sqrt(latent_dim)
    elif isinstance(z, torch.Tensor):
        norm = torch.linalg.norm(z, dim=-1, keepdim=True)
        return (z / (norm + eps)) * np.sqrt(latent_dim)
    else:  # JAX
        norm = jnp.linalg.norm(z, axis=-1, keepdims=True)
        return (z / (norm + eps)) * jnp.sqrt(latent_dim)


def safe_inner_product(f: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Computes inner product between Forward embedding F(s, z) and Backward embedding B(s').
    f: shape (..., D)
    b: shape (..., D)
    returns: scalar or tensor of shape (...)
    """
    return np.sum(f * b, axis=-1)


def safe_log_dot(f: np.ndarray, b: np.ndarray, eps: float = 1e-6, max_val: float = 1e6) -> np.ndarray:
    """
    Numerically safe log-inner product: log(max(F^T B, eps)).
    Guarantees no NaNs or -inf even if F^T B <= 0 due to approximation error.
    # ponytail: simple clip with eps prevents log domain breakdown.
    """
    dot = np.sum(f * b, axis=-1)
    clipped_dot = np.clip(dot, a_min=eps, a_max=max_val)
    return np.log(clipped_dot)


def compute_quasi_distance(f: np.ndarray, b: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Computes directed quasi-distance d(s_i, s_j) = -log(max(F(s_i, B(s_j))^T B(s_j), eps)).
    Higher successor measure value -> lower travel cost.
    """
    return -safe_log_dot(f, b, eps=eps)


def log_successor_bisection_score(f_sw: np.ndarray, b_w: np.ndarray, f_wg: np.ndarray, b_g: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Computes bisection score: log M(s -> w) + log M(w -> g).
    Maximizing this score finds the optimal midpoint w* that lies on the high-density path from s to g.
    """
    score_sw = safe_log_dot(f_sw, b_w, eps=eps)
    score_wg = safe_log_dot(f_wg, b_g, eps=eps)
    return score_sw + score_wg
