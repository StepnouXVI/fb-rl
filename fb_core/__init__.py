"""
fb_core: Shared zero-duplication framework for Forward-Backward RL Multi-Subgoal Planning.
"""

from fb_core.math_utils import (
    normalize_latent,
    safe_inner_product,
    safe_log_dot,
    compute_quasi_distance,
    log_successor_bisection_score,
)

__all__ = [
    "normalize_latent",
    "safe_inner_product",
    "safe_log_dot",
    "compute_quasi_distance",
    "log_successor_bisection_score",
]
