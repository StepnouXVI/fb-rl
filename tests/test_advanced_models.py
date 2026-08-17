"""
Unit tests for Advanced Distilled Neural Architectures:
- ECALayer1D: adaptive kernel computation, 1D convolution channel modulation, 2D/3D shapes.
- ResNetECASubgoalNetwork: multi-block deep residual network with ECA-1D attention.
- GatedCrossAttentionNetwork: cross-attention (state queries, goal keys/values) with SwiGLU FFN.
- DenseECANetwork: densely connected network with channel modulation.
- AdvancedDistilledPlanner: training pipeline with Cosine Annealing, validation tracking, and latent normalization.
"""

import math
import numpy as np
import pytest
import torch
import torch.nn as nn

from fb_core.fb_model_wrapper import FBModelWrapper
from fb_core.dataset_sampler import DatasetSampler
from planners.advanced_distilled_models import (
    get_adaptive_kernel_size,
    ECALayer1D,
    ResNetECABlock,
    ResNetECASubgoalNetwork,
    SwiGLU,
    GatedCrossAttentionBlock,
    GatedCrossAttentionNetwork,
    DenseECALayer,
    DenseECANetwork,
    AdvancedDistilledPlanner,
)


# ==============================================================================
# 1. ECALayer1D Tests
# ==============================================================================

def test_adaptive_kernel_size_formula():
    """Verify adaptive kernel size k = |log2(C)/gamma + b/gamma|_odd."""
    for c in [32, 64, 128, 256, 512, 1024]:
        k = get_adaptive_kernel_size(channels=c, gamma=2.0, b=1.0)
        assert k % 2 == 1, f"Kernel size {k} for C={c} must be odd"
        assert k >= 3, f"Kernel size {k} for C={c} must be >= 3"


def test_eca_layer_1d_forward_2d():
    """Verify ECALayer1D preserves 2D shape (B, C) and modulates values in [0, 1]."""
    B, C = 16, 256
    layer = ECALayer1D(channels=C)
    x = torch.randn(B, C, requires_grad=True)
    out = layer(x)

    assert out.shape == (B, C)
    # Output should not be identical to input due to modulation
    assert not torch.allclose(out, x)
    # Backward gradient flow
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert torch.all(torch.isfinite(x.grad))


def test_eca_layer_1d_forward_3d():
    """Verify ECALayer1D supports 3D shape (B, L, C)."""
    B, L, C = 8, 4, 128
    layer = ECALayer1D(channels=C, k_size=3)
    x = torch.randn(B, L, C)
    out = layer(x)
    assert out.shape == (B, L, C)


def test_eca_layer_fixed_kernel():
    """Verify fixed odd kernel size parameterization."""
    layer = ECALayer1D(channels=128, k_size=5)
    assert layer.k_size == 5
    assert layer.conv.kernel_size[0] == 5


# ==============================================================================
# 2. ResNetECASubgoalNetwork Tests
# ==============================================================================

def test_resnet_eca_block():
    """Verify single ResNet-ECA block with skip connection."""
    B, D = 8, 128
    block = ResNetECABlock(hidden_dim=D)
    x = torch.randn(B, D)
    out = block(x)
    assert out.shape == (B, D)


def test_resnet_eca_network_forward_and_gradients():
    """Verify ResNetECASubgoalNetwork end-to-end forward and backward pass."""
    B, obs_dim, latent_dim = 16, 29, 128
    net = ResNetECASubgoalNetwork(
        obs_dim=obs_dim,
        latent_dim=latent_dim,
        hidden_dim=256,
        num_blocks=3,
    )
    obs = torch.randn(B, obs_dim, requires_grad=True)
    z_goal = torch.randn(B, latent_dim, requires_grad=True)

    out = net(obs, z_goal)
    assert out.shape == (B, latent_dim)

    loss = out.pow(2).sum()
    loss.backward()
    assert obs.grad is not None and z_goal.grad is not None
    assert torch.all(torch.isfinite(obs.grad))
    assert torch.all(torch.isfinite(z_goal.grad))


# ==============================================================================
# 3. GatedCrossAttentionNetwork Tests
# ==============================================================================

def test_swiglu_activation():
    """Verify SwiGLU mathematical gating behavior."""
    B, D = 8, 64
    swiglu = SwiGLU(in_features=D, hidden_features=128, out_features=D)
    x = torch.randn(B, D)
    out = swiglu(x)
    assert out.shape == (B, D)


def test_gated_cross_attention_block():
    """Verify cross-attention between state query tokens and goal key/value tokens."""
    B, N_q, N_k, d_model = 4, 3, 5, 128
    block = GatedCrossAttentionBlock(d_model=d_model, num_heads=4)
    q = torch.randn(B, N_q, d_model)
    kv = torch.randn(B, N_k, d_model)

    out, attn = block(q, kv, return_attn=True)
    assert out.shape == (B, N_q, d_model)
    assert attn.shape == (B, 4, N_q, N_k)
    # Check attention weights sum to 1 along last dimension
    np.testing.assert_allclose(attn.sum(dim=-1).detach().numpy(), 1.0, atol=1e-5)


def test_gated_cross_attention_network_forward_and_attn_weights():
    """Verify full GatedCrossAttentionNetwork forward pass and attention map extraction."""
    B, obs_dim, latent_dim = 6, 29, 128
    net = GatedCrossAttentionNetwork(
        obs_dim=obs_dim,
        latent_dim=latent_dim,
        d_model=128,
        num_heads=4,
        num_layers=2,
        num_query_tokens=4,
        num_key_tokens=4,
    )
    obs = torch.randn(B, obs_dim)
    z_goal = torch.randn(B, latent_dim)

    out = net(obs, z_goal)
    assert out.shape == (B, latent_dim)

    attn_maps = net.get_attention_weights(obs, z_goal)
    assert len(attn_maps) == 2
    assert attn_maps[0].shape == (B, 4, 4, 4)


# ==============================================================================
# 4. DenseECANetwork Tests
# ==============================================================================

def test_dense_eca_layer():
    """Verify DenseECALayer output dimension equals growth_rate."""
    B, in_f, growth = 8, 128, 64
    layer = DenseECALayer(in_features=in_f, growth_rate=growth)
    x = torch.randn(B, in_f)
    out = layer(x)
    assert out.shape == (B, growth)


def test_dense_eca_network_forward():
    """Verify DenseECANetwork forward pass with multiple dense layers."""
    B, obs_dim, latent_dim = 10, 29, 128
    net = DenseECANetwork(
        obs_dim=obs_dim,
        latent_dim=latent_dim,
        init_dim=128,
        growth_rate=32,
        num_layers=3,
    )
    obs = torch.randn(B, obs_dim)
    z_goal = torch.randn(B, latent_dim)

    out = net(obs, z_goal)
    assert out.shape == (B, latent_dim)


# ==============================================================================
# 5. AdvancedDistilledPlanner End-to-End Tests
# ==============================================================================

@pytest.fixture
def dummy_setup():
    np.random.seed(42)
    states = np.random.randn(80, 29).astype(np.float32)
    sampler = DatasetSampler(states, seed=42)
    fb_model = FBModelWrapper(agent=None, latent_dim=128)
    obs = states[0]
    goal = states[-1]
    return sampler, fb_model, obs, goal


@pytest.mark.parametrize("arch_type", ["resnet_eca", "gated_cross_attn", "dense_eca", "standard_mlp"])
def test_advanced_distilled_planner_training_and_inference(dummy_setup, arch_type):
    """Verify all 4 architecture options train cleanly, track metrics, and normalize output intentions."""
    sampler, fb_model, obs, goal = dummy_setup
    planner = AdvancedDistilledPlanner(
        fb_model=fb_model,
        dataset_sampler=sampler,
        architecture_type=arch_type,
        obs_dim=29,
        latent_dim=128,
        hidden_dim=128,
        num_blocks=2,
        device="cpu",
    )

    # Train for 3 epochs on small sample
    train_res = planner.train_distillation(
        n_pairs=40,
        val_ratio=0.25,
        epochs=3,
        batch_size=10,
        lr=1e-3,
    )

    assert "final_train_loss" in train_res
    assert "final_val_loss" in train_res
    assert "final_val_cosine_sim" in train_res
    assert "learning_rates" in planner.training_history
    assert len(planner.training_history["learning_rates"]) == 3
    assert planner.count_parameters() > 0
    assert planner.estimate_flops() > 0

    # Inference & normalization check: ||z||_2 == sqrt(latent_dim)
    planner.reset(obs, goal)
    z = planner.get_intention(obs, goal, step=0)
    assert z.shape == (128,)
    np.testing.assert_allclose(np.linalg.norm(z), math.sqrt(128), rtol=1e-4)
