"""Neural network architectures for waypoint and sequence intention translation."""

from typing import Any, Dict, Optional
import jax
import jax.numpy as jnp
import flax.linen as nn


class SingleWaypointNetwork(nn.Module):
    """Direct translator from immediate next waypoint B(w_1) and state s_t to low-level intention z_cmd."""

    obs_dim: int = 29
    latent_dim: int = 128
    hidden_dim: int = 256
    n_layers: int = 3

    @nn.compact
    def __call__(self, state: jnp.ndarray, waypoint_z: jnp.ndarray) -> jnp.ndarray:
        """Forward state and target waypoint through adaptive gating and residual layers."""
        h_s = nn.gelu(nn.LayerNorm(name="s_ln_0")(nn.Dense(self.hidden_dim, name="s_enc_0")(state)))
        h_s = nn.LayerNorm(name="s_ln_1")(nn.Dense(self.hidden_dim, name="s_enc_1")(h_s))

        h_w = nn.gelu(nn.LayerNorm(name="w_ln_0")(nn.Dense(self.hidden_dim, name="w_enc_0")(waypoint_z)))
        h_w = nn.Dense(self.hidden_dim, name="w_enc_1")(h_w)

        gate = nn.sigmoid(nn.Dense(self.hidden_dim, name="gate_fc")(jnp.concatenate([h_s, h_w], axis=-1)))
        h_fused = gate * h_w + (1.0 - gate) * h_s

        for i in range(self.n_layers - 1):
            h_res = nn.gelu(nn.LayerNorm(name=f"refine_ln_{i}")(nn.Dense(self.hidden_dim, name=f"refine_fc_{i}")(h_fused)))
            h_fused = h_fused + h_res

        out = nn.Dense(self.latent_dim, name="out_head")(h_fused)
        return out / (jnp.linalg.norm(out, axis=-1, keepdims=True) + 1e-8) * jnp.sqrt(self.latent_dim)


class GatedAttentionNetwork(nn.Module):
    """Gated state-goal modulation network for O(1) amortized planning."""

    latent_dim: int = 128
    hidden_dim: int = 256
    n_layers: int = 3

    @nn.compact
    def __call__(self, state: jnp.ndarray, goal_latent: jnp.ndarray) -> jnp.ndarray:
        """Modulate state features with goal gate to produce direct intention vector."""
        h_s = state
        for _ in range(self.n_layers):
            h_s = nn.gelu(nn.LayerNorm()(nn.Dense(self.hidden_dim)(h_s)))

        h_g = goal_latent
        for _ in range(self.n_layers - 1):
            h_g = nn.gelu(nn.LayerNorm()(nn.Dense(self.hidden_dim)(h_g)))
        gate_g = nn.sigmoid(nn.Dense(self.hidden_dim)(h_g))

        h_fused = h_s * gate_g
        out = nn.gelu(nn.LayerNorm()(nn.Dense(self.hidden_dim)(h_fused)))
        out = nn.Dense(self.latent_dim)(out)
        return out / (jnp.linalg.norm(out, axis=-1, keepdims=True) + 1e-8) * jnp.sqrt(self.latent_dim)


class DenseECANetwork(nn.Module):
    """Dense feature concatenation network with 1D efficient channel attention."""

    latent_dim: int = 128
    hidden_dim: int = 256
    n_layers: int = 3

    @nn.compact
    def __call__(self, state: jnp.ndarray, goal_latent: jnp.ndarray) -> jnp.ndarray:
        """Process concatenated state-goal representation through densely connected ECA blocks."""
        x = jnp.concatenate([state, goal_latent], axis=-1)
        h0 = nn.gelu(nn.LayerNorm()(nn.Dense(self.hidden_dim)(x)))

        features = [h0]
        for _ in range(self.n_layers - 1):
            cat_feat = jnp.concatenate(features, axis=-1)
            h_next = nn.gelu(nn.LayerNorm()(nn.Dense(self.hidden_dim)(cat_feat)))

            w_eca = self.param(f"eca_w_{len(features)}", nn.initializers.ones, (3,))
            h_pad = jnp.pad(h_next, ((0, 0), (1, 1)), mode="edge")
            att = nn.sigmoid(w_eca[0] * h_pad[:, :-2] + w_eca[1] * h_pad[:, 1:-1] + w_eca[2] * h_pad[:, 2:])
            features.append(h_next * att)

        all_cat = jnp.concatenate(features, axis=-1)
        out = nn.Dense(self.latent_dim)(all_cat)
        return out / (jnp.linalg.norm(out, axis=-1, keepdims=True) + 1e-8) * jnp.sqrt(self.latent_dim)


def build_direct_translator(
    model_type: str = "gated_attn",
    latent_dim: int = 128,
    hidden_dim: int = 256,
    n_layers: int = 3,
) -> nn.Module:
    """Construct direct intention translator network definition."""
    if model_type in ("gated_attn", "gated", "cross_gated"):
        return GatedAttentionNetwork(latent_dim=latent_dim, hidden_dim=hidden_dim, n_layers=n_layers)
    elif model_type in ("dense_eca", "dense"):
        return DenseECANetwork(latent_dim=latent_dim, hidden_dim=hidden_dim, n_layers=n_layers)
    return GatedAttentionNetwork(latent_dim=latent_dim, hidden_dim=hidden_dim, n_layers=n_layers)


class PositionalEncoding(nn.Module):
    """Learnable positional embedding table for trajectory waypoint sequences."""

    dim: int
    max_len: int = 32

    @nn.compact
    def __call__(self, seq_len: int) -> jnp.ndarray:
        """Retrieve slice of positional embeddings up to seq_len."""
        pe = self.param("pe", nn.initializers.normal(stddev=0.02), (self.max_len, self.dim))
        return pe[:seq_len]


def compute_curvatures_helper(waypoint_seq: jnp.ndarray, curvatures: Optional[jnp.ndarray]) -> jnp.ndarray:
    """Ensure curvature features are 3D array or calculate cosine turning angles."""
    if curvatures is not None:
        return curvatures[:, :, None] if curvatures.ndim == 2 else curvatures
    batch_size, seq_len_k, _ = waypoint_seq.shape
    if seq_len_k <= 2:
        return jnp.ones((batch_size, seq_len_k, 1), dtype=waypoint_seq.dtype)
    v = waypoint_seq[:, 1:] - waypoint_seq[:, :-1]
    v_norm = jnp.linalg.norm(v, axis=-1, keepdims=True) + 1e-6
    v_u = v / v_norm
    cos_th = jnp.sum(v_u[:, :-1] * v_u[:, 1:], axis=-1, keepdims=True)
    pad = jnp.ones((batch_size, 1, 1), dtype=waypoint_seq.dtype)
    return jnp.concatenate([pad, cos_th, pad], axis=1)


class SequenceAttentionNetwork(nn.Module):
    """Sequence-aware waypoint attention transformer with ALiBi position decay and curvature tokens."""

    obs_dim: int = 29
    latent_dim: int = 128
    hidden_dim: int = 384
    num_heads: int = 6
    max_seq_len: int = 16
    n_layers: int = 3
    dropout_rate: float = 0.2
    alibi_slope: float = 0.4

    def _apply_alibi_cross_attn(
        self,
        s_q: jnp.ndarray,
        h_w: jnp.ndarray,
        seq_mask: Optional[jnp.ndarray],
        deterministic: bool,
    ):
        """Cross-attention from state query to trajectory tokens with ALiBi distance decay."""
        batch_size, seq_len_k, _ = h_w.shape
        head_dim = self.hidden_dim // self.num_heads
        q = nn.Dense(self.hidden_dim, name="cross_q")(s_q).reshape((batch_size, 1, self.num_heads, head_dim))
        k = nn.Dense(self.hidden_dim, name="cross_k")(h_w).reshape((batch_size, seq_len_k, self.num_heads, head_dim))
        v = nn.Dense(self.hidden_dim, name="cross_v")(h_w).reshape((batch_size, seq_len_k, self.num_heads, head_dim))

        logits = jnp.einsum("bqhd,bkhd->bhqk", q, k) / jnp.sqrt(head_dim)
        decay = jnp.arange(seq_len_k)[None, :].astype(jnp.float32)
        slopes = jnp.array([2.0 ** (-4.0 * h / max(self.num_heads - 1, 1)) for h in range(self.num_heads)])
        logits = logits + (-self.alibi_slope * slopes[None, :, None, None] * decay[:, None, None, :])
        if seq_mask is not None:
            logits = jnp.where(seq_mask[:, None, None, :], logits, -1e9)

        attn_probs = jax.nn.softmax(logits, axis=-1)
        weights = nn.Dropout(rate=self.dropout_rate)(attn_probs, deterministic=deterministic)
        cross_out = jnp.einsum("bhqk,bkhd->bqhd", weights, v).reshape((batch_size, 1, self.hidden_dim)).squeeze(1)
        mean_attn = jnp.mean(attn_probs, axis=1)[:, 0, :]
        return nn.Dense(self.hidden_dim, name="cross_out_proj")(cross_out), mean_attn

    @nn.compact
    def __call__(
        self,
        state: jnp.ndarray,
        waypoint_seq: jnp.ndarray,
        seq_mask: Optional[jnp.ndarray] = None,
        curvatures: Optional[jnp.ndarray] = None,
        deterministic: bool = True,
        return_attention: bool = False,
    ):
        """Transform waypoint sequence and current state into low-level intention vector."""
        batch_size, seq_len_k, _ = waypoint_seq.shape
        curv_3d = compute_curvatures_helper(waypoint_seq, curvatures)
        c_emb = nn.Dense(self.hidden_dim, name="curv_proj")(curv_3d)

        w_emb = nn.Dense(self.hidden_dim, name="w_proj")(waypoint_seq) + PositionalEncoding(dim=self.hidden_dim, max_len=self.max_seq_len)(seq_len_k)[None, :seq_len_k, :] + c_emb
        w_emb = nn.Dropout(rate=self.dropout_rate)(w_emb, deterministic=deterministic)

        sa_mask = seq_mask[:, None, None, :] if seq_mask is not None else None
        h_w = w_emb
        for l in range(self.n_layers):
            attn = nn.MultiHeadDotProductAttention(num_heads=self.num_heads, qkv_features=self.hidden_dim, dropout_rate=self.dropout_rate, deterministic=deterministic, name=f"self_attn_{l}")
            h_w = nn.LayerNorm(name=f"sa_ln_{l}")(h_w + attn(h_w, h_w, mask=sa_mask))
            h_mlp = nn.Dense(self.hidden_dim, name=f"sa_mlp_1_{l}")(nn.Dropout(rate=self.dropout_rate)(nn.gelu(nn.Dense(self.hidden_dim, name=f"sa_mlp_0_{l}")(h_w)), deterministic=deterministic))
            h_w = nn.LayerNorm(name=f"sa_ln_mlp_{l}")(h_w + h_mlp)

        s_q = nn.Dense(self.hidden_dim, name="s_query_1")(nn.gelu(nn.LayerNorm(name="s_q_ln")(nn.Dense(self.hidden_dim, name="s_query_0")(state))))[:, None, :]
        cross_out, mean_attn = self._apply_alibi_cross_attn(s_q, h_w, seq_mask, deterministic)

        s_rep = s_q.squeeze(1)
        gate = nn.sigmoid(nn.Dense(self.hidden_dim, name="fusion_gate")(jnp.concatenate([s_rep, cross_out], axis=-1)))
        fused = gate * cross_out + (1.0 - gate) * s_rep

        out = nn.Dense(self.latent_dim, name="out_head")(nn.gelu(nn.LayerNorm(name="head_ln")(nn.Dense(self.hidden_dim, name="head_fc")(fused))))
        normed = out / (jnp.linalg.norm(out, axis=-1, keepdims=True) + 1e-8) * jnp.sqrt(self.latent_dim)
        if return_attention:
            return normed, mean_attn
        return normed
