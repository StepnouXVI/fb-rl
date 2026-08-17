"""
Advanced Neural Architectures for Distilled Shortest-Path Topological Graph Planning.
Includes:
- ECALayer1D: 1D Efficient Channel Attention without dimensionality reduction.
- ResNetECASubgoalNetwork: Multi-block Deep ResNet with ECA-1D attention, LayerNorm, and GELU.
- GatedCrossAttentionNetwork: Cross-attention (state queries, goal keys/values) with SwiGLU feed-forward.
- DenseECANetwork: Densely Connected Network with ECA-1D channel modulation.
- AdvancedDistilledPlanner: Unified Hierarchical Planner with Cosine Annealing AdamW training.
"""

from typing import Optional, Dict, Any, List, Tuple, Union
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from fb_core.base_planner import BaseHierarchicalPlanner
from fb_core.fb_model_wrapper import FBModelWrapper
from fb_core.dataset_sampler import DatasetSampler
from fb_core.math_utils import normalize_latent
from planners.buffer_graph import BufferGraphPlanner


# ==============================================================================
# 1. Efficient Channel Attention (ECA-1D)
# ==============================================================================

def get_adaptive_kernel_size(channels: int, gamma: float = 2.0, b: float = 1.0) -> int:
    """
    Compute adaptive kernel size k for ECA-1D:
    k = | log2(C)/gamma + b/gamma |_odd
    Ensures k >= 3 and k is odd.
    """
    t = int(abs(math.log2(max(channels, 2)) / gamma + b / gamma))
    k = t if t % 2 == 1 else t + 1
    return max(k, 3)


class ECALayer1D(nn.Module):
    """
    1D Efficient Channel Attention (ECA-1D) Layer.
    Captures local cross-channel interaction without dimensionality reduction.
    Applies a 1D convolution of adaptive or fixed odd kernel size k across channels,
    followed by sigmoid modulation: omega = sigmoid(Conv1D_k(y)), x_tilde = x * omega.
    """

    def __init__(
        self,
        channels: int,
        k_size: Optional[int] = None,
        gamma: float = 2.0,
        b: float = 1.0,
    ):
        super().__init__()
        self.channels = channels
        if k_size is None:
            self.k_size = get_adaptive_kernel_size(channels, gamma=gamma, b=b)
        else:
            # Ensure odd
            self.k_size = k_size if k_size % 2 == 1 else k_size + 1
            self.k_size = max(self.k_size, 3)

        self.conv = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=self.k_size,
            padding=(self.k_size - 1) // 2,
            bias=False,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for ECA-1D.
        x: tensor of shape (B, C) or (B, L, C)
        returns: modulated tensor of same shape
        """
        if x.dim() == 2:
            # Shape (B, C) -> (B, 1, C)
            y = x.unsqueeze(1)
            omega = self.conv(y)  # (B, 1, C)
            omega = self.sigmoid(omega).squeeze(1)  # (B, C)
            return x * omega
        elif x.dim() == 3:
            # Shape (B, L, C)
            B, L, C = x.shape
            y = x.view(B * L, 1, C)
            omega = self.conv(y)
            omega = self.sigmoid(omega).view(B, L, C)
            return x * omega
        else:
            raise ValueError(f"ECALayer1D expects 2D (B, C) or 3D (B, L, C) input, got shape {x.shape}")


# ==============================================================================
# 2. ResNet with ECA-1D (ResNetECASubgoalNetwork)
# ==============================================================================

class ResNetECABlock(nn.Module):
    """
    Residual Block with LayerNorm, GELU, Linear projection, and ECA-1D channel attention.
    h = h + ECA(Linear(GELU(LayerNorm(Linear(GELU(LayerNorm(h)))))))
    """

    def __init__(
        self,
        hidden_dim: int,
        dropout: float = 0.0,
        k_size: Optional[int] = None,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.act1 = nn.GELU()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.ln2 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.act2 = nn.GELU()

        self.eca = ECALayer1D(channels=hidden_dim, k_size=k_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.ln1(x)
        out = self.act1(self.fc1(out))
        out = self.drop(out)
        out = self.ln2(out)
        out = self.act2(self.fc2(out))
        out = self.eca(out)
        return residual + out


class ResNetECASubgoalNetwork(nn.Module):
    """
    Deep Residual Subgoal Intention Network with 1D Efficient Channel Attention (ResNet-ECA).
    Maps (observation s, goal latent z_g) -> optimal intermediate waypoint latent z_w*.
    """

    def __init__(
        self,
        obs_dim: int = 29,
        latent_dim: int = 128,
        hidden_dim: int = 256,
        num_blocks: int = 4,
        dropout: float = 0.0,
        k_size: Optional[int] = None,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks

        in_dim = obs_dim + latent_dim
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        self.blocks = nn.ModuleList([
            ResNetECABlock(hidden_dim=hidden_dim, dropout=dropout, k_size=k_size)
            for _ in range(num_blocks)
        ])

        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, latent_dim),
        )

    def forward(self, obs: torch.Tensor, z_goal: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, z_goal], dim=-1)
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h)
        return self.head(h)


# ==============================================================================
# 3. Gated Cross-Attention Subgoal Controller (SwiGLU)
# ==============================================================================

class SwiGLU(nn.Module):
    """
    Swish Gated Linear Unit (SwiGLU) Feed-Forward Network:
    SwiGLU(x) = (Linear_gate(x) * SiLU(Linear_val(x))) * W_out
    """

    def __init__(self, in_features: int, hidden_features: Optional[int] = None, out_features: Optional[int] = None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or int(in_features * 8 / 3)  # Standard 8/3 expansion for SwiGLU
        self.w_gate = nn.Linear(in_features, hidden_features, bias=False)
        self.w_val = nn.Linear(in_features, hidden_features, bias=False)
        self.w_out = nn.Linear(hidden_features, out_features, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_out(F.silu(self.w_gate(x)) * self.w_val(x))


class GatedCrossAttentionBlock(nn.Module):
    """
    Cross-Attention Block where Queries come from State representation and
    Keys/Values come from Goal representation, followed by SwiGLU FFN.
    """

    def __init__(
        self,
        d_model: int = 256,
        num_heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.ln_q = nn.LayerNorm(d_model)
        self.ln_kv = nn.LayerNorm(d_model)

        self.swiglu = SwiGLU(in_features=d_model, hidden_features=int(d_model * 2), out_features=d_model)
        self.ln_ffn = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Cache for attention maps
        self.last_attn_weights: Optional[torch.Tensor] = None

    def forward(
        self,
        state_tokens: torch.Tensor,
        goal_tokens: torch.Tensor,
        return_attn: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        state_tokens: (B, N_q, d_model)
        goal_tokens: (B, N_k, d_model)
        """
        B, N_q, _ = state_tokens.shape
        _, N_k, _ = goal_tokens.shape

        q_norm = self.ln_q(state_tokens)
        kv_norm = self.ln_kv(goal_tokens)

        # Multi-head projections: (B, num_heads, N, head_dim)
        q = self.q_proj(q_norm).view(B, N_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(kv_norm).view(B, N_k, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(kv_norm).view(B, N_k, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, num_heads, N_q, N_k)
        attn_weights = F.softmax(scores, dim=-1)
        self.last_attn_weights = attn_weights.detach()

        attn_out = torch.matmul(attn_weights, v)  # (B, num_heads, N_q, head_dim)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, N_q, self.d_model)
        attn_out = self.dropout(self.out_proj(attn_out))

        # Residual 1
        h = state_tokens + attn_out

        # Residual 2: SwiGLU FFN
        h = h + self.dropout(self.swiglu(self.ln_ffn(h)))

        if return_attn:
            return h, attn_weights
        return h


class GatedCrossAttentionNetwork(nn.Module):
    """
    Subgoal Network with Gated Cross-Attention and SwiGLU layers.
    Embeds state into query tokens and goal into key/value tokens,
    then applies cross-attention and gated non-linearities.
    """

    def __init__(
        self,
        obs_dim: int = 29,
        latent_dim: int = 128,
        d_model: int = 256,
        num_heads: int = 4,
        num_layers: int = 3,
        num_query_tokens: int = 4,
        num_key_tokens: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.latent_dim = latent_dim
        self.d_model = d_model
        self.num_query_tokens = num_query_tokens
        self.num_key_tokens = num_key_tokens

        # State Tokenizer: projects obs to num_query_tokens x d_model
        self.state_proj = nn.Sequential(
            nn.Linear(obs_dim, d_model * num_query_tokens),
            nn.GELU(),
        )

        # Goal Tokenizer: projects z_goal to num_key_tokens x d_model
        self.goal_proj = nn.Sequential(
            nn.Linear(latent_dim, d_model * num_key_tokens),
            nn.GELU(),
        )

        # Cross-Attention Layers
        self.layers = nn.ModuleList([
            GatedCrossAttentionBlock(d_model=d_model, num_heads=num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])

        # ECA Channel modulation over final token representation
        self.eca = ECALayer1D(channels=d_model)

        # Output head: maps pooled / flattened tokens to latent_dim
        self.head = nn.Sequential(
            nn.LayerNorm(d_model * num_query_tokens),
            nn.Linear(d_model * num_query_tokens, d_model),
            nn.GELU(),
            nn.Linear(d_model, latent_dim),
        )

    def forward(self, obs: torch.Tensor, z_goal: torch.Tensor) -> torch.Tensor:
        B = obs.shape[0]
        q_tokens = self.state_proj(obs).view(B, self.num_query_tokens, self.d_model)
        kv_tokens = self.goal_proj(z_goal).view(B, self.num_key_tokens, self.d_model)

        h = q_tokens
        for layer in self.layers:
            h = layer(h, kv_tokens)

        h = self.eca(h)
        h_flat = h.view(B, -1)
        return self.head(h_flat)

    def get_attention_weights(self, obs: torch.Tensor, z_goal: torch.Tensor) -> List[torch.Tensor]:
        """Extract multi-head cross-attention weight tensors for inspection."""
        B = obs.shape[0]
        q_tokens = self.state_proj(obs).view(B, self.num_query_tokens, self.d_model)
        kv_tokens = self.goal_proj(z_goal).view(B, self.num_key_tokens, self.d_model)

        weights = []
        h = q_tokens
        for layer in self.layers:
            h, attn = layer(h, kv_tokens, return_attn=True)
            weights.append(attn)
        return weights


# ==============================================================================
# 4. Densely Connected Gated ResNet (Dense-ECA)
# ==============================================================================

class DenseECALayer(nn.Module):
    """
    A single dense layer with LayerNorm -> Linear -> GELU -> ECA-1D -> Growth Channel output.
    """

    def __init__(self, in_features: int, growth_rate: int, dropout: float = 0.0):
        super().__init__()
        self.ln = nn.LayerNorm(in_features)
        self.fc = nn.Linear(in_features, growth_rate)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.eca = ECALayer1D(channels=growth_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.ln(x)
        out = self.act(self.fc(out))
        out = self.drop(out)
        out = self.eca(out)
        return out


class DenseECANetwork(nn.Module):
    """
    Densely Connected Subgoal Network with Efficient Channel Attention (Dense-ECA).
    Each layer receives feature concatenation of all preceding layers:
    x_l = ECA(Layer( [x_0, x_1, ..., x_{l-1}] )).
    """

    def __init__(
        self,
        obs_dim: int = 29,
        latent_dim: int = 128,
        init_dim: int = 128,
        growth_rate: int = 64,
        num_layers: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers

        in_dim = obs_dim + latent_dim
        self.init_proj = nn.Sequential(
            nn.Linear(in_dim, init_dim),
            nn.LayerNorm(init_dim),
            nn.GELU(),
        )

        self.dense_layers = nn.ModuleList()
        current_dim = init_dim
        for _ in range(num_layers):
            layer = DenseECALayer(in_features=current_dim, growth_rate=growth_rate, dropout=dropout)
            self.dense_layers.append(layer)
            current_dim += growth_rate

        self.head = nn.Sequential(
            nn.LayerNorm(current_dim),
            nn.Linear(current_dim, current_dim // 2),
            nn.GELU(),
            nn.Linear(current_dim // 2, latent_dim),
        )

    def forward(self, obs: torch.Tensor, z_goal: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, z_goal], dim=-1)
        features = [self.init_proj(x)]
        for layer in self.dense_layers:
            inp = torch.cat(features, dim=-1)
            out = layer(inp)
            features.append(out)
        total_cat = torch.cat(features, dim=-1)
        return self.head(total_cat)


# ==============================================================================
# 5. Unified Advanced Distilled Hierarchical Planner
# ==============================================================================

class AdvancedDistilledPlanner(BaseHierarchicalPlanner):
    """
    Advanced Distilled Planner supporting ResNet-ECA, Gated Cross-Attention, Dense-ECA,
    and Standard MLP architectures.
    Includes AdamW optimizer, Cosine Annealing learning rate schedule, composite loss,
    and validation metrics tracking.
    """

    def __init__(
        self,
        fb_model: FBModelWrapper,
        dataset_sampler: Optional[DatasetSampler] = None,
        architecture_type: str = "resnet_eca",
        name: Optional[str] = None,
        obs_dim: int = 29,
        latent_dim: int = 128,
        hidden_dim: int = 256,
        num_blocks: int = 4,
        num_heads: int = 4,
        device: str = "cpu",
        config: Optional[Dict[str, Any]] = None,
    ):
        default_name = f"Distilled-{architecture_type.upper()}"
        super().__init__(name=name or default_name, config=config)
        self.fb_model = fb_model
        self.dataset_sampler = dataset_sampler
        self.architecture_type = architecture_type.lower()
        self.obs_dim = obs_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        # Device setup
        if device == "mps" and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif device == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # Instantiate selected architecture
        if self.architecture_type in ["resnet_eca", "resnet-eca"]:
            self.model: nn.Module = ResNetECASubgoalNetwork(
                obs_dim=obs_dim,
                latent_dim=latent_dim,
                hidden_dim=hidden_dim,
                num_blocks=num_blocks,
            )
        elif self.architecture_type in ["gated_cross_attn", "cross_attention", "gated_cross_attention"]:
            self.model = GatedCrossAttentionNetwork(
                obs_dim=obs_dim,
                latent_dim=latent_dim,
                d_model=hidden_dim,
                num_heads=num_heads,
                num_layers=num_blocks,
            )
        elif self.architecture_type in ["dense_eca", "dense-eca"]:
            self.model = DenseECANetwork(
                obs_dim=obs_dim,
                latent_dim=latent_dim,
                init_dim=hidden_dim // 2,
                growth_rate=64,
                num_layers=num_blocks,
            )
        elif self.architecture_type in ["standard_mlp", "mlp"]:
            from planners.distilled_mlp import SubgoalMLPNetwork
            self.model = SubgoalMLPNetwork(
                obs_dim=obs_dim,
                latent_dim=latent_dim,
                hidden_dims=[hidden_dim, hidden_dim],
                use_residual=False,
            )
        else:
            raise ValueError(f"Unknown architecture_type: {architecture_type}")

        self.model.to(self.device)
        self.is_trained = False
        self.training_history: Dict[str, List[float]] = {}

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def estimate_flops(self) -> int:
        """Analytical estimation of forward FLOPs for single sample."""
        total_flops = 0
        for m in self.model.modules():
            if isinstance(m, nn.Linear):
                # 2 * in_features * out_features (multiply-add)
                total_flops += 2 * m.in_features * m.out_features
            elif isinstance(m, nn.Conv1d):
                total_flops += 2 * m.in_channels * m.out_channels * m.kernel_size[0] * m.padding[0]
            elif isinstance(m, nn.LayerNorm):
                total_flops += 2 * m.normalized_shape[0]
        return total_flops

    def train_distillation(
        self,
        graph_planner: Optional[BufferGraphPlanner] = None,
        dataset: Optional[Dict[str, np.ndarray]] = None,
        n_pairs: int = 3000,
        val_ratio: float = 0.2,
        epochs: int = 30,
        batch_size: int = 64,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
    ) -> Dict[str, Any]:
        """
        Train the model using Dijkstra shortest-path transition pairs with
        Cosine Annealing learning rate schedule, AdamW optimizer, and composite loss.
        """
        if dataset is not None:
            inputs_obs = dataset["obs"]
            inputs_zg = dataset["z_goal"]
            targets_zw = dataset["z_waypoint"]
            n_total = len(inputs_obs)
        else:
            if graph_planner is None:
                assert self.dataset_sampler is not None, "dataset_sampler must be provided if graph_planner is None"
                graph_planner = BufferGraphPlanner(
                    fb_model=self.fb_model,
                    dataset_sampler=self.dataset_sampler,
                    n_landmarks=200,
                )

            print(f"[{self.name}] Sampling {n_pairs} Dijkstra shortest-path transition pairs...")
            states = self.dataset_sampler.sample_candidates(n_pairs * 2)
            start_states = states[:n_pairs]
            goal_states = states[n_pairs:]

            inputs_obs_list, inputs_zg_list, targets_zw_list = [], [], []
            for i in range(n_pairs):
                s = start_states[i]
                g = goal_states[i]
                z_g = self.fb_model.encode_backward(g)

                waypoints = graph_planner._find_shortest_path(s, g)
                target_waypoint = waypoints[1] if len(waypoints) > 1 else g
                z_w = self.fb_model.encode_backward(target_waypoint)

                inputs_obs_list.append(s)
                inputs_zg_list.append(z_g)
                targets_zw_list.append(z_w)

            inputs_obs = np.asarray(inputs_obs_list, dtype=np.float32)
            inputs_zg = np.asarray(inputs_zg_list, dtype=np.float32)
            targets_zw = np.asarray(targets_zw_list, dtype=np.float32)
            n_total = len(inputs_obs)

        # Train / Val Split
        n_val = max(int(n_total * val_ratio), 1)
        n_train = n_total - n_val

        obs_train = torch.tensor(inputs_obs[:n_train], dtype=torch.float32, device=self.device)
        zg_train = torch.tensor(inputs_zg[:n_train], dtype=torch.float32, device=self.device)
        zw_train = torch.tensor(targets_zw[:n_train], dtype=torch.float32, device=self.device)

        obs_val = torch.tensor(inputs_obs[n_train:], dtype=torch.float32, device=self.device)
        zg_val = torch.tensor(inputs_zg[n_train:], dtype=torch.float32, device=self.device)
        zw_val = torch.tensor(targets_zw[n_train:], dtype=torch.float32, device=self.device)

        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
        cos_sim = nn.CosineSimilarity(dim=-1)

        train_losses, val_losses, val_cos_sims, val_mses, lrs = [], [], [], [], []

        for ep in range(epochs):
            self.model.train()
            perm = torch.randperm(n_train)
            ep_losses = []

            for j in range(0, n_train, batch_size):
                idx = perm[j:j+batch_size]
                b_obs = obs_train[idx]
                b_zg = zg_train[idx]
                b_zw = zw_train[idx]

                pred_z = self.model(b_obs, b_zg)
                mse_loss = F.mse_loss(pred_z, b_zw)
                cos_loss = torch.mean(1.0 - cos_sim(pred_z, b_zw))
                loss = mse_loss + 0.5 * cos_loss

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                ep_losses.append(loss.item())

            scheduler.step()
            curr_lr = scheduler.get_last_lr()[0]
            lrs.append(curr_lr)

            # Validation evaluation
            self.model.eval()
            with torch.no_grad():
                pred_val = self.model(obs_val, zg_val)
                val_mse = F.mse_loss(pred_val, zw_val).item()
                val_cos = torch.mean(cos_sim(pred_val, zw_val)).item()
                val_loss = val_mse + 0.5 * (1.0 - val_cos)

            train_losses.append(float(np.mean(ep_losses)))
            val_losses.append(float(val_loss))
            val_cos_sims.append(float(val_cos))
            val_mses.append(float(val_mse))

        self.is_trained = True
        self.training_history = {
            "train_loss": train_losses,
            "val_loss": val_losses,
            "val_cosine_sim": val_cos_sims,
            "val_mse": val_mses,
            "learning_rates": lrs,
        }

        return {
            "final_loss": train_losses[-1] if train_losses else 0.0,
            "final_train_loss": train_losses[-1] if train_losses else 0.0,
            "final_val_loss": val_losses[-1] if val_losses else 0.0,
            "final_val_cosine_sim": val_cos_sims[-1] if val_cos_sims else 0.0,
            "final_val_mse": val_mses[-1] if val_mses else 0.0,
            "epochs": epochs,
            "n_train": n_train,
            "n_val": n_val,
            "param_count": self.count_parameters(),
        }

    def reset(self, initial_obs: np.ndarray, goal: np.ndarray) -> None:
        super().reset(initial_obs, goal)

    def get_intention(self, obs: np.ndarray, goal: np.ndarray, step: int = 0) -> np.ndarray:
        if not self.is_trained:
            self.train_distillation(n_pairs=300, epochs=15)

        z_g = self.fb_model.encode_backward(goal)

        self.model.eval()
        with torch.no_grad():
            obs_t = torch.tensor(obs[None, :], dtype=torch.float32, device=self.device)
            zg_t = torch.tensor(z_g[None, :], dtype=torch.float32, device=self.device)
            pred_z = self.model(obs_t, zg_t).cpu().numpy()[0]

        return normalize_latent(pred_z, latent_dim=self.latent_dim)
