import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ponytail: 1D Efficient Channel Attention (ECA) module
class ECALayer(nn.Module):
    def __init__(self, k_size=3):
        super().__init__()
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)

    def forward(self, x):
        # x: (B, C) -> (B, 1, C) -> conv -> sigmoid -> (B, C)
        y = x.unsqueeze(1)
        y = self.conv(y)
        return x * torch.sigmoid(y.squeeze(1))


class StandardMLP(nn.Module):
    def __init__(self, in_dim=157, out_dim=128, hidden_dim=256, n_layers=3):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()]
        for _ in range(n_layers - 2):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()])
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        z = self.net(x)
        return z / (torch.norm(z, dim=-1, keepdim=True) + 1e-8) * np.sqrt(z.shape[-1])


class DenseECANetwork(nn.Module):
    def __init__(self, in_dim=157, out_dim=128, hidden_dim=256, n_layers=3, k_size=3):
        super().__init__()
        self.input_layer = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
        self.blocks = nn.ModuleList()
        current_dim = hidden_dim
        for _ in range(n_layers - 1):
            block = nn.Sequential(
                nn.Linear(current_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                ECALayer(k_size=k_size),
            )
            self.blocks.append(block)
            current_dim += hidden_dim
        self.out_head = nn.Linear(current_dim, out_dim)

    def forward(self, x):
        features = [self.input_layer(x)]
        for block in self.blocks:
            cat_feat = torch.cat(features, dim=-1)
            new_feat = block(cat_feat)
            features.append(new_feat)
        out = self.out_head(torch.cat(features, dim=-1))
        return out / (torch.norm(out, dim=-1, keepdim=True) + 1e-8) * np.sqrt(out.shape[-1])


class GatedCrossAttentionNetwork(nn.Module):
    def __init__(self, obs_dim=29, latent_dim=128, hidden_dim=256, num_heads=4):
        super().__init__()
        self.obs_encoder = nn.Sequential(nn.Linear(obs_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
        self.goal_encoder = nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
        self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.gate = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.Sigmoid())
        self.fusion_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            ECALayer(k_size=3),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, x):
        # x is concatenation of [obs, goal_latent]
        obs = x[:, :29]
        goal = x[:, 29:]
        h_obs = self.obs_encoder(obs).unsqueeze(1)    # (B, 1, H) - Query
        h_goal = self.goal_encoder(goal).unsqueeze(1) # (B, 1, H) - Key/Value

        attn_out, _ = self.cross_attn(h_obs, h_goal, h_goal) # (B, 1, H)
        attn_out = attn_out.squeeze(1)
        h_obs_sq = h_obs.squeeze(1)

        g = self.gate(torch.cat([h_obs_sq, attn_out], dim=-1))
        fused = g * attn_out + (1.0 - g) * h_obs_sq
        out = self.fusion_mlp(fused)
        return out / (torch.norm(out, dim=-1, keepdim=True) + 1e-8) * np.sqrt(out.shape[-1])


class ResNetECANetwork(nn.Module):
    def __init__(self, in_dim=157, out_dim=128, hidden_dim=256, n_blocks=3, k_size=3):
        super().__init__()
        self.in_proj = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                ECALayer(k_size=k_size),
            )
            for _ in range(n_blocks)
        ])
        self.out_head = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        h = self.in_proj(x)
        for block in self.blocks:
            h = h + block(h)
        out = self.out_head(h)
        return out / (torch.norm(out, dim=-1, keepdim=True) + 1e-8) * np.sqrt(out.shape[-1])


class FiLMBlock(nn.Module):
    def __init__(self, hidden_dim=256, k_size=3):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.eca = ECALayer(k_size=k_size)

    def forward(self, h, gamma, beta):
        h_mod = (1.0 + gamma) * self.norm1(h) + beta
        res = F.gelu(self.fc1(h_mod))
        res = self.norm2(res)
        res = self.fc2(res)
        res = self.eca(res)
        return h + res


class FiLMResNetNetwork(nn.Module):
    def __init__(self, obs_dim=29, latent_dim=128, hidden_dim=256, n_blocks=3, k_size=3):
        super().__init__()
        self.obs_dim = obs_dim
        self.obs_proj = nn.Sequential(nn.Linear(obs_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
        self.goal_encoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.film_generators = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim * 2) for _ in range(n_blocks)
        ])
        self.blocks = nn.ModuleList([
            FiLMBlock(hidden_dim=hidden_dim, k_size=k_size) for _ in range(n_blocks)
        ])
        self.out_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, latent_dim)
        )

    def forward(self, x):
        obs = x[:, :self.obs_dim]
        goal = x[:, self.obs_dim:]
        h = self.obs_proj(obs)
        g_feat = self.goal_encoder(goal)
        for gen, block in zip(self.film_generators, self.blocks):
            params = gen(g_feat)
            gamma, beta = torch.chunk(params, 2, dim=-1)
            h = block(h, gamma, beta)
        out = self.out_head(h)
        return out / (torch.norm(out, dim=-1, keepdim=True) + 1e-8) * np.sqrt(out.shape[-1])


class TransformerEncoderNetwork(nn.Module):
    def __init__(self, obs_dim=29, latent_dim=128, hidden_dim=256, num_heads=4, n_layers=3):
        super().__init__()
        self.obs_dim = obs_dim
        self.obs_proj = nn.Linear(obs_dim, hidden_dim)
        self.goal_proj = nn.Linear(latent_dim, hidden_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers, enable_nested_tensor=False)
        self.out_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, latent_dim)
        )

    def forward(self, x):
        obs = x[:, :self.obs_dim]
        goal = x[:, self.obs_dim:]
        b = x.shape[0]
        tok_obs = self.obs_proj(obs).unsqueeze(1)
        tok_goal = self.goal_proj(goal).unsqueeze(1)
        cls_tok = self.cls_token.expand(b, -1, -1)
        tokens = torch.cat([cls_tok, tok_obs, tok_goal], dim=1)
        h = self.transformer(tokens)
        out = self.out_head(h[:, 0])
        return out / (torch.norm(out, dim=-1, keepdim=True) + 1e-8) * np.sqrt(out.shape[-1])


def build_student_model(model_type="mlp", obs_dim=29, latent_dim=128, hidden_dim=256, n_layers=3, num_heads=4):
    in_dim = obs_dim + latent_dim
    model_type = model_type.lower()
    if model_type in ["mlp", "standard_mlp"]:
        return StandardMLP(in_dim, latent_dim, hidden_dim=hidden_dim, n_layers=n_layers)
    elif model_type in ["dense_eca", "dense"]:
        return DenseECANetwork(in_dim, latent_dim, hidden_dim=hidden_dim, n_layers=n_layers)
    elif model_type in ["gated_attn", "cross_attn", "dense_gated_attn", "gated_cross_attn"]:
        return GatedCrossAttentionNetwork(obs_dim=obs_dim, latent_dim=latent_dim, hidden_dim=hidden_dim, num_heads=num_heads)
    elif model_type in ["resnet", "resnet_eca"]:
        return ResNetECANetwork(in_dim, latent_dim, hidden_dim=hidden_dim, n_blocks=n_layers)
    elif model_type in ["film", "film_resnet", "film_eca"]:
        return FiLMResNetNetwork(obs_dim=obs_dim, latent_dim=latent_dim, hidden_dim=hidden_dim, n_blocks=n_layers)
    elif model_type in ["transformer", "transformer_encoder"]:
        return TransformerEncoderNetwork(obs_dim=obs_dim, latent_dim=latent_dim, hidden_dim=hidden_dim, num_heads=num_heads, n_layers=n_layers)
    else:
        raise ValueError(f"Unknown student model_type: {model_type}")
