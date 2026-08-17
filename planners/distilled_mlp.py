"""
DistilledMLPPlanner: Lightweight Neural Controller Distilled from Topological Graph Search.
Trains a fast 3-layer MLP on optimal path waypoints for instant O(1) step inference.
"""

from typing import Optional, Dict, Any
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from fb_core.base_planner import BaseHierarchicalPlanner
from fb_core.fb_model_wrapper import FBModelWrapper
from fb_core.dataset_sampler import DatasetSampler
from fb_core.math_utils import normalize_latent
from planners.buffer_graph import BufferGraphPlanner


class SubgoalMLPNetwork(nn.Module):
    """
    3-Layer Lightweight MLP: (obs_dim + latent_dim) -> 256 -> 256 -> latent_dim.
    # ponytail: Standard PyTorch Sequential MLP, zero unnecessary abstractions.
    """

    def __init__(self, obs_dim: int = 29, latent_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, obs: torch.Tensor, z_goal: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, z_goal], dim=-1)
        return self.net(x)


class DistilledMLPPlanner(BaseHierarchicalPlanner):
    """
    Branch 3: Distilled Latent Multi-Subgoal MLP Controller.
    """

    def __init__(
        self,
        fb_model: FBModelWrapper,
        dataset_sampler: DatasetSampler,
        name: str = "Distilled Latent MLP (Branch 3)",
        obs_dim: int = 29,
        latent_dim: int = 128,
        device: str = "cpu",
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(name=name, config=config)
        self.fb_model = fb_model
        self.dataset_sampler = dataset_sampler
        self.obs_dim = obs_dim
        self.latent_dim = latent_dim
        
        # Check MPS availability
        if device == "mps" and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif device == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.model = SubgoalMLPNetwork(obs_dim=obs_dim, latent_dim=latent_dim).to(self.device)
        self.is_trained = False

    def train_distillation(
        self,
        graph_planner: Optional[BufferGraphPlanner] = None,
        n_pairs: int = 500,
        epochs: int = 25,
        batch_size: int = 64,
        lr: float = 1e-3,
    ) -> Dict[str, float]:
        """
        Generate dataset of optimal path transitions and train the lightweight MLP.
        """
        if graph_planner is None:
            graph_planner = BufferGraphPlanner(
                fb_model=self.fb_model,
                dataset_sampler=self.dataset_sampler,
                n_landmarks=100,
            )

        print(f"[{self.name}] Generating {n_pairs} distillation path pairs from graph search...")
        states = self.dataset_sampler.sample_candidates(n_pairs * 2)
        start_states = states[:n_pairs]
        goal_states = states[n_pairs:]

        inputs_obs = []
        inputs_zg = []
        targets_zw = []

        for i in range(n_pairs):
            s = start_states[i]
            g = goal_states[i]
            z_g = self.fb_model.encode_backward(g)
            
            # Find shortest path waypoint
            waypoints = graph_planner._find_shortest_path(s, g)
            # Pick first intermediate waypoint or goal
            target_waypoint = waypoints[1] if len(waypoints) > 1 else g
            z_w = self.fb_model.encode_backward(target_waypoint)

            inputs_obs.append(s)
            inputs_zg.append(z_g)
            targets_zw.append(z_w)

        obs_t = torch.tensor(np.asarray(inputs_obs), dtype=torch.float32, device=self.device)
        zg_t = torch.tensor(np.asarray(inputs_zg), dtype=torch.float32, device=self.device)
        zw_t = torch.tensor(np.asarray(targets_zw), dtype=torch.float32, device=self.device)

        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        cos_sim = nn.CosineSimilarity(dim=-1)
        num_samples = len(obs_t)

        self.model.train()
        losses = []
        for ep in range(epochs):
            perm = torch.randperm(num_samples)
            ep_losses = []
            for j in range(0, num_samples, batch_size):
                idx = perm[j:j+batch_size]
                b_obs = obs_t[idx]
                b_zg = zg_t[idx]
                b_zw = zw_t[idx]

                pred_z = self.model(b_obs, b_zg)
                
                # Combined Loss: MSE + Cosine Distance
                mse_loss = nn.functional.mse_loss(pred_z, b_zw)
                cos_loss = torch.mean(1.0 - cos_sim(pred_z, b_zw))
                loss = mse_loss + 0.5 * cos_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                ep_losses.append(loss.item())

            avg_ep_loss = float(np.mean(ep_losses))
            losses.append(avg_ep_loss)

        self.model.eval()
        self.is_trained = True
        return {"final_loss": losses[-1] if losses else 0.0, "epochs": epochs}

    def reset(self, initial_obs: np.ndarray, goal: np.ndarray) -> None:
        super().reset(initial_obs, goal)

    def get_intention(self, obs: np.ndarray, goal: np.ndarray, step: int = 0) -> np.ndarray:
        if not self.is_trained:
            # Self-train if not yet trained
            self.train_distillation(n_pairs=200, epochs=15)

        z_g = self.fb_model.encode_backward(goal)
        
        with torch.no_grad():
            obs_t = torch.tensor(obs[None, :], dtype=torch.float32, device=self.device)
            zg_t = torch.tensor(z_g[None, :], dtype=torch.float32, device=self.device)
            pred_z = self.model(obs_t, zg_t).cpu().numpy()[0]

        return normalize_latent(pred_z, latent_dim=self.latent_dim)
