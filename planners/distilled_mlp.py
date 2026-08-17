"""
DistilledMLPPlanner: Lightweight Neural Controller Distilled from Topological Graph Search.
Supports configurable architectures (depth, width, residual connections) and train/val split monitoring.
"""

from typing import Optional, Dict, Any, List
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
    Configurable MLP Network for Subgoal Intention Prediction.
    Supports arbitrary depth, width, LayerNorm, GELU activations, and optional Residual skips.
    # ponytail: Clean modular PyTorch Module with residual skip option.
    """

    def __init__(
        self,
        obs_dim: int = 29,
        latent_dim: int = 128,
        hidden_dims: Optional[List[int]] = None,
        use_residual: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256]
            
        self.use_residual = use_residual
        self.input_dim = obs_dim + latent_dim
        self.latent_dim = latent_dim

        # Build layers
        self.blocks = nn.ModuleList()
        in_d = self.input_dim
        for h_d in hidden_dims:
            block = nn.Sequential(
                nn.Linear(in_d, h_d),
                nn.LayerNorm(h_d),
                nn.GELU(),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            )
            self.blocks.append(block)
            in_d = h_d

        self.head = nn.Linear(in_d, latent_dim)

    def forward(self, obs: torch.Tensor, z_goal: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, z_goal], dim=-1)
        h = x
        for i, block in enumerate(self.blocks):
            h_next = block(h)
            if self.use_residual and h_next.shape == h.shape:
                h = h + h_next
            else:
                h = h_next
        return self.head(h)


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
        hidden_dims: Optional[List[int]] = None,
        use_residual: bool = False,
        device: str = "cpu",
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(name=name, config=config)
        self.fb_model = fb_model
        self.dataset_sampler = dataset_sampler
        self.obs_dim = obs_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims or [256, 256]
        self.use_residual = use_residual
        
        # Check MPS availability
        if device == "mps" and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif device == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.model = SubgoalMLPNetwork(
            obs_dim=obs_dim,
            latent_dim=latent_dim,
            hidden_dims=self.hidden_dims,
            use_residual=use_residual,
        ).to(self.device)
        self.is_trained = False
        self.training_history: Dict[str, List[float]] = {}

    def train_distillation(
        self,
        graph_planner: Optional[BufferGraphPlanner] = None,
        n_pairs: int = 1000,
        val_ratio: float = 0.2,
        epochs: int = 25,
        batch_size: int = 64,
        lr: float = 1e-3,
    ) -> Dict[str, Any]:
        """
        Generate dataset of optimal path transitions and train the lightweight MLP with train/val split.
        """
        if graph_planner is None:
            graph_planner = BufferGraphPlanner(
                fb_model=self.fb_model,
                dataset_sampler=self.dataset_sampler,
                n_landmarks=150,
            )

        print(f"[{self.name}] Generating {n_pairs} distillation path pairs (Train: {int(n_pairs*(1-val_ratio))}, Val: {int(n_pairs*val_ratio)})...")
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

        # Train / Val Split
        n_val = int(n_pairs * val_ratio)
        n_train = n_pairs - n_val

        obs_train = torch.tensor(np.asarray(inputs_obs[:n_train]), dtype=torch.float32, device=self.device)
        zg_train = torch.tensor(np.asarray(inputs_zg[:n_train]), dtype=torch.float32, device=self.device)
        zw_train = torch.tensor(np.asarray(targets_zw[:n_train]), dtype=torch.float32, device=self.device)

        obs_val = torch.tensor(np.asarray(inputs_obs[n_train:]), dtype=torch.float32, device=self.device)
        zg_val = torch.tensor(np.asarray(inputs_zg[n_train:]), dtype=torch.float32, device=self.device)
        zw_val = torch.tensor(np.asarray(targets_zw[n_train:]), dtype=torch.float32, device=self.device)

        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        cos_sim = nn.CosineSimilarity(dim=-1)

        train_losses, val_losses, val_cos_sims, val_mses = [], [], [], []

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
                mse_loss = nn.functional.mse_loss(pred_z, b_zw)
                cos_loss = torch.mean(1.0 - cos_sim(pred_z, b_zw))
                loss = mse_loss + 0.5 * cos_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                ep_losses.append(loss.item())

            # Validation
            self.model.eval()
            with torch.no_grad():
                pred_val = self.model(obs_val, zg_val)
                val_mse = nn.functional.mse_loss(pred_val, zw_val).item()
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
        }

        return {
            "final_train_loss": train_losses[-1] if train_losses else 0.0,
            "final_val_loss": val_losses[-1] if val_losses else 0.0,
            "final_val_cosine_sim": val_cos_sims[-1] if val_cos_sims else 0.0,
            "final_val_mse": val_mses[-1] if val_mses else 0.0,
            "epochs": epochs,
            "n_train": n_train,
            "n_val": n_val,
        }

    def reset(self, initial_obs: np.ndarray, goal: np.ndarray) -> None:
        super().reset(initial_obs, goal)

    def get_intention(self, obs: np.ndarray, goal: np.ndarray, step: int = 0) -> np.ndarray:
        if not self.is_trained:
            self.train_distillation(n_pairs=300, epochs=15)

        z_g = self.fb_model.encode_backward(goal)
        
        with torch.no_grad():
            obs_t = torch.tensor(obs[None, :], dtype=torch.float32, device=self.device)
            zg_t = torch.tensor(z_g[None, :], dtype=torch.float32, device=self.device)
            pred_z = self.model(obs_t, zg_t).cpu().numpy()[0]

        return normalize_latent(pred_z, latent_dim=self.latent_dim)
