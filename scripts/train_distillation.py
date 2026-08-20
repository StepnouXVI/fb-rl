import os, sys, hydra
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from omegaconf import DictConfig
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from src.agent_loader import load_pretrained_agent
from src.planners import BufferGraphPlanner
from src.models import build_student_model

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    model_type = cfg.distillation.get("model_type", "dense_eca")
    print(f"=== Distilling Buffer Graph Planner into Student ({model_type}) on {cfg.env.split} ===")
    os.makedirs(cfg.eval.output_dir, exist_ok=True)

    agent, env, train_ds, _, _ = load_pretrained_agent(cfg.eval.checkpoint_dir, cfg.env.split)
    n_pairs = cfg.distillation.n_pairs
    cache_path = os.path.join(cfg.eval.output_dir, f"distill_dataset_{cfg.env.split}_{n_pairs}pairs.npz")

    if os.path.exists(cache_path):
        print(f"Loading cached dataset from {cache_path}")
        data = np.load(cache_path)
        states = data["states"]
        goal_latents = data["goal_latents"]
        target_latents = data["target_latents"]
    else:
        teacher = BufferGraphPlanner(
            agent,
            train_ds["observations"],
            n_landmarks=1000,
            max_edge_radius=3.5,
            reachability_cutoff=35.0,
            lookahead_dist=2.6,
        )

        print(f"Generating {n_pairs} teacher demonstration pairs...")
        rng = np.random.default_rng(42)
        s_idxs = rng.choice(len(train_ds["observations"]), size=n_pairs)
        g_idxs = rng.choice(len(train_ds["observations"]), size=n_pairs)

        states = train_ds["observations"][s_idxs]
        goal_latents = np.asarray(
            agent.normalize_z(agent.network.select("backward_repr")(train_ds["observations"][g_idxs]))
        )

        target_latents = []
        for i in tqdm(range(n_pairs), desc="Generating targets"):
            subgoal_z = teacher.get_subgoal_latent(states[i], goal_latents[i], step=0)
            target_latents.append(subgoal_z)
        target_latents = np.asarray(target_latents, dtype=np.float32)
        np.savez_compressed(cache_path, states=states, goal_latents=goal_latents, target_latents=target_latents)
        print(f"Saved dataset to {cache_path}")

    # Detect device (MPS / CUDA / CPU)
    device_str = cfg.distillation.device
    if device_str == "auto":
        device = torch.device(
            "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
        )
    else:
        device = torch.device(device_str)
    print(f"Using acceleration device: {device}")

    # Train student model
    X = np.concatenate([states, goal_latents], axis=-1).astype(np.float32)
    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(target_latents))
    loader = DataLoader(dataset, batch_size=cfg.distillation.batch_size, shuffle=True)

    obs_dim = states.shape[-1]
    latent_dim = goal_latents.shape[-1]
    student = build_student_model(
        model_type=model_type,
        obs_dim=obs_dim,
        latent_dim=latent_dim,
        hidden_dim=cfg.distillation.hidden_dim,
        n_layers=cfg.distillation.get("n_layers", 3),
        num_heads=cfg.distillation.get("num_heads", 4),
    ).to(device)

    optimizer = torch.optim.AdamW(student.parameters(), lr=cfg.distillation.lr, weight_decay=1e-4)

    student.train()
    pbar = tqdm(range(1, cfg.distillation.epochs + 1), desc=f"Training Student ({model_type})")
    for epoch in pbar:
        total_loss = 0.0
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            pred = student(bx)
            cos_loss = 1.0 - torch.nn.functional.cosine_similarity(pred, by, dim=-1).mean()
            mse_loss = torch.nn.functional.mse_loss(pred, by)
            loss = cos_loss + 0.1 * mse_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(bx)
        mean_loss = total_loss / len(dataset)
        pbar.set_postfix({"loss": f"{mean_loss:.4f}"})

    save_path = os.path.join(cfg.eval.output_dir, f"distilled_{model_type}_{cfg.env.split}.pt")
    torch.save(student.state_dict(), save_path)
    print(f"Distilled model successfully saved to {save_path}")

    # Also update default distilled_mlp_medium.pt
    default_save_path = os.path.join(cfg.eval.output_dir, f"distilled_mlp_{cfg.env.split}.pt")
    torch.save(student.state_dict(), default_save_path)

if __name__ == "__main__":
    main()
