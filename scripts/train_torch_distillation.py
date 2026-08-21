import os
import sys
import time
import json
import argparse
from typing import Dict, Any, Tuple
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.agent_loader import load_pretrained_agent
from src.planners import BufferGraphPlanner
from src.models import build_student_model


def generate_or_load_golden_dataset(
    agent,
    train_obs: np.ndarray,
    n_pairs: int = 15000,
    noise_sigma: float = 0.05,
    save_path: str = None,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """
    Generates or loads offline golden demonstration dataset:
    [state (with noise), goal_z] -> [z_target from high_actor(w_1), a_target from low_actor(z_target)].
    """
    if save_path and os.path.exists(save_path):
        print(f"Loading cached golden dataset from {save_path}...")
        data = np.load(save_path)
        return {
            "states": data["states"],
            "goal_zs": data["goal_zs"],
            "z_targets": data["z_targets"],
            "a_targets": data["a_targets"],
        }

    print(f"Constructing Dijkstra teacher graph on {len(train_obs)} observations...")
    teacher = BufferGraphPlanner(
        agent,
        train_obs,
        n_landmarks=1000,
        max_edge_radius=3.5,
        reachability_cutoff=35.0,
        lookahead_dist=2.6,
    )

    print(f"Generating {n_pairs} golden demonstration pairs (noise_sigma={noise_sigma})...")
    rng = np.random.default_rng(seed)
    s_idxs = rng.choice(len(train_obs), size=n_pairs)
    g_idxs = rng.choice(len(train_obs), size=n_pairs)

    raw_states = train_obs[s_idxs].copy()
    raw_goals = train_obs[g_idxs].copy()

    # Apply Gaussian noise to states for recovery from drift / covariate shift
    noise = rng.normal(0.0, noise_sigma, size=raw_states.shape).astype(np.float32)
    noisy_states = (raw_states + noise).astype(np.float32)

    # Encode global goals B(g_j)
    goal_latents = np.asarray(
        agent.normalize_z(agent.network.select("backward_repr")(raw_goals))
    )

    z_targets = []
    # Batch compute next waypoints via Dijkstra and pass through frozen high_actor
    for i in tqdm(range(n_pairs), desc="Dijkstra Waypoints & High-Actor Targets"):
        subgoal_z = teacher.get_subgoal_latent(noisy_states[i], goal_latents[i], step=0)
        z_targets.append(subgoal_z)

    z_targets = np.asarray(z_targets, dtype=np.float32)

    # Compute reference actions a_target = low_actor(s_i, z_target)
    print("Computing reference actions through frozen low-level actor...")
    act_dist = agent.network.select("actor")(noisy_states, z_targets, goal_encoded=True, temperature=0.0)
    a_targets = np.asarray(act_dist.mode())

    dataset = {
        "states": noisy_states,
        "goal_zs": goal_latents,
        "z_targets": z_targets,
        "a_targets": a_targets,
    }

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        np.savez_compressed(
            save_path,
            states=noisy_states,
            goal_zs=goal_latents,
            z_targets=z_targets,
            a_targets=a_targets,
        )
        print(f"Golden dataset saved to {save_path}")

    return dataset


def train_torch_distillation(
    checkpoint_dir: str = "fb-test",
    split: str = "medium",
    model_type: str = "gated_attn",
    hidden_dim: int = 256,
    n_layers: int = 3,
    num_heads: int = 4,
    n_pairs: int = 20000,
    noise_sigma: float = 0.05,
    epochs: int = 50,
    batch_size: int = 256,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    l_mse: float = 0.1,
    l_cos: float = 1.0,
    l_goal: float = 0.05,
    device_str: str = "auto",
    use_mlflow: bool = True,
    output_dir: str = "results",
    seed: int = 42,
):
    print("=" * 80)
    print(f"=== PyTorch Latent Distillation: {model_type} on {split} ===")
    print(f"=== Objective: L_BC (Cosine Sim + MSE) + L_GoalAlignment ===")
    print("=" * 80)

    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, f"train_torch_distill_{model_type}_{split}.log")

    # 1. Device Selection
    if device_str == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"Using NVIDIA CUDA Device: {torch.cuda.get_device_name(0)}")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
            print("Using Apple Silicon MPS Device")
        else:
            device = torch.device("cpu")
            print("Using CPU Device")
    else:
        device = torch.device(device_str)
        print(f"Using specified device: {device}")

    # 2. MLflow setup
    mlflow_active = False
    if use_mlflow:
        try:
            import mlflow
            mlflow.set_experiment(f"latent_distillation_{split}")
            mlflow.start_run(run_name=f"distill_{model_type}_{split}")
            mlflow.log_params({
                "model_type": model_type,
                "split": split,
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
                "n_pairs": n_pairs,
                "noise_sigma": noise_sigma,
                "hidden_dim": hidden_dim,
                "n_layers": n_layers,
                "l_mse": l_mse,
                "l_cos": l_cos,
                "l_goal": l_goal,
            })
            mlflow_active = True
            print("MLflow experiment tracking initialized successfully.")
        except Exception as e:
            print(f"MLflow initialization skipped: {e}")

    # 3. Load agent & Golden dataset
    print(f"Loading pretrained FB agent on {split}...")
    agent, env, train_ds, _, _ = load_pretrained_agent(checkpoint_dir, split, seed=seed)

    data_cache_path = os.path.join(output_dir, f"golden_dataset_{split}_{n_pairs}p_s{int(noise_sigma*100)}.npz")
    dataset = generate_or_load_golden_dataset(
        agent=agent,
        train_obs=train_ds["observations"],
        n_pairs=n_pairs,
        noise_sigma=noise_sigma,
        save_path=data_cache_path,
        seed=seed,
    )

    # 4. PyTorch Dataset & DataLoaders
    X = np.concatenate([dataset["states"], dataset["goal_zs"]], axis=-1).astype(np.float32)
    Y_z = dataset["z_targets"].astype(np.float32)
    Y_g = dataset["goal_zs"].astype(np.float32)

    N = len(X)
    n_train = int(0.9 * N)
    n_val = N - n_train

    train_ds_torch = TensorDataset(
        torch.from_numpy(X[:n_train]),
        torch.from_numpy(Y_z[:n_train]),
        torch.from_numpy(Y_g[:n_train]),
    )
    val_ds_torch = TensorDataset(
        torch.from_numpy(X[n_train:]),
        torch.from_numpy(Y_z[n_train:]),
        torch.from_numpy(Y_g[n_train:]),
    )

    train_loader = DataLoader(train_ds_torch, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds_torch, batch_size=batch_size, shuffle=False)

    # 5. Build PyTorch Student Model
    obs_dim = dataset["states"].shape[-1]
    latent_dim = dataset["goal_zs"].shape[-1]

    model = build_student_model(
        model_type=model_type,
        obs_dim=obs_dim,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        num_heads=num_heads,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"PyTorch Student Model: {model_type} | Trainable Parameters: {total_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    # 6. Training Loop
    history = []
    best_val_loss = float("inf")
    save_path = os.path.join(output_dir, f"distilled_torch_{model_type}_{split}.pt")

    print(f"\nStarting PyTorch Training ({epochs} epochs, {len(train_loader)} batches/epoch)...")
    with open(log_file, "w") as lf:
        lf.write("Epoch,Train_Loss,Train_Cosine_Loss,Train_MSE_Loss,Train_Cos_Sim,Val_Loss,Val_Cos_Sim,Val_MSE\n")

        for epoch in range(1, epochs + 1):
            t0 = time.perf_counter()
            model.train()
            train_loss_sum = 0.0
            train_cos_loss_sum = 0.0
            train_mse_loss_sum = 0.0
            train_cossim_sum = 0.0

            for bx, by_z, by_g in train_loader:
                bx, by_z, by_g = bx.to(device), by_z.to(device), by_g.to(device)
                optimizer.zero_grad()

                pred_z = model(bx)

                # Cosine Similarity Loss
                cos_sim = F.cosine_similarity(pred_z, by_z, dim=-1)
                cos_loss = (1.0 - cos_sim).mean()

                # MSE Loss in latent space
                mse_loss = F.mse_loss(pred_z, by_z)

                # Goal Consistency (Alignment with global goal)
                goal_cos = F.cosine_similarity(pred_z, by_g, dim=-1).mean()
                goal_loss = -goal_cos

                # Total Loss
                total_loss = l_cos * cos_loss + l_mse * mse_loss + l_goal * goal_loss
                total_loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                bs = len(bx)
                train_loss_sum += total_loss.item() * bs
                train_cos_loss_sum += cos_loss.item() * bs
                train_mse_loss_sum += mse_loss.item() * bs
                train_cossim_sum += cos_sim.mean().item() * bs

            scheduler.step()

            train_loss = train_loss_sum / n_train
            train_cos_loss = train_cos_loss_sum / n_train
            train_mse_loss = train_mse_loss_sum / n_train
            train_cossim = train_cossim_sum / n_train

            # Validation Loop
            model.eval()
            val_loss_sum = 0.0
            val_cossim_sum = 0.0
            val_mse_sum = 0.0

            with torch.no_grad():
                for bx, by_z, by_g in val_loader:
                    bx, by_z, by_g = bx.to(device), by_z.to(device), by_g.to(device)
                    pred_z = model(bx)

                    cos_sim = F.cosine_similarity(pred_z, by_z, dim=-1)
                    cos_loss = (1.0 - cos_sim).mean()
                    mse_loss = F.mse_loss(pred_z, by_z)
                    goal_loss = -F.cosine_similarity(pred_z, by_g, dim=-1).mean()

                    v_loss = l_cos * cos_loss + l_mse * mse_loss + l_goal * goal_loss

                    bs = len(bx)
                    val_loss_sum += v_loss.item() * bs
                    val_cossim_sum += cos_sim.mean().item() * bs
                    val_mse_sum += mse_loss.item() * bs

            val_loss = val_loss_sum / n_val
            val_cossim = val_cossim_sum / n_val
            val_mse = val_mse_sum / n_val
            elapsed = time.perf_counter() - t0

            # Save best checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), save_path)
                best_tag = " [BEST SAVED]"
            else:
                best_tag = ""

            log_line = (
                f"Epoch {epoch:02d}/{epochs:02d} [{elapsed:.1f}s] | "
                f"Train Loss: {train_loss:.4f} (Cos: {train_cos_loss:.4f}, MSE: {train_mse_loss:.4f}) | "
                f"Train CosSim: {train_cossim:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val CosSim: {val_cossim:.4f} | Val MSE: {val_mse:.4f}{best_tag}"
            )
            print(log_line)

            csv_row = f"{epoch},{train_loss:.5f},{train_cos_loss:.5f},{train_mse_loss:.5f},{train_cossim:.5f},{val_loss:.5f},{val_cossim:.5f},{val_mse:.5f}\n"
            lf.write(csv_row)
            lf.flush()

            if mlflow_active:
                import mlflow
                mlflow.log_metrics({
                    "train_loss": train_loss,
                    "train_cos_loss": train_cos_loss,
                    "train_mse_loss": train_mse_loss,
                    "train_cos_sim": train_cossim,
                    "val_loss": val_loss,
                    "val_cos_sim": val_cossim,
                    "val_mse": val_mse,
                }, step=epoch)

            history.append({
                "epoch": epoch,
                "train_loss": train_loss,
                "train_cossim": train_cossim,
                "val_loss": val_loss,
                "val_cossim": val_cossim,
                "val_mse": val_mse,
                "elapsed": elapsed,
            })

    if mlflow_active:
        import mlflow
        mlflow.log_artifact(save_path)
        mlflow.end_run()

    print(f"\nTraining completed! Best model saved to {save_path}")
    return model, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="medium")
    parser.add_argument("--model_type", type=str, default="gated_attn")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--n_pairs", type=int, default=20000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--noise_sigma", type=float, default=0.05)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--no_mlflow", action="store_true")
    args = parser.parse_args()

    train_torch_distillation(
        split=args.split,
        model_type=args.model_type,
        epochs=args.epochs,
        batch_size=args.batch_size,
        n_pairs=args.n_pairs,
        lr=args.lr,
        noise_sigma=args.noise_sigma,
        device_str=args.device,
        use_mlflow=not args.no_mlflow,
    )
