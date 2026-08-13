import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import torch
import numpy as np
import importlib.metadata

def test_imports():
    print("=" * 50)
    print("Checking Python environment requirements...")
    print("=" * 50)

    # Python version
    print(f"[OK] Python version: {sys.version.split()[0]}")

    # Torch & MPS check (MacBook Pro M1 Pro)
    print(f"[OK] PyTorch version: {torch.__version__}")
    mps_available = torch.backends.mps.is_available()
    print(f"[OK] Apple Silicon MPS (GPU acceleration) available: {mps_available}")

    if mps_available:
        x = torch.ones(5, device="mps")
        print(f"[OK] MPS Tensor test successful: {x.sum().item()}")

    # Plotly
    import plotly
    import plotly.graph_objects as go
    fig = go.Figure(data=go.Scatter(x=[1, 2, 3], y=[4, 5, 6]))
    print(f"[OK] Plotly version: {plotly.__version__}")

    # Hydra
    import hydra
    from omegaconf import OmegaConf
    cfg = OmegaConf.create({"env_name": "antmaze-medium-navigate-v0", "seed": 42})
    print(f"[OK] Hydra version: {hydra.__version__} (Config test: {cfg.env_name})")

    # MLflow
    import mlflow
    print(f"[OK] MLflow version: {mlflow.__version__}")

    # OGBench
    import ogbench
    ogbench_ver = importlib.metadata.version("ogbench")
    print(f"[OK] OGBench installed successfully! Version: {ogbench_ver}")

    print("=" * 50)
    print("ALL ENVIRONMENT CHECKS PASSED SUCCESSFULLY!")
    print("=" * 50)

if __name__ == "__main__":
    test_imports()
