"""Aim experiment tracker wrapper supporting local RocksDB and batched remote transport."""

import os
from typing import Any, Dict, List, Optional

try:
    from aim import Figure, Run
    _AIM_AVAILABLE = True
except ImportError:
    _AIM_AVAILABLE = False
    Figure = None
    Run = None

from src.telemetry.aim_fast import patch as patch_aim_fast


class AimTracker:
    """Encapsulates Aim Run lifecycle, metric logging, and Plotly figure tracking."""

    def __init__(
        self,
        repo: str = "results/aim",
        experiment: str = "default",
        run_name: Optional[str] = None,
        batch_size: int = 50,
    ) -> None:
        """Initialize Run instance, applying fast remote batching patch if URI is remote."""
        if not _AIM_AVAILABLE:
            self.run = None
            return
        if repo.startswith("aim://"):
            patch_aim_fast(batch_size=batch_size)
        else:
            os.makedirs(repo, exist_ok=True)
        self.run = Run(repo=repo, experiment=experiment)
        if run_name:
            self.run.name = run_name

    @property
    def hash(self) -> Optional[str]:
        """Return unique Aim run hash if run is active."""
        return self.run.hash if self.run else None

    def set_params(self, params: Dict[str, Any]) -> None:
        """Assign hyperparameters dictionary to Aim run."""
        if self.run and params:
            self.run["hparams"] = dict(params)

    def add_tags(self, tags: List[str]) -> None:
        """Attach category tags to Aim run."""
        if self.run and tags:
            for t in tags:
                self.run.add_tag(str(t))

    def track(
        self,
        value: float,
        name: str,
        step: Optional[int] = None,
        epoch: Optional[int] = None,
        context: Optional[Dict[str, str]] = None,
    ) -> None:
        """Record numerical scalar metric into Aim repository."""
        if self.run:
            self.run.track(float(value), name=name, step=step, epoch=epoch, context=context)

    def track_figure(
        self,
        fig: Any,
        name: str,
        step: Optional[int] = None,
        context: Optional[Dict[str, str]] = None,
    ) -> None:
        """Convert Plotly figure into aim.Figure and track object."""
        if self.run and Figure and fig:
            aim_fig = Figure(fig)
            self.run.track(aim_fig, name=name, step=step, context=context)

    def close(self) -> None:
        """Flush buffered writes and close Aim run."""
        if self.run:
            self.run.close()
            self.run = None

    def __enter__(self) -> "AimTracker":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.close()
