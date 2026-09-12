"""Unified StagedAgent architecture and factory functions for reinforcement learning."""

from typing import Any, List, Optional, Type
import jax
import numpy as np

from src.contexts import (
    BaseMazeContext,
    SequenceAttentionContext,
    TopologicalPathContext,
)
from src.stages import (
    AttentionFilter,
    DijkstraPathBuilder,
    DirectIntentionTranslator,
    HighLevelActor,
    LowLevelActor,
    PathDatabaseLogger,
    PipelineStage,
    SequenceAttentionTranslator,
    SingleWaypointTranslator,
    SubgoalSelector,
)


class StagedAgent:
    """Sequential pipeline agent processing context through modular execution stages."""

    def __init__(
        self,
        stages: List[PipelineStage],
        context_cls: Type[BaseMazeContext] = BaseMazeContext,
        name: str = "StagedAgent",
    ) -> None:
        """Initialize pipeline stages and associate context type."""
        self.stages = list(stages)
        self.context_cls = context_cls
        self.name = name
        self.ctx: BaseMazeContext = self.context_cls()

    def reset(
        self,
        obs: Optional[np.ndarray] = None,
        goal_latent: Optional[np.ndarray] = None,
        **ctx_kwargs: Any,
    ) -> None:
        """Instantiate new context and reset all pipeline stages."""
        self.ctx = self.context_cls(obs=obs, goal_latent=goal_latent)
        for key, value in ctx_kwargs.items():
            if hasattr(self.ctx, key):
                setattr(self.ctx, key, value)
        for stage in self.stages:
            stage.reset(self.ctx)

    def sample_action(
        self,
        obs: np.ndarray,
        goal_latent: np.ndarray,
        step: int = 0,
        seed: Optional[Any] = None,
        temperature: float = 0.0,
    ) -> np.ndarray:
        """Update context with observation and process through all stages."""
        self.ctx.obs = obs
        self.ctx.goal_latent = goal_latent
        self.ctx.step = step
        self.ctx.seed = seed
        self.ctx.temperature = temperature
        for stage in self.stages:
            stage(self.ctx)
        return np.asarray(self.ctx.action)

    def warmup(self, obs: np.ndarray, goal_latent: np.ndarray) -> None:
        """Trigger JIT compilation across all stages to prevent runtime lag."""
        self.reset(obs, goal_latent)
        self.sample_action(
            obs,
            goal_latent,
            step=0,
            seed=jax.random.PRNGKey(0),
            temperature=0.0,
        )


def create_baseline_agent(
    agent: Any,
    dataset_states: Optional[np.ndarray] = None,
    use_high_actor: bool = True,
    name: str = "Single-Intention Baseline",
) -> StagedAgent:
    """Create baseline agent using high actor and low level policy."""
    stages: List[PipelineStage] = []
    if use_high_actor:
        stages.append(HighLevelActor(agent, dataset_states=dataset_states))
    stages.append(LowLevelActor(agent))
    return StagedAgent(stages=stages, context_cls=BaseMazeContext, name=name)


def create_dijkstra_teacher_agent(
    agent: Any,
    dataset_states: np.ndarray,
    n_landmarks: int = 1000,
    max_edge_radius: float = 3.5,
    reachability_cutoff: float = 35.0,
    lookahead_dist: float = 2.6,
    name: str = "Dijkstra Teacher Agent",
) -> StagedAgent:
    """Create teacher agent navigating along Dijkstra topological paths."""
    stages: List[PipelineStage] = [
        DijkstraPathBuilder(
            agent,
            dataset_states,
            n_landmarks=n_landmarks,
            max_edge_radius=max_edge_radius,
            reachability_cutoff=reachability_cutoff,
            lookahead_dist=lookahead_dist,
        ),
        PathDatabaseLogger(),
        HighLevelActor(agent, dataset_states=dataset_states),
        LowLevelActor(agent),
    ]
    return StagedAgent(stages=stages, context_cls=TopologicalPathContext, name=name)


def create_single_waypoint_agent(
    agent: Any,
    dataset_states: np.ndarray,
    checkpoint_path: Optional[str] = None,
    translator_params: Optional[Any] = None,
    n_landmarks: int = 1000,
    max_edge_radius: float = 3.5,
    reachability_cutoff: float = 35.0,
    lookahead_dist: float = 2.6,
    hidden_dim: int = 256,
    n_layers: int = 3,
    name: str = "Dijkstra Single Waypoint Agent",
) -> StagedAgent:
    """Create agent utilizing single-waypoint neural intention translator."""
    stages: List[PipelineStage] = [
        DijkstraPathBuilder(
            agent,
            dataset_states,
            n_landmarks=n_landmarks,
            max_edge_radius=max_edge_radius,
            reachability_cutoff=reachability_cutoff,
            lookahead_dist=lookahead_dist,
        ),
        PathDatabaseLogger(),
        SingleWaypointTranslator(
            params=translator_params,
            checkpoint_path=checkpoint_path,
            latent_dim=agent.config["latent_dim"],
            hidden_dim=hidden_dim,
            n_layers=n_layers,
        ),
        LowLevelActor(agent),
    ]
    return StagedAgent(stages=stages, context_cls=TopologicalPathContext, name=name)


def create_sequence_attention_agent(
    agent: Any,
    dataset_states: np.ndarray,
    checkpoint_path: Optional[str] = None,
    seq_params: Optional[Any] = None,
    n_landmarks: int = 1000,
    max_edge_radius: float = 3.5,
    reachability_cutoff: float = 35.0,
    lookahead_dist: float = 2.6,
    max_seq_len: int = 16,
    hidden_dim: int = 384,
    num_heads: int = 6,
    n_layers: int = 3,
    dropout_rate: float = 0.2,
    alibi_slope: float = 0.4,
    name: str = "Dijkstra Sequence Attention Agent",
) -> StagedAgent:
    """Create agent utilizing ALiBi-enhanced sequence attention transformer."""
    stages: List[PipelineStage] = [
        DijkstraPathBuilder(
            agent,
            dataset_states,
            n_landmarks=n_landmarks,
            max_edge_radius=max_edge_radius,
            reachability_cutoff=reachability_cutoff,
            lookahead_dist=lookahead_dist,
        ),
        PathDatabaseLogger(),
        SubgoalSelector(
            lookahead_dist=lookahead_dist,
            num_landmarks=4,
        ),
        SequenceAttentionTranslator(
            params=seq_params,
            checkpoint_path=checkpoint_path,
            latent_dim=agent.config["latent_dim"],
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            n_layers=n_layers,
            dropout_rate=dropout_rate,
            alibi_slope=alibi_slope,
        ),
        LowLevelActor(agent),
    ]
    return StagedAgent(stages=stages, context_cls=SequenceAttentionContext, name=name)


def create_direct_intention_agent(
    agent: Any,
    checkpoint_path: Optional[str] = None,
    params: Optional[Any] = None,
    model_type: str = "gated_attn",
    hidden_dim: int = 256,
    n_layers: int = 3,
    name: str = "Direct Intention Agent",
) -> StagedAgent:
    """Create direct amortized intention agent with O(1) planning."""
    stages: List[PipelineStage] = [
        DirectIntentionTranslator(
            checkpoint_path=checkpoint_path,
            params=params,
            model_type=model_type,
            latent_dim=agent.config["latent_dim"],
            hidden_dim=hidden_dim,
            n_layers=n_layers,
        ),
        LowLevelActor(agent),
    ]
    return StagedAgent(stages=stages, context_cls=BaseMazeContext, name=name)
