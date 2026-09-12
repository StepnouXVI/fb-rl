"""Comprehensive test suite for StagedAgent, context classes, and pipeline stages."""

import ast
from pathlib import Path
import pytest
import numpy as np

from src.agent_loader import load_pretrained_agent
from src.contexts import (
    BaseMazeContext,
    TopologicalPathContext,
    SequenceAttentionContext,
)
from src.stages import (
    PipelineStage,
    DijkstraPathBuilder,
    PathDatabaseLogger,
    StepDatabaseLogger,
    AttentionFilter,
    SubgoalSelector,
    SequenceAttentionTranslator,
    SingleWaypointTranslator,
    DirectIntentionTranslator,
    HighLevelActor,
    LowLevelActor,
)
from src.agent import (
    StagedAgent,
    create_baseline_agent,
    create_dijkstra_teacher_agent,
    create_direct_intention_agent,
    create_sequence_attention_agent,
    create_single_waypoint_agent,
)
from src.telemetry.db import TelemetryDatabase


@pytest.fixture(scope="module")
def agent_bundle():
    """Load pretrained agent, environment, and dataset bundle."""
    return load_pretrained_agent("fb-test", "medium")


def test_code_architecture_rules():
    """Verify naming, length, and comment rules across staged agent source files."""
    repo_root = Path(__file__).resolve().parent.parent
    target_files = [
        repo_root / "src" / "contexts.py",
        repo_root / "src" / "stages.py",
        repo_root / "src" / "agent.py",
        repo_root / "tests" / "test_staged_agent.py",
    ]

    for file_path in target_files:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        for idx, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith(chr(35)):
                raise AssertionError(f"Forbidden comment at {file_path}:{idx + 1}")

    for file_path in target_files[:3]:
        with open(file_path, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read())
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                length = node.end_lineno - node.lineno + 1
                assert length <= 60, f"{file_path} function {node.name} has {length} lines"

    stage_classes = [
        DijkstraPathBuilder,
        PathDatabaseLogger,
        StepDatabaseLogger,
        AttentionFilter,
        SubgoalSelector,
        SequenceAttentionTranslator,
        SingleWaypointTranslator,
        DirectIntentionTranslator,
        HighLevelActor,
        LowLevelActor,
    ]
    for cls in stage_classes:
        assert not cls.__name__.endswith("Stage"), f"{cls.__name__} has illegal suffix"
    assert StagedAgent.__name__ == "StagedAgent"


def test_context_structures():
    """Verify properties, field inheritance, and initialization of contexts."""
    base_ctx = BaseMazeContext()
    assert base_ctx.pos_xy is None
    base_ctx.obs = np.array([1.5, -2.5, 0.0, 1.0], dtype=np.float32)
    np.testing.assert_allclose(base_ctx.pos_xy, np.array([1.5, -2.5], dtype=np.float32))

    topo_ctx = TopologicalPathContext(obs=base_ctx.obs)
    assert hasattr(topo_ctx, "path_coords")
    assert hasattr(topo_ctx, "path_latents")
    assert hasattr(topo_ctx, "waypoint_coords")
    assert hasattr(topo_ctx, "current_path_idx")
    assert hasattr(topo_ctx, "replan_triggered")
    assert hasattr(topo_ctx, "lookahead_xy")
    assert hasattr(topo_ctx, "dist_to_lookahead")
    assert topo_ctx.pos_xy is not None

    seq_ctx = SequenceAttentionContext(obs=base_ctx.obs)
    assert hasattr(seq_ctx, "attention_targets")
    assert hasattr(seq_ctx, "attention_weights")
    assert hasattr(seq_ctx, "pad_seq")
    assert hasattr(seq_ctx, "seq_mask")
    assert hasattr(seq_ctx, "curv_arr")
    assert isinstance(seq_ctx, TopologicalPathContext)


def test_dijkstra_path_builder_and_logger(agent_bundle, tmp_path):
    """Test DijkstraPathBuilder path finding and PathDatabaseLogger recording."""
    agent, _, train_ds, _, _ = agent_bundle
    builder = DijkstraPathBuilder(agent, train_ds["observations"][:200], n_landmarks=30)
    db_file = str(tmp_path / "stage_test.db")
    db = TelemetryDatabase(db_file)
    db.insert_experiment("exp", "test")
    db.insert_run("run", "exp", "test", "med", 0)
    db.insert_episode(
        episode_id="ep1", run_id="run", seed=0, task_id=0, episode_idx=0,
        start_x=0.0, start_y=0.0, goal_x=0.0, goal_y=0.0, is_success=0,
        total_steps=0, total_reward=0.0, mean_speed=0.0, self_intersections=0,
        mean_cross_track_error=0.0, max_cross_track_error=0.0, mean_latency_ms=0.0
    )
    logger = PathDatabaseLogger()

    obs = train_ds["observations"][0]
    goal_z = np.asarray(
        agent.normalize_z(
            agent.network.select("backward_repr")(train_ds["observations"][10:11])
        )[0]
    )

    ctx = TopologicalPathContext(obs=obs, goal_latent=goal_z, db=db, episode_id="ep1")
    builder.reset(ctx)
    assert len(ctx.path_coords) > 0
    assert len(ctx.path_latents) > 0
    assert ctx.replan_triggered is True

    logger(ctx)
    cur = db.conn.cursor()
    cur.execute("SELECT COUNT(*) FROM planned_paths WHERE episode_id = 'ep1'")
    assert cur.fetchone()[0] >= 1
    cur.close()

    builder(ctx)
    assert ctx.lookahead_xy is not None
    assert ctx.dist_to_lookahead is not None
    db.close()


def test_step_database_logger(tmp_path):
    """Test StepDatabaseLogger writing step telemetry to SQLite."""
    db_file = str(tmp_path / "step_stage.db")
    db = TelemetryDatabase(db_file)
    db.insert_experiment("exp", "test")
    db.insert_run("run", "exp", "test", "med", 0)
    db.insert_episode(
        episode_id="ep_step", run_id="run", seed=0, task_id=0, episode_idx=0,
        start_x=0.0, start_y=0.0, goal_x=0.0, goal_y=0.0, is_success=0,
        total_steps=0, total_reward=0.0, mean_speed=0.0, self_intersections=0,
        mean_cross_track_error=0.0, max_cross_track_error=0.0, mean_latency_ms=0.0
    )
    logger = StepDatabaseLogger(db_getter=lambda: db)
    obs = np.zeros(29, dtype=np.float32)
    obs[0], obs[1] = 1.0, 2.0
    ctx = TopologicalPathContext(
        obs=obs, action=np.zeros(8, dtype=np.float32), step=1,
        reward=1.0, done=False, latency_ms=2.5, episode_id="ep_step"
    )
    logger(ctx)
    cur = db.conn.cursor()
    cur.execute("SELECT x, y, reward, latency_ms FROM steps WHERE episode_id = 'ep_step'")
    row = cur.fetchone()
    assert row is not None
    assert np.isclose(row[0], 1.0)
    assert np.isclose(row[1], 2.0)
    assert np.isclose(row[2], 1.0)
    assert np.isclose(row[3], 2.5)
    cur.close()
    db.close()


def test_dijkstra_replan_triggers(agent_bundle):
    """Test replan triggering upon large deviation (>4.8m) or stuck count (>45)."""
    agent, _, train_ds, _, _ = agent_bundle
    builder = DijkstraPathBuilder(agent, train_ds["observations"][:200], n_landmarks=30)
    obs = train_ds["observations"][0]
    goal_z = np.asarray(
        agent.normalize_z(
            agent.network.select("backward_repr")(train_ds["observations"][10:11])
        )[0]
    )
    ctx = TopologicalPathContext(obs=obs, goal_latent=goal_z)
    builder.reset(ctx)
    ctx.replan_triggered = False

    deviated_obs = obs.copy()
    deviated_obs[:2] += 10.0
    ctx.obs = deviated_obs
    builder(ctx)
    assert ctx.replan_triggered is True

    builder.reset(ctx)
    ctx.replan_triggered = False
    builder.stuck_count = 46
    builder(ctx)
    assert ctx.replan_triggered is True


def test_attention_filter():
    """Test SubgoalSelector sliding lookahead and landmark selection."""
    selector = SubgoalSelector(lookahead_dist=2.6, num_landmarks=4)

    path_pts = [
        np.array([0.0, 0.0]),
        np.array([2.0, 0.0]),
        np.array([4.0, 2.0]),
        np.array([6.0, 2.0]),
    ]
    goal_z = np.ones(128, dtype=np.float32)
    path_latents = [np.ones(128, dtype=np.float32) * (i + 1) for i in range(4)]

    ctx = SequenceAttentionContext(
        obs=np.array([0.1, 0.0, 0.0]),
        goal_latent=goal_z,
        path_coords=path_pts,
        path_latents=path_latents,
        waypoint_coords=[path_pts[0], path_pts[2], path_pts[3]],
        waypoint_indices=[0, 2, 3],
    )
    ctx.path_dists = [0.0, 2.0, 4.828, 6.828]

    selector.reset(ctx)
    selector(ctx)
    assert ctx.lookahead_xy is not None
    assert len(ctx.attention_targets) >= 1
    assert len(ctx.attention_latents) == len(ctx.attention_targets)


def test_single_and_direct_translators():
    """Test standalone inference of single-waypoint and direct intention translators."""
    sw_stage = SingleWaypointTranslator(latent_dim=128, hidden_dim=64, n_layers=2)
    di_stage = DirectIntentionTranslator(latent_dim=128, hidden_dim=64, n_layers=2)

    obs = np.zeros(29, dtype=np.float32)
    goal_z = np.ones(128, dtype=np.float32)
    ctx = TopologicalPathContext(obs=obs, goal_latent=goal_z)

    sw_stage(ctx)
    assert ctx.z_cmd is not None
    assert ctx.z_cmd.shape == (128,)

    base_ctx = BaseMazeContext(obs=obs, goal_latent=goal_z)
    di_stage(base_ctx)
    assert base_ctx.z_cmd is not None
    assert base_ctx.z_cmd.shape == (128,)


def test_high_and_low_level_actors(agent_bundle):
    """Test HighLevelActor subgoal selection and LowLevelActor action sampling."""
    agent, _, train_ds, _, _ = agent_bundle
    high_actor = HighLevelActor(agent, dataset_states=train_ds["observations"][:50])
    low_actor = LowLevelActor(agent)

    obs = train_ds["observations"][0]
    goal_z = np.asarray(
        agent.normalize_z(
            agent.network.select("backward_repr")(train_ds["observations"][5:6])
        )[0]
    )
    ctx = BaseMazeContext(obs=obs, goal_latent=goal_z)

    high_actor(ctx)
    assert ctx.z_cmd is not None
    assert ctx.z_cmd.shape == (agent.config["latent_dim"],)
    assert ctx.lookahead_xy is not None

    low_actor(ctx)
    assert ctx.action is not None
    assert ctx.action.shape == (8,)
    assert np.all(ctx.action >= -1.0) and np.all(ctx.action <= 1.0)


def test_staged_baseline_agent(agent_bundle):
    """Test baseline agent pipeline construction, reset, warmup, and sampling."""
    agent, _, train_ds, _, _ = agent_bundle
    staged = create_baseline_agent(agent, dataset_states=train_ds["observations"][:50])
    obs = train_ds["observations"][0]
    goal_z = np.asarray(
        agent.normalize_z(
            agent.network.select("backward_repr")(train_ds["observations"][10:11])
        )[0]
    )

    staged.warmup(obs, goal_z)
    staged.reset(obs, goal_z)
    action = staged.sample_action(obs, goal_z)
    assert action.shape == (8,)
    assert staged.ctx.z_cmd is not None


def test_staged_dijkstra_teacher_agent(agent_bundle):
    """Test Dijkstra teacher agent execution and subgoal state."""
    agent, _, train_ds, _, _ = agent_bundle
    staged = create_dijkstra_teacher_agent(
        agent, train_ds["observations"][:200], n_landmarks=30
    )
    obs = train_ds["observations"][0]
    goal_z = np.asarray(
        agent.normalize_z(
            agent.network.select("backward_repr")(train_ds["observations"][10:11])
        )[0]
    )

    staged.warmup(obs, goal_z)
    staged.reset(obs, goal_z)
    action = staged.sample_action(obs, goal_z)
    assert action.shape == (8,)
    assert len(staged.ctx.waypoint_coords) > 0


def test_staged_single_waypoint_agent(agent_bundle):
    """Test single waypoint agent pipeline with custom translator params."""
    agent, _, train_ds, _, _ = agent_bundle
    staged = create_single_waypoint_agent(
        agent,
        train_ds["observations"][:200],
        n_landmarks=30,
        hidden_dim=64,
        n_layers=2,
    )
    obs = train_ds["observations"][0]
    goal_z = np.asarray(
        agent.normalize_z(
            agent.network.select("backward_repr")(train_ds["observations"][10:11])
        )[0]
    )

    staged.warmup(obs, goal_z)
    staged.reset(obs, goal_z)
    action = staged.sample_action(obs, goal_z)
    assert action.shape == (8,)
    assert staged.ctx.z_cmd is not None


def test_staged_sequence_attention_agent(agent_bundle):
    """Test sequence attention agent execution."""
    agent, _, train_ds, _, _ = agent_bundle
    staged = create_sequence_attention_agent(
        agent,
        train_ds["observations"][:200],
        n_landmarks=30,
        max_seq_len=8,
        hidden_dim=64,
        num_heads=2,
        n_layers=1,
    )
    obs = train_ds["observations"][0]
    goal_z = np.asarray(
        agent.normalize_z(
            agent.network.select("backward_repr")(train_ds["observations"][10:11])
        )[0]
    )

    staged.warmup(obs, goal_z)
    staged.reset(obs, goal_z)
    action = staged.sample_action(obs, goal_z)
    assert action.shape == (8,)
    assert len(staged.ctx.attention_targets) > 0
    assert len(staged.ctx.attention_weights) == len(staged.ctx.attention_targets)
    np.testing.assert_allclose(sum(staged.ctx.attention_weights), 1.0, atol=1e-3)


def test_staged_direct_intention_agent(agent_bundle):
    """Test direct intention amortized agent execution."""
    agent, _, train_ds, _, _ = agent_bundle
    staged = create_direct_intention_agent(
        agent,
        hidden_dim=64,
        n_layers=2,
    )
    obs = train_ds["observations"][0]
    goal_z = np.asarray(
        agent.normalize_z(
            agent.network.select("backward_repr")(train_ds["observations"][10:11])
        )[0]
    )

    staged.warmup(obs, goal_z)
    staged.reset(obs, goal_z)
    action = staged.sample_action(obs, goal_z)
    assert action.shape == (8,)
    assert staged.ctx.z_cmd is not None
