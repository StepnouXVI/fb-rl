import os
import tempfile
import pytest
import numpy as np
import pandas as pd
from hydra import initialize, compose
from omegaconf import OmegaConf
import jax.numpy as jnp

from src.agent_loader import load_pretrained_agent
from src.planners import BufferGraphPlanner
from scripts.experiment_intention import (
    encode_coord_to_z,
    run_scenario,
    plot_scenario_results,
    draw_maze,
    normalize_maze_type,
)


@pytest.fixture(scope="module")
def loaded_agent_and_env():
    """Loads agent and environment once for fast unit tests."""
    agent, env, train_ds, val_ds, config = load_pretrained_agent(split="medium", seed=0)
    return agent, env, train_ds, val_ds, config


def test_hydra_experiment_config_parsing():
    """Verifies that configs/experiment.yaml parses correctly via Hydra with all expected defaults."""
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name="experiment", overrides=["env=antmaze_medium"])
        assert cfg.env.split == "medium"
        assert cfg.checkpoint_dir == "fb-test"
        assert cfg.target_radius == 1.0
        assert cfg.terminate_on_goal is True
        assert len(cfg.scenarios) >= 4

        modes = {s.mode for s in cfg.scenarios}
        assert "planner" in modes
        assert "direct_latent" in modes
        assert "custom_waypoints" in modes
        assert "custom_z" in modes


def test_hydra_scenario_override_and_filter():
    """Verifies that scenario_name filtering works as expected."""
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name="experiment", overrides=["scenario_name=sharp_turn"])
        assert cfg.scenario_name == "sharp_turn"
        sc_names = [s.name for s in cfg.scenarios]
        assert "sharp_turn" in sc_names


def test_ant_coordinate_positioning(loaded_agent_and_env):
    """Verifies precise ant positioning using set_xy and state synchronization."""
    _, env, _, _, _ = loaded_agent_and_env

    test_coords = [[0.0, 0.0], [8.0, 8.0], [20.0, 0.0], [0.0, 20.0]]
    for xy in test_coords:
        env.reset(seed=42)
        env.unwrapped.set_xy(np.asarray(xy, dtype=np.float32))
        ob = env.unwrapped.get_ob()
        actual_xy = env.unwrapped.get_xy()

        np.testing.assert_allclose(actual_xy, xy, atol=1e-3)
        np.testing.assert_allclose(ob[:2], xy, atol=1e-3)


def test_intention_encoding_and_modes(loaded_agent_and_env):
    """Verifies intention encoding for coordinates and latent representations."""
    agent, _, train_ds, _, _ = loaded_agent_and_env

    target_xy = [8.0, 8.0]
    z, nearest_state = encode_coord_to_z(agent, train_ds["observations"], target_xy)

    assert z.ndim == 1
    assert z.shape[0] == agent.config["latent_dim"]
    expected_norm = np.sqrt(agent.config["latent_dim"])
    np.testing.assert_allclose(np.linalg.norm(z), expected_norm, atol=1e-3)
    assert np.linalg.norm(nearest_state[:2] - np.asarray(target_xy)) < 3.0


def test_scenario_execution_planner_mode(loaded_agent_and_env):
    """Verifies execution of planner mode and telemetry collection."""
    agent, env, train_ds, _, config = loaded_agent_and_env

    sc = {
        "name": "test_planner_short",
        "start_xy": [0.0, 0.0],
        "target_xy": [8.0, 8.0],
        "mode": "planner",
        "max_steps": 5,
        "planner_cfg": {
            "n_landmarks": 100,
            "lookahead_dist": 2.6,
            "max_edge_radius": 3.5,
            "reachability_cutoff": 35.0,
        },
    }

    summary, df_telemetry, waypoints, planner_obj = run_scenario(
        agent=agent,
        env=env,
        train_ds=train_ds,
        config=config,
        scenario_cfg=sc,
        eval_temperature=0.0,
        target_radius=1.0,
        terminate_on_goal=True,
        seed=0,
    )

    assert summary["scenario_name"] == "test_planner_short"
    assert summary["mode"] == "planner"
    assert summary["num_steps"] <= 5
    assert len(df_telemetry) == summary["num_steps"] + 1
    assert len(waypoints) > 0
    assert planner_obj is not None

    required_cols = [
        "scenario_name",
        "mode",
        "step",
        "x",
        "y",
        "action_norm",
        "speed",
        "subgoal_x",
        "subgoal_y",
        "dist_to_subgoal",
        "dist_to_goal",
        "alignment_cos",
        "reached_goal",
        "reward",
        "done",
    ]
    for col in required_cols:
        assert col in df_telemetry.columns


def test_scenario_execution_direct_latent_mode(loaded_agent_and_env):
    """Verifies execution of direct_latent mode and telemetry collection."""
    agent, env, train_ds, _, config = loaded_agent_and_env

    sc = {
        "name": "test_direct_short",
        "start_xy": [0.0, 0.0],
        "target_xy": [0.0, 3.0],
        "mode": "direct_latent",
        "max_steps": 5,
    }

    summary, df_telemetry, waypoints, _ = run_scenario(
        agent=agent,
        env=env,
        train_ds=train_ds,
        config=config,
        scenario_cfg=sc,
        eval_temperature=0.0,
        target_radius=1.0,
        terminate_on_goal=True,
        seed=0,
    )

    assert summary["scenario_name"] == "test_direct_short"
    assert summary["mode"] == "direct_latent"
    assert summary["num_steps"] <= 5
    assert len(waypoints) == 1
    assert waypoints[0] == [0.0, 3.0]


def test_scenario_execution_custom_waypoints_mode(loaded_agent_and_env):
    """Verifies execution of custom_waypoints mode and telemetry collection."""
    agent, env, train_ds, _, config = loaded_agent_and_env

    sc = {
        "name": "test_cw_short",
        "start_xy": [0.0, 0.0],
        "target_xy": [8.0, 8.0],
        "mode": "custom_waypoints",
        "custom_waypoints": [[0.0, 4.0], [4.0, 8.0], [8.0, 8.0]],
        "max_steps": 5,
        "waypoint_threshold": 1.8,
    }

    summary, df_telemetry, waypoints, _ = run_scenario(
        agent=agent,
        env=env,
        train_ds=train_ds,
        config=config,
        scenario_cfg=sc,
        eval_temperature=0.0,
        target_radius=1.0,
        terminate_on_goal=True,
        seed=0,
    )

    assert summary["scenario_name"] == "test_cw_short"
    assert summary["mode"] == "custom_waypoints"
    assert summary["num_steps"] <= 5
    assert len(waypoints) == 3


def test_scenario_execution_custom_z_mode(loaded_agent_and_env):
    """Verifies execution of custom_z mode and telemetry collection."""
    agent, env, train_ds, _, config = loaded_agent_and_env

    rng = np.random.default_rng(42)
    latent_dim = agent.config["latent_dim"]
    rand_z = rng.standard_normal(latent_dim).tolist()

    sc = {
        "name": "test_custom_z_short",
        "start_xy": [0.0, 0.0],
        "target_xy": [4.0, 8.0],
        "mode": "custom_z",
        "custom_z": rand_z,
        "max_steps": 5,
    }

    summary, df_telemetry, waypoints, _ = run_scenario(
        agent=agent,
        env=env,
        train_ds=train_ds,
        config=config,
        scenario_cfg=sc,
        eval_temperature=0.0,
        target_radius=1.0,
        terminate_on_goal=True,
        seed=0,
    )

    assert summary["scenario_name"] == "test_custom_z_short"
    assert summary["mode"] == "custom_z"
    assert summary["num_steps"] <= 5


def test_plot_generation(loaded_agent_and_env):
    """Verifies that high-resolution dual-panel plot is saved without error."""
    agent, env, train_ds, _, config = loaded_agent_and_env

    sc = {
        "name": "test_plot_gen",
        "start_xy": [0.0, 0.0],
        "target_xy": [0.0, 3.0],
        "mode": "direct_latent",
        "max_steps": 10,
    }

    summary, df_telemetry, waypoints, _ = run_scenario(
        agent=agent,
        env=env,
        train_ds=train_ds,
        config=config,
        scenario_cfg=sc,
        eval_temperature=0.0,
        target_radius=1.0,
        terminate_on_goal=True,
        seed=0,
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        plot_path = os.path.join(tmp_dir, "test_plot.png")
        plot_scenario_results(
            summary=summary,
            df_telemetry=df_telemetry,
            waypoints=waypoints,
            output_path=plot_path,
            maze_type="medium",
            target_radius=1.0,
        )
        assert os.path.exists(plot_path)
        assert os.path.getsize(plot_path) > 10000


def test_multi_seed_scenario_execution(loaded_agent_and_env):
    """Verifies that running scenarios across multiple seeds works and produces separate summaries."""
    agent, env, train_ds, _, config = loaded_agent_and_env

    sc = {
        "name": "test_multi_seed",
        "start_xy": [0.0, 0.0],
        "target_xy": [0.0, 3.0],
        "mode": "direct_latent",
        "max_steps": 5,
    }

    seeds = [0, 42, 100]
    summaries = []
    for s in seeds:
        summary, df_telemetry, _, _ = run_scenario(
            agent=agent,
            env=env,
            train_ds=train_ds,
            config=config,
            scenario_cfg=sc,
            eval_temperature=0.0,
            target_radius=1.0,
            terminate_on_goal=True,
            seed=s,
        )
        assert summary["seed"] == s
        assert "seed" in df_telemetry.columns
        assert (df_telemetry["seed"] == s).all()
        summaries.append(summary)

    assert len(summaries) == 3
    df_all = pd.DataFrame(summaries)
    assert list(df_all["seed"]) == [0, 42, 100]

