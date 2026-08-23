import os
import tempfile
import pytest
import numpy as np
import pandas as pd
from hydra import initialize, compose
import jax.numpy as jnp

from src.agent_loader import load_pretrained_agent
from scripts.experiment_intention import (
    encode_coord_to_z,
    run_scenario,
    plot_scenario_results,
    draw_maze,
    normalize_maze_type,
)


@pytest.fixture(scope="module")
def loaded_agent_and_env():
    agent, env, train_ds, val_ds, config = load_pretrained_agent(split="medium", seed=0)
    return agent, env, train_ds, val_ds, config


def test_hydra_experiment_config_parsing():
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name="experiment", overrides=["env=antmaze_medium"])
        assert cfg.env.split == "medium" and cfg.target_radius == 1.0
        assert len(cfg.scenarios) >= 4
        modes = {s.mode for s in cfg.scenarios}
        assert {"planner", "direct_latent", "custom_waypoints"}.issubset(modes)


def test_ant_coordinate_positioning(loaded_agent_and_env):
    _, env, _, _, _ = loaded_agent_and_env
    for xy in [[0.0, 0.0], [8.0, 8.0], [20.0, 0.0], [0.0, 20.0]]:
        env.reset(seed=42)
        env.unwrapped.set_xy(np.asarray(xy, dtype=np.float32))
        np.testing.assert_allclose(env.unwrapped.get_xy(), xy, atol=1e-3)
        np.testing.assert_allclose(env.unwrapped.get_ob()[:2], xy, atol=1e-3)


def test_intention_encoding_and_modes(loaded_agent_and_env):
    agent, _, train_ds, _, _ = loaded_agent_and_env
    target_xy = [8.0, 8.0]
    z, nearest_state = encode_coord_to_z(agent, train_ds["observations"], target_xy)
    assert z.shape[0] == agent.config["latent_dim"]
    np.testing.assert_allclose(np.linalg.norm(z), np.sqrt(agent.config["latent_dim"]), atol=1e-3)
    assert np.linalg.norm(nearest_state[:2] - np.asarray(target_xy)) < 3.0


def test_scenario_execution_planner_mode(loaded_agent_and_env):
    agent, env, train_ds, _, config = loaded_agent_and_env
    sc = {"name": "test_planner", "start_xy": [0.0, 0.0], "target_xy": [8.0, 8.0], "mode": "planner", "max_steps": 5, "planner_cfg": {"n_landmarks": 50, "lookahead_dist": 2.6}}
    summary, df_tel, waypoints = run_scenario(agent, env, train_ds, config, sc, seed=0)
    assert summary["name"] == "test_planner" and summary["mode"] == "planner"
    assert len(df_tel) <= 6 and len(waypoints) > 0
    for col in ["step", "x", "y", "speed", "subgoal_x", "subgoal_y", "dist_to_goal"]:
        assert col in df_tel.columns


def test_scenario_execution_direct_and_custom(loaded_agent_and_env):
    agent, env, train_ds, _, config = loaded_agent_and_env
    sc_dir = {"name": "test_direct", "start_xy": [0.0, 0.0], "target_xy": [0.0, 3.0], "mode": "direct_latent", "max_steps": 5}
    s_dir, df_dir, w_dir = run_scenario(agent, env, train_ds, config, sc_dir, seed=0)
    assert s_dir["mode"] == "direct_latent" and len(w_dir) == 1

    sc_cw = {"name": "test_cw", "start_xy": [0.0, 0.0], "target_xy": [8.0, 8.0], "mode": "custom_waypoints", "custom_waypoints": [[0.0, 4.0], [8.0, 8.0]], "max_steps": 5}
    s_cw, df_cw, w_cw = run_scenario(agent, env, train_ds, config, sc_cw, seed=0)
    assert s_cw["mode"] == "custom_waypoints" and len(w_cw) == 2


def test_plot_generation(loaded_agent_and_env):
    agent, env, train_ds, _, config = loaded_agent_and_env
    sc = {"name": "test_plot", "start_xy": [0.0, 0.0], "target_xy": [0.0, 3.0], "mode": "direct_latent", "max_steps": 5}
    summary, df_tel, waypoints = run_scenario(agent, env, train_ds, config, sc, seed=0)
    with tempfile.TemporaryDirectory() as tmp_dir:
        plot_path = os.path.join(tmp_dir, "test_plot.png")
        plot_scenario_results(summary, df_tel, waypoints, plot_path, maze_type="medium", target_radius=1.0)
        assert os.path.exists(plot_path) and os.path.getsize(plot_path) > 1000
