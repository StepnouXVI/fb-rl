import pytest
from hydra import initialize, compose
from omegaconf import OmegaConf

def test_hydra_config_composition():
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name="config", overrides=["env=antmaze_medium", "planner=baseline"])
        assert cfg.env.split == "medium"
        assert cfg.planner.type == "baseline"
        assert cfg.eval.num_episodes > 0

def test_hydra_all_environments():
    for split in ["medium", "large", "giant", "teleport"]:
        with initialize(version_base=None, config_path="../configs"):
            cfg = compose(config_name="config", overrides=[f"env=antmaze_{split}"])
            assert cfg.env.split == split
