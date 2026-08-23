import os, sys, json, pickle, flax, numpy as np

for p in [
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "baseline_repo")),
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
]:
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)

from utils.env_utils import make_env_and_datasets
from utils.datasets import HGCDataset
from agents.fbpiswitch import FBpiSwitchAgent, get_config

def load_pretrained_agent(checkpoint_dir="fb-test", split="medium", seed=0, max_episode_steps=None):
    cfg_path = os.path.join(checkpoint_dir, split, "flags.json")
    params_path = os.path.join(checkpoint_dir, split, "params.pkl")
    if not os.path.exists(cfg_path) or not os.path.exists(params_path):
        raise FileNotFoundError(f"Checkpoint files not found at {checkpoint_dir}/{split}")

    with open(cfg_path, "r") as f:
        saved_flags = json.load(f)

    config = get_config()
    config.update(saved_flags["agent"])
    env_name = saved_flags.get("env_name", f"ogbench-antmaze-{split}-navigate-v0")

    env, train_dataset, val_dataset = make_env_and_datasets(
        env_name, frame_stack=config["frame_stack"], add_info=True
    )
    env.unwrapped._add_noise_to_goal = False

    if max_episode_steps is not None:
        curr = env
        while curr is not None:
            if hasattr(curr, "_max_episode_steps"):
                curr._max_episode_steps = int(max_episode_steps)
            curr = getattr(curr, "env", None)

    hgc_dataset = HGCDataset(train_dataset, config)
    ex_batch = hgc_dataset.sample(1)
    agent = FBpiSwitchAgent.create(seed, ex_batch, config)

    with open(params_path, "rb") as f:
        data = pickle.load(f)
    agent = flax.serialization.from_state_dict(agent, data["agent"])

    return agent, env, train_dataset, val_dataset, config
