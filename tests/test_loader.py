import pytest
import src
from src.agent_loader import load_pretrained_agent

def test_load_pretrained_agent():
    agent, env, train_ds, val_ds, config = load_pretrained_agent("fb-test", "medium")
    assert agent is not None
    assert "modules_forward_repr" in agent.network.params
    assert "modules_backward_repr" in agent.network.params
    assert "modules_actor" in agent.network.params
    assert train_ds["observations"].shape[1] == 29
