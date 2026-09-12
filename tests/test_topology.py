"""Unit tests for topological algorithms, graph search, and trajectory utilities."""

import pytest
import numpy as np
import jax.numpy as jnp
from src.agent_loader import load_pretrained_agent
from src.topology import (
    DijkstraGraph,
    backtrack_dijkstra_path,
    check_stuck_state,
    compute_continuous_lookahead,
    extract_corner_waypoints,
    extract_curvature_angles,
    find_connected_landmarks,
    jit_decode_latent_to_coords,
    march_along_path,
    project_agent_on_path,
    track_local_path_index,
)


@pytest.fixture(scope="module")
def setup_agent():
    return load_pretrained_agent("fb-test", "medium")


def test_backtrack_dijkstra_path():
    pred = np.full((5, 5), -9999, dtype=int)
    pred[0, 1] = 0
    pred[0, 2] = 1
    pred[0, 3] = 2
    path = backtrack_dijkstra_path(pred, 0, 3, 5)
    assert path == [0, 1, 2, 3]

    disconnected = backtrack_dijkstra_path(pred, 0, 4, 5)
    assert disconnected == [0, 4]


def test_extract_corner_waypoints():
    coords = [
        np.array([0.0, 0.0]),
        np.array([1.0, 0.0]),
        np.array([2.0, 0.0]),
        np.array([2.0, 1.0]),
        np.array([2.0, 2.0]),
    ]
    latents = [np.zeros(4) for _ in coords]
    filtered_c, filtered_l, idxs = extract_corner_waypoints(coords, latents, lookahead_dist=1.5)
    assert len(filtered_c) >= 3
    assert idxs[0] == 0
    assert idxs[-1] == 4


def test_track_local_path_index():
    coords = [np.array([float(i), 0.0]) for i in range(10)]
    best_idx, dist = track_local_path_index(np.array([3.1, 0.1]), coords, current_idx=2)
    assert best_idx == 3
    assert dist < 0.2


def test_extract_curvature_angles():
    coords = [
        np.array([0.0, 0.0]),
        np.array([1.0, 0.0]),
        np.array([1.0, 1.0]),
    ]
    curvs = extract_curvature_angles(coords)
    assert len(curvs) == 3
    assert abs(curvs[0]) < 1e-4
    assert curvs[-1] == 1.0


def test_check_stuck_state():
    history = []
    stuck_count = 0
    pos = np.array([1.0, 1.0])
    for _ in range(45):
        is_stuck, stuck_count = check_stuck_state(history, pos, dist_to_final=5.0, stuck_count=stuck_count)
    assert is_stuck is True
    assert stuck_count > 0


def test_compute_continuous_lookahead():
    coords = [np.array([0.0, float(i)]) for i in range(5)]
    lookahead_pt, seg_idx, s_agent = compute_continuous_lookahead(
        np.array([0.0, 0.5]), coords, lookahead_dist=1.5, current_seg=0
    )
    assert lookahead_pt[0] == 0.0
    assert abs(lookahead_pt[1] - 2.0) < 0.1
    assert seg_idx >= 0


def test_dijkstra_graph(setup_agent):
    agent, env, train_ds, _, _ = setup_agent
    graph = DijkstraGraph(agent, train_ds["observations"][:100], n_landmarks=20)
    obs = train_ds["observations"][0]
    goal_z = np.asarray(
        agent.normalize_z(agent.network.select("backward_repr")(train_ds["observations"][10:11]))[0]
    )
    coords, latents, states = graph.plan_path(obs[:2], goal_z)
    assert len(coords) >= 1
    assert len(latents) == len(coords)
    assert len(states) == len(coords)

    subgoal_z = graph.get_subgoal_latent(obs, goal_z)
    assert subgoal_z.shape == (128,)

    sg_z, sg_coord = graph.get_subgoal_and_coord(obs, goal_z)
    assert sg_z.shape == (128,)
    assert sg_coord.shape == (2,)
