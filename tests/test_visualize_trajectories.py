import os
import pytest
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scripts.visualize_trajectories import (
    MAZE_LAYOUTS,
    TELEPORT_INFO,
    TASK_COORDS,
    normalize_maze_type,
    auto_detect_maze_type,
    get_task_info,
    draw_maze,
    plot_path_with_teleports,
    extract_executed_waypoints,
    plot_trajectory_panel,
    plot_landmarks_panel,
    plot_multi_episode_panel,
    plot_overview_grid,
)


def test_maze_layouts_shapes():
    assert MAZE_LAYOUTS['medium'].shape == (8, 8)
    assert MAZE_LAYOUTS['large'].shape == (9, 12)
    assert MAZE_LAYOUTS['giant'].shape == (12, 16)
    assert MAZE_LAYOUTS['teleport'].shape == (9, 12)


def test_teleport_info_structure():
    assert 'teleport' in TELEPORT_INFO
    t_info = TELEPORT_INFO['teleport']
    assert len(t_info['teleport_in_xys']) == 2
    assert len(t_info['teleport_out_xys']) == 3
    assert (20.0, 12.0) in t_info['teleport_in_xys']
    assert (0.0, 16.0) in t_info['teleport_in_xys']
    assert (24.0, 0.0) in t_info['teleport_out_xys']
    assert (0.0, 20.0) in t_info['teleport_out_xys']
    assert (36.0, 20.0) in t_info['teleport_out_xys']


def test_task_coords_all_mazes():
    for m in ['medium', 'large', 'giant', 'teleport']:
        assert m in TASK_COORDS
        assert len(TASK_COORDS[m]) == 5
        for t_id in range(1, 6):
            t_info = get_task_info(t_id, maze_type=m)
            assert 'init' in t_info
            assert 'goal' in t_info
            assert 'name' in t_info


def test_normalize_maze_type():
    assert normalize_maze_type('antmaze-medium-navigate-v0') == 'medium'
    assert normalize_maze_type('antmaze-large-navigate-v0') == 'large'
    assert normalize_maze_type('antmaze-giant-navigate-v0') == 'giant'
    assert normalize_maze_type('antmaze-teleport-navigate-v0') == 'teleport'
    assert normalize_maze_type('LARGE') == 'large'
    assert normalize_maze_type('unknown_custom') == 'medium'


def test_auto_detect_maze_type():
    # Medium: max_coord <= 22.0
    df_med = pd.DataFrame({
        'method': ['A']*3, 'seed': [0]*3, 'task_id': [1]*3, 'episode': [0]*3, 'step': [0, 1, 2],
        'x': [0.0, 0.5, 1.0], 'y': [0.0, 0.5, 1.0]
    })
    assert auto_detect_maze_type(df_med) == 'medium'

    # Giant: max_coord > 38.0
    df_giant = pd.DataFrame({
        'method': ['A']*3, 'seed': [0]*3, 'task_id': [1]*3, 'episode': [0]*3, 'step': [0, 1, 2],
        'x': [48.0, 49.0, 50.0], 'y': [34.0, 34.5, 35.0]
    })
    assert auto_detect_maze_type(df_giant) == 'giant'

    # Large: 22 < max_coord <= 38.0 and smooth continuous trajectory (no teleports)
    df_large = pd.DataFrame({
        'method': ['A']*3, 'seed': [0]*3, 'task_id': [1]*3, 'episode': [0]*3, 'step': [0, 1, 2],
        'x': [28.0, 28.5, 29.0], 'y': [20.0, 20.2, 20.5]
    })
    assert auto_detect_maze_type(df_large) == 'large'

    # Teleport: 22 < max_coord <= 38.0 and discontinuous jump > 6.0
    df_teleport = pd.DataFrame({
        'method': ['A', 'A'], 'seed': [0, 0], 'task_id': [1, 1], 'episode': [0, 0], 'step': [0, 1],
        'x': [20.0, 24.0], 'y': [12.0, 0.0]
    })
    assert auto_detect_maze_type(df_teleport) == 'teleport'

    # Explicit split column
    df_col = pd.DataFrame({'split': ['teleport'], 'x': [0.0], 'y': [0.0]})
    assert auto_detect_maze_type(df_col) == 'teleport'


def test_draw_maze_all_types():
    for m in ['medium', 'large', 'giant', 'teleport']:
        fig, ax = plt.subplots()
        draw_maze(ax, maze_type=m, show_portals=True, show_portal_links=True)
        assert len(ax.patches) > 0
        plt.close(fig)


def test_draw_maze_custom_grid():
    custom_grid = np.array([
        [1, 1, 1, 1, 1],
        [1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1],
    ])
    fig, ax = plt.subplots()
    draw_maze(ax, custom_grid=custom_grid)
    assert len(ax.patches) == 12  # 5 + 2 + 5 = 12 wall cells
    plt.close(fig)


def test_plot_path_with_teleports():
    fig, ax = plt.subplots()
    xs = np.array([0.0, 1.0, 2.0, 20.0, 21.0])  # jump from 2.0 to 20.0
    ys = np.array([0.0, 1.0, 2.0, 15.0, 16.0])
    plot_path_with_teleports(ax, xs, ys, jump_threshold=5.0)
    # Should have 2 line segments and 1 FancyArrowPatch jump connector
    assert len(ax.lines) == 2
    assert len(ax.patches) == 1
    plt.close(fig)


def test_rendering_panels_and_grid(tmp_path):
    # Synthetic dataframe for testing end-to-end panel and grid plotting
    df_traj = pd.DataFrame({
        'method': ['Single-Intention Baseline', 'Single-Intention Baseline', 'Buffer Graph Dijkstra (Branch 2)', 'Buffer Graph Dijkstra (Branch 2)'],
        'seed': [0, 0, 0, 0],
        'task_id': [1, 1, 1, 1],
        'episode': [0, 0, 0, 0],
        'step': [0, 1, 0, 1],
        'x': [0.0, 20.0, 0.0, 20.0],
        'y': [0.0, 12.0, 0.0, 12.0],
        'reward': [0.0, 1.0, 0.0, 1.0],
        'done': [False, True, False, True],
    })
    df_sg = pd.DataFrame({
        'method': ['Buffer Graph Dijkstra (Branch 2)', 'Buffer Graph Dijkstra (Branch 2)'],
        'seed': [0, 0],
        'task_id': [1, 1],
        'episode': [0, 0],
        'step': [0, 1],
        'subgoal_x': [10.0, 20.0],
        'subgoal_y': [6.0, 12.0],
        'is_direct_goal': [False, True],
        'planned_waypoints': ['[]', '[]'],
    })
    df_lm = pd.DataFrame({
        'landmark_id': [0, 1],
        'x': [5.0, 15.0],
        'y': [5.0, 15.0],
    })

    # Test single panel on teleport maze
    fig, ax = plt.subplots()
    plot_trajectory_panel(ax, df_traj[df_traj['method'] == 'Single-Intention Baseline'], df_sg, title='Test Panel', maze_type='teleport')
    plt.close(fig)

    # Test landmarks panel
    fig, ax = plt.subplots()
    plot_landmarks_panel(ax, df_lm, df_sg, task_id=1, maze_type='large')
    plt.close(fig)

    # Test multi-episode panel
    fig, ax = plt.subplots()
    plot_multi_episode_panel(ax, df_traj, df_sg, title='Test Multi', maze_type='giant')
    plt.close(fig)

    # Test overview grid
    save_path = str(tmp_path / 'overview_test.png')
    plot_overview_grid(df_traj, df_sg, seed=0, maze_type='teleport', save_path=save_path)
    assert os.path.exists(save_path)
