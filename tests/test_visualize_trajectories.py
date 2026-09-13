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
    render_3panel_comparison,
    export_all_comparisons,
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


def test_task_coords_all_mazes():
    for m in ['medium', 'large', 'giant', 'teleport']:
        assert m in TASK_COORDS
        assert len(TASK_COORDS[m]) == 5
        for t_id in range(1, 6):
            t_info = get_task_info(t_id, maze_type=m)
            assert 'init' in t_info
            assert 'goal' in t_info


def test_normalize_maze_type():
    assert normalize_maze_type('antmaze-medium-navigate-v0') == 'medium'
    assert normalize_maze_type('antmaze-large-navigate-v0') == 'large'
    assert normalize_maze_type('antmaze-giant-navigate-v0') == 'giant'
    assert normalize_maze_type('antmaze-teleport-navigate-v0') == 'teleport'


def test_auto_detect_maze_type():
    df_med = pd.DataFrame({
        'method': ['A']*3, 'seed': [0]*3, 'task_id': [1]*3, 'episode': [0]*3, 'step': [0, 1, 2],
        'x': [0.0, 0.5, 1.0], 'y': [0.0, 0.5, 1.0]
    })
    assert auto_detect_maze_type(df_med) == 'medium'

    df_giant = pd.DataFrame({
        'method': ['A']*3, 'seed': [0]*3, 'task_id': [1]*3, 'episode': [0]*3, 'step': [0, 1, 2],
        'x': [48.0, 49.0, 50.0], 'y': [34.0, 34.5, 35.0]
    })
    assert auto_detect_maze_type(df_giant) == 'giant'

    df_large = pd.DataFrame({
        'method': ['A']*3, 'seed': [0]*3, 'task_id': [1]*3, 'episode': [0]*3, 'step': [0, 1, 2],
        'x': [28.0, 28.5, 29.0], 'y': [20.0, 20.2, 20.5]
    })
    assert auto_detect_maze_type(df_large) == 'large'

    df_teleport = pd.DataFrame({
        'method': ['A', 'A'], 'seed': [0, 0], 'task_id': [1, 1], 'episode': [0, 0], 'step': [0, 1],
        'x': [20.0, 24.0], 'y': [12.0, 0.0]
    })
    assert auto_detect_maze_type(df_teleport) == 'teleport'


def test_render_3panel_comparison_success_failed_split(tmp_path):
    out_dir = str(tmp_path / 'plots')

    df_succ = pd.DataFrame({
        'method': ['Buffer Graph Dijkstra (Branch 2)', 'Single-Intention Baseline'],
        'seed': [0, 0],
        'task_id': [1, 1],
        'episode': [0, 0],
        'step': [1, 1],
        'x': [20.0, 10.0],
        'y': [20.0, 10.0],
        'reward': [1.0, 0.0],
        'done': [True, True],
    })
    file_succ, outcome_succ = render_3panel_comparison(
        df_succ, seed=0, task_id=1, episode=0, maze_type='medium', output_dir=out_dir
    )
    assert outcome_succ == 'success'
    assert os.path.exists(file_succ)
    assert os.path.join(out_dir, 'medium', 'success') in file_succ

    df_fail = pd.DataFrame({
        'method': ['Buffer Graph Dijkstra (Branch 2)', 'Single-Intention Baseline'],
        'seed': [0, 0],
        'task_id': [1, 1],
        'episode': [1, 1],
        'step': [1, 1],
        'x': [5.0, 5.0],
        'y': [5.0, 5.0],
        'reward': [0.0, 0.0],
        'done': [True, True],
    })
    file_fail, outcome_fail = render_3panel_comparison(
        df_fail, seed=0, task_id=1, episode=1, maze_type='medium', output_dir=out_dir
    )
    assert outcome_fail == 'failed'
    assert os.path.exists(file_fail)
    assert os.path.join(out_dir, 'medium', 'failed') in file_fail


def test_export_all_comparisons(tmp_path):
    out_dir = str(tmp_path / 'plots')
    df_multi = pd.DataFrame({
        'method': [
            'Buffer Graph Dijkstra (Branch 2)', 'Buffer Graph Dijkstra (Branch 2)',
            'Single-Intention Baseline', 'Single-Intention Baseline'
        ],
        'seed': [0, 0, 0, 0],
        'task_id': [1, 1, 1, 1],
        'episode': [0, 1, 0, 1],
        'step': [1, 1, 1, 1],
        'x': [20.0, 5.0, 10.0, 5.0],
        'y': [20.0, 5.0, 10.0, 5.0],
        'reward': [1.0, 0.0, 0.0, 0.0],
        'done': [True, True, True, True],
    })
    counts = export_all_comparisons(df_multi, maze_type='large', output_dir=out_dir)
    assert counts['success'] == 1
    assert counts['failed'] == 1
    assert os.path.exists(os.path.join(out_dir, 'large', 'success'))
    assert os.path.exists(os.path.join(out_dir, 'large', 'failed'))
