import json
import os
import sqlite3
import sys
import time
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.telemetry.db import TelemetryDatabase
from src.telemetry.metrics import (
    compute_cross_track_errors,
    count_spatial_self_intersections,
)
from src.telemetry.profiler import ExecutionProfiler


def test_sqlite_pragmas_and_tables(tmp_path):
    """Verify database initialization, pragmas, tables, and indexes."""
    db_file = str(tmp_path / "telemetry.db")
    with TelemetryDatabase(db_file) as db:
        cur = db.conn.cursor()
        cur.execute("PRAGMA foreign_keys;")
        assert cur.fetchone()[0] == 1
        cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = {row[0] for row in cur.fetchall()}
        required = {
            "experiments", "runs", "landmarks", "episodes", "planned_paths", "steps",
            "training_runs", "training_epochs",
        }
        assert required.issubset(tables)
        cur.execute("SELECT name FROM sqlite_master WHERE type='index';")
        indexes = {row[0] for row in cur.fetchall()}
        assert "idx_runs_exp" in indexes
        assert "idx_steps_ep_idx" in indexes
        assert "idx_train_epochs_run" in indexes
        assert "idx_train_runs_model" in indexes
        cur.close()


def test_insert_and_cascade_delete(tmp_path):
    """Verify entity insertions and foreign key cascade deletion."""
    db_file = str(tmp_path / "telemetry_cascade.db")
    with TelemetryDatabase(db_file) as db:
        db.insert_experiment("exp1", "Experiment 1")
        db.insert_run("run1", "exp1", "dijkstra", "test", 42, {"lr": 0.01})
        db.insert_landmarks([[0.0, 1.0], [2.0, 3.0]], run_id="run1")
        ep_data = {
            "episode_id": "ep1", "run_id": "run1", "seed": 42, "task_id": 0, "episode_idx": 0,
            "start_x": 0.0, "start_y": 0.0, "goal_x": 10.0, "goal_y": 10.0, "is_success": 1,
            "total_steps": 10, "total_reward": 5.0, "mean_speed": 1.2, "self_intersections": 0,
            "mean_cross_track_error": 0.1, "max_cross_track_error": 0.3, "mean_latency_ms": 2.5,
        }
        db.insert_episode(ep_data)
        db.insert_planned_path({
            "path_id": "p1", "episode_id": "ep1", "replan_step": 0, "is_initial": True,
            "waypoints_json": [[0, 0], [10, 10]], "path_coords_json": [[0, 0], [10, 10]],
        })
        db.insert_steps_batch([{
            "step_id": "s1", "episode_id": "ep1", "step_idx": 0, "x": 0.1, "y": 0.1, "z": 0.5,
            "vx": 0.1, "vy": 0.1, "speed": 0.14, "action_norm": 0.5, "action_torques": [0.1, 0.2],
            "lookahead_x": 1.0, "lookahead_y": 1.0, "dist_to_lookahead": 1.2,
            "attention_targets": [0, 1], "attention_weights": [0.8, 0.2],
            "is_direct_goal": 0, "reward": 0.5, "done": 0, "latency_ms": 3.0,
        }])
        cur = db.conn.cursor()
        cur.execute("DELETE FROM experiments WHERE experiment_id = 'exp1';")
        db.conn.commit()
        for tbl in ("runs", "landmarks", "episodes", "planned_paths", "steps"):
            cur.execute(f"SELECT COUNT(*) FROM {tbl};")
            assert cur.fetchone()[0] == 0
        cur.close()


def test_get_failed_episodes(tmp_path):
    """Verify querying failed episodes across runs and experiments."""
    db_file = str(tmp_path / "failed.db")
    with TelemetryDatabase(db_file) as db:
        db.insert_experiment("exp_fail", "Fail Exp")
        db.insert_run("run_a", "exp_fail", "method_a", "train", 1)
        db.insert_run("run_b", "exp_fail", "method_b", "train", 2)
        base = {
            "seed": 1, "task_id": 0, "start_x": 0, "start_y": 0, "goal_x": 1, "goal_y": 1,
            "total_steps": 1, "total_reward": 0, "mean_speed": 0, "self_intersections": 0,
            "mean_cross_track_error": 0, "max_cross_track_error": 0, "mean_latency_ms": 0,
        }
        db.insert_episode(dict(base, episode_id="e1", run_id="run_a", episode_idx=0, is_success=1))
        db.insert_episode(dict(base, episode_id="e2", run_id="run_a", episode_idx=1, is_success=0))
        db.insert_episode(dict(base, episode_id="e3", run_id="run_b", episode_idx=0, is_success=0))
        all_failed = db.get_failed_episodes()
        assert len(all_failed) == 2
        run_a_failed = db.get_failed_episodes(run_id="run_a")
        assert len(run_a_failed) == 1
        assert run_a_failed[0]["episode_id"] == "e2"
        exp_failed = db.get_failed_episodes(experiment_id="exp_fail")
        assert len(exp_failed) == 2


def test_get_episode_telemetry(tmp_path):
    """Verify retrieval of complete ordered episode telemetry payload."""
    db_file = str(tmp_path / "telemetry_payload.db")
    with TelemetryDatabase(db_file) as db:
        db.insert_experiment("exp_payload", "Payload")
        db.insert_run("r1", "exp_payload", "method", "test", 100)
        ep = {
            "episode_id": "ep_test", "run_id": "r1", "seed": 100, "task_id": 1, "episode_idx": 5,
            "start_x": 0, "start_y": 0, "goal_x": 5, "goal_y": 5, "is_success": 1,
            "total_steps": 2, "total_reward": 10, "mean_speed": 1.5, "self_intersections": 0,
            "mean_cross_track_error": 0.05, "max_cross_track_error": 0.1, "mean_latency_ms": 1.8,
        }
        db.insert_episode(ep)
        db.insert_planned_path({
            "path_id": "path1", "episode_id": "ep_test", "replan_step": 0, "is_initial": True,
            "waypoints_json": [[0, 0], [5, 5]], "path_coords_json": [[0, 0], [5, 5]],
        })
        db.insert_steps_batch([
            {"step_id": "step1", "step_idx": 1, "x": 1.0, "y": 1.0, "speed": 1.4},
            {"step_id": "step0", "step_idx": 0, "x": 0.0, "y": 0.0, "speed": 0.0},
        ], episode_id="ep_test")
        payload = db.get_episode_telemetry("ep_test")
        assert payload is not None
        assert payload["episode"]["episode_id"] == "ep_test"
        assert len(payload["steps"]) == 2
        assert payload["steps"][0]["step_idx"] == 0
        assert payload["steps"][1]["step_idx"] == 1
        assert len(payload["planned_paths"]) == 1
        assert db.get_episode_telemetry("non_existent") is None


def test_finalize_episode(tmp_path):
    """Verify finalize_episode aggregates steps, computes metrics, and updates DB."""
    db_file = str(tmp_path / "finalize_test.db")
    with TelemetryDatabase(db_file) as db:
        db.insert_experiment("exp_fin", "Finalize Exp")
        db.insert_run("run_fin", "exp_fin", "method", "test", 1)
        ep = {
            "episode_id": "ep_fin", "run_id": "run_fin", "seed": 1, "task_id": 1, "episode_idx": 0,
            "start_x": 0.0, "start_y": 0.0, "goal_x": 2.0, "goal_y": 0.0, "is_success": 0,
            "total_steps": 0, "total_reward": 0.0, "mean_speed": 0.0, "self_intersections": 0,
            "mean_cross_track_error": 0.0, "max_cross_track_error": 0.0, "mean_latency_ms": 0.0,
        }
        db.insert_episode(ep)
        db.insert_planned_path({
            "path_id": "path_fin", "episode_id": "ep_fin", "replan_step": 0, "is_initial": True,
            "waypoints_json": [[0, 0], [2, 0]], "path_coords_json": [[0, 0], [2, 0]],
        })
        db.insert_steps_batch([
            {"step_id": "s0", "step_idx": 1, "x": 1.0, "y": 0.5, "speed": 1.0, "reward": 0.0, "latency_ms": 2.0},
            {"step_id": "s1", "step_idx": 2, "x": 2.0, "y": 0.1, "speed": 1.0, "reward": 1.0, "latency_ms": 3.0},
        ], episode_id="ep_fin")

        res = db.finalize_episode("ep_fin", target_radius=1.0)
        assert res["is_success"] is True
        assert res["total_steps"] == 2
        assert np.isclose(res["total_reward"], 1.0)
        assert res["mean_speed"] > 0.0
        assert res["mean_cross_track_error"] > 0.0
        assert np.isclose(res["mean_latency_ms"], 2.5)

        ep_after = db.get_episode_telemetry("ep_fin")["episode"]
        assert ep_after["is_success"] == 1
        assert ep_after["total_steps"] == 2
        assert np.isclose(ep_after["total_reward"], 1.0)


def test_cross_track_errors_straight_line():
    """Verify cross-track error along a straight line segment."""
    path = np.array([[0.0, 0.0], [10.0, 0.0]])
    traj_on = np.array([[0.0, 0.0], [5.0, 0.0], [10.0, 0.0]])
    errors_on = compute_cross_track_errors(traj_on, path)
    assert np.allclose(errors_on, 0.0, atol=1e-6)
    traj_offset = np.array([[0.0, 2.0], [5.0, 2.0], [10.0, 2.0]])
    errors_offset = compute_cross_track_errors(traj_offset, path)
    assert np.allclose(errors_offset, 2.0, atol=1e-6)
    assert len(compute_cross_track_errors([], path)) == 0
    single_pt_path = np.array([[1.0, 1.0]])
    pt_dist = compute_cross_track_errors(np.array([[1.0, 4.0]]), single_pt_path)
    assert np.isclose(pt_dist[0], 3.0)


def test_cross_track_errors_anti_wall_penetration():
    """Verify local window avoids snapping across walls in U-turn corridor."""
    path = np.array([[0.0, 0.0], [0.0, 10.0], [1.0, 10.0], [1.0, 0.0]])
    traj = np.array([[0.0, 2.0], [0.0, 5.0], [0.6, 5.0]])
    errors = compute_cross_track_errors(traj, path, window_size=2)
    assert np.isclose(errors[2], 0.6, atol=1e-5)


def test_spatial_self_intersections():
    """Verify CCW predicate and AABB pruning on trajectories."""
    straight = np.array([[0.0, float(i)] for i in range(20)])
    assert count_spatial_self_intersections(straight) == 0
    l_path = np.array([[0.0, float(i)] for i in range(10)] + [[float(i), 10.0] for i in range(1, 10)])
    assert count_spatial_self_intersections(l_path) == 0
    fig8 = np.array([[0.0, 0.0], [2.0, 2.0], [2.0, 0.0], [0.0, 2.0]])
    assert count_spatial_self_intersections(fig8, min_step_dist=0.1) == 1
    t = np.linspace(0, 2.5 * np.pi, 50)
    circle = np.stack([np.cos(t), np.sin(t)], axis=-1)
    assert count_spatial_self_intersections(circle, min_step_dist=0.1) >= 1
    jitter = np.array([[0.0, 0.0], [0.05, 0.02], [0.0, 0.05], [0.02, 0.0]])
    assert count_spatial_self_intersections(jitter, min_step_dist=0.35) == 0


def test_execution_profiler_timing_and_sync():
    """Verify inference latency measurement and asynchronous sync barriers."""
    profiler = ExecutionProfiler()
    with profiler:
        time.sleep(0.01)
    assert profiler.elapsed_ms >= 5.0
    assert profiler.elapsed_sec > 0.0
    assert profiler.call_count == 1
    mock_tensor = MagicMock()
    with profiler:
        profiler.sync({"tensor": mock_tensor})
    mock_tensor.block_until_ready.assert_called_once()
    assert profiler.call_count == 2
    assert profiler.mean_ms > 0.0
    profiler.reset()
    assert profiler.call_count == 0
    assert profiler.total_ms == 0.0


def test_evaluator_telemetry_integration(tmp_path):
    """Verify end-to-end telemetry recording during an evaluation episode rollout."""
    from src.agent import create_baseline_agent
    from src.agent_loader import load_pretrained_agent
    from src.evaluator import ZeroShotEvaluator

    db_path = str(tmp_path / "integration_telemetry.db")
    db = TelemetryDatabase(db_path)
    exp_id = db.insert_experiment(name="eval_test", notes="integration test")
    run_id = f"{exp_id}_baseline_medium_seed0"
    db.insert_run(run_id=run_id, experiment_id=exp_id, method="Baseline", split="medium", seed=0)

    agent_model, env, train_ds, _, cfg = load_pretrained_agent("fb-test", "medium", seed=0)
    agent = create_baseline_agent(agent_model, name="TestBaseline")

    evaluator = ZeroShotEvaluator(
        env, agent_model, train_ds, cfg, max_episode_steps=5, db=db, run_id=run_id
    )
    stats, trajs, _, _ = evaluator.evaluate_task(
        agent, task_id=1, num_episodes=1, eval_temperature=0.0, seed=0, run_id=run_id
    )

    with db.conn:
        cur = db.conn.cursor()
        cur.execute("SELECT COUNT(*) FROM episodes WHERE run_id = ?", (run_id,))
        assert cur.fetchone()[0] == 1
        cur.execute("SELECT COUNT(*) FROM steps")
        assert cur.fetchone()[0] >= 1
    db.close()


def test_metrics_db_aggregation(tmp_path):
    """Verify metrics calculation and LaTeX generation from SQLite database."""
    from src.metrics import aggregate_db_runs, aggregate_runs, bootstrap_ci, export_latex_table

    db_path = str(tmp_path / "metrics_test.db")
    db = TelemetryDatabase(db_path)
    exp_id = db.insert_experiment(name="metrics_exp")
    run1 = f"{exp_id}_baseline_0"
    run2 = f"{exp_id}_seq_0"
    db.insert_run(run_id=run1, experiment_id=exp_id, method="Single-Intention Baseline", split="medium", seed=1)
    db.insert_run(run_id=run2, experiment_id=exp_id, method="Dijkstra Sequence Attention", split="medium", seed=1)

    for i in range(5):
        db.insert_episode({
            "episode_id": f"ep1_{i}", "run_id": run1, "seed": 1, "task_id": 1, "episode_idx": i,
            "start_x": 0.0, "start_y": 0.0, "goal_x": 5.0, "goal_y": 5.0,
            "is_success": 1 if i < 3 else 0, "total_steps": 100 + i * 10, "total_reward": 1.0,
            "mean_speed": 0.5, "self_intersections": 0, "mean_cross_track_error": 0.2,
            "max_cross_track_error": 0.5, "mean_latency_ms": 1.2,
        })
        db.insert_episode({
            "episode_id": f"ep2_{i}", "run_id": run2, "seed": 1, "task_id": 1, "episode_idx": i,
            "start_x": 0.0, "start_y": 0.0, "goal_x": 5.0, "goal_y": 5.0,
            "is_success": 1, "total_steps": 80 + i * 5, "total_reward": 1.0,
            "mean_speed": 0.6, "self_intersections": 0, "mean_cross_track_error": 0.1,
            "max_cross_track_error": 0.3, "mean_latency_ms": 0.8,
        })

    summary = aggregate_db_runs(db, experiment_id=exp_id)
    assert "Single-Intention Baseline" in summary
    assert "Dijkstra Sequence Attention" in summary
    assert summary["Dijkstra Sequence Attention"]["success_rate"]["mean"] == 100.0
    assert summary["Single-Intention Baseline"]["success_rate"]["mean"] == 60.0

    low, high = bootstrap_ci([10.0, 20.0, 30.0, 40.0])
    assert low <= high

    latex = export_latex_table(summary)
    assert r"\begin{table}" in latex
    assert "Dijkstra Sequence Attention" in latex
    db.close()


def test_training_run_creation(tmp_path):
    """Verify creating training runs with auto-generated and explicit IDs."""
    db_file = str(tmp_path / "train_run.db")
    with TelemetryDatabase(db_file) as db:
        auto_id = db.create_training_run(
            run_name="auto_run",
            model_type="single_wp",
            split="medium",
            seed=0,
        )
        assert isinstance(auto_id, str) and len(auto_id) > 0
        cfg = {"lr": 1e-4, "batch_size": 64}
        custom_id = db.create_training_run(
            run_name="seq_attn_run",
            model_type="sequence_attention",
            split="large",
            seed=42,
            config=cfg,
            run_id="custom_train_1",
            best_checkpoint_path="/checkpoints/init.ckpt",
        )
        assert custom_id == "custom_train_1"
        runs = db.get_training_runs()
        assert len(runs) == 2
        run_dict = {r["run_id"]: r for r in runs}
        assert "custom_train_1" in run_dict
        entry = run_dict["custom_train_1"]
        assert entry["run_name"] == "seq_attn_run"
        assert entry["model_type"] == "sequence_attention"
        assert entry["split"] == "large"
        assert entry["seed"] == 42
        assert entry["status"] == "running"
        assert entry["best_checkpoint_path"] == "/checkpoints/init.ckpt"
        assert entry["completed_at"] is None
        assert json.loads(entry["config_json"]) == cfg


def test_training_epoch_logging_and_history(tmp_path):
    """Verify logging epoch metrics and retrieving chronological history."""
    db_file = str(tmp_path / "train_epochs.db")
    with TelemetryDatabase(db_file) as db:
        run_id = db.create_training_run(
            run_name="train_loop",
            model_type="gated_attn",
            split="medium",
            seed=123,
        )
        m0 = {
            "loss": 1.25,
            "loss_bc": 0.8,
            "loss_action": 0.3,
            "loss_reach": -0.1,
            "loss_goal": -0.05,
            "loss_aux": 0.05,
            "cos_sim": 0.65,
            "val_loss": 1.40,
            "val_cos_sim": 0.60,
        }
        db.insert_training_epoch(
            run_id=run_id,
            epoch=0,
            metrics=m0,
            checkpoint_path="/ckpts/ep0.pt",
            epoch_time_s=15.2,
            learning_rate=1e-3,
        )
        m1 = {
            "loss": 0.75,
            "loss_bc": 0.4,
            "loss_action": 0.2,
            "cos_sim": 0.85,
            "val_loss": 0.80,
            "val_cos_sim": 0.82,
            "learning_rate": 5e-4,
            "epoch_time_s": 14.8,
        }
        db.insert_training_epoch(
            run_id=run_id,
            epoch=1,
            metrics=m1,
            checkpoint_path="/ckpts/ep1.pt",
        )
        history = db.get_training_history(run_id)
        assert len(history) == 2
        assert history[0]["epoch"] == 0
        assert np.isclose(history[0]["loss"], 1.25)
        assert np.isclose(history[0]["learning_rate"], 1e-3)
        assert np.isclose(history[0]["epoch_time_s"], 15.2)
        assert history[0]["checkpoint_path"] == "/ckpts/ep0.pt"
        assert history[1]["epoch"] == 1
        assert np.isclose(history[1]["loss"], 0.75)
        assert np.isclose(history[1]["learning_rate"], 5e-4)
        assert np.isclose(history[1]["epoch_time_s"], 14.8)
        assert history[1]["checkpoint_path"] == "/ckpts/ep1.pt"


def test_finish_training_run_and_status(tmp_path):
    """Verify completing and failing training runs with best checkpoints."""
    db_file = str(tmp_path / "train_finish.db")
    with TelemetryDatabase(db_file) as db:
        r1 = db.create_training_run("r1", "sequence_attention", "medium", 1)
        r2 = db.create_training_run("r2", "single_wp", "medium", 2)
        db.finish_training_run(r1, best_checkpoint_path="/ckpts/best_r1.pt", status="completed")
        db.finish_training_run(r2, status="failed")
        runs = {r["run_id"]: r for r in db.get_training_runs()}
        assert runs[r1]["status"] == "completed"
        assert runs[r1]["completed_at"] is not None
        assert runs[r1]["best_checkpoint_path"] == "/ckpts/best_r1.pt"
        assert runs[r2]["status"] == "failed"
        assert runs[r2]["completed_at"] is not None


def test_training_cascade_deletion(tmp_path):
    """Verify foreign key cascade deletion for training runs and epochs."""
    db_file = str(tmp_path / "train_cascade.db")
    with TelemetryDatabase(db_file) as db:
        rid = db.create_training_run("run_del", "single_wp", "medium", 7)
        db.insert_training_epoch(rid, 0, {"loss": 0.5}, learning_rate=1e-3)
        db.insert_training_epoch(rid, 1, {"loss": 0.3}, learning_rate=1e-3)
        assert len(db.get_training_history(rid)) == 2
        cur = db.conn.cursor()
        cur.execute("DELETE FROM training_runs WHERE run_id = ?", (rid,))
        db.conn.commit()
        cur.execute("SELECT COUNT(*) FROM training_epochs WHERE run_id = ?", (rid,))
        assert cur.fetchone()[0] == 0
        cur.close()


def test_aim_fast_patch():
    """Verify runtime batching patch applies cleanly."""
    from src.telemetry.aim_fast import is_patched, patch
    patch(batch_size=25)
    assert is_patched() is True


def test_aim_tracker_lifecycle_and_metrics(tmp_path):
    """Verify AimTracker initialization, param logging, tags, and scalar tracking."""
    from src.telemetry import AimTracker
    aim_dir = str(tmp_path / "aim_test_repo")
    with AimTracker(repo=aim_dir, experiment="unit_test_exp", run_name="unit_run") as tracker:
        assert tracker.hash is not None
        tracker.set_params({"lr": 1e-3, "hidden_dim": 128})
        tracker.add_tags(["unit_test", "quick"])
        tracker.track(2.5, name="loss", step=0, epoch=0)
        tracker.track(0.75, name="loss", step=1, epoch=1)


def test_plotly_figure_builders_and_aim_figure(tmp_path):
    """Verify Plotly figure constructors and tracking inside AimTracker."""
    from src.telemetry import (
        AimTracker,
        build_pareto_figure,
        build_task_breakdown_figure,
        build_trajectory_figure,
    )
    traj = [{"x": 0.0, "y": 0.0, "attention_targets": "[[1.0, 1.0]]", "attention_weights": "[1.0]"}]
    path = [[0.0, 0.0], [2.0, 2.0]]
    fig_traj = build_trajectory_figure("Sequence Attention", path_coords=path, traj_steps=traj)
    fig_pareto = build_pareto_figure(["m1", "m2"], [1.0, 1.5], [80.0, 85.0], [2.0, 3.0])
    fig_task = build_task_breakdown_figure(["m1"], {"m1": [80.0, 85.0, 90.0, 75.0, 95.0]})
    assert fig_traj is not None
    assert fig_pareto is not None
    assert fig_task is not None

    aim_dir = str(tmp_path / "aim_fig_repo")
    with AimTracker(repo=aim_dir, experiment="fig_test_exp") as tracker:
        tracker.track_figure(fig_traj, name="trajectory_map", step=0)
        tracker.track_figure(fig_pareto, name="pareto", step=0)
