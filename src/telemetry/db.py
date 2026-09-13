import datetime
import json
import sqlite3
from typing import Any, Dict, List, Optional
import uuid
import numpy as np

from src.telemetry.metrics import (
    compute_cross_track_errors,
    count_spatial_self_intersections,
)


def _serialize_json(val):
    """Serialize value to JSON string if it is a container or array."""
    if val is None:
        return None
    if isinstance(val, (dict, list, tuple)):
        return json.dumps(val)
    if hasattr(val, "tolist"):
        return json.dumps(val.tolist())
    return str(val)


def _to_float(val):
    """Safely convert value to float or return None."""
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _current_timestamp():
    """Return current UTC timestamp in ISO format."""
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _compute_episode_trajectory(ep_row, step_rows, final_obs_xy=None):
    """Reconstruct continuous 2D planar trajectory without duplicate start point and with endpoint."""
    pts = []
    if ep_row["start_x"] is not None and ep_row["start_y"] is not None:
        start_pt = [float(ep_row["start_x"]), float(ep_row["start_y"])]
        if not step_rows or np.hypot(start_pt[0] - float(step_rows[0]["x"]), start_pt[1] - float(step_rows[0]["y"])) > 1e-4:
            pts.append(start_pt)
    for s in step_rows:
        pts.append([float(s["x"]), float(s["y"])])
    if final_obs_xy is not None and len(final_obs_xy) >= 2:
        end_pt = [float(final_obs_xy[0]), float(final_obs_xy[1])]
        if not pts or np.hypot(end_pt[0] - pts[-1][0], end_pt[1] - pts[-1][1]) > 1e-4:
            pts.append(end_pt)
    return np.asarray(pts, dtype=np.float64) if pts else np.empty((0, 2), dtype=np.float64)


def _compute_episode_cte(path_rows, traj):
    """Compute mean and max cross track error against the latest planned path."""
    path_coords = None
    if path_rows:
        raw_c = path_rows[-1].get("path_coords_json")
        if isinstance(raw_c, str):
            try:
                path_coords = json.loads(raw_c)
            except Exception:
                path_coords = None
        elif isinstance(raw_c, list):
            path_coords = raw_c
    mean_cte, max_cte = 0.0, 0.0
    if path_coords and len(path_coords) >= 2 and len(traj) >= 2:
        cte_arr = compute_cross_track_errors(traj, np.asarray(path_coords, dtype=np.float64))
        if len(cte_arr) > 0:
            mean_cte = float(np.mean(cte_arr))
            max_cte = float(np.max(cte_arr))
    return path_coords, mean_cte, max_cte


class TelemetryDatabase:
    """SQLite telemetry database with WAL mode for high-performance experiment tracking."""

    def __init__(self, db_path: str):
        """Initialize connection, configure pragmas, and setup schema."""
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self._apply_pragmas()
        self._create_experiment_tables()
        self._create_episode_tables()
        self._create_step_tables()
        self._create_training_tables()
        self._create_indexes()

    def _apply_pragmas(self):
        """Apply performance and integrity pragmas."""
        cur = self.conn.cursor()
        cur.execute("PRAGMA journal_mode = WAL;")
        cur.execute("PRAGMA synchronous = NORMAL;")
        cur.execute("PRAGMA cache_size = -64000;")
        cur.execute("PRAGMA foreign_keys = ON;")
        cur.close()

    def _create_experiment_tables(self):
        """Create experiments, runs, and landmarks tables."""
        cur = self.conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS experiments (
                experiment_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                created_at TEXT NOT NULL,
                git_commit TEXT,
                notes TEXT
            );
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                experiment_id TEXT NOT NULL REFERENCES experiments(experiment_id) ON DELETE CASCADE,
                method TEXT NOT NULL,
                split TEXT NOT NULL,
                seed INTEGER NOT NULL,
                config_json TEXT,
                created_at TEXT NOT NULL
            );
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS landmarks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
                landmark_idx INTEGER NOT NULL,
                x REAL NOT NULL,
                y REAL NOT NULL
            );
        """)
        self.conn.commit()
        cur.close()

    def _create_episode_tables(self):
        """Create episodes and planned paths tables."""
        cur = self.conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS episodes (
                episode_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
                seed INTEGER NOT NULL,
                task_id INTEGER NOT NULL,
                episode_idx INTEGER NOT NULL,
                start_x REAL NOT NULL,
                start_y REAL NOT NULL,
                goal_x REAL,
                goal_y REAL,
                is_success INTEGER NOT NULL,
                total_steps INTEGER NOT NULL,
                total_reward REAL NOT NULL,
                mean_speed REAL NOT NULL,
                self_intersections INTEGER NOT NULL,
                mean_cross_track_error REAL NOT NULL,
                max_cross_track_error REAL NOT NULL,
                mean_latency_ms REAL NOT NULL,
                created_at TEXT NOT NULL
            );
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS planned_paths (
                path_id TEXT PRIMARY KEY,
                episode_id TEXT NOT NULL REFERENCES episodes(episode_id) ON DELETE CASCADE,
                replan_step INTEGER NOT NULL,
                is_initial INTEGER NOT NULL,
                waypoints_json TEXT NOT NULL,
                path_coords_json TEXT NOT NULL
            );
        """)
        self.conn.commit()
        cur.close()

    def _create_step_tables(self):
        """Create step telemetry table."""
        cur = self.conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS steps (
                step_id TEXT PRIMARY KEY,
                episode_id TEXT NOT NULL REFERENCES episodes(episode_id) ON DELETE CASCADE,
                step_idx INTEGER NOT NULL,
                x REAL NOT NULL,
                y REAL NOT NULL,
                z REAL NOT NULL,
                vx REAL NOT NULL,
                vy REAL NOT NULL,
                speed REAL NOT NULL,
                action_norm REAL NOT NULL,
                action_torques TEXT,
                lookahead_x REAL,
                lookahead_y REAL,
                dist_to_lookahead REAL,
                attention_targets TEXT,
                attention_weights TEXT,
                is_direct_goal INTEGER NOT NULL,
                reward REAL NOT NULL,
                done INTEGER NOT NULL,
                latency_ms REAL NOT NULL
            );
        """)
        self.conn.commit()
        cur.close()

    def _create_training_tables(self):
        """Create training runs and training epochs tables."""
        cur = self.conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS training_runs (
                run_id TEXT PRIMARY KEY,
                run_name TEXT NOT NULL,
                model_type TEXT NOT NULL,
                split TEXT NOT NULL,
                seed INTEGER NOT NULL,
                config_json TEXT,
                best_checkpoint_path TEXT,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                completed_at TEXT
            );
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS training_epochs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL REFERENCES training_runs(run_id) ON DELETE CASCADE,
                epoch INTEGER NOT NULL,
                loss REAL NOT NULL,
                loss_bc REAL,
                loss_action REAL,
                loss_reach REAL,
                loss_goal REAL,
                loss_aux REAL,
                cos_sim REAL,
                val_loss REAL,
                val_cos_sim REAL,
                learning_rate REAL,
                epoch_time_s REAL,
                checkpoint_path TEXT,
                created_at TEXT NOT NULL
            );
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_train_epochs_run ON training_epochs(run_id, epoch);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_train_runs_model ON training_runs(model_type, split);")
        self.conn.commit()
        cur.close()

    def _create_indexes(self):
        """Create indexes for instant queries on foreign keys and filters."""
        cur = self.conn.cursor()
        cur.execute("CREATE INDEX IF NOT EXISTS idx_runs_exp ON runs(experiment_id);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_landmarks_run ON landmarks(run_id, landmark_idx);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_episodes_run ON episodes(run_id);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_episodes_success ON episodes(is_success);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_episodes_run_success ON episodes(run_id, is_success);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_planned_paths_ep ON planned_paths(episode_id, replan_step);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_steps_ep_idx ON steps(episode_id, step_idx);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_train_epochs_run ON training_epochs(run_id, epoch);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_train_runs_model ON training_runs(model_type, split);")
        self.conn.commit()
        cur.close()

    def insert_experiment(self, experiment_id: str = None, name: str = "default_experiment", git_commit: str = None, notes: str = None, created_at: str = None) -> str:
        """Insert an experiment record into the experiments table and return id."""
        exp_id = experiment_id or str(uuid.uuid4())
        ts = created_at or _current_timestamp()
        with self.conn:
            self.conn.execute(
                """
                INSERT OR REPLACE INTO experiments (experiment_id, name, created_at, git_commit, notes)
                VALUES (?, ?, ?, ?, ?)
                """,
                (exp_id, name, ts, git_commit, notes),
            )
        return exp_id

    def insert_run(self, run_id: str, experiment_id: str, method: str, split: str, seed: int, config_json = None, created_at: str = None, **kwargs):
        """Insert a run record into the runs table."""
        ts = created_at or _current_timestamp()
        cfg = config_json if config_json is not None else kwargs.get("config")
        cfg_str = _serialize_json(cfg)
        with self.conn:
            self.conn.execute(
                """
                INSERT OR REPLACE INTO runs (run_id, experiment_id, method, split, seed, config_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (run_id, experiment_id, method, split, seed, cfg_str, ts),
            )

    def insert_landmarks(self, landmarks, run_id: str = None):
        """Batch insert landmarks for a run from array or dicts."""
        if isinstance(landmarks, str) and not isinstance(run_id, str):
            run_id, landmarks = landmarks, run_id
        rows = []
        if isinstance(landmarks, (list, tuple)) and len(landmarks) > 0 and isinstance(landmarks[0], dict):
            for i, lm in enumerate(landmarks):
                r_id = lm.get("run_id", run_id)
                idx = lm.get("landmark_idx", i)
                rows.append((r_id, idx, float(lm["x"]), float(lm["y"])))
        else:
            for i, pt in enumerate(landmarks):
                rows.append((run_id, i, float(pt[0]), float(pt[1])))
        with self.conn:
            self.conn.executemany(
                """
                INSERT INTO landmarks (run_id, landmark_idx, x, y)
                VALUES (?, ?, ?, ?)
                """,
                rows,
            )

    def insert_episode(self, episode_data: dict = None, **kwargs):
        """Insert a single episode record into the episodes table."""
        data = dict(episode_data or {})
        data.update(kwargs)
        ts = data.get("created_at") or _current_timestamp()
        vals = (
            data["episode_id"],
            data["run_id"],
            int(data["seed"]),
            int(data["task_id"]),
            int(data["episode_idx"]),
            float(data["start_x"]),
            float(data["start_y"]),
            float(data["goal_x"]) if data.get("goal_x") is not None else None,
            float(data["goal_y"]) if data.get("goal_y") is not None else None,
            int(bool(data["is_success"])),
            int(data["total_steps"]),
            float(data["total_reward"]),
            float(data["mean_speed"]),
            int(data["self_intersections"]),
            float(data["mean_cross_track_error"]),
            float(data["max_cross_track_error"]),
            float(data["mean_latency_ms"]),
            ts,
        )
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO episodes (
                    episode_id, run_id, seed, task_id, episode_idx,
                    start_x, start_y, goal_x, goal_y, is_success,
                    total_steps, total_reward, mean_speed, self_intersections,
                    mean_cross_track_error, max_cross_track_error, mean_latency_ms, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(episode_id) DO UPDATE SET
                    goal_x = excluded.goal_x,
                    goal_y = excluded.goal_y,
                    is_success = excluded.is_success,
                    total_steps = excluded.total_steps,
                    total_reward = excluded.total_reward,
                    mean_speed = excluded.mean_speed,
                    self_intersections = excluded.self_intersections,
                    mean_cross_track_error = excluded.mean_cross_track_error,
                    max_cross_track_error = excluded.max_cross_track_error,
                    mean_latency_ms = excluded.mean_latency_ms
                """,
                vals,
            )

    def insert_planned_path(self, path_data: dict = None, **kwargs):
        """Insert a planned path with waypoints and continuous coordinates."""
        data = dict(path_data or {})
        data.update(kwargs)
        pid = data.get("path_id") or str(uuid.uuid4())
        wps = data.get("waypoints_json", data.get("waypoints", []))
        coords = data.get("path_coords_json", data.get("path_coords", []))
        vals = (
            pid,
            data["episode_id"],
            int(data["replan_step"]),
            int(bool(data.get("is_initial", False))),
            _serialize_json(wps),
            _serialize_json(coords),
        )
        with self.conn:
            self.conn.execute(
                """
                INSERT OR REPLACE INTO planned_paths (
                    path_id, episode_id, replan_step, is_initial,
                    waypoints_json, path_coords_json
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                vals,
            )

    def _prepare_step_row(self, s, idx, ep_id):
        """Format a single step dictionary into a tuple of column values."""
        step_id = s.get("step_id") or f"{ep_id}_{s.get('step_idx', idx)}"
        lh_x = float(s["lookahead_x"]) if s.get("lookahead_x") is not None else None
        lh_y = float(s["lookahead_y"]) if s.get("lookahead_y") is not None else None
        dist_lh = float(s["dist_to_lookahead"]) if s.get("dist_to_lookahead") is not None else None
        return (
            str(step_id),
            str(s.get("episode_id", ep_id)),
            int(s.get("step_idx", idx)),
            float(s.get("x") or 0.0),
            float(s.get("y") or 0.0),
            float(s.get("z") or 0.0),
            float(s.get("vx") or 0.0),
            float(s.get("vy") or 0.0),
            float(s.get("speed") or 0.0),
            float(s.get("action_norm") or 0.0),
            _serialize_json(s.get("action_torques")),
            lh_x,
            lh_y,
            dist_lh,
            _serialize_json(s.get("attention_targets")),
            _serialize_json(s.get("attention_weights")),
            int(bool(s.get("is_direct_goal", 0))),
            float(s.get("reward") or 0.0),
            int(bool(s.get("done", 0))),
            float(s.get("latency_ms") or 0.0),
        )

    def insert_steps_batch(self, steps: list, episode_id: str = None):
        """Batch insert step telemetry rows for an episode."""
        if isinstance(steps, str) and isinstance(episode_id, list):
            steps, episode_id = episode_id, steps
        if not steps:
            return
        rows = [self._prepare_step_row(s, i, episode_id) for i, s in enumerate(steps)]
        with self.conn:
            self.conn.executemany(
                """
                INSERT OR REPLACE INTO steps (
                    step_id, episode_id, step_idx, x, y, z, vx, vy, speed,
                    action_norm, action_torques, lookahead_x, lookahead_y,
                    dist_to_lookahead, attention_targets, attention_weights,
                    is_direct_goal, reward, done, latency_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )

    def get_failed_episodes(self, run_id: str = None, experiment_id: str = None):
        """Query failed episodes with optional run or experiment filter."""
        cur = self.conn.cursor()
        if run_id is not None:
            query = "SELECT * FROM episodes WHERE is_success = 0 AND run_id = ? ORDER BY episode_idx ASC"
            cur.execute(query, (run_id,))
        elif experiment_id is not None:
            query = """
                SELECT e.* FROM episodes e
                JOIN runs r ON e.run_id = r.run_id
                WHERE e.is_success = 0 AND r.experiment_id = ?
                ORDER BY e.created_at ASC
            """
            cur.execute(query, (experiment_id,))
        else:
            query = "SELECT * FROM episodes WHERE is_success = 0 ORDER BY created_at ASC"
            cur.execute(query)
        results = [dict(row) for row in cur.fetchall()]
        cur.close()
        return results

    def get_episode_telemetry(self, episode_id: str):
        """Query complete episode telemetry including ordered steps and planned paths."""
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM episodes WHERE episode_id = ?", (episode_id,))
        ep_row = cur.fetchone()
        if ep_row is None:
            cur.close()
            return None
        cur.execute("SELECT * FROM steps WHERE episode_id = ? ORDER BY step_idx ASC", (episode_id,))
        steps_rows = [dict(r) for r in cur.fetchall()]
        cur.execute("SELECT * FROM planned_paths WHERE episode_id = ? ORDER BY replan_step ASC", (episode_id,))
        paths_rows = [dict(r) for r in cur.fetchall()]
        cur.close()
        return {
            "episode": dict(ep_row),
            "steps": steps_rows,
            "planned_paths": paths_rows,
        }

    def _execute_episode_update(self, ep_id, vals):
        """Execute SQLite update for finalized episode record."""
        with self.conn:
            self.conn.execute(
                """
                UPDATE episodes SET
                    goal_x = ?, goal_y = ?, is_success = ?, total_steps = ?,
                    total_reward = ?, mean_speed = ?, self_intersections = ?,
                    mean_cross_track_error = ?, max_cross_track_error = ?,
                    mean_latency_ms = ?
                WHERE episode_id = ?
                """,
                (*vals, ep_id),
            )

    def finalize_episode(
        self,
        episode_id: str,
        target_radius: float = 1.0,
        final_obs_xy: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Aggregate step and path telemetry, compute episode metrics, and update DB."""
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM episodes WHERE episode_id = ?", (episode_id,))
        ep_row = cur.fetchone()
        if ep_row is None:
            cur.close()
            return {}
        cur.execute("SELECT * FROM steps WHERE episode_id = ? ORDER BY step_idx ASC", (episode_id,))
        step_rows = [dict(r) for r in cur.fetchall()]
        cur.execute("SELECT * FROM planned_paths WHERE episode_id = ? ORDER BY replan_step ASC", (episode_id,))
        path_rows = [dict(r) for r in cur.fetchall()]
        cur.close()

        total_steps = len(step_rows)
        total_reward = float(sum(s["reward"] for s in step_rows))
        mean_lat = float(np.mean([s["latency_ms"] for s in step_rows])) if step_rows else 0.0
        traj = _compute_episode_trajectory(ep_row, step_rows, final_obs_xy=final_obs_xy)
        mean_spd = (
            float(np.mean(np.linalg.norm(np.diff(traj, axis=0), axis=1)))
            if len(traj) > 1
            else (float(np.mean([s["speed"] for s in step_rows])) if step_rows else 0.0)
        )
        self_int = count_spatial_self_intersections(traj) if len(traj) >= 4 else 0
        path_coords, mean_cte, max_cte = _compute_episode_cte(path_rows, traj)

        raw_gx, raw_gy = ep_row["goal_x"], ep_row["goal_y"]
        goal_x = float(raw_gx) if raw_gx is not None else None
        goal_y = float(raw_gy) if raw_gy is not None else None
        if (goal_x is None or goal_y is None) and path_coords and len(path_coords) > 0:
            goal_x, goal_y = float(path_coords[-1][0]), float(path_coords[-1][1])
        has_goal = goal_x is not None and goal_y is not None
        dist_final = (
            float(np.hypot(traj[-1, 0] - goal_x, traj[-1, 1] - goal_y))
            if len(traj) > 0 and has_goal
            else float("inf")
        )
        is_succ = bool(
            total_steps > 0
            and (
                any(s["reward"] > 0.0 for s in step_rows)
                or (has_goal and dist_final <= target_radius)
            )
        )

        self._execute_episode_update(episode_id, (goal_x, goal_y, int(is_succ), total_steps, total_reward, mean_spd, self_int, mean_cte, max_cte, mean_lat))
        return {
            "episode_id": episode_id, "goal_x": goal_x, "goal_y": goal_y, "is_success": is_succ,
            "total_steps": total_steps, "total_reward": total_reward, "mean_speed": mean_spd,
            "self_intersections": self_int, "mean_cross_track_error": mean_cte,
            "max_cross_track_error": max_cte, "mean_latency_ms": mean_lat,
        }

    def create_training_run(
        self,
        run_name: str,
        model_type: str,
        split: str,
        seed: int,
        config: Any = None,
        run_id: str = None,
        best_checkpoint_path: str = None,
    ) -> str:
        """Insert a training run record into training_runs table and return run_id."""
        rid = run_id or str(uuid.uuid4())
        ts = _current_timestamp()
        cfg_str = _serialize_json(config)
        with self.conn:
            self.conn.execute(
                """
                INSERT OR REPLACE INTO training_runs (
                    run_id, run_name, model_type, split, seed,
                    config_json, best_checkpoint_path, status,
                    created_at, completed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    rid,
                    run_name,
                    model_type,
                    split,
                    int(seed),
                    cfg_str,
                    best_checkpoint_path,
                    "running",
                    ts,
                    None,
                ),
            )
        return rid

    def insert_training_epoch(
        self,
        run_id: str,
        epoch: int,
        metrics: Dict[str, Any],
        checkpoint_path: Optional[str] = None,
        epoch_time_s: Optional[float] = None,
        learning_rate: Optional[float] = None,
    ) -> None:
        """Insert a training epoch telemetry record into training_epochs table."""
        loss_val = _to_float(metrics.get("loss"))
        loss = 0.0 if loss_val is None else loss_val
        lr = learning_rate if learning_rate is not None else metrics.get("learning_rate")
        lr = lr if lr is not None else metrics.get("lr")
        time_s = epoch_time_s if epoch_time_s is not None else metrics.get("epoch_time_s")
        time_s = time_s if time_s is not None else metrics.get("time")
        ckpt = checkpoint_path if checkpoint_path is not None else metrics.get("checkpoint_path")
        vals = (
            run_id,
            int(epoch),
            loss,
            _to_float(metrics.get("loss_bc")),
            _to_float(metrics.get("loss_action")),
            _to_float(metrics.get("loss_reach")),
            _to_float(metrics.get("loss_goal")),
            _to_float(metrics.get("loss_aux")),
            _to_float(metrics.get("cos_sim")),
            _to_float(metrics.get("val_loss")),
            _to_float(metrics.get("val_cos_sim")),
            _to_float(lr),
            _to_float(time_s),
            str(ckpt) if ckpt is not None else None,
            _current_timestamp(),
        )
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO training_epochs (
                    run_id, epoch, loss, loss_bc, loss_action, loss_reach,
                    loss_goal, loss_aux, cos_sim, val_loss, val_cos_sim,
                    learning_rate, epoch_time_s, checkpoint_path, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                vals,
            )

    def finish_training_run(
        self,
        run_id: str,
        best_checkpoint_path: Optional[str] = None,
        status: str = "completed",
    ) -> None:
        """Update training run status, completion timestamp, and best checkpoint."""
        ts = _current_timestamp()
        with self.conn:
            self.conn.execute(
                """
                UPDATE training_runs
                SET status = ?,
                    completed_at = ?,
                    best_checkpoint_path = COALESCE(?, best_checkpoint_path)
                WHERE run_id = ?
                """,
                (status, ts, best_checkpoint_path, run_id),
            )

    def get_training_runs(self) -> List[Dict[str, Any]]:
        """Query all training runs ordered by creation time descending."""
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM training_runs ORDER BY created_at DESC")
        runs = [dict(row) for row in cur.fetchall()]
        cur.close()
        return runs

    def get_training_history(self, run_id: str) -> List[Dict[str, Any]]:
        """Query epoch history for a specific training run ordered by epoch ascending."""
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM training_epochs WHERE run_id = ? ORDER BY epoch ASC", (run_id,))
        epochs = [dict(row) for row in cur.fetchall()]
        cur.close()
        return epochs

    def close(self):
        """Close SQLite database connection."""
        self.conn.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
