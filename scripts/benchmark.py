"""Benchmark runner for StagedAgent architectures with SQLite telemetry logging."""

import argparse
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.agent import (
    create_baseline_agent,
    create_dijkstra_teacher_agent,
    create_direct_intention_agent,
    create_sequence_attention_agent,
    create_single_waypoint_agent,
)
from src.agent_loader import load_pretrained_agent
from src.evaluator import ZeroShotEvaluator
from src.telemetry.db import TelemetryDatabase


def _find_checkpoint(name: str, dirs: List[str]) -> Optional[str]:
    """Search directories for model checkpoint file."""
    for d in dirs:
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    return None


def _load_env_config(split: str) -> Any:
    """Load environment configuration YAML or return default parameters."""
    cfg_file = os.path.join(PROJECT_ROOT, "configs", "env", f"antmaze_{split}.yaml")
    if os.path.exists(cfg_file):
        try:
            return OmegaConf.load(cfg_file)
        except Exception:
            pass
    return OmegaConf.create({
        "name": f"antmaze-{split}-navigate-v0",
        "split": split,
        "max_episode_steps": 1500 if split in ["large", "giant"] else 1000,
        "planner": {
            "n_landmarks": 2000 if split == "large" else 1000,
            "lookahead_dist": 2.6,
            "max_edge_radius": 3.5,
            "reachability_cutoff": 35.0,
            "single_wp": {"hidden_dim": 384, "n_layers": 4},
            "sequence_attn": {"hidden_dim": 384, "num_heads": 6, "n_layers": 4},
        },
    })


def _resolve_checkpoints(split: str, dirs: List[str]) -> Tuple[str, str, str]:
    """Resolve required checkpoint paths across known candidate directories."""
    seq_ckpt = (
        _find_checkpoint(f"best_sequence_attention_{split}.pkl", dirs)
        or _find_checkpoint(f"best_enhanced_sequence_attn_{split}.pkl", dirs)
        or _find_checkpoint(f"best_enhanced_seq_attn_{split}.pkl", dirs)
    )
    sw_ckpt = (
        _find_checkpoint(f"best_single_waypoint_{split}.pkl", dirs)
        or _find_checkpoint(f"best_single_wp_{split}.pkl", dirs)
    )
    di_ckpt = (
        _find_checkpoint(f"distilled_gated_attn_{split}.pkl", dirs)
        or _find_checkpoint(f"distilled_jax_gated_attn_{split}.pkl", dirs)
    )
    if not seq_ckpt or not sw_ckpt or not di_ckpt:
        raise FileNotFoundError(
            f"Missing checkpoints in {dirs} for split {split}! seq={seq_ckpt}, sw={sw_ckpt}, di={di_ckpt}"
        )
    return seq_ckpt, sw_ckpt, di_ckpt


def _init_candidate_agents(
    agent: Any, train_obs: np.ndarray, split: str, env_cfg: Any = None
) -> List[Any]:
    """Instantiate all benchmark StagedAgent variants with configured stages."""
    dirs = [
        os.path.join(PROJECT_ROOT, "results", "checkpoints"),
        os.path.join(PROJECT_ROOT, "outputs", "checkpoints"),
        PROJECT_ROOT,
    ]
    p_cfg = getattr(env_cfg, "planner", {})
    n_landmarks = int(p_cfg.get("n_landmarks", 2000 if split == "large" else 1000))
    lookahead = float(p_cfg.get("lookahead_dist", 2.6))
    max_radius = float(p_cfg.get("max_edge_radius", 3.5))
    reach_cutoff = float(p_cfg.get("reachability_cutoff", 35.0))

    sw_cfg = p_cfg.get("single_wp", {})
    sa_cfg = p_cfg.get("sequence_attn", p_cfg.get("enhanced_seq_attn", {}))
    seq_ckpt, single_wp_ckpt, direct_ckpt = _resolve_checkpoints(split, dirs)

    return [
        create_sequence_attention_agent(
            agent, train_obs, seq_ckpt, n_landmarks=n_landmarks,
            lookahead_dist=lookahead, max_edge_radius=max_radius,
            reachability_cutoff=reach_cutoff, hidden_dim=int(sa_cfg.get("hidden_dim", 384)),
            num_heads=int(sa_cfg.get("num_heads", 6)), n_layers=int(sa_cfg.get("n_layers", 4)),
            name="1. Dijkstra + Sequence Attention",
        ),
        create_baseline_agent(agent, name="2. Single-Intention Baseline"),
        create_single_waypoint_agent(
            agent, train_obs, single_wp_ckpt, n_landmarks=n_landmarks,
            lookahead_dist=lookahead, max_edge_radius=max_radius,
            reachability_cutoff=reach_cutoff, hidden_dim=int(sw_cfg.get("hidden_dim", 384)),
            n_layers=int(sw_cfg.get("n_layers", 4)), name="3. Dijkstra + Single Waypoint Translator",
        ),
        create_dijkstra_teacher_agent(
            agent, train_obs, n_landmarks=n_landmarks, lookahead_dist=lookahead,
            max_edge_radius=max_radius, reachability_cutoff=reach_cutoff,
            name="4. Dijkstra Teacher (high_actor)",
        ),
        create_direct_intention_agent(
            agent, checkpoint_path=direct_ckpt, model_type="gated_attn",
            name="5. Direct Intention Planner [O(1)]",
        ),
    ]


def _warmup_agents(agents: List[Any], obs: np.ndarray, goal_latent: np.ndarray) -> None:
    """Perform dry-run step on each agent to trigger JIT compilation before timing."""
    for ag in agents:
        if hasattr(ag, "warmup"):
            ag.warmup(obs, goal_latent)


def _eval_agent_seed(
    evaluator: ZeroShotEvaluator,
    agent: Any,
    s: int,
    num_tasks: int,
    ep_per_task: int,
    run_id: str,
) -> Tuple[Dict[int, float], float]:
    """Evaluate an agent on one seed across all tasks and record task success."""
    task_sr: Dict[int, float] = {}
    latencies: List[float] = []
    for t_id in range(1, num_tasks + 1):
        stats, _, _, _ = evaluator.evaluate_task(
            agent, t_id, ep_per_task, seed=s, run_id=run_id
        )
        task_sr[t_id] = float(stats.get("success", 0.0)) * 100.0
        latencies.append(float(stats.get("latency_ms", 0.0)))
    mean_lat = float(np.mean(latencies)) if latencies else 0.0
    return task_sr, mean_lat


def _eval_single_agent(
    evaluator: ZeroShotEvaluator,
    agent: Any,
    seeds: List[int],
    num_tasks: int,
    ep_per_task: int,
    db: Optional[TelemetryDatabase],
    exp_id: str,
    split: str,
) -> Dict[str, Any]:
    """Evaluate candidate agent across all seeds and aggregate task metrics."""
    print(f"\nEvaluating: >>> {agent.name} <<< across {len(seeds)} seeds: {seeds}")
    seed_task_sr, seed_overall_sr, seed_latencies = defaultdict(list), [], []
    method_slug = agent.name.split(". ")[-1].replace(" ", "_").lower()

    for s in seeds:
        run_id = f"{exp_id}_{method_slug}_{split}_seed{s}"
        if db is not None:
            db.insert_run(
                run_id=run_id, experiment_id=exp_id, method=agent.name,
                split=split, seed=s, config_json={"num_tasks": num_tasks, "ep_per_task": ep_per_task}
            )
            for st in agent.stages:
                if hasattr(st, "landmark_coords"):
                    db.insert_landmarks(getattr(st, "landmark_coords"), run_id=run_id)
                    break

        task_sr, mean_lat = _eval_agent_seed(evaluator, agent, s, num_tasks, ep_per_task, run_id)
        for t_id, sr in task_sr.items():
            seed_task_sr[t_id].append(sr)
        seed_latencies.append(mean_lat)
        overall_sr = float(np.mean(list(task_sr.values())))
        seed_overall_sr.append(overall_sr)
        task_str = " | ".join([f"T{t}:{task_sr[t]:.0f}%" for t in range(1, num_tasks + 1)])
        print(f"  Seed {s:02d}: Overall = {overall_sr:5.1f}% | {task_str}")

    entry: Dict[str, Any] = {
        "Method": agent.name,
        "Success Rate (%)": f"{np.mean(seed_overall_sr):.1f} ± {np.std(seed_overall_sr):.1f}",
        "Latency (ms)": f"{np.mean(seed_latencies):.2f}",
    }
    for t_id in range(1, num_tasks + 1):
        entry[f"Task {t_id:02d} (%)"] = f"{np.mean(seed_task_sr[t_id]):.1f}"
    return entry


def _save_summary_tables(
    rows: List[Dict[str, Any]], output_dir: str, split: str, n_seeds: int
) -> pd.DataFrame:
    """Export benchmark summary to CSV, Markdown, and formatted terminal output."""
    df = pd.DataFrame(rows)
    os.makedirs(output_dir, exist_ok=True)
    csv_p = os.path.join(output_dir, f"benchmark_summary_{n_seeds}seeds_{split}.csv")
    md_p = os.path.join(output_dir, f"benchmark_summary_{n_seeds}seeds_{split}.md")
    df.to_csv(csv_p, index=False)
    try:
        md_text = df.to_markdown(index=False)
    except Exception:
        md_text = df.to_string()
    with open(md_p, "w", encoding="utf-8") as f:
        f.write(md_text)
    sep = "=" * 90
    print(f"\n{sep}\n=== FINAL BENCHMARK SUMMARY TABLE ===\n{sep}")
    print(md_text)
    print(f"\nSaved summary to {csv_p} and {md_p}")
    return df


def run_comprehensive_benchmark(
    checkpoint_dir: str = "fb-test",
    split: str = "medium",
    num_tasks: int = 5,
    episodes_per_task: int = 10,
    seeds: Optional[List[int]] = None,
    output_dir: str = "results/benchmarks",
    methods: Optional[List[str]] = None,
    db_path: str = "results/data/telemetry.db",
    experiment: str = "benchmark",
    **overrides: Any,
) -> pd.DataFrame:
    """Execute multi-seed benchmark across agent architectures with telemetry recording."""
    seed_list = seeds or list(range(1, 11))
    env_cfg = _load_env_config(split)
    max_steps = overrides.get("max_episode_steps") or getattr(
        env_cfg, "max_episode_steps", 1500 if split in ["large", "giant"] else 1000
    )
    agent, env, train_ds, _, cfg = load_pretrained_agent(
        checkpoint_dir, split, seed=seed_list[0], max_episode_steps=max_steps
    )
    available = len(getattr(env.unwrapped, "task_infos", getattr(env, "task_infos", [])))
    tasks_count = min(num_tasks, available) if available > 0 else num_tasks

    db = TelemetryDatabase(db_path)
    exp_id = db.insert_experiment(name=experiment, notes=f"Split {split} benchmark")

    evaluator = ZeroShotEvaluator(
        env, agent, train_ds, cfg, env_name=f"ogbench-antmaze-{split}-navigate-v0",
        max_episode_steps=max_steps, db=db,
    )
    all_agents = _init_candidate_agents(agent, train_ds["observations"], split, env_cfg)
    agents = (
        [a for a in all_agents if any(m.lower() in a.name.lower() for m in methods)]
        if methods else all_agents
    )

    latent_sample = evaluator.get_inferred_latent(1, seed=seed_list[0])
    _warmup_agents(agents, train_ds["observations"][0], latent_sample)

    try:
        rows = []
        for ag in agents:
            entry = _eval_single_agent(
                evaluator, ag, seed_list, tasks_count, episodes_per_task, db, exp_id, split
            )
            rows.append(entry)
        return _save_summary_tables(rows, output_dir, split, len(seed_list))
    finally:
        db.close()


def main() -> None:
    """Parse CLI arguments and launch comprehensive benchmark evaluation."""
    parser = argparse.ArgumentParser(description="Benchmark StagedAgent architectures.")
    parser.add_argument("--split", type=str, default="medium")
    parser.add_argument("--num_tasks", type=int, default=5)
    parser.add_argument("--episodes_per_task", type=int, default=10)
    parser.add_argument("--start_seed", type=int, default=1)
    parser.add_argument("--end_seed", type=int, default=10)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--methods", type=str, nargs="+", default=None)
    parser.add_argument("--output_dir", type=str, default="results/benchmarks")
    parser.add_argument("--db_path", type=str, default="results/data/telemetry.db")
    parser.add_argument("--experiment", type=str, default="benchmark")
    args = parser.parse_args()

    seed_list = (
        args.seeds if args.seeds is not None
        else list(range(args.start_seed, args.end_seed + 1))
    )
    run_comprehensive_benchmark(
        split=args.split, num_tasks=args.num_tasks,
        episodes_per_task=args.episodes_per_task, seeds=seed_list,
        output_dir=args.output_dir, methods=args.methods,
        db_path=args.db_path, experiment=args.experiment,
    )


if __name__ == "__main__":
    main()
