"""
PPO / DQN / baseline karşılaştırması, episode logları ve öğrenme eğrileri.

  python evaluate_agents.py
  python evaluate_agents.py --episodes 50 --skip-gifs
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import GRID_MAX_EPISODE_STEPS, RANDOM_SEED
from environment import GridParkingNavigationEnv, build_grid_nav_episode_configs
from evaluation.baselines import (
    bfs_first_action,
    greedy_manhattan_step,
    random_action,
    rollout_episode_return,
)
from paths import (
    DATA_PROCESSED,
    LOGS_DIR,
    MODELS_DIR,
    OUTPUT_DIR,
    PREDICTIONS_DIR,
    ensure_output,
)
from visualize_agent import draw_metrics_overlay, save_explained_gif_sb3


def _vecnormalize_path(algo: str) -> Optional[Path]:
    for p in (
        MODELS_DIR / f"vecnormalize_{algo}.pkl",
        MODELS_DIR / f"vecnormalize_{algo}",
    ):
        if p.exists():
            return p
    return None


def _moving_average(x: np.ndarray, window: int) -> np.ndarray:
    if len(x) == 0:
        return x
    w = max(1, int(window))
    return pd.Series(x).rolling(w, min_periods=1).mean().to_numpy(dtype=float)


def _load_monitor_rewards(algo: str) -> Optional[np.ndarray]:
    p = LOGS_DIR / "monitor" / algo / "mon_0.monitor.csv"
    if not p.exists():
        return None
    rows = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("r,l,t"):
                continue
            parts = line.split(",")
            if len(parts) >= 1:
                try:
                    rows.append(float(parts[0]))
                except ValueError:
                    continue
    if not rows:
        return None
    return np.array(rows, dtype=float)


def plot_learning_curves() -> None:
    """Monitor CSV'lerinden öğrenme eğrileri (`output/`)."""
    ensure_output()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    colors = {"ppo": "#2563eb", "dqn": "#dc2626"}
    for algo, color in colors.items():
        rewards = _load_monitor_rewards(algo)
        if rewards is None:
            continue
        x = np.arange(1, len(rewards) + 1)
        ma = _moving_average(rewards, window=max(10, len(rewards) // 50))
        success_proxy = (rewards > 0).astype(float)
        success_ma = _moving_average(success_proxy, window=max(20, len(rewards) // 40))
        axes[0, 0].plot(x, rewards, alpha=0.25, color=color)
        axes[0, 0].plot(x, ma, label=algo.upper(), color=color, linewidth=1.5)
        axes[0, 1].plot(x, ma, label=f"{algo.upper()} MA reward", color=color)
        axes[1, 0].plot(x, success_ma * 100.0, label=f"{algo.upper()} success proxy %", color=color)
    axes[0, 0].set_title("Episode reward")
    axes[0, 0].set_xlabel("Episode")
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].set_title("Moving average reward")
    axes[0, 1].set_xlabel("Episode")
    axes[0, 1].legend(fontsize=8)
    axes[1, 0].set_title("Success rate trend (proxy: reward>0)")
    axes[1, 0].set_ylabel("%")
    axes[1, 0].set_xlabel("Episode")
    axes[1, 0].legend(fontsize=8)

    ppo_r = _load_monitor_rewards("ppo")
    dqn_r = _load_monitor_rewards("dqn")
    if ppo_r is not None and dqn_r is not None:
        n = min(len(ppo_r), len(dqn_r))
        axes[1, 1].plot(
            _moving_average(ppo_r[:n], 30),
            label="PPO",
            color=colors["ppo"],
        )
        axes[1, 1].plot(
            _moving_average(dqn_r[:n], 30),
            label="DQN",
            color=colors["dqn"],
        )
        axes[1, 1].set_title("PPO vs DQN — moving average reward")
        axes[1, 1].set_xlabel("Episode (aligned)")
        axes[1, 1].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "reward_trend.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Ayrı başarı oranı grafiği
    plt.figure(figsize=(10, 4))
    for algo, color in colors.items():
        rewards = _load_monitor_rewards(algo)
        if rewards is None:
            continue
        x = np.arange(1, len(rewards) + 1)
        plt.plot(
            _moving_average((rewards > 0).astype(float), 40) * 100.0,
            label=algo.upper(),
            color=color,
        )
    plt.title("Success rate trend (proxy)")
    plt.xlabel("Episode")
    plt.ylabel("Success %")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "success_rate_trend.png", dpi=150, bbox_inches="tight")
    plt.close()


def _rollout_baseline_episode(
    episode_configs: list,
    action_fn: Callable[[GridParkingNavigationEnv], int],
    seed: int,
    *,
    scenario: str = "medium",
) -> Dict[str, Any]:
    env = GridParkingNavigationEnv(
        episode_configs,
        seed=RANDOM_SEED,
        max_episode_steps=GRID_MAX_EPISODE_STEPS,
        scenario=scenario,
    )
    env.reset(seed=seed)
    total_reward = 0.0
    done = False
    last_info: Dict[str, Any] = {}
    while not done:
        a = action_fn(env)
        _obs, r, term, trunc, info = env.step(a)
        total_reward += float(r)
        done = bool(term or trunc)
        last_info = info
    success = bool(last_info.get("success"))
    steps = int(last_info.get("step", last_info.get("steps", 0)))
    return {
        "episode_id": seed,
        "algorithm": action_fn.__name__ if hasattr(action_fn, "__name__") else "baseline",
        "start_position": last_info.get("start_position", []),
        "target_position": last_info.get("target_position", []),
        "total_reward": float(total_reward),
        "total_steps": steps,
        "success": success,
        "path_length": int(last_info.get("path_length", max(0, steps))),
        "invalid_moves": int(last_info.get("invalid_moves", 0)),
        "collision_count": int(last_info.get("collision_count", 0)),
        "visited_cells": int(last_info.get("visited_cells", 0)),
        "final_distance_to_goal": int(
            last_info.get("final_distance_to_goal", last_info.get("distance_to_goal", 0))
        ),
        "time_to_goal_steps": steps if success else None,
    }


def _rollout_sb3_episode(
    episode_configs: list,
    algo: str,
    seed: int,
    *,
    scenario: str = "medium",
) -> Dict[str, Any]:
    from stable_baselines3 import DQN, PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    vec_p = _vecnormalize_path(algo)
    if vec_p is None:
        raise FileNotFoundError(f"Model/VecNormalize yok: {algo}")

    def factory():
        return GridParkingNavigationEnv(
            episode_configs,
            seed=RANDOM_SEED,
            max_episode_steps=GRID_MAX_EPISODE_STEPS,
            scenario=scenario,
        )

    venv = DummyVecEnv([factory])
    vec = VecNormalize.load(str(vec_p), venv)
    vec.training = False
    vec.norm_reward = False
    load_path = str(MODELS_DIR / f"{algo}_agent")
    model = (
        PPO.load(load_path, env=vec, custom_objects={"lr_schedule": lambda _: 3e-4})
        if algo == "ppo"
        else DQN.load(load_path, env=vec)
    )
    try:
        vec.seed(int(seed))
        obs = vec.reset()
        total = 0.0
        done = False
        last_info: Dict[str, Any] = {}
        while not done:
            act, _ = model.predict(obs, deterministic=True)
            obs, r, dones, infos = vec.step(act)
            total += float(r[0])
            done = bool(dones[0])
            if isinstance(infos, (list, tuple)) and infos:
                last_info = dict(infos[0])
        success = bool(last_info.get("success"))
        steps = int(last_info.get("step", last_info.get("steps", 0)))
        return {
            "episode_id": seed,
            "algorithm": algo,
            "start_position": last_info.get("start_position", []),
            "target_position": last_info.get("target_position", []),
            "total_reward": float(total),
            "total_steps": steps,
            "success": success,
            "path_length": int(last_info.get("path_length", max(0, steps))),
            "invalid_moves": int(last_info.get("invalid_moves", 0)),
            "collision_count": int(last_info.get("collision_count", 0)),
            "visited_cells": int(last_info.get("visited_cells", 0)),
            "final_distance_to_goal": int(
                last_info.get("final_distance_to_goal", last_info.get("distance_to_goal", 0))
            ),
            "time_to_goal_steps": steps if success else None,
        }
    finally:
        vec.close()


def _aggregate_metrics(rows: List[Dict[str, Any]], label: str) -> Dict[str, Any]:
    if not rows:
        return {"algorithm": label}
    df = pd.DataFrame(rows)
    succ = df["success"].astype(bool)
    t_goal = df.loc[succ, "time_to_goal_steps"].dropna()
    return {
        "algorithm": label,
        "mean_reward": float(df["total_reward"].mean()),
        "mean_steps": float(df["total_steps"].mean()),
        "success_rate": float(succ.mean()),
        "mean_path_length": float(df["path_length"].mean()),
        "mean_invalid_moves": float(df["invalid_moves"].mean()),
        "mean_collision_count": float(df["collision_count"].mean()),
        "mean_time_to_goal_steps": float(t_goal.mean()) if len(t_goal) else None,
        "episodes": int(len(df)),
    }


def _save_comparison_charts(metrics_df: pd.DataFrame) -> None:
    ensure_output()
    metrics_df.to_csv(OUTPUT_DIR / "model_comparison_metrics.csv", index=False)

    algos = metrics_df["algorithm"].tolist()
    x = np.arange(len(algos))
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    specs = [
        ("mean_reward", "Mean reward", axes[0, 0]),
        ("mean_steps", "Mean steps", axes[0, 1]),
        ("success_rate", "Success rate", axes[0, 2]),
        ("mean_path_length", "Mean path length", axes[1, 0]),
        ("mean_collision_count", "Collisions", axes[1, 1]),
        ("mean_invalid_moves", "Invalid moves", axes[1, 2]),
    ]
    for col, title, ax in specs:
        if col not in metrics_df.columns:
            continue
        vals = metrics_df[col].astype(float).fillna(0.0).tolist()
        ax.bar(x, vals, color=plt.cm.Set2(np.linspace(0, 1, len(algos))))
        ax.set_xticks(x)
        ax.set_xticklabels(algos, rotation=20, ha="right", fontsize=8)
        ax.set_title(title)
    plt.suptitle("Agent comparison", y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "ppo_vs_dqn_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.bar(x, metrics_df["mean_path_length"], color=plt.cm.Pastel1(np.linspace(0, 1, len(algos))))
    plt.xticks(x, algos, rotation=15, ha="right")
    plt.ylabel("Mean path length")
    plt.title("Path length comparison")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "path_length_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()


def _write_episode_logs(rows: List[Dict[str, Any]]) -> None:
    path = OUTPUT_DIR / "rl_episode_logs.csv"
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def run_evaluation(
    n_episodes: int,
    scenario: str,
    skip_gifs: bool,
    gif_seed: int,
) -> None:
    ensure_output()
    pred = PREDICTIONS_DIR / "test_predictions.csv"
    cfgs = build_grid_nav_episode_configs(
        DATA_PROCESSED / "processed.parquet",
        pred if pred.exists() else None,
        split="test",
        base_seed=RANDOM_SEED,
    )

    policy_fns = {
        "random": random_action,
        "greedy_shortest_path": greedy_manhattan_step,
        "bfs_oracle": bfs_first_action,
    }
    all_logs: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []

    for name, fn in policy_fns.items():
        rows = []
        for ep in range(n_episodes):
            seed = RANDOM_SEED + ep
            row = _rollout_baseline_episode(cfgs, fn, seed, scenario=scenario)
            row["algorithm"] = name
            rows.append(row)
            all_logs.append(row)
        summary_rows.append(_aggregate_metrics(rows, name))

    for algo in ("ppo", "dqn"):
        try:
            rows = []
            for ep in range(n_episodes):
                seed = RANDOM_SEED + ep
                row = _rollout_sb3_episode(cfgs, algo, seed, scenario=scenario)
                rows.append(row)
                all_logs.append(row)
            summary_rows.append(_aggregate_metrics(rows, algo))
        except FileNotFoundError as exc:
            print(f"[eval_agents] Atlandı ({algo}): {exc}")

    _write_episode_logs(all_logs)
    metrics_df = pd.DataFrame(summary_rows)
    _save_comparison_charts(metrics_df)
    plot_learning_curves()

    if not skip_gifs:
        gif_seed = int(gif_seed)
        for algo in ("ppo", "dqn"):
            try:
                save_explained_gif_sb3(
                    cfgs,
                    algo,
                    OUTPUT_DIR / f"{algo}_agent_explained.gif",
                    seed=gif_seed,
                    scenario=scenario,
                    pick_shortest_success=True,
                )
                print(f"[eval_agents] GIF: {algo}")
            except Exception as exc:
                print(f"[eval_agents] GIF atlandı ({algo}): {exc}")

        # Baseline GIF (tek bölüm, aynı tohum)
        for name, fn in (
            ("random", random_action),
            ("greedy_shortest_path", greedy_manhattan_step),
        ):
            try:
                _save_baseline_gif(cfgs, fn, name, gif_seed, scenario)
            except Exception as exc:
                print(f"[eval_agents] Baseline GIF ({name}): {exc}")

    print(f"[eval_agents] metrics -> {OUTPUT_DIR / 'model_comparison_metrics.csv'}")
    print(f"[eval_agents] logs   -> {OUTPUT_DIR / 'rl_episode_logs.csv'}")


def _save_baseline_gif(
    episode_configs: list,
    action_fn: Callable,
    name: str,
    seed: int,
    scenario: str,
) -> None:
    import imageio.v2 as imageio

    env = GridParkingNavigationEnv(
        episode_configs,
        seed=RANDOM_SEED,
        render_mode="rgb_array",
        max_episode_steps=GRID_MAX_EPISODE_STEPS,
        scenario=scenario,
    )
    env.reset(seed=seed)
    frames: List[np.ndarray] = []
    f0 = env.render()
    if f0 is not None:
        info0 = {
            "step": 0,
            "instant_reward": 0.0,
            "cumulative_reward": 0.0,
            "distance_to_goal": env._manhattan(env._agent, env._goal),
            "action_name": "START",
            "path_length": 0,
            "collision_count": 0,
            "invalid_moves": 0,
            "visited_cells": 1,
            "status": "START",
        }
        frames.append(
            draw_metrics_overlay(np.array(f0), info0, algorithm=name, episode_display=1)
        )
    done = False
    while not done:
        a = action_fn(env)
        _obs, r, term, trunc, info = env.step(a)
        done = bool(term or trunc)
        fr = env.render()
        if fr is not None:
            info = dict(info)
            info.setdefault("instant_reward", float(r))
            frames.append(
                draw_metrics_overlay(np.array(fr), info, algorithm=name, episode_display=1)
            )
    out = OUTPUT_DIR / f"{name}_baseline_explained.gif"
    imageio.mimsave(str(out), frames, duration=0.35)
    print(f"[eval_agents] GIF: {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--scenario", default="medium")
    parser.add_argument("--skip-gifs", action="store_true")
    parser.add_argument("--gif-seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()
    run_evaluation(
        n_episodes=int(args.episodes),
        scenario=str(args.scenario),
        skip_gifs=bool(args.skip_gifs),
        gif_seed=int(args.gif_seed),
    )


if __name__ == "__main__":
    main()
