"""
Birleşik değerlendirme: LSTM + RL (SB3) + grid baseline politikaları.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, List

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from ml_config import RANDOM_SEED
from paths import DATA_PROCESSED, EVALUATION_DIR, MODELS_DIR, PREDICTIONS_DIR, ensure_all_standard_dirs

from env.grid_navigation_env import GridParkingNavigationEnv, build_grid_nav_episode_configs
from evaluation.baselines import (
    bfs_first_action,
    greedy_manhattan_step,
    oracle_episode_return,
    random_action,
    rollout_episode_return,
)


def _vecnormalize_file(algo: str) -> Path:
    pkl = MODELS_DIR / f"vecnormalize_{algo}.pkl"
    raw = MODELS_DIR / f"vecnormalize_{algo}"
    if pkl.exists():
        return pkl
    if raw.exists():
        return raw
    return pkl


def eval_lstm() -> dict:
    pred_path = PREDICTIONS_DIR / "test_predictions.csv"
    if not pred_path.exists():
        raise FileNotFoundError(f"LSTM tahminleri yok: {pred_path} — önce train_lstm.py")

    df = pd.read_csv(pred_path)
    y_t = df["y_true_occupancy_rate"].to_numpy(dtype=float)
    y_p = df["y_pred_occupancy_rate"].to_numpy(dtype=float)
    mae = float(mean_absolute_error(y_t, y_p))
    rmse = float(np.sqrt(mean_squared_error(y_t, y_p)))
    return {"lstm_mae": mae, "lstm_rmse": rmse, "n": len(df)}


def _freeze_vec_normalize(vec: VecNormalize) -> None:
    vec.training = False
    vec.norm_reward = False


def eval_sb3_algo(algo: str, episode_configs: list, n_episodes: int) -> dict:
    algo = algo.lower()
    model_zip = MODELS_DIR / f"{algo}_agent.zip"
    if not model_zip.exists():
        raise FileNotFoundError(f"RL model yok: {model_zip} — önce train_rl.py")

    vec_path = _vecnormalize_file(algo)
    if not vec_path.exists():
        raise FileNotFoundError(
            f"VecNormalize yok: {MODELS_DIR / f'vecnormalize_{algo}.pkl'} — önce train_rl.py"
        )

    def _make():
        return GridParkingNavigationEnv(episode_configs, seed=RANDOM_SEED, max_episode_steps=250)

    venv = DummyVecEnv([_make])
    vec = VecNormalize.load(str(vec_path), venv)
    _freeze_vec_normalize(vec)

    load_path = str(MODELS_DIR / f"{algo}_agent")
    if algo == "ppo":
        model = PPO.load(
            load_path,
            env=vec,
            custom_objects={"lr_schedule": lambda _: 3e-4},
        )
    else:
        model = DQN.load(load_path, env=vec)

    successes = 0
    returns: List[float] = []
    regrets: List[float] = []
    lengths: List[int] = []

    for ep in range(n_episodes):
        seed = RANDOM_SEED + ep
        r_star = oracle_episode_return(episode_configs, seed)
        vec.seed(int(seed))
        obs = vec.reset()
        ep_ret = 0.0
        done = False
        steps = 0
        last_info: dict = {}
        while not done:
            act, _ = model.predict(obs, deterministic=True)
            obs, r, dones, infos = vec.step(act)
            ep_ret += float(r[0])
            done = bool(dones[0])
            steps += 1
            inf = infos[0] if isinstance(infos, (list, tuple)) else infos
            last_info = inf if isinstance(inf, dict) else {}
        returns.append(ep_ret)
        regrets.append(float(r_star - ep_ret))
        lengths.append(steps)
        if last_info.get("success"):
            successes += 1

    return {
        "rl_algo": algo,
        "success_rate": successes / n_episodes,
        "mean_episode_return": float(np.mean(returns)),
        "mean_regret_vs_bfs_oracle": float(np.mean(regrets)),
        "mean_episode_length": float(np.mean(lengths)),
        "episodes": n_episodes,
    }


def eval_baseline_policy(
    name: str,
    action_fn: Callable[[GridParkingNavigationEnv], int],
    episode_configs: list,
    n_episodes: int,
) -> dict:
    env = GridParkingNavigationEnv(episode_configs, seed=RANDOM_SEED, max_episode_steps=250)
    successes = 0
    returns: List[float] = []
    regrets: List[float] = []
    lengths: List[int] = []

    for ep in range(n_episodes):
        seed = RANDOM_SEED + ep
        r_star = oracle_episode_return(episode_configs, seed)
        ret, ok, nstep = rollout_episode_return(env, action_fn, seed)
        returns.append(ret)
        regrets.append(float(r_star - ret))
        lengths.append(nstep)
        if ok:
            successes += 1

    return {
        "policy": name,
        "success_rate": successes / n_episodes,
        "mean_episode_return": float(np.mean(returns)),
        "mean_regret_vs_bfs_oracle": float(np.mean(regrets)),
        "mean_episode_length": float(np.mean(lengths)),
        "episodes": n_episodes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", choices=["lstm", "rl", "baselines", "all"], default="all")
    parser.add_argument("--rl-algo", choices=["ppo", "dqn"], default="ppo")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--skip-sb3", action="store_true")
    args = parser.parse_args()

    ensure_all_standard_dirs()
    out: dict = {}

    if args.part in ("lstm", "all"):
        out["lstm"] = eval_lstm()
        print("[eval] LSTM:", out["lstm"])

    episode_configs = None
    if args.part in ("rl", "baselines", "all"):
        episode_configs = build_grid_nav_episode_configs(
            DATA_PROCESSED / "processed.parquet",
            PREDICTIONS_DIR / "test_predictions.csv"
            if (PREDICTIONS_DIR / "test_predictions.csv").exists()
            else None,
            split="test",
            base_seed=RANDOM_SEED,
        )

    if args.part in ("baselines", "all"):
        assert episode_configs is not None
        out["baselines"] = {
            "random": eval_baseline_policy(
                "random", random_action, episode_configs, args.episodes
            ),
            "greedy_manhattan": eval_baseline_policy(
                "greedy_manhattan", greedy_manhattan_step, episode_configs, args.episodes
            ),
            "bfs_oracle": eval_baseline_policy(
                "bfs_oracle", bfs_first_action, episode_configs, args.episodes
            ),
        }
        print("[eval] baselines:", {k: v["mean_episode_return"] for k, v in out["baselines"].items()})

    if args.part in ("rl", "all") and not args.skip_sb3:
        assert episode_configs is not None
        out["rl_sb3"] = eval_sb3_algo(args.rl_algo, episode_configs, args.episodes)
        print("[eval] RL SB3:", out["rl_sb3"])

    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    report = EVALUATION_DIR / "metrics.json"
    with open(report, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"[eval] Rapor: {report}")


if __name__ == "__main__":
    main()
