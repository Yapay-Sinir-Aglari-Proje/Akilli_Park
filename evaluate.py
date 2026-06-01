"""
Birleşik değerlendirme: LSTM + RL (Stable-Baselines3) + ızgara baseline politikaları.

--part lstm: test_predictions.csv üzerinden MAE/RMSE
--part rl: PPO veya DQN + VecNormalize ile başarı oranı ve ortalama bölüm getirisi
--part baselines: rastgele, greedy Manhattan, BFS oracle ile karşılaştırma
--part all: hepsi; sonuç evaluation/reports/metrics.json dosyasına yazılır
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, List

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from ml_config import GRID_STEP_COST, RANDOM_SEED
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
    """SB3 bazen .pkl uzantısız kaydeder; her iki dosya adını da dene."""
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
    r2 = float(r2_score(y_t, y_p))
    denom = np.maximum(np.abs(y_t), 1e-6)
    mape = float(np.mean(np.abs((y_t - y_p) / denom)) * 100.0)
    return {
        "lstm_mae": mae,
        "lstm_rmse": rmse,
        "lstm_mape_pct": mape,
        "lstm_r2": r2,
        "n": len(df),
    }


def _freeze_vec_normalize(vec: VecNormalize) -> None:
    """Çıkarım modunda gözlem normalizasyonu sabit kalsın; ödül yeniden ölçeklenmesin."""
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

    rand_lengths: List[int] = []
    n_rand = min(50, n_episodes)
    for ep in range(n_rand):
        seed_r = RANDOM_SEED + 10_000 + ep
        env_r = GridParkingNavigationEnv(episode_configs, seed=RANDOM_SEED, max_episode_steps=250)
        _ret, _ok, nstep_r = rollout_episode_return(env_r, random_action, seed_r)
        rand_lengths.append(nstep_r)
    mean_random_episode_length = float(np.mean(rand_lengths)) if rand_lengths else 1.0

    successes = 0
    returns: List[float] = []
    regrets: List[float] = []
    lengths: List[int] = []
    routing_costs: List[float] = []
    visit_congestions: List[float] = []

    for ep in range(n_episodes):
        seed = RANDOM_SEED + ep
        # Aynı bölümde BFS ile mümkün olan üst sınır ödül; regret = r_star - elde edilen
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
        rc = last_info.get("routing_cost_proxy")
        if rc is None:
            rc = float(steps * GRID_STEP_COST)
        routing_costs.append(float(rc))
        vc = last_info.get("mean_visit_congestion")
        if vc is None:
            vc = 0.0
        visit_congestions.append(float(vc))

    mean_len = float(np.mean(lengths))
    traffic_reduction_pct = float(
        max(0.0, min(100.0, (1.0 - mean_len / max(mean_random_episode_length, 1e-6)) * 100.0))
    )

    return {
        "rl_algo": algo,
        "success_rate": successes / n_episodes,
        "mean_parking_search_steps": mean_len,
        "mean_episode_return": float(np.mean(returns)),
        "mean_regret_vs_bfs_oracle": float(np.mean(regrets)),
        "mean_episode_length": mean_len,
        "mean_routing_cost_proxy": float(np.mean(routing_costs)),
        "mean_visit_congestion": float(np.mean(visit_congestions)),
        "traffic_congestion_reduction_pct_vs_random": traffic_reduction_pct,
        "mean_random_episode_length_baseline": mean_random_episode_length,
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
    parser.add_argument(
        "--rl-algo",
        choices=["ppo", "dqn", "both"],
        default="ppo",
        help="'both' veya --part all: mevcut modeller için PPO ve DQN",
    )
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--skip-sb3", action="store_true")
    args = parser.parse_args()

    ensure_all_standard_dirs()
    out: dict = {}

    # İstenen parçalara göre metrikleri topla; sonunda JSON raporu yaz
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
        if args.rl_algo == "both" or args.part == "all":
            algos = ["ppo", "dqn"]
        else:
            algos = [args.rl_algo]
        rl_results: dict = {}
        for algo in algos:
            try:
                rl_results[algo] = eval_sb3_algo(algo, episode_configs, args.episodes)
                print(f"[eval] RL SB3 ({algo}):", rl_results[algo])
            except FileNotFoundError as exc:
                print(f"[eval] RL atlandı ({algo}): {exc}")
        if rl_results:
            out["rl_sb3"] = rl_results
            # Eski tek-algoritma okuyucuları için (Streamlit / notlar)
            if len(rl_results) == 1:
                out["rl_sb3_primary"] = next(iter(rl_results.values()))
            elif "ppo" in rl_results:
                out["rl_sb3_primary"] = rl_results["ppo"]

    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    report = EVALUATION_DIR / "metrics.json"
    with open(report, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"[eval] Rapor: {report}")


if __name__ == "__main__":
    main()
