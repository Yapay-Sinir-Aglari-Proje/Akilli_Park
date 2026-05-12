"""
PPO ve DQN (Stable-Baselines3) ile GridParkingNavigationEnv eğitimi.

- VecNormalize (eğitim sonu donuk kayıt)
- TensorBoard: logs/tensorboard/<algo>/
- Çıktı: models/ppo_agent.zip | models/dqn_agent.zip
"""

from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from ml_config import (
    GRID_MAX_EPISODE_STEPS,
    GRID_REWARD_CLIP,
    RANDOM_SEED,
    RL_N_ENVS,
    RL_REWARD_NORM,
    RL_TOTAL_TIMESTEPS,
)
from paths import DATA_PROCESSED, LOGS_DIR, MODELS_DIR, PREDICTIONS_DIR, ensure_all_standard_dirs
from utils.data_pipeline import build_processed_dataset
from utils.seeds import set_global_seed

from env.grid_navigation_env import GridParkingNavigationEnv, build_grid_nav_episode_configs


def _make_vec_env(
    n_envs: int,
    episode_configs: list,
    log_dir: Path,
    norm_reward: bool,
) -> VecNormalize:
    """Paralel ortam yoksa bile SB3 için vektörleştirilmiş tek ortam + gözlem/ödül normalizasyonu."""
    log_dir.mkdir(parents=True, exist_ok=True)

    def make_fn(rank: int):
        def _init():
            # Her kopya farklı tohumla üretilir; Monitor episode istatistiklerini CSV'ye yazar
            env = GridParkingNavigationEnv(
                episode_configs,
                seed=RANDOM_SEED + rank * 17,
                max_episode_steps=int(GRID_MAX_EPISODE_STEPS),
            )
            return Monitor(env, filename=str(log_dir / f"mon_{rank}"))

        return _init

    venv = DummyVecEnv([make_fn(i) for i in range(n_envs)])
    return VecNormalize(
        venv,
        norm_obs=True,
        norm_reward=norm_reward,
        clip_obs=10.0,
        clip_reward=float(GRID_REWARD_CLIP),
        training=True,
    )


def train_algo(algo: str, timesteps: int, n_envs: int) -> Path:
    """Veri ve (varsa) LSTM tahminlerinden bölüm konfigürasyonları üretir, SB3 ile eğitir, modeli kaydeder."""
    ensure_all_standard_dirs()
    pq = DATA_PROCESSED / "processed.parquet"
    if not pq.exists():
        build_processed_dataset()
    pred_path = PREDICTIONS_DIR / "test_predictions.csv"
    # Her zaman dilimi + lot düzeni bir RL bölümü; tahmin CSV opsiyonel (hedef lot seçiminde kullanılabilir)
    episode_configs = build_grid_nav_episode_configs(
        pq,
        pred_path if pred_path.exists() else None,
        split=None,
        base_seed=RANDOM_SEED,
    )

    tb = LOGS_DIR / "tensorboard" / algo
    run_log = LOGS_DIR / "monitor" / algo
    vec = _make_vec_env(n_envs, episode_configs, run_log, RL_REWARD_NORM)

    algo_l = algo.lower()
    if algo_l == "ppo":
        model = PPO(
            "MlpPolicy",
            vec,
            verbose=1,
            seed=RANDOM_SEED,
            tensorboard_log=str(tb),
            learning_rate=2.5e-4,
            n_steps=2048,
            batch_size=128,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.0,
            clip_range=0.2,
            n_epochs=10,
        )
    elif algo_l == "dqn":
        model = DQN(
            "MlpPolicy",
            vec,
            verbose=1,
            seed=RANDOM_SEED,
            tensorboard_log=str(tb),
            learning_rate=1e-4,
            buffer_size=150_000,
            learning_starts=2000,
            batch_size=64,
            gamma=0.99,
            exploration_fraction=0.25,
            exploration_final_eps=0.05,
        )
    else:
        raise ValueError(f"Bilinmeyen algoritma: {algo}")

    model.learn(total_timesteps=timesteps, progress_bar=True)
    # Kayıt anında istatistikleri dondur; aksi halde değerlendirme sırasında dağılım kayar
    vec.training = False
    vec.norm_reward = False
    base = MODELS_DIR / f"{algo_l}_agent"
    model.save(str(base))
    vn_path = MODELS_DIR / f"vecnormalize_{algo_l}.pkl"
    vec.save(str(vn_path))
    print(f"[RL] Model: {base}.zip")
    print(f"[RL] VecNormalize: {vn_path}")
    return MODELS_DIR / f"{algo_l}_agent.zip"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=["ppo", "dqn", "both"], default="both")
    parser.add_argument("--timesteps", type=int, default=RL_TOTAL_TIMESTEPS)
    parser.add_argument("--n-envs", type=int, default=max(1, RL_N_ENVS))
    args = parser.parse_args()

    set_global_seed(RANDOM_SEED)
    if args.algo == "both":
        train_algo("ppo", args.timesteps, args.n_envs)
        train_algo("dqn", args.timesteps, args.n_envs)
    else:
        train_algo(args.algo, args.timesteps, args.n_envs)


if __name__ == "__main__":
    main()
