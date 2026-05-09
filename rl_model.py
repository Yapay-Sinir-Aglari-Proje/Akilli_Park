"""
PPO ile GridParkingEnv (mode=train) üzerinde politika eğitimi.

Grid dünyada ajan boş park (yeşil) hücrelere giderek sabit hedefe ulaşmayı öğrenir;
episode boyunca hedef ve doluluk haritası statiktir (stabil PPO).

Aksiyon maskesi: gözlemin son 4 boyutu geçerli yönleri kodlar; MaskedActorCriticPolicy
logitleri maskeler (geçersiz yön olasılığı 0). Alternatif: stable-baselines3-contrib
MaskablePPO + ActionMasker.

- train_env / val_env / test_env aynı sınıftan, farklı Gymnasium tohumlarıyla kurulur.
- EvalCallback doğrulama ortamında periyodik ölçüm yapar; en iyi ağırlıklar
  `models/best_model.zip` altında saklanır (Stable-Baselines3 varsayılanı).
- Final politika: models/ppo_parking_model_final.zip
"""

from __future__ import annotations

from collections import Counter
from collections import deque
from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from ml_config import ENV_TYPE, USE_CONTINUOUS
from paths import LOGS_DIR, MODELS_DIR, ensure_logs_dir, ensure_models
from parking_rl.grid_parking_env import GridParkingEnv, TRAIN_PPO_ENV_KWARGS
from parking_rl.external_env import ExternalParkingEnv
from parking_rl.masked_policy import MaskedActorCriticPolicy, masked_policy_kwargs


class GaussianActionNoiseWrapper(gym.ActionWrapper):
    """Continuous env için eğitim sırasında aksiyona küçük Gaussian noise ekler."""

    def __init__(self, env: gym.Env, sigma: float = 0.1) -> None:
        super().__init__(env)
        self.sigma = float(sigma)
        if not isinstance(self.action_space, gym.spaces.Box):
            raise TypeError("GaussianActionNoiseWrapper sadece Box action space destekler.")

    def action(self, action: np.ndarray) -> np.ndarray:
        a = np.asarray(action, dtype=np.float32)
        noise = np.random.normal(loc=0.0, scale=self.sigma, size=a.shape).astype(np.float32)
        a_noisy = a + noise
        return np.clip(a_noisy, self.action_space.low, self.action_space.high).astype(np.float32)


def _resolve_best_model_path() -> Path | None:
    """Önce best_model.zip, yoksa best_model aranır."""
    for name in ("best_model.zip", "best_model"):
        p = MODELS_DIR / name
        if p.exists():
            return p
    return None


class ActionHistogramCallback(BaseCallback):
    """Toplanan ayrık aksiyonları logs/actions.csv olarak yazar (rl_visualizer)."""

    def __init__(self, out_path: Path):
        super().__init__(0)
        self.out_path = out_path
        self._counts: Counter[int] = Counter()

    def _on_step(self) -> bool:
        actions = self.locals.get("actions")
        if actions is not None:
            for a in np.asarray(actions).reshape(-1):
                self._counts[int(a)] += 1
        return True

    def _on_training_end(self) -> None:
        if not self._counts:
            return
        actions = sorted(self._counts.keys())
        df = pd.DataFrame({"action": actions, "count": [self._counts[k] for k in actions]})
        df.to_csv(self.out_path, index=False)


class RolloutActionLogCallback(BaseCallback):
    """Her rollout sonunda bu turda seçilen ayrık aksiyonların özetini stdout’a yazar."""

    def __init__(self) -> None:
        super().__init__(0)
        self._buf: list[int] = []

    def _on_step(self) -> bool:
        actions = self.locals.get("actions")
        if actions is not None:
            self._buf.extend(int(x) for x in np.asarray(actions).reshape(-1))
        return True

    def _on_rollout_end(self) -> None:
        if not self._buf:
            return
        c = Counter(self._buf)
        names = ("UP", "DOWN", "LEFT", "RIGHT")
        parts = [f"{names[k]}={c[k]}" for k in sorted(c.keys())]
        print(f"[PPO] Rollout aksiyon özeti ({len(self._buf)} adım): " + ", ".join(parts))
        self._buf.clear()


class EpisodeRewardMA100Callback(BaseCallback):
    """Episode ödül moving average (window=100) metriğini logger'a yazar."""

    def __init__(self, window_size: int = 100) -> None:
        super().__init__(0)
        self.window_size = int(window_size)
        self._recent_rewards: list[float] = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos")
        dones = self.locals.get("dones")
        if infos is None or dones is None:
            return True

        for info, done in zip(infos, dones):
            if not bool(done):
                continue
            episode_info = info.get("episode")
            if episode_info is None:
                continue
            reward = float(episode_info.get("r", 0.0))
            self._recent_rewards.append(reward)
            if len(self._recent_rewards) > self.window_size:
                self._recent_rewards = self._recent_rewards[-self.window_size :]
            if self._recent_rewards:
                ma_reward = float(np.mean(self._recent_rewards))
                self.logger.record("train/episode_reward_ma100", ma_reward)
        return True


class TrainingDiagnosticsCallback(BaseCallback):
    """Checkpoint metrikleri ve öğrenme trend raporu üretir."""

    def __init__(
        self,
        report_path: Path,
        checkpoints: list[int],
        window_size: int = 100,
        total_timesteps: int = 200_000,
    ) -> None:
        super().__init__(0)
        self.report_path = report_path
        self.checkpoints = sorted({int(x) for x in checkpoints})
        self.window_size = int(window_size)
        self.total_timesteps = int(total_timesteps)
        self._episodes: deque[dict[str, float]] = deque(maxlen=self.window_size)
        self._rows: list[dict[str, float]] = []
        self._next_checkpoint_idx = 0

    @staticmethod
    def _to_bool(value: object) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, np.integer)):
            return int(value) != 0
        if isinstance(value, (float, np.floating)):
            return float(value) != 0.0
        if isinstance(value, str):
            return value.lower().strip() in {"1", "true", "yes", "y"}
        return False

    def _on_step(self) -> bool:
        infos = self.locals.get("infos")
        dones = self.locals.get("dones")
        if infos is None or dones is None:
            return True

        for info, done in zip(infos, dones):
            if not bool(done):
                continue
            ep = info.get("episode")
            if ep is None:
                continue

            success = self._to_bool(
                info.get("is_success", info.get("success", info.get("terminated", False)))
            )
            # Grid env'de doğrudan collision yok; invalid_move "çarpışma-benzeri" sinyal olarak alınır.
            collision = self._to_bool(
                info.get(
                    "collision",
                    info.get("crashed", info.get("invalid_move", False)),
                )
            )
            self._episodes.append(
                {
                    "reward": float(ep.get("r", 0.0)),
                    "length": float(ep.get("l", 0.0)),
                    "success": float(success),
                    "collision": float(collision),
                }
            )

        while (
            self._next_checkpoint_idx < len(self.checkpoints)
            and self.num_timesteps >= self.checkpoints[self._next_checkpoint_idx]
        ):
            self._snapshot(self.checkpoints[self._next_checkpoint_idx])
            self._next_checkpoint_idx += 1
        return True

    def _snapshot(self, timestep: int) -> None:
        if not self._episodes:
            row = {
                "timestep": int(timestep),
                "mean_reward": 0.0,
                "success_rate": 0.0,
                "collision_rate": 0.0,
                "avg_episode_length": 0.0,
            }
        else:
            rewards = np.asarray([x["reward"] for x in self._episodes], dtype=np.float64)
            successes = np.asarray([x["success"] for x in self._episodes], dtype=np.float64)
            collisions = np.asarray([x["collision"] for x in self._episodes], dtype=np.float64)
            lengths = np.asarray([x["length"] for x in self._episodes], dtype=np.float64)
            row = {
                "timestep": int(timestep),
                "mean_reward": float(np.mean(rewards)),
                "success_rate": float(np.mean(successes)),
                "collision_rate": float(np.mean(collisions)),
                "avg_episode_length": float(np.mean(lengths)),
            }
        self._rows.append(row)
        print(
            f"[CHECKPOINT] t={row['timestep']} "
            f"mean_reward={row['mean_reward']:.4f} "
            f"success_rate={row['success_rate']:.4f} "
            f"collision_rate={row['collision_rate']:.4f} "
            f"avg_episode_length={row['avg_episode_length']:.2f}"
        )

    @staticmethod
    def _trend_label(values: np.ndarray, timesteps: np.ndarray, eps: float) -> str:
        if values.size < 2 or np.allclose(values, values[0]):
            return "flat"
        slope, _ = np.polyfit(timesteps, values, 1)
        changes = np.diff(values)
        flips = int(np.sum(np.sign(changes[1:]) != np.sign(changes[:-1]))) if changes.size > 1 else 0
        if abs(slope) < eps:
            return "flat"
        if flips >= max(2, int(0.5 * changes.size)):
            return "unstable"
        return "increasing" if slope > 0 else "decreasing"

    def _analyze_and_print_report(self) -> None:
        if not self._rows:
            print("[PPO] Uyarı: training_report için veri toplanamadı.")
            return

        timesteps = np.asarray([r["timestep"] for r in self._rows], dtype=np.float64)
        mean_rewards = np.asarray([r["mean_reward"] for r in self._rows], dtype=np.float64)
        success_rates = np.asarray([r["success_rate"] for r in self._rows], dtype=np.float64)

        reward_trend = self._trend_label(mean_rewards, timesteps, eps=1e-5)
        success_trend = self._trend_label(success_rates, timesteps, eps=1e-7)
        reward_std = float(np.std(mean_rewards))

        final_success = float(success_rates[-1])
        final_mean_reward = float(mean_rewards[-1])

        if reward_trend == "increasing" and success_trend == "increasing" and final_success >= 0.6:
            verdict = "LEARNING"
            stability = "stable"
        elif final_success >= 0.3 or reward_trend == "increasing" or success_trend == "increasing":
            verdict = "PARTIAL LEARNING"
            stability = "moderately stable" if reward_std < 10.0 else "unstable"
        else:
            verdict = "NO LEARNING"
            stability = "unstable"

        reason = (
            f"Checkpoint trendleri reward={reward_trend}, success={success_trend}; "
            f"final başarı oranı={final_success:.3f}, final mean reward={final_mean_reward:.3f}, "
            f"reward std={reward_std:.3f}."
        )

        print("\n==============================")
        print("PPO LEARNING DIAGNOSTIC REPORT")
        print("==============================\n")
        print(f"Reward Trend: {reward_trend}")
        print(f"Success Trend: {success_trend}")
        print(f"Stability: {stability}")
        print(f"Final Success Rate: {final_success:.4f}")
        print(f"Final Mean Reward: {final_mean_reward:.4f}\n")
        print("Verdict:")
        print(f"- {verdict}\n")
        print("Reason:")
        print(reason)
        print("\n==============================\n")

    def _on_training_end(self) -> None:
        # Final snapshot (200k) garanti edilir.
        if not self._rows or int(self._rows[-1]["timestep"]) != self.total_timesteps:
            self._snapshot(self.total_timesteps)

        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(self._rows).to_csv(self.report_path, index=False)
        print(f"[PPO] Training raporu yazıldı: {self.report_path}")
        self._analyze_and_print_report()


def main() -> None:
    ensure_models()
    ensure_logs_dir()

    env_type = ENV_TYPE.lower().strip()
    if env_type not in ("grid", "external"):
        raise ValueError(f"ENV_TYPE geçersiz: {ENV_TYPE}. Beklenen: 'grid' | 'external'")

    print(f"[PPO] Ortamlar kuruluyor (ENV_TYPE={env_type})...")
    # Episode ödül/uzunluk: logs/train.monitor.csv (rl_visualizer ile uyumlu)
    # debug_checks=False: her adımda grid/hedef doğrulaması yapmaz; eğitim CPU'da belirgin hızlanır.
    if env_type == "grid":
        train_env_raw = Monitor(GridParkingEnv(**TRAIN_PPO_ENV_KWARGS), str(LOGS_DIR / "train"))
        val_env_raw = Monitor(GridParkingEnv(**TRAIN_PPO_ENV_KWARGS))
        test_env_raw = Monitor(GridParkingEnv(**TRAIN_PPO_ENV_KWARGS))
    else:
        ext_kwargs = {"max_episode_steps": TRAIN_PPO_ENV_KWARGS.get("max_episode_steps", 200)}
        train_env_raw = Monitor(ExternalParkingEnv(**ext_kwargs), str(LOGS_DIR / "train"))
        val_env_raw = Monitor(ExternalParkingEnv(**ext_kwargs))
        test_env_raw = Monitor(ExternalParkingEnv(**ext_kwargs))

    if isinstance(train_env_raw.action_space, gym.spaces.Box):
        train_env_raw = GaussianActionNoiseWrapper(train_env_raw, sigma=0.1)

    train_env = VecNormalize(
        DummyVecEnv([lambda: train_env_raw]),
        norm_obs=True,
        norm_reward=True,
    )
    val_env = VecNormalize(
        DummyVecEnv([lambda: val_env_raw]),
        norm_obs=True,
        norm_reward=True,
        training=False,
    )
    val_env.obs_rms = train_env.obs_rms
    val_env.ret_rms = train_env.ret_rms

    # eval_freq: en az bir rollout tamamlanabilsin (n_steps ile uyumlu)
    eval_callback = EvalCallback(
        val_env,
        best_model_save_path=str(MODELS_DIR),
        log_path=str(MODELS_DIR / "ppo_eval_logs"),
        eval_freq=8_192,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
        verbose=1,
    )
    is_continuous = isinstance(train_env_raw.action_space, gym.spaces.Box)
    if bool(USE_CONTINUOUS) and not is_continuous:
        print("[PPO] Uyarı: USE_CONTINUOUS=True ancak env action space continuous değil.")
    if (not bool(USE_CONTINUOUS)) and is_continuous:
        print("[PPO] Uyarı: USE_CONTINUOUS=False ancak env action space continuous.")

    print("[PPO] Öğrenme başlıyor (200k timestep)...")
    # n_steps: her güncellemeden önce toplanan ortam adımı; düşük değer = daha sık güncelleme,
    # rollout turu daha kısa sürer (1024 tipik olarak 2048'e göre ~yarı süre, öğrenme hâlâ stabil).
    callbacks = [
        eval_callback,
        EpisodeRewardMA100Callback(window_size=100),
        TrainingDiagnosticsCallback(
            report_path=LOGS_DIR / "training_report.csv",
            checkpoints=[10_000, 20_000, 40_000, 80_000],
            window_size=100,
            total_timesteps=200_000,
        ),
    ]
    if is_continuous:
        policy_name = "MlpPolicy"
    else:
        policy_name = MaskedActorCriticPolicy
        callbacks.extend(
            [ActionHistogramCallback(LOGS_DIR / "actions.csv"), RolloutActionLogCallback()]
        )

    model = PPO(
        policy_name,
        train_env,
        policy_kwargs=None if is_continuous else masked_policy_kwargs(),
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        gamma=0.99,
        ent_coef=0.02,
    )
    # Kayıp eğrileri: logs/progress.csv (train/policy_gradient_loss, train/value_loss)
    model.set_logger(configure(str(LOGS_DIR), ["stdout", "csv"]))
    model.learn(
        total_timesteps=200_000,
        callback=callbacks,
    )
    vecnorm_path = MODELS_DIR / "vecnormalize.pkl"
    train_env.save(str(vecnorm_path))
    print(f"[PPO] VecNormalize istatistikleri kaydedildi: {vecnorm_path}")

    final_path = MODELS_DIR / "ppo_parking_model_final.zip"
    # Stable-Baselines3 kayıtta dosya adına otomatik .zip ekler
    model.save(str(MODELS_DIR / "ppo_parking_model_final"))
    print(f"[PPO] Final model kaydı: {final_path}")

    print("\n[PPO] En iyi model ile evaluate_policy...")
    best = _resolve_best_model_path()
    if best is None:
        print("[PPO] Uyarı: best_model bulunamadı, final model kullanılıyor.")
        eval_model = PPO.load(str(final_path))
    else:
        print(f"[PPO] Yüklenen en iyi model: {best}")
        eval_model = PPO.load(str(best))

    eval_envs = []
    for name, raw_env in (("train", train_env_raw), ("val", val_env_raw), ("test", test_env_raw)):
        wrapped = VecNormalize.load(str(vecnorm_path), DummyVecEnv([lambda e=raw_env: e]))
        wrapped.training = False
        wrapped.norm_reward = False
        eval_envs.append((name, wrapped))

    for name, env in eval_envs:
        mean_r, std_r = evaluate_policy(
            eval_model, env, n_eval_episodes=10, deterministic=True
        )
        print(f"  {name}: ortalama ödül = {mean_r:.3f} ± {std_r:.3f}")


if __name__ == "__main__":
    main()
