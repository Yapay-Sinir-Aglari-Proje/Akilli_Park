"""
Eğitilmiş PPO politikasını GridParkingEnv (mode=train) üzerinde değerlendirir.

- Öncelik: models/best_model.zip (EvalCallback çıktısı)
- Yoksa: models/ppo_parking_model_final.zip

Metrikler (100 bölüm):
    - Hedefe varış oranı (başarı %)
    - Ortalama park bulma süresi (adım sayısı)
    - Ortalama rota uzunluğu (Manhattan adım = grid'de yakıt/zaman proxy'si)
    - Ortalama ödül
    - Kümülatif reward trendi (matplotlib grafiği)
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO

from paths import MODELS_DIR, OUTPUT_DIR, ensure_output
from parking_rl.grid_parking_env import GridParkingEnv, TRAIN_PPO_ENV_KWARGS
# PPO.load() zip içindeki policy class'ını bulabilsin diye explicit import
from parking_rl.masked_policy import MaskedActorCriticPolicy  # noqa: F401


def load_eval_model():
    """En iyi model yolu yoksa final modele düş."""
    # SB3 ActorCriticPolicy zip yüklemesi için lr_schedule gerekir
    custom_objects = {"lr_schedule": lambda _: 3e-4}
    candidates = [
        MODELS_DIR / "best_model.zip",
        MODELS_DIR / "best_model",
        MODELS_DIR / "ppo_parking_model_final.zip",
    ]
    for p in candidates:
        if p.exists():
            return PPO.load(str(p), custom_objects=custom_objects), p
    raise FileNotFoundError(
        f"Hiçbir PPO ağırlığı bulunamadı. Önce rl_model.py çalıştırın. Aranan: {candidates}"
    )


def main() -> None:
    print("=" * 60)
    print(" PPO performans değerlendirmesi (GridParkingEnv train)")
    print("=" * 60)

    ensure_output()
    env = GridParkingEnv(**TRAIN_PPO_ENV_KWARGS)
    model, path = load_eval_model()
    print(f"Yüklenen model: {path}")
    print(f"Inference policy sınıfı: {type(model.policy).__name__}")

    n_episodes = 100
    successes = 0
    steps_list: list[int] = []
    rewards_list: list[float] = []

    for ep in range(n_episodes):
        obs, _info = env.reset(seed=ep + 2024)
        terminated = False
        truncated = False
        steps = 0
        ep_reward = 0.0

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _step_info = env.step(int(action))
            steps += 1
            ep_reward += float(reward)

        if terminated:
            successes += 1
        steps_list.append(steps)
        rewards_list.append(ep_reward)

    success_rate = 100.0 * successes / n_episodes
    avg_steps = float(np.mean(steps_list))
    avg_reward = float(np.mean(rewards_list))

    # Grid dünyada her adım = bir Manhattan birimi.
    # "Park bulma süresi", "rota uzunluğu" ve "yakıt maliyeti (1 birim/adım)" hep bu metriğe dayanır.
    avg_route_length = avg_steps
    avg_fuel_cost = avg_steps  # 1 birim yakıt/adım proxy'si

    print("-" * 60)
    print(f" Bölüm sayısı                       : {n_episodes}")
    print(f" Başarı (hedef) %                   : {success_rate:.1f}")
    print(f" Ortalama park bulma süresi (adım)  : {avg_steps:.2f}")
    print(f" Ortalama rota uzunluğu (Manhattan) : {avg_route_length:.2f}")
    print(f" Ortalama yakıt/zaman maliyeti      : {avg_fuel_cost:.2f}")
    print(f" Ortalama ödül                      : {avg_reward:.4f}")
    print("-" * 60)

    # Kümülatif reward trendi (bölüm bazında)
    cum_rewards = np.cumsum(rewards_list)
    plt.figure(figsize=(10, 4))
    plt.plot(np.arange(1, n_episodes + 1), cum_rewards, color="steelblue", linewidth=1.5)
    plt.xlabel("Bölüm")
    plt.ylabel("Kümülatif ödül")
    plt.title("Değerlendirme — kümülatif reward trendi (deterministik politika)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    cum_path = OUTPUT_DIR / "evaluate_cumulative_reward.png"
    plt.savefig(cum_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[evaluate] Kümülatif reward grafiği: {cum_path}")

    # Bölüm başına adım sayısı dağılımı (park bulma süresi histogramı)
    plt.figure(figsize=(10, 4))
    plt.hist(steps_list, bins=20, color="coral", edgecolor="white")
    plt.xlabel("Bölüm uzunluğu (adım)")
    plt.ylabel("Frekans")
    plt.title("Park bulma süresi dağılımı (100 bölüm)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    hist_path = OUTPUT_DIR / "evaluate_steps_histogram.png"
    plt.savefig(hist_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[evaluate] Adım histogramı: {hist_path}")


if __name__ == "__main__":
    main()