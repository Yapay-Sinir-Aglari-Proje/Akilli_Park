"""
LSTM ve RL feature pipeline için merkezi konfigürasyon dosyası.
"""

# ===== LSTM =====
LSTM_TIME_STEP: int = 12
LSTM_EPOCHS: int = 80
LSTM_BATCH_SIZE: int = 32

# ===== RANDOM SEED =====
RANDOM_SEED: int = 42

# ===== RL EĞİTİM =====
RL_TOTAL_TIMESTEPS: int = 120_000
RL_N_ENVS: int = 1
RL_REWARD_NORM: bool = True
# LSTM→RL coupling: eğitimde çoğunlukla gürültülü pred gözlemi (temsil drift’e karşı)
RL_PRED_NOISE_STD: float = 0.08
RL_PRED_CLEAN_PROB: float = 0.3
RL_REWARD_CLIP: float = 10.0
# Grid navigasyon: büyük hedef/timeout ödülleri için clip üst sınırı
GRID_REWARD_CLIP: float = 550.0

# ===== GRID NAVIGATION RL =====
GRID_HEIGHT: int = 15
GRID_WIDTH: int = 15
GRID_MAX_EPISODE_STEPS: int = 200
# Ödül mühendisliği (zigzag / loop azaltma)
GRID_LOOP_WINDOW: int = 10
GRID_LOOP_PENALTY: float = -4.0
GRID_REVISIT_PENALTY: float = -2.5
GRID_FIRST_VISIT_BONUS: float = 0.15
GRID_GOAL_BONUS: float = 300.0
GRID_TIMEOUT_PENALTY: float = -200.0
GRID_MANHATTAN_SHAPING_SCALE: float = 2.0
GRID_STEP_COST: float = 1.0

# ===== RL ENVIRONMENT =====
# Sadece bu değeri değiştir:
# - "grid"     -> GridParkingEnv (discrete)
# - "external" -> ExternalParkingEnv (continuous)
# Ek aliaslar: "g", "ext", "continuous", "discrete"
SELECTED_ENV: str = "grid"

_ENV_ALIASES = {
    "grid": "grid",
    "g": "grid",
    "discrete": "grid",
    "external": "external",
    "ext": "external",
    "continuous": "external",
}

_ENV_IS_CONTINUOUS = {
    "grid": False,
    "external": True,
}


def _normalize_env_name(raw_env_name: str) -> str:
    key = str(raw_env_name).strip().lower()
    if key in _ENV_ALIASES:
        return _ENV_ALIASES[key]
    allowed = ", ".join(sorted(_ENV_ALIASES.keys()))
    raise ValueError(
        f"Geçersiz SELECTED_ENV: {raw_env_name!r}. "
        f"Kullanılabilir değerler: {allowed}"
    )


# rl_model.py ve rl_animation.py bunları import ettiği için geriye uyumlu adları koruyoruz.
ENV_TYPE: str = _normalize_env_name(SELECTED_ENV)
USE_CONTINUOUS: bool = _ENV_IS_CONTINUOUS[ENV_TYPE]