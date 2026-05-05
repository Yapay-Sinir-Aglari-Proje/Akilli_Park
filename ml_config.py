"""
LSTM ve RL feature pipeline için merkezi konfigürasyon dosyası.
"""

# ===== LSTM =====
LSTM_TIME_STEP: int = 12
LSTM_EPOCHS: int = 80
LSTM_BATCH_SIZE: int = 32

# ===== RANDOM SEED =====
RANDOM_SEED: int = 42

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