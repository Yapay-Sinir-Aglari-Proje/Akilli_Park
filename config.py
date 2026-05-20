"""
Proje yapılandırması — `ml_config` ve `paths` için tek giriş noktası.

Modüler scriptler (`train_ppo.py`, `evaluate_agents.py`, …) bu dosyayı import eder.
"""

from __future__ import annotations

from ml_config import *  # noqa: F401,F403 — merkezi sabitler
from paths import (  # noqa: F401
    DATA_PROCESSED,
    DATA_RAW,
    EVALUATION_DIR,
    LOGS_DIR,
    MODELS_DIR,
    OUTPUT_DIR,
    PREDICTIONS_DIR,
    PROJECT_ROOT,
    ensure_all_standard_dirs,
    ensure_output,
)

# RL senaryo varsayılanı (ortam `scenario=` parametresi)
DEFAULT_RL_SCENARIO: str = "medium"

# Sunum GIF’leri
GIF_SECONDS_PER_FRAME: float = 0.35
GIF_PRESENTATION_MAX_ATTEMPTS: int = 80
