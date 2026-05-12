"""
Akıllı Park (BLM4502) — LSTM ve RL ortamı için merkezi konfigürasyon.

Grid gözlem boyutu değişirse (ör. env/ grid kanal sayısı) PPO/DQN ağırlıklarını yeniden eğitin.
"""

# ===== LSTM =====
LSTM_TIME_STEP: int = 12
LSTM_EPOCHS: int = 80
LSTM_BATCH_SIZE: int = 32

# Künye (5): MinMaxScaler veya StandardScaler — `utils.data_pipeline.build_processed_dataset`
FEATURE_SCALER: str = "minmax"  # "minmax" | "standard"

# ===== RANDOM SEED =====
RANDOM_SEED: int = 42

# ===== RL EĞİTİM =====
# Daha stabil / daha düz rota için önceki 120k üzerinde; PPO `ent_coef=0` ile birlikte kullanılır.
RL_TOTAL_TIMESTEPS: int = 280_000
RL_N_ENVS: int = 1
RL_REWARD_NORM: bool = True
# LSTM→RL bağlantısı: eğitimde çoğunlukla gürültülü tahmin gözlemi (dağılım kaymasına karşı dayanıklılık)
RL_PRED_NOISE_STD: float = 0.08
# Bu olasılıkla temiz (gürültüsüz) tahmin kullanılır; kalan kısımda gürültü eklenir
RL_PRED_CLEAN_PROB: float = 0.3
RL_REWARD_CLIP: float = 10.0
# Izgara navigasyon: hedef/timeout gibi büyük ödüller için ödül kırpma üst sınırı
GRID_REWARD_CLIP: float = 550.0

# ===== GRID NAVIGATION RL =====
GRID_HEIGHT: int = 15
GRID_WIDTH: int = 15
GRID_MAX_EPISODE_STEPS: int = 200
# --- Maliyet duyarlı ödül (künye / özgünlük): toplam skor ≈
#   (zaman maliyeti: -GRID_STEP_COST * adım)
# + (rota / mesafe şekillendirme: GRID_MANHATTAN_SHAPING_SCALE * (d_eski - d_yeni))
# + (tekrar ziyaret / salınım cezaları) + (hedef/timeout sabitleri)
# Ağırlıklar aşağıdaki sabitlerle doğrudan kontrol edilir; raporlama: `routing_cost_proxy` (ortam info).
# Daha kısa pencere: kısa A↔B salınımlarını daha çabuk cezalandırır.
GRID_LOOP_WINDOW: int = 8
GRID_LOOP_PENALTY: float = -6.0
GRID_REVISIT_PENALTY: float = -4.0
# Çok yüksek olunca koridorlarda gereksiz keşif / zigzag artar; düz rota için düşürüldü.
GRID_FIRST_VISIT_BONUS: float = 0.08
GRID_GOAL_BONUS: float = 300.0
GRID_TIMEOUT_PENALTY: float = -200.0
# Hedefe yaklaşmayı adım başına güçlendirir (Manhattan d_new - d_old pozitifken).
GRID_MANHATTAN_SHAPING_SCALE: float = 4.5
GRID_STEP_COST: float = 1.25

# ===== RL ORTAMI (künye: grid tabanlı simülasyon) =====
SELECTED_ENV: str = "grid"

_ENV_ALIASES = {
    "grid": "grid",
    "g": "grid",
    "discrete": "grid",
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


ENV_TYPE: str = _normalize_env_name(SELECTED_ENV)
USE_CONTINUOUS: bool = False