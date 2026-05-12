"""
FastAPI servisi — Akıllı Park (BLM4502) hibrit tahmin + RL API.

Parking Birmingham veri seti; LSTM tahmini RL gözlemine bağlıdır (gözlem boyutu değiştiyse
PPO/DQN modellerini yeniden eğitin).

- GET /health: servis, seed, gözlem boyutu, proje künye özeti
- POST /predict: son test bağlamıyla bir sonraki aggregate doluluk oranı (LSTM)
- POST /act?algo=ppo|dqn: seçilen SB3 ajanı ile tek bölüm oynatma
- POST /simulate/reset ve POST /simulate: grid ortamında elle adım adım deneme
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ml_config import GRID_HEIGHT, GRID_WIDTH, RANDOM_SEED
from paths import DATA_PROCESSED, MODELS_DIR, PREDICTIONS_DIR
from utils.data_pipeline import load_feature_scaler
from utils.lstm_core import (
    inv_occupancy_rate,
    load_aggregate_frame,
    load_model_from_checkpoint,
    make_sequences,
    prepend_context,
)
from utils.seeds import set_global_seed

from env.grid_navigation_env import (
    GridParkingNavigationEnv,
    build_grid_nav_episode_configs,
)
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

set_global_seed(RANDOM_SEED)

app = FastAPI(title="Akıllı Park API", version="2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_lstm_bundle: dict | None = None
_rl_cache: dict[str, tuple] = {}
_cached_env: GridParkingNavigationEnv | None = None
_episode_cfgs = None


def _episode_configs():
    """İlk çağrıda test split’inden bölüm listesi üretilir; sonraki isteklerde önbellek kullanılır."""
    global _episode_cfgs
    if _episode_cfgs is None:
        _episode_cfgs = build_grid_nav_episode_configs(
            DATA_PROCESSED / "processed.parquet",
            PREDICTIONS_DIR / "test_predictions.csv"
            if (PREDICTIONS_DIR / "test_predictions.csv").exists()
            else None,
            split="test",
            base_seed=RANDOM_SEED,
        )
    return _episode_cfgs


def _get_lstm():
    """LSTM modelini ve checkpoint sözlüğünü tek sefer yükler (sunucu ömrü boyunca)."""
    global _lstm_bundle
    if _lstm_bundle is None:
        ckpt = torch.load(MODELS_DIR / "lstm_model.pt", map_location="cpu")
        mm = ckpt["input_cols"]
        model = load_model_from_checkpoint(ckpt)
        _lstm_bundle = {"model": model, "ckpt": ckpt, "mm": mm}
    return _lstm_bundle


PROJECT_KUNYE = {
    "proje": "Akıllı Park",
    "ders": "BLM4502 Yapay Sinir Ağlarına Giriş",
    "veri_seti": "Parking Birmingham (Kaggle)",
    "tahmin_modelleri": "LSTM, GRU, Temporal Transformer (train_lstm.py); XGBoost, RF (train_tabular_baselines.py)",
    "rl_algoritmalari": "PPO, DQN (Stable-Baselines3); API: POST /act?algo=ppo|dqn",
    "egitim_stratejileri": "early stopping, LR scheduling, weighted loss, Optuna (--optuna-trials)",
    "arayuz_streamlit": "ui_streamlit/app.py",
    "arayuz_react": "frontend/ (Vite + React, FastAPI CORS; npm run dev)",
}


@app.get("/health")
def health():
    obs_dim = 5 * int(GRID_HEIGHT) * int(GRID_WIDTH) + 2
    return {
        "status": "ok",
        "seed": RANDOM_SEED,
        "env": "grid_navigation_hybrid",
        "observation_dim": obs_dim,
        "project": PROJECT_KUNYE,
    }


class PredictResponse(BaseModel):
    last_updated: str
    y_pred_occupancy_rate: float


@app.post("/predict", response_model=PredictResponse)
def predict_next():
    pq = DATA_PROCESSED / "processed.parquet"
    if not pq.exists():
        raise HTTPException(500, "processed.parquet yok")
    b = _get_lstm()
    model, ckpt, mm = b["model"], b["ckpt"], b["mm"]
    ts = int(ckpt["time_step"])
    agg = load_aggregate_frame(pq)
    train_part = agg[agg["split"] == "train"][mm].to_numpy(dtype=np.float32)
    val_part = agg[agg["split"] == "val"][mm].to_numpy(dtype=np.float32)
    test_part = agg[agg["split"] == "test"][mm].to_numpy(dtype=np.float32)
    tv = prepend_context(train_part[-ts:], val_part)
    test_block = prepend_context(tv[-ts:], test_part)
    X, _ = make_sequences(test_block, ts)
    if len(X) == 0:
        raise HTTPException(500, "Yeterli test verisi yok")
    with torch.no_grad():
        pred = float(model(torch.from_numpy(X[-1:]).float()).numpy()[0])
    scaler, sk = load_feature_scaler(MODELS_DIR)
    rate = float(inv_occupancy_rate(np.array([pred], dtype=np.float64), scaler, sk)[0])
    agg_test = agg[agg["split"] == "test"].reset_index(drop=True)
    t_last = agg_test["LastUpdated"].iloc[-1]
    return PredictResponse(last_updated=str(t_last), y_pred_occupancy_rate=rate)


class ActResponse(BaseModel):
    algo: str
    actions: list[int]
    total_reward: float
    success: bool


def _vecnormalize_path(algo: str) -> Path:
    for p in (MODELS_DIR / f"vecnormalize_{algo}.pkl", MODELS_DIR / f"vecnormalize_{algo}"):
        if p.exists():
            return p
    raise HTTPException(500, f"VecNormalize yok: vecnormalize_{algo}")


def _get_rl_agent(algo: str):
    """PPO veya DQN + eşleşen VecNormalize (künye 6.2)."""
    global _rl_cache
    algo_l = algo.strip().lower()
    if algo_l not in ("ppo", "dqn"):
        raise HTTPException(400, "algo parametresi 'ppo' veya 'dqn' olmalıdır")
    if algo_l not in _rl_cache:
        cfgs = _episode_configs()

        def _make():
            return GridParkingNavigationEnv(cfgs, seed=RANDOM_SEED, max_episode_steps=250)

        venv = DummyVecEnv([_make])
        vec = VecNormalize.load(str(_vecnormalize_path(algo_l)), venv)
        vec.training = False
        vec.norm_reward = False
        base = MODELS_DIR / f"{algo_l}_agent"
        if algo_l == "ppo":
            model = PPO.load(
                str(base),
                env=vec,
                custom_objects={"lr_schedule": lambda _: 3e-4},
            )
        else:
            model = DQN.load(str(base), env=vec)
        _rl_cache[algo_l] = (model, vec)
    return _rl_cache[algo_l]


@app.post("/act", response_model=ActResponse)
def act(algo: str = Query("ppo", description="ppo veya dqn")):
    """Bir bölümü seçilen algoritma ile baştan sona oynatır; eylem dizisi döner."""
    algo_l = algo.strip().lower()
    model, vec = _get_rl_agent(algo_l)
    obs = vec.reset()
    actions: list[int] = []
    total = 0.0
    done = False
    success = False
    while not done:
        a, _ = model.predict(obs, deterministic=True)
        actions.append(int(a[0]))
        obs, r, dones, infos = vec.step(a)
        total += float(r[0])
        done = bool(dones[0])
        inf = infos[0] if isinstance(infos, (list, tuple)) else infos
        if isinstance(inf, dict) and inf.get("success"):
            success = True
    return ActResponse(algo=algo_l, actions=actions, total_reward=total, success=success)


class StepBody(BaseModel):
    action: int


class SimResponse(BaseModel):
    observation: list[float]
    reward: float
    terminated: bool
    truncated: bool
    info: dict


@app.post("/simulate/reset")
def simulate_reset():
    global _cached_env
    _cached_env = GridParkingNavigationEnv(
        _episode_configs(),
        seed=RANDOM_SEED,
        max_episode_steps=250,
    )
    obs, _ = _cached_env.reset(seed=RANDOM_SEED)
    return {"observation": obs.tolist()}


@app.post("/simulate", response_model=SimResponse)
def simulate_step(body: StepBody):
    global _cached_env
    if _cached_env is None:
        simulate_reset()
    assert _cached_env is not None
    obs, r, term, trunc, info = _cached_env.step(int(body.action))
    info_out = {k: v for k, v in info.items() if isinstance(v, (bool, int, float))}
    return SimResponse(
        observation=obs.tolist(),
        reward=float(r),
        terminated=bool(term),
        truncated=bool(trunc),
        info=info_out,
    )
