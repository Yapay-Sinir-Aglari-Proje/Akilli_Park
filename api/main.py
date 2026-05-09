"""
FastAPI: /predict (LSTM), /act (PPO grid: tek bölüm oynatır), /simulate (grid adımı).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from ml_config import RANDOM_SEED
from paths import DATA_PROCESSED, MODELS_DIR, PREDICTIONS_DIR
from utils.lstm_core import OccupancyLSTM, load_aggregate_frame, make_sequences, prepend_context
from utils.seeds import set_global_seed

from env.grid_navigation_env import (
    GridParkingNavigationEnv,
    build_grid_nav_episode_configs,
)
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

set_global_seed(RANDOM_SEED)

app = FastAPI(title="Akıllı Otopark API", version="2.0")

_lstm_bundle: dict | None = None
_ppo_model = None
_ppo_vec = None
_cached_env: GridParkingNavigationEnv | None = None
_episode_cfgs = None


def _episode_configs():
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
    global _lstm_bundle
    if _lstm_bundle is None:
        ckpt = torch.load(MODELS_DIR / "lstm_model.pt", map_location="cpu")
        mm = ckpt["input_cols"]
        model = OccupancyLSTM(
            input_dim=len(mm),
            hidden=ckpt.get("hidden", 64),
            num_layers=ckpt.get("num_layers", 2),
            dropout=ckpt.get("dropout", 0.2),
        )
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        _lstm_bundle = {"model": model, "ckpt": ckpt, "mm": mm}
    return _lstm_bundle


@app.get("/health")
def health():
    return {"status": "ok", "seed": RANDOM_SEED, "env": "grid_navigation"}


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
    scaler = joblib.load(MODELS_DIR / "processed_feature_scaler.joblib")
    from utils.data_pipeline import FEATURE_COLS_FOR_SCALING

    i = FEATURE_COLS_FOR_SCALING.index("occupancy_rate")
    lo, hi = float(scaler.data_min_[i]), float(scaler.data_max_[i])
    rate = float(np.clip(pred * (hi - lo) + lo, 0.0, 1.0))
    agg_test = agg[agg["split"] == "test"].reset_index(drop=True)
    t_last = agg_test["LastUpdated"].iloc[-1]
    return PredictResponse(last_updated=str(t_last), y_pred_occupancy_rate=rate)


class ActResponse(BaseModel):
    actions: list[int]
    total_reward: float
    success: bool


@app.post("/act", response_model=ActResponse)
def act():
    """Bir bölümü PPO ile baştan sona oynatır; eylem dizisi döner."""
    global _ppo_model, _ppo_vec
    if _ppo_model is None:
        cfgs = _episode_configs()

        def _make():
            return GridParkingNavigationEnv(cfgs, seed=RANDOM_SEED, max_episode_steps=250)

        venv = DummyVecEnv([_make])
        for p in (MODELS_DIR / "vecnormalize_ppo.pkl", MODELS_DIR / "vecnormalize_ppo"):
            if p.exists():
                _ppo_vec = VecNormalize.load(str(p), venv)
                break
        else:
            raise HTTPException(500, "VecNormalize yok")
        _ppo_vec.training = False
        _ppo_vec.norm_reward = False
        _ppo_model = PPO.load(
            str(MODELS_DIR / "ppo_agent"),
            env=_ppo_vec,
            custom_objects={"lr_schedule": lambda _: 3e-4},
        )

    obs = _ppo_vec.reset()
    actions: list[int] = []
    total = 0.0
    done = False
    success = False
    while not done:
        a, _ = _ppo_model.predict(obs, deterministic=True)
        actions.append(int(a[0]))
        obs, r, dones, infos = _ppo_vec.step(a)
        total += float(r[0])
        done = bool(dones[0])
        inf = infos[0] if isinstance(infos, (list, tuple)) else infos
        if isinstance(inf, dict) and inf.get("success"):
            success = True
    return ActResponse(actions=actions, total_reward=total, success=success)


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
