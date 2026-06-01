"""
Zaman serisi tahmin ortak yapı taşları (aggregate seri).

- `load_aggregate_frame`, `make_sequences`, `prepend_context`
- Tahmin modelleri (künye 6.1): `OccupancyLSTM`, `OccupancyGRU`, `OccupancyTemporalTransformer`
  — tanımlar `utils.forecast_models` içindedir; buradan yeniden dışa aktarılır.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from utils.data_pipeline import FEATURE_COLS_FOR_SCALING
from utils.forecast_models import (
    OccupancyGRU,
    OccupancyLSTM,
    OccupancyTemporalTransformer,
    build_forecast_model,
    load_model_from_checkpoint,
)


def mm_cols() -> list[str]:
    """Ölçeklenmiş özellik sütun adları (data_pipeline.FEATURE_COLS_FOR_SCALING + '_mm')."""
    return [f"{c}_mm" for c in FEATURE_COLS_FOR_SCALING]


def load_aggregate_frame(parquet_path) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path)
    cols = mm_cols()
    agg = (
        df.groupby("LastUpdated", sort=False)
        .agg({**{c: "mean" for c in cols}, "split": "first"})
        .reset_index()
        .sort_values("LastUpdated", kind="mergesort")
        .reset_index(drop=True)
    )
    return agg


def prepend_context(prev: np.ndarray, block: np.ndarray) -> np.ndarray:
    """Önceki split’in son satırlarını başa ekleyerek val/test dizilerinde bağlam kaybını önler."""
    if len(prev) == 0:
        return block
    return np.vstack([prev, block])


def make_sequences(data: np.ndarray, time_step: int) -> tuple[np.ndarray, np.ndarray]:
    x_list, y_list = [], []
    for i in range(len(data) - time_step):
        x_list.append(data[i : i + time_step])
        y_list.append(data[i + time_step, 0])
    return np.asarray(x_list, dtype=np.float32), np.asarray(y_list, dtype=np.float32)


def inv_occupancy_rate(
    y_mm: np.ndarray,
    scaler: MinMaxScaler | StandardScaler,
    scaler_kind: str | None = None,
) -> np.ndarray:
    """
    Tahmin hedefi occupancy_rate_mm → gerçek [0,1] doluluk oranı.

    scaler_kind: 'minmax' | 'standard' — None ise scaler tipinden çıkarım.
    """
    i = FEATURE_COLS_FOR_SCALING.index("occupancy_rate")
    if isinstance(scaler, StandardScaler):
        kind = "standard"
    else:
        kind = (scaler_kind or "minmax").lower()
        if kind != "standard":
            kind = "minmax"
    if kind == "standard":
        m = float(scaler.mean_[i])
        s = float(scaler.scale_[i])
        if s < 1e-12:
            s = 1.0
        out = y_mm * s + m
        return np.clip(out, 0.0, 1.0).astype(np.float64)
    lo, hi = float(scaler.data_min_[i]), float(scaler.data_max_[i])
    out = y_mm * (hi - lo) + lo
    return np.clip(out, 0.0, 1.0).astype(np.float64)
