"""PyTorch LSTM ile aggregate doluluk oranı tahmini."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from paths import DATA_PROCESSED, MODELS_DIR
from utils.data_pipeline import FEATURE_COLS_FOR_SCALING
from utils.lstm_core import (
    OccupancyLSTM,
    load_aggregate_frame,
    make_sequences,
    mm_cols,
    prepend_context,
)


def load_lstm_bundle(path: Path | None = None) -> tuple[OccupancyLSTM, dict, int]:
    path = path or (MODELS_DIR / "lstm_model.pt")
    ckpt = torch.load(path, map_location="cpu")
    mm = ckpt["input_cols"]
    model = OccupancyLSTM(
        input_dim=len(mm),
        hidden=ckpt.get("hidden", 64),
        num_layers=ckpt.get("num_layers", 2),
        dropout=ckpt.get("dropout", 0.2),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    ts = int(ckpt["time_step"])
    return model, ckpt, ts


def predict_occupancy_rate_series(
    parquet_path: Path | None = None,
    model_path: Path | None = None,
) -> pd.DataFrame:
    """Test dilimindeki zamanlar için aggregate tahmin."""
    parquet_path = parquet_path or (DATA_PROCESSED / "processed.parquet")
    model, _, ts = load_lstm_bundle(model_path)

    agg = load_aggregate_frame(parquet_path)
    mm = mm_cols()
    train_part = agg[agg["split"] == "train"][mm].to_numpy(dtype=np.float32)
    val_part = agg[agg["split"] == "val"][mm].to_numpy(dtype=np.float32)
    test_part = agg[agg["split"] == "test"][mm].to_numpy(dtype=np.float32)

    tv = prepend_context(train_part[-ts:], val_part)
    test_block = prepend_context(tv[-ts:], test_part)
    X_test, _ = make_sequences(test_block, ts)

    with torch.no_grad():
        pred = model(torch.from_numpy(X_test)).numpy()

    scaler = joblib.load(MODELS_DIR / "processed_feature_scaler.joblib")
    i = FEATURE_COLS_FOR_SCALING.index("occupancy_rate")
    lo, hi = float(scaler.data_min_[i]), float(scaler.data_max_[i])
    pred_rate = np.clip(pred * (hi - lo) + lo, 0.0, 1.0)

    agg_test = agg[agg["split"] == "test"].reset_index(drop=True)
    times = agg_test["LastUpdated"].iloc[: len(pred_rate)]

    return pd.DataFrame(
        {
            "LastUpdated": times.values,
            "y_pred_occupancy_rate": pred_rate,
        }
    )
