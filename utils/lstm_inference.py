"""
PyTorch tahmin modeli (LSTM / GRU / Transformer) ile toplu (aggregate) doluluk oranı tahmini.

`train_lstm.py` ile kaydedilen `lstm_model.pt` ve `processed_feature_scaler.joblib`
dosyalarını okur; test dilimindeki her zaman adımı için bir tahmin üretir.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from paths import DATA_PROCESSED, MODELS_DIR
from utils.data_pipeline import load_feature_scaler
from utils.lstm_core import (
    inv_occupancy_rate,
    load_aggregate_frame,
    load_model_from_checkpoint,
    make_sequences,
    mm_cols,
    prepend_context,
)


def load_lstm_bundle(path: Path | None = None) -> tuple[nn.Module, dict, int]:
    """Checkpoint’ten model yükler (künye mimarileri); time_step meta verisini döner."""
    path = path or (MODELS_DIR / "lstm_model.pt")
    ckpt = torch.load(path, map_location="cpu")
    model = load_model_from_checkpoint(ckpt)
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
        pred = model(torch.from_numpy(X_test).float()).numpy()

    scaler, sk = load_feature_scaler(MODELS_DIR)
    pred_rate = inv_occupancy_rate(pred.astype(np.float64), scaler, sk)

    agg_test = agg[agg["split"] == "test"].reset_index(drop=True)
    times = agg_test["LastUpdated"].iloc[: len(pred_rate)]

    return pd.DataFrame(
        {
            "LastUpdated": times.values,
            "y_pred_occupancy_rate": pred_rate,
        }
    )
