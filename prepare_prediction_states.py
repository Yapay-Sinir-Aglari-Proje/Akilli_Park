"""
LSTM tahminlerini RL state CSV’sine çevirir (per-lot, LSTM oranı capacity ile ölçeklenir).

Koordinatlar: utils.coordinates.stable_parking_coordinates
Model: models/lstm_model.pt (PyTorch)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from ml_config import RANDOM_SEED
from paths import DATA_PROCESSED, MODELS_DIR, PREDICTIONS_DIR, ensure_data_processed
from utils.coordinates import stable_parking_coordinates
from utils.lstm_inference import predict_occupancy_rate_series
from utils.seeds import set_global_seed

LOGGER = logging.getLogger(__name__)

REQUIRED_COLUMNS = {
    "timestamp",
    "parking_id",
    "predicted_occupancy",
    "capacity",
    "latitude",
    "longitude",
}

DEFAULT_OUTPUT_PATH = DATA_PROCESSED / "rl_prediction_states.csv"


def load_or_create_predictions(
    lstm_predictions_path: Path | None = None,
) -> pd.DataFrame:
    lstm_predictions_path = lstm_predictions_path or (PREDICTIONS_DIR / "test_predictions.csv")
    if lstm_predictions_path.exists():
        p = pd.read_csv(lstm_predictions_path)
        p["ts"] = pd.to_datetime(p["LastUpdated"])
        return p
    LOGGER.warning("Tahmin CSV yok, LSTM çıktısı üretiliyor.")
    pred = predict_occupancy_rate_series()
    pred["ts"] = pd.to_datetime(pred["LastUpdated"])
    return pred


def create_rl_features(predictions: pd.DataFrame) -> pd.DataFrame:
    df = predictions.copy()
    if (df["capacity"] <= 0).any():
        raise ValueError("capacity > 0 olmalı.")

    df["predicted_occupancy"] = df["predicted_occupancy"].clip(lower=0)
    df["predicted_occupancy"] = np.minimum(df["predicted_occupancy"], df["capacity"])

    df["predicted_empty_slots"] = df["capacity"] - df["predicted_occupancy"]
    df["occupancy_ratio"] = (df["predicted_occupancy"] / df["capacity"]).clip(0.0, 1.0)

    grouped = df.groupby("timestamp")["predicted_empty_slots"]
    min_vals = grouped.transform("min")
    max_vals = grouped.transform("max")
    denominator = (max_vals - min_vals).replace(0, np.nan)
    normalized = (df["predicted_empty_slots"] - min_vals) / denominator
    df["availability_score"] = normalized.fillna(1.0).clip(0.0, 1.0)

    conditions = [
        df["occupancy_ratio"] < 0.5,
        (df["occupancy_ratio"] >= 0.5) & (df["occupancy_ratio"] < 0.8),
        df["occupancy_ratio"] >= 0.8,
    ]
    df["risk_level"] = np.select(conditions, ["LOW", "MEDIUM", "HIGH"], default="HIGH")

    hour = df["timestamp"].dt.hour.astype(float)
    df["time_sin"] = np.sin(2 * np.pi * hour / 24.0)
    df["time_cos"] = np.cos(2 * np.pi * hour / 24.0)
    df["distance_placeholder"] = 0.0
    df["expected_wait_time"] = df["occupancy_ratio"] * 10.0
    return df


def build_state_vector(df: pd.DataFrame) -> pd.DataFrame:
    vector_columns: List[str] = [
        "occupancy_ratio",
        "predicted_empty_slots",
        "availability_score",
        "expected_wait_time",
        "time_sin",
        "time_cos",
    ]

    def _to_vector(row: pd.Series) -> List[float]:
        return [float(row[col]) for col in vector_columns]

    out = df.copy()
    out["rl_state_vector"] = out[vector_columns].apply(_to_vector, axis=1)
    return out


def build_per_lot_predictions_from_test(
    pred_rates: pd.DataFrame,
    processed_parquet: Path = DATA_PROCESSED / "processed.parquet",
) -> pd.DataFrame:
    """Aggregate tahmin oranını her lot satırına yayar."""
    df = pd.read_parquet(processed_parquet)
    df = df[df["split"] == "test"].copy()
    df["LastUpdated"] = pd.to_datetime(df["LastUpdated"])
    rate_map = {
        pd.Timestamp(r.ts): float(r.y_pred_occupancy_rate)
        for r in pred_rates.itertuples()
    }
    coords = stable_parking_coordinates(
        sorted(df["SystemCodeNumber"].astype(str).unique())
    )
    rows = []
    for _, row in df.iterrows():
        ts = pd.Timestamp(row["LastUpdated"])
        cap = float(row["Capacity"])
        pid = str(row["SystemCodeNumber"])
        rate = rate_map.get(ts, np.nan)
        if np.isnan(rate):
            continue
        rate = float(np.clip(rate, 0.0, 1.0))
        pred_occ = float(np.clip(rate * cap, 0.0, cap))
        lat, lon = coords.get(pid, (0.0, 0.0))
        rows.append(
            {
                "timestamp": ts,
                "parking_id": pid,
                "predicted_occupancy": pred_occ,
                "capacity": cap,
                "latitude": lat,
                "longitude": lon,
            }
        )
    return pd.DataFrame(rows)


def save_rl_states(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def main(
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> None:
    set_global_seed(RANDOM_SEED)
    ensure_data_processed()
    pred_rates = load_or_create_predictions()
    predictions = build_per_lot_predictions_from_test(pred_rates)
    if predictions.empty:
        raise ValueError("RL tahmin tablosu boş.")
    rl_features = create_rl_features(predictions)
    rl_states = build_state_vector(rl_features)
    save_rl_states(rl_states, output_path)
    LOGGER.info("RL state CSV hazır: %s", output_path)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    main()
