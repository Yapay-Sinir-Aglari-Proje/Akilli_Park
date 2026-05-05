"""
LSTM çıktısını RL tarafının okuyacağı feature tablosuna çevirir.

Akış:
1) train/val/test işlenmiş CSV’leri oku (zaman sırası korunmuş parçalar).
2) LSTM ile aynı şekilde her parça için zaman başına ortalama doluluk oranı üret.
3) Scaler **eğitimde kaydedilen** joblib dosyasından yüklenir (test üzerinde yeniden fit YOK).
4) Tam zaman ekseni üzerinde kaydırmalı pencere ile tahmin; her test zamanı için
   global tahmin edilen oran, satırdaki Capacity ile çarpılarak predicted_occupancy olur.
5) Çıktı: data/processed/rl_prediction_states.csv

Koordinat yoksa layout.stable_parking_coordinates ile deterministik lat/lon üretilir.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

from ml_config import LSTM_TIME_STEP
from paths import DATA_PROCESSED, MODELS_DIR, ensure_data_processed
from parking_rl.layout import stable_parking_coordinates

LOGGER = logging.getLogger(__name__)

REQUIRED_COLUMNS = {
    "timestamp",
    "parking_id",
    "predicted_occupancy",
    "capacity",
    "latitude",
    "longitude",
}

DEFAULT_INPUT_PATH = DATA_PROCESSED / "lstm_predictions.csv"
DEFAULT_OUTPUT_PATH = DATA_PROCESSED / "rl_prediction_states.csv"
DEFAULT_MODEL_PATH = MODELS_DIR / "lstm_parking_model.h5"
DEFAULT_SCALER_PATH = MODELS_DIR / "lstm_occupancy_scaler.joblib"


def aggregate_mean_occupancy_rate(df: pd.DataFrame) -> pd.DataFrame:
    """lstm_model.py ile aynı mantık (tek kaynak tutarlılığı)."""
    occ = pd.to_numeric(df["Occupancy"], errors="coerce")
    cap = pd.to_numeric(df["Capacity"], errors="coerce")
    tmp = pd.DataFrame({"LastUpdated": df["LastUpdated"], "occ": occ, "cap": cap})
    tmp = tmp.dropna()
    tmp = tmp[tmp["cap"] > 0]
    tmp["occupancy_rate"] = tmp["occ"] / tmp["cap"]
    return (
        tmp.groupby("LastUpdated", sort=False)["occupancy_rate"]
        .mean()
        .reset_index()
        .sort_values("LastUpdated", kind="mergesort")
    )


def create_sequences(data: np.ndarray, time_step: int) -> np.ndarray:
    """Sadece X tensörü (tahmin için)."""
    X = []
    for i in range(len(data) - time_step):
        X.append(data[i : i + time_step])
    return np.asarray(X, dtype=np.float32).reshape(-1, time_step, 1)


def prepend_context(prev_tail: np.ndarray, block: np.ndarray) -> np.ndarray:
    if len(prev_tail) == 0:
        return block
    return np.vstack([prev_tail, block])


def generate_lstm_predictions(
    output_path: Path = DEFAULT_INPUT_PATH,
    model_path: Path = DEFAULT_MODEL_PATH,
    scaler_path: Path = DEFAULT_SCALER_PATH,
    time_step: int = LSTM_TIME_STEP,
) -> pd.DataFrame:
    """
    Birleşik zaman ekseninde LSTM çalıştırır; test dilimindeki her zaman için
    tahmin edilen *ortalama doluluk oranını* üretir ve lstm_predictions.csv yazar.
    """
    ensure_data_processed()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        raise FileNotFoundError(
            f"LSTM model yok: {model_path}. Önce `python lstm_model.py` çalıştırın."
        )
    if not scaler_path.exists():
        raise FileNotFoundError(
            f"Scaler yok: {scaler_path}. LSTM eğitimi scaler kaydetmeli."
        )

    model = load_model(model_path, compile=False)
    scaler = joblib.load(scaler_path)

    train_df = pd.read_csv(DATA_PROCESSED / "train.csv")
    val_df = pd.read_csv(DATA_PROCESSED / "val.csv")
    test_df = pd.read_csv(DATA_PROCESSED / "test.csv")

    for name, d in ("train", train_df), ("val", val_df), ("test", test_df):
        if d.empty:
            raise ValueError(f"{name}.csv boş.")

    for d in (train_df, val_df, test_df):
        d["LastUpdated"] = pd.to_datetime(d["LastUpdated"], errors="coerce")

    agg_train = aggregate_mean_occupancy_rate(train_df)
    agg_val = aggregate_mean_occupancy_rate(val_df)
    agg_test = aggregate_mean_occupancy_rate(test_df)

    train_v = agg_train["occupancy_rate"].to_numpy(float).reshape(-1, 1)
    val_v = agg_val["occupancy_rate"].to_numpy(float).reshape(-1, 1)
    test_v = agg_test["occupancy_rate"].to_numpy(float).reshape(-1, 1)

    train_s = scaler.transform(train_v)
    val_s = scaler.transform(val_v)
    test_s = scaler.transform(test_v)

    # Test bloğu: lstm_model.py ile aynı bağlam (train+val sonu + test)
    tv = np.vstack([train_s, val_s])
    test_block = prepend_context(tv[-time_step:], test_s)

    if len(test_block) <= time_step:
        raise ValueError("Test zaman serisi LSTM penceresi için çok kısa.")

    X_test = create_sequences(test_block, time_step)
    pred_scaled = model.predict(X_test, verbose=0)
    pred_rate = scaler.inverse_transform(pred_scaled).reshape(-1)

    # pred_rate[i], birleşik test bloğunda i. test zaman adımının tahmini oranına karşılık gelir
    # (bağlam satırları X üretiminde kullanıldı; yine de her agg_test zamanı için bir çıktı üretilir)
    test_times = agg_test["LastUpdated"].to_numpy()
    if len(pred_rate) != len(test_times):
        raise RuntimeError(
            f"Tahmin ({len(pred_rate)}) ile test zaman sayısı ({len(test_times)}) uyuşmuyor."
        )

    rate_by_time = {
        pd.Timestamp(test_times[i]): float(pred_rate[i]) for i in range(len(pred_rate))
    }

    coords_map = stable_parking_coordinates(
        sorted(test_df["SystemCodeNumber"].astype(str).unique())
    )

    rows = []
    for _, row in test_df.iterrows():
        ts = pd.Timestamp(row["LastUpdated"])
        cap = float(row["Capacity"])
        pid = str(row["SystemCodeNumber"])
        rate = rate_by_time.get(ts, np.nan)
        if np.isnan(rate):
            continue
        rate = float(np.clip(rate, 0.0, 1.0))
        pred_occ = float(np.clip(rate * cap, 0.0, cap))
        lat, lon = coords_map.get(pid, (0.0, 0.0))
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

    predictions = pd.DataFrame(rows)
    if predictions.empty:
        raise ValueError("Tahmin tablosu boş; zaman eşleşmesi veya veri filtresi kontrol edin.")

    predictions.to_csv(output_path, index=False)
    LOGGER.info("LSTM tahminleri yazıldı: %s", output_path)
    return predictions


def load_predictions(input_path: Path) -> pd.DataFrame:
    """Tahmin CSV varsa oku; yoksa LSTM ile üret."""
    if not input_path.exists():
        LOGGER.warning("Tahmin dosyası yok (%s), LSTM ile üretiliyor.", input_path)
        return generate_lstm_predictions(output_path=input_path)

    predictions = pd.read_csv(input_path)
    if predictions.empty:
        raise ValueError(f"Tahmin CSV boş: {input_path}")

    missing = REQUIRED_COLUMNS - set(predictions.columns)
    if missing:
        raise ValueError(f"Eksik kolonlar: {sorted(missing)}")

    predictions["timestamp"] = pd.to_datetime(predictions["timestamp"], errors="coerce")
    if predictions["timestamp"].isna().any():
        raise ValueError("Geçersiz timestamp satırları var.")
    return predictions


def _normalize_empty_slots_within_timestamp(df: pd.DataFrame) -> pd.Series:
    grouped = df.groupby("timestamp")["predicted_empty_slots"]
    min_vals = grouped.transform("min")
    max_vals = grouped.transform("max")
    denominator = (max_vals - min_vals).replace(0, np.nan)
    normalized = (df["predicted_empty_slots"] - min_vals) / denominator
    return normalized.fillna(1.0).clip(0.0, 1.0)


def create_rl_features(predictions: pd.DataFrame) -> pd.DataFrame:
    df = predictions.copy()
    if (df["capacity"] <= 0).any():
        raise ValueError("capacity > 0 olmalı.")

    df["predicted_occupancy"] = df["predicted_occupancy"].clip(lower=0)
    df["predicted_occupancy"] = np.minimum(df["predicted_occupancy"], df["capacity"])

    df["predicted_empty_slots"] = df["capacity"] - df["predicted_occupancy"]
    df["occupancy_ratio"] = (df["predicted_occupancy"] / df["capacity"]).clip(0.0, 1.0)
    df["availability_score"] = _normalize_empty_slots_within_timestamp(df)

    conditions = [
        df["occupancy_ratio"] < 0.5,
        (df["occupancy_ratio"] >= 0.5) & (df["occupancy_ratio"] < 0.8),
        df["occupancy_ratio"] >= 0.8,
    ]
    df["risk_level"] = np.select(
        conditions, ["LOW", "MEDIUM", "HIGH"], default="HIGH"
    )

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


def save_rl_states(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def main(
    input_path: Path = DEFAULT_INPUT_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> None:
    ensure_data_processed()
    predictions = load_predictions(input_path)
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
