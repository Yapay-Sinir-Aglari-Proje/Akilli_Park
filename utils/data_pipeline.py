"""
Birmingham otopark verisi: özellik üretimi + zaman bazlı bölme + MinMax (sızıntı yok).

- Lag / rolling yalnızca geçmiş gözlemleri kullanır (shift + rolling).
- MinMaxScaler yalnızca train satırlarında fit edilir.
- Çıktı: data/processed.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from ml_config import RANDOM_SEED
from paths import DATA_PROCESSED, DATA_RAW, MODELS_DIR, ensure_all_standard_dirs


LAG_STEPS = (1, 3, 6, 12, 24)
ROLL_WINDOWS = (7, 24)

FEATURE_COLS_FOR_SCALING = [
    "occupancy_rate",
    "lag_1",
    "lag_3",
    "lag_6",
    "lag_12",
    "lag_24",
    "roll_mean_7",
    "roll_mean_24",
    "hour",
    "day_of_week",
    "is_weekend",
]


def _load_raw(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    before = len(df)
    df = df.drop_duplicates()
    cap = pd.to_numeric(df["Capacity"], errors="coerce")
    occ = pd.to_numeric(df["Occupancy"], errors="coerce")
    df = df.copy()
    df["Capacity"] = cap
    df["Occupancy"] = occ
    df = df.dropna(subset=["Capacity", "Occupancy"])
    df = df[df["Capacity"] > 0]
    df = df[(df["Occupancy"] >= 0) & (df["Occupancy"] <= df["Capacity"])]
    df["LastUpdated"] = pd.to_datetime(df["LastUpdated"], errors="coerce")
    df = df.dropna(subset=["LastUpdated"])
    df = df.sort_values("LastUpdated", kind="mergesort").reset_index(drop=True)
    print(f"[pipeline] Temizlik: {before} -> {len(df)} satır")
    return df


def _split_masks_by_timestamp(
    timestamps: pd.Series,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> pd.Series:
    unique_times = timestamps.drop_duplicates().sort_values(kind="mergesort")
    n_t = len(unique_times)
    if n_t < 3:
        raise ValueError(f"En az 3 benzersiz zaman gerekli, gelen: {n_t}")

    idx_train = int(n_t * train_ratio)
    idx_val_end = int(n_t * (train_ratio + val_ratio))
    idx_train = max(1, min(idx_train, n_t - 2))
    idx_val_end = max(idx_train + 1, min(idx_val_end, n_t - 1))

    train_times = set(unique_times.iloc[:idx_train])
    val_times = set(unique_times.iloc[idx_train:idx_val_end])
    test_times = set(unique_times.iloc[idx_val_end:])

    if not test_times:
        last_t = unique_times.iloc[-1]
        test_times = {last_t}
        val_times.discard(last_t)
        train_times.discard(last_t)

    def _label(ts: pd.Timestamp) -> str:
        if ts in train_times:
            return "train"
        if ts in val_times:
            return "val"
        if ts in test_times:
            return "test"
        return "train"

    return timestamps.map(_label)


def _add_per_lot_history_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["occupancy_rate"] = df["Occupancy"] / df["Capacity"]

    df["hour"] = df["LastUpdated"].dt.hour.astype(np.float64)
    df["day_of_week"] = df["LastUpdated"].dt.dayofweek.astype(np.float64)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(np.float64)

    lot_key = df["SystemCodeNumber"].astype(str)
    df["_lot"] = lot_key
    df = df.sort_values(["_lot", "LastUpdated"], kind="mergesort")

    g = df.groupby("_lot", sort=False)["occupancy_rate"]
    for k in LAG_STEPS:
        df[f"lag_{k}"] = g.shift(k)
    for w in ROLL_WINDOWS:
        # Yalnızca geçmiş: t anında t-1.. dahil pencere
        df[f"roll_mean_{w}"] = g.transform(
            lambda s, ww=w: s.shift(1).rolling(ww, min_periods=1).mean()
        )

    df = df.drop(columns=["_lot"])
    df = df.dropna().reset_index(drop=True)
    return df


def _scale_train_only(df: pd.DataFrame, scaler_dir: Path) -> tuple[pd.DataFrame, MinMaxScaler]:
    scaler = MinMaxScaler()
    train_mask = df["split"] == "train"
    scaler.fit(df.loc[train_mask, FEATURE_COLS_FOR_SCALING].to_numpy(dtype=np.float64))

    scaled = scaler.transform(df[FEATURE_COLS_FOR_SCALING].to_numpy(dtype=np.float64))
    scaled_df = pd.DataFrame(
        scaled,
        columns=[f"{c}_mm" for c in FEATURE_COLS_FOR_SCALING],
        index=df.index,
    )
    out = pd.concat([df.reset_index(drop=True), scaled_df], axis=1)

    scaler_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, scaler_dir / "processed_feature_scaler.joblib")
    print(f"[pipeline] MinMaxScaler kaydedildi: {scaler_dir / 'processed_feature_scaler.joblib'}")
    return out, scaler


def build_processed_dataset(
    raw_path: Path | None = None,
    output_path: Path | None = None,
) -> pd.DataFrame:
    ensure_all_standard_dirs()
    raw_path = raw_path or (DATA_RAW / "parking.csv")
    output_path = output_path or (DATA_PROCESSED / "processed.parquet")

    if not raw_path.exists():
        raise FileNotFoundError(f"Ham veri yok: {raw_path}")

    df = _load_raw(raw_path)
    df["split"] = _split_masks_by_timestamp(df["LastUpdated"])
    df = _add_per_lot_history_features(df)

    out, _ = _scale_train_only(df, MODELS_DIR)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_path, index=False)
    print(f"[pipeline] Parquet yazıldı: {output_path} ({len(out)} satır)")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="processed.parquet üret")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()
    np.random.seed(args.seed)
    build_processed_dataset()


if __name__ == "__main__":
    main()
