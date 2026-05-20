"""
Parking Birmingham (Kaggle) verisi — proje künyesi ile uyumlu özellik üretimi.

- Aykırı değer: lot bazında IQR (`utils.outliers`, `ml_config.OUTLIER_*`) — split öncesi.
- Eksik Occupancy: lot (SystemCodeNumber) içinde zamana göre ileri doldurma (ffill).
- Lag / rolling mean ve rolling variance yalnızca geçmişe bakan shift + rolling.
- Zaman öznitelikleri: saat, gün, hafta içi/sonu; hava benzeri türetilmiş mevsimsel sin/cos.
- Ölçekleme (künye 5): `FEATURE_SCALER` ile **MinMaxScaler** veya **StandardScaler** (train’de fit).
- Train/val/test: benzersiz zaman damgası ekseninde %70 / %15 / %15.
- Çıktı: data/processed.parquet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Literal, Tuple

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from ml_config import (
    FEATURE_SCALER,
    OUTLIER_FILTER_ENABLED,
    OUTLIER_IQR_MULTIPLIER,
    OUTLIER_MIN_LOT_SAMPLES,
    RANDOM_SEED,
)
from paths import DATA_PROCESSED, DATA_RAW, MODELS_DIR, ensure_all_standard_dirs
from utils.outliers import remove_occupancy_rate_outliers_iqr


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
    "roll_var_7",
    "roll_var_24",
    "hour",
    "day_of_week",
    "is_weekend",
    "month_sin",
    "month_cos",
    "day_of_year_sin",
    "day_of_year_cos",
]


def _load_raw(csv_path: Path) -> pd.DataFrame:
    """Ham CSV’yi data_preparation ile uyumlu temel temizlikten geçirir (data_preparation.py’ye benzer)."""
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
    """Her satıra train/val/test etiketi; kesimler benzersiz zaman ekseninde (veri sızıntısı yok)."""
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
    """
    Lot bazında gecikme, kayan ortalama ve kayan varyans; yalnızca geçmişe bakan shift/rolling.

    Künye (öznitelik çıkarımı): lag, rolling mean, moving variance, saat/gün/hafta sonu,
    normalizasyona giren mevsimsel döngüsel kodlama (hava/ mevsim benzeri türetilmiş sinyal).
    """
    df = df.copy()
    lot_key = df["SystemCodeNumber"].astype(str)
    df["_lot"] = lot_key
    df = df.sort_values(["_lot", "LastUpdated"], kind="mergesort")

    # Eksik doluluk: aynı lot içinde zamana göre ileri/geri doldurma (künye: eksik veri tamamlama)
    df["Occupancy"] = pd.to_numeric(df["Occupancy"], errors="coerce")
    df["Occupancy"] = df.groupby("_lot", sort=False)["Occupancy"].ffill()
    df["Occupancy"] = df.groupby("_lot", sort=False)["Occupancy"].bfill()

    df["occupancy_rate"] = df["Occupancy"] / df["Capacity"]

    df["hour"] = df["LastUpdated"].dt.hour.astype(np.float64)
    df["day_of_week"] = df["LastUpdated"].dt.dayofweek.astype(np.float64)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(np.float64)
    month = df["LastUpdated"].dt.month.astype(np.float64)
    df["month_sin"] = np.sin(2.0 * np.pi * (month - 1.0) / 12.0)
    df["month_cos"] = np.cos(2.0 * np.pi * (month - 1.0) / 12.0)
    doy = df["LastUpdated"].dt.dayofyear.astype(np.float64)
    df["day_of_year_sin"] = np.sin(2.0 * np.pi * (doy - 1.0) / 366.0)
    df["day_of_year_cos"] = np.cos(2.0 * np.pi * (doy - 1.0) / 366.0)

    g = df.groupby("_lot", sort=False)["occupancy_rate"]
    for k in LAG_STEPS:
        df[f"lag_{k}"] = g.shift(k)
    for w in ROLL_WINDOWS:
        # Yalnızca geçmiş: t anında t-1.. dahil pencere
        df[f"roll_mean_{w}"] = g.transform(
            lambda s, ww=w: s.shift(1).rolling(ww, min_periods=1).mean()
        )
        df[f"roll_var_{w}"] = g.transform(
            lambda s, ww=w: s.shift(1).rolling(ww, min_periods=1).var()
        )
        df[f"roll_var_{w}"] = df[f"roll_var_{w}"].fillna(0.0)

    df = df.drop(columns=["_lot"])
    df = df.dropna().reset_index(drop=True)
    return df


ScalerKind = Literal["minmax", "standard"]


def _scale_train_only(
    df: pd.DataFrame,
    scaler_dir: Path,
    scaler_kind: ScalerKind,
) -> tuple[pd.DataFrame, MinMaxScaler | StandardScaler]:
    train_mask = df["split"] == "train"
    X_train_only = df.loc[train_mask, FEATURE_COLS_FOR_SCALING].to_numpy(dtype=np.float64)
    X_all = df[FEATURE_COLS_FOR_SCALING].to_numpy(dtype=np.float64)

    if scaler_kind == "standard":
        scaler: MinMaxScaler | StandardScaler = StandardScaler()
        scaler.fit(X_train_only)
        scaled = scaler.transform(X_all)
        label = "StandardScaler"
    else:
        scaler = MinMaxScaler()
        scaler.fit(X_train_only)
        scaled = scaler.transform(X_all)
        label = "MinMaxScaler"

    scaled_df = pd.DataFrame(
        scaled,
        columns=[f"{c}_mm" for c in FEATURE_COLS_FOR_SCALING],
        index=df.index,
    )
    out = pd.concat([df.reset_index(drop=True), scaled_df], axis=1)

    scaler_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = scaler_dir / "processed_feature_scaler.joblib"
    joblib.dump({"kind": scaler_kind, "scaler": scaler}, bundle_path)
    print(f"[pipeline] {label} ({scaler_kind}) kaydedildi: {bundle_path}")
    return out, scaler


def load_feature_scaler(models_dir: Path | None = None) -> Tuple[Any, str]:
    """Eski düz joblib (yalnızca scaler) ve yeni {'kind','scaler'} biçimini destekler."""
    models_dir = models_dir or MODELS_DIR
    path = models_dir / "processed_feature_scaler.joblib"
    obj = joblib.load(path)
    if isinstance(obj, dict) and "scaler" in obj:
        return obj["scaler"], str(obj.get("kind", "minmax")).lower()
    return obj, "minmax"


def apply_outlier_filter(df: pd.DataFrame, enabled: bool | None = None) -> pd.DataFrame:
    """Ham/temiz tabloda künye aykırı değer adımı (split ve lag öncesi)."""
    use = OUTLIER_FILTER_ENABLED if enabled is None else enabled
    if not use:
        print("[pipeline] Aykırı değer filtresi kapalı.")
        return df
    out, stats = remove_occupancy_rate_outliers_iqr(
        df,
        iqr_multiplier=OUTLIER_IQR_MULTIPLIER,
        min_samples_per_lot=OUTLIER_MIN_LOT_SAMPLES,
    )
    print(
        f"[pipeline] Aykırı değer (IQR×{stats['iqr_multiplier']}): "
        f"{stats['rows_before']} -> {stats['rows_after']} satır "
        f"({stats['removed']} silindi, %{stats['removed_pct']})"
    )
    return out


def build_processed_dataset(
    raw_path: Path | None = None,
    output_path: Path | None = None,
    scaler_kind: ScalerKind | None = None,
    outlier_filter: bool | None = None,
) -> pd.DataFrame:
    ensure_all_standard_dirs()
    raw_path = raw_path or (DATA_RAW / "parking.csv")
    output_path = output_path or (DATA_PROCESSED / "processed.parquet")

    if not raw_path.exists():
        raise FileNotFoundError(f"Ham veri yok: {raw_path}")

    df = _load_raw(raw_path)
    df = apply_outlier_filter(df, enabled=outlier_filter)
    df["split"] = _split_masks_by_timestamp(df["LastUpdated"])
    df = _add_per_lot_history_features(df)

    sk = scaler_kind or str(FEATURE_SCALER).strip().lower()
    if sk not in ("minmax", "standard"):
        raise ValueError(f"Geçersiz scaler_kind: {sk!r} (minmax | standard)")
    out, _ = _scale_train_only(df, MODELS_DIR, sk)  # type: ignore[arg-type]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_path, index=False)
    print(f"[pipeline] Parquet yazıldı: {output_path} ({len(out)} satır)")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="processed.parquet üret")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument(
        "--scaler",
        choices=["minmax", "standard"],
        default=None,
        help="Özellik ölçekleyici (varsayılan: ml_config.FEATURE_SCALER)",
    )
    parser.add_argument(
        "--no-outliers",
        action="store_true",
        help="Lot bazında IQR aykırı değer filtresini kapat (varsayılan: ml_config)",
    )
    args = parser.parse_args()
    np.random.seed(args.seed)
    build_processed_dataset(
        scaler_kind=args.scaler,
        outlier_filter=False if args.no_outliers else None,
    )


if __name__ == "__main__":
    main()
