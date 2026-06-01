"""
Künye bölüm 6.1 — XGBoost ve Random Forest ile aggregate doluluk tahmini.

LSTM ile aynı pencere (make_sequences): giriş vektörü son `time_step` adımın tüm
ölçekli özelliklerinin düzleştirilmiş birleşimidir. Metrikler: MAE, RMSE, MAPE, R².
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from ml_config import LSTM_TIME_STEP, RANDOM_SEED
from paths import DATA_PROCESSED, EVALUATION_DIR, MODELS_DIR, ensure_all_standard_dirs
from utils.data_pipeline import build_processed_dataset, load_feature_scaler
from utils.lstm_core import (
    inv_occupancy_rate,
    load_aggregate_frame,
    make_sequences,
    mm_cols,
    prepend_context,
)

try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None  # type: ignore[misc, assignment]


def _mape_pct(y_t: np.ndarray, y_p: np.ndarray) -> float:
    denom = np.maximum(np.abs(y_t), 1e-6)
    return float(np.mean(np.abs((y_t - y_p) / denom)) * 100.0)


def main() -> None:
    ensure_all_standard_dirs()
    pq = DATA_PROCESSED / "processed.parquet"
    if not pq.exists():
        build_processed_dataset(output_path=pq)

    ts = LSTM_TIME_STEP
    agg = load_aggregate_frame(pq)
    mm = mm_cols()
    scaler, sk = load_feature_scaler(MODELS_DIR)

    train_part = agg[agg["split"] == "train"][mm].to_numpy(dtype=np.float32)
    val_part = agg[agg["split"] == "val"][mm].to_numpy(dtype=np.float32)
    test_part = agg[agg["split"] == "test"][mm].to_numpy(dtype=np.float32)

    X_train, y_train = make_sequences(train_part, ts)
    val_block = prepend_context(train_part[-ts:], val_part)
    X_val, y_val = make_sequences(val_block, ts)
    tv = np.vstack([train_part, val_part])
    test_block = prepend_context(tv[-ts:], test_part)
    X_test, y_test = make_sequences(test_block, ts)

    X_train_f = X_train.reshape(len(X_train), -1)
    X_val_f = X_val.reshape(len(X_val), -1)
    X_test_f = X_test.reshape(len(X_test), -1)
    X_tv_f = np.vstack([X_train_f, X_val_f])
    y_tv = np.concatenate([y_train, y_val])

    results: dict = {"seed": RANDOM_SEED, "time_step": ts, "models": {}}

    estimators: dict = {
        "random_forest": RandomForestRegressor(
            n_estimators=250,
            random_state=RANDOM_SEED,
            n_jobs=-1,
            max_depth=24,
        ),
    }
    if XGBRegressor is not None:
        estimators["xgboost"] = XGBRegressor(
            n_estimators=400,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=RANDOM_SEED,
            n_jobs=-1,
            tree_method="hist",
        )
    else:
        results["note"] = "xgboost paketi yok; yalnızca RandomForest eğitildi."

    for name, reg in estimators.items():
        reg.fit(X_tv_f, y_tv)
        pred_mm = reg.predict(X_test_f).astype(np.float64)
        y_true_o = inv_occupancy_rate(y_test, scaler, sk)
        y_pred_o = inv_occupancy_rate(pred_mm, scaler, sk)
        mae = float(mean_absolute_error(y_true_o, y_pred_o))
        rmse = float(np.sqrt(mean_squared_error(y_true_o, y_pred_o)))
        mape = _mape_pct(y_true_o, y_pred_o)
        r2 = float(r2_score(y_true_o, y_pred_o))
        results["models"][name] = {"mae": mae, "rmse": rmse, "mape_pct": mape, "r2": r2}
        print(f"[tabular:{name}] MAE={mae:.6f} RMSE={rmse:.6f} MAPE={mape:.2f}% R2={r2:.4f}")

    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    out_path = EVALUATION_DIR / "tabular_baselines.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[tabular] Rapor: {out_path}")


if __name__ == "__main__":
    main()
