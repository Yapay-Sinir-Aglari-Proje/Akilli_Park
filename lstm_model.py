"""
LSTM ile otopark doluluk oranı tahmini (zaman serisi).

Önemli tasarım kararları:
- Ham veri çoklu otopark satırı içerir; tek bir LSTM için her zaman diliminde
  tüm otoparkların doluluk oranı ortalaması alınır.
- Model tek feature ile çalışır: occupancy_rate.
- MinMaxScaler yalnızca train üzerinde fit edilir; val/test sadece transform.
- Val/test dizileri için bir önceki parçanın son time_step gözlemi eklenir.
- Model parametreleri ml_config.py dosyasından alınır.
"""

from __future__ import annotations

import random

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential

from ml_config import (
    LSTM_TIME_STEP,
    LSTM_EPOCHS,
    LSTM_BATCH_SIZE,
    RANDOM_SEED,
)
from paths import DATA_PROCESSED, MODELS_DIR, OUTPUT_DIR, ensure_models, ensure_output


np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


def aggregate_mean_occupancy_rate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aynı LastUpdated altındaki tüm kayıtlar için occupancy_rate ortalaması.
    Böylece model gerçekten zaman ekseninde tek değişkenli seri görür.
    """

    occ = pd.to_numeric(df["Occupancy"], errors="coerce")
    cap = pd.to_numeric(df["Capacity"], errors="coerce")

    tmp = pd.DataFrame(
        {
            "LastUpdated": df["LastUpdated"],
            "occ": occ,
            "cap": cap,
        }
    )

    tmp = tmp.dropna()
    tmp = tmp[tmp["cap"] > 0]
    tmp = tmp[(tmp["occ"] >= 0) & (tmp["occ"] <= tmp["cap"])]

    tmp["occupancy_rate"] = tmp["occ"] / tmp["cap"]

    agg = (
        tmp.groupby("LastUpdated", sort=False)["occupancy_rate"]
        .mean()
        .reset_index()
        .sort_values("LastUpdated", kind="mergesort")
    )

    return agg


def create_sequences(data: np.ndarray, time_step: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Ölçeklenmiş zaman serisinden LSTM için X ve y üretir.

    Örnek:
    time_step = 12 ise:
    X = önceki 12 zaman adımı
    y = sonraki 1 zaman adımı
    """

    X, y = [], []

    for i in range(len(data) - time_step):
        X.append(data[i : i + time_step])
        y.append(data[i + time_step])

    X_arr = np.array(X, dtype=np.float32)
    y_arr = np.array(y, dtype=np.float32)

    return X_arr, y_arr


def prepend_context(prev_tail: np.ndarray, block: np.ndarray) -> np.ndarray:
    """
    Önceki split'in son time_step satırını başa ekler.
    Böylece validation/test başlangıcında bağlam kaybı olmaz.
    """

    if len(prev_tail) == 0:
        return block

    return np.vstack([prev_tail, block])


def main() -> None:
    ensure_models()
    ensure_output()

    train_df = pd.read_csv(DATA_PROCESSED / "train.csv")
    val_df = pd.read_csv(DATA_PROCESSED / "val.csv")
    test_df = pd.read_csv(DATA_PROCESSED / "test.csv")

    for name, d in ("train", train_df), ("val", val_df), ("test", test_df):
        if d.empty:
            raise ValueError(f"{name}.csv boş; data_preparation çıktısını kontrol edin.")

    train_df["LastUpdated"] = pd.to_datetime(train_df["LastUpdated"], errors="coerce")
    val_df["LastUpdated"] = pd.to_datetime(val_df["LastUpdated"], errors="coerce")
    test_df["LastUpdated"] = pd.to_datetime(test_df["LastUpdated"], errors="coerce")

    agg_train = aggregate_mean_occupancy_rate(train_df)
    agg_val = aggregate_mean_occupancy_rate(val_df)
    agg_test = aggregate_mean_occupancy_rate(test_df)

    train_v = agg_train["occupancy_rate"].to_numpy(dtype=float).reshape(-1, 1)
    val_v = agg_val["occupancy_rate"].to_numpy(dtype=float).reshape(-1, 1)
    test_v = agg_test["occupancy_rate"].to_numpy(dtype=float).reshape(-1, 1)

    scaler = MinMaxScaler()

    train_scaled = scaler.fit_transform(train_v)
    val_scaled = scaler.transform(val_v)
    test_scaled = scaler.transform(test_v)

    time_step = LSTM_TIME_STEP

    if len(train_scaled) <= time_step:
        raise ValueError(
            f"Train serisi çok kısa: en az {time_step + 1} zaman gerekli, var {len(train_scaled)}"
        )

    X_train, y_train = create_sequences(train_scaled, time_step)

    val_block = prepend_context(train_scaled[-time_step:], val_scaled)
    X_val, y_val = create_sequences(val_block, time_step)

    tv_scaled = np.vstack([train_scaled, val_scaled])
    test_block = prepend_context(tv_scaled[-time_step:], test_scaled)
    X_test, y_test = create_sequences(test_block, time_step)

    print(
        f"[LSTM] Şekiller — X_train {X_train.shape}, "
        f"X_val {X_val.shape}, X_test {X_test.shape}"
    )

    model = Sequential(
        [
            Input(shape=(time_step, 1)),
            LSTM(64, return_sequences=True),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(1),
        ]
    )

    model.compile(
        optimizer="adam",
        loss="mse",
    )

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1,
    )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=LSTM_EPOCHS,
        batch_size=LSTM_BATCH_SIZE,
        callbacks=[early_stop],
        verbose=1,
    )

    model_path = MODELS_DIR / "lstm_parking_model.keras"
    scaler_path = MODELS_DIR / "lstm_occupancy_scaler.joblib"

    model.save(model_path)
    joblib.dump(scaler, scaler_path)

    print(f"[LSTM] Model: {model_path}")
    print(f"[LSTM] Scaler: {scaler_path}")

    plt.figure(figsize=(8, 4))
    plt.plot(history.history["loss"], label="Train loss")
    plt.plot(history.history["val_loss"], label="Val loss")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title("LSTM Eğitim ve Validation Loss")
    plt.tight_layout()

    loss_fig = OUTPUT_DIR / "lstm_loss_curve.png"
    plt.savefig(loss_fig, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"[LSTM] Loss eğrisi: {loss_fig}")

    test_pred = model.predict(X_test, verbose=0)

    y_test_2d = y_test.reshape(-1, 1)

    pred_inv = scaler.inverse_transform(test_pred)
    true_inv = scaler.inverse_transform(y_test_2d)

    rmse = float(np.sqrt(mean_squared_error(true_inv, pred_inv)))
    mae = float(mean_absolute_error(true_inv, pred_inv))

    print(f"[LSTM] Test RMSE: {rmse:.6f}, MAE: {mae:.6f}")

    metrics_path = OUTPUT_DIR / "lstm_metrics.txt"

    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write("LSTM Model Performansi\n")
        f.write("======================\n")
        f.write("Model tipi: Tek degiskenli zaman serisi LSTM\n")
        f.write("Kullanilan feature: occupancy_rate\n")
        f.write(f"Time step: {time_step}\n")
        f.write(f"Epochs: {LSTM_EPOCHS}\n")
        f.write(f"Batch size: {LSTM_BATCH_SIZE}\n")
        f.write(f"Random seed: {RANDOM_SEED}\n")
        f.write(f"RMSE: {rmse:.6f}\n")
        f.write(f"MAE: {mae:.6f}\n")

    print(f"[LSTM] Metrikler kaydedildi: {metrics_path}")

    plt.figure(figsize=(14, 4))

    lim = min(200, len(true_inv))

    plt.plot(true_inv[:lim], label="Gerçek")
    plt.plot(pred_inv[:lim], label="Tahmin", linestyle="--")

    plt.legend()
    plt.xlabel("Adım")
    plt.ylabel("Ortalama doluluk oranı")
    plt.title("LSTM Gerçek ve Tahmin Karşılaştırması")
    plt.tight_layout()

    cmp_fig = OUTPUT_DIR / "lstm_test_tahmin_karsilastirma.png"
    plt.savefig(cmp_fig, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"[LSTM] Tahmin-gerçek karşılaştırma grafiği: {cmp_fig}")


if __name__ == "__main__":
    main()