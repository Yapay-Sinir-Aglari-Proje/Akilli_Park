"""
Zenginleştirilmiş zaman serisi ile PyTorch LSTM eğitimi (2 katman + dropout).

Çıktılar:
- models/lstm_model.pt
- models/processed_feature_scaler.joblib (pipeline ile aynı; yoksa pipeline çalışır)
- predictions/test_predictions.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.utils.data import DataLoader, TensorDataset

from ml_config import LSTM_BATCH_SIZE, LSTM_EPOCHS, LSTM_TIME_STEP, RANDOM_SEED
from paths import DATA_PROCESSED, MODELS_DIR, PREDICTIONS_DIR, ensure_all_standard_dirs
from utils.data_pipeline import build_processed_dataset
from utils.lstm_core import (
    OccupancyLSTM,
    inv_occupancy_rate,
    load_aggregate_frame,
    make_sequences,
    mm_cols,
    prepend_context as _prepend_context,
)
from utils.seeds import set_global_seed


def _ensure_parquet(path: Path) -> None:
    if not path.exists():
        print("[train_lstm] processed.parquet yok; veri hattı çalıştırılıyor...")
        build_processed_dataset(output_path=path)


def train_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=LSTM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=LSTM_BATCH_SIZE)
    parser.add_argument("--time-step", type=int, default=LSTM_TIME_STEP)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=8)
    args = parser.parse_args()

    set_global_seed(RANDOM_SEED)
    ensure_all_standard_dirs()

    parquet_path = DATA_PROCESSED / "processed.parquet"
    _ensure_parquet(parquet_path)

    agg = load_aggregate_frame(parquet_path)
    mm = mm_cols()
    scaler = joblib.load(MODELS_DIR / "processed_feature_scaler.joblib")

    train_part = agg[agg["split"] == "train"][mm].to_numpy(dtype=np.float32)
    val_part = agg[agg["split"] == "val"][mm].to_numpy(dtype=np.float32)
    test_part = agg[agg["split"] == "test"][mm].to_numpy(dtype=np.float32)

    ts = args.time_step
    if len(train_part) <= ts:
        raise ValueError("Train serisi pencere için çok kısa.")

    X_train, y_train = make_sequences(train_part, ts)
    val_block = _prepend_context(train_part[-ts:], val_part)
    X_val, y_val = make_sequences(val_block, ts)
    tv = np.vstack([train_part, val_part])
    test_block = _prepend_context(tv[-ts:], test_part)
    X_test, y_test = make_sequences(test_block, ts)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = OccupancyLSTM(input_dim=len(mm)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.L1Loss()

    train_ds = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(y_train),
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False
    )

    X_val_t = torch.from_numpy(X_val).to(device)
    y_val_t = torch.from_numpy(y_val).to(device)

    best_val = float("inf")
    stale = 0
    best_state: dict | None = None

    for epoch in range(args.epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            vpred = model(X_val_t)
            vloss = float(loss_fn(vpred, y_val_t).item())
        print(f"[LSTM] epoch {epoch + 1}/{args.epochs} val_mae_mm: {vloss:.6f}")
        if vloss < best_val - 1e-6:
            best_val = vloss
            stale = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= args.patience:
                print("[LSTM] early stopping")
                break

    if best_state:
        model.load_state_dict(best_state)

    model_path = MODELS_DIR / "lstm_model.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_cols": mm,
            "time_step": ts,
            "hidden": 64,
            "num_layers": 2,
            "dropout": 0.2,
            "seed": RANDOM_SEED,
        },
        model_path,
    )
    print(f"[LSTM] Kaydedildi: {model_path}")

    model.eval()
    with torch.no_grad():
        test_pred = model(torch.from_numpy(X_test).to(device)).cpu().numpy()

    y_true_o = inv_occupancy_rate(y_test, scaler)
    y_pred_o = inv_occupancy_rate(test_pred, scaler)

    rmse = float(np.sqrt(mean_squared_error(y_true_o, y_pred_o)))
    mae = float(mean_absolute_error(y_true_o, y_pred_o))
    print(f"[LSTM] Test MAE (occ rate): {mae:.6f}, RMSE: {rmse:.6f}")

    agg_test = agg[agg["split"] == "test"].reset_index(drop=True)
    n_out = len(y_true_o)
    if n_out > len(agg_test):
        raise RuntimeError(
            f"Tahmin sayısı ({n_out}) aggregate test satırından ({len(agg_test)}) fazla."
        )
    # y_test[i] ↔ test_part i. satırı (bağlam test_block'un başında); zaman damgası i.
    test_times = agg_test["LastUpdated"].iloc[:n_out]

    pred_df = pd.DataFrame(
        {
            "LastUpdated": test_times.to_numpy(),
            "y_true_occupancy_rate": y_true_o,
            "y_pred_occupancy_rate": y_pred_o,
        }
    )
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = PREDICTIONS_DIR / "test_predictions.csv"
    pred_df.to_csv(out_csv, index=False)
    print(f"[LSTM] Tahmin CSV: {out_csv}")


if __name__ == "__main__":
    train_main()
