"""
Künye (özgünlük): farklı zaman pencereleri (sequence length) için kısa LSTM kıyaslaması.

Her `time_step` için aynı aggregate veri üzerinde kısa eğitim + doğrulama MAE (ölçekli)
yazılır; `evaluation/reports/time_window_comparison.json` üretilir.
Ana `models/lstm_model.pt` dosyası **değiştirilmez**.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ml_config import RANDOM_SEED
from paths import DATA_PROCESSED, EVALUATION_DIR, MODELS_DIR, ensure_all_standard_dirs
from utils.data_pipeline import load_feature_scaler
from utils.forecast_models import OccupancyLSTM
from utils.lstm_core import (
    inv_occupancy_rate,
    load_aggregate_frame,
    make_sequences,
    mm_cols,
    prepend_context,
)
from utils.seeds import set_global_seed


def _run_one(ts: int, epochs: int, device: torch.device) -> dict:
    agg = load_aggregate_frame(DATA_PROCESSED / "processed.parquet")
    mm = mm_cols()
    train_part = agg[agg["split"] == "train"][mm].to_numpy(dtype=np.float32)
    val_part = agg[agg["split"] == "val"][mm].to_numpy(dtype=np.float32)
    if len(train_part) <= ts:
        return {"time_step": ts, "error": "train çok kısa"}

    X_train, y_train = make_sequences(train_part, ts)
    val_block = prepend_context(train_part[-ts:], val_part)
    X_val, y_val = make_sequences(val_block, ts)
    model = OccupancyLSTM(input_dim=len(mm), hidden=64, num_layers=2, dropout=0.2).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.L1Loss()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=32,
        shuffle=True,
    )
    Xv = torch.from_numpy(X_val).to(device)
    yv = torch.from_numpy(y_val).to(device)
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        vpred = model(Xv)
        val_mae_mm = float(loss_fn(vpred, yv).item())
    scaler, sk = load_feature_scaler(MODELS_DIR)
    with torch.no_grad():
        pred_v = vpred.cpu().numpy()
    y_val_inv = inv_occupancy_rate(y_val, scaler, sk)
    pred_inv = inv_occupancy_rate(pred_v, scaler, sk)
    val_mae_rate = float(np.mean(np.abs(y_val_inv - pred_inv)))
    return {
        "time_step": ts,
        "val_mae_scaled_space": val_mae_mm,
        "val_mae_occupancy_rate": val_mae_rate,
        "n_train_seq": int(len(X_train)),
        "n_val_seq": int(len(X_val)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", default="8,12,16,24", help="Virgülle ayrılmış pencere uzunlukları")
    parser.add_argument("--epochs", type=int, default=22)
    args = parser.parse_args()
    set_global_seed(RANDOM_SEED)
    ensure_all_standard_dirs()
    pq = DATA_PROCESSED / "processed.parquet"
    if not pq.exists():
        raise FileNotFoundError(f"Önce processed.parquet üretin: {pq}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ts_list = [int(x.strip()) for x in args.timesteps.split(",") if x.strip()]
    rows = []
    for ts in ts_list:
        print(f"[time_window] time_step={ts} ...")
        rows.append(_run_one(ts, args.epochs, device))

    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    out = EVALUATION_DIR / "time_window_comparison.json"
    out.write_text(json.dumps({"epochs_per_run": args.epochs, "runs": rows}, indent=2), encoding="utf-8")
    print(f"[time_window] Rapor: {out}")


if __name__ == "__main__":
    main()
