"""
Parking Birmingham aggregate serisi — künye (6.1, 7.2, 7.3):

- Mimari: LSTM | GRU | Temporal Transformer (PyTorch)
- Eğitim: erken durdurma, ReduceLROnPlateau, isteğe bağlı weighted L1 (yüksek doluluğa ağırlık)
- Hiperparametre: Optuna ile çok deneme (--optuna-trials); ardından en iyi hiperparametrelerle tam eğitim

Çıktı: models/lstm_model.pt (tüm mimariler için aynı dosya adı), predictions/test_predictions.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, TensorDataset

from ml_config import LSTM_BATCH_SIZE, LSTM_EPOCHS, LSTM_TIME_STEP, RANDOM_SEED
from paths import DATA_PROCESSED, MODELS_DIR, PREDICTIONS_DIR, ensure_all_standard_dirs
from utils.data_pipeline import build_processed_dataset, load_feature_scaler
from utils.forecast_models import build_forecast_model
from utils.lstm_core import (
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


def _batch_weighted_l1(pred: torch.Tensor, target: torch.Tensor, xb: torch.Tensor) -> torch.Tensor:
    """Künye weighted loss: son zaman adımındaki occupancy_rate_mm (kanal 0) yüksekse daha büyük ağırlık."""
    last_occ = xb[:, -1, 0].clamp(0.0, 1.0)
    w = 1.0 + 2.0 * last_occ
    err = torch.abs(pred - target)
    return (err * w).mean() / w.mean()


def _train_loop(
    model: nn.Module,
    device: torch.device,
    train_loader: DataLoader,
    X_val_t: torch.Tensor,
    y_val_t: torch.Tensor,
    epochs: int,
    patience: int,
    lr: float,
    weighted_loss: bool,
    log_prefix: str = "[LSTM]",
) -> tuple[dict[str, torch.Tensor], float]:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=3, min_lr=1e-6
    )
    loss_fn = nn.L1Loss()

    best_val = float("inf")
    stale = 0
    best_state: dict[str, torch.Tensor] | None = None

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            if weighted_loss:
                loss = _batch_weighted_l1(pred, yb, xb)
            else:
                loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            vpred = model(X_val_t)
            vloss = float(loss_fn(vpred, y_val_t).item())
        scheduler.step(vloss)
        print(f"{log_prefix} epoch {epoch + 1}/{epochs} val_mae_mm: {vloss:.6f}")
        if vloss < best_val - 1e-6:
            best_val = vloss
            stale = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                print(f"{log_prefix} early stopping")
                break

    if best_state:
        model.load_state_dict(best_state)
    assert best_state is not None
    return best_state, best_val


def _checkpoint_payload(
    cell: str,
    state: dict[str, torch.Tensor],
    mm: list[str],
    ts: int,
    hp: dict[str, Any],
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "state_dict": state,
        "input_cols": mm,
        "time_step": ts,
        "cell_type": cell,
        "seed": RANDOM_SEED,
    }
    out.update(hp)
    return out


def _default_hp(cell: str) -> dict[str, Any]:
    if cell == "transformer":
        return {
            "d_model": 64,
            "nhead": 4,
            "num_layers": 2,
            "dropout": 0.2,
            "max_seq_len": 128,
            "dim_feedforward": None,
        }
    return {"hidden": 64, "num_layers": 2, "dropout": 0.2}


def _suggest_hp_from_trial(trial: Any, cell: str) -> dict[str, Any]:
    """Optuna trial — künye 7.2 hiperparametre araması."""
    if cell == "transformer":
        d_model = trial.suggest_categorical("d_model", [32, 64, 96, 128])
        nhead_choices = [h for h in (2, 4) if d_model % h == 0]
        nhead = trial.suggest_categorical("nhead", nhead_choices)
        return {
            "d_model": d_model,
            "nhead": nhead,
            "num_layers": trial.suggest_int("num_layers", 1, 3),
            "dropout": trial.suggest_float("dropout", 0.1, 0.35),
            "max_seq_len": 128,
            "dim_feedforward": None,
        }
    hidden = trial.suggest_categorical("hidden", [32, 64, 96, 128])
    return {
        "hidden": hidden,
        "num_layers": trial.suggest_int("num_layers", 1, 3),
        "dropout": trial.suggest_float("dropout", 0.1, 0.35),
    }


def run_optuna(
    cell: str,
    input_dim: int,
    device: torch.device,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_trials: int,
    epochs_per_trial: int,
    patience: int,
    weighted_loss: bool,
) -> dict[str, Any]:
    try:
        import optuna
    except ImportError as e:
        raise RuntimeError(
            "Optuna gerekli: `pip install optuna` (künye 7.2 hiperparametre optimizasyonu)."
        ) from e

    def objective(trial: optuna.Trial) -> float:
        hp = _suggest_hp_from_trial(trial, cell)
        lr = trial.suggest_float("lr", 1e-4, 3e-3, log=True)
        bs = trial.suggest_categorical("batch_size", [16, 32, 64])
        model = build_forecast_model(cell, input_dim, hp).to(device)
        train_ds = TensorDataset(
            torch.from_numpy(X_train),
            torch.from_numpy(y_train),
        )
        train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, drop_last=False)
        X_val_t = torch.from_numpy(X_val).to(device)
        y_val_t = torch.from_numpy(y_val).to(device)
        _state, best_val = _train_loop(
            model,
            device,
            train_loader,
            X_val_t,
            y_val_t,
            epochs=epochs_per_trial,
            patience=max(2, patience // 2),
            lr=lr,
            weighted_loss=weighted_loss,
            log_prefix=f"[Optuna trial {trial.number}]",
        )
        return float(best_val)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    best = study.best_params
    if cell == "transformer":
        hp_out = {
            "d_model": int(best["d_model"]),
            "nhead": int(best["nhead"]),
            "num_layers": int(best["num_layers"]),
            "dropout": float(best["dropout"]),
            "max_seq_len": 128,
            "dim_feedforward": None,
        }
    else:
        hp_out = {
            "hidden": int(best["hidden"]),
            "num_layers": int(best["num_layers"]),
            "dropout": float(best["dropout"]),
        }
    meta = {
        "best_val_mae_mm": float(study.best_value),
        "best_params": dict(best),
        "arch_hp": hp_out,
    }
    out_path = MODELS_DIR / "lstm_optuna_best.json"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[Optuna] En iyi doğrulama MAE (ölçekli): {study.best_value:.6f}")
    print(f"[Optuna] Parametreler kaydedildi: {out_path}")
    return {"lr": float(best["lr"]), "batch_size": int(best["batch_size"]), **hp_out}


def train_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cell",
        choices=["lstm", "gru", "transformer"],
        default="lstm",
        help="Künye 6.1 tahmin mimarisi",
    )
    parser.add_argument("--epochs", type=int, default=LSTM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=LSTM_BATCH_SIZE)
    parser.add_argument("--time-step", type=int, default=LSTM_TIME_STEP)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument(
        "--weighted-loss",
        action="store_true",
        help="Künye 7.2: yoğun doluluğa daha yüksek ağırlıklı L1 kaybı",
    )
    parser.add_argument("--optuna-trials", type=int, default=0, help="0 = kapalı; künye Optuna araması")
    parser.add_argument("--optuna-epochs-per-trial", type=int, default=35)
    args = parser.parse_args()

    set_global_seed(RANDOM_SEED)
    ensure_all_standard_dirs()

    parquet_path = DATA_PROCESSED / "processed.parquet"
    _ensure_parquet(parquet_path)

    agg = load_aggregate_frame(parquet_path)
    mm = mm_cols()
    scaler, sk = load_feature_scaler(MODELS_DIR)

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
    cell = args.cell
    input_dim = len(mm)

    hp = _default_hp(cell)
    lr = args.lr
    batch_size = args.batch_size

    if args.optuna_trials > 0:
        o = run_optuna(
            cell,
            input_dim,
            device,
            X_train,
            y_train,
            X_val,
            y_val,
            n_trials=args.optuna_trials,
            epochs_per_trial=args.optuna_epochs_per_trial,
            patience=args.patience,
            weighted_loss=args.weighted_loss,
        )
        lr = o["lr"]
        batch_size = int(o["batch_size"])
        hp = {k: v for k, v in o.items() if k not in ("lr", "batch_size")}

    model = build_forecast_model(cell, input_dim, hp).to(device)
    train_ds = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(y_train),
    )
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=False
    )
    X_val_t = torch.from_numpy(X_val).to(device)
    y_val_t = torch.from_numpy(y_val).to(device)

    best_state, _ = _train_loop(
        model,
        device,
        train_loader,
        X_val_t,
        y_val_t,
        epochs=args.epochs,
        patience=args.patience,
        lr=lr,
        weighted_loss=args.weighted_loss,
    )

    model_path = MODELS_DIR / "lstm_model.pt"
    ckpt = _checkpoint_payload(cell, best_state, mm, ts, hp)
    torch.save(ckpt, model_path)
    print(f"[train] Kaydedildi: {model_path} (cell={cell})")

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_pred = model(torch.from_numpy(X_test).to(device)).cpu().numpy()

    y_true_o = inv_occupancy_rate(y_test, scaler, sk)
    y_pred_o = inv_occupancy_rate(test_pred, scaler, sk)

    rmse = float(np.sqrt(mean_squared_error(y_true_o, y_pred_o)))
    mae = float(mean_absolute_error(y_true_o, y_pred_o))
    r2 = float(r2_score(y_true_o, y_pred_o))
    denom = np.maximum(np.abs(y_true_o), 1e-6)
    mape = float(np.mean(np.abs((y_true_o - y_pred_o) / denom)) * 100.0)
    print(
        f"[train] Test — MAE: {mae:.6f}, RMSE: {rmse:.6f}, MAPE: {mape:.3f}%, R²: {r2:.4f}"
    )

    agg_test = agg[agg["split"] == "test"].reset_index(drop=True)
    n_out = len(y_true_o)
    if n_out > len(agg_test):
        raise RuntimeError(
            f"Tahmin sayısı ({n_out}) aggregate test satırından ({len(agg_test)}) fazla."
        )
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
    print(f"[train] Tahmin CSV: {out_csv}")


if __name__ == "__main__":
    train_main()
