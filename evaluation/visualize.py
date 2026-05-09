"""
Basit otopark ızgarası + doluluk heatmap + tahmin/gerçek çizimi.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from paths import DATA_PROCESSED, OUTPUT_DIR, PREDICTIONS_DIR, ensure_output


def plot_prediction_vs_actual(
    predictions_csv: Path | None = None,
    out_path: Path | None = None,
    max_points: int = 300,
) -> Path:
    predictions_csv = predictions_csv or (PREDICTIONS_DIR / "test_predictions.csv")
    ensure_output()
    out_path = out_path or (OUTPUT_DIR / "pred_vs_actual.png")
    df = pd.read_csv(predictions_csv).tail(max_points)
    plt.figure(figsize=(12, 4))
    plt.plot(df["y_true_occupancy_rate"].values, label="Gerçek", color="tab:blue")
    plt.plot(df["y_pred_occupancy_rate"].values, label="LSTM", color="tab:orange", linestyle="--")
    plt.xlabel("Test adımı")
    plt.ylabel("Doluluk oranı")
    plt.title("LSTM: tahmin vs gerçek (aggregate)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path


def plot_parking_heatmap(
    processed_parquet: Path | None = None,
    timestamp: str | None = None,
    out_path: Path | None = None,
) -> Path:
    """Tek zaman diliminde lot bazlı doluluk ızgarası (sıralı bar)."""
    processed_parquet = processed_parquet or (DATA_PROCESSED / "processed.parquet")
    ensure_output()
    out_path = out_path or (OUTPUT_DIR / "occupancy_heatmap.png")
    df = pd.read_parquet(processed_parquet)
    df["LastUpdated"] = pd.to_datetime(df["LastUpdated"])
    ts = timestamp or str(df["LastUpdated"].iloc[len(df) // 2])
    snap = df[df["LastUpdated"] == ts]
    if snap.empty:
        snap = df.groupby("LastUpdated").head(1).iloc[:30]
        ts = str(snap["LastUpdated"].iloc[0])
    occ = (snap["Occupancy"] / snap["Capacity"]).to_numpy()
    labs = snap["SystemCodeNumber"].astype(str).to_numpy()
    plt.figure(figsize=(12, 4))
    plt.bar(np.arange(len(occ)), occ, color="teal", alpha=0.85)
    plt.xticks(np.arange(len(occ)), labs, rotation=75, ha="right", fontsize=7)
    plt.ylabel("Doluluk oranı")
    plt.title(f"Lot doluluk — {ts}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path


if __name__ == "__main__":
    print(plot_prediction_vs_actual())
    print(plot_parking_heatmap())
