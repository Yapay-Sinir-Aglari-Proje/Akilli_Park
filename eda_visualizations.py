"""
Gelişmiş EDA grafikleri — `output/` altına kayıt.

  python eda_visualizations.py
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from eda import load_processed_timeseries
from paths import OUTPUT_DIR, PREDICTIONS_DIR, ensure_output


def plot_occupancy_hourly_heatmap(df: pd.DataFrame, out_path) -> None:
    """Saatlik ortalama doluluk heatmap (gün × saat)."""
    d = df.copy()
    d["hour"] = d["LastUpdated"].dt.hour.astype(int)
    d["dow"] = d["LastUpdated"].dt.dayofweek.astype(int)
    piv = (
        d.groupby(["dow", "hour"], observed=False)["occupancy_rate"]
        .mean()
        .unstack(fill_value=np.nan)
    )
    piv = piv.reindex(index=range(7), columns=range(24))
    labels = ["Pzt", "Sal", "Çar", "Per", "Cum", "Cmt", "Paz"]
    plt.figure(figsize=(12, 4.5))
    im = plt.imshow(piv.to_numpy(dtype=float), aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    plt.colorbar(im, label="Ortalama doluluk")
    plt.yticks(range(7), labels)
    plt.xticks(range(0, 24, 2), [str(h) for h in range(0, 24, 2)])
    plt.xlabel("Saat")
    plt.ylabel("Haftanın günü")
    plt.title("Saatlik doluluk heatmap")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_weekday_weekend(df: pd.DataFrame, out_path) -> None:
    d = df.copy()
    d["is_weekend"] = d["LastUpdated"].dt.dayofweek >= 5
    g = d.groupby("is_weekend")["occupancy_rate"].mean()
    labels = ["Hafta içi", "Hafta sonu"]
    vals = [float(g.get(False, np.nan)), float(g.get(True, np.nan))]
    plt.figure(figsize=(6, 4))
    plt.bar(labels, vals, color=["#3b82f6", "#f97316"])
    plt.ylabel("Ortalama doluluk oranı")
    plt.title("Hafta içi / hafta sonu karşılaştırması")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_top_lots(df: pd.DataFrame, out_dir) -> None:
    if "SystemCodeNumber" not in df.columns:
        return
    lot = (
        df.groupby("SystemCodeNumber")["occupancy_rate"]
        .mean()
        .sort_values(ascending=False)
    )
    top = lot.head(10)
    bottom = lot.tail(10).sort_values()
    for subset, name, color in (
        (top, "en_yogun_10_otopark.png", "#dc2626"),
        (bottom, "en_bos_10_otopark.png", "#16a34a"),
    ):
        plt.figure(figsize=(10, 4))
        short = [str(x)[-10:] for x in subset.index]
        plt.barh(short[::-1], subset.values[::-1], color=color)
        plt.xlabel("Ortalama doluluk oranı")
        plt.title("En yoğun 10" if "yogun" in name else "En boş 10")
        plt.tight_layout()
        plt.savefig(out_dir / name, dpi=150, bbox_inches="tight")
        plt.close()


def plot_train_test_split_timeline(df: pd.DataFrame, out_path) -> None:
    """Zaman serisi + train/val/test bölgeleri (dosya sırasına göre yaklaşık)."""
    n = len(df)
    if n < 10:
        return
    i_train = int(n * 0.7)
    i_val = int(n * 0.85)
    t = df["LastUpdated"]
    y = df["occupancy_rate"]
    plt.figure(figsize=(12, 4))
    plt.plot(t, y, linewidth=0.5, alpha=0.7, color="#64748b")
    plt.axvspan(t.iloc[0], t.iloc[i_train - 1], alpha=0.12, color="#22c55e", label="Train")
    plt.axvspan(t.iloc[i_train], t.iloc[i_val - 1], alpha=0.12, color="#eab308", label="Val")
    plt.axvspan(t.iloc[i_val], t.iloc[-1], alpha=0.12, color="#ef4444", label="Test")
    plt.xlabel("Zaman")
    plt.ylabel("Doluluk oranı")
    plt.title("Train / val / test zaman serisi bölünmesi (yaklaşık %70 / %15 / %15)")
    plt.legend(handles=[
        Patch(facecolor="#22c55e", alpha=0.3, label="Train"),
        Patch(facecolor="#eab308", alpha=0.3, label="Val"),
        Patch(facecolor="#ef4444", alpha=0.3, label="Test"),
    ])
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_actual_vs_predicted(out_path) -> None:
    pred_path = PREDICTIONS_DIR / "test_predictions.csv"
    if not pred_path.exists():
        print(f"[eda_viz] Tahmin dosyası yok: {pred_path}")
        return
    pr = pd.read_csv(pred_path)
    if "y_true_occupancy_rate" not in pr.columns or "y_pred_occupancy_rate" not in pr.columns:
        return
    yt = pr["y_true_occupancy_rate"].astype(float)
    yp = pr["y_pred_occupancy_rate"].astype(float)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].scatter(yt, yp, alpha=0.35, s=8, c="#2563eb")
    lims = [0, 1]
    axes[0].plot(lims, lims, "k--", linewidth=1)
    axes[0].set_xlabel("Gerçek doluluk")
    axes[0].set_ylabel("Tahmin")
    axes[0].set_title("Gerçek vs tahmin (scatter)")
    n_show = min(400, len(pr))
    idx = np.linspace(0, len(pr) - 1, n_show, dtype=int)
    axes[1].plot(range(n_show), yt.iloc[idx].values, label="Gerçek", alpha=0.8)
    axes[1].plot(range(n_show), yp.iloc[idx].values, label="Tahmin", alpha=0.8)
    axes[1].set_xlabel("Örnek (alt küme)")
    axes[1].set_title("Zaman dilimi karşılaştırması (alt küme)")
    axes[1].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def run_all() -> None:
    ensure_output()
    df = load_processed_timeseries()
    plot_occupancy_hourly_heatmap(df, OUTPUT_DIR / "occupancy_hourly_heatmap.png")
    plot_weekday_weekend(df, OUTPUT_DIR / "weekday_weekend_comparison.png")
    plot_top_lots(df, OUTPUT_DIR)
    plot_train_test_split_timeline(df, OUTPUT_DIR / "train_test_split_timeline.png")
    plot_actual_vs_predicted(OUTPUT_DIR / "actual_vs_predicted.png")
    print(f"[eda_viz] Ciktilar: {OUTPUT_DIR}")


def main() -> None:
    run_all()


if __name__ == "__main__":
    main()
