"""
Keşifsel veri analizi (EDA): işlenmiş train/val/test birleşiminde grafikler.

Grafikler `output/` altına kaydedilir (ekranda göstermek için Agg backend;
sunucuda çalışırken pencere açılmaz).
"""

from __future__ import annotations

import matplotlib

# Türkçe karakterli başlıklar ve sunucu ortamı için etkileşimsiz backend
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from paths import DATA_PROCESSED, OUTPUT_DIR, ensure_output


def load_processed_timeseries() -> pd.DataFrame:
    """
    Zaman sırasını koruyarak train → val → test dosyalarını üst üste birleştirir.
    (Shuffle yok; EDA da aynı kronolojiyi yansıtır.)
    """
    paths = [
        DATA_PROCESSED / "train.csv",
        DATA_PROCESSED / "val.csv",
        DATA_PROCESSED / "test.csv",
    ]
    chunks = []
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(
                f"İşlenmiş veri yok: {p}. Önce `python data_preparation.py` çalıştırın."
            )
        chunks.append(pd.read_csv(p))
    df = pd.concat(chunks, ignore_index=True)
    df["LastUpdated"] = pd.to_datetime(df["LastUpdated"], errors="coerce")
    df = df.dropna(subset=["LastUpdated"])
    df["Occupancy"] = pd.to_numeric(df["Occupancy"], errors="coerce")
    df["Capacity"] = pd.to_numeric(df["Capacity"], errors="coerce")
    df = df.dropna(subset=["Occupancy", "Capacity"])
    df = df[df["Capacity"] > 0]
    df["occupancy_rate"] = df["Occupancy"] / df["Capacity"]
    df = df.sort_values("LastUpdated", kind="mergesort")
    return df


def _save_fig(path, title: str) -> None:
    """Ortak kayıt: başlık + sıkı bbox."""
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_time_series(df: pd.DataFrame, out_dir) -> None:
    """Zaman serisi doluluk oranı (tüm otopark satırları kronolojik)."""
    plt.figure(figsize=(12, 4))
    plt.plot(df["LastUpdated"], df["occupancy_rate"], linewidth=0.8)
    plt.xlabel("Zaman")
    plt.ylabel("Doluluk oranı")
    _save_fig(out_dir / "1_zaman_serisi_doluluk.png", "Zaman serisi doluluk oranı")


def plot_hourly_average(df: pd.DataFrame, out_dir) -> None:
    """Saatlik ortalama doluluk."""
    df = df.copy()
    df["hour"] = df["LastUpdated"].dt.hour
    hourly = df.groupby("hour", sort=True)["occupancy_rate"].mean()
    plt.figure(figsize=(10, 4))
    hourly.plot(kind="bar", color="steelblue")
    plt.xlabel("Saat")
    plt.ylabel("Ortalama doluluk oranı")
    _save_fig(out_dir / "2_saatlik_ortalama.png", "Saatlik ortalama doluluk")


def plot_daily_average(df: pd.DataFrame, out_dir) -> None:
    """Haftanın gününe göre ortalama doluluk."""
    cats = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]
    df = df.copy()
    df["day"] = pd.Categorical(
        df["LastUpdated"].dt.day_name(), categories=cats, ordered=True
    )
    daily = df.groupby("day", observed=False)["occupancy_rate"].mean()
    plt.figure(figsize=(10, 4))
    daily.plot(kind="bar", color="coral")
    plt.xlabel("Gün")
    plt.ylabel("Ortalama doluluk oranı")
    _save_fig(out_dir / "3_gunluk_ortalama.png", "Günlük ortalama doluluk")


def plot_histogram(df: pd.DataFrame, out_dir) -> None:
    """Doluluk oranı histogramı."""
    plt.figure(figsize=(8, 4))
    df["occupancy_rate"].hist(bins=40, color="seagreen", edgecolor="white")
    plt.xlabel("Doluluk oranı")
    plt.ylabel("Frekans")
    _save_fig(out_dir / "4_histogram_doluluk_orani.png", "Doluluk oranı dağılımı")


def plot_correlation(df: pd.DataFrame, out_dir) -> None:
    """
    occupancy_rate ile Capacity ve Occupancy arasındaki Pearson korelasyonu.
    Heatmap yerine çubuk grafik + küçük korelasyon tablosu (matplotlib).
    """
    cols = ["occupancy_rate", "Capacity", "Occupancy"]
    corr_mat = df[cols].corr(method="pearson")
    occ_vs_cap = corr_mat.loc["occupancy_rate", "Capacity"]
    occ_vs_occ = corr_mat.loc["occupancy_rate", "Occupancy"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    labels = ["Capacity", "Occupancy"]
    values = [occ_vs_cap, occ_vs_occ]
    axes[0].barh(labels, values, color=["steelblue", "darkorange"])
    axes[0].axvline(0, color="gray", linewidth=0.8)
    axes[0].set_xlabel("Pearson korelasyonu (occupancy_rate ile)")
    axes[0].set_title("Doluluk oranı — değişken ilişkisi")

    table_data = []
    for i, r in enumerate(cols):
        table_data.append([f"{corr_mat.loc[r, c]:.3f}" for c in cols])
    axes[1].axis("off")
    tbl = axes[1].table(
        cellText=table_data,
        rowLabels=cols,
        colLabels=cols,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.1, 1.5)
    axes[1].set_title("Korelasyon matrisi (tablo)")

    plt.suptitle("Korelasyon analizi", y=1.02, fontsize=12)
    plt.tight_layout()
    plt.savefig(out_dir / "5_korelasyon.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_outliers(df: pd.DataFrame, out_dir) -> None:
    """occupancy_rate için kutu grafiği; aşırı değerler gösterilir (veri silinmez)."""
    plt.figure(figsize=(8, 4))
    plt.boxplot(
        df["occupancy_rate"].dropna(),
        vert=True,
        showfliers=True,
        patch_artist=True,
        boxprops=dict(facecolor="lightsteelblue", color="steelblue"),
        medianprops=dict(color="darkred", linewidth=2),
        flierprops=dict(marker="o", markersize=3, alpha=0.35, markerfacecolor="coral"),
    )
    plt.ylabel("Doluluk oranı")
    plt.xticks([1], ["occupancy_rate"])
    _save_fig(out_dir / "6_outlier_boxplot.png", "Doluluk oranı — aykırı değerler (boxplot)")


def plot_rolling_statistics(df: pd.DataFrame, out_dir, window: int = 20) -> None:
    """
    occupancy_rate için kayan ortalama; ham seri üzerine overlay + yalnızca kayan ortalama paneli.
    (Mevcut `plot_time_series` çıktısı değişmez; bu grafik ayrı dosyada kaydedilir.)
    """
    rolling_mean = df["occupancy_rate"].rolling(window=window, min_periods=1).mean()

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    axes[0].plot(
        df["LastUpdated"],
        df["occupancy_rate"],
        linewidth=0.6,
        alpha=0.65,
        label="Doluluk oranı (ham)",
    )
    axes[0].plot(
        df["LastUpdated"],
        rolling_mean,
        linewidth=1.2,
        color="darkorange",
        label=f"Kayan ortalama (pencere={window})",
    )
    axes[0].set_ylabel("Doluluk oranı")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].set_title("Ham seri + kayan ortalama (overlay)")

    axes[1].plot(
        df["LastUpdated"],
        rolling_mean,
        linewidth=1.0,
        color="darkorange",
        label=f"Kayan ortalama (pencere={window})",
    )
    axes[1].set_xlabel("Zaman")
    axes[1].set_ylabel("Doluluk oranı")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].set_title("Yalnızca kayan ortalama")

    plt.suptitle(
        f"Kayan ortalama — occupancy_rate (window={window})",
        y=1.01,
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(out_dir / "7_rolling_mean.png", dpi=150, bbox_inches="tight")
    plt.close()


def main() -> None:
    ensure_output()
    out_dir = OUTPUT_DIR
    df = load_processed_timeseries()

    plot_time_series(df, out_dir)
    plot_hourly_average(df, out_dir)
    plot_daily_average(df, out_dir)
    plot_histogram(df, out_dir)
    plot_correlation(df, out_dir)
    plot_outliers(df, out_dir)
    plot_rolling_statistics(df, out_dir)

    print(f"[EDA] Grafikler kaydedildi: {out_dir}")


if __name__ == "__main__":
    main()
