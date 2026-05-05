"""
Otopark ham CSV verisini temizler ve zaman serisine uygun train/val/test böler.

Akış (sıra önemli):
1) CSV oku
2) Yinelenen satırları sil
3) Occupancy doğrula: 0 <= Occupancy <= Capacity, Capacity > 0
4) LastUpdated -> datetime (hatalıları düşür)
5) Zamana göre sırala (shuffle yok)
6) Benzersiz zaman damgası ekseninde %70 / %15 / %15 böl (aynı timestamp iki split’e düşmez)

Çıktılar:
- data/processed/train.csv, val.csv, test.csv
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from paths import DATA_PROCESSED, DATA_RAW, ensure_all_standard_dirs


def load_data(csv_path: Path) -> pd.DataFrame:
    """Ham CSV’yi okur; boyut bilgisini loglar."""
    df = pd.read_csv(csv_path)
    print(f"[data_prep] İlk boyut: {df.shape}")
    return df


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Tamamen aynı satırları kaldırır (tekrar ölçümler)."""
    before = len(df)
    df = df.drop_duplicates()
    print(f"[data_prep] Duplicate sonrası: {before} -> {len(df)} satır")
    return df


def validate_occupancy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Occupancy fiziksel olarak mantıklı olsun.
    Capacity 0 veya negatif satırlar elenir; Occupancy [0, Capacity] aralığına çekilmez,
    aralık dışındakiler silinir (veri hatası varsayımı).
    """
    before = len(df)
    cap = pd.to_numeric(df["Capacity"], errors="coerce")
    occ = pd.to_numeric(df["Occupancy"], errors="coerce")
    df = df.copy()
    df["Capacity"] = cap
    df["Occupancy"] = occ

    df = df.dropna(subset=["Capacity", "Occupancy"])
    df = df[df["Capacity"] > 0]
    df = df[(df["Occupancy"] >= 0) & (df["Occupancy"] <= df["Capacity"])]

    print(f"[data_prep] Occupancy doğrulama: {before} -> {len(df)} satır")
    return df


def parse_and_sort_time(df: pd.DataFrame) -> pd.DataFrame:
    """LastUpdated alanını datetime yapar, geçersizleri atar, zamana göre sıralar."""
    before = len(df)
    df = df.copy()
    df["LastUpdated"] = pd.to_datetime(df["LastUpdated"], errors="coerce")
    df = df.dropna(subset=["LastUpdated"])
    df = df.sort_values("LastUpdated", kind="mergesort").reset_index(drop=True)
    print(f"[data_prep] Tarih işleme: {before} -> {len(df)} satır (geçersiz zaman atıldı)")
    return df


def split_by_timestamps(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Zaman serisini bozmadan böler: önce benzersiz LastUpdated sırası,
    sonra her satır kendi zaman damgasına göre tek bir split’e düşer.

    Not: Satır sayısına göre değil, benzersiz zaman sayısına göre oran uygulanır;
    böylece aynı anda çekilmiş çoklu otopark okumaları hep aynı parçada kalır.
    """
    unique_times = df["LastUpdated"].drop_duplicates().sort_values(kind="mergesort")
    n_t = len(unique_times)
    if n_t < 3:
        raise ValueError(f"En az 3 benzersiz zaman gerekli (train/val/test), gelen: {n_t}")

    # Kesme indeksleri: önce oransal, sonra boş küme kalmaması için sıkıştır
    idx_train = int(n_t * train_ratio)
    idx_val_end = int(n_t * (train_ratio + val_ratio))
    idx_train = max(1, min(idx_train, n_t - 2))
    idx_val_end = max(idx_train + 1, min(idx_val_end, n_t - 1))

    train_times = set(unique_times.iloc[:idx_train])
    val_times = set(unique_times.iloc[idx_train:idx_val_end])
    test_times = set(unique_times.iloc[idx_val_end:])

    if not test_times:
        # Çok seyrek veride son zamanı teste al
        last_t = unique_times.iloc[-1]
        test_times = {last_t}
        val_times.discard(last_t)
        train_times.discard(last_t)

    train_df = df[df["LastUpdated"].isin(train_times)].copy()
    val_df = df[df["LastUpdated"].isin(val_times)].copy()
    test_df = df[df["LastUpdated"].isin(test_times)].copy()

    print(
        f"[data_prep] Zaman bazlı split — benzersiz zaman: {n_t}, "
        f"train/val/test zaman sayısı: {len(train_times)}/{len(val_times)}/{len(test_times)}"
    )
    print(
        f"[data_prep] Satır sayıları — train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}"
    )
    return train_df, val_df, test_df


def main() -> None:
    ensure_all_standard_dirs()

    input_path = DATA_RAW / "parking.csv"
    train_path = DATA_PROCESSED / "train.csv"
    val_path = DATA_PROCESSED / "val.csv"
    test_path = DATA_PROCESSED / "test.csv"

    if not input_path.exists():
        raise FileNotFoundError(f"Ham veri bulunamadı: {input_path}")

    df = load_data(input_path)
    df = drop_duplicates(df)
    df = validate_occupancy(df)
    df = parse_and_sort_time(df)

    print("[data_prep] Eksik değer özeti (temizlik sonrası):")
    print(df.isnull().sum())

    train_df, val_df, test_df = split_by_timestamps(df)

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"[data_prep] Kaydedildi:\n  {train_path}\n  {val_path}\n  {test_path}")


if __name__ == "__main__":
    main()
