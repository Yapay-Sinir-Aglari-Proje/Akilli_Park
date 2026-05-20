"""
Künye (Bölüm 5): aykırı değer temizleme — lot bazında IQR (occupancy_rate).

Fiziksel sınır kontrolünden (0 <= Occupancy <= Capacity) sonra uygulanır;
çok seyrek lotlarda IQR güvenilmez olduğu için `min_samples_per_lot` altında satır korunur.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def remove_occupancy_rate_outliers_iqr(
    df: pd.DataFrame,
    *,
    lot_col: str = "SystemCodeNumber",
    iqr_multiplier: float = 3.0,
    min_samples_per_lot: int = 30,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Her otopark (lot) için occupancy_rate üzerinde IQR tabanlı aykırı satırları çıkarır.

    Returns:
        (temiz DataFrame, özet istatistik sözlüğü)
    """
    if df.empty:
        return df, {"removed": 0, "kept": 0, "lots_skipped": 0}

    work = df.copy()
    if "occupancy_rate" not in work.columns:
        work["occupancy_rate"] = work["Occupancy"] / work["Capacity"]

    before = len(work)
    keep_mask = np.ones(before, dtype=bool)
    lots_skipped = 0
    lots_filtered = 0

    for lot, grp in work.groupby(lot_col, sort=False):
        idx = grp.index.to_numpy()
        n = len(grp)
        if n < min_samples_per_lot:
            lots_skipped += 1
            continue

        rates = grp["occupancy_rate"].to_numpy(dtype=np.float64)
        q1 = float(np.percentile(rates, 25))
        q3 = float(np.percentile(rates, 75))
        iqr = q3 - q1
        if iqr <= 1e-12:
            continue

        lo = q1 - iqr_multiplier * iqr
        hi = q3 + iqr_multiplier * iqr
        inlier = (rates >= lo) & (rates <= hi)
        if not inlier.all():
            lots_filtered += 1
            keep_mask[idx[~inlier]] = False

    out = work.loc[keep_mask].reset_index(drop=True)
    removed = before - len(out)
    stats: dict[str, Any] = {
        "method": "per_lot_iqr",
        "iqr_multiplier": iqr_multiplier,
        "min_samples_per_lot": min_samples_per_lot,
        "rows_before": before,
        "rows_after": len(out),
        "removed": removed,
        "removed_pct": round(100.0 * removed / max(before, 1), 4),
        "lots_skipped_sparse": lots_skipped,
        "lots_with_removals": lots_filtered,
    }
    return out, stats
