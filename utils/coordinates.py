"""Deterministik otopark konumları (gerçek koordinat yoksa)."""

from __future__ import annotations

import hashlib
from typing import Dict, List, Tuple


def stable_parking_coordinates(parking_ids: List[str]) -> Dict[str, Tuple[float, float]]:
    """
    Her parking_id için [0,1]x[0,1] içinde stabil lat/lon benzeri koordinat.
    """
    out: Dict[str, Tuple[float, float]] = {}
    for pid in sorted(set(parking_ids)):
        h = hashlib.sha256(pid.encode("utf-8")).digest()
        lat = int.from_bytes(h[:8], "big") / (2**64)
        lon = int.from_bytes(h[8:16], "big") / (2**64)
        out[str(pid)] = (float(lat), float(lon))
    return out
