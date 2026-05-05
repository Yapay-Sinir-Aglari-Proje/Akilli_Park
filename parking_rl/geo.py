"""Coğrafi mesafe ve koordinat normalizasyonu (araç–otopark)."""

from __future__ import annotations

import math
from typing import Iterable, Tuple


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """İki nokta arası büyük daire mesafesi (km)."""
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * r * math.asin(min(1.0, math.sqrt(a)))


def bbox_normalize(
    lat: float,
    lon: float,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
) -> Tuple[float, float]:
    """Enlem/boylamı [0, 1] aralığına sıkıştırır (sınır genişletilmiş kutu)."""
    eps = 1e-9
    nx = (lat - lat_min) / max(lat_max - lat_min, eps)
    ny = (lon - lon_min) / max(lon_max - lon_min, eps)
    return max(0.0, min(1.0, float(nx))), max(0.0, min(1.0, float(ny)))


def max_pairwise_haversine_km(coords: Iterable[Tuple[float, float]]) -> float:
    """Ağdaki en büyük otopark arası mesafe (km) — mesafe normalizasyonu için."""
    pts = list(coords)
    best = 1e-6
    for i, (a1, o1) in enumerate(pts):
        for a2, o2 in pts[i + 1 :]:
            best = max(best, haversine_km(a1, o1, a2, o2))
    return best
