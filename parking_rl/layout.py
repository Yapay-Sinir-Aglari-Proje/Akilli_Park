"""Otopark ve araç konumlarının modellenmesi."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple


# Birmingham merkezine yakın referans (Bull Ring / şehir çekirdeği)
DEFAULT_ORIGIN: Tuple[float, float] = (52.4778, -1.8942)


def stable_parking_coordinates(
    parking_ids: Sequence[str],
    base_lat: float = DEFAULT_ORIGIN[0],
    base_lon: float = DEFAULT_ORIGIN[1],
) -> Dict[str, Tuple[float, float]]:
    """
    CSV'de koordinat yoksa, her SystemCodeNumber için deterministik sentetik lat/lon üretir.
    Böylece aynı otopark her çalıştırmada aynı konumda kalır.
    """
    out: Dict[str, Tuple[float, float]] = {}
    for pid in parking_ids:
        h = hashlib.sha256(pid.encode("utf-8")).digest()
        dlat = (int.from_bytes(h[0:2], "big") / 65535.0 - 0.5) * 0.07
        dlon = (int.from_bytes(h[2:4], "big") / 65535.0 - 0.5) * 0.09
        out[pid] = (base_lat + dlat, base_lon + dlon)
    return out


@dataclass
class ParkingLot:
    """Otopark varlığı: kimlik, indeks ve coğrafi konum."""

    parking_id: str
    index: int
    latitude: float
    longitude: float


@dataclass
class Vehicle:
    """Araç durumu: düzlemdeki konumu (WGS84)."""

    latitude: float
    longitude: float

    def set_position(self, lat: float, lon: float) -> None:
        self.latitude = lat
        self.longitude = lon


def build_parking_lots(parking_ids: Sequence[str]) -> List[ParkingLot]:
    """Sıralı kimlik listesinden ParkingLot listesi."""
    coords = stable_parking_coordinates(parking_ids)
    lots: List[ParkingLot] = []
    for i, pid in enumerate(sorted(parking_ids)):
        lat, lon = coords[pid]
        lots.append(ParkingLot(parking_id=pid, index=i, latitude=lat, longitude=lon))
    return lots


def parking_bounding_box(
    lots: Sequence[ParkingLot], margin_deg: float = 0.01
) -> Tuple[float, float, float, float]:
    """lat_min, lat_max, lon_min, lon_max (+ kenar payı)."""
    lats = [p.latitude for p in lots]
    lons = [p.longitude for p in lots]
    return (
        min(lats) - margin_deg,
        max(lats) + margin_deg,
        min(lons) - margin_deg,
        max(lons) + margin_deg,
    )
