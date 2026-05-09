"""
Gymnasium uyumlu akıllı otopark simülasyon ortamı.

State: zaman ilerlemesi, aracın normalize konumu, her otopark için doluluk oranı
      ve araca göre normalize edilmiş mesafe.

Action: Discrete — hedef otopark indeksi (hangi otoparka yönelindiği).

Reward: adım maliyeti + varışta boş yer oranına bonus − yoğunluk cezası.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from parking_rl.geo import bbox_normalize, haversine_km, max_pairwise_haversine_km
from parking_rl.layout import (
    DEFAULT_ORIGIN,
    ParkingLot,
    Vehicle,
    build_parking_lots,
    parking_bounding_box,
)


class SmartParkingEnv(gym.Env):
    """
    Zaman serisi otopark verisi üzerinde hareket eden araç; ajan bir otopark seçer,
    araç her adımda hedefe doğru hareket eder, zaman bir adım ilerler.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        data_path: str | Path = "data/processed/train.csv",
        max_episode_steps: int = 120,
        arrival_threshold_km: float = 0.08,
        move_step_deg: float = 0.002,
        step_cost: float = -0.02,
        availability_weight: float = 8.0,
        congestion_weight: float = 3.0,
        vehicle_origin: Optional[Tuple[float, float]] = None,
        randomize_start_time: bool = True,
    ):
        super().__init__()

        self.data_path = Path(data_path)
        self.max_episode_steps = int(max_episode_steps)
        self.arrival_threshold_km = float(arrival_threshold_km)
        self.move_step_deg = float(move_step_deg)
        self.step_cost = float(step_cost)
        self.availability_weight = float(availability_weight)
        self.congestion_weight = float(congestion_weight)
        self.randomize_start_time = randomize_start_time

        df = pd.read_csv(self.data_path)
        # Sayısal alanları zorunlu kıl; NaN / kirli satırlar PPO sırasında patlamasın
        df["Occupancy"] = pd.to_numeric(df["Occupancy"], errors="coerce")
        df["Capacity"] = pd.to_numeric(df["Capacity"], errors="coerce")
        df["SystemCodeNumber"] = df["SystemCodeNumber"].astype(str)
        df["LastUpdated"] = pd.to_datetime(df["LastUpdated"], errors="coerce")
        df = df.dropna(subset=["LastUpdated", "Occupancy", "Capacity"])
        df = df[df["Capacity"] > 0]
        df = df[(df["Occupancy"] >= 0) & (df["Occupancy"] <= df["Capacity"])]
        df = df.sort_values("LastUpdated", kind="mergesort")

        if df.empty:
            raise ValueError(f"Ortam verisi boş veya geçersiz: {self.data_path}")

        self._df = df
        self.parking_ids: List[str] = sorted(df["SystemCodeNumber"].unique().tolist())
        self.lots: List[ParkingLot] = build_parking_lots(self.parking_ids)
        self._lot_by_id: Dict[str, ParkingLot] = {p.parking_id: p for p in self.lots}

        self.default_capacity: Dict[str, float] = (
            df.groupby("SystemCodeNumber")["Capacity"].max().astype(float).to_dict()
        )

        self._times: List[pd.Timestamp] = sorted(df["LastUpdated"].unique().tolist())
        self._build_snapshots()

        lat0, lon0 = vehicle_origin if vehicle_origin is not None else DEFAULT_ORIGIN
        self._origin_lat = float(lat0)
        self._origin_lon = float(lon0)

        self.lat_min, self.lat_max, self.lon_min, self.lon_max = parking_bounding_box(
            self.lots, margin_deg=0.012
        )
        # Araç başlangıcı kutunun dışına taşmasın
        self._origin_lat = float(np.clip(self._origin_lat, self.lat_min, self.lat_max))
        self._origin_lon = float(np.clip(self._origin_lon, self.lon_min, self.lon_max))

        coords = [(p.latitude, p.longitude) for p in self.lots]
        self._max_net_km = max_pairwise_haversine_km(coords)
        self._max_net_km = max(self._max_net_km, 0.5)

        n = len(self.lots)
        obs_dim = 1 + 2 + 2 * n
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(n)

        self.vehicle = Vehicle(self._origin_lat, self._origin_lon)
        self._time_idx = 0
        self._step_count = 0
        self._target_idx = 0
        self._last_snap: Dict[str, Tuple[float, float]] = {}

    def _build_snapshots(self) -> None:
        self._snapshots: List[Dict[str, Tuple[float, float]]] = []
        for ts in self._times:
            sub = self._df[self._df["LastUpdated"] == ts]
            snap: Dict[str, Tuple[float, float]] = {}
            for _, row in sub.iterrows():
                pid = str(row["SystemCodeNumber"])
                snap[pid] = (float(row["Occupancy"]), float(row["Capacity"]))
            self._snapshots.append(snap)

    def _merge_snapshot(self, t_idx: int) -> Dict[str, Tuple[float, float]]:
        raw = self._snapshots[t_idx]
        merged: Dict[str, Tuple[float, float]] = {}
        for pid in self.parking_ids:
            if pid in raw:
                merged[pid] = raw[pid]
            elif pid in self._last_snap:
                merged[pid] = self._last_snap[pid]
            else:
                cap = max(1.0, self.default_capacity.get(pid, 1.0))
                merged[pid] = (0.0, cap)
        self._last_snap = merged.copy()
        return merged

    def _get_obs(self, snap: Dict[str, Tuple[float, float]]) -> np.ndarray:
        time_norm = float(self._time_idx / max(1, len(self._times) - 1))
        vlat, vlon = bbox_normalize(
            self.vehicle.latitude,
            self.vehicle.longitude,
            self.lat_min,
            self.lat_max,
            self.lon_min,
            self.lon_max,
        )
        occ_list: List[float] = []
        dist_list: List[float] = []
        for pid in self.parking_ids:
            occ, cap = snap[pid]
            cap = max(1.0, cap)
            occ = float(np.clip(occ, 0.0, cap))
            occ_list.append(occ / cap)
            lot = self._lot_by_id[pid]
            d_km = haversine_km(
                self.vehicle.latitude,
                self.vehicle.longitude,
                lot.latitude,
                lot.longitude,
            )
            dist_list.append(float(np.clip(d_km / self._max_net_km, 0.0, 1.0)))
        vec = np.array(
            [time_norm, vlat, vlon] + occ_list + dist_list, dtype=np.float32
        )
        # Gym Box ile uyumlu, son güvenlik: NaN/inf üretimini engelle
        vec = np.nan_to_num(vec, nan=0.0, posinf=1.0, neginf=0.0)
        return np.clip(vec, 0.0, 1.0)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Bölüm başında zaman ve araç konumunu sıfırlar.

        - randomize_start_time=True ise, episode sonunda zaman taşması olmaması için
          başlangıç indeksi üstten kısıtlanır (sonsuz döngü değil, güvenli aralık).
        - _last_snap temizlenir ki önceki bölümden kirli doluluk taşınmasın.
        """
        super().reset(seed=seed)
        self._step_count = 0
        self._last_snap = {}

        if self.randomize_start_time and len(self._times) > 1:
            # Kalan zaman adımı en az 1 olsun; yoksa hep t=0’dan başlarız
            hi = max(0, len(self._times) - self.max_episode_steps - 1)
            self._time_idx = int(self.np_random.integers(0, max(1, hi + 1)))
        else:
            self._time_idx = 0

        self.vehicle.set_position(self._origin_lat, self._origin_lon)
        self._target_idx = 0

        snap = self._merge_snapshot(self._time_idx)
        return self._get_obs(snap), {
            "time_index": self._time_idx,
            "parking_ids": list(self.parking_ids),
        }

    def _advance_vehicle_toward(self, lot_idx: int) -> None:
        lot = self.lots[lot_idx]
        dlat = lot.latitude - self.vehicle.latitude
        dlon = lot.longitude - self.vehicle.longitude
        dist = math.sqrt(dlat * dlat + dlon * dlon)
        if dist < 1e-14:
            return
        step = min(self.move_step_deg, dist)
        self.vehicle.latitude += (dlat / dist) * step
        self.vehicle.longitude += (dlon / dist) * step

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Tek adım: aksiyon = hedef otopark indeksi.

        Ödül mantığı (ekip için özet):
        - Her adımda küçük negatif step_cost (zaman/yakıt proxy’si).
        - Hedefe yeterince yakınsa: boş yer oranı (empty_frac) kadar bonus,
          doluluk oranı kadar ceza (yoğunluk).

        Döngü güvenliği: terminated (varış) veya truncated (max adım) en geç
        max_episode_steps içinde gelir; zaman indeksi son zaman damgasında
        sabitlenir ama adım sayacı yine de artar → sessiz sonsuz döngü yok.
        """
        self._target_idx = int(action)
        if not self.action_space.contains(self._target_idx):
            raise ValueError(f"Geçersiz aksiyon: {action}")

        self._advance_vehicle_toward(self._target_idx)

        # Zamanı ilerlet; veri sonuna gelindiyse son snapshot’ta kalınır
        self._time_idx = min(self._time_idx + 1, len(self._times) - 1)
        snap = self._merge_snapshot(self._time_idx)

        lot = self.lots[self._target_idx]
        dist_km = haversine_km(
            self.vehicle.latitude,
            self.vehicle.longitude,
            lot.latitude,
            lot.longitude,
        )
        arrived = dist_km <= self.arrival_threshold_km

        occ, cap = snap[lot.parking_id]
        cap = max(1.0, cap)
        occ = float(np.clip(occ, 0.0, cap))
        empty_frac = (cap - occ) / cap
        occ_ratio = occ / cap

        reward = self.step_cost
        if arrived:
            reward += self.availability_weight * empty_frac
            reward -= self.congestion_weight * occ_ratio

        self._step_count += 1
        terminated = bool(arrived)
        truncated = self._step_count >= self.max_episode_steps

        obs = self._get_obs(snap)
        info: Dict[str, Any] = {
            "distance_to_target_km": dist_km,
            "target_parking_id": lot.parking_id,
            "target_index": self._target_idx,
            "arrived": arrived,
            "occupancy_ratio": occ_ratio,
            "empty_fraction": empty_frac,
            "time_index": self._time_idx,
        }
        return obs, float(reward), terminated, truncated, info

    def render(self) -> None:
        print(
            f"[SmartParking] t={self._time_idx} araç=({self.vehicle.latitude:.5f},{self.vehicle.longitude:.5f}) "
            f"hedef={self.lots[self._target_idx].parking_id}"
        )


def main() -> None:
    """Rastgele politika ile kısa doğrulama."""
    env = SmartParkingEnv(
        data_path="data/processed/train.csv",
        max_episode_steps=80,
        randomize_start_time=True,
    )
    obs, _ = env.reset(seed=42)
    total = 0.0
    for _ in range(80):
        a = env.action_space.sample()
        obs, r, term, trunc, info = env.step(a)
        total += r
        if term or trunc:
            break
    print(f"Örnek episode toplam ödül (rastgele politika): {total:.3f}")
    print(f"Gözlem boyutu: {obs.shape}, Aksiyon sayısı: {env.action_space.n}")


if __name__ == "__main__":
    main()
