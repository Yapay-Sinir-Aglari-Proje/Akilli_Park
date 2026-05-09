"""
Grid tabanlı çok adımlı navigasyon ortamı (varsayılan).

Geriye dönük isimler: ParkingRoutingEnv, build_routing_scenarios.
"""

from __future__ import annotations

from env.grid_navigation_env import (
    GridNavEpisodeConfig,
    GridParkingNavigationEnv,
    build_grid_nav_episode_configs,
)

ParkingRoutingEnv = GridParkingNavigationEnv
build_routing_scenarios = build_grid_nav_episode_configs

__all__ = [
    "GridNavEpisodeConfig",
    "GridParkingNavigationEnv",
    "ParkingRoutingEnv",
    "build_grid_nav_episode_configs",
    "build_routing_scenarios",
]
