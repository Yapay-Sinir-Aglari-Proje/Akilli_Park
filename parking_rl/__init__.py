"""Akıllı otopark RL simülasyonu: ortam, konum ve ödül modeli.

Kullanım:
    from parking_rl.smart_parking_env import SmartParkingEnv
    from parking_rl.grid_parking_env import GridParkingEnv
"""

__all__ = ["SmartParkingEnv", "GridParkingEnv", "ExternalParkingEnv"]


def __getattr__(name: str):
    if name == "SmartParkingEnv":
        from parking_rl.smart_parking_env import SmartParkingEnv

        return SmartParkingEnv
    if name == "GridParkingEnv":
        from parking_rl.grid_parking_env import GridParkingEnv

        return GridParkingEnv
    if name == "ExternalParkingEnv":
        from parking_rl.external_env import ExternalParkingEnv

        return ExternalParkingEnv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
