"""
Eğitilmiş PPO ile tek bölüm oynatıp output/parking_agent.gif üretir.

Kullanım:
  python -m evaluation.record_parking_gif --fps 8 --seed 42
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml_config import RANDOM_SEED
from paths import DATA_PROCESSED, MODELS_DIR, OUTPUT_DIR, PREDICTIONS_DIR, ensure_output

from env.grid_navigation_env import build_grid_nav_episode_configs, save_episode_gif_ppo


def _vecnormalize_file() -> Path:
    pkl = MODELS_DIR / "vecnormalize_ppo.pkl"
    raw = MODELS_DIR / "vecnormalize_ppo"
    if pkl.exists():
        return pkl
    if raw.exists():
        return raw
    raise FileNotFoundError("VecNormalize bulunamadı — train_rl.py --algo ppo çalıştırın.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--out", type=Path, default=OUTPUT_DIR / "parking_agent.gif")
    parser.add_argument(
        "--park-frames",
        type=int,
        default=12,
        help="Başarıda GIF sonuna eklenecek kutlama kare sayısı (0=kapat).",
    )
    args = parser.parse_args()

    ensure_output()
    cfgs = build_grid_nav_episode_configs(
        DATA_PROCESSED / "processed.parquet",
        PREDICTIONS_DIR / "test_predictions.csv"
        if (PREDICTIONS_DIR / "test_predictions.csv").exists()
        else None,
        split="test",
        base_seed=RANDOM_SEED,
    )
    vec_p = _vecnormalize_file()
    out = save_episode_gif_ppo(
        cfgs,
        MODELS_DIR / "ppo_agent",
        vec_p,
        args.out,
        fps=args.fps,
        seed=args.seed,
        park_celebration_frames=args.park_frames,
    )
    print(f"[gif] {out}")


if __name__ == "__main__":
    main()
