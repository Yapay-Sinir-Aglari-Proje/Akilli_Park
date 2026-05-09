"""
Grid PPO bölümü GIF: output/rl_rollout.gif

  python -m evaluation.animate_rl_rollout --fps 8
  python -m evaluation.animate_rl_rollout --park-frames 16
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


def _vec_path() -> Path:
    for p in (MODELS_DIR / "vecnormalize_ppo.pkl", MODELS_DIR / "vecnormalize_ppo"):
        if p.exists():
            return p
    raise FileNotFoundError("vecnormalize_ppo bulunamadı")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--out", type=Path, default=OUTPUT_DIR / "rl_rollout.gif")
    parser.add_argument(
        "--park-frames",
        type=int,
        default=12,
        help="Hedefe varış sonrası GIF'e eklenecek kutlama kare sayısı (0=kapat).",
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
    save_episode_gif_ppo(
        cfgs,
        MODELS_DIR / "ppo_agent",
        _vec_path(),
        args.out,
        fps=args.fps,
        seed=args.seed,
        park_celebration_frames=args.park_frames,
    )
    print(f"[anim] {args.out}")


if __name__ == "__main__":
    main()
