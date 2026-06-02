"""
Grid PPO bölümü GIF: output/rl_rollout.gif

  python -m evaluation.animate_rl_rollout
  python -m evaluation.animate_rl_rollout --seconds-per-frame 6
  python -m evaluation.animate_rl_rollout --presentation --max-attempts 100
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
    parser.add_argument(
        "--seconds-per-frame",
        type=float,
        default=6.0,
        help="Kare başına süre (saniye); varsayılan 6 (çok yavaş inceleme).",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Verilirse kare süresi = 1/fps saniye (--seconds-per-frame yerine).",
    )
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--out", type=Path, default=OUTPUT_DIR / "rl_rollout.gif")
    parser.add_argument(
        "--presentation",
        action="store_true",
        help="Birden fazla bölüm dene; success + en kısa adımlı rotayı GIF yap.",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=80,
        help="--presentation ile en fazla kaç tohum denensin.",
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
    spf = float(args.seconds_per_frame)
    if args.fps is not None:
        spf = 1.0 / max(float(args.fps), 0.05)
    save_episode_gif_ppo(
        cfgs,
        MODELS_DIR / "ppo_agent",
        _vec_path(),
        args.out,
        seconds_per_frame=spf,
        seed=args.seed,
        pick_shortest_success=bool(args.presentation),
        max_candidate_episodes=int(args.max_attempts),
    )
    tag = " [presentation]" if args.presentation else ""
    print(f"[anim]{tag} {args.out}  (~{spf:.2f}s/kare)")


if __name__ == "__main__":
    main()
