"""
Eğitilmiş PPO ile bölüm oynatıp output/parking_agent.gif üretir.

Kullanım:
  python -m evaluation.record_parking_gif --seed 42
  python -m evaluation.record_parking_gif --seconds-per-frame 6
  python -m evaluation.record_parking_gif --presentation --max-attempts 100
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
    parser.add_argument(
        "--seconds-per-frame",
        type=float,
        default=6.0,
        help="Her GIF karesinin ekranda kalma süresi (saniye). Varsayılan: 6 (çok yavaş inceleme).",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="İsteğe bağlı: saniye başına kare (örn. 0.2 → kare başına 5 sn). Verilirse --seconds-per-frame ezilir.",
    )
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--out", type=Path, default=OUTPUT_DIR / "parking_agent.gif")
    parser.add_argument(
        "--presentation",
        action="store_true",
        help="Birden fazla bölüm dene; yalnızca success olanlar arasından en kısa adımlıyı GIF yap (tez/sunum).",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=80,
        help="--presentation ile en fazla kaç farklı tohum (bölüm) denensin.",
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
    spf = float(args.seconds_per_frame)
    if args.fps is not None:
        spf = 1.0 / max(float(args.fps), 0.05)
    out = save_episode_gif_ppo(
        cfgs,
        MODELS_DIR / "ppo_agent",
        vec_p,
        args.out,
        seconds_per_frame=spf,
        seed=args.seed,
        pick_shortest_success=bool(args.presentation),
        max_candidate_episodes=int(args.max_attempts),
    )
    tag = " [presentation: shortest success]" if args.presentation else ""
    print(f"[gif]{tag} {out}  (~{spf:.2f}s/kare)")


if __name__ == "__main__":
    main()
