"""
Sunum için 3 ayrı PPO GIF’i: kısa (best case), orta (20–40 adım bandı), uzun ama temiz başarı.

  python -m evaluation.record_sunum_gif_set
  python -m evaluation.record_sunum_gif_set --seconds-per-frame 5 --scan 300
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import imageio.v2 as imageio
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from env.grid_navigation_env import (
    GridParkingNavigationEnv,
    build_grid_nav_episode_configs,
    rollout_ppo_gif_frames,
)
from ml_config import GRID_MAX_EPISODE_STEPS, RANDOM_SEED
from paths import DATA_PROCESSED, MODELS_DIR, OUTPUT_DIR, PREDICTIONS_DIR, ensure_output


def _vecnormalize_ppo() -> Path:
    for p in (MODELS_DIR / "vecnormalize_ppo.pkl", MODELS_DIR / "vecnormalize_ppo"):
        if p.exists():
            return p
    raise FileNotFoundError("vecnormalize_ppo bulunamadı — önce train_rl.py --algo ppo")


def _write_gif(path: Path, frames: List[Any], seconds_per_frame: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    spf = float(max(seconds_per_frame, 0.05))
    imageio.mimsave(str(path), frames, duration=spf)


def _load_vec_model(
    episode_configs: list,
    vec_pkl: Path,
    seed: int,
    max_steps: int,
) -> Tuple[Any, Any]:
    def factory():
        return GridParkingNavigationEnv(
            episode_configs,
            seed=seed,
            render_mode="rgb_array",
            max_episode_steps=max_steps,
        )

    venv = DummyVecEnv([factory])
    vec = VecNormalize.load(str(vec_pkl), venv)
    vec.training = False
    vec.norm_reward = False
    model = PPO.load(
        str(MODELS_DIR / "ppo_agent"),
        env=vec,
        custom_objects={"lr_schedule": lambda _: 3e-4},
    )
    return vec, model


def _ok_metrics(r: Dict[str, Any], max_revisit: int, max_loop: int) -> bool:
    inf = r["info"]
    rev = int(inf.get("revisit_count") or inf.get("revisit_events") or 0)
    loops = int(inf.get("loop_count") or inf.get("loop_penalty_events") or 0)
    return rev <= max_revisit and loops <= max_loop


def _pick_medium_primary(
    successful: List[Dict[str, Any]],
    short_seed: int,
    mid_lo: int,
    mid_hi: int,
    mid_target: int,
) -> Dict[str, Any]:
    pool = [r for r in successful if r["seed"] != short_seed]
    if not pool:
        return min(successful, key=lambda r: (r["steps"], r["seed"]))
    band = [r for r in pool if mid_lo <= r["steps"] <= mid_hi]
    if band:
        return min(band, key=lambda r: (abs(r["steps"] - mid_target), r["seed"]))
    return min(pool, key=lambda r: (abs(r["steps"] - mid_target), r["seed"]))


def _pick_long_after_medium(
    successful: List[Dict[str, Any]],
    short_seed: int,
    medium: Dict[str, Any],
    long_min: int,
    long_max: int,
    max_revisit: int,
    max_loop: int,
) -> Tuple[Dict[str, Any], str]:
    """Kısa + orta dışı; mümkünse ortadan daha uzun ve [long_min, long_max] bandına yakın."""
    med_steps = int(medium["steps"])
    used = {short_seed, int(medium["seed"])}
    pool = [r for r in successful if r["seed"] not in used]
    if not pool:
        pool = [r for r in successful if r["seed"] != short_seed]

    def ok(r: Dict[str, Any]) -> bool:
        return _ok_metrics(r, max_revisit, max_loop)

    harder = [r for r in pool if ok(r) and r["steps"] > med_steps]
    strict = [r for r in harder if long_min <= r["steps"] <= long_max]
    if strict:
        return max(strict, key=lambda r: (r["steps"], -r["seed"])), "strict_longer_than_orta"

    relaxed = [
        r
        for r in harder
        if ok(r) and max(38, long_min - 15) <= r["steps"] <= min(long_max + 35, int(GRID_MAX_EPISODE_STEPS) - 3)
    ]
    if relaxed:
        return max(relaxed, key=lambda r: (r["steps"], -r["seed"])), "relaxed_longer_than_orta"

    any_clean = [r for r in pool if ok(r)]
    if any_clean:
        longer = [r for r in any_clean if r["steps"] > med_steps]
        if longer:
            return max(longer, key=lambda r: (r["steps"], -r["seed"])), "longest_clean_longer_than_orta"
        return max(any_clean, key=lambda r: (r["steps"], -r["seed"])), "longest_clean_remaining"

    return max(pool, key=lambda r: (r["steps"], -r["seed"])), "fallback_any_remaining"


def main() -> None:
    parser = argparse.ArgumentParser(description="Sunum: 3 farklı uzunlukta başarılı PPO GIF’i.")
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR, help="Çıktı klasörü")
    parser.add_argument("--base-seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--scan", type=int, default=350, help="Kaç farklı tohum taransın")
    parser.add_argument("--seconds-per-frame", type=float, default=5.0)
    parser.add_argument("--mid-lo", type=int, default=20, help="Orta GIF ideal alt adım")
    parser.add_argument("--mid-hi", type=int, default=40, help="Orta GIF ideal üst adım")
    parser.add_argument("--mid-target", type=int, default=30)
    parser.add_argument("--long-min", type=int, default=45)
    parser.add_argument("--long-max", type=int, default=92)
    parser.add_argument("--max-revisit", type=int, default=16)
    parser.add_argument("--max-loop", type=int, default=6)
    args = parser.parse_args()

    ensure_output()
    cfgs = build_grid_nav_episode_configs(
        DATA_PROCESSED / "processed.parquet",
        PREDICTIONS_DIR / "test_predictions.csv"
        if (PREDICTIONS_DIR / "test_predictions.csv").exists()
        else None,
        split="test",
        base_seed=int(args.base_seed),
    )
    vec_p = _vecnormalize_ppo()
    max_steps = int(GRID_MAX_EPISODE_STEPS)
    vec, model = _load_vec_model(cfgs, vec_p, args.base_seed, max_steps)

    successful: List[Dict[str, Any]] = []
    try:
        for k in range(max(1, int(args.scan))):
            seed_k = int(args.base_seed) + k
            frames, ok, steps, info = rollout_ppo_gif_frames(vec, model, seed_k, max_steps)
            if ok:
                successful.append(
                    {"seed": seed_k, "steps": steps, "frames": frames, "info": dict(info)}
                )
    finally:
        vec.close()

    if len(successful) < 1:
        raise RuntimeError(
            f"{args.scan} denemede başarılı bölüm yok — scan artırın veya evaluate ile PPO kontrol edin."
        )

    short = min(successful, key=lambda r: (r["steps"], r["seed"]))
    medium = _pick_medium_primary(
        successful,
        short["seed"],
        int(args.mid_lo),
        int(args.mid_hi),
        int(args.mid_target),
    )
    long_r, long_tag = _pick_long_after_medium(
        successful,
        short["seed"],
        medium,
        int(args.long_min),
        int(args.long_max),
        int(args.max_revisit),
        int(args.max_loop),
    )

    out_dir = Path(args.out_dir)
    paths = {
        "kisa": out_dir / "sunum_ppo_episode_kisa.gif",
        "orta": out_dir / "sunum_ppo_episode_orta.gif",
        "uzun": out_dir / "sunum_ppo_episode_uzun.gif",
    }
    _write_gif(paths["kisa"], short["frames"], float(args.seconds_per_frame))
    _write_gif(paths["orta"], medium["frames"], float(args.seconds_per_frame))
    _write_gif(paths["uzun"], long_r["frames"], float(args.seconds_per_frame))

    seeds = (short["seed"], medium["seed"], long_r["seed"])
    if len(set(seeds)) < 3:
        print(
            "  [uyarı] Bazı GIF'ler aynı tohumdan; --scan değerini artırın (ör. 500).",
            file=sys.stderr,
        )

    def _line(label: str, r: Dict[str, Any], note: str = "") -> str:
        rev = int(r["info"].get("revisit_count") or r["info"].get("revisit_events") or 0)
        lp = int(r["info"].get("loop_count") or r["info"].get("loop_penalty_events") or 0)
        extra = f" {note}" if note else ""
        return f"  {label}: seed={r['seed']} steps={r['steps']} revisit={rev} loop={lp}{extra}"

    print("[sunum-gif-set]")
    print(_line("Kısa (best / min adım)", short))
    print(_line("Orta (ana sunum bandı)", medium))
    print(_line("Uzun (zor ama başarılı)", long_r, f"[seçim={long_tag}]"))
    for k, p in paths.items():
        print(f"  -> {p}")


if __name__ == "__main__":
    main()
