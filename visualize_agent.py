"""
RL ajan animasyonları: canlı metrik paneli + açıklanabilir GIF çıktıları.

  python visualize_agent.py --algo ppo
  python visualize_agent.py --algo dqn --presentation
  python visualize_agent.py --all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from config import (
    GIF_PRESENTATION_MAX_ATTEMPTS,
    GIF_SECONDS_PER_FRAME,
    GRID_MAX_EPISODE_STEPS,
    RANDOM_SEED,
)
from environment import (
    GridParkingNavigationEnv,
    build_grid_nav_episode_configs,
)
from paths import DATA_PROCESSED, MODELS_DIR, OUTPUT_DIR, PREDICTIONS_DIR, ensure_output


def _vecnormalize_path(algo: str) -> Path:
    for p in (
        MODELS_DIR / f"vecnormalize_{algo}.pkl",
        MODELS_DIR / f"vecnormalize_{algo}",
    ):
        if p.exists():
            return p
    raise FileNotFoundError(
        f"VecNormalize bulunamadı ({algo}). Önce: python train_{algo}.py veya train_rl.py --algo {algo}"
    )


def _load_font(size: int) -> ImageFont.ImageFont:
    for name in ("arial.ttf", "segoeui.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_metrics_overlay(
    frame: np.ndarray,
    info: Dict[str, Any],
    *,
    algorithm: str,
    episode_display: int = 0,
) -> np.ndarray:
    """Grid karesinin altına canlı metrik paneli ekler."""
    img = Image.fromarray(np.asarray(frame, dtype=np.uint8))
    panel_h = max(118, img.height // 5)
    canvas = Image.new("RGB", (img.width, img.height + panel_h), (248, 248, 252))
    canvas.paste(img, (0, 0))
    draw = ImageDraw.Draw(canvas)
    font = _load_font(max(11, img.width // 42))
    font_title = _load_font(max(13, img.width // 36))

    status = str(info.get("status", "RUNNING"))
    success = bool(info.get("success"))
    if success:
        status = "SUCCESS"

    lines = [
        f"{algorithm.upper()}  |  Episode #{episode_display}  |  {status}",
        (
            f"Step: {info.get('step', info.get('steps', '?'))}  |  "
            f"Reward: {float(info.get('instant_reward', 0)):.2f}  |  "
            f"Cumulative: {float(info.get('cumulative_reward', 0)):.1f}"
        ),
        (
            f"Dist→Goal: {info.get('distance_to_goal', '?')}  |  "
            f"Action: {info.get('action_name', '?')}  |  "
            f"Path len: {info.get('path_length', '?')}"
        ),
        (
            f"Collisions: {info.get('collision_count', 0)}  |  "
            f"Invalid moves: {info.get('invalid_moves', 0)}  |  "
            f"Visited cells: {info.get('visited_cells', '?')}"
        ),
    ]
    draw.rectangle([0, img.height, img.width, img.height + panel_h], fill=(235, 240, 248))
    draw.text((8, img.height + 4), lines[0], fill=(20, 40, 90), font=font_title)
    y = img.height + 26
    for line in lines[1:]:
        draw.text((8, y), line, fill=(40, 40, 50), font=font)
        y += 22
    return np.asarray(canvas)


def _unwrap_vec(vec: Any) -> GridParkingNavigationEnv:
    w = vec.venv.envs[0]
    u = w.unwrapped
    if not isinstance(u, GridParkingNavigationEnv):
        u = getattr(u, "unwrapped", u)
    assert isinstance(u, GridParkingNavigationEnv)
    return u


def rollout_sb3_explained_frames(
    vec: Any,
    model: Any,
    rollout_seed: int,
    max_steps: int,
    *,
    algorithm: str,
    episode_display: int,
) -> Tuple[List[np.ndarray], bool, int, Dict[str, Any]]:
    """SB3 modeli ile bölüm oynatır; her karede metrik paneli vardır."""
    vec.seed(int(rollout_seed))
    obs = vec.reset()
    env0 = _unwrap_vec(vec)
    frames: List[np.ndarray] = []
    last_info: Dict[str, Any] = {}

    f0 = env0.render()
    if f0 is not None:
        info0 = {
            "step": 0,
            "instant_reward": 0.0,
            "cumulative_reward": 0.0,
            "distance_to_goal": env0._manhattan(env0._agent, env0._goal),
            "action_name": "START",
            "path_length": 0,
            "collision_count": 0,
            "invalid_moves": 0,
            "visited_cells": len(env0._visited),
            "status": "START",
            "success": False,
        }
        frames.append(
            draw_metrics_overlay(
                np.array(f0),
                info0,
                algorithm=algorithm,
                episode_display=episode_display,
            )
        )

    done = False
    while not done:
        act, _ = model.predict(obs, deterministic=True)
        obs, reward, dones, infos = vec.step(act)
        done = bool(dones[0])
        if isinstance(infos, (list, tuple)) and infos and infos[0]:
            last_info = dict(infos[0])
        else:
            last_info = {}
        last_info.setdefault("instant_reward", float(reward[0]) if hasattr(reward, "__len__") else reward)
        env0 = _unwrap_vec(vec)
        fr = env0.render()
        if fr is not None:
            frames.append(
                draw_metrics_overlay(
                    np.array(fr),
                    last_info,
                    algorithm=algorithm,
                    episode_display=episode_display,
                )
            )

    env_check = _unwrap_vec(vec)
    reached = bool(last_info.get("success")) or (
        done and env_check._agent == env_check._goal
    )
    n_steps = int(last_info.get("steps", last_info.get("step", 0)))
    if n_steps <= 0 and reached:
        n_steps = max(0, len(frames) - 1)
    if not frames:
        raise RuntimeError("GIF için kare üretilemedi.")
    return frames, reached, n_steps, last_info


def save_explained_gif_sb3(
    episode_configs: list,
    algo: str,
    out_path: Path,
    *,
    seed: int = RANDOM_SEED,
    seconds_per_frame: float = GIF_SECONDS_PER_FRAME,
    max_steps: int = GRID_MAX_EPISODE_STEPS,
    scenario: str = "medium",
    pick_shortest_success: bool = False,
    max_candidate_episodes: int = GIF_PRESENTATION_MAX_ATTEMPTS,
    episode_display: int = 1,
) -> Path:
    from stable_baselines3 import DQN, PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    algo_l = algo.lower()
    vec_p = _vecnormalize_path(algo_l)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def factory():
        return GridParkingNavigationEnv(
            episode_configs,
            seed=seed,
            render_mode="rgb_array",
            max_episode_steps=max_steps,
            scenario=scenario,
        )

    venv = DummyVecEnv([factory])
    vec = VecNormalize.load(str(vec_p), venv)
    vec.training = False
    vec.norm_reward = False
    load_path = str(MODELS_DIR / f"{algo_l}_agent")
    if algo_l == "ppo":
        model = PPO.load(load_path, env=vec, custom_objects={"lr_schedule": lambda _: 3e-4})
    else:
        model = DQN.load(load_path, env=vec)

    try:
        if not pick_shortest_success:
            frames, _, _, _ = rollout_sb3_explained_frames(
                vec,
                model,
                int(seed),
                max_steps,
                algorithm=algo_l,
                episode_display=episode_display,
            )
        else:
            best: Optional[Tuple[int, List[np.ndarray]]] = None
            for k in range(max(1, max_candidate_episodes)):
                frames_k, ok, n_steps, _ = rollout_sb3_explained_frames(
                    vec,
                    model,
                    int(seed) + k,
                    max_steps,
                    algorithm=algo_l,
                    episode_display=episode_display,
                )
                if ok and (best is None or n_steps < best[0]):
                    best = (n_steps, frames_k)
            if best is None:
                raise RuntimeError(
                    f"{max_candidate_episodes} denemede başarılı bölüm yok — eğitimi veya --max-attempts artırın."
                )
            frames = best[1]
        spf = float(max(seconds_per_frame, 0.05))
        imageio.mimsave(str(out_path), frames, duration=spf)
    finally:
        vec.close()
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Açıklanabilir RL ajan GIF'leri üretir.")
    parser.add_argument("--algo", choices=["ppo", "dqn", "all"], default="all")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--scenario", default="medium")
    parser.add_argument("--seconds-per-frame", type=float, default=GIF_SECONDS_PER_FRAME)
    parser.add_argument("--presentation", action="store_true")
    parser.add_argument("--max-attempts", type=int, default=GIF_PRESENTATION_MAX_ATTEMPTS)
    args = parser.parse_args()

    ensure_output()
    pred = PREDICTIONS_DIR / "test_predictions.csv"
    cfgs = build_grid_nav_episode_configs(
        DATA_PROCESSED / "processed.parquet",
        pred if pred.exists() else None,
        split="test",
        base_seed=RANDOM_SEED,
    )
    algos = ["ppo", "dqn"] if args.algo == "all" else [args.algo]
    for algo in algos:
        out = OUTPUT_DIR / f"{algo}_agent_explained.gif"
        try:
            p = save_explained_gif_sb3(
                cfgs,
                algo,
                out,
                seed=args.seed,
                seconds_per_frame=args.seconds_per_frame,
                scenario=args.scenario,
                pick_shortest_success=bool(args.presentation),
                max_candidate_episodes=int(args.max_attempts),
            )
            print(f"[viz] {p}")
        except FileNotFoundError as exc:
            print(f"[viz] Atlandı ({algo}): {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
