"""
Çok adımlı otopark grid navigasyon ortamı (Gymnasium).

Eylem: 0=up, 1=down, 2=left, 3=right
Gözlem: [duvar | park yerleri | hedef | ajan] düzleştirilmiş (4 * H * W)
render: human | rgb_array (GIF/video için)

Ödül (ml_config GRID_* sabitleri):
- Adım maliyeti: -GRID_STEP_COST
- Manhattan shaping: GRID_MANHATTAN_SHAPING_SCALE * (d_old - d_new) her geçerli adımda
- İlk ziyaret hücre bonusu / tekrar ziyaret cezası: visited set
- A↔B salınım (son GRID_LOOP_WINDOW pozisyon): GRID_LOOP_PENALTY
- Hedef: +GRID_GOAL_BONUS; süre aşımı (truncated, başarısız): GRID_TIMEOUT_PENALTY
- Tüm ödüller GRID_REWARD_CLIP ile kırpılır; train_rl VecNormalize clip_reward aynı ölçekte.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from PIL import Image, ImageDraw, ImageFont

from ml_config import (
    GRID_FIRST_VISIT_BONUS,
    GRID_GOAL_BONUS,
    GRID_HEIGHT,
    GRID_LOOP_PENALTY,
    GRID_LOOP_WINDOW,
    GRID_MANHATTAN_SHAPING_SCALE,
    GRID_MAX_EPISODE_STEPS,
    GRID_REVISIT_PENALTY,
    GRID_REWARD_CLIP,
    GRID_STEP_COST,
    GRID_TIMEOUT_PENALTY,
    GRID_WIDTH,
)
from utils.coordinates import stable_parking_coordinates


ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3
DR_DC = [(-1, 0), (1, 0), (0, -1), (0, 1)]


@dataclass
class GridNavEpisodeConfig:
    walls: np.ndarray
    parking_cells: List[Tuple[int, int]]
    goal_cell: Tuple[int, int]
    agent_start: Tuple[int, int]
    height: int
    width: int


def _place_lots_on_grid(
    lot_ids: List[str],
    height: int,
    width: int,
) -> Dict[str, Tuple[int, int]]:
    coords = stable_parking_coordinates(lot_ids)
    used: Set[Tuple[int, int]] = set()
    out: Dict[str, Tuple[int, int]] = {}
    interior_h = max(1, height - 2)
    interior_w = max(1, width - 2)
    for lid in lot_ids:
        lat, lon = coords[str(lid)]
        r = 1 + int(lat * 0.999 * (interior_h - 1)) if interior_h > 1 else 1
        c = 1 + int(lon * 0.999 * (interior_w - 1)) if interior_w > 1 else 1
        r = int(np.clip(r, 1, height - 2))
        c = int(np.clip(c, 1, width - 2))
        k = 0
        while (r, c) in used and k < height * width:
            c = 1 + (c % (width - 2))
            r = 1 + (r % (height - 2))
            k += 1
        used.add((r, c))
        out[str(lid)] = (r, c)
    return out


def build_grid_nav_episode_configs(
    processed_parquet: Path | str,
    predictions_csv: Optional[Path | str] = None,
    split: Optional[str] = None,
    max_episodes: int = 5000,
    height: int = GRID_HEIGHT,
    width: int = GRID_WIDTH,
    base_seed: int = 42,
) -> List[GridNavEpisodeConfig]:
    path = Path(processed_parquet)
    df_full = pd.read_parquet(path)
    lot_ids = sorted(df_full["SystemCodeNumber"].astype(str).unique())
    cap_median = df_full.groupby("SystemCodeNumber")["Capacity"].median().reindex(lot_ids)

    df = df_full if split is None else df_full[df_full["split"] == split].copy()
    if df.empty:
        raise ValueError(f"Split boş: {split!r}")

    df["LastUpdated"] = pd.to_datetime(df["LastUpdated"])

    walls = np.ones((height, width), dtype=bool)
    walls[1 : height - 1, 1 : width - 1] = False

    configs: List[GridNavEpisodeConfig] = []
    rng = np.random.default_rng(base_seed)

    for _ts, g in df.groupby("LastUpdated", sort=False):
        if len(configs) >= max_episodes:
            break
        g = g.copy()
        g["SystemCodeNumber"] = g["SystemCodeNumber"].astype(str)
        idx = g.set_index("SystemCodeNumber")
        cap = idx["Capacity"].astype(float).reindex(lot_ids)
        occ = idx["Occupancy"].astype(float).reindex(lot_ids)
        cap = cap.fillna(cap_median)
        if cap.isna().any():
            continue
        occ = occ.fillna(cap)
        mask = np.array([1.0 if lid in idx.index else 0.0 for lid in lot_ids], dtype=np.float32)
        occ_ratio = (occ / cap).to_numpy(dtype=np.float32)
        occ_ratio = np.clip(occ_ratio, 0.0, 1.0)

        cell_map = _place_lots_on_grid(lot_ids, height, width)
        parking_cells = [cell_map[lid] for lid in lot_ids]

        valid_lot_idx = [i for i in range(len(lot_ids)) if mask[i] > 0.5]
        if not valid_lot_idx:
            continue
        best_i = int(min(valid_lot_idx, key=lambda i: float(occ_ratio[i])))
        goal_cell = cell_map[lot_ids[best_i]]

        blocked: Set[Tuple[int, int]] = set(parking_cells)
        blocked.discard(goal_cell)
        free_cells = [
            (r, c)
            for r in range(1, height - 1)
            for c in range(1, width - 1)
            if not walls[r, c] and (r, c) not in blocked and (r, c) != goal_cell
        ]
        if not free_cells:
            continue
        agent_start = free_cells[int(rng.integers(0, len(free_cells)))]

        configs.append(
            GridNavEpisodeConfig(
                walls=walls.copy(),
                parking_cells=parking_cells,
                goal_cell=goal_cell,
                agent_start=agent_start,
                height=height,
                width=width,
            )
        )

    if not configs:
        raise ValueError("GridNav episode üretilemedi.")
    return configs


class GridParkingNavigationEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        episode_configs: List[GridNavEpisodeConfig],
        seed: int = 42,
        max_episode_steps: int = GRID_MAX_EPISODE_STEPS,
        wall_penalty: float = 0.5,
        render_mode: Optional[str] = None,
        cell_pixels: int = 24,
        reward_debug: bool = False,
        **kwargs: Any,
    ):
        kwargs.pop("mode", None)
        super().__init__()
        self._rng = np.random.default_rng(seed)
        self.episode_configs = episode_configs
        self.max_episode_steps = max_episode_steps
        self.wall_penalty = wall_penalty
        self.render_mode = render_mode
        self.cell_pixels = cell_pixels
        self.reward_debug = reward_debug

        self.step_cost = float(GRID_STEP_COST)
        self.manhattan_scale = float(GRID_MANHATTAN_SHAPING_SCALE)
        self.goal_bonus = float(GRID_GOAL_BONUS)
        self.timeout_penalty = float(GRID_TIMEOUT_PENALTY)
        self.first_visit_bonus = float(GRID_FIRST_VISIT_BONUS)
        self.revisit_penalty = float(GRID_REVISIT_PENALTY)
        self.loop_penalty = float(GRID_LOOP_PENALTY)
        self._loop_window = int(GRID_LOOP_WINDOW)
        self.reward_clip = float(GRID_REWARD_CLIP)

        self.height = episode_configs[0].height
        self.width = episode_configs[0].width
        flat_dim = 4 * self.height * self.width
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(flat_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(4)

        self._walls: Optional[np.ndarray] = None
        self._parking_set: Set[Tuple[int, int]] = set()
        self._goal: Tuple[int, int] = (0, 0)
        self._agent: Tuple[int, int] = (0, 0)
        self._trail: List[Tuple[int, int]] = []
        self._steps = 0
        self._fig = None
        self._ax = None
        self._recent_positions: deque[Tuple[int, int]] = deque(maxlen=self._loop_window)
        self._visited: Set[Tuple[int, int]] = set()
        self._episode_loop_events = 0
        self._episode_revisit_events = 0
        # GIF vb. için: hedefe varış sonrası "park etti" kareleri (_render_rgb_array)
        self._celebrate_parking: bool = False

    def _manhattan(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _detect_two_cell_oscillation(self) -> bool:
        """Son pozisyonlarda A↔B↔A↔B (veya uzatılmış) deseni."""
        p = list(self._recent_positions)
        if len(p) < 4:
            return False
        if p[-1] == p[-3] and p[-2] == p[-4] and p[-1] != p[-2]:
            return True
        if len(p) >= 6 and p[-1] == p[-3] == p[-5] and p[-2] == p[-4] == p[-6] and p[-1] != p[-2]:
            return True
        return False

    def _clip(self, r: float) -> float:
        return float(np.clip(r, -self.reward_clip, self.reward_clip))

    @staticmethod
    def _draw_park_checkmark(
        drw: ImageDraw.ImageDraw,
        x0: int,
        y0: int,
        cs: int,
        stroke: Tuple[int, int, int] = (255, 255, 255),
        width: int = 3,
    ) -> None:
        """Hücre içinde kalın onay işareti (✓)."""
        pad = max(2, cs // 6)
        p1 = (x0 + pad, y0 + cs // 2 - pad // 2)
        p2 = (x0 + cs // 3, y0 + cs - pad)
        p3 = (x0 + cs - pad, y0 + pad * 2)
        w = max(width, cs // 8)
        drw.line([p1, p2], fill=stroke, width=w)
        drw.line([p2, p3], fill=stroke, width=w)

    def _observation(self) -> np.ndarray:
        assert self._walls is not None
        H, W = self.height, self.width
        wall_f = self._walls.astype(np.float32)
        park = np.zeros((H, W), dtype=np.float32)
        for r, c in self._parking_set:
            park[r, c] = 1.0
        goal = np.zeros((H, W), dtype=np.float32)
        goal[self._goal[0], self._goal[1]] = 1.0
        agent = np.zeros((H, W), dtype=np.float32)
        agent[self._agent[0], self._agent[1]] = 1.0
        return np.concatenate(
            [wall_f.ravel(), park.ravel(), goal.ravel(), agent.ravel()]
        ).astype(np.float32)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        idx = int(self._rng.integers(0, len(self.episode_configs)))
        cfg = self.episode_configs[idx]
        self._walls = cfg.walls.copy()
        self._parking_set = set(cfg.parking_cells)
        self._goal = cfg.goal_cell
        self._agent = cfg.agent_start
        self._trail = [self._agent]
        self._steps = 0
        self._recent_positions = deque([self._agent], maxlen=self._loop_window)
        self._visited = {self._agent}
        self._episode_loop_events = 0
        self._episode_revisit_events = 0
        self._celebrate_parking = False
        return self._observation(), {}

    def step(self, action: int):
        self._steps += 1
        a = int(action)
        if a < 0 or a > 3:
            obs = self._observation()
            r = self._clip(-self.step_cost)
            truncated = self._steps >= self.max_episode_steps
            info: Dict[str, Any] = {"invalid": True}
            if truncated:
                r = self._clip(r + self.timeout_penalty)
                info["timeout"] = True
                info["loop_penalty_events"] = self._episode_loop_events
                info["revisit_events"] = self._episode_revisit_events
                info["loop_count"] = self._episode_loop_events
                info["revisit_count"] = self._episode_revisit_events
            return obs, r, False, truncated, info

        dr, dc = DR_DC[a]
        nr, nc = self._agent[0] + dr, self._agent[1] + dc
        hit_wall = (
            nr < 0
            or nc < 0
            or nr >= self.height
            or nc >= self.width
            or self._walls[nr, nc]
        )
        if hit_wall:
            reward = self._clip(-self.step_cost - self.wall_penalty)
            truncated = self._steps >= self.max_episode_steps
            info: Dict[str, Any] = {"collision": True, "success": False}
            if truncated:
                reward = self._clip(reward + self.timeout_penalty)
                info["timeout"] = True
                info["loop_penalty_events"] = self._episode_loop_events
                info["revisit_events"] = self._episode_revisit_events
                info["loop_count"] = self._episode_loop_events
                info["revisit_count"] = self._episode_revisit_events
            return self._observation(), reward, False, truncated, info

        old_dist = self._manhattan(self._agent, self._goal)
        self._agent = (nr, nc)
        self._trail.append(self._agent)
        self._recent_positions.append(self._agent)

        new_dist = self._manhattan(self._agent, self._goal)
        reward = -self.step_cost
        reward += self.manhattan_scale * float(old_dist - new_dist)

        was_revisit = self._agent in self._visited
        if was_revisit:
            reward += self.revisit_penalty
            self._episode_revisit_events += 1
        else:
            reward += self.first_visit_bonus
            self._visited.add(self._agent)

        loop_hit = False
        if self._detect_two_cell_oscillation():
            reward += self.loop_penalty
            self._episode_loop_events += 1
            loop_hit = True

        terminated = self._agent == self._goal
        truncated = self._steps >= self.max_episode_steps

        if terminated:
            reward += self.goal_bonus

        if truncated and not terminated:
            reward += self.timeout_penalty

        reward = self._clip(reward)

        info: Dict[str, Any] = {
            "success": bool(terminated),
            "collision": False,
            "distance_to_goal": new_dist,
            "steps": self._steps,
            "loop_step": loop_hit,
            "revisit_step": was_revisit,
        }
        if terminated or truncated:
            info["loop_penalty_events"] = self._episode_loop_events
            info["revisit_events"] = self._episode_revisit_events
            info["loop_count"] = self._episode_loop_events
            info["revisit_count"] = self._episode_revisit_events

        if self.reward_debug and (terminated or truncated):
            print(
                f"[GridNav] steps={self._steps} term={terminated} trunc={truncated} "
                f"loops={self._episode_loop_events} revisits={self._episode_revisit_events} "
                f"last_r={reward:.2f}"
            )

        return self._observation(), reward, terminated, truncated, info

    def _cell_rgb(self, r: int, c: int) -> Tuple[int, int, int]:
        assert self._walls is not None
        if self._walls[r, c]:
            return (45, 45, 55)
        if (r, c) == self._goal:
            return (220, 70, 70)
        if (r, c) in self._parking_set:
            return (70, 160, 90)
        return (235, 235, 238)

    def render(self):
        if self.render_mode is None:
            return None
        arr = self._render_rgb_array()
        if self.render_mode == "rgb_array":
            return arr
        import matplotlib.pyplot as plt

        if self._fig is None:
            self._fig, self._ax = plt.subplots(figsize=(6, 6))
        self._ax.clear()
        self._ax.imshow(arr, origin="upper")
        self._ax.set_axis_off()
        self._ax.set_title("Grid navigasyon")
        plt.pause(0.001)
        self._fig.canvas.draw_idle()
        return None

    def _render_rgb_array(self) -> np.ndarray:
        cs = self.cell_pixels
        H, W = self.height, self.width
        assert self._walls is not None
        img = Image.new("RGB", (W * cs, H * cs), (255, 255, 255))
        drw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("arial.ttf", size=max(10, cs // 2))
        except OSError:
            font = ImageFont.load_default()

        for r in range(H):
            for c in range(W):
                x0, y0 = c * cs, r * cs
                drw.rectangle(
                    [x0, y0, x0 + cs - 1, y0 + cs - 1],
                    fill=self._cell_rgb(r, c),
                    outline=(180, 180, 180),
                )

        if len(self._trail) >= 2:
            pts = [(c * cs + cs // 2, r * cs + cs // 2) for r, c in self._trail]
            drw.line(pts, fill=(255, 200, 50), width=max(2, cs // 8))

        for r, c in self._parking_set:
            if (r, c) == self._goal:
                continue
            x0, y0 = c * cs, r * cs
            drw.text((x0 + cs // 4, y0 + cs // 5), "P", fill=(20, 80, 20), font=font)

        gr, gc = self._goal
        gx, gy = gc * cs, gr * cs
        drw.text((gx + cs // 6, gy + cs // 5), "G", fill=(255, 255, 255), font=font)

        ar, ac = self._agent
        ax_, ay_ = ac * cs, ar * cs
        if self._celebrate_parking:
            margin = max(1, cs // 10)
            drw.rounded_rectangle(
                [ax_ + margin, ay_ + margin, ax_ + cs - 1 - margin, ay_ + cs - 1 - margin],
                radius=max(2, cs // 6),
                fill=(72, 160, 86),
                outline=(18, 95, 32),
                width=max(2, cs // 10),
            )
            self._draw_park_checkmark(drw, ax_, ay_, cs)
        else:
            drw.text((ax_ + cs // 6, ay_ + cs // 5), "\U0001F697", fill=(20, 20, 120), font=font)

        if self._celebrate_parking:
            banner_h = max(14, cs)
            drw.rectangle([0, H * cs - banner_h, W * cs, H * cs], fill=(40, 100, 50))
            try:
                banner_font = ImageFont.truetype("arial.ttf", size=max(11, cs // 2))
            except OSError:
                banner_font = font
            drw.text(
                (cs // 2, H * cs - banner_h + 2),
                "Park edildi",
                fill=(255, 255, 255),
                font=banner_font,
            )

        return np.asarray(img)

    def close(self):
        if self._fig is not None:
            import matplotlib.pyplot as plt

            plt.close(self._fig)
            self._fig = None
            self._ax = None


def save_episode_gif_ppo(
    episode_configs: List[GridNavEpisodeConfig],
    model_base_path: Path,
    vec_pkl_path: Path,
    out_path: Path,
    fps: int = 8,
    seed: int = 42,
    max_steps: int = 300,
    park_celebration_frames: int = 12,
) -> Path:
    """VecNormalize + PPO ile tek bölüm oynatıp GIF yazar.

    Başarılı hedef varışında `park_celebration_frames` kadar ek kare: yeşil hücre,
    onay işareti ve altta \"Park edildi\" bandı.
    """
    import imageio.v2 as imageio
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def factory():
        return GridParkingNavigationEnv(
            episode_configs,
            seed=seed,
            render_mode="rgb_array",
            max_episode_steps=max_steps,
        )

    venv = DummyVecEnv([factory])
    vec = VecNormalize.load(str(vec_pkl_path), venv)
    vec.training = False
    vec.norm_reward = False
    model = PPO.load(
        str(model_base_path),
        env=vec,
        custom_objects={"lr_schedule": lambda _: 3e-4},
    )

    def _unwrap() -> GridParkingNavigationEnv:
        w = vec.venv.envs[0]
        u = w.unwrapped
        if not isinstance(u, GridParkingNavigationEnv):
            u = getattr(u, "unwrapped", u)
        assert isinstance(u, GridParkingNavigationEnv)
        return u

    frames: List[np.ndarray] = []
    vec.seed(int(seed))
    obs = vec.reset()
    env0 = _unwrap()
    f0 = env0.render()
    if f0 is not None:
        frames.append(np.array(f0))
    done = False
    last_info: Dict[str, Any] = {}
    while not done:
        act, _ = model.predict(obs, deterministic=True)
        obs, _r, dones, infos = vec.step(act)
        done = bool(dones[0])
        if isinstance(infos, (list, tuple)) and infos and infos[0]:
            last_info = infos[0]
        env0 = _unwrap()
        fr = env0.render()
        if fr is not None:
            frames.append(np.array(fr))

    env_check = _unwrap()
    reached_goal = bool(last_info.get("success")) or (
        done and env_check._agent == env_check._goal
    )
    if reached_goal and park_celebration_frames > 0:
        env0 = env_check
        env0._celebrate_parking = True
        try:
            for _ in range(int(park_celebration_frames)):
                fr = env0.render()
                if fr is not None:
                    frames.append(np.array(fr))
        finally:
            env0._celebrate_parking = False

    if not frames:
        raise RuntimeError("GIF için kare üretilemedi.")

    imageio.mimsave(str(out_path), frames, fps=fps)
    env0.close()
    return out_path
