"""
Çok adımlı otopark grid navigasyon ortamı (Gymnasium).

Proje künyesi (hibrit RL durumu): araç konumu, otopark doluluk haritası (ızgara üzerinde
oranlar), hedef ve duvarlar; ayrıca skaler olarak kısa vadeli LSTM tahmini (aggregate)
ve hedef otopark için tahmini boşalma / yoğunluk vekili (0–1).

Eylem: 0=up, 1=down, 2=left, 3=right
Gözlem: [duvar | park alanı maskesi | hedef | ajan | doluluk ısı haritası] düzleştirilmiş
(5 * H * W) + [lstm_aggregate_pred, predicted_emptying_norm] (2 skaler)
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

import math
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


# Eylem kodları: yukarı / aşağı / sol / sağ (satır, sütun) delta sırası DR_DC ile eşleşir
ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3
DR_DC = [(-1, 0), (1, 0), (0, -1), (0, 1)]


@dataclass
class GridNavEpisodeConfig:
    """
    occ_ratios: park_cells ile aynı sırada 0–1 doluluk (Birmingham anlık).
    lstm_aggregate_pred: LSTM aggregate tahmin; NaN ise reset’te lot ortalaması kullanılır.
    predicted_emptying_norm: hedef lot doluluğu 0–1 (boşalma süresi vekili).
    """

    walls: np.ndarray
    parking_cells: List[Tuple[int, int]]
    goal_cell: Tuple[int, int]
    agent_start: Tuple[int, int]
    height: int
    width: int
    occ_ratios: np.ndarray
    lstm_aggregate_pred: float = float("nan")
    predicted_emptying_norm: float = 0.0


def _place_lots_on_grid(
    lot_ids: List[str],
    height: int,
    width: int,
) -> Dict[str, Tuple[int, int]]:
    """Gerçek GPS yoksa hash tabanlı stabil (lat,lon) → ızgara hücresi eşlemesi; çakışmalarda kaydırma."""
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
    """
    Parquet zaman dilimlerinden duvarlı ızgara, park hücreleri, hedef ve başlangıç konumu üretir.

    predictions_csv (opsiyonel): `LastUpdated`, `y_pred_occupancy_rate` sütunları — LSTM
    çıktısı RL durumundaki lstm_aggregate_pred skalerine bağlanır (künye: tahmin → RL state).
    """
    path = Path(processed_parquet)
    df_full = pd.read_parquet(path)
    lot_ids = sorted(df_full["SystemCodeNumber"].astype(str).unique())
    cap_median = df_full.groupby("SystemCodeNumber")["Capacity"].median().reindex(lot_ids)

    pred_by_ts: dict[pd.Timestamp, float] = {}
    if predictions_csv is not None:
        pc = Path(predictions_csv)
        if pc.exists():
            pr = pd.read_csv(pc)
            pr["LastUpdated"] = pd.to_datetime(pr["LastUpdated"], errors="coerce")
            pr = pr.dropna(subset=["LastUpdated"])
            pred_by_ts = {
                pd.Timestamp(ts): float(v)
                for ts, v in pr.groupby("LastUpdated")["y_pred_occupancy_rate"].first().items()
            }

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

        ts_key = pd.Timestamp(_ts)
        raw_lstm = pred_by_ts.get(ts_key, float("nan"))
        if not math.isnan(raw_lstm):
            raw_lstm = float(np.clip(raw_lstm, 0.0, 1.0))
        predicted_emptying_norm = float(np.clip(float(occ_ratio[best_i]), 0.0, 1.0))
        occ_ratios_f = occ_ratio.astype(np.float32, copy=False)

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
                occ_ratios=occ_ratios_f,
                lstm_aggregate_pred=float(raw_lstm),
                predicted_emptying_norm=predicted_emptying_norm,
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
        self._extra_dim = 2
        flat_dim = 5 * self.height * self.width + self._extra_dim
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
        self._occ_heatmap = np.zeros((self.height, self.width), dtype=np.float32)
        self._lstm_agg_scalar = 0.0
        self._pred_emptying = 0.0
        self._routing_time_cost = 0.0
        self._routing_wall_cost = 0.0
        self._occ_path_accum = 0.0
        self._grid_moves = 0

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
        occ_map = np.clip(self._occ_heatmap, 0.0, 1.0)
        aux = np.array(
            [float(self._lstm_agg_scalar), float(self._pred_emptying)],
            dtype=np.float32,
        )
        return np.concatenate(
            [
                wall_f.ravel(),
                park.ravel(),
                goal.ravel(),
                agent.ravel(),
                occ_map.ravel(),
                aux,
            ]
        ).astype(np.float32)

    def _merge_episode_cost_metrics(
        self, info: Dict[str, Any], terminated: bool, truncated: bool
    ) -> Dict[str, Any]:
        """Künye 7.3 RL: yönlendirme maliyeti ve rota üzeri yoğunluk vekilleri."""
        if not (terminated or truncated):
            return info
        out = dict(info)
        gm = max(1, int(self._grid_moves))
        out["routing_cost_proxy"] = float(self._routing_time_cost + self._routing_wall_cost)
        out["mean_visit_congestion"] = float(self._occ_path_accum / gm)
        return out

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

        self._occ_heatmap = np.zeros((self.height, self.width), dtype=np.float32)
        for (r, c), ov in zip(cfg.parking_cells, cfg.occ_ratios):
            self._occ_heatmap[int(r), int(c)] = float(np.clip(float(ov), 0.0, 1.0))
        mean_occ = float(np.clip(float(np.mean(cfg.occ_ratios)), 0.0, 1.0))
        if math.isnan(cfg.lstm_aggregate_pred):
            self._lstm_agg_scalar = mean_occ
        else:
            self._lstm_agg_scalar = float(np.clip(cfg.lstm_aggregate_pred, 0.0, 1.0))
        self._pred_emptying = float(np.clip(cfg.predicted_emptying_norm, 0.0, 1.0))

        self._routing_time_cost = 0.0
        self._routing_wall_cost = 0.0
        self._occ_path_accum = 0.0
        self._grid_moves = 0

        return self._observation(), {}

    def step(self, action: int):
        self._steps += 1
        self._routing_time_cost += float(self.step_cost)
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
            info = self._merge_episode_cost_metrics(info, False, truncated)
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
            self._routing_wall_cost += float(self.wall_penalty)
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
            info = self._merge_episode_cost_metrics(info, False, truncated)
            return self._observation(), reward, False, truncated, info

        old_dist = self._manhattan(self._agent, self._goal)
        self._agent = (nr, nc)
        self._grid_moves += 1
        self._occ_path_accum += float(self._occ_heatmap[self._agent[0], self._agent[1]])
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

        info = self._merge_episode_cost_metrics(info, terminated, truncated)

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
        if (r, c) in self._parking_set:
            # Künye arayüz: yeşil (düşük doluluk), sarı (orta), kırmızı (yüksek).
            # Hedef (G) de bir lot hücresi olduğu için aynı renk skalası kullanılır — böylece
            # sunumda “en boş lot = yeşil” ile çelişen sabit kırmızı hedef rengi oluşmaz.
            o = float(self._occ_heatmap[r, c])
            if o < 0.33:
                return (70, 160, 90)
            if o < 0.66:
                return (220, 200, 80)
            return (190, 60, 60)
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

        # Rota: emoji/font bağımsız, kalın çizgi (sarı park hücrelerinden ayrışır)
        if len(self._trail) >= 2:
            pts = [(c * cs + cs // 2, r * cs + cs // 2) for r, c in self._trail]
            lw = max(6, cs // 3)
            drw.line(pts, fill=(25, 25, 90), width=lw + 4)
            drw.line(pts, fill=(0, 210, 255), width=lw)
            pr = max(3, cs // 7)
            for r, c in self._trail:
                cx, cy = c * cs + cs // 2, r * cs + cs // 2
                drw.ellipse(
                    [cx - pr, cy - pr, cx + pr, cy + pr],
                    fill=(0, 170, 220),
                    outline=(0, 90, 140),
                    width=1,
                )

        for r, c in self._parking_set:
            if (r, c) == self._goal:
                continue
            x0, y0 = c * cs, r * cs
            drw.text((x0 + cs // 4, y0 + cs // 5), "P", fill=(20, 80, 20), font=font)

        gr, gc = self._goal
        gx, gy = gc * cs, gr * cs
        margin = max(2, cs // 14)
        drw.rounded_rectangle(
            [gx + margin, gy + margin, gx + cs - 1 - margin, gy + cs - 1 - margin],
            radius=max(3, cs // 8),
            outline=(255, 210, 40),
            width=max(3, cs // 7),
        )
        # G: arka plan yeşil/sarı/kırmızı olabileceği için okunaklı kontrast
        o_goal = float(self._occ_heatmap[gr, gc])
        g_fill = (15, 25, 90) if o_goal < 0.5 else (255, 255, 255)
        drw.text((gx + cs // 6, gy + cs // 5), "G", fill=g_fill, font=font)

        ar, ac = self._agent
        ax_, ay_ = ac * cs, ar * cs
        # Windows’ta Arial ile 🚗 çoğu zaman boş — her zaman görünür ajan çizimi
        margin = max(2, cs // 8)
        drw.rounded_rectangle(
            [ax_ + margin, ay_ + margin, ax_ + cs - 1 - margin, ay_ + cs - 1 - margin],
            radius=max(3, cs // 7),
            fill=(55, 125, 255),
            outline=(20, 55, 160),
            width=max(2, cs // 12),
        )
        drw.text(
            (ax_ + max(2, cs // 5), ay_ + cs // 6),
            "A",
            fill=(255, 255, 255),
            font=font,
        )

        return np.asarray(img)

    def close(self):
        if self._fig is not None:
            import matplotlib.pyplot as plt

            plt.close(self._fig)
            self._fig = None
            self._ax = None


def _unwrap_vec_to_grid_nav(vec: Any) -> GridParkingNavigationEnv:
    w = vec.venv.envs[0]
    u = w.unwrapped
    if not isinstance(u, GridParkingNavigationEnv):
        u = getattr(u, "unwrapped", u)
    assert isinstance(u, GridParkingNavigationEnv)
    return u


def rollout_ppo_gif_frames(
    vec: Any,
    model: Any,
    rollout_seed: int,
    max_steps: int,
) -> Tuple[List[np.ndarray], bool, int, Dict[str, Any]]:
    """Tek PPO bölümü: RGB kareleri, success, adım sayısı, son info (revisit/loop vb.)."""
    vec.seed(int(rollout_seed))
    obs = vec.reset()
    env0 = _unwrap_vec_to_grid_nav(vec)
    frames: List[np.ndarray] = []
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
        env0 = _unwrap_vec_to_grid_nav(vec)
        fr = env0.render()
        if fr is not None:
            frames.append(np.array(fr))

    env_check = _unwrap_vec_to_grid_nav(vec)
    reached_goal = bool(last_info.get("success")) or (
        done and env_check._agent == env_check._goal
    )

    if not frames:
        raise RuntimeError("GIF için kare üretilemedi.")
    n_steps = int(last_info.get("steps", 0))
    if n_steps <= 0 and reached_goal:
        n_steps = max(0, len(frames) - 1)
    return frames, reached_goal, n_steps, last_info


def save_episode_gif_ppo(
    episode_configs: List[GridNavEpisodeConfig],
    model_base_path: Path,
    vec_pkl_path: Path,
    out_path: Path,
    seconds_per_frame: float = 6.0,
    seed: int = 42,
    max_steps: int = GRID_MAX_EPISODE_STEPS,
    *,
    pick_shortest_success: bool = False,
    max_candidate_episodes: int = 80,
) -> Path:
    """VecNormalize + PPO ile bölüm oynatıp GIF yazar.

    Tez / sunum (Seçenek A): ``pick_shortest_success=True`` iken ``seed, seed+1, ...``
    tohumlarıyla en fazla ``max_candidate_episodes`` bölüm dener; yalnızca
    ``success`` olanlar arasından **en az adımda** hedefe varanı GIF olarak kaydeder.
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

    try:
        if not pick_shortest_success:
            frames, _, _, _ = rollout_ppo_gif_frames(vec, model, int(seed), max_steps)
        else:
            best: Optional[Tuple[int, List[np.ndarray]]] = None
            for k in range(max(1, int(max_candidate_episodes))):
                frames_k, ok, n_steps, _ = rollout_ppo_gif_frames(
                    vec, model, int(seed) + k, max_steps
                )
                if ok and (best is None or n_steps < best[0]):
                    best = (n_steps, frames_k)
            if best is None:
                raise RuntimeError(
                    f"Sunum modu: {max_candidate_episodes} denemede başarılı bölüm yok "
                    f"(success==True). GIF üretilemedi — evaluate.py ile başarı oranını "
                    f"kontrol edin veya max_candidate_episodes artırın."
                )
            frames = best[1]
        # GIF: kare başına süre (saniye). Çok düşük fps’ten daha kontrollü; inceleme için varsayılan yavaş.
        spf = float(max(seconds_per_frame, 0.05))
        imageio.mimsave(str(out_path), frames, duration=spf)
    finally:
        vec.close()
    return out_path
