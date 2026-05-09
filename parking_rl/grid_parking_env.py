"""
Grid tabanlı akıllı park ortamı — PPO için train / demo modları.

Hücre tipleri (saklanan grid; hedef ayrı tutulur, alt hücre her zaman boş park=1):
  0: bina/engel (gri, geçilemez)
  1: boş park (yeşil)
  2: dolu park (kırmızı, geçilemez)
Gözlemde hedef hücresi 3 (sarı) olarak kodlanır; fiziksel olarak hedef yine 1'dir.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

Mode = Literal["train", "demo"]

# Grid kodları (gözlem / çizim ile uyumlu)
CELL_BUILDING = 0
CELL_EMPTY = 1
CELL_OCCUPIED = 2
CELL_GOAL = 3

_REWARD_GOAL = 100.0
_REWARD_STEP = -1.0
INVALID_ACTION_PENALTY = -10.0


def detect_abab_position_oscillation(
    trail: Sequence[Tuple[int, int]],
    *,
    window: int = 4,
) -> bool:
    """
    Son dört konumda iki farklı hücre arasında A→B→A→B ping-pong var mı
    (state repetition / 2-periyot osilasyon). `window` şimdilik 4 olmalı.
    """
    if window != 4 or len(trail) < 4:
        return False
    a, b, c, d = trail[-4], trail[-3], trail[-2], trail[-1]
    return bool(a == c and b == d and a != b)


def detect_immediate_return_pingpong(trail: Sequence[Tuple[int, int]]) -> bool:
    """
    Son üç konum X→Y→X (X≠Y): bir adımda gidilip hemen önceki hücreye dönüş
    (sağ–sol / yukarı–aşağı tek adım geri alma). ABAB’den önce tetiklenir.
    """
    if len(trail) < 3:
        return False
    a, b, c = trail[-3], trail[-2], trail[-1]
    return bool(a == c and a != b)


def detect_position_pingpong(trail: Sequence[Tuple[int, int]]) -> Tuple[bool, str]:
    """ABAB veya tek adım geri dönüş; (True, sebep) veya (False, '')."""
    if detect_abab_position_oscillation(trail):
        return True, "abab"
    if detect_immediate_return_pingpong(trail):
        return True, "undo"
    return False, ""


# rl_model / evaluate_performance ile tek kaynak (MDP tutarlılığı)
TRAIN_PPO_ENV_KWARGS: Dict[str, Any] = {
    "size": 10,
    "mode": "train",
    "max_episode_steps": 200,
    "debug_checks": False,
    "oscillation_penalty": -2.5,
    "revisit_penalty": -0.65,
    "same_cell_stuck_penalty": -0.45,
    "distance_shaping_coef": 2.0,
}


class GridParkingEnv(gym.Env):
    """
    Aksiyonlar: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT (origin='lower' ile uyumlu).
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        *,
        size: int = 10,
        mode: Mode = "train",
        max_episode_steps: int = 200,
        building_ratio: float = 0.18,
        occupied_ratio: float = 0.12,
        # Demo: her adımda yeşil→kırmızı / kırmızı→yeşil olasılığı (hücre başına)
        demo_flip_prob_green: float = 0.04,
        demo_flip_prob_red: float = 0.03,
        debug_checks: bool = True,
        step_debug_log: bool = False,
        # Ödül: Manhattan shaping katsayısı; >1 hedefe yaklaşmayı güçlendirir.
        distance_shaping_coef: float = 1.0,
        # Son dört konum A,B,A,B ise (iki hücre arasında ping-pong) bu kadar ek ödül (genelde negatif).
        oscillation_penalty: float = 0.0,
        # Geçerli adımla aynı hücreye 2.+ girişte uygulanır (loop / geri iz sürer).
        revisit_penalty: float = 0.0,
        # Geçersiz adım sonrası trail’de aynı konum iki kez üst üste (duvara tekrar çarpma).
        same_cell_stuck_penalty: float = 0.0,
    ) -> None:
        super().__init__()
        if mode not in ("train", "demo"):
            raise ValueError('mode "train" veya "demo" olmalı')
        self.size = int(size)
        self.mode: Mode = mode
        self.max_episode_steps = int(max_episode_steps)
        self.building_ratio = float(building_ratio)
        self.occupied_ratio = float(occupied_ratio)
        self.demo_flip_prob_green = float(demo_flip_prob_green)
        self.demo_flip_prob_red = float(demo_flip_prob_red)
        self.debug_checks = bool(debug_checks)
        self.step_debug_log = bool(step_debug_log)
        self.distance_shaping_coef = float(distance_shaping_coef)
        self.oscillation_penalty = float(oscillation_penalty)
        self.revisit_penalty = float(revisit_penalty)
        self.same_cell_stuck_penalty = float(same_cell_stuck_penalty)
        self._position_trail: List[Tuple[int, int]] = []
        self._cell_visit_counts: Dict[Tuple[int, int], int] = {}
        self._episode_oscillation_total = 0
        self._episode_same_cell_stuck = 0
        self._episode_action_counts = [0, 0, 0, 0]

        # Son 4 boyut: aksiyon maskesi (1 geçerli, 0 duvar/sınır/dolu)
        obs_dim = self.size * self.size + 4 + 4
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(4)

        self.grid = np.zeros((self.size, self.size), dtype=np.int32)
        self.agent: Tuple[int, int] = (0, 0)
        self.goal: Tuple[int, int] = (0, 0)
        self._step_count = 0
        self._episode_train_goal: Tuple[int, int] = (0, 0)
        self._last_goal_change_step: Optional[int] = None
        self._goal_flash_remaining = 0

    def _neighbors(self, r: int, c: int) -> List[Tuple[int, int]]:
        out: List[Tuple[int, int]] = []
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.size and 0 <= nc < self.size:
                out.append((nr, nc))
        return out

    def _reachable_empty(
        self, grid: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]
    ) -> bool:
        gr, gc = goal
        if grid[start[0], start[1]] != CELL_EMPTY:
            return False
        if grid[gr, gc] != CELL_EMPTY:
            return False
        seen = {start}
        stack = [start]
        while stack:
            r, c = stack.pop()
            if (r, c) == goal:
                return True
            for nr, nc in self._neighbors(r, c):
                if (nr, nc) in seen:
                    continue
                if int(grid[nr, nc]) != CELL_EMPTY:
                    continue
                seen.add((nr, nc))
                stack.append((nr, nc))
        return False

    def _build_static_grid(self) -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, int]]:
        """Train/demo başlangıç layout: bina (0), boş (1), dolu (2)."""
        n_cells = self.size * self.size
        n_build = int(round(self.building_ratio * n_cells))
        n_build = max(0, min(n_build, n_cells - 2))

        for _ in range(500):
            g = np.full((self.size, self.size), CELL_EMPTY, dtype=np.int32)
            inner = [(r, c) for r in range(self.size) for c in range(self.size)]
            self.np_random.shuffle(inner)
            for r, c in inner[:n_build]:
                g[r, c] = CELL_BUILDING

            parking = [
                (r, c)
                for r in range(self.size)
                for c in range(self.size)
                if int(g[r, c]) == CELL_EMPTY
            ]
            self.np_random.shuffle(parking)
            n_occ = int(round(len(parking) * self.occupied_ratio))
            n_occ = max(0, min(n_occ, max(0, len(parking) - 2)))
            for r, c in parking[:n_occ]:
                g[r, c] = CELL_OCCUPIED

            greens = [(r, c) for r, c in parking[n_occ:] if int(g[r, c]) == CELL_EMPTY]
            if len(greens) < 2:
                continue

            self.np_random.shuffle(greens)
            start = greens[0]
            goal = greens[1]
            if self._reachable_empty(g, start, goal):
                return g, start, goal

        g = np.full((self.size, self.size), CELL_EMPTY, dtype=np.int32)
        return g, (0, 0), (self.size - 1, self.size - 1)

    def _grid_for_obs(self) -> np.ndarray:
        """Hedef hücresi 3 olarak gösterilen kopya (alt değer her zaman 1 olmalı)."""
        disp = self.grid.copy()
        gr, gc = self.goal
        if self.debug_checks and int(disp[gr, gc]) != CELL_EMPTY:
            raise RuntimeError(
                f"[GridParkingEnv] Hata: hedef hücresi yeşil (1) değil: değer={disp[gr, gc]} konum=({gr},{gc})"
            )
        disp[gr, gc] = CELL_GOAL
        return disp

    def _get_obs(self) -> np.ndarray:
        disp = self._grid_for_obs()
        ar, ac = self.agent
        gr, gc = self.goal
        s = float(self.size)
        mask = self.get_valid_actions(None)
        vec = np.concatenate(
            [
                (disp.astype(np.float32).flatten() / 3.0),
                np.array([ar, ac, gr, gc], dtype=np.float32) / s,
                mask.astype(np.float32),
            ]
        )
        return np.clip(vec, 0.0, 1.0)

    def _assert_goal_green(self, where: str) -> None:
        if not self.debug_checks:
            return
        gr, gc = self.goal
        v = int(self.grid[gr, gc])
        if v != CELL_EMPTY:
            raise RuntimeError(f"[GridParkingEnv] {where}: hedef yeşil değil (değer={v})")

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)

        self.grid, self.agent, self.goal = self._build_static_grid()
        self._step_count = 0
        self._episode_train_goal = tuple(self.goal)
        self._last_goal_change_step = None
        self._goal_flash_remaining = 0

        self._assert_goal_green("reset sonrası")
        self._sync_goal_cell_empty()
        self._position_trail = [tuple(self.agent)]
        self._cell_visit_counts = {tuple(self.agent): 1}
        self._episode_oscillation_total = 0
        self._episode_same_cell_stuck = 0
        self._episode_action_counts = [0, 0, 0, 0]

        obs = self._get_obs()
        vm = self.get_valid_actions(None)
        info: Dict[str, Any] = {
            "mode": self.mode,
            "goal": tuple(self.goal),
            "agent": tuple(self.agent),
            "valid_actions": vm.tolist(),
            "action_mask": vm.copy(),
        }
        return obs, info

    def _passable(self, r: int, c: int) -> bool:
        return (
            0 <= r < self.size
            and 0 <= c < self.size
            and int(self.grid[r, c]) == CELL_EMPTY
        )

    @staticmethod
    def _manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
        return abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1]))

    def _passable_from_disp(self, disp: np.ndarray, r: int, c: int) -> bool:
        """Gözlemdeki 0–3 kodlarıyla geçilebilirlik (hedef hücresi gösterimi 3)."""
        n = self.size
        if not (0 <= r < n and 0 <= c < n):
            return False
        v = int(disp[r, c])
        return v == CELL_EMPTY or v == CELL_GOAL

    def get_valid_actions(self, state: Optional[np.ndarray] = None) -> np.ndarray:
        """
        4 aksiyon maskesi: 1.0 geçerli, 0.0 duvar (0), sınır dışı veya dolu (2).
        state=None iken mevcut self.agent / self.grid; aksi halde gözlem vektöründen konum+ızgara çözülür.
        """
        n = self.size
        dr_dc = [(1, 0), (-1, 0), (0, -1), (0, 1)]
        out = np.zeros(4, dtype=np.float32)
        if state is None:
            r, c = self.agent
            for i, (dr, dc) in enumerate(dr_dc):
                nr, nc = r + dr, c + dc
                out[i] = 1.0 if self._passable(nr, nc) else 0.0
            return out

        vec = np.asarray(state, dtype=np.float32).reshape(-1)
        flat_len = n * n
        need = flat_len + 4
        if vec.shape[0] < need:
            raise ValueError(f"state boyutu yetersiz: beklenen>={need}, gelen={vec.shape[0]}")
        s = float(n)
        r = int(np.clip(round(float(vec[flat_len]) * s), 0, n - 1))
        c = int(np.clip(round(float(vec[flat_len + 1]) * s), 0, n - 1))
        disp = np.clip(np.round(vec[:flat_len].astype(np.float64) * 3.0), 0, 3).astype(np.int32).reshape(
            n, n
        )
        for i, (dr, dc) in enumerate(dr_dc):
            nr, nc = r + dr, c + dc
            out[i] = 1.0 if self._passable_from_disp(disp, nr, nc) else 0.0
        return out

    def _legal_action_indices(self) -> List[int]:
        """Geçerli yön indeksleri (0=UP,1=DOWN,2=LEFT,3=RIGHT)."""
        m = self.get_valid_actions(None)
        return [i for i in range(4) if float(m[i]) >= 0.5]

    def _sync_goal_cell_empty(self) -> None:
        """Hedef hücresi tanım gereği boş park (1) olmalı; demo/flip sonrası tutarlılık onarımı."""
        gr, gc = self.goal
        v = int(self.grid[gr, gc])
        if v == CELL_BUILDING:
            raise RuntimeError(f"Hedef bina (0) üzerinde: ({gr},{gc})")
        if v != CELL_EMPTY:
            self.grid[gr, gc] = CELL_EMPTY

    def _apply_demo_dynamics(self) -> bool:
        """Yeşil↔kırmızı dinamikleri. Hedef kırmızı olursa yeni hedef seçilir. Dönüş: hedef değişti mi."""
        assert self.mode == "demo"
        g = self.grid
        ar, ac = self.agent
        gr, gc = self.goal

        # Yeşil → kırmızı (ajanın üzerindeki hücreyi koru; hedef yeşili de doluya dönebilir)
        greens = [
            (r, c)
            for r in range(self.size)
            for c in range(self.size)
            if int(g[r, c]) == CELL_EMPTY and (r, c) != (ar, ac)
        ]
        for r, c in greens:
            if self.np_random.random() < self.demo_flip_prob_green:
                g[r, c] = CELL_OCCUPIED

        reds = [
            (r, c)
            for r in range(self.size)
            for c in range(self.size)
            if int(g[r, c]) == CELL_OCCUPIED
        ]
        for r, c in reds:
            if self.np_random.random() < self.demo_flip_prob_red:
                g[r, c] = CELL_EMPTY

        goal_changed = False
        if int(g[gr, gc]) != CELL_EMPTY:
            print("Doluluk oranı nedeniyle hedef değiştirildi.")
            candidates = [
                (r, c)
                for r in range(self.size)
                for c in range(self.size)
                if int(g[r, c]) == CELL_EMPTY and self._reachable_empty(g, self.agent, (r, c))
            ]
            if not candidates:
                candidates = [
                    (r, c)
                    for r in range(self.size)
                    for c in range(self.size)
                    if int(g[r, c]) == CELL_EMPTY
                ]
            if candidates:
                idx = int(self.np_random.integers(0, len(candidates)))
                self.goal = candidates[idx]
                goal_changed = True
                self._last_goal_change_step = self._step_count
                self._goal_flash_remaining = 8
            else:
                raise RuntimeError("[GridParkingEnv] Demo: yeşil hücre kalmadı, hedef seçilemiyor.")

        self._assert_goal_green("demo dinamik sonrası")
        self._sync_goal_cell_empty()
        return goal_changed

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        if not self.action_space.contains(int(action)):
            raise ValueError(f"Geçersiz aksiyon: {action}")

        self._sync_goal_cell_empty()

        valid_before = self.get_valid_actions(None)
        goal_before = tuple(self.goal)
        r, c = self.agent
        prev_dist = self._manhattan((r, c), goal_before)

        dr_dc = [(1, 0), (-1, 0), (0, -1), (0, 1)]  # UP, DOWN, LEFT, RIGHT
        dr, dc = dr_dc[int(action) % 4]
        nr, nc = r + dr, c + dc
        invalid = not self._passable(nr, nc)
        if invalid:
            pass  # ajan yerinde kalır; ödül aşağıda
        else:
            self.agent = (nr, nc)

        at_goal = self.agent == self.goal
        goal_changed = False

        if not at_goal and self.mode == "demo":
            goal_changed = self._apply_demo_dynamics()
            at_goal = self.agent == self.goal
        elif self.mode == "train" and self.debug_checks:
            if tuple(self.goal) != self._episode_train_goal:
                raise RuntimeError(
                    "[GridParkingEnv][DEBUG] Train modunda hedef değişti; bu olmamalı."
                )

        cur_dist = self._manhattan(self.agent, tuple(self.goal))
        shaping_unscaled = float(prev_dist - cur_dist)
        shaping = self.distance_shaping_coef * shaping_unscaled

        if at_goal:
            base_reward = _REWARD_GOAL
        elif invalid:
            base_reward = float(INVALID_ACTION_PENALTY) + shaping
        else:
            base_reward = float(_REWARD_STEP) + shaping

        self._episode_action_counts[int(action) % 4] += 1

        oscillation_hit = False
        oscillation_kind = ""
        self._position_trail.append(tuple(self.agent))
        if len(self._position_trail) > 64:
            self._position_trail.pop(0)
        if self.oscillation_penalty != 0.0:
            hit, oscillation_kind = detect_position_pingpong(self._position_trail)
            if hit:
                oscillation_hit = True
                self._episode_oscillation_total += 1

        revisit_component = 0.0
        if not invalid:
            pos = tuple(self.agent)
            prev_n = int(self._cell_visit_counts.get(pos, 0))
            self._cell_visit_counts[pos] = prev_n + 1
            if (
                self.revisit_penalty != 0.0
                and prev_n >= 1
                and not at_goal
            ):
                revisit_component = float(self.revisit_penalty)

        stuck_same_cell = (
            not at_goal
            and len(self._position_trail) >= 2
            and self._position_trail[-1] == self._position_trail[-2]
        )
        stuck_component = 0.0
        if stuck_same_cell and self.same_cell_stuck_penalty != 0.0:
            stuck_component = float(self.same_cell_stuck_penalty)
            self._episode_same_cell_stuck += 1

        oscillation_component = float(self.oscillation_penalty) if oscillation_hit else 0.0
        reward = (
            float(base_reward)
            + oscillation_component
            + revisit_component
            + stuck_component
        )

        self._step_count += 1
        terminated = bool(at_goal)
        truncated = self._step_count >= self.max_episode_steps
        step_idx = self._step_count

        flash_active = self._goal_flash_remaining > 0
        if self._goal_flash_remaining > 0:
            self._goal_flash_remaining -= 1

        obs = self._get_obs()
        legal = self._legal_action_indices()
        info: Dict[str, Any] = {
            "mode": self.mode,
            "invalid_move": invalid,
            "is_valid_move": not invalid,
            "legal_actions": legal,
            "valid_actions": valid_before.tolist(),
            "action_mask": valid_before.copy(),
            "goal": tuple(self.goal),
            "agent": tuple(self.agent),
            "goal_flash": flash_active,
            "goal_reassigned_this_step": goal_changed,
            "last_goal_change_step": self._last_goal_change_step,
            "oscillation_penalty_applied": oscillation_hit,
            "oscillation_kind": oscillation_kind or None,
            "distance_shaping_unscaled": shaping_unscaled,
            "reward_base": float(base_reward),
            "reward_oscillation": oscillation_component,
            "reward_revisit": revisit_component,
            "reward_same_cell_stuck": stuck_component,
            "revisit_penalty_hit": bool(revisit_component != 0.0),
            "same_cell_stuck_hit": bool(stuck_component != 0.0),
            "episode_oscillation_total": int(self._episode_oscillation_total),
            "episode_same_cell_stuck": int(self._episode_same_cell_stuck),
            "episode_unique_cells": int(len(self._cell_visit_counts)),
            "episode_action_counts": list(self._episode_action_counts),
        }

        if terminated or truncated:
            names = ("UP", "DOWN", "LEFT", "RIGHT")
            dist = {
                names[i]: int(self._episode_action_counts[i]) for i in range(4)
            }
            info["episode_summary"] = {
                "total_oscillations": int(self._episode_oscillation_total),
                "same_cell_stuck_steps": int(self._episode_same_cell_stuck),
                "unique_cells_visited": int(len(self._cell_visit_counts)),
                "action_distribution": dist,
                "steps": int(self._step_count),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
            }

        if self.step_debug_log:
            names = ("UP", "DOWN", "LEFT", "RIGHT")
            tried = names[int(action) % 4]
            tail = self._position_trail[-8:] if len(self._position_trail) > 8 else self._position_trail
            print(
                f"[step {step_idx}] state(agent={tuple(self.agent)}, goal={tuple(self.goal)}) "
                f"action={int(action)}({tried}) valid={valid_before.tolist()} ok={not invalid} "
                f"reward={float(reward):.4f} (base={base_reward:.3f} osc={oscillation_component:.3f} "
                f"rev={revisit_component:.3f} stuck={stuck_component:.3f}) shaping_u={shaping_unscaled:.2f} "
                f"osc={oscillation_hit!s}:{oscillation_kind or '-'} stuck_hit={stuck_same_cell} "
                f"trail_tail={tail}"
            )

        return obs, float(reward), terminated, truncated, info

    def to_draw_state(
        self,
        *,
        step: int,
        action: Any,
        reward: float,
        prev_agent: Optional[Tuple[int, int]] = None,
        goal_flash: bool = False,
        banner: Optional[str] = None,
        invalid_move: bool = False,
    ) -> Dict[str, Any]:
        """Animasyon: imshow için sadece 0–2; hedef ve ajan ayrı katmanlarda çizilir."""
        ar, ac = self.agent
        dx, dy = 0.0, 0.0
        if prev_agent is not None:
            pr, pc = prev_agent
            dx = (ac + 0.5) - (pc + 0.5)
            dy = (ar + 0.5) - (pr + 0.5)
        return {
            "kind": "grid_parking",
            "grid_base": self.grid.copy(),
            "goal": tuple(self.goal),
            "agent": (ar, ac),
            "step": int(step),
            "action": action,
            "reward": float(reward),
            "arrow": (dx, dy) if prev_agent is not None else None,
            "goal_flash": bool(goal_flash),
            "banner": banner,
            "invalid_move": bool(invalid_move),
        }
