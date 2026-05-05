"""
External continuous parking environment adapter.

Bu sınıf, GitHub'daki parking-env yaklaşımına (steering + throttle) uyumlu
continuous action space sağlar ve proje içindeki env arayüzüyle aynı sözleşmeyi
korur: reset(), step(), observation_space, action_space.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class ExternalParkingEnv(gym.Env):
    """
    Continuous action wrapper:
      action[0] -> steering
      action[1] -> throttle / acceleration

    Öncelikli olarak parking-v0 (highway-env) ortamını kullanır.
    parking-v0 mevcut değilse, basit bir kinematic fallback senaryosuna düşer.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        *,
        max_episode_steps: int = 120,
    ) -> None:
        super().__init__()
        self.max_episode_steps = int(max_episode_steps)
        self._step_count = 0
        self._use_fallback = False

        self._base_env: gym.Env | None = None
        self._last_distance: float | None = None
        self._last_success = False
        self._last_collision = False
        self._episode_count = 0
        self._near_success_count = 0

        try:
            self._base_env = gym.make("parking-v0")
            base_action = self._base_env.action_space
            if not isinstance(base_action, spaces.Box):
                raise TypeError("parking-v0 action space continuous (Box) olmalı.")
            self.action_space = spaces.Box(
                low=np.asarray(base_action.low, dtype=np.float32),
                high=np.asarray(base_action.high, dtype=np.float32),
                dtype=np.float32,
            )
            # PPO için sadeleştirilmiş dense-feature state:
            # [distance, occupancy_proxy, target_dir_x, target_dir_y]
            self.observation_space = spaces.Box(
                low=np.array([0.0, 0.0, -1.0, -1.0], dtype=np.float32),
                high=np.array([10.0, 1.0, 1.0, 1.0], dtype=np.float32),
                dtype=np.float32,
            )
        except Exception:
            # Basit fallback: iç durum yine 6D tutulur, gözlem 4D'ye sadeleşir.
            self._use_fallback = True
            self._base_env = None
            self.action_space = spaces.Box(
                low=np.array([-1.0, -1.0], dtype=np.float32),
                high=np.array([1.0, 1.0], dtype=np.float32),
                dtype=np.float32,
            )
            self.observation_space = spaces.Box(
                low=np.array([0.0, 0.0, -1.0, -1.0], dtype=np.float32),
                high=np.array([3.0, 1.0, 1.0, 1.0], dtype=np.float32),
                dtype=np.float32,
            )
            self._fallback_state = np.zeros(6, dtype=np.float32)

    @staticmethod
    def _flatten_obs(obs: Any) -> np.ndarray:
        if isinstance(obs, dict):
            parts = []
            for key in ("observation", "achieved_goal", "desired_goal"):
                if key in obs:
                    parts.append(np.asarray(obs[key], dtype=np.float32).reshape(-1))
            if not parts:
                for v in obs.values():
                    parts.append(np.asarray(v, dtype=np.float32).reshape(-1))
            vec = np.concatenate(parts, dtype=np.float32)
        else:
            vec = np.asarray(obs, dtype=np.float32).reshape(-1)
        vec = np.nan_to_num(vec, nan=0.0, posinf=1.0, neginf=-1.0)
        return vec.astype(np.float32)

    def _compute_distance(self, obs_vec: np.ndarray) -> float:
        if obs_vec.size >= 12:
            # [achieved_goal(6), desired_goal(6)] benzeri formatlara tolerans
            # Son 6 ile önceki 6 arasındaki L2 mesafe.
            half = obs_vec.size // 2
            a = obs_vec[max(0, half - 6) : half]
            b = obs_vec[half : half + 6]
            if a.size == b.size and a.size > 0:
                return float(np.linalg.norm(a - b))
        if obs_vec.size >= 6:
            return float(np.linalg.norm(obs_vec[:2] - obs_vec[-2:]))
        return float(np.linalg.norm(obs_vec))

    @staticmethod
    def _safe_direction(src_xy: np.ndarray, dst_xy: np.ndarray) -> np.ndarray:
        delta = (dst_xy - src_xy).astype(np.float32)
        norm = float(np.linalg.norm(delta))
        if norm <= 1e-8:
            return np.zeros(2, dtype=np.float32)
        return np.clip(delta / norm, -1.0, 1.0).astype(np.float32)

    def _extract_occupancy_proxy(self, obs_raw: Any, obs_vec: np.ndarray) -> float:
        if isinstance(obs_raw, dict):
            for key in ("occupancy", "occupancy_ratio", "occupied_ratio"):
                if key in obs_raw:
                    arr = np.asarray(obs_raw[key], dtype=np.float32).reshape(-1)
                    if arr.size > 0:
                        return float(np.clip(np.mean(arr), 0.0, 1.0))
        # parking-v0 gözleminde doğrudan doluluk sinyali yoksa nötr proxy
        _ = obs_vec
        return 0.5

    def _build_dense_state(self, obs_raw: Any, obs_vec: np.ndarray) -> np.ndarray:
        distance = float(self._compute_distance(obs_vec))
        occupancy = float(self._extract_occupancy_proxy(obs_raw, obs_vec))

        direction = np.zeros(2, dtype=np.float32)
        if isinstance(obs_raw, dict):
            achieved = np.asarray(obs_raw.get("achieved_goal", []), dtype=np.float32).reshape(-1)
            desired = np.asarray(obs_raw.get("desired_goal", []), dtype=np.float32).reshape(-1)
            if achieved.size >= 2 and desired.size >= 2:
                direction = self._safe_direction(achieved[:2], desired[:2])
        elif obs_vec.size >= 6:
            direction = self._safe_direction(obs_vec[:2], obs_vec[-2:])

        state = np.array(
            [distance, occupancy, float(direction[0]), float(direction[1])],
            dtype=np.float32,
        )
        return np.nan_to_num(state, nan=0.0, posinf=10.0, neginf=-10.0).astype(np.float32)

    def _curriculum_stage(self) -> str:
        # İlk %30: yakın başlangıçlar, orta %40: orta mesafe, son %30: tam rastgele.
        span = 1000.0
        progress = min(1.0, float(self._episode_count) / span)
        if progress < 0.3:
            return "early"
        if progress < 0.7:
            return "mid"
        return "late"

    def _curriculum_distance_bounds(self) -> Tuple[float, float]:
        stage = self._curriculum_stage()
        if stage == "early":
            return (5.0, 8.0)
        if stage == "mid":
            return (8.0, 15.0)
        return (0.0, float("inf"))

    def _reward_components(
        self,
        *,
        distance: float,
        success: bool,
        soft_success: bool,
        collision: bool,
        previous_distance: float,
    ) -> Tuple[float, Dict[str, float]]:
        step_penalty = float(self._step_count)
        progress_reward = 0.2 * float(previous_distance - distance)
        success_reward = 100.0 if success else 0.0
        near_success_bonus = 20.0 if distance < 2.0 else 0.0
        progressive_success_shaping = 50.0 if distance < 1.0 else 0.0
        soft_success_bonus = 10.0 if soft_success and not success else 0.0
        collision_penalty = -50.0 if collision else 0.0
        distance_penalty = -0.05 * float(distance)
        step_penalty_term = -0.01 * step_penalty
        reward_raw = (
            success_reward
            + near_success_bonus
            + progressive_success_shaping
            + soft_success_bonus
            + collision_penalty
            + distance_penalty
            + progress_reward
            + step_penalty_term
        )
        reward = reward_raw / 100.0
        components = {
            "distance_penalty": float(distance_penalty),
            "progress_reward": float(progress_reward),
            "success_reward": float(success_reward),
            "near_success_bonus": float(near_success_bonus),
            "progressive_success_shaping": float(progressive_success_shaping),
            "soft_success_bonus": float(soft_success_bonus),
            "collision_penalty": float(collision_penalty),
            "step_penalty": float(step_penalty_term),
            "reward_raw": float(reward_raw),
            "reward_normalized": float(reward),
        }
        return reward, components

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self._step_count = 0
        self._last_success = False
        self._last_collision = False
        self._near_success_count = 0
        self._episode_count += 1

        if self._use_fallback:
            rng = self.np_random
            stage = self._curriculum_stage()
            x, y = rng.uniform(-0.75, 0.75), rng.uniform(-0.75, 0.75)
            angle = float(rng.uniform(-np.pi, np.pi))
            if stage == "early":
                radius = float(rng.uniform(0.45, 0.65))
            elif stage == "mid":
                radius = float(rng.uniform(0.65, 1.0))
            else:
                radius = float(rng.uniform(0.2, 1.2))
            tx = float(np.clip(x + radius * np.cos(angle), -0.75, 0.75))
            ty = float(np.clip(y + radius * np.sin(angle), -0.75, 0.75))
            self._fallback_state = np.array([x, y, 0.0, 0.0, tx, ty], dtype=np.float32)
            dense_obs = self._build_dense_state(self._fallback_state.copy(), self._fallback_state.copy())
            self._last_distance = float(dense_obs[0])
            return dense_obs, {
                "env_type": "external_fallback",
                "collision": False,
                "curriculum_stage": stage,
            }

        assert self._base_env is not None
        stage = self._curriculum_stage()
        min_dist, max_dist = self._curriculum_distance_bounds()
        dense_obs = None
        info: Dict[str, Any] = {}
        for _ in range(25):
            obs_raw, info = self._base_env.reset(seed=seed, options=options)
            obs = self._flatten_obs(obs_raw)
            candidate = self._build_dense_state(obs_raw, obs)
            d = float(candidate[0])
            if d >= min_dist and d <= max_dist:
                dense_obs = candidate
                break
        if dense_obs is None:
            dense_obs = candidate
        self._last_distance = float(dense_obs[0])
        return dense_obs, {
            "env_type": "external",
            "curriculum_stage": stage,
            **dict(info),
        }

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self._step_count += 1
        a = np.asarray(action, dtype=np.float32).reshape(-1)
        if a.size < 2:
            a = np.pad(a, (0, max(0, 2 - a.size)))
        a = np.clip(a[:2], self.action_space.low[:2], self.action_space.high[:2])

        if self._use_fallback:
            x, y, heading, speed, tx, ty = self._fallback_state.tolist()
            steering = float(a[0])
            throttle = float(a[1])
            heading = float(np.clip(heading + 0.08 * steering, -1.0, 1.0))
            speed = float(np.clip(speed + 0.05 * throttle, -1.0, 1.0))
            x = float(np.clip(x + speed * np.cos(np.pi * heading) * 0.03, -1.0, 1.0))
            y = float(np.clip(y + speed * np.sin(np.pi * heading) * 0.03, -1.0, 1.0))
            self._fallback_state = np.array([x, y, heading, speed, tx, ty], dtype=np.float32)
            obs_raw = self._fallback_state.copy()
            obs = self._build_dense_state(obs_raw, obs_raw)
            dist = float(obs[0])
            prev_dist = float(self._last_distance if self._last_distance is not None else dist)
            collision = bool(abs(x) >= 0.98 or abs(y) >= 0.98)
            success = bool(dist < 0.5 and not collision)
            soft_success = bool(dist < 2.0)
            if soft_success:
                self._near_success_count += 1
            reward, reward_components = self._reward_components(
                distance=dist,
                success=success,
                soft_success=soft_success,
                collision=collision,
                previous_distance=prev_dist,
            )
            self._last_distance = dist
            self._last_success = success
            self._last_collision = collision
            terminated = success
            truncated = self._step_count >= self.max_episode_steps
            info: Dict[str, Any] = {
                "env_type": "external_fallback",
                "is_success": success,
                "success": success,
                "soft_success_triggered": soft_success,
                "near_success_count": int(self._near_success_count),
                "collision": collision,
                "reward_components": reward_components,
            }
            info.update(reward_components)
            return obs, float(reward), terminated, truncated, info

        assert self._base_env is not None
        obs_raw, _base_reward, base_terminated, truncated, info = self._base_env.step(a)
        flat_obs = self._flatten_obs(obs_raw)
        obs = self._build_dense_state(obs_raw, flat_obs)
        dist = float(obs[0])
        prev_dist = float(self._last_distance if self._last_distance is not None else dist)

        collision = bool(info.get("crashed", False) or info.get("collision", False))
        if not collision:
            vehicle = getattr(getattr(self._base_env, "unwrapped", None), "vehicle", None)
            collision = bool(getattr(vehicle, "crashed", False)) if vehicle is not None else False
        success = bool(dist < 0.5 and not collision)
        soft_success = bool(dist < 2.0)
        if soft_success:
            self._near_success_count += 1

        reward, reward_components = self._reward_components(
            distance=dist,
            success=success,
            soft_success=soft_success,
            collision=collision,
            previous_distance=prev_dist,
        )
        self._last_distance = dist
        self._last_success = success
        self._last_collision = collision

        force_trunc = self._step_count >= self.max_episode_steps
        terminated = bool(base_terminated or success)
        info_out: Dict[str, Any] = dict(info)
        info_out.update(
            {
                "is_success": success,
                "success": success,
                "soft_success_triggered": soft_success,
                "near_success_count": int(self._near_success_count),
                "collision": collision,
                "reward_components": reward_components,
            }
        )
        info_out.update(reward_components)
        return obs, float(reward), bool(terminated), bool(truncated or force_trunc), info_out
