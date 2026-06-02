"""
Grid navigasyon baseline politikaları + BFS oracle toplam ödülü.
"""

from __future__ import annotations

from collections import deque
from typing import List

import numpy as np

from env.grid_navigation_env import (
    DR_DC,
    GridNavEpisodeConfig,
    GridParkingNavigationEnv,
)


def random_action(env: GridParkingNavigationEnv) -> int:
    return int(env._rng.integers(0, 4))


def greedy_manhattan_step(env: GridParkingNavigationEnv) -> int:
    """Bir adımda hedefe Manhattan mesafesini en çok azaltan yasal hareket."""
    assert env._walls is not None
    ar, ac = env._agent
    gr, gc = env._goal
    best_a = 0
    best_d = 10**9
    for a in range(4):
        dr, dc = DR_DC[a]
        nr, nc = ar + dr, ac + dc
        if nr < 0 or nc < 0 or nr >= env.height or nc >= env.width or env._walls[nr, nc]:
            continue
        d = abs(nr - gr) + abs(nc - gc)
        if d < best_d:
            best_d = d
            best_a = a
    if best_d == 10**9:
        return random_action(env)
    return best_a


def _walkable(env: GridParkingNavigationEnv, r: int, c: int) -> bool:
    assert env._walls is not None
    if r < 0 or c < 0 or r >= env.height or c >= env.width:
        return False
    return not env._walls[r, c]


def bfs_first_action(env: GridParkingNavigationEnv) -> int:
    """Ajan → hedef en kısa yolun ilk eylemi (BFS)."""
    assert env._walls is not None
    start = env._agent
    goal = env._goal
    if start == goal:
        return 0
    q: deque = deque([start])
    came_from: dict[tuple[int, int], tuple[int, int, int]] = {}
    seen = {start}
    while q:
        r, c = q.popleft()
        if (r, c) == goal:
            break
        for a in range(4):
            dr, dc = DR_DC[a]
            nr, nc = r + dr, c + dc
            if not _walkable(env, nr, nc):
                continue
            if (nr, nc) in seen:
                continue
            seen.add((nr, nc))
            came_from[(nr, nc)] = (r, c, a)
            q.append((nr, nc))
    if goal not in seen:
        return greedy_manhattan_step(env)
    cell = goal
    last_a = 0
    while cell != start:
        pr, pc, a = came_from[cell]
        last_a = a
        cell = (pr, pc)
    return int(last_a)


def oracle_episode_return(episode_configs: List[GridNavEpisodeConfig], seed: int) -> float:
    """Aynı seed ile BFS oracle tam bölüm toplam ödülü."""
    e = GridParkingNavigationEnv(episode_configs, seed=0, max_episode_steps=250)
    e.reset(seed=seed)
    total = 0.0
    done = False
    while not done:
        a = bfs_first_action(e)
        _o, r, term, trunc, _ = e.step(a)
        total += float(r)
        done = bool(term or trunc)
    return total


def rollout_episode_return(
    env: GridParkingNavigationEnv,
    action_fn,
    seed: int,
) -> tuple[float, bool, int]:
    """Tek bölüm: toplam ödül, başarı, adım sayısı."""
    env.reset(seed=seed)
    total = 0.0
    done = False
    steps = 0
    success = False
    while not done:
        a = action_fn(env)
        _o, r, term, trunc, info = env.step(a)
        total += float(r)
        steps += 1
        done = bool(term or trunc)
        if info.get("success"):
            success = True
    return total, success, steps
