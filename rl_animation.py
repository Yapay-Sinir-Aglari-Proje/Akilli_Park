"""
RL ajanının park etmeyi nasıl öğrendiğini animasyonla gösterir.

Öncelik: GridParkingEnv (train/demo) + eğitilmiş PPO (models/).
İsteğe bağlı: --legacy-smart ile SmartParkingEnv + PPO (grid modeliyle gözlem uyumu gerekir).

Çıktı: animations/training_animation.mp4 (ffmpeg varsa) veya .gif (yoksa).
"""

from __future__ import annotations

import argparse
import math
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.animation as mplanim
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors
from matplotlib import patches as mpatches
from matplotlib import transforms as mtransforms

from ml_config import ENV_TYPE
from paths import DATA_PROCESSED, MODELS_DIR, PROJECT_ROOT
from parking_rl.external_env import ExternalParkingEnv
from parking_rl.grid_parking_env import (
    CELL_BUILDING,
    CELL_EMPTY,
    CELL_GOAL,
    CELL_OCCUPIED,
    INVALID_ACTION_PENALTY,
    GridParkingEnv,
    TRAIN_PPO_ENV_KWARGS,
)

# --- Görsel (kullanıcı spesifikasyonu) ---
_COLOR_AGENT = "#1f77b4"  # mavi
_COLOR_GOAL = "#ffdd57"  # sarı
_COLOR_GOAL_FLASH = "#fff4a3"  # hedef yanıp sönme (bonus)
_COLOR_OCCUPIED = "#d62728"  # kırmızı
_COLOR_EMPTY = "#2ca02c"  # yeşil — boş park
_COLOR_BUILDING = "#7f7f7f"  # bina / engel (gri)
_COLOR_GRID = "#d3d3d3"  # ızgara çizgisi
_COLOR_ARROW = "#ff7f0e"  # turuncu
_COLOR_BACKGROUND = "#e8e8e8"  # SmartParking rasterında otoparkın düşmediği hücre
_COLOR_ASPHALT = "#050505"
_COLOR_FRAME = "#9a9a9a"
_COLOR_SLOT_BODY = "#c7d317"
_COLOR_SLOT_CAP = "#20c05c"
_COLOR_SLOT_EMPTY_EDGE = "#0a8f3c"
_COLOR_SLOT_OCC_EDGE = "#0f7f3c"
_COLOR_SLOT_GOAL = "#f7f6b5"

# SmartParking raster: arka plan kutusu (grid ile aynı kod 0)
CELL_BACKGROUND = CELL_BUILDING

_ANIM_DIR = PROJECT_ROOT / "animations"
_DEFAULT_MP4 = _ANIM_DIR / "training_animation.mp4"
_DEFAULT_GIF = _ANIM_DIR / "training_animation.gif"

# Kayıtlı video/GIF kare hızı (düşük = daha yavaş dosya)
_DEFAULT_FILE_FPS = 6
# Ekrandaki plt.show önizlemesi: FuncAnimation interval (ms); yüksek = daha yavaş
_DEFAULT_PREVIEW_MS_PER_FRAME = 280

# External görsel akış ayarları (yalnızca animasyon katmanı)
_EXTERNAL_AGENT_START_X = 1.1
_EXTERNAL_AGENT_START_Y = 8.4
_EXTERNAL_PROGRESS_SCALE = 4.2  # büyük değer = hedefe daha hızlı görsel yaklaşım

_ACTION_NAMES_DEMO = ("UP", "DOWN", "LEFT", "RIGHT")
_ACTION_LABELS_TR = {
    "UP": "YUKARI ↑",
    "DOWN": "ASAGI ↓",
    "LEFT": "SOL ←",
    "RIGHT": "SAG →",
}


def _legal_actions_label(legal: Any) -> str:
    if not isinstance(legal, list) or not legal:
        return ""
    parts = [_ACTION_NAMES_DEMO[int(i) % 4] for i in legal]
    return "Geçerli: " + ", ".join(parts)


def _present_action_label(action: Any) -> str:
    txt = str(action)
    return _ACTION_LABELS_TR.get(txt, txt)


def _distance_label(agent: Tuple[float, float], goal: Tuple[float, float]) -> str:
    dr = float(goal[0]) - float(agent[0])
    dc = float(goal[1]) - float(agent[1])
    dist = math.hypot(dr, dc)
    return f"{dist:.2f}"


def _ensure_anim_dir() -> Path:
    _ANIM_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[rl_animation] Çıktı klasörü hazır: {_ANIM_DIR.resolve()}")
    return _ANIM_DIR


def _ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def create_grid_env(
    *,
    size: int = 10,
    mode: str = "demo",
    max_steps: int = 200,
    building_ratio: float = 0.18,
    occupied_ratio: float = 0.12,
    debug_checks: bool = False,
    step_debug_log: bool = False,
    match_train_mdp: bool = False,
) -> GridParkingEnv:
    """GridParkingEnv: train (statik hedef / grid) veya demo (dinamik doluluk).

    match_train_mdp: True ise rl_model ile aynı ödül / MDP parametreleri (TRAIN_PPO_ENV_KWARGS)
    üzerine boyut, mod ve max_steps gibi alanlar uygulanır.
    """
    m = "demo" if str(mode).lower() == "demo" else "train"
    kw: Dict[str, Any] = {
        "size": size,
        "mode": m,
        "max_episode_steps": max_steps,
        "building_ratio": building_ratio,
        "occupied_ratio": occupied_ratio,
        "debug_checks": debug_checks,
        "step_debug_log": step_debug_log,
    }
    if match_train_mdp:
        merged = dict(TRAIN_PPO_ENV_KWARGS)
        merged.update(kw)
        return GridParkingEnv(**merged)  # type: ignore[arg-type]
    return GridParkingEnv(**kw)  # type: ignore[arg-type]


def draw_grid_lines(ax: plt.Axes, h: int, w: int) -> None:
    for g in np.arange(0, w + 1, 1):
        ax.axvline(g, color=_COLOR_GRID, linewidth=0.7, zorder=2)
    for g in np.arange(0, h + 1, 1):
        ax.axhline(g, color=_COLOR_GRID, linewidth=0.7, zorder=2)


def draw_obstacles_and_parking(ax: plt.Axes, grid_base: np.ndarray) -> None:
    """Bina (0), boş (1), dolu (2) — sarı hedef ayrı katmanda (üzerine yazılmaz)."""
    h, w = grid_base.shape
    cmap = mcolors.ListedColormap([_COLOR_BUILDING, _COLOR_EMPTY, _COLOR_OCCUPIED])
    norm = mcolors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)
    ax.imshow(
        grid_base,
        origin="lower",
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
        extent=(0, w, 0, h),
        zorder=1,
    )


def _draw_parking_slot(
    ax: plt.Axes,
    r: int,
    c: int,
    *,
    state: str,
    goal_flash: bool = False,
) -> None:
    x = c + 0.5
    y = r + 0.5
    body_w = 0.40
    body_h = 0.66
    cap_h = 0.17
    edge_w = 1.6

    body_color = _COLOR_SLOT_BODY
    cap_color = _COLOR_SLOT_CAP
    edge_color = _COLOR_SLOT_EMPTY_EDGE
    alpha = 0.95

    if state == "occupied":
        edge_color = _COLOR_SLOT_OCC_EDGE
    elif state == "goal":
        body_color = _COLOR_GOAL_FLASH if goal_flash else _COLOR_SLOT_GOAL
        cap_color = "#ffa767" if goal_flash else "#ffd05f"
        edge_color = "#f8b400"
        edge_w = 2.2 if goal_flash else 1.8

    body = mpatches.Rectangle(
        (x - body_w / 2, y - body_h / 2),
        body_w,
        body_h,
        facecolor=body_color,
        edgecolor=edge_color,
        linewidth=edge_w,
        alpha=alpha,
        zorder=4,
    )
    cap = mpatches.Rectangle(
        (x - body_w / 2, y + body_h / 2 - cap_h),
        body_w,
        cap_h,
        facecolor=cap_color,
        edgecolor=cap_color,
        linewidth=0.8,
        alpha=alpha,
        zorder=5,
    )
    ax.add_patch(body)
    ax.add_patch(cap)


def draw_target_square(ax: plt.Axes, *, x: float, y: float) -> None:
    marker = mpatches.Rectangle(
        (x - 0.14, y - 0.14),
        0.28,
        0.28,
        facecolor="#6ec6ff",
        edgecolor="#8de0ff",
        linewidth=1.0,
        zorder=12,
    )
    ax.add_patch(marker)


def draw_stylized_parking_world(
    ax: plt.Axes,
    grid: np.ndarray,
    *,
    goal: Optional[Tuple[int, int]] = None,
    goal_flash: bool = False,
) -> None:
    """Sunum için yüksek kontrast park sahnesi (yalnızca görsel katman)."""
    h, w = grid.shape
    ax.set_facecolor(_COLOR_ASPHALT)

    # Çerçeve (örnek görseldeki gri sınır etkisi)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(_COLOR_FRAME)
        spine.set_linewidth(2.2)

    goal_set = set()
    if goal is not None:
        goal_set.add((int(goal[0]), int(goal[1])))

    for r in range(h):
        for c in range(w):
            cell = int(grid[r, c])
            if (r, c) in goal_set or cell == CELL_GOAL:
                _draw_parking_slot(ax, r, c, state="goal", goal_flash=goal_flash)
            elif cell == CELL_EMPTY:
                _draw_parking_slot(ax, r, c, state="empty")
            elif cell == CELL_OCCUPIED:
                _draw_parking_slot(ax, r, c, state="occupied")
            # CELL_BUILDING/CELL_BACKGROUND için boş bırakıp asfalt etkisini koruyoruz.

    # Ortadaki ayırıcı şerit: görseldeki "koridor" hissini verir.
    if h >= 8:
        mid_y = h * 0.60
        ax.plot([w * 0.25, w * 0.75], [mid_y, mid_y], color=_COLOR_FRAME, linewidth=3.0, zorder=3)


def draw_goal_marker(ax: plt.Axes, goal: Tuple[int, int], *, goal_flash: bool) -> None:
    gr, gc = goal
    face = _COLOR_GOAL_FLASH if goal_flash else _COLOR_GOAL
    edge = "#c9a000" if not goal_flash else "#ff9800"
    lw = 2.8 if goal_flash else 1.8
    ax.scatter(
        [gc + 0.5],
        [gr + 0.5],
        s=520,
        marker="s",
        facecolors=face,
        edgecolors=edge,
        linewidths=lw,
        zorder=8,
    )


def draw_agent_marker(ax: plt.Axes, agent: Tuple[int, int]) -> None:
    ar, ac = agent
    car = mpatches.Rectangle(
        (ac + 0.5 - 0.17, ar + 0.5 - 0.13),
        0.34,
        0.26,
        angle=0,
        facecolor="#2db8ff",
        edgecolor="white",
        linewidth=1.2,
        zorder=10,
    )
    ax.add_patch(car)


def draw_agent_marker_xy(ax: plt.Axes, *, x: float, y: float) -> None:
    car = mpatches.Rectangle(
        (x - 0.17, y - 0.13),
        0.34,
        0.26,
        angle=0,
        facecolor="#2db8ff",
        edgecolor="white",
        linewidth=1.2,
        zorder=10,
    )
    ax.add_patch(car)


def draw_agent_car_xy(
    ax: plt.Axes,
    *,
    x: float,
    y: float,
    direction: Optional[Tuple[float, float]] = None,
) -> None:
    angle_deg = 0.0
    if direction is not None:
        dx, dy = float(direction[0]), float(direction[1])
        if math.hypot(dx, dy) > 1e-9:
            angle_deg = math.degrees(math.atan2(dy, dx)) - 90.0
    body_w = 0.32
    body_h = 0.56
    body = mpatches.Rectangle(
        (x - body_w / 2.0, y - body_h / 2.0),
        body_w,
        body_h,
        angle=0.0,
        facecolor="#ff9f1a",
        edgecolor="#ffd08a",
        linewidth=0.7,
        zorder=13,
    )
    rot = mtransforms.Affine2D().rotate_deg_around(x, y, angle_deg)
    body.set_transform(rot + ax.transData)
    ax.add_patch(body)


def draw_trajectory_path(
    ax: plt.Axes,
    trajectory: List[Tuple[float, float]],
    *,
    color_rgb: Tuple[float, float, float] = (0.53, 0.33, 0.96),
) -> None:
    if len(trajectory) < 2:
        return
    nseg = len(trajectory) - 1
    for i in range(nseg):
        x0, y0 = trajectory[i]
        x1, y1 = trajectory[i + 1]
        alpha = 0.18 + 0.75 * ((i + 1) / max(1, nseg))
        ax.plot(
            [x0, x1],
            [y0, y1],
            linestyle=(0, (4, 4)),
            linewidth=2.0,
            color=color_rgb,
            alpha=alpha,
            solid_capstyle="round",
            zorder=9,
        )


def draw_agent_heading(
    ax: plt.Axes,
    agent: Tuple[int, int],
    arrow: Optional[Tuple[float, float]],
) -> None:
    if arrow is None:
        return
    dx, dy = float(arrow[0]), float(arrow[1])
    mag = math.hypot(dx, dy)
    if mag <= 1e-9:
        return
    dx /= mag
    dy /= mag
    ax.quiver(
        agent[1] + 0.5,
        agent[0] + 0.5,
        dx * 0.55,
        dy * 0.55,
        angles="xy",
        scale_units="xy",
        scale=1.0,
        color="#4fd7ff",
        width=0.012,
        zorder=11,
    )


def draw_explain_panel(ax: plt.Axes) -> None:
    panel = (
        "Mavi: Agent (araç)\n"
        "Açık mavi ok: seçilen yön\n"
        "Sarı slot: Hedef park\n"
        "Yeşil/Kırmızı kenar: Boş/Dolu"
    )
    ax.text(
        0.985,
        0.985,
        panel,
        transform=ax.transAxes,
        va="top",
        ha="right",
        fontsize=7.2,
        color="#f0f0f0",
        bbox={
            "boxstyle": "round,pad=0.18",
            "facecolor": "#0f0f0f",
            "edgecolor": "#8a8a8a",
            "alpha": 0.82,
        },
        zorder=20,
    )


def _external_dense_to_scene(
    *,
    step: int,
    distance: float,
    direction: Tuple[float, float],
) -> Tuple[np.ndarray, Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
    """4D dense external observation'u görsel sahneye dönüştürür."""
    h, w = 10, 10
    grid = np.full((h, w), CELL_BUILDING, dtype=int)

    # Referans görseldeki gibi park slotları
    slot_cells = [(8, 2), (8, 6), (6, 4), (3, 2), (3, 6), (0, 3), (0, 4)]
    for r, c in slot_cells:
        grid[r, c] = CELL_EMPTY

    target_xy = (6.8, 0.7)

    dx, dy = float(direction[0]), float(direction[1])
    mag = math.hypot(dx, dy)
    if mag > 1e-9:
        dx /= mag
        dy /= mag
    else:
        dx, dy = 0.5, -0.8

    # Agent üstten başlasın ve bariyeri "üstünden geçmeden" sağ uçtan dolaşsın.
    # distance dar aralıkta kaldığında da görünür hareket için güçlü ölçek kullanılır.
    start_x, start_y = _EXTERNAL_AGENT_START_X, _EXTERNAL_AGENT_START_Y
    end_x, end_y = target_xy
    distance_progress = float(np.clip((0.95 - distance) * _EXTERNAL_PROGRESS_SCALE, 0.0, 1.0))
    # İlk karede agent tam başlangıç noktasında görünsün.
    step_progress = float(np.clip(step / 10.0, 0.0, 1.0))
    progress = min(distance_progress, step_progress) if step <= 10 else distance_progress

    # Gri bariyer (draw_stylized_parking_world ile uyumlu): y ~= h*0.60, x in [w*0.25, w*0.75]
    # Path'i 4 parçalı tanımlıyoruz:
    # start -> solda aşağı in -> sağ üst geçit -> sağ alt geçit -> hedef
    # Böylece üst sıradaki park aracına teğet geçmez.
    drop_x = 1.35
    drop_y = 7.15
    gate_x = 8.55
    gate_top_y = 6.18
    gate_bot_y = 5.00

    p0 = (start_x, start_y)
    p1 = (drop_x, drop_y)
    p2 = (gate_x, gate_top_y)
    p3 = (gate_x, gate_bot_y)
    p4 = (end_x, end_y)

    seg_lens = [
        math.hypot(p1[0] - p0[0], p1[1] - p0[1]),
        math.hypot(p2[0] - p1[0], p2[1] - p1[1]),
        math.hypot(p3[0] - p2[0], p3[1] - p2[1]),
        math.hypot(p4[0] - p3[0], p4[1] - p3[1]),
    ]
    total_len = max(1e-6, seg_lens[0] + seg_lens[1] + seg_lens[2] + seg_lens[3])
    travel = progress * total_len

    if travel <= seg_lens[0]:
        t = travel / max(1e-6, seg_lens[0])
        ag_c = p0[0] + (p1[0] - p0[0]) * t
        ag_r = p0[1] + (p1[1] - p0[1]) * t
    elif travel <= seg_lens[0] + seg_lens[1]:
        t = (travel - seg_lens[0]) / max(1e-6, seg_lens[1])
        ag_c = p1[0] + (p2[0] - p1[0]) * t
        ag_r = p1[1] + (p2[1] - p1[1]) * t
    elif travel <= seg_lens[0] + seg_lens[1] + seg_lens[2]:
        t = (travel - seg_lens[0] - seg_lens[1]) / max(1e-6, seg_lens[2])
        ag_c = p2[0] + (p3[0] - p2[0]) * t
        ag_r = p2[1] + (p3[1] - p2[1]) * t
    else:
        t = (travel - seg_lens[0] - seg_lens[1] - seg_lens[2]) / max(1e-6, seg_lens[3])
        ag_c = p3[0] + (p4[0] - p3[0]) * t
        ag_r = p3[1] + (p4[1] - p3[1]) * t

    # Küçük yön jitter'ı (çok agresif değil): hareket hissi verir.
    ag_c += dx * 0.08
    ag_r += dy * 0.08

    # Park halindeki araçlarla görsel çakışmayı engelle.
    # Slot merkezine çok yaklaşıldığında agenti en kısa yönde dışarı iter.
    min_clearance = 0.58
    for sr, sc in slot_cells:
        cx = float(sc) + 0.5
        cy = float(sr) + 0.5
        vx = ag_c - cx
        vy = ag_r - cy
        d = math.hypot(vx, vy)
        if d < min_clearance:
            if d < 1e-6:
                vx, vy, d = 1.0, 0.0, 1.0
            scale = min_clearance / d
            ag_c = cx + vx * scale
            ag_r = cy + vy * scale

    ag_c = float(np.clip(ag_c, 0.6, w - 0.6))
    ag_r = float(np.clip(ag_r, 0.6, h - 0.6))
    return grid, (ag_r, ag_c), target_xy, (dx, dy)


def _draw_scene_common(
    ax: plt.Axes,
    *,
    grid: np.ndarray,
    goal: Optional[Tuple[int, int]],
    goal_flash: bool,
    agent_xy: Tuple[float, float],
    arrow: Optional[Tuple[float, float]],
    trail_from: Optional[Tuple[float, float]] = None,
    show_default_agent: bool = True,
) -> None:
    """Ortak sahne katmanı: dünya + araç + yön oku + opsiyonel iz."""
    draw_stylized_parking_world(ax, grid, goal=goal, goal_flash=goal_flash)

    ax_x, ax_y = float(agent_xy[0]), float(agent_xy[1])
    if show_default_agent:
        draw_agent_marker_xy(ax, x=ax_x, y=ax_y)

    if trail_from is not None:
        px, py = float(trail_from[0]), float(trail_from[1])
        ax.plot([px, ax_x], [py, ax_y], color="#86e3ff", linewidth=2.2, alpha=0.65, zorder=7)

    if arrow is not None:
        dx, dy = float(arrow[0]), float(arrow[1])
        mag = math.hypot(dx, dy)
        if mag > 1e-9:
            dx /= mag
            dy /= mag
            ax.quiver(
                ax_x,
                ax_y,
                dx * 0.55,
                dy * 0.55,
                angles="xy",
                scale_units="xy",
                scale=1.0,
                color="#4fd7ff",
                width=0.012,
                zorder=11,
            )

    ax.set_xlim(0, grid.shape[1])
    ax.set_ylim(0, grid.shape[0])
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    draw_explain_panel(ax)


def draw_frame(ax: plt.Axes, state: Dict[str, Any]) -> None:
    """draw_grid → park durumu → hedef → ajan sırası; SmartParking için eski 'grid' yolu korunur."""
    ax.clear()

    if state.get("kind") == "grid_parking":
        grid_base = np.asarray(state["grid_base"], dtype=int)
        h, w = grid_base.shape
        goal = state["goal"]
        ag_r, ag_c = state["agent"]
        ag_r, ag_c = float(ag_r), float(ag_c)

        arrow = state.get("arrow")

        prev_agent = state.get("prev_agent")
        trail_from: Optional[Tuple[float, float]] = None
        if isinstance(prev_agent, (tuple, list)) and len(prev_agent) == 2:
            pr, pc = float(prev_agent[0]), float(prev_agent[1])
            trail_from = (pc + 0.5, pr + 0.5)

        _draw_scene_common(
            ax,
            grid=grid_base,
            goal=goal,
            goal_flash=bool(state.get("goal_flash")),
            agent_xy=(ag_c + 0.5, ag_r + 0.5),
            arrow=arrow,
            trail_from=trail_from,
        )

        step = int(state.get("step", 0))
        action = _present_action_label(state.get("action", "-"))
        _ = float(state.get("reward", 0.0))
        distance_text = _distance_label((ag_r, ag_c), (float(goal[0]), float(goal[1])))
        banner = state.get("banner")
        title = f"Step: {step}   |   Action: {action}   |   Hedefe Uzaklık: {distance_text}"
        ax.set_title(title, fontsize=13, color="#f2f2f2")
        if banner:
            ax.text(
                0.5,
                -0.06,
                banner,
                transform=ax.transAxes,
                ha="center",
                fontsize=10,
                color="#e0e0e0",
            )
        return

    if state.get("kind") == "external_dense":
        ax.clear()
        distance = float(state.get("distance", 0.0))
        dx, dy = state.get("arrow", (0.0, 0.0))
        step = int(state.get("step", 0))
        env_success = bool(state.get("success", False))

        grid, (ag_r, ag_c), target_xy, (ndx, ndy) = _external_dense_to_scene(
            step=step,
            distance=distance,
            direction=(dx, dy),
        )
        draw_stylized_parking_world(ax, grid, goal=None, goal_flash=False)
        trajectory = state.get("trajectory", [])
        if isinstance(trajectory, list):
            traj_xy: List[Tuple[float, float]] = []
            for p in trajectory:
                if isinstance(p, (tuple, list)) and len(p) == 2:
                    traj_xy.append((float(p[0]), float(p[1])))
            draw_trajectory_path(ax, traj_xy)

        prev_xy = state.get("prev_xy")
        heading_for_car: Tuple[float, float] = (ndx, ndy)
        if isinstance(prev_xy, (tuple, list)) and len(prev_xy) == 2:
            px, py = float(prev_xy[0]), float(prev_xy[1])
            move_dx = ag_c - px
            move_dy = ag_r - py
            if math.hypot(move_dx, move_dy) > 1e-9:
                heading_for_car = (move_dx, move_dy)
            ax.plot(
                [px, ag_c],
                [py, ag_r],
                color="#80d4ff",
                linewidth=2.4,
                alpha=0.95,
                solid_capstyle="round",
                zorder=12,
            )
        draw_agent_car_xy(ax, x=ag_c, y=ag_r, direction=heading_for_car)
        draw_target_square(ax, x=target_xy[0], y=target_xy[1])
        ax.text(
            0.02,
            0.98,
            f"step: {step}   success: {env_success}   distance: {distance:.3f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            color="#e8e8e8",
        )
        ax.set_xlim(0, grid.shape[1])
        ax.set_ylim(0, grid.shape[0])
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        return

    grid = np.asarray(state["grid"], dtype=int)
    h, w = grid.shape
    ag_r, ag_c = state["agent"]
    ag_r, ag_c = float(ag_r), float(ag_c)

    cmap = mcolors.ListedColormap(
        [_COLOR_BACKGROUND, _COLOR_EMPTY, _COLOR_OCCUPIED, _COLOR_GOAL]
    )
    norm = mcolors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)

    # SmartParking/legacy grid görünümünü de aynı sunum stiline yaklaştır.
    _ = (cmap, norm)  # Eski renk eşlemesi bilinçli olarak korunuyor; render stili değişti.
    _draw_scene_common(
        ax,
        grid=grid,
        goal=None,
        goal_flash=bool(state.get("goal_flash", False)),
        agent_xy=(ag_c + 0.5, ag_r + 0.5),
        arrow=state.get("arrow"),
    )

    step = int(state.get("step", 0))
    action = _present_action_label(state.get("action", "-"))
    _ = float(state.get("reward", 0.0))
    goal_cells = np.argwhere(grid == CELL_GOAL)
    if goal_cells.size > 0:
        # Birden fazla hedef varsa en yakını ile sunumda anlaşılır uzaklık ver.
        dvals = [
            math.hypot(float(gr) - ag_r, float(gc) - ag_c) for gr, gc in goal_cells.tolist()
        ]
        distance_text = f"{min(dvals):.2f}"
    else:
        distance_text = "-"
    ax.set_title(
        f"Step: {step}   |   Action: {action}   |   Hedefe Uzaklık: {distance_text}",
        fontsize=13,
        color="#f2f2f2",
    )


def _latlon_to_cell(
    lat: float,
    lon: float,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    rows: int,
    cols: int,
) -> Tuple[int, int]:
    lat_span = max(lat_max - lat_min, 1e-9)
    lon_span = max(lon_max - lon_min, 1e-9)
    c = int(np.clip((lon - lon_min) / lon_span * cols, 0, cols - 1))
    r = int(np.clip((lat - lat_min) / lat_span * rows, 0, rows - 1))
    return r, c


def _smart_parking_to_state(
    env: Any,
    *,
    step: int,
    action: Any,
    reward: float,
    info: Dict[str, Any],
    grid_shape: Tuple[int, int] = (18, 18),
) -> Dict[str, Any]:
    """SmartParkingEnv anlık görüntüsünü draw_frame uyumlu state'e çevirir."""
    rows, cols = grid_shape
    lat_min, lat_max = float(env.lat_min), float(env.lat_max)
    lon_min, lon_max = float(env.lon_min), float(env.lon_max)

    # Tüm kutu başta arka plan: otopark koordinatı bu hücreye düşmüyorsa yeşil değil gri kalır
    grid = np.full((rows, cols), CELL_BACKGROUND, dtype=int)

    target_idx = int(info.get("target_index", getattr(env, "_target_idx", 0)))
    t_idx = int(info.get("time_index", getattr(env, "_time_idx", 0)))
    snap = env._merge_snapshot(t_idx)  # noqa: SLF001

    for i, lot in enumerate(env.lots):
        r, c = _latlon_to_cell(
            lot.latitude, lot.longitude, lat_min, lat_max, lon_min, lon_max, rows, cols
        )
        occ, cap = snap.get(lot.parking_id, (0.0, 1.0))
        cap = max(1.0, float(cap))
        occ_ratio = float(np.clip(float(occ) / cap, 0.0, 1.0))
        if i == target_idx:
            grid[r, c] = CELL_GOAL
        elif occ_ratio >= 0.85:
            grid[r, c] = CELL_OCCUPIED
        else:
            grid[r, c] = CELL_EMPTY

    v = env.vehicle
    ag_r, ag_c = _latlon_to_cell(
        v.latitude, v.longitude, lat_min, lat_max, lon_min, lon_max, rows, cols
    )
    tlot = env.lots[target_idx]
    tr, tc = _latlon_to_cell(
        tlot.latitude, tlot.longitude, lat_min, lat_max, lon_min, lon_max, rows, cols
    )
    dx = (tc + 0.5) - (ag_c + 0.5)
    dy = (tr + 0.5) - (ag_r + 0.5)

    return {
        "grid": grid,
        "agent": (ag_r, ag_c),
        "step": int(step),
        "action": action,
        "reward": float(reward),
        "arrow": (dx, dy),
    }


def _policy_action(policy: Any, obs: Any, env: Any) -> Any:
    if hasattr(policy, "predict"):
        act, _ = policy.predict(obs, deterministic=True)
        arr = np.asarray(act)
        if arr.shape == ():
            return int(arr.reshape(()))
        return arr
    out = policy(obs)
    arr = np.asarray(out)
    if arr.shape == ():
        return int(arr.reshape(()))
    return arr


def _coerce_external_action(raw_action: Any) -> np.ndarray:
    """External env için [steering, throttle] continuous action garantisi."""
    # Grid/discrete model çıktısı external'da tek skalar gelebilir.
    # Bu durumda aracı gerçekten hareket ettirmek için anlamlı continuous aksiyona map edilir.
    if np.isscalar(raw_action):
        a = int(raw_action)
        # 0: ileri, 1: geri, 2: sol+ileri, 3: sağ+ileri
        table = {
            0: np.array([0.0, 1.0], dtype=np.float32),
            1: np.array([0.0, -1.0], dtype=np.float32),
            2: np.array([-0.85, 0.70], dtype=np.float32),
            3: np.array([0.85, 0.70], dtype=np.float32),
        }
        return table.get(a % 4, np.array([0.0, 0.8], dtype=np.float32))

    arr = np.asarray(raw_action, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return np.array([0.0, 0.8], dtype=np.float32)
    if arr.size == 1:
        # Tek değer geldiyse gaz eksik kalmasın diye ileri throttle ekle.
        return np.array([float(np.clip(arr[0], -1.0, 1.0)), 0.8], dtype=np.float32)
    return np.array(
        [float(np.clip(arr[0], -1.0, 1.0)), float(np.clip(arr[1], -1.0, 1.0))],
        dtype=np.float32,
    )


def _external_to_state(
    *,
    step: int,
    action: Any,
    reward: float,
    obs: Any,
    info: Dict[str, Any],
) -> Dict[str, Any]:
    obs_vec = np.asarray(obs, dtype=np.float32).reshape(-1)
    distance = float(obs_vec[0]) if obs_vec.size > 0 else float(info.get("distance", 0.0))
    occupancy = float(obs_vec[1]) if obs_vec.size > 1 else float(info.get("occupancy", 0.5))
    dx = float(obs_vec[2]) if obs_vec.size > 2 else 0.0
    dy = float(obs_vec[3]) if obs_vec.size > 3 else 0.0
    if isinstance(action, str):
        action_value: Any = action
    elif np.isscalar(action):
        action_value = float(action)
    else:
        action_value = np.asarray(action).tolist()
    return {
        "kind": "external_dense",
        "step": int(step),
        "action": action_value,
        "reward": float(reward),
        "distance": distance,
        "occupancy": occupancy,
        "arrow": (dx, dy),
        "success": bool(info.get("is_success", info.get("success", False))),
        "soft_success": bool(info.get("soft_success_triggered", False)),
        "collision": bool(info.get("collision", False)),
    }


def _random_grid_action(env: GridParkingEnv) -> int:
    """Geçerli yönlerden üret (maskesiz rastgele aksiyon spam'ini önler)."""
    legal = env._legal_action_indices()
    if not legal:
        return int(env.action_space.sample())
    j = int(env.np_random.integers(0, len(legal)))
    return int(legal[j])


def _inject_prev_agent_markers(trajectory: List[Dict[str, Any]]) -> None:
    """Hareket izi için her kareye bir önceki ajan konumunu ekler."""
    prev: Optional[Tuple[int, int]] = None
    for st in trajectory:
        agent = st.get("agent")
        if isinstance(agent, (tuple, list)) and len(agent) == 2:
            cur = (int(agent[0]), int(agent[1]))
            st["prev_agent"] = prev
            prev = cur


def run_episode(
    env: Any,
    policy: Optional[Any] = None,
    *,
    seed: int = 42,
    max_steps: int = 200,
    mode: Optional[str] = None,
    log_episode_summary: bool = False,
) -> List[Dict[str, Any]]:
    """
    Bir episode boyunca her adım için draw_frame uyumlu state listesi döndürür.
    mode: 'grid_parking' | 'smart' | None (None ise ortam tipinden seçilir).
    """
    if mode is None:
        if isinstance(env, GridParkingEnv):
            mode = "grid_parking"
        elif isinstance(env, ExternalParkingEnv):
            mode = "external"
        else:
            mode = "smart"

    trajectory: List[Dict[str, Any]] = []
    external_traj: List[Tuple[float, float]] = []
    obs, info = env.reset(seed=seed)
    prev_agent: Optional[Tuple[int, int]] = None
    last_step_info: Dict[str, Any] = dict(info) if isinstance(info, dict) else {}

    if mode == "grid_parking":
        assert isinstance(env, GridParkingEnv)
        s0 = env.to_draw_state(
            step=0,
            action="-",
            reward=0.0,
            prev_agent=None,
            goal_flash=False,
            banner=None,
            invalid_move=False,
        )
        s0["env_mode"] = f"mode={env.mode}"
        trajectory.append(s0)
        prev_agent = tuple(env.agent)
    elif mode == "external":
        init_info = dict(info) if isinstance(info, dict) else {}
        st0 = _external_to_state(step=0, action="-", reward=0.0, obs=obs, info=init_info)
        _grid0, (ag_r0, ag_c0), _target0, _dir0 = _external_dense_to_scene(
            step=0,
            distance=float(st0.get("distance", 0.0)),
            direction=tuple(st0.get("arrow", (0.0, 0.0))),
        )
        external_traj.append((ag_c0, ag_r0))
        st0["trajectory"] = list(external_traj)
        st0["prev_xy"] = None
        trajectory.append(
            st0
        )
    else:
        init_info: Dict[str, Any] = dict(info) if isinstance(info, dict) else {}
        init_info.setdefault("target_index", int(getattr(env, "_target_idx", 0)))
        init_info.setdefault("time_index", int(getattr(env, "_time_idx", 0)))
        trajectory.append(
            _smart_parking_to_state(env, step=0, action="-", reward=0.0, info=init_info)
        )

    for t in range(max_steps):
        if policy is None:
            if isinstance(env, GridParkingEnv):
                action = _random_grid_action(env)
            elif hasattr(env, "action_space") and hasattr(env.action_space, "sample"):
                action = env.action_space.sample()
            else:
                action = 0
        else:
            action = _policy_action(policy, obs, env)

        if mode == "external":
            action = _coerce_external_action(action)

        if mode == "grid_parking":
            prev_agent = tuple(env.agent)

        obs, reward, terminated, truncated, step_info = env.step(action)
        last_step_info = step_info
        merged = {**(info if isinstance(info, dict) else {}), **step_info}

        if mode == "grid_parking":
            assert isinstance(env, GridParkingEnv)
            aname = _ACTION_NAMES_DEMO[int(action) % 4]
            flash = bool(step_info.get("goal_flash", False))
            banner = (
                "Doluluk oranı nedeniyle hedef değiştirildi."
                if step_info.get("goal_reassigned_this_step")
                else None
            )
            st = env.to_draw_state(
                step=t + 1,
                action=aname,
                reward=float(reward),
                prev_agent=prev_agent,
                goal_flash=flash,
                banner=banner,
                invalid_move=bool(step_info.get("invalid_move", False)),
            )
            la = step_info.get("legal_actions")
            if isinstance(la, list):
                st["legal_actions"] = la
            st["env_mode"] = f"mode={env.mode}"
            trajectory.append(st)
        elif mode == "external":
            st = _external_to_state(
                step=t + 1,
                action=action,
                reward=float(reward),
                obs=obs,
                info=merged,
            )
            _gridx, (ag_rx, ag_cx), _targetx, _dirx = _external_dense_to_scene(
                step=t + 1,
                distance=float(st.get("distance", 0.0)),
                direction=tuple(st.get("arrow", (0.0, 0.0))),
            )
            prev_xy = external_traj[-1] if external_traj else None
            external_traj.append((ag_cx, ag_rx))
            st["trajectory"] = list(external_traj)
            st["prev_xy"] = prev_xy
            trajectory.append(st)
        else:
            trajectory.append(
                _smart_parking_to_state(
                    env,
                    step=t + 1,
                    action=int(action),
                    reward=float(reward),
                    info=merged,
                )
            )

        if terminated or truncated:
            break
        info = merged

    if (
        log_episode_summary
        and mode == "grid_parking"
        and isinstance(env, GridParkingEnv)
    ):
        summ = last_step_info.get("episode_summary")
        if summ is not None:
            print(f"[rl_animation] episode_summary: {summ}")
        else:
            print("[rl_animation] episode_summary: (yok — ortam episode bitirmeden kesildi olabilir)")

    if len(trajectory) == 0:
        print("[rl_animation] Uyarı: trajectory boş; tek karelik yedek state eklendi.")
        if isinstance(env, GridParkingEnv):
            st = env.to_draw_state(
                step=0,
                action="N/A",
                reward=0.0,
                prev_agent=None,
                goal_flash=False,
                invalid_move=False,
            )
            st["env_mode"] = f"mode={env.mode}"
            trajectory.append(st)

    _inject_prev_agent_markers(trajectory)
    print(f"[rl_animation] Episode toplam kare: {len(trajectory)}")
    return trajectory


def _try_load_smart_stack() -> Tuple[Optional[Any], Optional[Any], str]:
    """(env, policy, mode) — başarısızsa (None, None, '')."""
    train_csv = DATA_PROCESSED / "train.csv"
    if not train_csv.is_file():
        print(f"[rl_animation] SmartParkingEnv atlandı — dosya yok: {train_csv}")
        return None, None, ""

    try:
        from parking_rl.smart_parking_env import SmartParkingEnv
    except Exception as exc:
        print(f"[rl_animation] SmartParkingEnv import hatası: {exc}")
        return None, None, ""

    try:
        env = SmartParkingEnv(
            data_path=train_csv,
            max_episode_steps=120,
            randomize_start_time=False,
        )
        print(f"[rl_animation] SmartParkingEnv yüklendi: {train_csv}")
    except Exception as exc:
        print(f"[rl_animation] SmartParkingEnv kurulamadı: {exc}")
        return None, None, ""

    policy = _load_ppo_policy()
    if policy is None:
        print("[rl_animation] PPO bulunamadı veya yüklenemedi; SmartParking için rastgele politika.")

    return env, policy, "smart"


def _load_ppo_policy() -> Optional[Any]:
    for name in (
        "best_model.zip",
        "best_model",
        "ppo_parking_model_final.zip",
        "ppo_parking_model_final",
    ):
        p = MODELS_DIR / name
        if not p.exists():
            continue
        try:
            from stable_baselines3 import PPO

            pol = PPO.load(
                str(p),
                custom_objects={"lr_schedule": lambda _: 3e-4},
            )
            print(f"[rl_animation] PPO modeli yüklendi: {p}")
            print(f"[rl_animation] policy sınıfı (inference): {type(pol.policy).__name__}")
            return pol
        except Exception as exc:
            print(f"[rl_animation] PPO yüklenemedi ({p}): {exc}")
    return None


def _try_load_grid_stack(
    *,
    grid_mode: str = "demo",
    step_debug_log: bool = False,
    match_train_mdp: bool = False,
) -> Tuple[Any, Optional[Any], str]:
    """GridParkingEnv + (varsa) PPO — grid eğitimi ile uyumlu."""
    policy = _load_ppo_policy()
    if policy is None:
        print(
            "[rl_animation] PPO bulunamadı; grid ortamında rastgele politika "
            "(yalnızca geçerli yönlerden) kullanılacak — SB3 inference yok."
        )
    env = create_grid_env(
        mode=grid_mode,
        max_steps=220,
        debug_checks=False,
        step_debug_log=step_debug_log,
        match_train_mdp=match_train_mdp,
    )
    return env, policy, "grid_parking"


def _try_load_external_stack() -> Tuple[Any, Optional[Any], str]:
    policy = _load_ppo_policy()
    env = ExternalParkingEnv(max_episode_steps=220)
    return env, policy, "external"


def _save_animation(
    fig: plt.Figure,
    trajectory: List[Dict[str, Any]],
    *,
    fps: int = _DEFAULT_FILE_FPS,
) -> Path:
    _ensure_anim_dir()

    def _update(frame_idx: int) -> None:
        if frame_idx < len(trajectory):
            draw_frame(fig.axes[0], trajectory[frame_idx])

    n_frames = max(1, len(trajectory))
    anim = mplanim.FuncAnimation(
        fig,
        _update,
        frames=n_frames,
        interval=max(50, int(1000 / max(1, fps))),
        blit=False,
    )

    used_ffmpeg = _ffmpeg_available()
    if used_ffmpeg:
        out_path = _DEFAULT_MP4
        print(f"[rl_animation] ffmpeg bulundu; MP4 kaydediliyor: {out_path}")
        try:
            writer = mplanim.FFMpegWriter(fps=fps, metadata={"title": "Akıllı Park RL"})
            anim.save(str(out_path), writer=writer, dpi=120)
        except Exception as exc:
            print(f"[rl_animation] MP4 kaydı başarısız: {exc}")
            used_ffmpeg = False

    if not used_ffmpeg:
        out_path = _DEFAULT_GIF
        print(
            "[rl_animation] ffmpeg bulunamadı veya MP4 yazılamadı; "
            "Pillow ile GIF kaydediliyor."
        )
        try:
            writer = mplanim.PillowWriter(fps=fps)
            anim.save(str(out_path), writer=writer, dpi=120)
        except Exception as exc:
            print(f"[rl_animation] GIF kaydı başarısız: {exc}")
            # Son çare: tek kare PNG
            out_path = _ANIM_DIR / "training_animation_fallback.png"
            draw_frame(fig.axes[0], trajectory[0])
            fig.savefig(str(out_path), dpi=120, bbox_inches="tight")
            print(f"[rl_animation] Yedek statik görüntü yazıldı: {out_path}")

    plt.close(fig)
    if out_path.exists():
        print(f"[rl_animation] Dosya boyutu: {out_path.stat().st_size} byte — {out_path}")
    return out_path


def generate_animation(
    *,
    episode_seed: int = 42,
    fps: int = _DEFAULT_FILE_FPS,
    preview_ms_per_frame: int = _DEFAULT_PREVIEW_MS_PER_FRAME,
    use_smart_parking: bool = False,
    grid_mode: str = "demo",
    step_debug_log: bool = False,
    show_preview: bool = True,
    match_train_mdp: bool = False,
    log_episode_summary: bool = False,
    plain_train_reward: bool = False,
) -> Path:
    """
    GridParkingEnv (varsayılan) veya SmartParking ile animasyon üretir;
    MP4 veya GIF yazar; show_preview=True ise plt.show() ile önizleme açar.

    Args:
        episode_seed: Episode tohumu (grid reset ile uyumlu).
        fps: Kaydedilen MP4/GIF saniyedeki kare sayısı (2–4 genelde rahat izlenir).
        preview_ms_per_frame: Ekranda plt.show ile oynatırken kareler arası süre (ms);
            büyütmek animasyonu yavaşlatır (ör. 800–1200).
        use_smart_parking: True ise CSV tabanlı SmartParkingEnv + uyumlu PPO dener.
        grid_mode: train veya demo — GridParkingEnv modu.
        step_debug_log: True ise her adımda konsola ACTION/VALID/OK/reward yazdırılır.
        show_preview: False ise sadece dosya yazılır (plt.show atlanır, CI / başsız ortam).
        match_train_mdp: True ise rl_model ile aynı ödül (TRAIN_PPO_ENV_KWARGS); train modunda
            varsayılan olarak zaten açılır (plain_train_reward ile kapatılır).
        log_episode_summary: Bölüm sonunda osilasyon / benzersiz hücre / aksiyon dağılımı yazdır.
        plain_train_reward: True ise train modunda bile osilasyon/tekrar cezası olmadan eski ödül.
    """
    env: Any
    policy: Optional[Any]
    mode: str

    use_train_mdp = (not plain_train_reward) and (
        match_train_mdp or str(grid_mode).lower() == "train"
    )
    if str(grid_mode).lower() == "train" and use_train_mdp:
        print(
            "[rl_animation] Train ödülü: TRAIN_PPO_ENV_KWARGS (osilasyon + tekrar ziyaret cezası, "
            "güçlü shaping) — kapatmak için --plain-train-reward"
        )

    if use_smart_parking:
        env, policy, mode = _try_load_smart_stack()
        if env is None:
            print("[rl_animation] SmartParking yok; grid yığınına düşülüyor.")
            env, policy, mode = _try_load_grid_stack(
                grid_mode=grid_mode,
                step_debug_log=step_debug_log,
                match_train_mdp=use_train_mdp,
            )
    else:
        if str(ENV_TYPE).lower().strip() == "external":
            env, policy, mode = _try_load_external_stack()
        else:
            env, policy, mode = _try_load_grid_stack(
                grid_mode=grid_mode,
                step_debug_log=step_debug_log,
                match_train_mdp=use_train_mdp,
            )

    print(f"[rl_animation] Animasyon modu: {mode}, seed={episode_seed}")

    trajectory = run_episode(
        env,
        policy,
        seed=episode_seed,
        max_steps=220,
        mode=mode,
        log_episode_summary=log_episode_summary or step_debug_log,
    )

    fig, ax = plt.subplots(figsize=(8.0, 8.0))
    draw_frame(ax, trajectory[0])

    out_path = _save_animation(fig, trajectory, fps=fps)

    if not show_preview:
        print("[rl_animation] Önizleme atlandı (--no-show). Çıktı:", out_path)
        return out_path

    preview_interval = max(120, int(preview_ms_per_frame))
    print(
        f"[rl_animation] Önizleme hızı: ~{preview_interval} ms/kare "
        f"(daha yavaş için generate_animation(preview_ms_per_frame=900) gibi artırın)."
    )

    # Gösterim (kaydedilen fig kapatıldı; önizleme için yeniden çiz)
    preview_fig, preview_ax = plt.subplots(figsize=(8.0, 8.0))

    def _preview_update(i: int) -> None:
        if i < len(trajectory):
            draw_frame(preview_ax, trajectory[i])

    preview_anim = mplanim.FuncAnimation(
        preview_fig,
        _preview_update,
        frames=max(1, len(trajectory)),
        interval=preview_interval,
        blit=False,
    )
    preview_fig._rl_animation_ref = preview_anim  # GC ile animasyonun silinmesini önle
    # show() animasyonu ekranda tutar
    print("[rl_animation] Pencere açılıyor (plt.show) — kapatınca script sonlanır.")
    plt.show()
    plt.close(preview_fig)

    return out_path


if __name__ == "__main__":
    print("Animasyon başlatılıyor...")
    parser = argparse.ArgumentParser(description="Akıllı Park grid RL animasyonu")
    parser.add_argument(
        "--mode",
        choices=("train", "demo"),
        default="demo",
        help="GridParkingEnv modu: train=statik hedef/grid, demo=dinamik doluluk",
    )
    parser.add_argument("--seed", type=int, default=42, help="Episode tohumu")
    parser.add_argument(
        "--legacy-smart",
        action="store_true",
        help="CSV SmartParkingEnv + PPO yolunu dene (grid modeliyle uyumsuz olabilir)",
    )
    parser.add_argument(
        "--step-debug-log",
        action="store_true",
        help="Grid ortamında her adım: state, action, mask, reward, osilasyon bayrağı",
    )
    parser.add_argument(
        "--train-mdp",
        action="store_true",
        help="Demo modunda bile rl_model ödül ayarlarını kullan (nadiren gerekir)",
    )
    parser.add_argument(
        "--plain-train-reward",
        action="store_true",
        help="Train modunda eski ödül (osilasyon/tekrar cezası yok); varsayılan train’de cezalar açık",
    )
    parser.add_argument(
        "--episode-summary",
        action="store_true",
        help="Bölüm bitince episode_summary (osilasyon sayısı, benzersiz hücre, aksiyon dağılımı)",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="MP4/GIF kaydettikten sonra plt.show açma (başsız / otomasyon)",
    )
    args = parser.parse_args()
    try:
        generate_animation(
            episode_seed=args.seed,
            grid_mode=args.mode,
            use_smart_parking=bool(args.legacy_smart),
            step_debug_log=bool(args.step_debug_log),
            show_preview=not bool(args.no_show),
            match_train_mdp=bool(args.train_mdp),
            log_episode_summary=bool(args.episode_summary),
            plain_train_reward=bool(args.plain_train_reward),
        )
    except Exception as exc:
        print(f"[rl_animation] KRİTİK HATA: {exc}")
        _ensure_anim_dir()
        # Mutlaka bir çıktı
        emergency = _ANIM_DIR / "training_animation_error_note.txt"
        emergency.write_text(
            f"Animasyon üretilemedi.\nSebep: {exc!r}\n",
            encoding="utf-8",
        )
        print(f"[rl_animation] Hata notu yazıldı: {emergency}")
        raise
    print("Animasyon tamamlandı!")
