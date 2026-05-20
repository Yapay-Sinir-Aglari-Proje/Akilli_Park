"""
Izgara navigasyon RL için ödül bileşenleri (Akıllı Park).

Bu modül, `GridParkingNavigationEnv.step` içindeki ödül mantığını tek yerde toplar;
her bileşen raporlanabilir ve README / sunumda açıklanabilir.

Bileşenler (mantık özeti):
- Zaman cezası: her geçerli adımda `-step_cost` (ajanı aceleye iter).
- Mesafe şekillendirme: hedefe yaklaşınca pozitif, uzaklaşınca negatif
  (`manhattan_scale * (old_dist - new_dist)`).
- Engel / duvar çarpışması: duvara veya sınır dışına temasta ek `-wall_penalty`
  (dolmuş park alanı duvar değil; model için önemli engel tipi duvar/harita sınırıdır).
- Geçersiz eylem indeksi: yalnızca küçük adım maliyeti (policy hatası vekili).
- İlk ziyaret bonusu / tekrar ziyaret cezası: gereksiz dolaşımı azaltır.
- A↔B salınım (zigzag) cezası: kısa pencerede osilasyon tespiti.
- Hedef ödülü: terminalde büyük pozitif `goal_bonus`.
- Süre aşımı: `timeout_penalty` (truncated).
- Kısa yol bonusu: BFS ile bulunan en kısa yol uzunluğuna göre verimlilik ödülü.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass
class RewardParts:
    """Adım bazlı ödül kırılımı (loglama / hata ayıklama)."""

    time_penalty: float
    distance_shaping: float
    visit_adjustment: float
    loop_penalty: float
    goal_bonus: float
    timeout_penalty: float
    path_efficiency_bonus: float
    wall_collision_extra: float
    clipped_total: float


def clip_reward(value: float, reward_clip: float) -> float:
    """Öğrenme stabilitesi için ödül kırpma (`GRID_REWARD_CLIP`)."""
    lo, hi = -float(reward_clip), float(reward_clip)
    if value < lo:
        return float(lo)
    if value > hi:
        return float(hi)
    return float(value)


def compute_goal_path_efficiency_bonus(
    *,
    success: bool,
    shortest_path_len: int,
    path_edges: int,
    scale: float,
    reward_clip: float,
) -> float:
    """
    Başarılı bölümde: gerçek rota uzunluğu (kenar sayısı) ile ideal en kısa yol arasındaki
    verimliliğe küçük ek ödül.

    Formül (basit ve açıklanabilir): bonus = scale * shortest / max(actual_edges, shortest)
    - Tam optimalda ~ `scale`.
    - Dolambaçlı rotada oran düşer.
    """
    if not success or shortest_path_len <= 0 or scale <= 0:
        return 0.0
    actual = max(int(path_edges), int(shortest_path_len))
    ratio = float(shortest_path_len) / float(actual)
    return clip_reward(float(scale) * ratio, reward_clip)


def compute_grid_step_reward(
    *,
    step_cost: float,
    manhattan_scale: float,
    old_dist: int,
    new_dist: int,
    was_revisit: bool,
    first_visit_bonus: float,
    revisit_penalty: float,
    loop_hit: bool,
    loop_penalty: float,
    goal_bonus: float,
    timeout_penalty: float,
    terminated: bool,
    truncated: bool,
    reward_clip: float,
    shortest_path_len: int,
    path_edges: int,
    path_efficiency_scale: float,
) -> Tuple[float, RewardParts]:
    """
    Grid üzerinde **geçerli hücreye yapılan bir adım** için ödül.

    Duvar/sınır çarpışması ve geçersiz aksiyon indeksi ortam tarafında ayrı ele alınır
    (`-step_cost` ve isteğe bağlı `-wall_penalty`).
    """
    time_penalty = -float(step_cost)
    distance_shaping = float(manhattan_scale) * float(old_dist - new_dist)
    visit_adjustment = float(revisit_penalty if was_revisit else first_visit_bonus)
    loop_p = float(loop_penalty) if loop_hit else 0.0
    g_bonus = 0.0
    t_pen = 0.0
    path_eff_bonus = 0.0

    total = time_penalty + distance_shaping + visit_adjustment + loop_p

    if terminated:
        g_bonus = float(goal_bonus)
        total += g_bonus
        path_eff_bonus = compute_goal_path_efficiency_bonus(
            success=True,
            shortest_path_len=int(shortest_path_len),
            path_edges=int(path_edges),
            scale=float(path_efficiency_scale),
            reward_clip=float(reward_clip),
        )
        total += path_eff_bonus

    if truncated and not terminated:
        t_pen = float(timeout_penalty)
        total += t_pen

    clipped = clip_reward(total, reward_clip)
    parts = RewardParts(
        time_penalty=time_penalty,
        distance_shaping=distance_shaping,
        visit_adjustment=visit_adjustment,
        loop_penalty=loop_p,
        goal_bonus=g_bonus,
        timeout_penalty=t_pen,
        path_efficiency_bonus=path_eff_bonus,
        wall_collision_extra=0.0,
        clipped_total=clipped,
    )
    return clipped, parts
