"""
RL eğitim metriklerinin akademik kalitede görselleştirilmesi.

CSV ve NumPy çıktılarından grafik üretir; zorunlu dosya eksiklerinde uyarı verir.
"""

from __future__ import annotations

import io
import logging
import warnings
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from paths import PROJECT_ROOT

_LOGGER = logging.getLogger(__name__)

# Varsayılan arama dizinleri (log / çıktı / kök)
_DEFAULT_SEARCH_DIRS: Tuple[Path, ...] = (
    PROJECT_ROOT / "logs",
    PROJECT_ROOT / "output",
    PROJECT_ROOT,
)

_FIGSIZE: Tuple[float, float] = (10.0, 6.0)
_DPI: int = 300
_PLOTS_DIR: Path = PROJECT_ROOT / "plots"


def _ensure_plots_dir() -> Path:
    """plots klasörünü oluşturur ve yolunu döndürür."""
    _PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    return _PLOTS_DIR


def _find_file(candidates: Sequence[str], search_dirs: Iterable[Path]) -> Optional[Path]:
    """Verilen aday dosya adlarından ilkinin var olan yolunu bulur."""
    for base in search_dirs:
        for name in candidates:
            p = base / name
            if p.is_file():
                return p
    return None


def _find_monitor_csv(search_dirs: Iterable[Path]) -> Optional[Path]:
    """SB3 Monitor çıktısı (*monitor.csv) dosyalarından ilki."""
    for base in search_dirs:
        if not base.is_dir():
            continue
        found = sorted(base.glob("*monitor.csv"))
        if found:
            return found[0]
    return None


def _read_monitor_episodes(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    train.monitor.csv: # ile başlayan meta satırı atlanır; sütunlar r, l, t.

    Returns:
        episode_index, rewards, lengths
    """
    raw = path.read_text(encoding="utf-8", errors="replace")
    df = pd.read_csv(io.StringIO(raw), comment="#")
    if "r" not in df.columns or "l" not in df.columns:
        raise KeyError("Monitor CSV r/l sütunları bekleniyor")
    n = len(df)
    return np.arange(n, dtype=float), df["r"].to_numpy(dtype=float), df["l"].to_numpy(dtype=float)


def _read_csv_flexible(path: Path) -> pd.DataFrame:
    """CSV dosyasını okur; UTF-8 ve latin-1 ile dener."""
    try:
        return pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1")


def _pick_xy_columns(
    df: pd.DataFrame, x_names: Tuple[str, ...], y_names: Tuple[str, ...]
) -> Tuple[str, str]:
    """Sütun adlarını küçük harfe indirger ve x/y sütunlarını seçer."""
    lower = {c.lower(): c for c in df.columns}
    x_col = None
    for name in x_names:
        if name.lower() in lower:
            x_col = lower[name.lower()]
            break
    y_col = None
    for name in y_names:
        if name.lower() in lower:
            y_col = lower[name.lower()]
            break
    if x_col is None or y_col is None:
        raise KeyError(f"Beklenen sütunlar bulunamadı: x={x_names}, y={y_names}")
    return x_col, y_col


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    """Basit hareketli ortalama (kenarlar kısaltılır)."""
    if window <= 1:
        return values.astype(float)
    window = min(window, len(values))
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(values.astype(float), kernel, mode="valid")


def plot_reward_trend(
    search_dirs: Optional[Sequence[Path]] = None,
    window: int = 10,
    out_path: Optional[Path] = None,
) -> bool:
    """
    Episode ödül eğilimini ve hareketli ortalamayı çizer.

    Args:
        search_dirs: CSV aranacak dizinler (None ise varsayılan).
        window: Hareketli ortalama penceresi.
        out_path: PNG çıktı yolu (None ise plots/reward_trend.png).

    Returns:
        Grafik üretildiyse True, veri yoksa False.
    """
    dirs = tuple(search_dirs) if search_dirs is not None else _DEFAULT_SEARCH_DIRS
    csv_path = _find_file(("rewards.csv", "Rewards.csv"), dirs)
    monitor_path = _find_monitor_csv(dirs) if csv_path is None else None
    plot_dir = _ensure_plots_dir()
    target = out_path or (plot_dir / "reward_trend.png")

    if csv_path is None and monitor_path is None:
        warnings.warn(
            "rewards.csv ve *monitor.csv bulunamadı; ödül eğilimi grafiği atlandı.",
            UserWarning,
            stacklevel=2,
        )
        return False

    if csv_path is not None:
        df = _read_csv_flexible(csv_path)
        try:
            ep_col, r_col = _pick_xy_columns(df, ("episode", "ep"), ("reward", "return", "score"))
        except KeyError as exc:
            warnings.warn(f"rewards.csv sütun hatası ({exc}); grafik atlandı.", UserWarning, stacklevel=2)
            return False
        episodes = df[ep_col].to_numpy(dtype=float)
        rewards = df[r_col].to_numpy(dtype=float)
    else:
        assert monitor_path is not None
        try:
            episodes, rewards, _ = _read_monitor_episodes(monitor_path)
        except (KeyError, OSError, UnicodeDecodeError, pd.errors.ParserError) as exc:
            warnings.warn(f"Monitor CSV okunamadı ({exc}); grafik atlandı.", UserWarning, stacklevel=2)
            return False
    ma = _moving_average(rewards, window)
    win_eff = 1 if window <= 1 else min(window, len(rewards))
    ma_episodes = episodes[win_eff - 1 : win_eff - 1 + len(ma)]

    fig, ax = plt.subplots(figsize=_FIGSIZE, dpi=_DPI)
    ax.plot(episodes, rewards, color="#4c72b0", alpha=0.35, linewidth=1.0, label="Episode reward")
    ax.plot(
        ma_episodes,
        ma,
        color="#c44e52",
        linewidth=2.0,
        label=f"Moving average (w={window})",
    )
    ax.set_title("Episode Reward Trend During Reinforcement Learning")
    ax.set_xlabel("Episode index")
    ax.set_ylabel("Cumulative reward per episode")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.legend(frameon=True, fancybox=False, edgecolor="0.6")
    fig.tight_layout()
    fig.savefig(target, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_episode_length_trend(
    search_dirs: Optional[Sequence[Path]] = None,
    out_path: Optional[Path] = None,
) -> bool:
    """Episode uzunluklarının zamana göre dağılımını çizer."""
    dirs = tuple(search_dirs) if search_dirs is not None else _DEFAULT_SEARCH_DIRS
    csv_path = _find_file(("lengths.csv", "Lengths.csv"), dirs)
    monitor_path = _find_monitor_csv(dirs) if csv_path is None else None
    plot_dir = _ensure_plots_dir()
    target = out_path or (plot_dir / "episode_length.png")

    if csv_path is None and monitor_path is None:
        warnings.warn(
            "lengths.csv ve *monitor.csv bulunamadı; episode length grafiği atlandı.",
            UserWarning,
            stacklevel=2,
        )
        return False

    if csv_path is not None:
        df = _read_csv_flexible(csv_path)
        try:
            ep_col, len_col = _pick_xy_columns(
                df, ("episode", "ep"), ("length", "steps", "episode_length")
            )
        except KeyError as exc:
            warnings.warn(f"lengths.csv sütun hatası ({exc}); grafik atlandı.", UserWarning, stacklevel=2)
            return False
        ep_vals = df[ep_col].to_numpy()
        len_vals = df[len_col].to_numpy()
    else:
        assert monitor_path is not None
        try:
            ep_vals, _, len_vals = _read_monitor_episodes(monitor_path)
        except (KeyError, OSError, UnicodeDecodeError, pd.errors.ParserError) as exc:
            warnings.warn(f"Monitor CSV okunamadı ({exc}); grafik atlandı.", UserWarning, stacklevel=2)
            return False

    fig, ax = plt.subplots(figsize=_FIGSIZE, dpi=_DPI)
    ax.plot(
        ep_vals,
        len_vals,
        color="#55a868",
        linewidth=1.4,
        marker="o",
        markersize=2.5,
        alpha=0.85,
    )
    ax.set_title("Episode Length Trend")
    ax.set_xlabel("Episode index")
    ax.set_ylabel("Steps until termination")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    fig.tight_layout()
    fig.savefig(target, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_loss_curve(
    search_dirs: Optional[Sequence[Path]] = None,
    out_path: Optional[Path] = None,
) -> bool:
    """
    PPO (policy + value) veya DQN (tek loss) eğrilerini tek figürde gösterir.

    Öncelik: policy_loss.csv + value_loss.csv varsa PPO; aksi halde loss.csv.
    """
    dirs = tuple(search_dirs) if search_dirs is not None else _DEFAULT_SEARCH_DIRS
    plot_dir = _ensure_plots_dir()
    target = out_path or (plot_dir / "loss_curve.png")

    p_policy = _find_file(("policy_loss.csv",), dirs)
    p_value = _find_file(("value_loss.csv",), dirs)
    p_loss = _find_file(("loss.csv", "Loss.csv"), dirs)
    p_progress = _find_file(("progress.csv",), dirs)

    fig, ax = plt.subplots(figsize=_FIGSIZE, dpi=_DPI)
    plotted = False

    if p_policy is not None and p_value is not None:
        df_p = _read_csv_flexible(p_policy)
        df_v = _read_csv_flexible(p_value)
        try:
            x_p, y_p = _pick_xy_columns(
                df_p, ("timestep", "step", "update", "iteration"), ("loss", "policy_loss")
            )
            x_v, y_v = _pick_xy_columns(
                df_v, ("timestep", "step", "update", "iteration"), ("loss", "value_loss")
            )
        except KeyError as exc:
            warnings.warn(f"PPO loss CSV sütun hatası ({exc}); grafik atlandı.", UserWarning, stacklevel=2)
            plt.close(fig)
            return False
        ax.plot(df_p[x_p], df_p[y_p], color="#8172b3", linewidth=1.6, label="Policy loss")
        ax.plot(df_v[x_v], df_v[y_v], color="#ccb974", linewidth=1.6, label="Value loss")
        ax.set_title("Optimization Loss Curves (Proximal Policy Optimization)")
        plotted = True
    elif p_loss is not None:
        df = _read_csv_flexible(p_loss)
        try:
            x_c, y_c = _pick_xy_columns(
                df, ("timestep", "step", "update", "iteration", "episode"), ("loss", "td_error")
            )
        except KeyError as exc:
            warnings.warn(f"loss.csv sütun hatası ({exc}); grafik atlandı.", UserWarning, stacklevel=2)
            plt.close(fig)
            return False
        ax.plot(df[x_c], df[y_c], color="#64b5cd", linewidth=1.6, label="Training loss")
        ax.set_title("Optimization Loss Curve (Value-Based RL)")
        plotted = True
    elif p_progress is not None:
        df = _read_csv_flexible(p_progress)
        lower = {c.lower(): c for c in df.columns}

        def _col_exact(canonical: str) -> Optional[str]:
            return lower.get(canonical.lower())

        x_col = _col_exact("time/iterations") or _col_exact("time/total_timesteps")
        pg_col = _col_exact("train/policy_gradient_loss")
        vl_col = _col_exact("train/value_loss")
        if x_col and pg_col and vl_col:
            ax.plot(df[x_col], df[pg_col], color="#8172b3", linewidth=1.6, label="Policy loss")
            ax.plot(df[x_col], df[vl_col], color="#ccb974", linewidth=1.6, label="Value loss")
            ax.set_title("Optimization Loss Curves (Proximal Policy Optimization)")
            plotted = True
        else:
            loss_col = _col_exact("train/loss")
            if x_col and loss_col:
                ax.plot(df[x_col], df[loss_col], color="#64b5cd", linewidth=1.6, label="Training loss")
                ax.set_title("Optimization Loss Curve (PPO, combined)")
                plotted = True
        if not plotted:
            warnings.warn(
                "progress.csv var fakat beklenen sütunlar yok; loss grafiği atlandı.",
                UserWarning,
                stacklevel=2,
            )
            plt.close(fig)
            return False
    else:
        warnings.warn(
            "policy_loss/value_loss, loss.csv veya progress.csv bulunamadı; loss grafiği atlandı.",
            UserWarning,
            stacklevel=2,
        )
        plt.close(fig)
        return False

    ax.set_xlabel("Training step (or logged index)")
    ax.set_ylabel("Loss")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.legend(frameon=True, fancybox=False, edgecolor="0.6")
    fig.tight_layout()
    fig.savefig(target, bbox_inches="tight")
    plt.close(fig)
    return plotted


def plot_action_distribution(
    search_dirs: Optional[Sequence[Path]] = None,
    out_path: Optional[Path] = None,
) -> bool:
    """Aksiyon kimliklerinin frekans dağılımını çubuk grafik olarak gösterir."""
    dirs = tuple(search_dirs) if search_dirs is not None else _DEFAULT_SEARCH_DIRS
    csv_path = _find_file(("actions.csv", "Actions.csv"), dirs)
    plot_dir = _ensure_plots_dir()
    target = out_path or (plot_dir / "action_distribution.png")

    if csv_path is None:
        _LOGGER.debug("actions.csv yok; aksiyon dağılımı atlandı (PPO sonrası rl_model ile üretilir).")
        return False

    df = _read_csv_flexible(csv_path)
    lower = {c.lower(): c for c in df.columns}
    if "action" in lower and "count" in lower:
        ac, cc = lower["action"], lower["count"]
        unique = np.rint(df[ac].to_numpy(dtype=float)).astype(int)
        counts = df[cc].to_numpy(dtype=float)
        order = np.argsort(unique)
        unique = unique[order]
        counts = counts[order]
    else:
        try:
            _, a_col = _pick_xy_columns(df, ("episode", "step", "timestep"), ("action", "a", "act"))
        except KeyError:
            if "action" in df.columns:
                a_col = "action"
            elif len(df.columns) >= 2:
                a_col = df.columns[-1]
            else:
                warnings.warn("actions.csv uygun aksiyon sütunu içermiyor.", UserWarning, stacklevel=2)
                return False

        actions = df[a_col].to_numpy()
        actions_int = np.rint(actions).astype(int)
        unique, counts = np.unique(actions_int, return_counts=True)
        counts = counts.astype(float)

    fig, ax = plt.subplots(figsize=_FIGSIZE, dpi=_DPI)
    ax.bar(unique.astype(float), counts, color="#dd8452", edgecolor="0.2", linewidth=0.4)
    ax.set_title("Empirical Action Distribution")
    ax.set_xlabel("Discrete action index")
    ax.set_ylabel("Frequency (count)")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    fig.tight_layout()
    fig.savefig(target, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_value_heatmap(
    search_dirs: Optional[Sequence[Path]] = None,
    out_path: Optional[Path] = None,
) -> bool:
    """value_map.npy üzerinden durum-değer ısı haritası üretir."""
    dirs = tuple(search_dirs) if search_dirs is not None else _DEFAULT_SEARCH_DIRS
    npy_path = _find_file(("value_map.npy",), dirs)
    plot_dir = _ensure_plots_dir()
    target = out_path or (plot_dir / "value_heatmap.png")

    if npy_path is None:
        _LOGGER.debug("value_map.npy yok; ısı haritası atlandı (tabanlı RL çıktısı, PPO ile üretilmez).")
        return False

    arr = np.load(npy_path)
    if arr.ndim != 2:
        warnings.warn(
            f"value_map.npy beklenen 2B dizi değil (shape={arr.shape}).",
            UserWarning,
            stacklevel=2,
        )
        return False

    fig, ax = plt.subplots(figsize=_FIGSIZE, dpi=_DPI)
    im = ax.imshow(arr, origin="lower", cmap="viridis", aspect="auto")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Estimated state value")
    ax.set_title("State–Value Function over Discretized Spatial Grid")
    ax.set_xlabel("Grid column index")
    ax.set_ylabel("Grid row index")
    fig.tight_layout()
    fig.savefig(target, bbox_inches="tight")
    plt.close(fig)
    return True


def visualize_all(
    search_dirs: Optional[Sequence[Path]] = None,
) -> List[str]:
    """
    Tüm standart RL görselleştirmelerini sırayla üretir.

    Returns:
        Üretilen çıktı dosya yollarının listesi (başarılı olanlar).
    """
    _ensure_plots_dir()
    produced: List[str] = []
    if plot_reward_trend(search_dirs=search_dirs):
        produced.append(str(_PLOTS_DIR / "reward_trend.png"))
    if plot_episode_length_trend(search_dirs=search_dirs):
        produced.append(str(_PLOTS_DIR / "episode_length.png"))
    if plot_loss_curve(search_dirs=search_dirs):
        produced.append(str(_PLOTS_DIR / "loss_curve.png"))
    if plot_action_distribution(search_dirs=search_dirs):
        produced.append(str(_PLOTS_DIR / "action_distribution.png"))
    if plot_value_heatmap(search_dirs=search_dirs):
        produced.append(str(_PLOTS_DIR / "value_heatmap.png"))
    return produced


if __name__ == "__main__":
    produced = visualize_all()
    if produced:
        print(f"PNG çıktıları ({len(produced)} adet), proje altında plots/:")
        for p in produced:
            print(f"  {p}")
    else:
        print("Grafik üretilmedi; logs/ altında train.monitor.csv, progress.csv vb. gerekir.")
