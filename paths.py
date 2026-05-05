"""
Proje kök dizini ve standart klasör yolları.

Tüm scriptler Path nesnesi ile çalışsın diye tek yerde tanımlanır;
CI veya farklı çalışma dizininde de göreli yollar kırılmasın.
"""

from __future__ import annotations

from pathlib import Path

# Repo kökü (bu dosyanın bulunduğu dizin)
PROJECT_ROOT: Path = Path(__file__).resolve().parent

DATA_RAW: Path = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED: Path = PROJECT_ROOT / "data" / "processed"
MODELS_DIR: Path = PROJECT_ROOT / "models"
OUTPUT_DIR: Path = PROJECT_ROOT / "output"
# RL eğitim çıktıları (Monitor + SB3 progress.csv); rl_visualizer burayı da arar.
LOGS_DIR: Path = PROJECT_ROOT / "logs"


def ensure_logs_dir() -> None:
    """RL ve diğer script logları için klasör."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)


def ensure_data_processed() -> None:
    """İşlenmiş veri çıktı klasörünü oluşturur."""
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)


def ensure_models() -> None:
    """Model kayıt klasörünü oluşturur."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


def ensure_output() -> None:
    """EDA ve grafik çıktıları için klasör."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def ensure_all_standard_dirs() -> None:
    """Pipeline’ın ihtiyaç duyduğu tüm klasörleri tek seferde hazırlar."""
    ensure_data_processed()
    ensure_models()
    ensure_output()
    ensure_logs_dir()
