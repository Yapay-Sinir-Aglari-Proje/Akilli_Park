"""
Tek tip rastgelelik: Python, NumPy, (varsa) PyTorch ve PYTHONHASHSEED.

Aynı seed ile `train_lstm`, `train_rl` ve değerlendirme scriptleri tekrarlanabilir olur.
"""

from __future__ import annotations

import os
import random

import numpy as np

from ml_config import RANDOM_SEED


def set_global_seed(seed: int = RANDOM_SEED) -> None:
    """Tüm ilgili kütüphanelerde rastgele tohumu ayarlar (CUDA varsa GPU tohumu dahil)."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
