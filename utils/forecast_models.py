"""
Künye (BLM4502) bölüm 6.1 — zaman serisi tahmin mimarileri: LSTM, GRU, Temporal Transformer.

Hepsi aynı giriş şeklini kullanır: (batch, seq_len, input_dim) → skaler occupancy_rate_mm tahmini.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class OccupancyLSTM(nn.Module):
    """Çok katmanlı LSTM + dropout + doğrusal başlık."""

    def __init__(
        self,
        input_dim: int,
        hidden: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.rnn_type = "lstm"
        self.lstm = nn.LSTM(
            input_dim,
            hidden,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, _ = self.lstm(x)
        last = self.dropout(y[:, -1, :])
        return self.head(last).squeeze(-1)


class OccupancyGRU(nn.Module):
    """Çok katmanlı GRU + dropout + doğrusal başlık."""

    def __init__(
        self,
        input_dim: int,
        hidden: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.rnn_type = "gru"
        self.gru = nn.GRU(
            input_dim,
            hidden,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, _ = self.gru(x)
        last = self.dropout(y[:, -1, :])
        return self.head(last).squeeze(-1)


class OccupancyTemporalTransformer(nn.Module):
    """
    Encoder-only Transformer; künyede belirtilen positional encoding (öğrenilebilir embedding).
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.2,
        max_seq_len: int = 128,
        dim_feedforward: int | None = None,
    ):
        super().__init__()
        self.rnn_type = "transformer"
        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) nhead ({nhead}) ile bölünmeli")
        df = dim_feedforward or 4 * d_model
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=df,
            dropout=dropout,
            batch_first=True,
            norm_first=False,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        h = self.input_proj(x) + self.pos_embedding[:, :t, :]
        h = self.encoder(h)
        last = self.dropout(h[:, -1, :])
        return self.head(last).squeeze(-1)


def build_forecast_model(
    cell_type: str,
    input_dim: int,
    hp: dict[str, Any],
) -> nn.Module:
    """cell_type: lstm | gru | transformer"""
    ct = cell_type.strip().lower()
    if ct == "lstm":
        return OccupancyLSTM(
            input_dim,
            hidden=int(hp.get("hidden", 64)),
            num_layers=int(hp.get("num_layers", 2)),
            dropout=float(hp.get("dropout", 0.2)),
        )
    if ct == "gru":
        return OccupancyGRU(
            input_dim,
            hidden=int(hp.get("hidden", 64)),
            num_layers=int(hp.get("num_layers", 2)),
            dropout=float(hp.get("dropout", 0.2)),
        )
    if ct == "transformer":
        return OccupancyTemporalTransformer(
            input_dim,
            d_model=int(hp.get("d_model", 64)),
            nhead=int(hp.get("nhead", 4)),
            num_layers=int(hp.get("num_layers", 2)),
            dropout=float(hp.get("dropout", 0.2)),
            max_seq_len=int(hp.get("max_seq_len", 128)),
            dim_feedforward=hp.get("dim_feedforward"),
        )
    raise ValueError(f"Bilinmeyen cell_type: {cell_type!r}")


def load_model_from_checkpoint(ckpt: dict, input_dim: int | None = None) -> nn.Module:
    """Kayıtlı state_dict ile model kurar (API ve inference)."""
    mm = ckpt["input_cols"]
    dim = input_dim if input_dim is not None else len(mm)
    cell = str(ckpt.get("cell_type", "lstm")).lower()
    hp = {
        "hidden": ckpt.get("hidden", 64),
        "num_layers": ckpt.get("num_layers", 2),
        "dropout": ckpt.get("dropout", 0.2),
        "d_model": ckpt.get("d_model", 64),
        "nhead": ckpt.get("nhead", 4),
        "dim_feedforward": ckpt.get("dim_feedforward"),
        "max_seq_len": ckpt.get("max_seq_len", 128),
    }
    m = build_forecast_model(cell, dim, hp)
    m.load_state_dict(ckpt["state_dict"])
    m.eval()
    return m
