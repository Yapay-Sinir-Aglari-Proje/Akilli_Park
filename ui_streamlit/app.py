"""
Streamlit kontrol paneli: LSTM, RL önerisi, KPI.
Çalıştırma: streamlit run ui_streamlit/app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from evaluation.visualize import plot_prediction_vs_actual
from ml_config import RANDOM_SEED
from paths import EVALUATION_DIR, PREDICTIONS_DIR

st.set_page_config(page_title="Akıllı Otopark", layout="wide")
st.title("Akıllı Otopark — LSTM + RL")

tab_ts, tab_rl, tab_kpi = st.tabs(
    ["Zaman Serisi Tahmini", "RL Simülasyon / Öneri", "KPI"]
)

with tab_ts:
    st.subheader("LSTM aggregate doluluk tahmini")
    pred_path = PREDICTIONS_DIR / "test_predictions.csv"
    if pred_path.exists():
        df = pd.read_csv(pred_path)
        st.line_chart(
            df.rename(
                columns={
                    "y_true_occupancy_rate": "Gerçek",
                    "y_pred_occupancy_rate": "Tahmin",
                }
            )[["Gerçek", "Tahmin"]]
            .head(500)
        )
        out = plot_prediction_vs_actual()
        st.caption(f"Kaydedilen grafik: `{out}`")
    else:
        st.warning("Önce `python train_lstm.py` çalıştırın.")

with tab_rl:
    st.subheader("RL politika önerisi (PPO)")
    st.write(
        "Grid navigasyon GIF (PPO): `python -m evaluation.record_parking_gif` → `output/parking_agent.gif`"
    )
    st.write("Alternatif: `python -m evaluation.animate_rl_rollout` → `output/rl_rollout.gif`")
    st.write("API: `POST /act` (önce `uvicorn api.main:app --app-dir .`)")
    if st.button("evaluate.py --part rl çalıştır (terminal)"):
        st.info("Terminalde: `python evaluate.py --part rl --rl-algo ppo`")

with tab_kpi:
    st.subheader("Metrikler")
    st.write(f"Sabit seed: **{RANDOM_SEED}**")
    rep = EVALUATION_DIR / "metrics.json"
    if rep.exists():
        st.json(json.loads(rep.read_text(encoding="utf-8")))
    else:
        st.info("`python evaluate.py` ile metrics.json üretin.")
    pred_path = PREDICTIONS_DIR / "test_predictions.csv"
    if pred_path.exists():
        df = pd.read_csv(pred_path)
        err = (df["y_pred_occupancy_rate"] - df["y_true_occupancy_rate"]).abs().mean()
        st.metric("Ortalama mutlak hata (LSTM)", f"{err:.4f}")
        congestion = float(df["y_pred_occupancy_rate"].mean())
        st.metric("Tahmini ortalama doluluk (yoğunluk skoru)", f"{congestion:.3f}")
