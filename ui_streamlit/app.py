"""
Akıllı Park — proje künyesi (BLM4502) ile uyumlu Streamlit kontrol paneli.

Modüller:
1) Zaman serisi tahmin ve model performansı
2) Keşifsel veri analizi (EDA çıktıları)
3) Sistem durumu ve özet göstergeler
4) Pekiştirmeli öğrenme simülasyonu / şeffaflık
5) Karar destek ve otopark önerisi (yeşil / sarı / kırmızı doluluk kodu)

Çalıştırma: streamlit run ui_streamlit/app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from evaluation.visualize import plot_prediction_vs_actual
from ml_config import GRID_HEIGHT, GRID_WIDTH, RANDOM_SEED
from paths import DATA_PROCESSED, EVALUATION_DIR, OUTPUT_DIR, PREDICTIONS_DIR

st.set_page_config(page_title="Akıllı Park", layout="wide")
st.title("Akıllı Park — Hibrit tahmin + RL")
st.caption(
    "Parking Birmingham · LSTM aggregate tahmin · Grid PPO/DQN · "
    "TRAKYA ÜNİVERSİTESİ BLM4502 proje künyesi ile uyumlu mimari"
)

tab_ts, tab_eda, tab_dash, tab_rl, tab_decision = st.tabs(
    [
        "1) Zaman serisi tahmin & performans",
        "2) Keşifsel veri analizi (EDA)",
        "3) Sistem özeti (dashboard)",
        "4) RL simülasyon & görselleştirme",
        "5) Karar destek & öneri",
    ]
)


def _occupancy_level(rate: float) -> tuple[str, str]:
    if rate < 0.33:
        return "Düşük (yeşil)", "#2e7d32"
    if rate < 0.66:
        return "Orta (sarı)", "#f9a825"
    return "Yüksek (kırmızı)", "#c62828"


with tab_ts:
    st.subheader("Zaman serisi tahmin ve model performansı")
    st.markdown(
        "Bu modülde aggregate doluluk tahmini gerçek değerlerle birlikte sunulur; "
        "künyedeki MAE, RMSE, MAPE ve R² metrikleri `evaluate.py` çıktısında toplanır. "
        "**Zaman filtresi** ve **maksimum nokta** künye 10. bölüm etkileşimi ile uyumludur."
    )
    pred_path = PREDICTIONS_DIR / "test_predictions.csv"
    if pred_path.exists():
        df = pd.read_csv(pred_path)
        df["LastUpdated"] = pd.to_datetime(df["LastUpdated"], errors="coerce")
        df = df.dropna(subset=["LastUpdated"])
        tmin, tmax = df["LastUpdated"].min(), df["LastUpdated"].max()
        c1, c2, c3 = st.columns(3)
        with c1:
            d0 = st.date_input("Başlangıç", value=tmin.date(), min_value=tmin.date(), max_value=tmax.date())
        with c2:
            d1 = st.date_input("Bitiş", value=tmax.date(), min_value=tmin.date(), max_value=tmax.date())
        with c3:
            max_pts = st.slider("Grafikte en fazla nokta", 50, 3000, 800, 50)
        m = (df["LastUpdated"] >= pd.Timestamp(d0)) & (df["LastUpdated"] <= pd.Timestamp(d1) + pd.Timedelta(days=1))
        sub = df.loc[m].sort_values("LastUpdated")
        if sub.empty:
            st.warning("Seçilen aralıkta veri yok.")
        else:
            chart_df = (
                sub.rename(
                    columns={
                        "y_true_occupancy_rate": "Gerçek doluluk oranı",
                        "y_pred_occupancy_rate": "Tahmin",
                    }
                )[["LastUpdated", "Gerçek doluluk oranı", "Tahmin"]]
                .set_index("LastUpdated")
                .head(int(max_pts))
            )
            st.line_chart(chart_df)
        out = plot_prediction_vs_actual()
        st.caption(f"Kayıtlı grafik: `{out}`")
    else:
        st.warning("Önce `python train_lstm.py` ile tahmin CSV üretin.")

    rep = EVALUATION_DIR / "metrics.json"
    if rep.exists():
        m = json.loads(rep.read_text(encoding="utf-8")).get("lstm", {})
        if m:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("MAE", f"{m.get('lstm_mae', 0):.5f}")
            c2.metric("RMSE", f"{m.get('lstm_rmse', 0):.5f}")
            c3.metric("MAPE %", f"{m.get('lstm_mape_pct', 0):.2f}")
            c4.metric("R²", f"{m.get('lstm_r2', 0):.4f}")
    else:
        st.info("`python evaluate.py --part lstm` ile metrikleri güncelleyin.")


with tab_eda:
    st.subheader("Keşifsel veri analizi")
    st.markdown(
        "Saatlik / günlük desenler, histogram ve korelasyon grafikleri `eda.py` ile "
        "`output/` klasörüne yazılır (künye iş paketi 9.2)."
    )
    patterns = sorted(OUTPUT_DIR.glob("*.png")) if OUTPUT_DIR.exists() else []
    if patterns:
        for p in patterns[:12]:
            st.image(str(p), caption=p.name, use_container_width=True)
    else:
        st.warning("Grafik yok. Çalıştırın: `python eda.py`")


with tab_dash:
    st.subheader("Sistem durumu ve özet göstergeler")
    st.metric("Sabit seed", str(RANDOM_SEED))
    st.metric("Grid gözlem boyutu (hibrit RL)", f"{5 * GRID_HEIGHT * GRID_WIDTH + 2} (5×H×W + 2 skaler)")

    pq = DATA_PROCESSED / "processed.parquet"
    if pq.exists():
        dfp = pd.read_parquet(pq)
        dfp["occ_rate"] = dfp["Occupancy"] / dfp["Capacity"]
        st.metric("İşlenmiş satır sayısı", f"{len(dfp):,}")
        st.metric("Ortalama doluluk (tüm lotlar)", f"{dfp['occ_rate'].mean():.3f}")
        last = dfp.sort_values("LastUpdated").iloc[-1]
        st.write("Son kayıt zamanı:", str(last["LastUpdated"]))
    else:
        st.warning("`data/processed/processed.parquet` bulunamadı — önce veri hattını çalıştırın.")

    rep = EVALUATION_DIR / "metrics.json"
    if rep.exists():
        st.json(json.loads(rep.read_text(encoding="utf-8")))
    else:
        st.info("`python evaluate.py` ile `evaluation/reports/metrics.json` oluşturun.")


with tab_rl:
    st.subheader("Pekiştirmeli öğrenme tabanlı yönlendirme")
    st.markdown(
        "**Durum uzayı (künye):** duvar ve boş alanlar, park alanları maskesi, hedef hücre, "
        "araç konumu, **otopark doluluk ısı haritası**, **LSTM aggregate** ve **boşalma vekili** skalerleri.\n\n"
        "**Adım adım simülasyon:** aşağıdan senaryo seçip ortamı yükleyin; aksiyon seçerek kare kare ilerleyin "
        "(künye 10.4 — şeffaflık)."
    )
    st.code(
        "python train_rl.py --algo ppo   # veya dqn / both\n"
        "python -m evaluation.record_parking_gif\n"
        "python experiments/compare_time_windows.py",
        language="bash",
    )
    st.write(
        "API: `uvicorn api.main:app --app-dir .` → `POST /act?algo=ppo|dqn` · "
        "React arayüz: `frontend/` (`npm run dev`)"
    )

    scenario = st.selectbox(
        "Senaryo (bölüm havuzu)",
        ("test (künye değerlendirme)", "train (alternatif senaryo)"),
    )
    max_eps = st.slider("Maks. bölüm sayısı (hız için sınırlı)", 30, 600, 150, 10)
    split_kw = "test" if scenario.startswith("test") else None

    if st.button("Grid ortamını yükle / sıfırla"):
        try:
            from env.grid_navigation_env import (
                GridParkingNavigationEnv,
                build_grid_nav_episode_configs,
            )
            from paths import DATA_PROCESSED, PREDICTIONS_DIR

            pred_csv = PREDICTIONS_DIR / "test_predictions.csv"
            cfgs = build_grid_nav_episode_configs(
                DATA_PROCESSED / "processed.parquet",
                pred_csv if pred_csv.exists() else None,
                split=split_kw,
                max_episodes=int(max_eps),
                base_seed=RANDOM_SEED,
            )
            st.session_state["_rl_cfgs"] = cfgs
            st.session_state["_rl_env"] = GridParkingNavigationEnv(
                cfgs,
                seed=RANDOM_SEED,
                render_mode="rgb_array",
                max_episode_steps=200,
            )
            st.session_state["_rl_obs"], _ = st.session_state["_rl_env"].reset(seed=RANDOM_SEED)
            st.session_state["_rl_log"] = []
            st.success(f"{len(cfgs)} bölüm konfigürasyonu yüklendi.")
        except Exception as e:
            st.error(str(e))

    env = st.session_state.get("_rl_env")
    if env is not None:
        act_labels = {0: "0 ↑ yukarı", 1: "1 ↓ aşağı", 2: "2 ← sol", 3: "3 → sağ"}
        a = st.selectbox("Aksiyon", [0, 1, 2, 3], format_func=lambda x: act_labels[x])
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Bir adım uygula"):
                obs, r, term, trunc, info = env.step(int(a))
                st.session_state["_rl_obs"] = obs
                st.session_state["_rl_log"].append(
                    {"reward": float(r), "terminated": bool(term), "truncated": bool(trunc), "info": {k: v for k, v in info.items() if isinstance(v, (bool, int, float))}}
                )
                if term or trunc:
                    st.info("Bölüm bitti — tekrar yüklemek için yukarıdaki butonu kullanın.")
        with col_b:
            if st.button("Günlüğü temizle"):
                st.session_state["_rl_log"] = []
        fr = env.render()
        if fr is not None:
            st.image(Image.fromarray(np.asarray(fr)), caption="Grid (doluluk renk kodlu)", use_container_width=True)
        if st.session_state.get("_rl_log"):
            st.json(st.session_state["_rl_log"][-8:])

    gif1 = ROOT / "output" / "parking_agent.gif"
    gif2 = ROOT / "output" / "rl_rollout.gif"
    if gif1.exists():
        st.image(str(gif1), caption="PPO park GIF (varsa)", use_container_width=True)
    if gif2.exists():
        st.image(str(gif2), caption="RL rollout GIF (varsa)", use_container_width=True)


with tab_decision:
    st.subheader("Karar destek ve öneri")
    st.markdown(
        "Hibrit öneri: seçilen dilimde son zaman diliminde **en düşük gözlemlenen doluluğa** sahip lot; "
        "aynı zaman damgası için LSTM tahmini varsa birlikte gösterilir."
    )
    split_dec = st.radio("Veri dilimi", ("test", "train", "val"), horizontal=True)
    if split_dec != "test":
        st.caption("`test_predictions.csv` yalnızca test dilimini içerir; LSTM satırı o dilimde eşleşmezse boş kalır.")
    pred_path = PREDICTIONS_DIR / "test_predictions.csv"
    pq = DATA_PROCESSED / "processed.parquet"
    if not pq.exists():
        st.error("Önce `utils.data_pipeline` ile `processed.parquet` oluşturun.")
    else:
        dfp = pd.read_parquet(pq)
        dfp = dfp[dfp["split"] == split_dec].copy()
        dfp["LastUpdated"] = pd.to_datetime(dfp["LastUpdated"], errors="coerce")
        dfp = dfp.dropna(subset=["LastUpdated"])
        dfp["occ_rate"] = dfp["Occupancy"] / dfp["Capacity"]
        last_t = dfp["LastUpdated"].max()
        snap = dfp[dfp["LastUpdated"] == last_t]
        if snap.empty:
            st.warning("Test diliminde kayıt yok.")
        else:
            best_idx = snap["occ_rate"].idxmin()
            row = snap.loc[best_idx]
            rate = float(row["occ_rate"])
            label, color = _occupancy_level(rate)
            st.markdown(
                f"**Önerilen otopark (SystemCodeNumber):** `{row['SystemCodeNumber']}`  \n"
                f"**Gözlemlenen doluluk oranı:** {rate:.3f}  \n"
                f"**Seviye:** <span style='color:{color};font-weight:bold'>{label}</span>",
                unsafe_allow_html=True,
            )
            lstm_txt = "—"
            if pred_path.exists():
                pr = pd.read_csv(pred_path)
                pr["LastUpdated"] = pd.to_datetime(pr["LastUpdated"], errors="coerce")
                m = pr[pr["LastUpdated"] == last_t]
                if not m.empty:
                    lstm_txt = f"{float(m['y_pred_occupancy_rate'].iloc[0]):.3f} (aggregate LSTM)"
            st.write("Aynı zaman için LSTM aggregate tahmin:", lstm_txt)
            st.caption(
                "Bu panel künyedeki ‘karar destek’ bileşeninin sadeleştirilmiş örneğidir; "
                "tam rota maliyeti RL eğitimi ve API üzerinden üretilir."
            )
