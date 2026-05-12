"""
Akıllı Park — proje künyesi ile uyumlu Streamlit kontrol paneli.

Modüller:
1) Zaman serisi tahmin ve model performansı
2) Keşifsel veri analizi (EDA çıktıları)
3) Sistem durumu ve özet göstergeler
4) Pekiştirmeli öğrenme simülasyonu / şeffaflık
5) Karar destek ve açıklanabilir otopark önerisi

Çalıştırma:
streamlit run ui_streamlit/app.py
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

st.title("Akıllı Park — Hibrit Tahmin + RL")
st.caption(
    "Parking Birmingham · LSTM/GRU/Transformer yaklaşımı · PPO/DQN grid yönlendirme · "
    "Açıklanabilir karar destek paneli"
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
    """Doluluk oranına göre seviye etiketi ve renk döndürür."""
    if rate < 0.33:
        return "Düşük (yeşil)", "#2e7d32"
    if rate < 0.66:
        return "Orta (sarı)", "#f9a825"
    return "Yüksek (kırmızı)", "#c62828"


def _safe_read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


# ---------------------------------------------------------------------
# 1) Zaman Serisi Tahmin & Performans
# ---------------------------------------------------------------------
with tab_ts:
    st.subheader("Zaman serisi tahmin ve model performansı")
    st.markdown(
        "Bu modülde tahmin edilen otopark doluluk oranları gerçek değerlerle birlikte gösterilir. "
        "MAE, RMSE, MAPE ve R² metrikleri `evaluate.py` çıktısından okunur."
    )

    pred_path = PREDICTIONS_DIR / "test_predictions.csv"

    if pred_path.exists():
        df = pd.read_csv(pred_path)
        df["LastUpdated"] = pd.to_datetime(df["LastUpdated"], errors="coerce")
        df = df.dropna(subset=["LastUpdated"])

        if df.empty:
            st.warning("Tahmin dosyası var ancak geçerli tarih içeren kayıt bulunamadı.")
        else:
            tmin, tmax = df["LastUpdated"].min(), df["LastUpdated"].max()

            c1, c2, c3 = st.columns(3)
            with c1:
                d0 = st.date_input(
                    "Başlangıç",
                    value=tmin.date(),
                    min_value=tmin.date(),
                    max_value=tmax.date(),
                )
            with c2:
                d1 = st.date_input(
                    "Bitiş",
                    value=tmax.date(),
                    min_value=tmin.date(),
                    max_value=tmax.date(),
                )
            with c3:
                max_pts = st.slider("Grafikte en fazla nokta", 50, 3000, 800, 50)

            mask = (
                (df["LastUpdated"] >= pd.Timestamp(d0))
                & (df["LastUpdated"] <= pd.Timestamp(d1) + pd.Timedelta(days=1))
            )

            sub = df.loc[mask].sort_values("LastUpdated")

            if sub.empty:
                st.warning("Seçilen tarih aralığında veri yok.")
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

                try:
                    out = plot_prediction_vs_actual()
                    st.caption(f"Kayıtlı grafik: `{out}`")
                except Exception as e:
                    st.info(f"Tahmin grafiği dosyaya yazılamadı: {e}")
    else:
        st.warning("Tahmin CSV bulunamadı. Önce `python train_lstm.py` çalıştırılmalı.")

    rep = EVALUATION_DIR / "metrics.json"

    if rep.exists():
        metrics = _safe_read_json(rep)
        lstm_metrics = metrics.get("lstm", {})

        if lstm_metrics:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("MAE", f"{lstm_metrics.get('lstm_mae', 0):.5f}")
            c2.metric("RMSE", f"{lstm_metrics.get('lstm_rmse', 0):.5f}")
            c3.metric("MAPE %", f"{lstm_metrics.get('lstm_mape_pct', 0):.2f}")
            c4.metric("R²", f"{lstm_metrics.get('lstm_r2', 0):.4f}")
        else:
            st.info("`metrics.json` içinde LSTM metrikleri bulunamadı.")
    else:
        st.info("`python evaluate.py --part lstm` ile metrikleri oluşturabilirsiniz.")


# ---------------------------------------------------------------------
# 2) EDA
# ---------------------------------------------------------------------
with tab_eda:
    st.subheader("Keşifsel veri analizi")
    st.markdown(
        "Bu bölümde `eda.py` tarafından oluşturulan saatlik, günlük, histogram, korelasyon "
        "ve benzeri analiz grafikleri gösterilir."
    )

    patterns = sorted(OUTPUT_DIR.glob("*.png")) if OUTPUT_DIR.exists() else []

    if patterns:
        for p in patterns[:12]:
            st.image(str(p), caption=p.name, use_container_width=True)
    else:
        st.warning("EDA grafiği bulunamadı. Çalıştırın: `python eda.py`")


# ---------------------------------------------------------------------
# 3) Dashboard
# ---------------------------------------------------------------------
with tab_dash:
    st.subheader("Sistem durumu ve özet göstergeler")

    c1, c2 = st.columns(2)
    c1.metric("Sabit seed", str(RANDOM_SEED))
    c2.metric(
        "Grid gözlem boyutu",
        f"{5 * GRID_HEIGHT * GRID_WIDTH + 2} = 5×H×W + 2 skaler",
    )

    pq = DATA_PROCESSED / "processed.parquet"

    if pq.exists():
        dfp = pd.read_parquet(pq)
        dfp["LastUpdated"] = pd.to_datetime(dfp["LastUpdated"], errors="coerce")
        dfp = dfp.dropna(subset=["LastUpdated"])

        if not dfp.empty:
            dfp["occ_rate"] = dfp["Occupancy"] / dfp["Capacity"]

            c1, c2, c3 = st.columns(3)
            c1.metric("İşlenmiş satır sayısı", f"{len(dfp):,}")
            c2.metric("Ortalama doluluk", f"{dfp['occ_rate'].mean():.3f}")
            c3.metric("Otopark sayısı", f"{dfp['SystemCodeNumber'].nunique()}")

            last = dfp.sort_values("LastUpdated").iloc[-1]
            st.write("Son kayıt zamanı:", str(last["LastUpdated"]))

            last_t = dfp["LastUpdated"].max()
            snap = dfp[dfp["LastUpdated"] == last_t].copy()

            if not snap.empty:
                snap["occ_rate"] = snap["Occupancy"] / snap["Capacity"]
                best = snap.loc[snap["occ_rate"].idxmin()]
                worst = snap.loc[snap["occ_rate"].idxmax()]

                c1, c2 = st.columns(2)
                c1.success(
                    f"En uygun otopark: {best['SystemCodeNumber']} "
                    f"(%{float(best['occ_rate']) * 100:.1f} dolu)"
                )
                c2.error(
                    f"En yoğun otopark: {worst['SystemCodeNumber']} "
                    f"(%{float(worst['occ_rate']) * 100:.1f} dolu)"
                )
        else:
            st.warning("İşlenmiş veri dosyasında geçerli kayıt bulunamadı.")
    else:
        st.warning("`data/processed/processed.parquet` bulunamadı. Önce veri hattını çalıştırın.")

    rep = EVALUATION_DIR / "metrics.json"

    if rep.exists():
        st.subheader("Genel metrik dosyası")
        st.json(_safe_read_json(rep))
    else:
        st.info("`python evaluate.py` ile `evaluation/reports/metrics.json` oluşturabilirsiniz.")


# ---------------------------------------------------------------------
# 4) RL Simülasyon & Görselleştirme
# ---------------------------------------------------------------------
with tab_rl:
    st.subheader("Pekiştirmeli öğrenme tabanlı yönlendirme")
    st.markdown(
        "**Durum uzayı:** duvar/boş alan bilgisi, park alanları maskesi, hedef hücre, "
        "araç konumu, otopark doluluk ısı haritası, LSTM aggregate tahmini ve boşalma vekili skalerleri.\n\n"
        "Bu bölümde ajan adım adım ilerletilebilir. Her adımda reward, terminal/truncated bilgisi "
        "ve ortamdan dönen sayısal bilgiler izlenebilir."
    )

    st.code(
        "python train_rl.py --algo ppo\n"
        "python train_rl.py --algo dqn\n"
        "python train_rl.py --algo both\n"
        "python -m evaluation.record_parking_gif",
        language="bash",
    )

    scenario = st.selectbox(
        "Senaryo",
        ("test (değerlendirme)", "train (alternatif senaryo)"),
    )

    max_eps = st.slider("Maksimum bölüm sayısı", 30, 600, 150, 10)
    split_kw = "test" if scenario.startswith("test") else None

    if st.button("Grid ortamını yükle / sıfırla"):
        try:
            from env.grid_navigation_env import (
                GridParkingNavigationEnv,
                build_grid_nav_episode_configs,
            )

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
        act_labels = {
            0: "0 ↑ yukarı",
            1: "1 ↓ aşağı",
            2: "2 ← sol",
            3: "3 → sağ",
        }

        a = st.selectbox("Aksiyon seç", [0, 1, 2, 3], format_func=lambda x: act_labels[x])

        col_a, col_b = st.columns(2)

        with col_a:
            if st.button("Bir adım uygula"):
                obs, reward, terminated, truncated, info = env.step(int(a))
                st.session_state["_rl_obs"] = obs

                clean_info = {
                    k: v
                    for k, v in info.items()
                    if isinstance(v, (bool, int, float, str))
                }

                st.session_state["_rl_log"].append(
                    {
                        "aksiyon": act_labels[int(a)],
                        "reward": float(reward),
                        "terminated": bool(terminated),
                        "truncated": bool(truncated),
                        "info": clean_info,
                    }
                )

                if terminated or truncated:
                    st.info("Bölüm bitti. Yeni bölüm için ortamı tekrar sıfırlayın.")

        with col_b:
            if st.button("Günlüğü temizle"):
                st.session_state["_rl_log"] = []

        frame = env.render()

        if frame is not None:
            st.image(
                Image.fromarray(np.asarray(frame)),
                caption="Grid ortamı — ajan, hedef ve doluluk bilgisi",
                use_container_width=True,
            )

        if st.session_state.get("_rl_log"):
            st.subheader("Ajan karar günlüğü")

            log_df = pd.DataFrame(st.session_state["_rl_log"])

            st.dataframe(log_df.tail(10), use_container_width=True)

            last_step = st.session_state["_rl_log"][-1]
            c1, c2, c3 = st.columns(3)
            c1.metric("Son aksiyon", last_step["aksiyon"])
            c2.metric("Son reward", f"{last_step['reward']:.3f}")
            c3.metric(
                "Bölüm durumu",
                "Bitti" if last_step["terminated"] or last_step["truncated"] else "Devam ediyor",
            )

    gif1 = ROOT / "output" / "parking_agent.gif"
    gif2 = ROOT / "output" / "rl_rollout.gif"

    if gif1.exists():
        st.image(str(gif1), caption="PPO park GIF", use_container_width=True)

    if gif2.exists():
        st.image(str(gif2), caption="RL rollout GIF", use_container_width=True)


# ---------------------------------------------------------------------
# 5) Karar Destek & Açıklanabilir Öneri
# ---------------------------------------------------------------------
with tab_decision:
    st.subheader("Karar destek ve açıklanabilir otopark önerisi")
    st.markdown(
        "Bu modül her otoparkı yalnızca doluluk oranına göre değil; **boş kapasite**, "
        "**doluluk seviyesi**, **temsili mesafe maliyeti** ve **uygunluk skoru** üzerinden değerlendirir. "
        "Amaç, sistemin neden belirli bir otoparkı önerdiğini açıklanabilir hale getirmektir."
    )

    split_dec = st.radio("Veri dilimi", ("test", "train", "val"), horizontal=True)

    pq = DATA_PROCESSED / "processed.parquet"
    pred_path = PREDICTIONS_DIR / "test_predictions.csv"

    if not pq.exists():
        st.error("Önce `utils.data_pipeline` ile `processed.parquet` oluşturun.")
    else:
        dfp = pd.read_parquet(pq)
        dfp = dfp[dfp["split"] == split_dec].copy()

        dfp["LastUpdated"] = pd.to_datetime(dfp["LastUpdated"], errors="coerce")
        dfp = dfp.dropna(subset=["LastUpdated"])

        if dfp.empty:
            st.warning("Seçilen veri diliminde kayıt yok.")
        else:
            dfp["occ_rate"] = dfp["Occupancy"] / dfp["Capacity"]

            last_t = dfp["LastUpdated"].max()
            snap = dfp[dfp["LastUpdated"] == last_t].copy()

            if snap.empty:
                st.warning("Son zaman diliminde kayıt bulunamadı.")
            else:
                st.caption(f"Analiz edilen zaman damgası: `{last_t}`")

                snap = snap.sort_values("SystemCodeNumber").reset_index(drop=True)

                snap["occupancy_pct"] = snap["occ_rate"] * 100
                snap["empty_capacity"] = snap["Capacity"] - snap["Occupancy"]

                # Gerçek koordinat yoksa UI üzerinde karar mantığını göstermek için
                # otopark sırasına göre temsili bir mesafe maliyeti oluşturuyoruz.
                snap["demo_distance"] = np.arange(1, len(snap) + 1)

                occ_cost = snap["occ_rate"].astype(float)
                dist_cost = snap["demo_distance"] / max(float(snap["demo_distance"].max()), 1.0)

                # Düşük skor daha iyi:
                # %70 doluluk maliyeti + %30 temsili mesafe maliyeti
                snap["decision_score"] = 0.70 * occ_cost + 0.30 * dist_cost

                # Kullanıcıya daha anlaşılır göstermek için uygunluk skorunu tersliyoruz.
                # Yüksek uygunluk daha iyi anlamına gelir.
                snap["suitability_score"] = 1.0 - snap["decision_score"]

                best_idx = snap["decision_score"].idxmin()
                best = snap.loc[best_idx]

                rate = float(best["occ_rate"])
                label, color = _occupancy_level(rate)

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Önerilen otopark", str(best["SystemCodeNumber"]))
                c2.metric("Doluluk", f"%{rate * 100:.1f}")
                c3.metric("Boş kapasite", f"{int(best['empty_capacity'])}")
                c4.metric("Uygunluk skoru", f"{float(best['suitability_score']):.3f}")

                st.markdown(
                    f"""
                    <div style="
                        padding:16px;
                        border-radius:12px;
                        border:1px solid #ddd;
                        background-color:#fafafa;
                        margin-top:10px;
                        margin-bottom:10px;
                    ">
                        <h4 style="margin-bottom:8px;">Neden bu otopark seçildi?</h4>
                        <ul>
                            <li>
                                <b>Doluluk seviyesi:</b>
                                <span style="color:{color};font-weight:bold">{label}</span>
                                — %{rate * 100:.1f}
                            </li>
                            <li>
                                <b>Boş kapasite:</b> {int(best["empty_capacity"])} araçlık boş alan
                            </li>
                            <li>
                                <b>Temsili mesafe maliyeti:</b> {int(best["demo_distance"])}
                            </li>
                            <li>
                                <b>Karar skoru:</b> {float(best["decision_score"]):.3f}
                                <span style="color:#666;">(düşük skor daha iyi)</span>
                            </li>
                            <li>
                                <b>Seçilme nedeni:</b>
                                Düşük doluluk oranı ve kabul edilebilir rota maliyeti nedeniyle
                                en uygun aday olarak belirlenmiştir.
                            </li>
                        </ul>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.divider()

                st.subheader("Otopark doluluklarının görsel karşılaştırması")

                table = snap[
                    [
                        "SystemCodeNumber",
                        "Capacity",
                        "Occupancy",
                        "empty_capacity",
                        "occupancy_pct",
                        "demo_distance",
                        "decision_score",
                        "suitability_score",
                    ]
                ].copy()

                table = table.rename(
                    columns={
                        "SystemCodeNumber": "Otopark",
                        "Capacity": "Kapasite",
                        "Occupancy": "Dolu",
                        "empty_capacity": "Boş Kapasite",
                        "occupancy_pct": "Doluluk %",
                        "demo_distance": "Mesafe Maliyeti",
                        "decision_score": "Karar Skoru",
                        "suitability_score": "Uygunluk Skoru",
                    }
                )

                st.dataframe(
                    table.style.format(
                        {
                            "Doluluk %": "{:.1f}",
                            "Karar Skoru": "{:.3f}",
                            "Uygunluk Skoru": "{:.3f}",
                        }
                    )
                    .background_gradient(subset=["Doluluk %"])
                    .background_gradient(subset=["Uygunluk Skoru"]),
                    use_container_width=True,
                )

                st.subheader("Doluluk ve uygunluk grafiği")

                chart_table = table.copy()
                chart_table["Uygunluk Skoru"] = chart_table["Uygunluk Skoru"] * 100

                st.bar_chart(
                    chart_table.set_index("Otopark")[["Doluluk %", "Uygunluk Skoru"]],
                    use_container_width=True,
                )

                st.divider()

                st.subheader("LSTM aggregate tahmin bilgisi")

                if pred_path.exists():
                    pr = pd.read_csv(pred_path)
                    pr["LastUpdated"] = pd.to_datetime(pr["LastUpdated"], errors="coerce")
                    matched = pr[pr["LastUpdated"] == last_t]

                    if not matched.empty:
                        pred_val = float(matched["y_pred_occupancy_rate"].iloc[0])
                        true_val = float(matched["y_true_occupancy_rate"].iloc[0])

                        c1, c2 = st.columns(2)
                        c1.metric("Aggregate gerçek doluluk", f"{true_val:.3f}")
                        c2.metric("Aggregate LSTM tahmini", f"{pred_val:.3f}")
                    else:
                        st.info("Bu zaman damgası için LSTM tahmini bulunamadı.")
                else:
                    st.info("`test_predictions.csv` bulunamadığı için LSTM tahmini gösterilemiyor.")

                st.caption(
                    "Not: Bu panel karar destek modülünün açıklanabilir sürümüdür. "
                    "Buradaki mesafe maliyeti, UI üzerinde karar mantığını göstermek için "
                    "basitleştirilmiş temsili bir değerdir. Gerçek rota maliyeti ve ajan hareketleri "
                    "RL ortamı/API üzerinden üretilir."
                )