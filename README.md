# Akıllı Park

Trakya Üniversitesi BLM4502 kapsamında geliştirilen **hibrit akıllı otopark** projesi: Parking Birmingham verisi üzerinde **zaman serisi tahmini** (LSTM / GRU / Transformer ve isteğe bağlı tabular modeller), **ızgara tabanlı pekiştirmeli öğrenme** (Stable-Baselines3: PPO, DQN), **FastAPI** arka uç ve **React (Vite)** ile **Streamlit** arayüzleri.

## Proje amacı

Gerçek otopark doluluk verisinden **kısa vadeli doluluk tahmini** üretmek ve bu bilgiyi kullanarak grid dünyasında aracı **en uygun (düşük yoğunluklu) hedef otoparka** yönlendiren bir **PPO/DQN** ajanını eğitmek. Sunum çıktıları: açıklanabilir ödül, canlı metrik panelli animasyonlar, baseline karşılaştırması ve öğrenme eğrileri.

## Veri seti

**Parking Birmingham** (`data/raw/parking.csv`): lot kimliği, kapasite, anlık doluluk, zaman damgası. `data_preparation.py` ile train/val/test ayrımı ve `processed.parquet` üretilir.

## Zaman serisi tahmini

LSTM (veya `--cell gru|transformer`) ile lot/zaman bazlı **doluluk oranı** tahmini; test metrikleri ve `predictions/test_predictions.csv`. Bu tahmin, RL gözleminde `lstm_aggregate_pred` skalerine bağlanır.

## RL ortam yapısı

`environment.py` / `env/grid_navigation_env.py`: **15×15** grid, duvarlar, park lotları (doluluk renk skalası), hedef lot (G), ajan (A). Gözlem: 5 kanallı ızgara + 2 skaler (LSTM tahmini, hedef lot doluluğu). Eylemler: yukarı/aşağı/sol/sağ.

**Senaryo parametresi** (`scenario=`): `low`, `medium`, `high`, `dynamic` — lot doluluklarını ölçekler veya dinamik olarak günceller.

## Ödül fonksiyonu

`reward_utils.py` içinde bileşenler:

| Bileşen | Etki |
|--------|------|
| Zaman cezası | Her adımda `-GRID_STEP_COST` |
| Manhattan shaping | Hedefe yaklaşınca +, uzaklaşınca − |
| İlk ziyaret / tekrar | Keşif teşviki / gereksiz dolaşım cezası |
| Salınım (zigzag) | A↔B döngü cezası |
| Duvar çarpışması | Ek `-wall_penalty` |
| Hedef | Büyük `+GRID_GOAL_BONUS` |
| Süre aşımı | `GRID_TIMEOUT_PENALTY` |
| Kısa yol bonusu | BFS optimaline yakınlık (başarıda) |
| Yüksek doluluk lotu | `GRID_HIGH_OCC_*` ile dolu alan cezası |

## PPO / DQN karşılaştırması

`evaluate_agents.py` aynı test bölümlerinde **PPO, DQN, random, greedy Manhattan, BFS oracle** metriklerini toplar; `output/model_comparison_metrics.csv`, `ppo_vs_dqn_comparison.png`, `rl_episode_logs.csv` üretir.

## Çıktı görselleri (`output/`)

| Dosya | Açıklama |
|-------|----------|
| `ppo_agent_explained.gif` | PPO + canlı metrik paneli |
| `dqn_agent_explained.gif` | DQN + canlı metrik paneli |
| `ppo_vs_dqn_comparison.png` | Metrik bar chart |
| `reward_trend.png` / `success_rate_trend.png` | Öğrenme eğrileri |
| `path_length_comparison.png` | Ortalama rota uzunluğu |
| `occupancy_hourly_heatmap.png` | Saat × gün doluluk |
| `actual_vs_predicted.png` | LSTM gerçek vs tahmin |
| `rl_episode_logs.csv` | Bölüm bazlı karar logu |

Bu depo 5 kişilik ekip çalışması için düzenlenmiştir; aşağıdaki sıra, herkesin aynı ortamda projeyi ayağa kaldırması içindir.

## Gereksinimler

- **Python 3.10+** (3.11 önerilir)
- **Git**
- **Node.js 18+** ve **npm** (yalnızca `frontend/` için)
- Ham veri: `data/raw/parking.csv` (Parking Birmingham / Kaggle; depoda yoksa ekleyin)

PyTorch CPU sürümü `requirements.txt` ile kurulur; GPU kullanacaksanız [PyTorch kurulum sayfasından](https://pytorch.org/get-started/locally/) uygun wheel seçebilirsiniz.

---

## GitHub’dan çektikten sonra — terminalde sıra

Tüm komutları **proje kök dizininde** (`README.md` ve `requirements.txt`’nin olduğu klasör) çalıştırın.

### 1) Depoya girin

```powershell
cd Akilli_Park-master
```

*(Klasör adı farklıysa `git clone` ile oluşan dizin adını kullanın.)*

### 2) Sanal ortam oluşturun ve etkinleştirin

**Windows (PowerShell):**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Linux / macOS:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3) pip güncelleyin ve bağımlılıkları kurun

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4) Ham veriyi yerleştirin

`data/raw/parking.csv` dosyasının var olduğundan emin olun. Yoksa Kaggle vb. kaynaktan indirip bu yola koyun.

### 5) Train / val / test CSV üretin

```powershell
python data_preparation.py
```

Çıktı: `data/processed/train.csv`, `val.csv`, `test.csv`.

**Aykırı değer (künye Bölüm 5):** Varsayılan olarak lot bazında IQR filtresi açıktır (`ml_config.py`: `OUTLIER_FILTER_ENABLED`, `OUTLIER_IQR_MULTIPLIER=3.0`). Kapatmak için:

```powershell
python data_preparation.py --no-outliers
```

Filtreyi açtıktan veya `ml_config` değiştirdikten sonra **tüm pipeline’ı yeniden** çalıştırın (parquet, LSTM, RL, evaluate).

### 6) (İsteğe bağlı) Keşifsel veri analizi grafikleri

```powershell
python eda.py
```

Grafikler `output/` altına yazılır.

### 7) Derin öğrenme tahmin modeli (LSTM veya seçilen mimari)

```powershell
python train_lstm.py --cell lstm
```

İlk çalıştırmada gerekirse `data/processed/processed.parquet` ve ölçekleyici üretilir. Çıktılar: `models/lstm_model.pt`, `predictions/test_predictions.csv`.

*İsteğe bağlı:* hiperparametre araması için örneğin `--optuna-trials 20` (daha uzun sürer).

### 8) (İsteğe bağlı) Tabular baseline (XGBoost / Random Forest)

```powershell
python train_tabular_baselines.py
```

Metrikler `evaluation/reports/` altına yazılır.

### 9) Pekiştirmeli öğrenme (PPO + DQN)

```powershell
python train_rl.py --algo both
```

Çıktı: `models/ppo_agent.zip`, `models/dqn_agent.zip`, ilgili `vecnormalize_*.pkl`. Eğitim süresi makineye göre uzun olabilir; hızlı deneme için `--timesteps` değerini düşürebilirsiniz.

TensorBoard (isteğe bağlı):

```powershell
tensorboard --logdir logs/tensorboard
```

### 10) Birleşik değerlendirme

```powershell
python evaluate.py --part all
```

`--part all` ile PPO ve DQN metrikleri (model dosyaları varsa) `rl_sb3.ppo` / `rl_sb3.dqn` altında yazılır. Yalnızca bir algoritma: `python evaluate.py --part rl --rl-algo dqn`.

Rapor: `evaluation/reports/metrics.json`.

### 10b) RL karşılaştırma ve sunum çıktıları

```powershell
python evaluate_agents.py --episodes 40
python visualize_agent.py --presentation
python eda_visualizations.py
```

Modüler scriptler: `train_ppo.py`, `train_dqn.py`, `environment.py`, `reward_utils.py`, `config.py`.

---

## Uygulamaları çalıştırma

Aynı sanal ortamı kullanın. **API** ve **React** genelde iki ayrı terminaldedir.

### FastAPI (arka uç)

Proje kökünde:

```powershell
uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

Sağlık kontrolü: tarayıcıda `http://127.0.0.1:8000/docs`

### React + Vite (ön uç)

```powershell
cd frontend
npm install
npm run dev
```

Varsayılan: `http://127.0.0.1:5173` — Vite, `/api` isteklerini `8000` portuna yönlendirir (`vite.config.js`).

### Streamlit kontrol paneli

Proje köküne dönüp:

```powershell
cd ..
streamlit run ui_streamlit/app.py
```

---

## Klasör özeti

| Yol | Açıklama |
|-----|----------|
| `data/raw/` | Ham `parking.csv` |
| `data/processed/` | Bölünmüş CSV ve `processed.parquet` |
| `models/` | LSTM checkpoint, RL zip, ölçekleyici |
| `predictions/` | LSTM test tahmin CSV |
| `evaluation/reports/` | `metrics.json` vb. |
| `logs/` | RL Monitor CSV, TensorBoard |
| `output/` | EDA + sunum GIF/PNG/CSV (tüm grafikler) |
| `reward_utils.py` | Açıklanabilir ödül bileşenleri |
| `api/` | FastAPI |
| `frontend/` | React arayüz |
| `ui_streamlit/` | Streamlit uygulaması |

---

## Ekip için notlar

- **Veri ve modeller:** Büyük dosyalar Git LFS veya paylaşılan sürücü ile de paylaşılabilir; herkes aynı `RANDOM_SEED` ve `ml_config.py` ile tekrarlanabilir sonuç alır.
- **Paralel iş:** Biri veri/EDA, biri LSTM, biri RL, biri API/UI, biri raporlama/değerlendirme üzerinde çalışırken yukarıdaki sıraya uyun; API ve arayüzler için 7–9 adımları tamamlamış veya repodaki hazır model dosyalarını kullanmış olmanız gerekir.
- **Sorun giderme:** `data_preparation.py` “Ham veri bulunamadı” diyorsa `data/raw/parking.csv` yolunu kontrol edin. `evaluate.py` LSTM/RL hatası veriyorsa önce ilgili `train_*.py` scriptlerinin başarıyla bittiğinden emin olun.
