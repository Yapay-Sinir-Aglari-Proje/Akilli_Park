# Trakya Üniversitesi - Bilgisayar Mühendisliği Yapay Sinir Ağları Projesi
# Akıllı Park


Akıllı otopark sistemleri için geliştirilmiş; Veri İşleme, Keşifsel Veri Analizi (EDA), LSTM tabanlı doluluk tahmini ve Gymnasium tabanlı Pekiştirmeli Öğrenme (RL) simülasyon ortamını içeren kapsamlı bir yapay zeka projesi.

## Proje Ekibi
* Ulaş Ekrem Emili
* İrem Su Erdemir
* Hatice Pınar Yılmaz
* Beyza Yılmaz
* Yusuf Güde

---

## Çalıştırma Adımları

Sistemi uçtan uca test etmek ve çalıştırmak için aşağıdaki adımları sırasıyla izleyin:

1. **Bağımlılıkların Kurulması:** `pip3 install -r requirements.txt`
2. **Veri Hazırlama:** `python3 data_preparation.py` *(Çıktılar `data/processed/` klasörüne gider)*
3. **Keşifsel Analiz (EDA):** `python3 eda.py` *(Grafikler `output/` klasörüne kaydedilir)*
4. **LSTM Modeli Eğitimi:** `python3 lstm_model.py` *(Model `models/lstm_parking_model.h5` olarak kaydedilir)*
5. **RL State (Durum) Hazırlığı:** `python3 prepare_prediction_states.py` *(Tahminleri ve RL matrislerini hazırlar)*
6. **Simülasyon Ortamı Testi:** `python3 -m parking_rl.smart_parking_env`
7. **RL Ajanının Eğitilmesi (PPO):** `python3 rl_model.py` *(En iyi model `models/best_model.zip` olarak kaydedilir)*
8. **Performans Analizi:** `python3 evaluate_performance.py` *(Modelin başarı oranı, park bulma süresi ve maliyet metriklerini raporlar)*


---

# Bağımlılıklar
pandas, numpy, matplotlib, scikit-learn, tensorflow, gymnasium, stable-baselines3
