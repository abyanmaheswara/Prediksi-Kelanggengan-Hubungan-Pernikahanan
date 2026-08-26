# 💍 Prediksi Kelanggengan Hubungan / Pernikahan

Aplikasi berbasis Machine Learning untuk memprediksi potensi kelanggengan atau risiko perceraian hubungan pernikahan berdasarkan berbagai faktor psikologis, ekonomi, dan demografi.

---

## 🚀 Fitur Utama
- **Front-End Interaktif:** Dibangun menggunakan [Streamlit](https://streamlit.io/) dengan antarmuka Bahasa Indonesia yang intuitif.
- **Model Machine Learning Terbaik:** Menggunakan **Gradient Boosting Classifier** (Akurasi ~78.7%) yang telah diuji dan dibandingkan dengan berbagai algoritma lain (Decision Tree, Random Forest, Logistic Regression).
- **Interpretasi Model:** Visualisasi faktor-faktor paling berpengaruh (*Feature Importance*) menggunakan Plotly.
- **Forwarding Publik:** Terintegrasi dengan script otomatis **Ngrok Tunnel** untuk keperluan demo online.

---

## 📁 Struktur Proyek
```text
├── app.py                          # Aplikasi web Streamlit
├── run_ngrok.py                    # Script menjalankan Streamlit + Ngrok
├── marriage_longevity_master.csv   # Dataset pernikahan
├── artifacts/                      # Model & Preprocessing scaler/encoder
│   ├── best_model.pkl
│   ├── scaler.pkl
│   ├── encoders.pkl
│   └── features.pkl
└── README.md
```

---

## 🛠️ Cara Menjalankan Aplikasi

### 1. Install Dependencies
```bash
pip install streamlit pandas joblib scikit-learn plotly pyngrok
```

### 2. Jalankan secara Lokal
```bash
streamlit run app.py
```
Akses di browser pada `http://localhost:8501`.

### 3. Jalankan dengan Ngrok (Akses Online)
```bash
python run_ngrok.py
```
Salin URL publik yang muncul di terminal.
