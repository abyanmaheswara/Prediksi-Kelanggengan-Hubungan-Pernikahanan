import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Prediksi Kelanggengan Pernikahan", page_icon="💍")

st.title("💍 Prediksi Kelanggengan Hubungan / Pernikahan")
st.write("Isi data di bawah untuk memprediksi risiko perceraian berdasarkan model Machine Learning.")

with st.expander("🔍 Bagaimana Cara Kerja Prediksi Ini?"):
    st.markdown("""
    **Aplikasi ini menggunakan model Machine Learning** yang sudah dilatih dari ribuan data pernikahan. Berikut langkah-langkahnya:

    1. **Pengumpulan Data** — Kamu mengisi data seperti usia, pendapatan, jumlah anak, tingkat konflik, dll.
    2. **Normalisasi (Scaling)** — Data kamu dinormalisasi agar setiap fitur punya skala yang setara, sehingga model tidak bias terhadap fitur dengan angka besar.
    3. **Prediksi Model** — Model menganalisis pola dari data pelatihan dan membandingkannya dengan data kamu untuk menghitung probabilitas risiko perceraian.
    4. **Hasil** — Model menghasilkan:
       - **Prediksi**: Berisiko cerai atau kemungkinan langgeng
       - **Probabilitas**: Persentase keyakinan model (0% = sangat langgeng, 100% = sangat berisiko cerai)
    5. **Faktor Berpengaruh** — Grafik menunjukkan faktor mana yang paling memengaruhi keputusan model secara umum.

    > ⚠️ **Catatan**: Prediksi ini hanya berdasarkan pola statistik dan **bukan diagnosis pasti**. Banyak faktor kehidupan nyata yang tidak bisa ditangkap oleh data.
    """)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "artifacts", "best_model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "artifacts", "scaler.pkl"))
encoders = joblib.load(os.path.join(BASE_DIR, "artifacts", "encoders.pkl"))
features = joblib.load(os.path.join(BASE_DIR, "artifacts", "features.pkl"))

df_ref = pd.read_csv(os.path.join(BASE_DIR, "marriage_longevity_master.csv"))

# Mapping nama kolom ke label Bahasa Indonesia
LABEL_ID = {
    "marriage_number": "Pernikahan Ke-",
    "age_at_marriage": "Usia Saat Menikah",
    "age_gap_years": "Selisih Usia (Tahun)",
    "education_level": "Tingkat Pendidikan",
    "household_income_usd": "Pendapatan Rumah Tangga (USD)",
    "financial_stress": "Tingkat Stres Keuangan",
    "both_employed": "Keduanya Bekerja",
    "cohabited_before": "Tinggal Bersama Sebelum Menikah",
    "premarital_counseling": "Konseling Pranikah",
    "child_before_marriage": "Anak Sebelum Menikah",
    "n_children": "Jumlah Anak",
    "religious_attendance": "Keaktifan Beribadah",
    "criticism": "Tingkat Kritik",
    "contempt": "Tingkat Penghinaan",
    "defensiveness": "Tingkat Defensif",
    "stonewalling": "Tingkat Menghindar (Stonewalling)",
    "repair_attempt_success": "Keberhasilan Memperbaiki Konflik",
    "positive_negative_ratio": "Rasio Positif-Negatif",
    "conflict_frequency_weekly": "Frekuensi Konflik per Minggu",
    "shared_activities_weekly": "Aktivitas Bersama per Minggu",
    "years_married": "Lama Menikah (Tahun)",
}

# Mapping opsi kategorikal ke Bahasa Indonesia
OPSI_ID = {
    "education_level": {
        "high_school": "SMA / Sederajat",
        "less_than_hs": "Di Bawah SMA",
        "bachelors": "Sarjana (S1)",
        "graduate": "Pascasarjana (S2/S3)",
        "some_college": "Pernah Kuliah",
    },
    "religious_attendance": {
        "never": "Tidak Pernah",
        "rarely": "Jarang",
        "monthly": "Bulanan",
        "weekly": "Mingguan",
    },

}

def get_label(feat):
    """Ambil label Indonesia, fallback ke nama kolom asli."""
    return LABEL_ID.get(feat, feat)

st.subheader("📝 Masukkan Data")

user_input = {}

with st.form("form_prediksi"):
    cols = st.columns(2)

    # Fitur yang seharusnya Ya / Tidak (binary 0/1)
    FITUR_YA_TIDAK = {"both_employed", "cohabited_before", "premarital_counseling", "child_before_marriage"}

    # Fitur yang harus bilangan bulat
    FITUR_BULAT = {"marriage_number", "n_children", "age_at_marriage", "age_gap_years",
                   "conflict_frequency_weekly", "shared_activities_weekly"}

    # Fitur skala 0-10 (step 0.5)
    FITUR_SKALA = {"financial_stress", "criticism", "contempt", "defensiveness",
                   "stonewalling", "repair_attempt_success"}

    for i, feat in enumerate(features):
        col = cols[i % 2]
        with col:
            label = get_label(feat)
            if feat in encoders:
                options = list(encoders[feat].classes_)
                # Terjemahkan opsi jika tersedia
                if feat in OPSI_ID and OPSI_ID[feat]:
                    display_options = [OPSI_ID[feat].get(opt, opt) for opt in options]
                    pilihan_display = st.selectbox(label, display_options)
                    reverse_map = {v: k for k, v in OPSI_ID[feat].items()}
                    pilihan_asli = reverse_map.get(pilihan_display, pilihan_display)
                    user_input[feat] = encoders[feat].transform([pilihan_asli])[0]
                else:
                    pilihan = st.selectbox(label, options)
                    user_input[feat] = encoders[feat].transform([pilihan])[0]
            elif feat in FITUR_YA_TIDAK:
                pilihan = st.selectbox(label, ["Tidak", "Ya"])
                user_input[feat] = 1 if pilihan == "Ya" else 0
            elif feat in FITUR_BULAT:
                min_val = int(df_ref[feat].min())
                max_val = int(df_ref[feat].max())
                mean_val = int(round(df_ref[feat].mean()))
                user_input[feat] = st.slider(label, min_val, max_val, mean_val, step=1)
            elif feat == "household_income_usd":
                min_val = int(df_ref[feat].min())
                max_val = int(df_ref[feat].max())
                mean_val = int(round(df_ref[feat].mean() / 1000) * 1000)
                user_input[feat] = st.slider(label, min_val, max_val, mean_val, step=1000)
            elif feat in FITUR_SKALA:
                user_input[feat] = st.slider(label, 0.0, 10.0, 5.0, step=0.5)
            else:
                min_val = float(df_ref[feat].min())
                max_val = float(df_ref[feat].max())
                mean_val = float(df_ref[feat].mean())
                user_input[feat] = st.slider(label, min_val, max_val, mean_val, step=0.5)

    submit = st.form_submit_button("🔮 Prediksi Sekarang")

if submit:
    input_df = pd.DataFrame([user_input])[features]
    input_scaled = scaler.transform(input_df)

    pred = model.predict(input_scaled)[0]
    proba = model.predict_proba(input_scaled)[0][1]

    st.subheader("📊 Hasil Prediksi")
    if pred == 1:
        st.error("⚠️ Berisiko tinggi bercerai")
    else:
        st.success("✅ Kemungkinan besar akan langgeng")

    st.metric("Probabilitas Risiko Bercerai", f"{proba*100:.1f}%")

    # Tampilkan ringkasan data yang diisi
    with st.expander("📋 Lihat Data yang Kamu Isi", expanded=False):
        ringkasan = pd.DataFrame({
            "Faktor": [get_label(f) for f in features],
            "Nilai": [user_input[f] for f in features]
        })
        st.dataframe(ringkasan, use_container_width=True, hide_index=True)

    if hasattr(model, "feature_importances_"):
        import plotly.express as px

        imp = pd.DataFrame({
            "Fitur": [get_label(f) for f in features],
            "Pengaruh": model.feature_importances_
        }).sort_values("Pengaruh", ascending=True).tail(10)

        st.subheader("📈 Faktor Paling Berpengaruh")
        st.caption("💡 Grafik ini menunjukkan **seberapa penting** setiap faktor dalam keputusan model, "
                   "bukan nilai yang kamu isi. Semakin panjang bar-nya, semakin besar pengaruh faktor tersebut terhadap prediksi.")

        fig = px.bar(
            imp,
            x="Pengaruh",
            y="Fitur",
            orientation="h",
            color="Pengaruh",
            color_continuous_scale="RdYlGn_r",
            text=imp["Pengaruh"].apply(lambda x: f"{x:.3f}"),
        )
        fig.update_layout(
            height=400,
            margin=dict(l=10, r=10, t=10, b=10),
            yaxis_title="",
            xaxis_title="Tingkat Pengaruh",
            coloraxis_showscale=False,
            font=dict(size=13),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)