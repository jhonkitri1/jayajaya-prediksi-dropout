import streamlit as st
import pandas as pd
import joblib

# Konfigurasi Halaman
st.set_page_config(page_title="Jaya Jaya Institut - Prediksi Dropout", page_icon="🎓", layout="centered")

# Memuat Model Machine Learning yang BARU (Hanya 6 Fitur)
@st.cache_resource
def load_model():
    return joblib.load('model/rf_model_new.joblib')

model = load_model()

# Header Aplikasi
st.title("🎓 Sistem Deteksi Dini Dropout")
st.subheader("Jaya Jaya Institut")
st.write("Aplikasi ini memprediksi probabilitas mahasiswa putus studi (dropout) berdasarkan 6 fitur utama yang terbukti paling berpengaruh dari hasil analisis data.")

st.markdown("---")

# Input Form untuk User
st.header("Masukkan Profil Mahasiswa")

col1, col2 = st.columns(2)

with col1:
    tuition_fees = st.selectbox("Status Pembayaran Uang Kuliah?", ["Lunas (1)", "Menunggak (0)"])
    tuition_val = 1 if tuition_fees == "Lunas (1)" else 0
    
    scholarship = st.selectbox("Pemegang Beasiswa?", ["Ya (1)", "Tidak (0)"])
    scholarship_val = 1 if scholarship == "Ya (1)" else 0
    
    age = st.number_input("Usia Saat Mendaftar", min_value=15, max_value=60, value=20)

with col2:
    admission_grade = st.number_input("Nilai Masuk (Admission Grade)", min_value=0.0, max_value=200.0, value=120.0)
    sem1_grade = st.number_input("Nilai Semester 1", min_value=0.0, max_value=20.0, value=12.0)
    sem2_grade = st.number_input("Nilai Semester 2", min_value=0.0, max_value=20.0, value=12.0)

# Tombol Prediksi
if st.button("Deteksi Potensi Dropout", use_container_width=True):
    # Menyusun data HANYA dengan 6 fitur, persis sesuai urutan training
    data = {
        'Tuition_fees_up_to_date': tuition_val,
        'Scholarship_holder': scholarship_val,
        'Age_at_enrollment': age,
        'Admission_grade': admission_grade,
        'Curricular_units_1st_sem_grade': sem1_grade,
        'Curricular_units_2nd_sem_grade': sem2_grade
    }
    
    # Konversi ke DataFrame
    df_input = pd.DataFrame([data])
    
    # Melakukan Prediksi
    prediction = model.predict(df_input)
    
    st.markdown("---")
    st.header("Hasil Analisis:")
    
    if prediction[0] == 1:
        st.error("**Peringatan Tinggi:** Mahasiswa ini berisiko besar untuk DROPOUT. Segera jadwalkan sesi bimbingan akademik dan finansial.")
    else:
        st.success("**Aman:** Mahasiswa ini diprediksi akan LULUS (Graduate). Terus pertahankan performanya.")
