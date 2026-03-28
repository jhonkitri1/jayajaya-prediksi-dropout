import streamlit as st
import pandas as pd
import joblib

# Konfigurasi Halaman
st.set_page_config(page_title="Jaya Jaya Institut - Prediksi Dropout", page_icon="🎓", layout="centered")

# Memuat Model Machine Learning
@st.cache_resource
def load_model():
    return joblib.load('model/rf_model.joblib')

model = load_model()

# Header Aplikasi
st.title("🎓 Sistem Deteksi Dini Dropout")
st.subheader("Jaya Jaya Institut")
st.write("Aplikasi ini menggunakan Machine Learning untuk memprediksi probabilitas mahasiswa putus studi (dropout) berdasarkan profil akademik dan finansial mereka.")

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
    # Membuat dictionary dengan default value untuk semua 36 fitur (menggunakan nilai rata-rata/median dataset)
    # Ini agar model tidak error karena kekurangan kolom
    data = {
        'Marital_status': 1, 'Application_mode': 1, 'Application_order': 1, 'Course': 1, 
        'Daytime_evening_attendance': 1, 'Previous_qualification': 1, 'Previous_qualification_grade': 130.0, 
        'Nacionality': 1, 'Mothers_qualification': 1, 'Fathers_qualification': 1, 
        'Mothers_occupation': 1, 'Fathers_occupation': 1, 'Admission_grade': admission_grade, 
        'Displaced': 1, 'Educational_special_needs': 0, 'Debtor': 0, 
        'Tuition_fees_up_to_date': tuition_val, 'Gender': 0, 'Scholarship_holder': scholarship_val, 
        'Age_at_enrollment': age, 'International': 0, 'Curricular_units_1st_sem_credited': 0, 
        'Curricular_units_1st_sem_enrolled': 6, 'Curricular_units_1st_sem_evaluations': 6, 
        'Curricular_units_1st_sem_approved': 5, 'Curricular_units_1st_sem_grade': sem1_grade, 
        'Curricular_units_1st_sem_without_evaluations': 0, 'Curricular_units_2nd_sem_credited': 0, 
        'Curricular_units_2nd_sem_enrolled': 6, 'Curricular_units_2nd_sem_evaluations': 6, 
        'Curricular_units_2nd_sem_approved': 5, 'Curricular_units_2nd_sem_grade': sem2_grade, 
        'Curricular_units_2nd_sem_without_evaluations': 0, 'Unemployment_rate': 10.0, 
        'Inflation_rate': 1.0, 'GDP': 1.0
    }
    
    # Konversi ke DataFrame
    df_input = pd.DataFrame([data])
    
    # Melakukan Prediksi
    prediction = model.predict(df_input)
    
    st.markdown("---")
    st.header("Hasil Analisis:")
    
    if prediction[0] == 1:
        st.error("Peringatan Tinggi: Mahasiswa ini berisiko besar untuk DROPOUT. Segera jadwalkan sesi bimbingan akademik dan finansial.")
    else:
        st.success("Aman: Mahasiswa ini diprediksi akan LULUS (Graduate). Terus pertahankan performanya.")
