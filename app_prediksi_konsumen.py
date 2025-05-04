import streamlit as st
import numpy as np
import tensorflow as tf
import joblib
import os

# Konfigurasi logging untuk debug (tetap dipertahankan untuk troubleshooting)
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Konfigurasi halaman
st.set_page_config(page_title="Prediksi Konsumen", layout="centered")

st.title("🧠 Prediksi Konsumen Berdasarkan Umur dan Gaji")
st.markdown("Masukkan data umur dan gaji pengguna untuk memprediksi apakah akan membeli produk atau tidak.")

# Path untuk model dan scaler
model_path = "model_prediksi_konsumen.keras"  # Sesuai dengan format file yang Anda gunakan
scaler_path = "scaler.pkl"  # Sesuai dengan nama file scaler Anda

# Cek keberadaan file (ditampilkan lebih sederhana)
model_exists = os.path.exists(model_path)
scaler_exists = os.path.exists(scaler_path)

# Status file tidak ditampilkan lagi untuk tampilan lebih bersih

model_loaded = False
scaler_loaded = False

if model_exists and scaler_exists:
    try:
        logger.info(f"Mencoba memuat model dari {model_path}")
        model = tf.keras.models.load_model(model_path)
        logger.info("Model berhasil dimuat!")
        model_loaded = True
        
        logger.info(f"Mencoba memuat scaler dari {scaler_path}")
        scaler = joblib.load(scaler_path)
        logger.info("Scaler berhasil dimuat!")
        scaler_loaded = True
        
        st.success("Model dan scaler berhasil dimuat! ✅")
    except Exception as e:
        logger.error(f"Error detail: {type(e).__name__}: {str(e)}")
        st.error(f"Gagal memuat model atau scaler: {str(e)}")
        # Detail error tetap dipertahankan untuk troubleshooting
        st.expander("Detail Error").exception(e)
else:
    st.error("Model atau scaler tidak ditemukan. Silakan pastikan kedua file berikut ada:")
    st.code(f"Path model: {os.path.abspath(model_path)}")
    st.code(f"Path scaler: {os.path.abspath(scaler_path)}")

# Input
age = st.number_input("Umur", min_value=18, max_value=100, value=30)
salary = st.number_input("Gaji Perkiraan (dalam USD)", min_value=0, value=50000)

# Tombol Prediksi
if st.button("Prediksi"):
    if not (model_loaded and scaler_loaded):
        st.error("Model atau scaler belum siap digunakan.")
    else:
        try:
            # Siapkan input dan lakukan scaling
            input_data = np.array([[age, salary]])
            # Data input tidak ditampilkan lagi
            
            scaled_input = scaler.transform(input_data)
            # Data scaling tidak ditampilkan lagi

            # Prediksi
            prediction = model.predict(scaled_input)
            # Nilai prediksi mentah tidak ditampilkan lagi
            
            pred_class = int(prediction[0][0] > 0.5)
            confidence = prediction[0][0] if pred_class == 1 else 1 - prediction[0][0]

            # Tampilkan hasil
            st.subheader("📊 Hasil Prediksi")
            st.write(f"**Akurasi Prediksi (Confidence):** {confidence * 100:.2f}%")

            if pred_class == 1:
                st.success("✅ Prediksi: Pengguna kemungkinan AKAN membeli produk.")
            else:
                st.warning("❌ Prediksi: Pengguna kemungkinan TIDAK membeli produk.")
        except Exception as e:
            logger.error(f"Error prediksi: {type(e).__name__}: {str(e)}")
            st.error(f"Terjadi kesalahan saat prediksi: {str(e)}")
            # Detail error tetap dipertahankan untuk troubleshooting
            st.expander("Detail Error").exception(e)