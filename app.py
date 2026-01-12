import streamlit as st
import numpy as np
import librosa
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import tempfile
import os

# Konfigurasi target dimensi (sesuaikan dengan nilai saat pelatihan)
TARGET_HEIGHT = 96  
TARGET_WIDTH = 96  
TARGET_CHANNELS = 3  

# Inisialisasi LabelEncoder (sesuaikan dengan label saat pelatihan)
le = LabelEncoder()
le.fit(['Acoustic guitar', 'Piano'])  # Ganti sesuai kelas Anda

# Load model
@st.cache_resource
def load_trained_model():
    return load_model('instrument_classifier.h5')

model = load_trained_model()

def preprocess_audio_for_prediction(audio_path, label_encoder, target_h, target_w, target_c):
    # Load audio file
    data, sample_rate = librosa.load(audio_path, sr=None)

    # Extract MFCC features
    mfccs = librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=20)  # pastikan n_mfcc sesuai
    mfccs_processed = np.mean(mfccs.T, axis=0)  # rata-rata sepanjang waktu → (20,)

    # Reshape: (1, 20, 1, 1)
    reshaped_features = mfccs_processed[np.newaxis, :, np.newaxis, np.newaxis]

    current_height = reshaped_features.shape[1]  # 20
    current_width = reshaped_features.shape[2]   # 1

    pad_h = max(0, target_h - current_height)
    pad_w = max(0, target_w - current_width)

    padded_features = np.pad(
        reshaped_features,
        ((0, 0), (0, pad_h), (0, pad_w), (0, 0)),
        mode='constant',
        constant_values=0
    )

    # Ulangi channel untuk mencocokkan input model (misal: RGB → 3 channel)
    final_features = np.repeat(padded_features, target_c, axis=-1)
    return final_features

# UI Streamlit
st.title("🎵 Klasifikasi Instrumen Musik")
st.write("Unggah file audio (.wav) untuk mengklasifikasikan instrumen: **Piano** atau **Acoustic Guitar**.")

uploaded_file = st.file_uploader("Pilih file audio", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")

    # Simpan file sementara
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_audio_path = tmp_file.name

    try:
        # Preproses audio
        processed_audio = preprocess_audio_for_prediction(
            tmp_audio_path, le, TARGET_HEIGHT, TARGET_WIDTH, TARGET_CHANNELS
        )

        # Prediksi
        prediction_probs = model.predict(processed_audio)
        predicted_class_idx = np.argmax(prediction_probs, axis=1)[0]
        confidence = np.max(prediction_probs)

        predicted_label = le.inverse_transform([predicted_class_idx])[0]

        # Tampilkan hasil
        st.success(f"🔍 **Prediksi Instrumen**: `{predicted_label}`")
        st.info(f"📈 **Confidence**: {confidence:.4f}")

        # Opsional: tampilkan probabilitas per kelas
        class_names = le.classes_
        st.write("**Distribusi Probabilitas:**")
        prob_dict = {class_name: float(prob) for class_name, prob in zip(class_names, prediction_probs[0])}
        st.json(prob_dict)

    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses audio: {e}")
    finally:
        # Hapus file sementara
        if os.path.exists(tmp_audio_path):
            os.remove(tmp_audio_path)
else:
    st.info("Silakan unggah file audio dalam format `.wav`.")