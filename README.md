# 🎵 Klasifikasi Instrumen Musik (Piano vs Acoustic Guitar)

Aplikasi berbasis **Streamlit** untuk mengklasifikasikan instrumen musik dari file audio **(.wav)** menggunakan **Deep Learning (CNN)** dan ekstraksi fitur **MFCC**.  
Instrumen yang didukung saat ini:
- 🎹 Piano  
- 🎸 Acoustic Guitar  

---

## 📌 Fitur Utama
- Upload file audio format **.wav**
- Pemutaran audio langsung di aplikasi
- Prediksi jenis instrumen musik
- Menampilkan **confidence score**
- Menampilkan **distribusi probabilitas** setiap kelas
- Antarmuka sederhana dan interaktif

---

## 🖼️ Tampilan Aplikasi

### Halaman Upload Audio
![Upload Audio](Screenshot%202026-01-18%20165108.png)

### Hasil Prediksi Instrumen
![Hasil Prediksi](Screenshot%202026-01-18%20165138.png)

---

## ⚙️ Teknologi yang Digunakan
- **Python**
- **Streamlit** (Web App)
- **Librosa** (Audio Processing)
- **TensorFlow / Keras** (Deep Learning Model)
- **Scikit-learn** (Label Encoding)
- **NumPy**

---

## 🧠 Alur Kerja Sistem
1. Pengguna mengunggah file audio `.wav`
2. Audio diproses menggunakan **Librosa**
3. Ekstraksi fitur **MFCC**
4. MFCC dirata-ratakan dan di-*padding* agar sesuai input model
5. Data diprediksi menggunakan model CNN (`instrument_classifier.h5`)
6. Sistem menampilkan:
   - Prediksi instrumen
   - Nilai confidence
   - Distribusi probabilitas

---


---

## 🧪 Preprocessing Audio
- Sample rate asli audio digunakan
- Ekstraksi **20 MFCC**
- Rata-rata MFCC sepanjang waktu
- Padding ke ukuran:
  - Height: 96
  - Width: 96
  - Channels: 3 (RGB-like)

---

## 🚀 Cara Menjalankan Aplikasi

### 1. Clone Repository

git clone https://github.com/username/klasifikasi-instrumen-musik.git
cd klasifikasi-instrumen-musik


### 2. Install Dependensi
pip install streamlit librosa tensorflow scikit-learn numpy

### 3. Jalankan Streamlit
streamlit run app.py
http://localhost:8501

---

## Identitas Mahasiswa
 - Nama: Bondan Tiur Mahendra
 - NIM: 202211420067



