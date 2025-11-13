# 🎯 Amazon Toy Recommender

Aplikasi **rekomendasi mainan Amazon** berbasis **Content-Based Filtering** yang dibangun menggunakan **Streamlit** dan **Machine Learning (TF-IDF)**.  
Proyek ini bertujuan membantu pengguna menemukan mainan serupa berdasarkan deskripsi, kategori, harga, rating, dan fitur lainnya.

---

## 🚀 Fitur Utama

✅ **Rekomendasi Mainan Serupa**  
Pengguna dapat mencari mainan berdasarkan nama produk, lalu sistem akan memberikan rekomendasi mainan yang paling mirip.

✅ **Pencarian Pintar (TF-IDF)**  
Menggunakan teknik **TF-IDF Vectorization** pada deskripsi produk, kategori, dan subkategori untuk menilai kesamaan antar mainan.

✅ **Antarmuka Interaktif**  
Dibangun dengan **Streamlit** sehingga mudah digunakan langsung di browser.

✅ **Tema Gelap (Dark Mode)**  
Tampilan modern dan nyaman di mata dengan tema gelap menyeluruh.

---

## 🧠 Teknologi yang Digunakan

| Komponen | Teknologi |
|-----------|------------|
| Bahasa Pemrograman | Python 3.11+ |
| Framework UI | Streamlit |
| Machine Learning | scikit-learn |
| Vectorization | TF-IDF (Term Frequency - Inverse Document Frequency) |
| Data Source | Dataset mainan Amazon (Kaggle) |
| Deployment | Streamlit Cloud |

---

## 📁 Struktur Folder

```bash
amazon-toy-recommender/
│
├── app.py # File utama Streamlit
├── model_utils.py # Modul pemrosesan data & model
├── data/
│ └── amazon_toys.csv # Dataset mainan Amazon
├── requirements.txt # Daftar dependensi
├── .env # Variabel lingkungan (jika digunakan)
└── README.md # Dokumentasi proyek
```

---

## ⚙️ Cara Menjalankan Secara Lokal

1. **Clone repository**
   ```bash
   git clone https://github.com/XiaoFai17/amazon-toy-recommender.git
   cd amazon-toy-recommender
   ```
2. **Buat virtual environment dan aktifkan**
   ```bash
   python -m venv venv
   source venv/bin/activate        # Mac (bash/zsh)
   venv\Scripts\activate           # Windows PowerShell
   ```
3. **Install dependensi**
   ```bash
   pip install -r requirements.txt
   ```
4. **Jalankan aplikasi**
   ```bash
   streamlit run app.py
   ```
5. **Buka di browser**
   ```bash
   http://localhost:8501
   ```
---
## 💡 **Preview**
Kamu bisa mencoba aplikasi yang sudah di-deploy di sini:

👉 https://amazon-toy-recommender.streamlit.app/