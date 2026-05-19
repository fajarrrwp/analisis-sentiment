# Analisis Sentimen Opini Publik Pilgub Jatim dengan _Machine Learning_ dan _Lexicon-Based_

Proyek ini adalah aplikasi web yang dibangun menggunakan Flask untuk melakukan analisis sentimen terhadap data teks, khususnya opini publik dari Twitter mengenai Pemilihan Gubernur Jawa Timur (Pilgub Jatim). Aplikasi ini mengimplementasikan dua pendekatan utama: _Lexicon-Based_ untuk pelabelan data awal dan _Machine Learning_ (Naive Bayes & Logistic Regression) untuk klasifikasi teks.

## Fitur Utama

- **Klasifikasi Teks Tunggal:** Menganalisis sentimen (positif, negatif, atau netral) dari satu kalimat atau paragraf yang dimasukkan oleh pengguna.
- **Klasifikasi _Batch_:** Memungkinkan pengguna untuk mengunggah file `.csv` atau `.txt` yang berisi banyak data teks untuk dianalisis sekaligus.
- **Perbandingan Model:** Menampilkan perbandingan akurasi dan metrik evaluasi lainnya (precision, recall, f1-score) antara model Naive Bayes dan Logistic Regression.
- **Visualisasi Data:** Menghasilkan _WordCloud_ untuk setiap sentimen dan _Confusion Matrix_ untuk evaluasi model.
- **Sistem Pengguna:** Fitur registrasi dan login untuk pengguna, memungkinkan penyimpanan riwayat klasifikasi secara personal.
- **_Dashboard_ Pengguna:** Halaman _dashboard_ yang menampilkan ringkasan dan statistik aktivitas klasifikasi pengguna.
- **Riwayat Klasifikasi:** Menyimpan dan menampilkan riwayat analisis yang pernah dilakukan oleh pengguna yang telah login, dengan fitur filter berdasarkan kata kunci, sentimen, dan rentang tanggal.

## Alur Kerja Proyek

1. **Pengumpulan Data**: Data teks mentah berupa _tweet_ terkait Pilgub Jatim dikumpulkan dalam format `.csv`.
2. **Pra-pemrosesan Data (_Preprocessing_)**: Teks mentah dibersihkan melalui serangkaian proses:
   - **_Case Folding_**: Mengubah semua teks menjadi huruf kecil.
   - **Penghapusan Karakter & Simbol**: Menghilangkan _hashtag_, URL, angka, dan karakter yang tidak relevan.
   - **_Tokenization_**: Memecah kalimat menjadi token (kata-kata).
   - **Normalisasi**: Mengubah kata-kata tidak baku menjadi kata baku menggunakan kamus slang (`lexicons/kbba.txt`).
   - **_Stopword Removal_**: Menghapus kata-kata umum yang tidak memiliki makna sentimen.
   - **_Stemming_**: Mengubah kata-kata ke bentuk dasarnya.
3. **Pelabelan Awal (_Lexicon-Based_)**: Data yang telah bersih kemudian diberi label sentimen secara otomatis menggunakan kamus leksikon di `lexicons/`.
4. **Pelatihan Model _Machine Learning_**:
   - **_Feature Extraction_**: Teks diubah menjadi representasi numerik menggunakan TF-IDF.
   - **Pelatihan**: Model **Naive Bayes** dan **Logistic Regression** dilatih.
   - **Penyimpanan Model**: _Vectorizer_ dan model yang telah dilatih disimpan di `models/`.
5. **Aplikasi Web (Flask)**: Aplikasi web mengintegrasikan semua fungsi.

## Teknologi yang Digunakan

- **Backend**: Python, Flask
- **Frontend**: HTML, CSS, Bootstrap 5, Chart.js
- **Machine Learning**: Scikit-learn (Naive Bayes, Logistic Regression, TF-IDF)
- **Pra-pemrosesan Teks**: Pandas, NLTK, Sastrawi
- **Database**: SQLAlchemy + SQLite (untuk manajemen pengguna dan riwayat)

## Struktur Proyek

```
analisis-sentiment/
├── README.md
├── .gitignore
└── app/
    ├── app.py                # Logika utama aplikasi web Flask
    ├── pengujian.py          # Script benchmark waktu prediksi model
    ├── requirements.txt
    ├── data/                 # Dataset (CSV)
    │   ├── PilgubJatim1.csv
    │   ├── PilgubJatim1_no_hashtags.csv
    │   ├── Data_tokens_stemmed.csv
    │   ├── tokens_stemmed.csv
    │   ├── hasil4.csv
    │   └── hasil4_dataset_string.csv
    ├── lexicons/             # Kamus leksikon & slang
    │   ├── kbba.txt
    │   ├── positive.tsv
    │   ├── negative.tsv
    │   └── netral.tsv
    ├── models/               # Vectorizer & model terlatih
    │   ├── vectorizer.pkl
    │   ├── naive_bayes_model.pkl
    │   ├── logistic_regression_model.pkl
    │   └── text_label_encoder.pkl
    ├── notebooks/            # Notebook eksplorasi & pelatihan
    │   ├── preprocessing.ipynb
    │   ├── Labeling.ipynb
    │   ├── netral.ipynb
    │   └── database.ipynb
    ├── static/               # Asset statis (CSS)
    │   └── css/style.css
    ├── templates/            # Template HTML
    │   ├── index.html
    │   ├── login.html
    │   ├── register.html
    │   ├── dashboard.html
    │   ├── history.html
    │   ├── klasifikasi_batch.html
    │   ├── hasil_batch.html
    │   ├── model_comparison.html
    │   └── wordcloud.html
    ├── instance/             # SQLite DB (dibuat otomatis saat run)
    └── temp_files/           # File sementara hasil batch (auto-managed)
```

> Catatan: Path data/lexicon/model di dalam `app.py` dan `pengujian.py` sudah disesuaikan dengan struktur subfolder ini. Notebook di `notebooks/` masih merujuk path datar (relative ke direktori notebook). Jika ingin menjalankan ulang notebook, jalankan dari direktori `app/` atau sesuaikan path di sel notebook.

## Cara Menjalankan Aplikasi

1. **Clone Repositori**
   ```bash
   git clone https://github.com/fajarrrwp/analisis-sentiment.git
   cd analisis-sentiment
   ```

2. **Buat Lingkungan Virtual (Direkomendasikan)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Untuk Windows: venv\Scripts\activate
   ```

3. **Instal Dependensi**
   ```bash
   pip install -r app/requirements.txt
   ```

4. **Unduh _Corpus_ NLTK (jika diperlukan)**
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   ```

5. **Jalankan Aplikasi Flask**
   ```bash
   cd app
   python app.py
   ```

6. **Akses Aplikasi**
   Buka _browser_ Anda dan akses `http://127.0.0.1:5000`.

## Pelatihan Ulang Model

Jika file `.pkl` di `models/` belum tersedia atau ingin dilatih ulang, jalankan notebook di `notebooks/` secara berurutan: `preprocessing.ipynb` → `Labeling.ipynb` → notebook training (jika ada). Pastikan output `vectorizer.pkl`, `naive_bayes_model.pkl`, `logistic_regression_model.pkl`, dan `text_label_encoder.pkl` disimpan di folder `app/models/`.
