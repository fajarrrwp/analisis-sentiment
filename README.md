# Analisis Sentimen Opini Publik Pilgub Jatim dengan _Machine Learning_ dan _Lexicon-Based_

Proyek ini adalah aplikasi web yang dibangun menggunakan Flask untuk melakukan analisis sentimen terhadap data teks, khususnya opini publik dari Twitter mengenai Pemilihan Gubernur Jawa Timur (Pilgub Jatim). Aplikasi ini mengimplementasikan dua pendekatan utama: _Lexicon-Based_ untuk pelabelan data awal dan _Machine Learning_ (Naive Bayes & Logistic Regression) untuk klasifikasi teks.

## 🌟 Fitur Utama

-   **Klasifikasi Teks Tunggal:** Menganalisis sentimen (positif, negatif, atau netral) dari satu kalimat atau paragraf yang dimasukkan oleh pengguna.
-   **Klasifikasi _Batch_:** Memungkinkan pengguna untuk mengunggah file `.csv` atau `.txt` yang berisi banyak data teks untuk dianalisis sekaligus.
-   **Perbandingan Model:** Menampilkan perbandingan akurasi dan metrik evaluasi lainnya (precision, recall, f1-score) antara model Naive Bayes dan Logistic Regression.
-   **Visualisasi Data:** Menghasilkan _WordCloud_ untuk setiap sentimen dan _Confusion Matrix_ untuk evaluasi model.
-   **Sistem Pengguna:** Fitur registrasi dan login untuk pengguna, memungkinkan penyimpanan riwayat klasifikasi secara personal.
-   **_Dashboard_ Pengguna:** Halaman _dashboard_ yang menampilkan ringkasan dan statistik aktivitas klasifikasi pengguna.
-   **Riwayat Klasifikasi:** Menyimpan dan menampilkan riwayat analisis yang pernah dilakukan oleh pengguna yang telah login, dengan fitur filter berdasarkan kata kunci, sentimen, dan rentang tanggal.

## ⚙️ Alur Kerja Proyek

Proyek ini memiliki alur kerja yang terstruktur mulai dari pengolahan data mentah hingga menjadi aplikasi web yang fungsional:

1.  **Pengumpulan Data**: Data teks mentah berupa _tweet_ terkait Pilgub Jatim dikumpulkan dalam format `.csv`.
2.  **Pra-pemrosesan Data (_Preprocessing_)**: Teks mentah dibersihkan melalui serangkaian proses untuk menghasilkan data yang siap diolah:
    -   **_Case Folding_**: Mengubah semua teks menjadi huruf kecil.
    -   **Penghapusan Karakter & Simbol**: Menghilangkan _hashtag_, URL, angka, dan karakter yang tidak relevan.
    -   **_Tokenization_**: Memecah kalimat menjadi token (kata-kata).
    -   **Normalisasi**: Mengubah kata-kata tidak baku menjadi kata baku menggunakan kamus slang (`kbba.txt`).
    -   **_Stopword Removal_**: Menghapus kata-kata umum yang tidak memiliki makna sentimen (misalnya: "yang", "di", "dan").
    -   **_Stemming_**: Mengubah kata-kata ke bentuk dasarnya.
3.  **Pelabelan Awal (_Lexicon-Based_)**: Data yang telah bersih kemudian diberi label sentimen (positif, negatif, netral) secara otomatis menggunakan kamus leksikon sentimen.
4.  **Pelatihan Model _Machine Learning_**: Data yang telah memiliki label digunakan untuk melatih dua model klasifikasi:
    -   **_Feature Extraction_**: Teks diubah menjadi representasi numerik menggunakan TF-IDF.
    -   **Pelatihan**: Model **Naive Bayes** dan **Logistic Regression** dilatih menggunakan data fitur tersebut.
    -   **Penyimpanan Model**: _Vectorizer_ dan model yang telah dilatih disimpan ke dalam file `.pkl` untuk digunakan oleh aplikasi web.
5.  **Aplikasi Web (Flask)**: Aplikasi web berfungsi sebagai antarmuka utama yang mengintegrasikan semua fungsi, memungkinkan pengguna berinteraksi dengan model yang telah dilatih.

## 🛠️ Teknologi yang Digunakan

-   **Backend**: Python, Flask
-   **Frontend**: HTML, CSS, Bootstrap 5, Chart.js
-   **Machine Learning**: Scikit-learn (Naive Bayes, Logistic Regression, TF-IDF)
-   **Pra-pemrosesan Teks**: Pandas, NLTK, Sastrawi
-   **Database**: SQLAlchemy (untuk manajemen pengguna dan riwayat)

## 🚀 Cara Menjalankan Aplikasi

Untuk menjalankan aplikasi ini di lingkungan lokal, ikuti langkah-langkah berikut:

1.  **Clone Repositori**
    ```bash
    git clone [https://github.com/fajarrrwp/analisis-sentiment.git](https://github.com/fajarrrwp/analisis-sentiment.git)
    cd "analisis-sentiment/analisis-sentiment-main/main-final - 2 - Login"
    ```

2.  **Buat Lingkungan Virtual (Opsional tapi Direkomendasikan)**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Untuk Windows: venv\Scripts\activate
    ```

3.  **Instal Dependensi**
    Pastikan Anda memiliki file `requirements.txt`. Jika belum ada, Anda bisa membuatnya dari daftar _library_ yang diimpor di `app.py`.
    ```bash
    pip install -r requirements.txt
    ```
    *Catatan: Jika `requirements.txt` tidak tersedia, instal _library_ berikut secara manual: `flask`, `flask_sqlalchemy`, `flask_login`, `werkzeug`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `wordcloud`, `nltk`, `sastrawi`.*

4.  **Unduh _Corpus_ NLTK (jika diperlukan)**
    Jalankan Python interpreter dan unduh paket `punkt` dan `stopwords`.
    ```python
    import nltk
    nltk.download('punkt')
    nltk.download('stopwords')
    ```

5.  **Latih Model (Jika File `.pkl` Belum Ada)**
    Jika file-file model seperti `vectorizer.pkl`, `naive_bayes_model.pkl`, dll. belum tersedia, Anda perlu menjalankannya dari _Jupyter Notebook_ yang relevan untuk melatih dan menyimpan model terlebih dahulu.

6.  **Jalankan Aplikasi Flask**
    ```bash
    python app.py
    ```

7.  **Akses Aplikasi**
    Buka _browser_ Anda dan akses alamat `http://127.0.0.1:5000`.

## 📂 Struktur File Penting

-   `app.py`: Logika utama aplikasi web Flask.
-   `preprocessing.ipynb`: _Notebook_ untuk semua langkah pra-pemrosesan teks.
-   `Labeling.ipynb`: _Notebook_ untuk proses pelabelan data menggunakan metode _lexicon-based_.
-   `model_training.ipynb` (diasumsikan): _Notebook_ untuk melatih model _machine learning_ dan menyimpan hasilnya.
-   `/templates`: Berisi file-file HTML yang menjadi tampilan antarmuka aplikasi.
-   `/static`: Berisi file-file statis seperti CSS dan JavaScript.
-   `*.pkl`: File-file model dan _vectorizer_ yang telah dilatih.
-   `*.csv`, `*.tsv`, `*.txt`: Dataset dan file kamus leksikon.
-   `instance/sentiment_data.db`: _Database_ SQLite untuk menyimpan data pengguna dan riwayat.
