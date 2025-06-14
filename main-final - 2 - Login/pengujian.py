import pandas as pd
import joblib
import time
from sklearn.model_selection import train_test_split
import ast

# Konfigurasi (sesuaikan dengan nama file Anda)
VECTORIZER_PATH = 'vectorizer.pkl'
NAIVE_BAYES_MODEL_PATH = 'naive_bayes_model.pkl'
LOGISTIC_REGRESSION_MODEL_PATH = 'logistic_regression_model.pkl'
DATASET_PATH = 'hasil4.csv'

def preprocess_text_for_testing(series_text):
    def convert_stringified_list_to_text(s):
        if pd.isna(s): return ""
        if isinstance(s, str) and s.startswith('[') and s.endswith(']'):
            try: return ' '.join(map(str, ast.literal_eval(s)))
            except: return s
        return str(s)
    return series_text.apply(convert_stringified_list_to_text)

# Muat semua komponen
try:
    vectorizer = joblib.load(VECTORIZER_PATH)
    nb_model = joblib.load(NAIVE_BAYES_MODEL_PATH)
    lr_model = joblib.load(LOGISTIC_REGRESSION_MODEL_PATH)
    df = pd.read_csv(DATASET_PATH)
    df.dropna(subset=['tokens_stemmed', 'label'], inplace=True)

    X_texts = preprocess_text_for_testing(df['tokens_stemmed'])
    y_labels = df['label']

    # Vektorisasi semua teks
    X_vec = vectorizer.transform(X_texts)

    # Ambil test set (bisa disamakan dengan split di pelatihan, atau sampel acak)
    _, X_test, _, _ = train_test_split(X_vec, y_labels, test_size=0.2, random_state=42)

    print(f"Menguji waktu prediksi pada {X_test.shape[0]} sampel data uji...")

    # Uji Naive Bayes
    start_time_nb = time.time()
    for i in range(X_test.shape[0]):
        nb_model.predict(X_test[i])
    end_time_nb = time.time()
    total_time_nb = end_time_nb - start_time_nb
    avg_time_nb_ms = (total_time_nb / X_test.shape[0]) * 1000

    # Uji Logistic Regression
    start_time_lr = time.time()
    for i in range(X_test.shape[0]):
        lr_model.predict(X_test[i])
    end_time_lr = time.time()
    total_time_lr = end_time_lr - start_time_lr
    avg_time_lr_ms = (total_time_lr / X_test.shape[0]) * 1000

    print("\n--- Hasil Uji Efisiensi ---")
    print(f"Waktu Prediksi Rata-rata Naive Bayes: {avg_time_nb_ms:.4f} ms per teks")
    print(f"Waktu Prediksi Rata-rata Logistic Regression: {avg_time_lr_ms:.4f} ms per teks")

except FileNotFoundError as e:
    print(f"Error: File tidak ditemukan. Pastikan semua file .pkl dan CSV ada. ({e})")
except Exception as e:
    print(f"Terjadi error: {e}")