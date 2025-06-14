import io
import base64
import logging
import ast # Untuk safe_process_text
import os
import uuid
from datetime import datetime, date
import json # Untuk mengubah dict ke string JSON
import random # Ditambahkan untuk shuffle contoh cepat

from flask import Flask, render_template, request, session, send_file, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg') # Penting untuk environment tanpa GUI seperti server Flask
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter

# --- Konfigurasi & Konstanta ---
APP_SECRET_KEY = 'fajar_ganti_ini_dengan_kunci_rahasia_yang_sangat_aman_dan_unik_sekali_lagi!' # SANGAT PENTING: Ganti!
VECTORIZER_PATH = 'vectorizer.pkl'
NAIVE_BAYES_MODEL_PATH = 'naive_bayes_model.pkl'
LOGISTIC_REGRESSION_MODEL_PATH = 'logistic_regression_model.pkl'
LABEL_ENCODER_PATH = 'text_label_encoder.pkl'
DATASET_PATH = 'hasil4.csv'
HISTORY_SESSION_KEY = 'classification_history_anonymous' 
TEMP_FOLDER = 'temp_files'

# --- Inisialisasi Aplikasi Flask & Ekstensi ---
app = Flask(__name__)
app.config['SECRET_KEY'] = APP_SECRET_KEY
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///sentiment_data.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login_route' 
login_manager.login_message = "Mohon login terlebih dahulu untuk mengakses halaman ini."
login_manager.login_message_category = "warning"

# --- Pengaturan Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Model Database (SQLAlchemy) ---
class User(UserMixin, db.Model):
    __tablename__ = 'user'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    histories = db.relationship('History', backref='author', lazy='dynamic')

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def __repr__(self):
        return f'<User {self.username}>'

class History(db.Model):
    __tablename__ = 'history'
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    input_text = db.Column(db.Text, nullable=False)
    naive_bayes_label = db.Column(db.String(50))
    naive_bayes_probability = db.Column(db.String(10))
    naive_bayes_all_probs = db.Column(db.Text)
    naive_bayes_accuracy = db.Column(db.String(10))
    logistic_regression_label = db.Column(db.String(50))
    logistic_regression_probability = db.Column(db.String(10))
    logistic_regression_all_probs = db.Column(db.Text)
    logistic_regression_accuracy = db.Column(db.String(10))
    actual_label_retrieved = db.Column(db.String(100))
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

    def __repr__(self):
        return f'<History {self.id} for User ID {self.user_id} at {self.timestamp}>'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# --- Pemuatan Resource Model Machine Learning & Dataset Global ---
vectorizer, naive_bayes_model, logistic_regression_model, text_label_encoder, df_global = None, None, None, None, pd.DataFrame()

def load_ml_resources():
    global vectorizer, naive_bayes_model, logistic_regression_model, text_label_encoder, df_global
    try:
        vectorizer = joblib.load(VECTORIZER_PATH)
        naive_bayes_model = joblib.load(NAIVE_BAYES_MODEL_PATH)
        logistic_regression_model = joblib.load(LOGISTIC_REGRESSION_MODEL_PATH)
        text_label_encoder = joblib.load(LABEL_ENCODER_PATH)
        logger.info("Vectorizer, model ML, dan label encoder berhasil dimuat.")
        
        df_temp = pd.read_csv(DATASET_PATH, encoding='utf-8', encoding_errors='replace')
        logger.info(f"Dataset global '{DATASET_PATH}' berhasil dimuat.")
        if 'tokens_stemmed' in df_temp.columns:
            def safe_process_text(s):
                if pd.isna(s): return ""
                if isinstance(s, str) and s.startswith('[') and s.endswith(']'):
                    try: return ' '.join(map(str, ast.literal_eval(s)))
                    except: return s 
                return str(s)
            df_temp['processed_text_for_lookup'] = df_temp['tokens_stemmed'].apply(safe_process_text)
        else:
            logger.warning(f"Kolom 'tokens_stemmed' tidak ditemukan di {DATASET_PATH}. 'processed_text_for_lookup' akan kosong.")
            df_temp['processed_text_for_lookup'] = "" 
        df_global = df_temp
        logger.info("Dataset global telah diproses untuk lookup.")
    except FileNotFoundError as fnf_error:
        logger.error(f"File resource penting tidak ditemukan: {fnf_error}. Aplikasi mungkin tidak berfungsi dengan benar.")
    except Exception as e:
        logger.error(f"Gagal memuat resource .pkl atau dataset global: {e}. Aplikasi mungkin tidak berfungsi dengan benar.")

if not os.path.exists(TEMP_FOLDER):
    try:
        os.makedirs(TEMP_FOLDER)
        logger.info(f"Folder sementara '{TEMP_FOLDER}' berhasil dibuat.")
    except OSError as e:
        logger.error(f"Gagal membuat folder sementara '{TEMP_FOLDER}': {e}")

# --- Fungsi Helper ---
def classify_text_with_model(text_input, model_object, current_vectorizer, current_label_encoder, model_name_for_log="Unknown"):
    if not all([model_object, current_vectorizer, current_label_encoder]):
        logger.error(f"[{model_name_for_log}] Salah satu komponen model (model, vectorizer, encoder) belum dimuat.")
        return {"label": "Error: Model/Komponen Hilang", "probability": 0.0, "all_probabilities": {}}
    if not text_input or not isinstance(text_input, str) or not text_input.strip():
        return {"label": "Input Tidak Valid/Kosong", "probability": 0.0, "all_probabilities": {}}
    try:
        text_vec_sparse = current_vectorizer.transform([text_input])
        prediction_numeric = model_object.predict(text_vec_sparse)
        prediction_string = current_label_encoder.inverse_transform(prediction_numeric)[0]
        probabilities_all_classes = model_object.predict_proba(text_vec_sparse)[0]
        predicted_class_index = prediction_numeric[0]
        probability_of_predicted_class = probabilities_all_classes[predicted_class_index]
        all_probs_dict = {label: prob for label, prob in zip(current_label_encoder.classes_, probabilities_all_classes)}
        return {"label": prediction_string, "probability": probability_of_predicted_class, "all_probabilities": all_probs_dict}
    except Exception as e:
        logger.error(f"[{model_name_for_log}] Error saat klasifikasi '{text_input[:50]}...': {e}")
        return {"label": "Error Prediksi", "probability": 0.0, "all_probabilities": {}}

def get_actual_label_from_df(text_input, dataframe, text_column_for_lookup='processed_text_for_lookup', label_column='label'):
    if dataframe.empty or text_column_for_lookup not in dataframe.columns or label_column not in dataframe.columns:
        return "N/A (Dataframe Error)"
    if not isinstance(text_input, str): text_input = str(text_input)
    valid_rows = dataframe[dataframe[text_column_for_lookup].notna()]
    matching_rows = valid_rows[valid_rows[text_column_for_lookup].astype(str).str.contains(text_input, case=False, na=False)]
    if not matching_rows.empty:
        return matching_rows[label_column].values[0] 
    return "Data tidak ditemukan di CSV"

def calculate_single_instance_accuracy(prediction, actual_label):
    if actual_label == "N/A (Dataframe Error)" or actual_label == "Data tidak ditemukan di CSV" or \
       prediction.startswith("Error") or prediction == "Input Tidak Valid/Kosong":
        return 0.0 
    return 1.0 if prediction == actual_label else 0.0
    
def generate_wordcloud_base64(text_series, width=400, height=200):
    valid_texts = text_series.dropna().astype(str)
    text_for_wc = ' '.join(valid_texts) if not (valid_texts.empty or valid_texts.str.strip().eq('').all()) else "tidak ada data"
    try:
        wc = WordCloud(width=width, height=height, background_color='white').generate(text_for_wc)
        img_io = io.BytesIO(); wc.to_image().save(img_io, format='PNG'); img_io.seek(0)
        return base64.b64encode(img_io.getvalue()).decode('utf-8')
    except Exception as e: logger.error(f"Error wordcloud: {e}"); return None

def get_quick_examples_from_dataset():
    quick_examples = []
    if df_global is not None and not df_global.empty and \
       'label' in df_global.columns and 'processed_text_for_lookup' in df_global.columns:
        labels_to_sample = ['positif', 'negatif', 'netral']
        for sentiment_label in labels_to_sample:
            sentiment_samples_df = df_global[df_global['label'] == sentiment_label]
            if not sentiment_samples_df.empty:
                try:
                    sample_row = sentiment_samples_df.sample(1).iloc[0]
                    quick_examples.append({
                        'text': sample_row['processed_text_for_lookup'],
                        'label': sentiment_label 
                    })
                except Exception as e:
                    logger.warning(f"Gagal mengambil sampel untuk label '{sentiment_label}' (quick examples): {e}")
        if quick_examples: random.shuffle(quick_examples)
        else: logger.warning("Tidak ada sampel cocok untuk contoh cepat (helper).")
    else: logger.warning("Dataset global kosong/kolom hilang, contoh cepat tidak dimuat (helper).")
    return quick_examples

def get_word_importances_lr(text_input, current_vectorizer, lr_model_obj, current_label_encoder, top_n=5):
    if not text_input or not text_input.strip() or not all([current_vectorizer, lr_model_obj, current_label_encoder]):
        return [], [] 
    try:
        text_vec = current_vectorizer.transform([text_input])
        pred_numeric = lr_model_obj.predict(text_vec)[0]
        
        if lr_model_obj.coef_.shape[0] > 1: 
            class_coefficients = lr_model_obj.coef_[pred_numeric]
        else: 
            class_coefficients = lr_model_obj.coef_[0] 
        
        feature_names = current_vectorizer.get_feature_names_out()
        word_coeffs = list(zip(feature_names, class_coefficients))
        input_tokens = set(text_input.lower().split())
        relevant_word_coeffs = [{'word': word, 'coeff': coeff} for word, coeff in word_coeffs if word in input_tokens]
        
        sorted_coeffs = sorted(relevant_word_coeffs, key=lambda x: x['coeff'], reverse=True)
        
        influential_words_positive = [{'word': item['word'], 'importance': item['coeff']} for item in sorted_coeffs if item['coeff'] > 0][:top_n]
        influential_words_negative = [{'word': item['word'], 'importance': item['coeff']} for item in reversed(sorted_coeffs) if item['coeff'] < 0][:top_n]
        
        return influential_words_positive, influential_words_negative
    except Exception as e:
        logger.error(f"Error mendapatkan word importances: {e}")
        return [], []

# --- Routes ---
@app.route('/')
def home():
    history_preview = []
    if current_user.is_authenticated:
        history_preview = History.query.filter_by(user_id=current_user.id)\
                                     .order_by(History.timestamp.desc())\
                                     .limit(3).all()
    else:
        session_history = session.get(HISTORY_SESSION_KEY, [])
        history_preview = session_history[-3:][::-1] 
    
    quick_examples_data = get_quick_examples_from_dataset()
            
    return render_template('index.html', 
                           history=history_preview, 
                           quick_examples=quick_examples_data)

@app.route('/register', methods=['GET', 'POST'])
def register_route():
    if current_user.is_authenticated: return redirect(url_for('home'))
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        error = None
        if not username: error = 'Username wajib diisi.'
        elif not password: error = 'Password wajib diisi.'
        elif len(password) < 6: error = 'Password minimal 6 karakter.'
        elif password != confirm_password: error = 'Konfirmasi password tidak cocok.'
        
        if error is None:
            existing_user = User.query.filter_by(username=username).first()
            if existing_user:
                flash(f"Username '{username}' sudah terdaftar. Silakan gunakan username lain.", 'danger')
            else:
                new_user = User(username=username)
                new_user.set_password(password)
                db.session.add(new_user)
                db.session.commit()
                logger.info(f"User baru terdaftar: {username}")
                flash(f"Registrasi berhasil untuk user {username}! Silakan login.", 'success')
                return redirect(url_for('login_route'))
        else:
            flash(error, 'danger')
    return render_template('register.html', title="Registrasi")

@app.route('/login', methods=['GET', 'POST'])
def login_route():
    if current_user.is_authenticated: return redirect(url_for('home'))
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        remember_me = True if request.form.get('remember_me') else False
        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password):
            login_user(user, remember=remember_me)
            logger.info(f"User '{username}' berhasil login.")
            flash(f"Selamat datang kembali, {username}!", 'success')
            next_page = request.args.get('next')
            return redirect(next_page or url_for('home'))
        else:
            logger.warning(f"Upaya login gagal untuk username: {username}")
            flash('Login gagal. Periksa kembali username dan password Anda.', 'danger')
    return render_template('login.html', title="Login")

@app.route('/logout')
@login_required
def logout_route():
    logger.info(f"User '{current_user.username}' melakukan logout.")
    logout_user()
    flash('Anda telah berhasil logout.', 'info')
    return redirect(url_for('home'))

# Di dalam app.py, di bagian Routes

@app.route('/dashboard')
@login_required # Hanya pengguna yang login bisa mengakses dashboard
def dashboard_route():
    user_histories = History.query.filter_by(author=current_user).order_by(History.timestamp.desc()).all()

    total_classifications = len(user_histories)

    # Hitung distribusi sentimen untuk Naive Bayes
    nb_sentiments_list = [h.naive_bayes_label for h in user_histories if h.naive_bayes_label and not h.naive_bayes_label.startswith("Error")]
    nb_sentiment_counts = Counter(nb_sentiments_list)

    # Hitung distribusi sentimen untuk Logistic Regression
    lr_sentiments_list = [h.logistic_regression_label for h in user_histories if h.logistic_regression_label and not h.logistic_regression_label.startswith("Error")]
    lr_sentiment_counts = Counter(lr_sentiments_list)

    # Ambil beberapa aktivitas terbaru (misalnya 5 terakhir)
    recent_activities = user_histories[:5]

    # Data untuk chart (kirim sebagai dict agar mudah di-JSON-kan di template)
    chart_data_nb = dict(nb_sentiment_counts)
    chart_data_lr = dict(lr_sentiment_counts)

    return render_template('dashboard.html', 
                           title="Dashboard Pengguna",
                           total_classifications=total_classifications,
                           nb_sentiment_counts=nb_sentiment_counts, # Untuk tampilan list
                           lr_sentiment_counts=lr_sentiment_counts, # Untuk tampilan list
                           chart_data_nb=chart_data_nb, # Untuk Chart.js
                           chart_data_lr=chart_data_lr, # Untuk Chart.js
                           recent_activities=recent_activities)

@app.route('/classify', methods=['POST'])
def classify_text_input_route():
    if not all([vectorizer, naive_bayes_model, logistic_regression_model, text_label_encoder]):
        flash("Model tidak siap, silakan coba lagi nanti atau hubungi administrator.", "danger")
        return redirect(url_for('home'))

    input_text = request.form.get('text_input', '').strip()
    if not input_text:
        flash("Input teks tidak boleh kosong.", "warning")
        return redirect(url_for('home'))

    nb_prediction_data = classify_text_with_model(input_text, naive_bayes_model, vectorizer, text_label_encoder, "NaiveBayes")
    lr_prediction_data = classify_text_with_model(input_text, logistic_regression_model, vectorizer, text_label_encoder, "LogisticRegression")

    nb_result_str = nb_prediction_data["label"]
    nb_probability_float = nb_prediction_data["probability"]
    nb_all_probs_dict = nb_prediction_data["all_probabilities"]
    lr_result_str = lr_prediction_data["label"]
    lr_probability_float = lr_prediction_data["probability"]
    lr_all_probs_dict = lr_prediction_data["all_probabilities"]
    actual_label = get_actual_label_from_df(input_text, df_global)
    nb_accuracy_float = calculate_single_instance_accuracy(nb_result_str, actual_label)
    lr_accuracy_float = calculate_single_instance_accuracy(lr_result_str, actual_label)

    lr_positive_influencers, lr_negative_influencers = [], []
    if lr_result_str not in ["Error: Komponen Hilang", "Input Tidak Valid/Kosong", "Error Prediksi"]:
        if logistic_regression_model and vectorizer and text_label_encoder:
            lr_positive_influencers, lr_negative_influencers = get_word_importances_lr(
                input_text, vectorizer, logistic_regression_model, text_label_encoder
            )

    if current_user.is_authenticated:
        try:
            new_history_entry = History(
                input_text=input_text,
                naive_bayes_label=nb_result_str,
                naive_bayes_probability=f"{nb_probability_float:.2%}",
                naive_bayes_all_probs=json.dumps(nb_all_probs_dict),
                naive_bayes_accuracy=f"{nb_accuracy_float:.2f}",
                logistic_regression_label=lr_result_str,
                logistic_regression_probability=f"{lr_probability_float:.2%}",
                logistic_regression_all_probs=json.dumps(lr_all_probs_dict),
                logistic_regression_accuracy=f"{lr_accuracy_float:.2f}",
                actual_label_retrieved=actual_label,
                user_id=current_user.id
            )
            db.session.add(new_history_entry)
            db.session.commit()
            logger.info(f"Riwayat klasifikasi disimpan ke DB untuk user ID: {current_user.id}")
        except Exception as e_db:
            db.session.rollback(); logger.error(f"Gagal menyimpan riwayat ke DB: {e_db}")
            flash("Gagal menyimpan riwayat klasifikasi Anda.", "danger")
    else:
        temp_history_anon = session.get(HISTORY_SESSION_KEY, [])
        history_entry_anon = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'input_text': input_text,
            'naive_bayes_label': nb_result_str, 'naive_bayes_accuracy': f"{nb_accuracy_float:.2f}",
            'naive_bayes_probability': f"{nb_probability_float:.2%}", 'naive_bayes_all_probs': nb_all_probs_dict,
            'logistic_regression_label': lr_result_str, 'logistic_regression_accuracy': f"{lr_accuracy_float:.2f}",
            'logistic_regression_probability': f"{lr_probability_float:.2%}", 'logistic_regression_all_probs': lr_all_probs_dict,
            'actual_label_retrieved': actual_label
        }
        temp_history_anon.append(history_entry_anon)
        session[HISTORY_SESSION_KEY] = temp_history_anon

    current_classification_data = {
        'last_input_text': input_text, 'nb_result': nb_result_str,
        'nb_probability': f"{nb_probability_float:.2%}", 'nb_all_probs': nb_all_probs_dict,
        'lr_result': lr_result_str, 'lr_probability': f"{lr_probability_float:.2%}",
        'lr_all_probs': lr_all_probs_dict, 'nb_accuracy_str': f"{nb_accuracy_float:.2f}",
        'lr_accuracy_str': f"{lr_accuracy_float:.2f}", 'actual_label_for_display': actual_label,
        'lr_positive_influencers': lr_positive_influencers, 
        'lr_negative_influencers': lr_negative_influencers
    }
    
    history_preview = []
    if current_user.is_authenticated:
        history_preview = History.query.filter_by(user_id=current_user.id).order_by(History.timestamp.desc()).limit(3).all()
    else:
        session_history = session.get(HISTORY_SESSION_KEY, [])
        history_preview = session_history[-3:][::-1]

    quick_examples_data = get_quick_examples_from_dataset()

    return render_template('index.html', 
                           **current_classification_data, 
                           history=history_preview,
                           quick_examples=quick_examples_data)

@app.route('/klasifikasi_batch', methods=['GET', 'POST'])
def klasifikasi_batch_route():
    if request.method == 'POST':
        if not current_user.is_authenticated:
            flash("Anda harus login untuk melakukan klasifikasi dataset batch.", "warning")
            return redirect(url_for('login_route', next=request.url))
        
        if 'dataset_file' not in request.files:
            flash("Tidak ada file yang dipilih.", "danger")
            return render_template('klasifikasi_batch.html', error="Tidak ada file yang dipilih.")
        file = request.files['dataset_file']
        kolom_teks = request.form.get('kolom_teks', 'teks').strip()
        if not kolom_teks: kolom_teks = 'teks'

        if file.filename == '':
            flash("Tidak ada file yang dipilih.", "danger")
            return render_template('klasifikasi_batch.html', error="Tidak ada file yang dipilih.")

        if file and (file.filename.lower().endswith('.csv') or file.filename.lower().endswith('.txt')):
            try:
                df_upload = None
                if file.filename.lower().endswith('.txt'):
                    string_io = io.StringIO(file.stream.read().decode("UTF-8", errors='replace'), newline=None)
                    lines = [line.strip() for line in string_io if line.strip()]
                    if not lines: 
                        flash("File TXT kosong atau tidak berisi data teks yang valid.", "warning")
                        return render_template('klasifikasi_batch.html', error="File TXT kosong atau tidak berisi data teks yang valid.")
                    df_upload = pd.DataFrame({kolom_teks: lines})
                elif file.filename.lower().endswith('.csv'):
                    df_upload = pd.read_csv(file.stream, encoding='utf-8', encoding_errors='replace')
                
                if df_upload is None or df_upload.empty:
                    flash("Gagal memuat data dari file atau file kosong.", "warning")
                    return render_template('klasifikasi_batch.html', error="Gagal memuat data dari file atau file kosong.")
                if kolom_teks not in df_upload.columns:
                    flash(f"Kolom '{kolom_teks}' tidak ditemukan di file. Kolom tersedia: {', '.join(df_upload.columns.tolist())}", "danger")
                    return render_template('klasifikasi_batch.html', error=f"Kolom '{kolom_teks}' tidak ditemukan.")
                
                hasil_klasifikasi = []
                if not all([vectorizer, naive_bayes_model, logistic_regression_model, text_label_encoder]):
                     flash("Model tidak siap untuk klasifikasi. Hubungi administrator.", "danger")
                     return render_template('klasifikasi_batch.html', error="Model tidak siap.")

                for index, row in df_upload.iterrows():
                    teks_input = str(row[kolom_teks]) if pd.notna(row[kolom_teks]) else ""
                    pred_nb_data = {"label": "Input Kosong/NaN", "probability": 0.0}
                    pred_lr_data = {"label": "Input Kosong/NaN", "probability": 0.0}
                    if teks_input.strip():
                        pred_nb_data = classify_text_with_model(teks_input, naive_bayes_model, vectorizer, text_label_encoder, "NaiveBayes-Batch")
                        pred_lr_data = classify_text_with_model(teks_input, logistic_regression_model, vectorizer, text_label_encoder, "LogisticRegression-Batch")
                    
                    data_baris = row.to_dict()
                    data_baris['Prediksi_Naive_Bayes'] = pred_nb_data["label"]
                    data_baris['Probabilitas_NB'] = f"{pred_nb_data['probability']:.2%}"
                    data_baris['Prediksi_Logistic_Regression'] = pred_lr_data["label"]
                    data_baris['Probabilitas_LR'] = f"{pred_lr_data['probability']:.2%}"
                    hasil_klasifikasi.append(data_baris)
                
                df_hasil = pd.DataFrame(hasil_klasifikasi)
                temp_file_id = str(uuid.uuid4())
                temp_file_name = f"{temp_file_id}.csv"
                temp_file_path = os.path.join(TEMP_FOLDER, temp_file_name)
                try:
                    df_hasil.to_csv(temp_file_path, index=False, encoding='utf-8')
                    session['download_batch_file_id'] = temp_file_id
                    logger.info(f"File batch sementara: {temp_file_path} untuk user session.")
                except Exception as e_write:
                    logger.error(f"Gagal simpan file batch sementara '{temp_file_path}': {e_write}")
                    flash(f"Gagal mempersiapkan file unduhan: {e_write}", "danger")
                    return render_template('klasifikasi_batch.html', error=f"Gagal mempersiapkan file unduhan: {e_write}")

                summary_nb = df_hasil['Prediksi_Naive_Bayes'].value_counts().to_dict()
                summary_lr = df_hasil['Prediksi_Logistic_Regression'].value_counts().to_dict()
                flash("Dataset berhasil diklasifikasikan!", "success")
                return render_template('hasil_batch.html', 
                                       table_preview_html=df_hasil.head(20).to_html(classes='table table-sm table-striped table-hover', header="true", index=False, border=0),
                                       kolom_hasil=df_hasil.columns.tolist(), nama_file=file.filename,
                                       total_teks=len(df_hasil), summary_nb=summary_nb, summary_lr=summary_lr)
            except Exception as e:
                logger.error(f"Error proses batch '{file.filename}': {e}")
                flash(f"Terjadi kesalahan saat memproses file: {e}", "danger")
                return render_template('klasifikasi_batch.html', error=f"Terjadi kesalahan: {e}")
        else:
            flash("Format file tidak didukung (hanya .csv atau .txt).", "warning")
            return render_template('klasifikasi_batch.html', error="Format file tidak didukung.")
    return render_template('klasifikasi_batch.html')

@app.route('/unduh_hasil_batch')
def unduh_hasil_batch_route():
    temp_file_id = session.pop('download_batch_file_id', None)
    if temp_file_id:
        temp_file_name = f"{temp_file_id}.csv"
        temp_file_path = os.path.join(TEMP_FOLDER, temp_file_name)
        if os.path.exists(temp_file_path):
            try:
                response = send_file(temp_file_path, as_attachment=True, download_name='hasil_klasifikasi_batch.csv', mimetype='text/csv; charset=utf-8')
                return response
            except Exception as e_send: 
                logger.error(f"Error kirim file '{temp_file_path}': {e_send}")
            finally:
                if os.path.exists(temp_file_path):
                    try: 
                        os.remove(temp_file_path)
                        logger.info(f"File sementara '{temp_file_path}' dihapus.")
                    except OSError as e_remove: 
                        logger.error(f"Gagal hapus file sementara '{temp_file_path}': {e_remove}")
        else: 
            logger.warning(f"File sementara tidak ditemukan: '{temp_file_path}' (ID: {temp_file_id})")
    else: 
        logger.warning("Gagal unduh: tidak ada ID file di session.")
    flash("Tidak ada data unduhan, file tidak ditemukan, atau sesi berakhir. Ulangi klasifikasi batch.", "warning")
    return redirect(url_for('klasifikasi_batch_route'))

@app.route('/model_comparison')
def model_comparison_view():
    if df_global.empty or 'processed_text_for_lookup' not in df_global.columns or 'label' not in df_global.columns:
        flash("Data untuk perbandingan model tidak tersedia.", "warning")
        return render_template('model_comparison.html', error_message="Data perbandingan tidak tersedia.")
    if not all([vectorizer, naive_bayes_model, logistic_regression_model, text_label_encoder]):
         flash("Model/vectorizer tidak siap untuk perbandingan.", "danger")
         return render_template('model_comparison.html', error_message="Model/vectorizer tidak siap.")

    X_texts = df_global['processed_text_for_lookup']
    y_true_str = df_global['label']
    cm_nb_base64, cm_lr_base64, nb_report, lr_report = None, None, {}, {}
    nb_acc, lr_acc = 0.0, 0.0
    try:
        y_true_encoded = text_label_encoder.transform(y_true_str)
        X_vec = vectorizer.transform(X_texts)
        nb_pred_num = naive_bayes_model.predict(X_vec)
        lr_pred_num = logistic_regression_model.predict(X_vec)
        nb_acc = accuracy_score(y_true_encoded, nb_pred_num)
        lr_acc = accuracy_score(y_true_encoded, lr_pred_num)
        report_names = list(text_label_encoder.classes_)
        nb_report = classification_report(y_true_encoded, nb_pred_num, target_names=report_names, output_dict=True, zero_division=0)
        lr_report = classification_report(y_true_encoded, lr_pred_num, target_names=report_names, output_dict=True, zero_division=0)
        
        cm_nb = confusion_matrix(y_true_encoded, nb_pred_num, labels=text_label_encoder.transform(report_names))
        plt.figure(figsize=(6, 4)); sns.heatmap(cm_nb, annot=True, fmt='d', cmap='Blues', xticklabels=report_names, yticklabels=report_names, cbar=False)
        plt.xlabel('Prediksi'); plt.ylabel('Aktual'); plt.title('Confusion Matrix - Naive Bayes', fontsize=10)
        img_io_nb = io.BytesIO(); plt.savefig(img_io_nb, format='PNG', bbox_inches='tight'); plt.close(); img_io_nb.seek(0)
        cm_nb_base64 = base64.b64encode(img_io_nb.getvalue()).decode('utf-8')

        cm_lr = confusion_matrix(y_true_encoded, lr_pred_num, labels=text_label_encoder.transform(report_names))
        plt.figure(figsize=(6, 4)); sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Greens', xticklabels=report_names, yticklabels=report_names, cbar=False)
        plt.xlabel('Prediksi'); plt.ylabel('Aktual'); plt.title('Confusion Matrix - Logistic Regression', fontsize=10)
        img_io_lr = io.BytesIO(); plt.savefig(img_io_lr, format='PNG', bbox_inches='tight'); plt.close(); img_io_lr.seek(0)
        cm_lr_base64 = base64.b64encode(img_io_lr.getvalue()).decode('utf-8')
    except Exception as e:
        logger.error(f"Error perbandingan model: {e}")
        flash(f"Error saat membuat perbandingan: {str(e)}", "danger")
        return render_template('model_comparison.html', error_message=f"Error: {str(e)}", 
                               nb_accuracy_overall=f"{nb_acc:.2f}", lr_accuracy_overall=f"{lr_acc:.2f}",
                               nb_classification_report=nb_report, lr_classification_report=lr_report,
                               cm_nb_base64=cm_nb_base64, cm_lr_base64=cm_lr_base64)
    return render_template('model_comparison.html', 
                           nb_accuracy_overall=f"{nb_acc:.2f}", lr_accuracy_overall=f"{lr_acc:.2f}",
                           nb_classification_report=nb_report, lr_classification_report=lr_report,
                           cm_nb_base64=cm_nb_base64, cm_lr_base64=cm_lr_base64)

@app.route('/wordcloud_images')
def wordcloud_images_view():
    if df_global.empty or 'processed_text_for_lookup' not in df_global.columns or 'label' not in df_global.columns:
        flash("Data untuk wordcloud tidak tersedia.", "warning")
        return redirect(url_for('home')) 
    positive = df_global[df_global['label'] == 'positif']['processed_text_for_lookup']
    negative = df_global[df_global['label'] == 'negatif']['processed_text_for_lookup']
    neutral = df_global[df_global['label'] == 'netral']['processed_text_for_lookup']
    return render_template('wordcloud.html',
                           positive_wordcloud_img=generate_wordcloud_base64(positive),
                           negative_wordcloud_img=generate_wordcloud_base64(negative),
                           neutral_wordcloud_img=generate_wordcloud_base64(neutral))

# Di dalam app.py

@app.route('/history')
@login_required
def classification_history_view():
    # Ambil parameter filter dari URL, dengan nilai default jika tidak ada
    search_query = request.args.get('search_query', '').strip()
    sentiment_filter = request.args.get('sentiment_filter', 'semua').strip().lower() # 'semua', 'positif', 'negatif', 'netral'
    start_date_str = request.args.get('start_date', '').strip()
    end_date_str = request.args.get('end_date', '').strip()

    # Query dasar untuk riwayat pengguna yang login
    query = History.query.filter_by(user_id=current_user.id) # atau filter_by(author=current_user)

    # Terapkan filter pencarian teks
    if search_query:
        query = query.filter(History.input_text.ilike(f'%{search_query}%'))

    # Terapkan filter sentimen (contoh: berdasarkan Naive Bayes label)
    # Anda bisa memilih salah satu model atau keduanya (dengan OR)
    if sentiment_filter and sentiment_filter != 'semua':
        # Pastikan nilai sentiment_filter cocok dengan yang disimpan di DB (misal, 'positif', bukan 'Positif')
        query = query.filter(db.func.lower(History.naive_bayes_label) == sentiment_filter)
        # Jika ingin filter berdasarkan kedua model (salah satu cocok):
        # from sqlalchemy import or_
        # query = query.filter(or_(
        #     db.func.lower(History.naive_bayes_label) == sentiment_filter,
        #     db.func.lower(History.logistic_regression_label) == sentiment_filter
        # ))

    # Terapkan filter tanggal
    start_date_obj = None
    if start_date_str:
        try:
            start_date_obj = datetime.strptime(start_date_str, '%Y-%m-%d').date()
            # Filter dari awal hari pada start_date
            query = query.filter(History.timestamp >= datetime.combine(start_date_obj, datetime.min.time()))
        except ValueError:
            flash('Format tanggal awal tidak valid. Gunakan format YYYY-MM-DD.', 'warning')

    end_date_obj = None
    if end_date_str:
        try:
            end_date_obj = datetime.strptime(end_date_str, '%Y-%m-%d').date()
            # Filter hingga akhir hari pada end_date
            query = query.filter(History.timestamp <= datetime.combine(end_date_obj, datetime.max.time()))
        except ValueError:
            flash('Format tanggal akhir tidak valid. Gunakan format YYYY-MM-DD.', 'warning')

    # Urutkan hasil setelah semua filter diterapkan (terbaru dulu)
    query = query.order_by(History.timestamp.desc())

    # Paginasi untuk query yang sudah difilter
    page = request.args.get('page', 1, type=int)
    user_history_page_data = query.paginate(page=page, per_page=10, error_out=False) 

    # Kirim nilai filter yang aktif kembali ke template agar form bisa diisi ulang
    active_filters = {
        'search_query': search_query,
        'sentiment_filter': sentiment_filter,
        'start_date': start_date_str,
        'end_date': end_date_str
    }

    # Ambil daftar label unik untuk dropdown filter sentimen (jika text_label_encoder ada)
    sentiment_options = ['semua']
    if text_label_encoder:
        sentiment_options.extend(list(text_label_encoder.classes_))
    else: # Fallback jika encoder tidak ada
        sentiment_options.extend(['positif', 'negatif', 'netral'])


    return render_template('history.html', 
                           title="Riwayat Klasifikasi Saya", 
                           history_page_data=user_history_page_data,
                           active_filters=active_filters,
                           sentiment_options=sentiment_options) # Kirim opsi sentimen

# --- Main ---
if __name__ == '__main__':
    with app.app_context():
        db.create_all() 
        logger.info("Tabel database diperiksa/dibuat jika belum ada.")
        load_ml_resources()

    if not all([vectorizer, naive_bayes_model, logistic_regression_model, text_label_encoder]):
        logger.critical("APLIKASI GAGAL DIMULAI: Resource ML penting (vectorizer/model/encoder) gagal dimuat.")
    else:
        logger.info("Semua resource ML berhasil dimuat, menjalankan aplikasi Flask...")
        app.run(debug=True, host='0.0.0.0', port=5000)