import hashlib
import io
import json
import os
import pickle
import random
import sqlite3
import time
import threading  # Ditambahkan untuk menangani background processing agar sistem tidak down sementara
from flask import Flask, Response, redirect, render_template, request, url_for, flash, session
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Konfigurasi Jalur File Dinamis Berbasis Struktur Baru
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Lokasi folder 'model/'

DB_PATH = os.path.join(BASE_DIR, "data", "experts.db")
VECTORIZER_PATH = os.path.join(BASE_DIR, "saved_models", "tfidf_vectorizer.pkl")
MATRIX_PATH = os.path.join(BASE_DIR, "saved_models", "tfidf_matrix.pkl")

# Mengakses folder 'config/' dan 'templates/' yang naik 1 tingkat dari folder 'model/'
CONFIG_JSON_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "config", "config.json"))
STOPWORDS_JSON_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "config", "indonesian_stopwords.json"))
TEMPLATE_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "templates"))

app = Flask(__name__, template_folder=TEMPLATE_DIR)
app.secret_key = "kunci_rahasia_sistem_pakar"  # Ditambahkan untuk mendukung session flash message

# Lock untuk sinkronisasi akses ke model global
model_lock = threading.Lock()

def load_indonesian_stopwords():
    """
    Membaca daftar kata umum (stopwords) bahasa Indonesia dari berkas JSON eksternal.
    """
    if os.path.exists(STOPWORDS_JSON_PATH):
        with open(STOPWORDS_JSON_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

# Muat daftar kata umum ke dalam variabel global aplikasi
INDONESIAN_STOPWORDS = load_indonesian_stopwords()

# Inisialisasi database
def init_db():
    """
    Make tabel database SQLite jika belum tersedia di dalam sistem.
    """
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS experts (
            id TEXT PRIMARY KEY,
            name TEXT,
            department TEXT,
            email TEXT,
            description TEXT,
            UNIQUE(name, email)
        )
    """
    )
    conn.commit()
    conn.close()

init_db()

# Pipeline pelatihan ulang model ai (Proses Ekstraksi & Fit Model)
def retrain_tfidf_model():
    """
    Melatih ulang model TF-IDF secara terpusat berdasarkan data kompetensi terbaru dari database.
    """
    try:
        # Langkah 1: Pastikan tabel terinisialisasi dan ambil semua deskripsi pakar dari database SQLite
        init_db()
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT description FROM experts")
        all_descriptions = [row[0] for row in cursor.fetchall()]
        conn.close()

        # Fallback jika database masih kosong
        if not all_descriptions:
            all_descriptions = ["pakar kompetensi umum internal perusahaan"]

        # Langkah 2: Inisialisasi TfidfVectorizer dengan stop words bahasa Indonesia
        new_vectorizer = TfidfVectorizer(lowercase=True, stop_words=INDONESIAN_STOPWORDS)
        try:
            # Langkah 3: Hitung bobot TF-IDF (Term Frequency - Inverse Document Frequency) dari seluruh deskripsi pakar
            new_tfidf_matrix = new_vectorizer.fit_transform(all_descriptions)
        except ValueError as e:
            # Mengatasi error jika vocabulary kosong (misal semua teks hanya stopwords)
            print(f"PERINGATAN: TfidfVectorizer gagal melatih model ({e}). Menggunakan kata kunci fallback.")
            all_descriptions = ["pakar kompetensi umum internal perusahaan"]
            new_vectorizer = TfidfVectorizer(lowercase=True, stop_words=INDONESIAN_STOPWORDS)
            new_tfidf_matrix = new_vectorizer.fit_transform(all_descriptions)

        os.makedirs(os.path.dirname(VECTORIZER_PATH), exist_ok=True)
        
        # Langkah 4: Simpan model dan matriks bobot yang telah dilatih menggunakan Pickle (.pkl) secara aman
        tmp_vectorizer_path = VECTORIZER_PATH + ".tmp"
        tmp_matrix_path = MATRIX_PATH + ".tmp"
        
        with open(tmp_vectorizer_path, "wb") as f:
            pickle.dump(new_vectorizer, f)
        with open(tmp_matrix_path, "wb") as f:
            pickle.dump(new_tfidf_matrix, f)
            
        os.replace(tmp_vectorizer_path, VECTORIZER_PATH)
        os.replace(tmp_matrix_path, MATRIX_PATH)
        print("INFO: Model TF-IDF berhasil dilatih ulang dan disimpan secara aman.")
    except Exception as e:
        print(f"ERROR: Gagal melakukan retrain model TF-IDF: {e}")

# Muat sumber daya ke memori (Proses Caching data ke RAM)
def load_resources():
    """
    Memuat kembali model TF-IDF, matriks bobot teks, dan seluruh data pakar ke dalam memori aplikasi.
    """
    global vectorizer, tfidf_matrix, df_experts
    
    with model_lock:
        try:
            # Pastikan tabel database terinisialisasi
            init_db()
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM experts")
            row_count = cursor.fetchone()[0]
            conn.close()

            if row_count == 0:
                print("INFO: Database kosong. Sumber daya model tidak akan dimuat ke memori.")
                vectorizer = None
                tfidf_matrix = None
                df_experts = pd.DataFrame(columns=["ID", "Name", "Department", "Email", "Description"])
                return

            # Cek apakah berkas model hilang atau jumlah data tidak sinkron dengan database
            model_missing = not os.path.exists(VECTORIZER_PATH) or not os.path.exists(MATRIX_PATH)
            need_retrain = model_missing

            if not model_missing:
                try:
                    with open(MATRIX_PATH, "rb") as f:
                        temp_matrix = pickle.load(f)
                    if temp_matrix.shape[0] != row_count:
                        need_retrain = True
                        print(f"INFO: Model out-of-sync (Model: {temp_matrix.shape[0]}, DB: {row_count}).")
                except Exception:
                    need_retrain = True

            if need_retrain:
                print("INFO: Memicu pelatihan ulang model AI secara otomatis agar sinkron...")
                retrain_tfidf_model()

            # Langkah 5: Muat model vectorizer dan tfidf_matrix yang tersimpan ke RAM untuk melayani pencarian cepat
            with open(VECTORIZER_PATH, "rb") as f:
                vectorizer = pickle.load(f)
            with open(MATRIX_PATH, "rb") as f:
                tfidf_matrix = pickle.load(f)

            # Langkah 6: Muat seluruh data pakar dari SQLite ke Pandas DataFrame global
            init_db()
            conn = sqlite3.connect(DB_PATH)
            df_experts = pd.read_sql_query(
                "SELECT id AS ID, name AS Name, department AS Department, email AS Email, description AS Description FROM experts",
                conn,
            )
            conn.close()
            print("INFO: Sumber daya model sukses dimuat ke dalam memori.")
        except Exception as e:
            print(f"ERROR: Gagal memuat sumber daya model ke memori: {e}")

# Fungsi pembungkus untuk menjalankan pelatihan ulang secara asynchronous di background thread
def start_background_retrain():
    def run():
        retrain_tfidf_model()
        load_resources()
    threading.Thread(target=run, daemon=True).start()

try:
    load_resources()
except Exception:
    retrain_tfidf_model()
    load_resources()

# Engine AI Generator Deskripsi (Dinamis Berbasis TF-IDF & JSON)
def ai_description_generator(department, keywords):
    """
    Menghasilkan teks narasi deskripsi keahlian secara dinamis menggunakan model TF-IDF 
    berdasarkan konfigurasi ambang batas dari config.json.
    """
    global vectorizer, tfidf_matrix
    
    threshold = 0.05  
    max_ref = 2       
    
    if os.path.exists(CONFIG_JSON_PATH):
        try:
            with open(CONFIG_JSON_PATH, "r", encoding="utf-8") as f:
                config_data = json.load(f)
                threshold = config_data.get("ai_generator", {}).get("min_similarity_threshold", threshold)
                max_ref = config_data.get("ai_generator", {}).get("max_reference_fallback", max_ref)
        except Exception as e:
            print(f"PERINGATAN: Gagal memuat config.json: {e}")

    try:
        init_db()
        conn = sqlite3.connect(DB_PATH)
        df_db = pd.read_sql_query("SELECT department, description FROM experts", conn)
        conn.close()
    except Exception:
        df_db = pd.DataFrame(columns=["department", "description"])

    df_filtered = df_db[df_db["department"] == department].copy()

    # Salin objek global ke lokal di bawah lock
    with model_lock:
        local_vectorizer = vectorizer
        local_tfidf_matrix = tfidf_matrix

    if not df_filtered.empty and local_vectorizer is not None and local_tfidf_matrix is not None and keywords.strip():
        try:
            # Ubah kata kunci jadi vektor TF-IDF: Transformasi input keywords user ke format numerik
            keywords_vec = local_vectorizer.transform([keywords])
            # Hitung kemiripan (cosine similarity): Bandingkan keywords_vec dengan seluruh deskripsi di database
            sim_scores = cosine_similarity(keywords_vec, local_tfidf_matrix)[0]
            
            df_db["score"] = sim_scores
            # Saring pakar berdasarkan departemen yang sesuai dan melewati batas similarity threshold
            df_match = df_db[(df_db["department"] == department) & (df_db["score"] >= threshold)]
            
            if not df_match.empty:
                # Urutkan berdasarkan skor kemiripan tertinggi, ambil teks referensi teratas, lalu gabungkan sebagai saran deskripsi baru
                df_match = df_match.sort_values(by="score", ascending=False)
                top_descriptions = df_match["description"].head(max_ref).tolist()
                referenced_text = " ".join(top_descriptions)
                return f"Pakar rekomendasi berbasis kompetensi {department}: {referenced_text}"
                
        except Exception as e:
            print(f"PERINGATAN: Gagal memproses similarity pada generator: {e}")

    clean_keys = [k.strip() for k in keywords.split(",") if k.strip()]
    keys_str = ", ".join(clean_keys) if clean_keys else "kompetensi terkait"

    fallback_templates = [
        f"Spesialis berpengalaman di divisi {department}. Memiliki rekam jejak mendalam dalam penanganan area {keys_str} serta optimasi sistem kerja internal.",
        f"Personel ahli pada bidang {department}. Berfokus penuh pada implementasi teknologi {keys_str}, standarisasi operasional, dan manajemen risiko divisi."
    ]
    return random.choice(fallback_templates)

@app.route("/", methods=["GET", "POST"])
def index():
    """
    Menangani halaman utama aplikasi. Jika kueri kosong, mengembalikan seluruh data dari DB.
    """
    query = ""
    recommendations = []
    active_tab = "search"

    try:
        init_db()
        conn = sqlite3.connect(DB_PATH)
        df_current = pd.read_sql_query(
            "SELECT id AS ID, name AS Name, department AS Department, email AS Email, description AS Description FROM experts",
            conn,
        )
        conn.close()
    except Exception:
        df_current = pd.DataFrame(columns=["ID", "Name", "Department", "Email", "Description"])

    if request.method == "POST" and "search_submit" in request.form:
        query = request.form.get("query", "").strip()
        
        if query:
            # Ambil salinan lokal di bawah lock untuk menjaga konsistensi shape data
            with model_lock:
                local_df_experts = df_experts.copy() if df_experts is not None else pd.DataFrame(columns=["ID", "Name", "Department", "Email", "Description"])
                local_vectorizer = vectorizer
                local_tfidf_matrix = tfidf_matrix

            if query and not local_df_experts.empty and local_vectorizer is not None and local_tfidf_matrix is not None:
                try:
                    # Pengolahan data pencarian (transformasi input & cosine similarity)
                    # Langkah A: Transformasi teks kueri pencarian user ke dalam vektor TF-IDF
                    query_vector = local_vectorizer.transform([query])
                    
                    # Langkah B: Hitung skor kedekatan / kemiripan (cosine similarity) antara kueri dengan seluruh deskripsi pakar
                    cosine_sim = cosine_similarity(query_vector, local_tfidf_matrix)[0]

                    df_temp = local_df_experts.copy()
                    df_temp["Score"] = cosine_sim
                    
                    if "Score" in df_temp.columns:
                        # Langkah C: Saring data pakar yang memiliki nilai kecocokan di atas ambang batas (Score > 0.05)
                        # Langkah D: Urutkan data berdasarkan skor kemiripan terbesar ke terkecil (descending)
                        df_result = df_temp[df_temp["Score"] > 0.05].sort_values(by="Score", ascending=False)
                        recommendations = df_result.to_dict(orient="records")
                except Exception as e:
                    print(f"ERROR: Terjadi kegagalan saat pencarian TF-IDF: {e}")
                    recommendations = []

            # Simpan hasil kueri pencarian dan rekomendasi ke session (Penerapan Pola PRG)
            session["search_query"] = query
            session["search_results"] = recommendations
        else:
            session.pop("search_query", None)
            session.pop("search_results", None)

        return redirect(url_for("index"))

    # Untuk request GET:
    # Ambil data kueri & rekomendasi dari session jika diarahkan dari POST, lalu hapus instan dari session
    # sehingga saat user menekan refresh/F5, halaman kembali ke state awal yang kosong
    query = session.pop("search_query", "")
    recommendations = session.pop("search_results", [])

    return render_template(
        "index.html",
        query=query,
        recommendations=recommendations,
        active_tab=active_tab,
        experts=df_current.to_dict(orient="records") if not df_current.empty else [],
    )

@app.route("/generate-ai", methods=["POST"])
def generate_ai():
    department = request.form.get('divisi')
    keywords = request.form.get('keywords')
    generated_text = ai_description_generator(department, keywords)
    return {"status": "success", "text": generated_text}

@app.route("/add-expert", methods=["POST"])
def add_expert():
    name = request.form.get("name", "").strip()
    department = request.form.get("department", "").strip()
    email = request.form.get("email", "").strip()
    description = request.form.get("description", "").strip()

    if name and department and email and description:
        init_db()
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM experts")
        existing_ids = {row[0] for row in cursor.fetchall()}

        while True:
            seed = f"{name}{time.time_ns()}{random.randint(1000, 9999)}"
            git_hash_id = hashlib.md5(seed.encode()).hexdigest()[:7]
            if git_hash_id not in existing_ids:
                break

        cursor.execute(
            "INSERT OR IGNORE INTO experts (id, name, department, email, description) VALUES (?, ?, ?, ?, ?)",
            (git_hash_id, name, department, email, description),
        )
        conn.commit()
        conn.close()

        # Jalankan retraining secara sinkron agar model langsung siap digunakan
        retrain_tfidf_model()
        load_resources()
        flash("Pakar berhasil ditambahkan! AI selesai me-retrain model.", "success")

    return redirect(url_for("index") + "?tab=manage")

@app.route("/edit-expert/<expert_id>", methods=["POST"])
def edit_expert(expert_id):
    name = request.form.get("name", "").strip()
    department = request.form.get("department", "").strip()
    email = request.form.get("email", "").strip()
    description = request.form.get("description", "").strip()

    if name and department and email and description:
        init_db()
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE experts SET name=?, department=?, email=?, description=? WHERE id=?",
            (name, department, email, description, expert_id),
        )
        conn.commit()
        conn.close()

        # Jalankan retraining secara sinkron agar model langsung siap digunakan
        retrain_tfidf_model()
        load_resources()
        flash("Pakar berhasil diperbarui! AI selesai me-retrain model.", "success")

    return redirect(url_for("index") + "?tab=manage")

@app.route("/delete-expert/<expert_id>", methods=["GET", "POST"])
def delete_expert(expert_id):
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM experts WHERE id=?", (expert_id,))
    conn.commit()
    conn.close()

    # Jalankan retraining secara sinkron agar model langsung siap digunakan
    retrain_tfidf_model()
    load_resources()
    flash("Pakar berhasil dihapus! AI selesai me-retrain model.", "success")
    return redirect(url_for("index") + "?tab=manage")

@app.route("/delete-all-experts", methods=["GET", "POST"])
def delete_all_experts():
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM experts")
    conn.commit()
    conn.close()

    # Jalankan retraining secara sinkron agar model langsung siap digunakan
    retrain_tfidf_model()
    load_resources()
    flash("Semua pakar berhasil dihapus! AI selesai me-retrain model.", "success")

    return redirect(url_for("index") + "?tab=manage")

@app.route("/download-template", methods=["GET"])
def download_template():
    output = io.StringIO()
    output.write("Nama,Divisi,Email,Deskripsi Keahlian\n")
    output.write(
        'contoh: Prof. Andi,IT Infrastructure,andi@company.com,"Pakar infrastruktur server Linux dan jaringan internet LAN wifi."\n'
    )
    response = Response(output.getvalue(), mimetype="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=template_pakar_baru.csv"
    return response

@app.route("/import-csv", methods=["POST"])
def import_csv():
    if "file_csv" not in request.files:
        return redirect(url_for("index") + "?tab=manage")

    file = request.files["file_csv"]
    if file.filename == "":
        return redirect(url_for("index") + "?tab=manage")

    if file and file.filename.endswith(".csv"):
        try:
            uploaded_df = pd.read_csv(file)
            uploaded_df.columns = [col.strip() for col in uploaded_df.columns]

            required_cols = ["Nama", "Divisi", "Email", "Deskripsi Keahlian"]
            if not all(col in uploaded_df.columns for col in required_cols):
                return "Error: Format kolom CSV tidak sesuai template contoh!", 400

            init_db()
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute("SELECT id, name, department, description FROM experts")
            db_rows = cursor.fetchall()
            existing_ids = {row[0] for row in db_rows}
            # Kunci tuple unik untuk mendeteksi duplikat (nama, divisi, deskripsi)
            existing_records = {(row[1].strip().lower(), row[2].strip().lower(), row[3].strip().lower()) for row in db_rows}

            for _, row in uploaded_df.iterrows():
                if pd.isna(row["Nama"]) or "contoh:" in str(row["Nama"]):
                    continue

                name_val = str(row["Nama"]).strip()
                dept_val = str(row["Divisi"]).strip()
                email_val = str(row["Email"]).strip()
                desc_val = str(row["Deskripsi Keahlian"]).strip()

                # Lewati penambahan pakar jika data nama, divisi, dan deskripsi keahlian sama persis
                record_key = (name_val.lower(), dept_val.lower(), desc_val.lower())
                if record_key in existing_records:
                    continue

                while True:
                    seed = f"{name_val}{time.time_ns()}{random.randint(1000, 9999)}"
                    git_hash_id = hashlib.md5(seed.encode()).hexdigest()[:7]
                    if git_hash_id not in existing_ids:
                        existing_ids.add(git_hash_id)
                        break

                cursor.execute(
                    "INSERT OR IGNORE INTO experts (id, name, department, email, description) VALUES (?, ?, ?, ?, ?)",
                    (git_hash_id, name_val, dept_val, email_val, desc_val),
                )
                existing_records.add(record_key)

            conn.commit()
            conn.close()

            # Jalankan retraining secara sinkron agar model langsung siap digunakan
            retrain_tfidf_model()
            load_resources()
            flash("Import CSV berhasil dijalankan! AI selesai me-retrain model.", "success")

        except Exception as e:
            return f"Terjadi kesalahan saat memproses data CSV: {str(e)}", 500

    return redirect(url_for("index") + "?tab=manage")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)