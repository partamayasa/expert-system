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
PROBLEM_VECTORIZER_PATH = os.path.join(BASE_DIR, "saved_models", "problem_tfidf_vectorizer.pkl")
PROBLEM_MATRIX_PATH = os.path.join(BASE_DIR, "saved_models", "problem_tfidf_matrix.pkl")

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
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS problem_solving (
            id TEXT PRIMARY KEY,
            expert_id TEXT,
            problem TEXT,
            solution TEXT,
            FOREIGN KEY(expert_id) REFERENCES experts(id) ON DELETE CASCADE
        )
    """
    )
    conn.commit()
    conn.close()

init_db()

# Inisialisasi variabel model global
vectorizer = None
tfidf_matrix = None
df_experts = pd.DataFrame(columns=["ID", "Name", "Department", "Email", "Description"])

problem_vectorizer = None
problem_tfidf_matrix = None
df_problems = pd.DataFrame(columns=["ID", "ExpertID", "Problem", "Solution"])

# Pipeline pelatihan ulang model ai (Proses Ekstraksi & Fit Model)
def retrain_tfidf_model():
    """
    Melatih ulang model TF-IDF secara terpusat berdasarkan data kompetensi terbaru dan riwayat masalah dari database.
    """
    try:
        # Langkah 1: Pastikan tabel terinisialisasi dan ambil data dari database SQLite
        init_db()
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute("SELECT description FROM experts")
        all_descriptions = [row[0] for row in cursor.fetchall()]
        
        cursor.execute("SELECT problem, solution FROM problem_solving")
        all_problems = [f"{row[0]} {row[1]}" for row in cursor.fetchall()]
        
        conn.close()

        # Fallback jika database pakar kosong
        if not all_descriptions:
            all_descriptions = ["pakar kompetensi umum internal perusahaan"]

        # Fallback jika database masalah kosong
        if not all_problems:
            all_problems = ["kendala masalah sistem jaringan komputer internet koneksi"]

        # Langkah 2: Inisialisasi TfidfVectorizer untuk Expert
        new_vectorizer = TfidfVectorizer(lowercase=True, stop_words=INDONESIAN_STOPWORDS)
        try:
            new_tfidf_matrix = new_vectorizer.fit_transform(all_descriptions)
        except ValueError as e:
            print(f"PERINGATAN: TfidfVectorizer gagal melatih model expert ({e}). Menggunakan kata kunci fallback.")
            all_descriptions = ["pakar kompetensi umum internal perusahaan"]
            new_vectorizer = TfidfVectorizer(lowercase=True, stop_words=INDONESIAN_STOPWORDS)
            new_tfidf_matrix = new_vectorizer.fit_transform(all_descriptions)

        # Langkah 2b: Inisialisasi TfidfVectorizer untuk Problems
        new_problem_vectorizer = TfidfVectorizer(lowercase=True, stop_words=INDONESIAN_STOPWORDS)
        try:
            new_problem_tfidf_matrix = new_problem_vectorizer.fit_transform(all_problems)
        except ValueError as e:
            print(f"PERINGATAN: TfidfVectorizer gagal melatih model problem ({e}). Menggunakan kata kunci fallback.")
            all_problems = ["kendala masalah sistem jaringan komputer internet koneksi"]
            new_problem_vectorizer = TfidfVectorizer(lowercase=True, stop_words=INDONESIAN_STOPWORDS)
            new_problem_tfidf_matrix = new_problem_vectorizer.fit_transform(all_problems)

        os.makedirs(os.path.dirname(VECTORIZER_PATH), exist_ok=True)
        
        # Simpan model expert secara aman
        tmp_vectorizer_path = VECTORIZER_PATH + ".tmp"
        tmp_matrix_path = MATRIX_PATH + ".tmp"
        with open(tmp_vectorizer_path, "wb") as f:
            pickle.dump(new_vectorizer, f)
        with open(tmp_matrix_path, "wb") as f:
            pickle.dump(new_tfidf_matrix, f)
        os.replace(tmp_vectorizer_path, VECTORIZER_PATH)
        os.replace(tmp_matrix_path, MATRIX_PATH)
        
        # Simpan model problem secara aman
        tmp_problem_vectorizer_path = PROBLEM_VECTORIZER_PATH + ".tmp"
        tmp_problem_matrix_path = PROBLEM_MATRIX_PATH + ".tmp"
        with open(tmp_problem_vectorizer_path, "wb") as f:
            pickle.dump(new_problem_vectorizer, f)
        with open(tmp_problem_matrix_path, "wb") as f:
            pickle.dump(new_problem_tfidf_matrix, f)
        os.replace(tmp_problem_vectorizer_path, PROBLEM_VECTORIZER_PATH)
        os.replace(tmp_problem_matrix_path, PROBLEM_MATRIX_PATH)
        
        print("INFO: Model TF-IDF Expert & Problem berhasil dilatih ulang dan disimpan secara aman.")
    except Exception as e:
        print(f"ERROR: Gagal melakukan retrain model TF-IDF: {e}")

# Muat sumber daya ke memori (Proses Caching data ke RAM)
def load_resources():
    """
    Memuat kembali model TF-IDF, matriks bobot teks, data pakar, dan riwayat masalah ke dalam memori aplikasi.
    """
    global vectorizer, tfidf_matrix, df_experts
    global problem_vectorizer, problem_tfidf_matrix, df_problems
    
    with model_lock:
        try:
            # Pastikan tabel database terinisialisasi
            init_db()
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM experts")
            row_count = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM problem_solving")
            problem_count = cursor.fetchone()[0]
            conn.close()

            # Cek apakah berkas model hilang
            expert_model_missing = not os.path.exists(VECTORIZER_PATH) or not os.path.exists(MATRIX_PATH)
            problem_model_missing = not os.path.exists(PROBLEM_VECTORIZER_PATH) or not os.path.exists(PROBLEM_MATRIX_PATH)
            
            need_retrain = expert_model_missing or problem_model_missing

            # Validasi sinkronisasi baris dengan bentuk matriks
            if not expert_model_missing and row_count > 0:
                try:
                    with open(MATRIX_PATH, "rb") as f:
                        temp_matrix = pickle.load(f)
                    if temp_matrix.shape[0] != row_count:
                        need_retrain = True
                        print(f"INFO: Model Expert out-of-sync (Model: {temp_matrix.shape[0]}, DB: {row_count}).")
                except Exception:
                    need_retrain = True

            if not problem_model_missing and problem_count > 0:
                try:
                    with open(PROBLEM_MATRIX_PATH, "rb") as f:
                        temp_p_matrix = pickle.load(f)
                    if temp_p_matrix.shape[0] != problem_count:
                        need_retrain = True
                        print(f"INFO: Model Problem out-of-sync (Model: {temp_p_matrix.shape[0]}, DB: {problem_count}).")
                except Exception:
                    need_retrain = True

            if need_retrain:
                print("INFO: Memicu pelatihan ulang model AI secara otomatis agar sinkron...")
                retrain_tfidf_model()

            # Muat model expert
            if os.path.exists(VECTORIZER_PATH) and os.path.exists(MATRIX_PATH):
                with open(VECTORIZER_PATH, "rb") as f:
                    vectorizer = pickle.load(f)
                with open(MATRIX_PATH, "rb") as f:
                    tfidf_matrix = pickle.load(f)
            else:
                vectorizer = None
                tfidf_matrix = None

            # Muat model problem
            if os.path.exists(PROBLEM_VECTORIZER_PATH) and os.path.exists(PROBLEM_MATRIX_PATH):
                with open(PROBLEM_VECTORIZER_PATH, "rb") as f:
                    problem_vectorizer = pickle.load(f)
                with open(PROBLEM_MATRIX_PATH, "rb") as f:
                    problem_tfidf_matrix = pickle.load(f)
            else:
                problem_vectorizer = None
                problem_tfidf_matrix = None

            # Muat data pakar dari SQLite ke Pandas DataFrame global
            conn = sqlite3.connect(DB_PATH)
            df_experts = pd.read_sql_query(
                "SELECT id AS ID, name AS Name, department AS Department, email AS Email, description AS Description FROM experts",
                conn,
            )
            df_problems = pd.read_sql_query(
                "SELECT id AS ID, expert_id AS ExpertID, problem AS Problem, solution AS Solution FROM problem_solving",
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
                local_df_problems = df_problems.copy() if df_problems is not None else pd.DataFrame(columns=["ID", "ExpertID", "Problem", "Solution"])
                local_vectorizer = vectorizer
                local_tfidf_matrix = tfidf_matrix
                local_problem_vectorizer = problem_vectorizer
                local_problem_tfidf_matrix = problem_tfidf_matrix



            # 2. Cari berdasarkan Riwayat Masalah (Problem Solving)
            if not local_df_problems.empty and local_problem_vectorizer is not None and local_problem_tfidf_matrix is not None:
                try:
                    query_vector = local_problem_vectorizer.transform([query])
                    cosine_sim = cosine_similarity(query_vector, local_problem_tfidf_matrix)[0]
                    df_temp = local_df_problems.copy()
                    df_temp["Score"] = cosine_sim
                    
                    if "Score" in df_temp.columns:
                        df_result = df_temp[df_temp["Score"] > 0.05]
                        problems_match = df_result.to_dict(orient="records")
                        
                        for p_match in problems_match:
                            p_match["MatchType"] = "problem"
                            expert_id = p_match["ExpertID"]
                            
                            # Cari detail pakar terkait
                            exp_rows = local_df_experts[local_df_experts["ID"] == expert_id]
                            if not exp_rows.empty:
                                exp_data = exp_rows.iloc[0]
                                p_match["ExpertName"] = exp_data["Name"]
                                p_match["ExpertEmail"] = exp_data["Email"]
                                p_match["ExpertDepartment"] = exp_data["Department"]
                                p_match["ExpertDescription"] = exp_data["Description"]
                            else:
                                p_match["ExpertName"] = "Tidak Diketahui"
                                p_match["ExpertEmail"] = "-"
                                p_match["ExpertDepartment"] = "-"
                                p_match["ExpertDescription"] = "-"
                            
                            # Ambil riwayat lengkap masalah pakar ini
                            history_rows = local_df_problems[local_df_problems["ExpertID"] == expert_id]
                            p_match["ExpertHistory"] = history_rows.to_dict(orient="records")
                            recommendations.append(p_match)
                except Exception as e:
                    print(f"ERROR: Terjadi kegagalan saat pencarian TF-IDF masalah: {e}")

            # Urutkan berdasarkan skor kemiripan tertinggi secara global
            recommendations = sorted(recommendations, key=lambda x: x["Score"], reverse=True)

            # Simpan hasil kueri pencarian dan rekomendasi ke session (Penerapan Pola PRG)
            session["search_query"] = query
            session["search_results"] = recommendations
        else:
            session.pop("search_query", None)
            session.pop("search_results", None)

        return redirect(url_for("index"))

    # Untuk request GET:
    # Ambil data kueri & rekomendasi dari session jika diarahkan dari POST, lalu hapus instan dari session
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
    cursor.execute("DELETE FROM problem_solving WHERE expert_id=?", (expert_id,))
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
    cursor.execute("DELETE FROM problem_solving")
    cursor.execute("DELETE FROM experts")
    conn.commit()
    conn.close()

    # Jalankan retraining secara sinkron agar model langsung siap digunakan
    retrain_tfidf_model()
    load_resources()
    flash("Semua pakar berhasil dihapus! AI selesai me-retrain model.", "success")

    return redirect(url_for("index") + "?tab=manage")

@app.route("/get-problems/<expert_id>", methods=["GET"])
def get_problems(expert_id):
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, problem, solution FROM problem_solving WHERE expert_id=?", (expert_id,))
    rows = cursor.fetchall()
    conn.close()
    
    problems_list = []
    for row in rows:
        problems_list.append({
            "id": row[0],
            "problem": row[1],
            "solution": row[2]
        })
    return {"status": "success", "problems": problems_list}

@app.route("/add-problem/<expert_id>", methods=["POST"])
def add_problem(expert_id):
    problem = request.form.get("problem", "").strip()
    solution = request.form.get("solution", "").strip()
    
    if not problem or not solution:
        return {"status": "error", "message": "Masalah dan solusi tidak boleh kosong!"}, 400
        
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("SELECT id FROM problem_solving")
    existing_ids = {row[0] for row in cursor.fetchall()}
    while True:
        seed = f"{expert_id}{time.time_ns()}{random.randint(1000, 9999)}"
        git_hash_id = hashlib.md5(seed.encode()).hexdigest()[:7]
        if git_hash_id not in existing_ids:
            break
            
    cursor.execute(
        "INSERT INTO problem_solving (id, expert_id, problem, solution) VALUES (?, ?, ?, ?)",
        (git_hash_id, expert_id, problem, solution)
    )
    conn.commit()
    conn.close()
    
    retrain_tfidf_model()
    load_resources()
    
    return {"status": "success", "message": "Riwayat masalah berhasil ditambahkan!"}

@app.route("/edit-problem/<problem_id>", methods=["POST"])
def edit_problem(problem_id):
    problem = request.form.get("problem", "").strip()
    solution = request.form.get("solution", "").strip()
    
    if not problem or not solution:
        return {"status": "error", "message": "Masalah dan solusi tidak boleh kosong!"}, 400
        
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE problem_solving SET problem=?, solution=? WHERE id=?",
        (problem, solution, problem_id)
    )
    conn.commit()
    conn.close()
    
    retrain_tfidf_model()
    load_resources()
    
    return {"status": "success", "message": "Riwayat masalah berhasil diperbarui!"}

@app.route("/delete-problem/<problem_id>", methods=["POST"])
def delete_problem(problem_id):
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM problem_solving WHERE id=?", (problem_id,))
    conn.commit()
    conn.close()
    
    retrain_tfidf_model()
    load_resources()
    
    return {"status": "success", "message": "Riwayat masalah berhasil dihapus!"}

@app.route("/download-template", methods=["GET"])
def download_template():
    output = io.StringIO()
    output.write("Nama,Divisi,Email,Deskripsi Keahlian,Riwayat Masalah\n")
    output.write(
        'Budi Santoso,Teknologi,budi.santoso@nextgen.id,"Pakar infrastruktur server Linux, virtualisasi Docker, dan keamanan jaringan.","[{""problem"": ""Virtualisasi Docker container tiba-tiba crash karena disk space penuh"", ""solution"": ""Jalankan docker system prune -a --volumes untuk membersihkan cache.""}]"\n'
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

                # Jika terdapat kolom Riwayat Masalah dan tidak kosong
                if "Riwayat Masalah" in uploaded_df.columns and not pd.isna(row["Riwayat Masalah"]):
                    try:
                        import json
                        problems_data = json.loads(str(row["Riwayat Masalah"]))
                        if isinstance(problems_data, list):
                            cursor.execute("SELECT id FROM problem_solving")
                            p_ids = {r[0] for r in cursor.fetchall()}
                            for p_item in problems_data:
                                p_text = p_item.get("problem", "").strip()
                                s_text = p_item.get("solution", "").strip()
                                if p_text and s_text:
                                    while True:
                                        p_seed = f"{git_hash_id}{time.time_ns()}{random.random()}"
                                        p_git_hash = hashlib.md5(p_seed.encode()).hexdigest()[:7]
                                        if p_git_hash not in p_ids:
                                            p_ids.add(p_git_hash)
                                            break
                                    cursor.execute(
                                        "INSERT INTO problem_solving (id, expert_id, problem, solution) VALUES (?, ?, ?, ?)",
                                        (p_git_hash, git_hash_id, p_text, s_text)
                                    )
                    except Exception as json_err:
                        print(f"ERROR: Gagal memproses JSON Riwayat Masalah: {json_err}")

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