import hashlib
import io
import json
import os
import pickle
import random
import sqlite3
import time
# Tambahkan flash ke dalam komponen import Werkzeug/Flask
from flask import Flask, Response, redirect, render_template, request, url_for, flash
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- Konfigurasi Jalur File Dinamis Berbasis Struktur Baru ---
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


# --- 1. Inisialisasi database ---
def init_db():
    """
    Membuat tabel database SQLite jika belum tersedia di dalam sistem.
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


# --- 2. Pipeline pelatihan ulang model ai ---
def retrain_tfidf_model():
    """
    Melatih ulang model TF-IDF secara terpusat berdasarkan data kompetensi terbaru dari database.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT description FROM experts")
    all_descriptions = [row[0] for row in cursor.fetchall()]
    conn.close()

    if not all_descriptions:
        all_descriptions = ["pakar kompetensi umum internal perusahaan"]

    new_vectorizer = TfidfVectorizer(lowercase=True, stop_words=INDONESIAN_STOPWORDS)
    new_tfidf_matrix = new_vectorizer.fit_transform(all_descriptions)

    os.makedirs(os.path.dirname(VECTORIZER_PATH), exist_ok=True)
    with open(VECTORIZER_PATH, "wb") as f:
        pickle.dump(new_vectorizer, f)
    with open(MATRIX_PATH, "wb") as f:
        pickle.dump(new_tfidf_matrix, f)


# --- 3. Muat sumber daya ke memori ---
def load_resources():
    """
    Memuat kembali model TF-IDF, matriks bobot teks, dan seluruh data pakar ke dalam memori aplikasi.
    """
    global vectorizer, tfidf_matrix, df_experts
    
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

    if not os.path.exists(VECTORIZER_PATH) or not os.path.exists(MATRIX_PATH):
        retrain_tfidf_model()

    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)
    with open(MATRIX_PATH, "rb") as f:
        tfidf_matrix = pickle.load(f)

    conn = sqlite3.connect(DB_PATH)
    df_experts = pd.read_sql_query(
        "SELECT id AS ID, name AS Name, department AS Department, email AS Email, description AS Description FROM experts",
        conn,
    )
    conn.close()


try:
    load_resources()
except Exception:
    retrain_tfidf_model()
    load_resources()


# --- Engine AI Generator Deskripsi (Dinamis Berbasis TF-IDF & JSON) ---
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
        conn = sqlite3.connect(DB_PATH)
        df_db = pd.read_sql_query("SELECT department, description FROM experts", conn)
        conn.close()
    except Exception:
        df_db = pd.DataFrame(columns=["department", "description"])

    df_filtered = df_db[df_db["department"] == department].copy()

    if not df_filtered.empty and vectorizer is not None and tfidf_matrix is not None and keywords.strip():
        try:
            keywords_vec = vectorizer.transform([keywords])
            sim_scores = cosine_similarity(keywords_vec, tfidf_matrix)[0]
            
            df_db["score"] = sim_scores
            df_match = df_db[(df_db["department"] == department) & (df_db["score"] >= threshold)]
            
            if not df_match.empty:
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
        
        if not query:
            return render_template(
                "index.html",
                query="",
                recommendations=[],
                active_tab="search",
                experts=df_current.to_dict(orient="records") if not df_current.empty else [],
            )

        if query and not df_current.empty and vectorizer is not None and tfidf_matrix is not None:
            query_vector = vectorizer.transform([query])
            cosine_sim = cosine_similarity(query_vector, tfidf_matrix)[0]

            df_temp = df_current.copy()
            df_temp["Score"] = cosine_sim
            
            if "Score" in df_temp.columns:
                df_result = df_temp[df_temp["Score"] > 0.05].sort_values(by="Score", ascending=False)
                recommendations = df_result.to_dict(orient="records")

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

        retrain_tfidf_model()
        load_resources()

    return redirect(url_for("index") + "?tab=manage")


@app.route("/edit-expert/<expert_id>", methods=["POST"])
def edit_expert(expert_id):
    name = request.form.get("name", "").strip()
    department = request.form.get("department", "").strip()
    email = request.form.get("email", "").strip()
    description = request.form.get("description", "").strip()

    if name and department and email and description:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE experts SET name=?, department=?, email=?, description=? WHERE id=?",
            (name, department, email, description, expert_id),
        )
        conn.commit()
        conn.close()

        retrain_tfidf_model()
        load_resources()
        
        # Ditambahkan: Kirim flash message sukses ke index.html
        flash("Data pakar berhasil diperbarui!", "success")

    return redirect(url_for("index") + "?tab=manage")


@app.route("/delete-expert/<expert_id>", methods=["GET", "POST"])
def delete_expert(expert_id):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM experts WHERE id=?", (expert_id,))
    conn.commit()
    conn.close()

    retrain_tfidf_model()
    load_resources()

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

            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM experts")
            existing_ids = {row[0] for row in cursor.fetchall()}

            for _, row in uploaded_df.iterrows():
                if pd.isna(row["Nama"]) or "contoh:" in str(row["Nama"]):
                    continue

                name_val = str(row["Nama"]).strip()
                dept_val = str(row["Divisi"]).strip()
                email_val = str(row["Email"]).strip()
                desc_val = str(row["Deskripsi Keahlian"]).strip()

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

            conn.commit()
            conn.close()

            retrain_tfidf_model()
            load_resources()

        except Exception as e:
            return f"Terjadi kesalahan saat memproses data CSV: {str(e)}", 500

    return redirect(url_for("index") + "?tab=manage")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)