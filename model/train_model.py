import sqlite3
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import json
import os

def train_model():
    print("--- Memulai Proses Pelatihan Model AI dari Database ---")
    
    # Konfigurasi path relatif berbasis file ini berada (di dalam folder 'model')
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(BASE_DIR, "data", "experts.db")
    model_dir = os.path.join(BASE_DIR, "saved_models")
    stopwords_path = os.path.abspath(os.path.join(BASE_DIR, "..", "config", "indonesian_stopwords.json"))
    
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        
    # 1. Muat Stopwords dari folder config yang baru
    indonesian_stopwords = []
    if os.path.exists(stopwords_path):
        with open(stopwords_path, "r", encoding="utf-8") as f:
            indonesian_stopwords = json.load(f)
            print("SUCCESS: Stopwords Indonesia berhasil dimuat.")
    
    # 2. Ambil data langsung dari SQLite
    try:
        conn = sqlite3.connect(db_path)
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
        
        query = "SELECT description FROM experts"
        df = pd.read_sql_query(query, conn)
        
        query_problems = "SELECT problem, solution FROM problem_solving"
        df_problems = pd.read_sql_query(query_problems, conn)
        conn.close()
        
        if df.empty:
            print("PERINGATAN: Database expert kosong. Tidak ada data untuk dilatih.")
        else:
            print(f"SUCCESS: Menemukan {len(df)} data pakar untuk proses training.")
            
        if df_problems.empty:
            print("PERINGATAN: Database problem_solving kosong. Menggunakan data fallback untuk model problem.")
        else:
            print(f"SUCCESS: Menemukan {len(df_problems)} data problem solving untuk proses training.")
        
    except Exception as e:
        print(f"ERROR: Error saat membaca database: {e}")
        return
    
    # 3. Proses Training TF-IDF untuk Kompetensi Pakar
    vectorizer = TfidfVectorizer(lowercase=True, stop_words=indonesian_stopwords)
    try:
        if not df.empty:
            tfidf_matrix = vectorizer.fit_transform(df['description'])
        else:
            raise ValueError("Dataframe expert kosong")
    except ValueError as e:
        print(f"PERINGATAN: TfidfVectorizer gagal melatih model expert ({e}). Menggunakan kata kunci fallback.")
        fallback_descriptions = ["pakar kompetensi umum internal perusahaan"]
        vectorizer = TfidfVectorizer(lowercase=True, stop_words=indonesian_stopwords)
        tfidf_matrix = vectorizer.fit_transform(fallback_descriptions)
    
    # 4. Proses Training TF-IDF untuk Riwayat Masalah (Problem Solving)
    problem_vectorizer = TfidfVectorizer(lowercase=True, stop_words=indonesian_stopwords)
    try:
        if not df_problems.empty:
            problem_texts = df_problems['problem'] + ' ' + df_problems['solution']
            problem_tfidf_matrix = problem_vectorizer.fit_transform(problem_texts)
        else:
            raise ValueError("Dataframe problem kosong")
    except ValueError as e:
        print(f"PERINGATAN: TfidfVectorizer gagal melatih model problem ({e}). Menggunakan kata kunci fallback.")
        fallback_problems = ["kendala masalah sistem jaringan komputer internet koneksi"]
        problem_vectorizer = TfidfVectorizer(lowercase=True, stop_words=indonesian_stopwords)
        problem_tfidf_matrix = problem_vectorizer.fit_transform(fallback_problems)
    
    # 5. Simpan Berkas Model Bisnis Baru (.pkl)
    vectorizer_path = os.path.join(model_dir, "tfidf_vectorizer.pkl")
    matrix_path = os.path.join(model_dir, "tfidf_matrix.pkl")
    problem_vectorizer_path = os.path.join(model_dir, "problem_tfidf_vectorizer.pkl")
    problem_matrix_path = os.path.join(model_dir, "problem_tfidf_matrix.pkl")
    
    tmp_vectorizer_path = vectorizer_path + ".tmp"
    tmp_matrix_path = matrix_path + ".tmp"
    tmp_problem_vectorizer_path = problem_vectorizer_path + ".tmp"
    tmp_problem_matrix_path = problem_matrix_path + ".tmp"
    
    with open(tmp_vectorizer_path, "wb") as f:
        pickle.dump(vectorizer, f)
    with open(tmp_matrix_path, "wb") as f:
        pickle.dump(tfidf_matrix, f)
        
    with open(tmp_problem_vectorizer_path, "wb") as f:
        pickle.dump(problem_vectorizer, f)
    with open(tmp_problem_matrix_path, "wb") as f:
        pickle.dump(problem_tfidf_matrix, f)
        
    os.replace(tmp_vectorizer_path, vectorizer_path)
    os.replace(tmp_matrix_path, matrix_path)
    os.replace(tmp_problem_vectorizer_path, problem_vectorizer_path)
    os.replace(tmp_problem_matrix_path, problem_matrix_path)
        
    print("SUCCESS: Seluruh Model AI sukses diperbarui dan disinkronisasikan ke database.")

if __name__ == "__main__":
    train_model()