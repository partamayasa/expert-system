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
        query = "SELECT description FROM experts"
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        if df.empty:
            print("ERROR: Database kosong. Tidak ada data untuk dilatih.")
            return

        print(f"SUCCESS: Menemukan {len(df)} data pakar untuk proses training.")
        
    except Exception as e:
        print(f"ERROR: Error saat membaca database: {e}")
        return
    
    # 3. Proses Training TF-IDF
    vectorizer = TfidfVectorizer(lowercase=True, stop_words=indonesian_stopwords)
    tfidf_matrix = vectorizer.fit_transform(df['description'])
    
    # 4. Simpan Berkas Model Bisnis Baru (.pkl)
    vectorizer_path = os.path.join(model_dir, "tfidf_vectorizer.pkl")
    matrix_path = os.path.join(model_dir, "tfidf_matrix.pkl")
    
    with open(vectorizer_path, "wb") as f:
        pickle.dump(vectorizer, f)
        
    with open(matrix_path, "wb") as f:
        pickle.dump(tfidf_matrix, f)
        
    print("SUCCESS: Model AI sukses diperbarui dan disinkronisasikan ke database.")

if __name__ == "__main__":
    train_model()