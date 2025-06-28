import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- Konfigurasi ---
CASE_BASE_PATH = Path("data/processed/cases.json")
QUERY_PATH = Path("data/eval/queries.json")
OUTPUT_PATH = Path("data/results/retrieved_cases.json")
TOP_K_SIMILAR_CASES = 5 # Jumlah kasus serupa teratas yang akan diambil

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def initialize_directories() -> bool:
    """
    Memastikan direktori keluaran untuk hasil retrieval ada.
    
    Returns:
        bool: True jika pembuatan direktori berhasil, False jika sebaliknya
    """
    try:
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory '{OUTPUT_PATH.parent}' ensured.")
        return True
    except OSError as e:
        logger.error(f"Error creating directory {OUTPUT_PATH.parent}: {e}")
        return False

def load_json_data(file_path: Path) -> Optional[List[Dict[str, Any]]]:
    """
    Memuat dan memvalidasi data JSON dari file.
    
    Args:
        file_path (Path): Path ke file JSON.
        
    Returns:
        Optional[List[Dict[str, Any]]]: Daftar data jika berhasil, None jika sebaliknya.
    """
    if not file_path.exists():
        logger.error(f"File '{file_path}' not found.")
        return None

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        if not isinstance(data, list):
            logger.error(f"Invalid data format in '{file_path}'. Expected a JSON array, got {type(data).__name__}.")
            return None
            
        if not data:
            logger.warning(f"No data found in '{file_path}'.")
            return []
            
        logger.info(f"Successfully loaded {len(data)} entries from '{file_path}'.")
        return data
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON from '{file_path}'. Invalid JSON format: {e}")
        return None
    except UnicodeDecodeError as e:
        logger.error(f"Encoding error reading '{file_path}': {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error reading '{file_path}': {e}")
        return None

def extract_case_text_for_retrieval(case: Dict[str, Any]) -> Optional[str]:
    """
    Mengekstrak teks yang paling relevan dari sebuah kasus untuk tujuan retrieval.
    Prioritas diberikan pada 'ringkasan_fakta', diikuti oleh kombinasi bidang lainnya.
    Ini mirip dengan logika create_query_text tetapi disesuaikan untuk representasi kasus.
    
    Args:
        case (Dict[str, Any]): Kamus yang merepresentasikan satu kasus.
        
    Returns:
        Optional[str]: Teks yang diekstrak dari kasus, atau None jika tidak ada teks yang cocok ditemukan.
    """
    # Urutan prioritas bidang untuk mengekstrak teks kasus
    field_combinations = [
        ["ringkasan_fakta"],
        ["jenis_perkara", "pasal", "status_hukuman"],
        ["jenis_perkara", "pasal"],
        ["no_perkara", "jenis_perkara", "tanggal"],
    ]
    
    for fields_to_try in field_combinations:
        text_parts = []
        for field in fields_to_try:
            if field in case and isinstance(case[field], str):
                value = case[field].strip()
                # Lewati jika itu placeholder, terlalu pendek, atau hanya karakter berulang
                if (value and 
                    value not in ["===", "---", "...", "N/A", "null", "undefined"] and
                    len(set(value)) > 1 and 
                    len(value) >= 10):
                    
                    # Pemotongan teks untuk menjaga relevansi
                    if field == "pasal" and len(value) > 200:
                        value = value[:200] + "..."
                    elif field == "status_hukuman" and len(value) > 300:
                        value = value[:300] + "..."
                    
                    text_parts.append(value)
        
        if text_parts:
            # Gabungkan bagian teks. Gunakan titik sebagai pemisah.
            return ". ".join(text_parts)
            
    return None # Tidak ada teks yang cocok ditemukan

def main():
    """
    Fungsi utama untuk melakukan proses retrieval kasus.
    """
    if not initialize_directories():
        return

    # Muat data kasus
    cases = load_json_data(CASE_BASE_PATH)
    if cases is None:
        logger.error("Failed to load case base data. Exiting.")
        return

    # Muat data kueri
    queries = load_json_data(QUERY_PATH)
    if queries is None:
        logger.error("Failed to load queries data. Exiting.")
        return

    if not cases:
        logger.warning("No cases found in the case base. Retrieval cannot proceed.")
        return
    if not queries:
        logger.warning("No queries found. Retrieval cannot proceed.")
        return

    # Ekstrak teks yang relevan dari kasus dan kueri
    case_ids = []
    case_texts = []
    for i, case in enumerate(cases):
        if not isinstance(case, dict):
            logger.warning(f"Skipping case at index {i}: not a dictionary.")
            continue
        extracted_text = extract_case_text_for_retrieval(case)
        if extracted_text:
            case_ids.append(case.get("case_id", f"case_{i}"))
            case_texts.append(extracted_text)
        else:
            logger.warning(f"Skipping case {case.get('case_id', f'case_{i}')} due to no suitable text content.")

    query_ids = []
    query_texts = []
    for i, query in enumerate(queries):
        if not isinstance(query, dict):
            logger.warning(f"Skipping query at index {i}: not a dictionary.")
            continue
        # Gunakan bidang 'text' yang sudah diproses dari file queries.json
        query_text = query.get("text")
        if query_text and isinstance(query_text, str) and query_text.strip():
            query_ids.append(query.get("query_id", f"query_{i}"))
            query_texts.append(query_text.strip())
        else:
            logger.warning(f"Skipping query {query.get('query_id', f'query_{i}')} due to missing or empty 'text' field.")

    if not case_texts:
        logger.error("No valid case texts extracted for retrieval. Cannot proceed.")
        return
    if not query_texts:
        logger.error("No valid query texts extracted for retrieval. Cannot proceed.")
        return
        
    logger.info(f"Extracted {len(case_texts)} valid case texts and {len(query_texts)} valid query texts.")

    # Gabungkan semua teks untuk TF-IDF fit
    corpus = case_texts + query_texts

    # TF-IDF vectorization
    logger.info("Performing TF-IDF vectorization...")
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000) # Batasi fitur untuk performa
    tfidf_matrix = vectorizer.fit_transform(corpus)
    logger.info(f"TF-IDF matrix created with {tfidf_matrix.shape[1]} features.")

    # Pisahkan matriks kembali menjadi bagian kasus dan kueri
    case_matrix = tfidf_matrix[:len(case_texts)]
    query_matrix = tfidf_matrix[len(case_texts):]

    # Hitung cosine similarity
    logger.info("Calculating cosine similarities...")
    similarities = cosine_similarity(query_matrix, case_matrix)
    logger.info("Cosine similarity calculation complete.")

    # Ambil top-K similar case untuk setiap kueri
    results = []
    for i, sim_scores in enumerate(similarities):
        # Menggunakan argpartition untuk kinerja yang lebih baik pada array besar
        # Mengambil TOP_K_SIMILAR_CASES indeks terbesar
        top_indices = sim_scores.argpartition(-TOP_K_SIMILAR_CASES)[-TOP_K_SIMILAR_CASES:]
        
        # Urutkan indeks-indeks ini berdasarkan skor similaritasnya secara menurun
        sorted_top_indices = top_indices[sim_scores[top_indices].argsort()[::-1]]

        top_cases_ids = [case_ids[j] for j in sorted_top_indices]
        similarity_scores = [float(sim_scores[j]) for j in sorted_top_indices] # Konversi ke float untuk JSON serialisasi

        results.append({
            "query_id": query_ids[i],
            "top_k_case_ids": top_cases_ids,
            "similarity_scores": similarity_scores
        })
    logger.info(f"Top {TOP_K_SIMILAR_CASES} similar cases retrieved for {len(results)} queries.")

    # Simpan hasil
    try:
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        logger.info(f"✅ Retrieval completed. Results saved to '{OUTPUT_PATH}'")
    except Exception as e:
        logger.error(f"Failed to write retrieval results to '{OUTPUT_PATH}': {e}")

# --- Titik Masuk ---
if __name__ == "__main__":
    main()
