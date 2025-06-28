import json
import csv
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
import pandas as pd
from collections import Counter # Digunakan lagi untuk extract_pasals

# --- Konfigurasi Awal ---
# Mendefinisikan jalur file input dan output.
QUERY_FILE = Path("data/eval/queries.json")
CASE_FILE = Path("data/processed/cases.json") # Diperlukan untuk ground truth prediction
RETRIEVED_CASES_FILE = Path("data/results/retrieved_cases.json") # Hasil retrieval
PREDICTIONS_FILE = Path("data/results/predictions.csv") # Hasil prediksi

RETRIEVAL_METRICS_FILE = Path("data/eval/retrieval_metrics.csv")
PREDICTION_METRICS_FILE = Path("data/eval/prediction_metrics.csv")

# Mengatur logging untuk memberikan informasi, peringatan, dan kesalahan selama eksekusi.
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# --- Fungsi Utilitas ---

def initialize_directories(file_path: Path) -> bool:
    """
    Memastikan direktori (folder) tempat file output akan disimpan sudah ada.
    Jika belum ada, fungsi ini akan membuatnya.
    
    Args:
        file_path (Path): Objek Path dari file yang akan disimpan.
                          Digunakan untuk mendapatkan direktori induknya.
        
    Returns:
        bool: True jika direktori berhasil dipastikan/dibuat, False jika terjadi error.
    """
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory '{file_path.parent}' ensured.")
        return True
    except OSError as e:
        logger.error(f"Error creating directory {file_path.parent}: {e}")
        return False

def load_json_data(file_path: Path) -> Optional[List[Dict[str, Any]]]:
    """
    Memuat data dari file JSON yang diberikan.
    Fungsi ini juga melakukan validasi dasar untuk memastikan file ada, 
    formatnya adalah JSON, dan isinya berupa daftar.
    
    Args:
        file_path (Path): Path ke file JSON yang akan dimuat.
        
    Returns:
        Optional[List[Dict[str, Any]]]: Daftar kamus (data JSON) jika berhasil dimuat.
                                         Mengembalikan None jika ada kesalahan (file tidak ditemukan,
                                         format JSON salah, dll.).
                                         Mengembalikan daftar kosong jika file ada tapi isinya kosong.
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

def extract_pasals(text: Optional[str]) -> List[str]:
    """
    Mengekstrak semua referensi pasal dari sebuah string teks menggunakan ekspresi reguler.
    Fungsi ini juga akan membersihkan dan menghilangkan duplikasi pasal yang ditemukan.
    Contoh: "Pasal 10, Pasal 10 Ayat (1) huruf a" akan menjadi ["Pasal 10", "Pasal 10 Ayat (1) Huruf A"].
    
    Args:
        text (Optional[str]): String teks yang mungkin berisi pasal-pasal.
                              Bisa berupa None jika tidak ada teks.
        
    Returns:
        List[str]: Daftar string pasal yang sudah diekstrak, dibersihkan, dan unik.
    """
    pasal_pattern = re.compile(
        r"Pasal\s+\d+(?:\s+Ayat\s+\(\d+\))?(?:\s+huruf\s+[a-zA-Z])?",
        re.IGNORECASE
    )
    pasals = pasal_pattern.findall(text or "")
    pasals = [p.title().strip() for p in pasals if p.strip()] 
    return list(dict.fromkeys(pasals))  

# --- Fungsi Evaluasi ---

def eval_retrieval():
    """
    Mengevaluasi kinerja retrieval menggunakan Mean Reciprocal Rank (MRR).
    Asumsi:
    - queries.json berisi 'query_id' dan 'case_id' (case_id adalah ground truth relevant case).
    - retrieved_cases.json berisi 'query_id' dan 'top_k_case_ids' (hasil retrieval yang diurutkan).
    """
    logger.info("Starting retrieval evaluation...")

    # Pastikan direktori output ada
    if not initialize_directories(RETRIEVAL_METRICS_FILE):
        return

    queries = load_json_data(QUERY_FILE)
    retrieved_data = load_json_data(RETRIEVED_CASES_FILE)

    if queries is None or retrieved_data is None:
        logger.error("Failed to load necessary data for retrieval evaluation. Exiting.")
        return
    if not queries or not retrieved_data:
        logger.warning("No data to evaluate for retrieval. Exiting.")
        return

    # Buat kamus untuk akses cepat data retrieved berdasarkan query_id
    retrieved_dict = {item.get("query_id"): item.get("top_k_case_ids", []) 
                      for item in retrieved_data if item.get("query_id")}

    reciprocal_ranks = []

    for query_entry in queries:
        query_id = query_entry.get("query_id")
        ground_truth_case_id = query_entry.get("case_id") # Asumsi: case_id adalah ground truth

        if not query_id or not ground_truth_case_id:
            logger.warning(f"Skipping query entry due to missing ID or ground truth: {query_entry}")
            continue

        predicted_cases = retrieved_dict.get(query_id, [])
        
        # Hitung Reciprocal Rank (RR)
        # MRR adalah rata-rata RR untuk setiap kueri. RR adalah 1/rank dari dokumen relevan pertama.
        # Jika tidak ada dokumen relevan yang ditemukan, RR adalah 0.
        rank = 0
        for i, case_id in enumerate(predicted_cases):
            if str(case_id) == str(ground_truth_case_id): # Konversi ke str untuk perbandingan yang konsisten
                rank = i + 1
                break
        
        if rank > 0:
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0) # Tidak ada dokumen relevan yang ditemukan

    # Hitung Mean Reciprocal Rank (MRR)
    if reciprocal_ranks:
        mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)
    else:
        mrr = 0.0
        logger.warning("No queries processed for MRR calculation.")

    # Simpan metrik retrieval
    try:
        df_metrics = pd.DataFrame([{"MRR": mrr}])
        df_metrics.to_csv(RETRIEVAL_METRICS_FILE, index=False)
        logger.info(f"✅ Retrieval metrics (MRR: {mrr:.4f}) saved to '{RETRIEVAL_METRICS_FILE}'")
    except Exception as e:
        logger.error(f"❌ Failed to save retrieval metrics: {e}")

def eval_prediction():
    """
    Mengevaluasi kinerja prediksi solusi menggunakan metrik Precision, Recall, dan F1-score
    (Micro-average) untuk masalah multi-label classification.
    Asumsi:
    - predictions.csv berisi 'query_id' dan 'predicted_solution'.
    - queries.json berisi 'query_id' dan 'case_id'.
    - cases.json berisi 'case_id' dan 'pasal' (ground truth solution).
    """
    logger.info("Starting prediction evaluation...")

    # Pastikan direktori output ada
    if not initialize_directories(PREDICTION_METRICS_FILE):
        return

    # Muat data
    try:
        predictions_df = pd.read_csv(PREDICTIONS_FILE)
    except FileNotFoundError:
        logger.error(f"Prediction file '{PREDICTIONS_FILE}' not found.")
        return
    except Exception as e:
        logger.error(f"Error loading prediction file '{PREDICTIONS_FILE}': {e}")
        return

    queries = load_json_data(QUERY_FILE)
    cases = load_json_data(CASE_FILE)

    if queries is None or cases is None:
        logger.error("Failed to load necessary data for prediction evaluation. Exiting.")
        return
    if not queries or not cases:
        logger.warning("No data to evaluate for prediction. Exiting.")
        return

    # Buat kamus untuk akses cepat data kasus dan kueri
    case_dict = {c.get("case_id"): c.get("pasal", "") 
                 for c in cases if c.get("case_id")}
    query_case_map = {q.get("query_id"): q.get("case_id") 
                      for q in queries if q.get("query_id") and q.get("case_id")}

    total_tp = 0
    total_fp = 0
    total_fn = 0

    for index, row in predictions_df.iterrows():
        query_id = str(row.get("query_id"))
        predicted_solution_str = str(row.get("predicted_solution", ""))

        original_case_id = query_case_map.get(query_id)
        if not original_case_id:
            logger.warning(f"Skipping prediction for query_id '{query_id}': Original case_id not found in queries data.")
            continue

        ground_truth_pasal_str = case_dict.get(original_case_id)
        if ground_truth_pasal_str is None:
            logger.warning(f"Skipping prediction for query_id '{query_id}': Ground truth pasal not found for case_id '{original_case_id}'.")
            continue

        # Ekstrak pasal dari string ground truth dan prediksi
        true_pasals = set(extract_pasals(ground_truth_pasal_str))
        pred_pasals = set(extract_pasals(predicted_solution_str))

        # Hitung True Positives (TP), False Positives (FP), False Negatives (FN) per kueri
        tp = len(true_pasals.intersection(pred_pasals))
        fp = len(pred_pasals - true_pasals)
        fn = len(true_pasals - pred_pasals)

        total_tp += tp
        total_fp += fp
        total_fn += fn
    
    # Hitung Micro-averaged Precision, Recall, F1-score
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    # Simpan metrik prediksi
    try:
        df_metrics = pd.DataFrame([{"precision": precision, "recall": recall, "f1": f1}])
        df_metrics.to_csv(PREDICTION_METRICS_FILE, index=False)
        logger.info(f"✅ Prediction metrics (Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}) saved to '{PREDICTION_METRICS_FILE}'")
    except Exception as e:
        logger.error(f"❌ Failed to save prediction metrics: {e}")

# --- Titik Masuk Skrip ---
if __name__ == "__main__":
    eval_retrieval()
    eval_prediction()
