import os
import json
import csv
import re
from pathlib import Path
from collections import Counter
from typing import List, Dict, Any, Optional, Tuple 
import logging

# --- Konfigurasi Awal ---
# Mendefinisikan jalur file input dan output untuk menjaga semua path di satu tempat.
RETRIEVAL_FILE = Path("data/results/retrieved_cases.json")
CASE_FILE = Path("data/processed/cases.json")
OUTPUT_FILE = Path("data/results/predictions.csv")

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
        # parent.mkdir(parents=True, exist_ok=True) akan membuat semua direktori induk yang diperlukan
        # dan tidak akan menimbulkan error jika direktori sudah ada.
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
            
        # Memastikan data yang dimuat adalah sebuah list (daftar)
        if not isinstance(data, list):
            logger.error(f"Invalid data format in '{file_path}'. Expected a JSON array, got {type(data).__name__}.")
            return None
            
        # Memperingatkan jika file JSON kosong
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
    # Ekspresi reguler untuk mencocokkan pola "Pasal X Ayat (Y) huruf Z"
    # re.IGNORECASE memastikan pencocokan tidak sensitif huruf besar/kecil.
    pasal_pattern = re.compile(
        r"Pasal\s+\d+(?:\s+Ayat\s+\(\d+\))?(?:\s+huruf\s+[a-zA-Z])?",
        re.IGNORECASE
    )
    
    # Mencari semua pola pasal dalam teks (jika teks bukan None)
    pasals = pasal_pattern.findall(text or "")
    
    # Membersihkan setiap pasal:
    # 1. Mengubah ke format Judul (huruf pertama setiap kata kapital).
    # 2. Menghilangkan spasi ekstra di awal/akhir.
    # 3. Memastikan pasal tidak kosong setelah dibersihkan.
    pasals = [p.title().strip() for p in pasals if p.strip()] 
    
    # Menghilangkan duplikasi sambil menjaga urutan asli yang ditemukan.
    # list(dict.fromkeys(pasals)) adalah cara yang Pythonic untuk melakukan ini.
    return list(dict.fromkeys(pasals))  

def majority_vote(cases: List[Dict[str, Any]]) -> str:
    """
    Melakukan 'majority vote' untuk menentukan solusi yang diprediksi 
    berdasarkan pasal-pasal yang paling sering muncul di antara kasus-kasus yang relevan.
    
    Args:
        cases (List[Dict[str, Any]]): Daftar kamus kasus yang relevan (misalnya, top-k kasus yang diambil).
        
    Returns:
        str: String yang berisi 5 pasal paling sering muncul (atau kurang jika tidak ada 5),
             dipisahkan dengan "; ". Mengembalikan "N/A" jika tidak ada pasal yang dapat ditemukan.
    """
    pasal_counter = Counter() # Counter adalah alat yang efisien untuk menghitung frekuensi item.
    
    for case in cases:
        # Memastikan setiap item yang diproses adalah kamus
        if not isinstance(case, dict):
            logger.warning(f"Skipping non-dictionary case in majority vote: {case}")
            continue
        
        # Mengekstrak pasal dari bidang 'pasal' setiap kasus
        pasals = extract_pasals(case.get("pasal", ""))
        
        # Menambahkan pasal yang ditemukan ke counter
        pasal_counter.update(pasals)
    
    # Jika tidak ada pasal yang ditemukan sama sekali dari semua kasus
    if not pasal_counter:
        return "N/A" 
        
    # Mengambil 5 pasal yang paling sering muncul (top common)
    # Jika kurang dari 5, itu akan mengembalikan semua yang ditemukan.
    top_pasals = [pasal for pasal, _ in pasal_counter.most_common(5)]
    
    # Menggabungkan pasal-pasal teratas menjadi satu string
    return "; ".join(top_pasals)

# --- Fungsi Utama ---

def main():
    """
    Fungsi utama yang mengatur alur kerja prediksi.
    Ini akan:
    1. Memastikan direktori output ada.
    2. Memuat data hasil retrieval dan data kasus dasar.
    3. Memproses setiap kueri, menemukan kasus-kasus yang relevan.
    4. Melakukan majority vote pada pasal-pasal kasus relevan untuk memprediksi solusi.
    5. Menyimpan hasil prediksi ke file CSV.
    """
    # Langkah 1: Pastikan direktori output untuk file CSV ada.
    if not initialize_directories(OUTPUT_FILE):
        return # Keluar jika direktori tidak dapat dibuat

    logger.info("Starting prediction process...")

    # Langkah 2: Muat data hasil retrieval (kasus-kasus yang paling mirip dengan kueri).
    retrieved_data = load_json_data(RETRIEVAL_FILE)
    if retrieved_data is None:
        logger.error("Failed to load retrieval data. Exiting.")
        return
    if not retrieved_data:
        logger.warning("No retrieval data found. Nothing to predict.")
        return

    # Langkah 3: Muat data kasus dasar (semua kasus yang diketahui).
    case_data = load_json_data(CASE_FILE)
    if case_data is None:
        logger.error("Failed to load case base data. Exiting.")
        return
    if not case_data:
        logger.warning("No case base data found. Cannot perform majority voting.")
        return

    # Langkah 4: Buat kamus untuk pencarian kasus yang efisien berdasarkan case_id.
    # Ini memungkinkan kita dengan cepat mengambil detail kasus hanya dengan ID-nya.
    case_dict = {c.get("case_id"): c for c in case_data if c.get("case_id")}
    if not case_dict:
        logger.error("No valid case_ids found in case base. Cannot map retrieved cases.")
        return

    results = [] # Daftar untuk menyimpan hasil prediksi akhir
    processed_queries = 0 # Penghitung kueri yang berhasil diproses

    # Langkah 5: Iterasi (ulang) melalui setiap entri kueri dari data retrieval.
    for query_entry in retrieved_data:
        # Validasi bahwa entri kueri adalah kamus yang valid.
        if not isinstance(query_entry, dict):
            logger.warning(f"Skipping non-dictionary entry in retrieval data: {query_entry}")
            continue

        query_id = query_entry.get("query_id", "UNKNOWN_QUERY")
        # Mengambil daftar ID kasus teratas yang diambil untuk kueri ini.
        # Penting: Menggunakan kunci "top_k_case_ids" sesuai output skrip retrieval.
        top_case_ids = query_entry.get("top_k_case_ids", []) 

        # Validasi bahwa top_case_ids adalah daftar.
        if not isinstance(top_case_ids, list):
            logger.warning(f"Skipping query {query_id}: 'top_k_case_ids' is not a list. Type: {type(top_case_ids).__name__}")
            continue
            
        top_cases = [] # Daftar untuk menyimpan objek kasus lengkap dari top_case_ids
        # Mengambil objek kasus lengkap dari case_dict menggunakan ID yang diambil
        for cid in top_case_ids:
            if cid in case_dict:
                top_cases.append(case_dict[cid])
            else:
                logger.warning(f"Case ID '{cid}' for query '{query_id}' not found in case base. Skipping this case for voting.")

        # Melakukan majority vote untuk memprediksi solusi
        if not top_cases:
            logger.warning(f"No valid top-k cases found for query {query_id}. Skipping majority vote, setting predicted solution to 'N/A'.")
            predicted = "N/A"
        else:
            predicted = majority_vote(top_cases)
        
        # Menambahkan hasil prediksi untuk kueri ini ke daftar 'results'
        results.append({
            "query_id": query_id,
            "predicted_solution": predicted,
            # Pastikan hanya mengambil 5 ID teratas jika ada lebih, sesuai format output.
            "top_5_case_ids": top_case_ids[:5]  
        })
        processed_queries += 1
    
    logger.info(f"Processed {processed_queries} queries for prediction.")

    # Langkah 6: Simpan hasil prediksi ke file CSV.
    try:
        with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
            # Membuat objek writer CSV. quoting=csv.QUOTE_ALL memastikan semua bidang di-quote.
            writer = csv.writer(f, quoting=csv.QUOTE_ALL)
            
            # Menulis baris header untuk CSV
            writer.writerow(["query_id", "predicted_solution", "top_5_case_ids"])
            
            # Menulis setiap baris hasil ke CSV
            for r in results:
                # Menggabungkan daftar ID kasus teratas menjadi satu string yang dipisahkan koma
                top_ids_str = ", ".join(map(str, r.get("top_5_case_ids", [])))
                writer.writerow([
                    r.get("query_id", ""), # Menggunakan .get() untuk menghindari KeyError jika bidang hilang
                    r.get("predicted_solution", ""),
                    top_ids_str
                ])

        logger.info(f"✅ Prediksi berhasil disimpan ke: '{OUTPUT_FILE}'")
    except Exception as e:
        logger.error(f"❌ Gagal menulis prediksi ke '{OUTPUT_FILE}': {e}")

# --- Titik Masuk Skrip ---
# Bagian ini memastikan bahwa fungsi main() dipanggil hanya ketika skrip dijalankan langsung.
if __name__ == "__main__":
    main()
