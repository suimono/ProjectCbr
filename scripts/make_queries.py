import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple # Menambahkan Tuple
import logging

# --- Konfigurasi ---
PROCESSED_FILE = Path("data/processed/cases.json")
QUERIES_FILE = Path("data/eval/queries.json")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def initialize_directories() -> bool:
    """
    Memastikan direktori keluaran untuk kueri ada.
    
    Returns:
        bool: True jika pembuatan direktori berhasil, False jika sebaliknya
    """
    try:
        QUERIES_FILE.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory '{QUERIES_FILE.parent}' ensured.")
        return True
    except OSError as e:
        logger.error(f"Error creating directory {QUERIES_FILE.parent}: {e}")
        return False

def load_cases_data() -> Optional[List[Dict[str, Any]]]:
    """
    Memuat dan memvalidasi data kasus dari file JSON.
    
    Returns:
        Optional[List[Dict[str, Any]]]: Daftar kasus jika berhasil, None jika sebaliknya
    """
    if not PROCESSED_FILE.exists():
        logger.error(f"Processed cases file '{PROCESSED_FILE}' not found. "
                     "Please ensure 'cases.json' exists in 'data/processed/'.")
        return None

    try:
        with open(PROCESSED_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        # Memvalidasi struktur data
        if not isinstance(data, list):
            logger.error(f"Invalid data format in '{PROCESSED_FILE}'. "
                         f"Expected a JSON array, got {type(data).__name__}.")
            return None
            
        if not data:
            logger.warning(f"No cases found in '{PROCESSED_FILE}'.")
            return []
            
        return data
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON from '{PROCESSED_FILE}'. "
                     f"Invalid JSON format: {e}")
        return None
    except UnicodeDecodeError as e:
        logger.error(f"Encoding error reading '{PROCESSED_FILE}': {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error reading '{PROCESSED_FILE}': {e}")
        return None

def create_query_text(case: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    """
    Membuat teks kueri yang bermakna dari bidang kasus yang tersedia dan mengembalikan teks
    bersama dengan string yang menunjukkan bidang mana yang digunakan.
    
    Args:
        case: Kamus kasus
        
    Returns:
        Tuple[Optional[str], Optional[str]]: (Teks kueri yang dihasilkan, string bidang yang digunakan)
                                              atau (None, None) jika tidak ada konten yang cocok ditemukan
    """
    # Urutan prioritas bidang yang akan digunakan untuk teks kueri
    field_combinations = [
        # Coba pertama: ringkasan_fakta jika valid
        (["ringkasan_fakta"], "ringkasan_fakta"),
        
        # Coba kedua: gabungkan informasi kunci kasus
        (["jenis_perkara", "pasal", "status_hukuman"], "jenis_perkara, pasal, status_hukuman"),
        
        # Coba ketiga: info kasus dasar
        (["jenis_perkara", "pasal"], "jenis_perkara, pasal"),
        
        # Coba keempat: hanya jenis kasus dan info dasar
        (["no_perkara", "jenis_perkara", "tanggal"], "no_perkara, jenis_perkara, tanggal"),
    ]
    
    for fields_to_try, combination_name in field_combinations:
        text_parts = []
        
        for field in fields_to_try:
            if field in case and isinstance(case[field], str):
                value = case[field].strip()
                
                # Lewati jika itu placeholder atau terlalu pendek
                if (value and 
                    value not in ["===", "---", "...", "N/A", "null", "undefined"] and
                    len(set(value)) > 1 and  # Bukan hanya karakter berulang
                    len(value) >= 10):   # Panjang minimum
                    
                    # Bersihkan teks
                    if field == "pasal" and len(value) > 200:
                        # Potong deskripsi pasal yang panjang
                        value = value[:200] + "..."
                    elif field == "status_hukuman" and len(value) > 300:
                        # Potong deskripsi status yang panjang
                        value = value[:300] + "..."
                    
                    text_parts.append(value)
        
        # Jika kami menemukan konten yang valid, gabungkan
        if text_parts:
            if len(text_parts) == 1:
                return text_parts[0], combination_name
            else:
                # Buat teks kueri terstruktur
                combined_text = ". ".join(text_parts)
                return combined_text, combination_name
    
    return None, None # Kembalikan None untuk keduanya jika tidak ada konten yang cocok ditemukan

def process_cases_to_queries(cases: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """
    Mengubah kasus yang valid menjadi format kueri menggunakan ekstraksi teks yang fleksibel.
    
    Args:
        cases: Daftar kamus kasus
        
    Returns:
        List[Dict[str, str]]: Daftar kueri yang diformat
    """
    queries = []
    valid_count = 0
    
    for i, case in enumerate(cases):
        if not isinstance(case, dict):
            logger.warning(f"Skipping case at index {i}: not a dictionary")
            continue
        
        # Coba buat teks kueri yang bermakna dan dapatkan bidang yang digunakan
        query_text, fields_used = create_query_text(case)
        
        if query_text:
            query_data = {
                "query_id": f"query_{valid_count:04d}",
                "text": query_text,
                # Tambahkan metadata untuk referensi
                "case_id": case.get("case_id", f"case_{i}"),
                "no_perkara": case.get("no_perkara", ""),
                "jenis_perkara": case.get("jenis_perkara", ""),
                "source": "generated_from_case_fields",
                "fields_used_for_query": fields_used # Tambahkan bidang baru ini
            }
            queries.append(query_data)
            valid_count += 1
            # Log informasi bidang yang digunakan untuk setiap kueri
            logger.info(f"Query for case {i} (ID: {case.get('case_id', f'case_{i}')}) generated using: {fields_used}")
        else:
            logger.warning(f"Skipping case at index {i}: no suitable text content found")
    
    logger.info(f"Processed {len(cases)} cases, {valid_count} valid queries created, "
               f"{len(cases) - valid_count} entries skipped")
    
    return queries

def save_queries(queries: List[Dict[str, str]]) -> bool:
    """
    Menyimpan kueri ke file JSON.
    
    Args:
        queries: Daftar kamus kueri
        
    Returns:
        bool: True jika berhasil, False jika sebaliknya
    """
    if not queries:
        logger.warning("No queries to save.")
        return False
        
    try:
        # Buat cadangan jika file sudah ada
        if QUERIES_FILE.exists():
            backup_file = QUERIES_FILE.with_suffix('.json.backup')
            QUERIES_FILE.rename(backup_file)
            logger.info(f"Existing file backed up to '{backup_file}'")
        
        with open(QUERIES_FILE, "w", encoding="utf-8") as f:
            json.dump(queries, f, indent=4, ensure_ascii=False)
            
        logger.info(f"Successfully saved {len(queries)} queries to '{QUERIES_FILE}'")
        return True
        
    except Exception as e:
        logger.error(f"Failed to write queries to '{QUERIES_FILE}': {e}")
        return False

def analyze_data_structure(cases: List[Dict[str, Any]]) -> None:
    """
    Menganalisis dan mencetak struktur data kasus untuk membantu debugging.
    
    Args:
        cases: Daftar kamus kasus
    """
    if not cases:
        logger.info("No cases to analyze.")
        return
    
    logger.info(f"=== DATA ANALYSIS ===")
    logger.info(f"Total cases: {len(cases)}")
    
    # Analisis struktur kasus pertama
    if cases:
        first_case = cases[0]
        logger.info(f"Keys in first case: {list(first_case.keys()) if isinstance(first_case, dict) else 'Not a dictionary'}")
        
        # Tampilkan contoh bidang yang tersedia
        if isinstance(first_case, dict):
            logger.info("Sample field values:")
            for key, value in first_case.items():
                if isinstance(value, str):
                    preview = value[:100] + "..." if len(value) > 100 else value
                    logger.info(f"  {key}: '{preview}'")
                else:
                    logger.info(f"  {key}: {type(value).__name__} - {value}")
    
    # Periksa bidang teks potensial
    potential_text_fields = []
    if isinstance(first_case, dict):
        for key, value in first_case.items():
            if isinstance(value, str) and len(value.strip()) > 10 and value.strip() != "===":
                potential_text_fields.append(key)
    
    if potential_text_fields:
        logger.info(f"Potential text fields for queries: {potential_text_fields}")
    else:
        logger.warning("No suitable text fields found for creating queries!")
    
    logger.info("=== END ANALYSIS ===")

def make_queries() -> bool:
    """
    Fungsi utama untuk membuat queries.json dari cases.json.
    
    Returns:
        bool: True jika berhasil, False jika sebaliknya
    """
    # Inisialisasi direktori
    if not initialize_directories():
        return False
    
    # Muat dan validasi data kasus
    cases = load_cases_data()
    if cases is None:
        return False
        
    if not cases:
        logger.warning("No cases to process.")
        return True  # Bukan kesalahan, hanya data kosong
    
    # Analisis struktur data untuk debugging
    analyze_data_structure(cases)
    
    # Proses kasus menjadi kueri
    queries = process_cases_to_queries(cases)
    
    # Simpan kueri (bahkan jika kosong, untuk tujuan debugging)
    if queries:
        return save_queries(queries)
    else:
        logger.error("No valid queries generated. Please check your data source.")
        logger.info("Suggestions:")
        logger.info("1. Verify that 'ringkasan_fakta' field contains actual case summaries")
        logger.info("2. Check if there are other text fields that should be used instead")
        logger.info("3. Ensure your data processing pipeline is working correctly")
        return False

# --- Titik Masuk ---
if __name__ == "__main__":
    success = make_queries()
    exit_code = 0 if success else 1
    exit(exit_code)
