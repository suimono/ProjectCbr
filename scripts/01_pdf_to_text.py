import os
import json
import fitz  # PyMuPDF
from tqdm import tqdm
from datetime import datetime
import logging
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

# --- Setup Logging ---
# Mengatur sistem logging untuk mencatat informasi, peringatan, dan kesalahan.
# Log akan ditampilkan di konsol dan juga disimpan ke file log.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Konfigurasi Path ---
# Mendefinisikan semua direktori input dan output. Menggunakan pathlib.Path 
# untuk penanganan path yang lebih modern dan platform-agnostik.
INPUT_DIR = Path("data/pdf")
RAW_TXT_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
OUTPUT_CASES_JSON = PROCESSED_DIR / "cases.json" # Menyimpan semua metadata kasus
OUTPUT_TEXT_MAP_JSON = PROCESSED_DIR / "case_raw_texts.json" # Menyimpan teks mentah (opsional, bisa dihilangkan jika tidak diperlukan)

# --- Fungsi Utilitas Direktori ---

def ensure_directories():
    """
    Memastikan semua direktori yang diperlukan untuk input dan output sudah ada.
    Jika sebuah direktori belum ada, fungsi ini akan membuatnya.
    """
    for directory in [INPUT_DIR, RAW_TXT_DIR, PROCESSED_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
        logger.info(f"Directory ensured: {directory}")

# --- Fungsi Generasi ID ---

def generate_case_id_from_filename(filename_stem: str) -> str:
    """
    Menghasilkan case_id yang bersih dari nama file dasar (tanpa ekstensi).
    Ini tidak lagi mencakup timestamp karena case_id yang sudah ada harus unik.
    Timestamp akan ditambahkan saat kasus diproses.
    
    Args:
        filename_stem (str): Nama file tanpa ekstensi (misal: "putusan_123_abc").
        
    Returns:
        str: Case ID yang dibersihkan.
    """
    # Bersihkan nama file: hapus karakter non-alfanumerik atau non-underscore/dash, 
    # ganti spasi dengan underscore, hindari underscore berturut-turut.
    clean_name = re.sub(r'[^\w\-.]', '_', filename_stem.lower())
    clean_name = re.sub(r'_+', '_', clean_name).strip('_')
    return clean_name

# --- Fungsi Ekstraksi Teks PDF ---

def extract_text_blocks_improved(page: fitz.Page) -> str:
    """
    Mengekstrak teks dari halaman PDF menggunakan metode 'blocks' PyMuPDF,
    dengan penyortiran yang lebih baik dan filter untuk block yang valid.
    """
    try:
        blocks = page.get_text("blocks")
        if not blocks:
            return ""
        
        # Urutkan blok berdasarkan posisi (dari atas ke bawah, dari kiri ke kanan)
        # Menggunakan pembulatan untuk menangani sedikit perbedaan koordinat.
        blocks.sort(key=lambda b: (round(b[1]), round(b[0])))
        
        text_parts = []
        for block in blocks:
            # Format block: [x0, y0, x1, y1, text, block_no, block_type]
            if len(block) >= 5 and block[4] and isinstance(block[4], str):
                text = block[4].strip()
                if text: # Pastikan teks tidak kosong setelah strip
                    text_parts.append(text)
        
        return "\n".join(text_parts)
    except Exception as e:
        logger.debug(f"Block extraction failed on page (improved): {e}")
        return ""

def extract_text_dict_improved(page: fitz.Page) -> str:
    """
    Mengekstrak teks dari halaman PDF menggunakan metode 'dict' PyMuPDF,
    dengan iterasi yang lebih hati-hati melalui struktur kamus.
    """
    try:
        text_dict = page.get_text("dict")
        if not text_dict or "blocks" not in text_dict:
            return ""
        
        page_text_parts = []
        for block in text_dict.get("blocks", []):
            if "lines" not in block:
                continue
            
            block_lines = []
            for line in block["lines"]:
                if "spans" not in line:
                    continue
                
                line_text = "".join([span.get("text", "") for span in line["spans"]])
                if line_text.strip():
                    block_lines.append(line_text.strip())
            
            if block_lines:
                page_text_parts.append(" ".join(block_lines)) # Gabungkan baris dalam blok dengan spasi
        
        return "\n".join(page_text_parts) # Gabungkan blok dengan newline
    except Exception as e:
        logger.debug(f"Dict extraction failed on page (improved): {e}")
        return ""

def extract_text_from_pdf(file_path: Path) -> Optional[str]:
    """
    Mengekstrak teks dari file PDF menggunakan berbagai metode PyMuPDF 
    dan memilih hasil terbaik.
    
    Args:
        file_path (Path): Path ke file PDF.
        
    Returns:
        Optional[str]: Seluruh teks yang diekstrak dari PDF, atau None jika gagal.
    """
    try:
        doc = fitz.open(file_path)
        full_text_parts = []
        
        for page_num in range(len(doc)):
            try:
                page = doc[page_num]
                page_text_candidates = {}
                
                # Coba berbagai metode ekstraksi dan simpan hasilnya
                methods = {
                    "standard_sorted": lambda p: p.get_text("text", sort=True),
                    "standard_unsorted": lambda p: p.get_text("text"),
                    "blocks_custom": extract_text_blocks_improved,
                    "dict_custom": extract_text_dict_improved
                }
                
                for method_name, method_func in methods.items():
                    try:
                        result = method_func(page)
                        if result and result.strip():
                            page_text_candidates[method_name] = result
                    except Exception as e:
                        logger.debug(f"Method '{method_name}' failed on page {page_num + 1} of {file_path.name}: {e}")
                
                best_page_text = ""
                # Pilih teks terbaik: yang paling panjang (setelah dibersihkan)
                for method_name, text in page_text_candidates.items():
                    cleaned_current_text = clean_extracted_text(text)
                    if len(cleaned_current_text) > len(best_page_text):
                        best_page_text = cleaned_current_text
                        logger.debug(f"Page {page_num + 1}: Using {method_name} method (length {len(best_page_text)})")
                
                if best_page_text:
                    full_text_parts.append(f"\n=== HALAMAN {page_num + 1} ===\n")
                    full_text_parts.append(best_page_text)
                else:
                    logger.warning(f"No meaningful text extracted from page {page_num + 1} of {file_path.name}")
                    
            except Exception as e:
                logger.error(f"Error processing page {page_num + 1} of {file_path.name}: {e}")
                # Lanjutkan ke halaman berikutnya meskipun ada error pada satu halaman
                continue 
        
        doc.close()
        return "\n".join(full_text_parts).strip()
        
    except fitz.FileDataError as e:
        logger.error(f"PDF file corrupted or invalid: {file_path}: {e}")
        return None
    except Exception as e:
        logger.error(f"Error opening or processing PDF {file_path}: {e}")
        return None

# --- Fungsi Pembersihan Teks ---

def clean_extracted_text(text: str) -> str:
    """
    Membersihkan teks yang diekstrak dari PDF.
    Fokus pada normalisasi spasi/baris baru dan penghapusan artefak yang sangat jelas.
    Tujuan utama adalah mempertahankan sebanyak mungkin teks asli tanpa pemotongan yang tidak disengaja.
    
    Args:
        text (str): Teks mentah yang diekstrak dari PDF.
        
    Returns:
        str: Teks yang sudah bersih dan dinormalisasi dengan filtering minimal.
    """
    if not text:
        return ""
    
    # 1. Normalisasi spasi dan baris baru
    text = text.replace('\x00', ' ') # Hapus karakter null (sering muncul dari PDF)
    text = re.sub(r'[\r\n]+', '\n', text) # Normalisasi CRLF ke LF, hapus multiple newlines menjadi single newline
    text = re.sub(r'[ \t]+', ' ', text) # Normalisasi spasi dan tab ganda menjadi single space
    text = text.strip() # Hapus spasi/newline di awal/akhir setelah normalisasi

    if not text: # Jika teks menjadi kosong setelah normalisasi awal
        return ""
    
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        line = line.strip()
        if not line: # Lewati baris yang kosong setelah di-strip
            continue
        
        # 2. Filter baris yang tidak diinginkan (artefak yang sangat spesifik dan jelas)
        # Filter ini dirancang untuk sangat konservatif, hanya membuang yang pasti artefak.
        
        # Filter nomor halaman yang sangat jelas:
        # Contoh: "1", "- 2 -", "-- 3 --", "Page 4"
        if re.fullmatch(r'^\s*[-_]?\s*\d+\s*[-_]?\s*$', line, re.IGNORECASE) or \
           re.fullmatch(r'^\s*[Pp][Aa][Gg][Ee]\s+\d+\s*$', line, re.IGNORECASE):
            continue
        
        # Filter garis horizontal atau deretan simbol sederhana yang berulang:
        # Contoh: "------------", "*********", "========"
        # Hanya filter jika baris *seluruhnya* terdiri dari satu atau dua karakter non-alphanumeric berulang.
        # Ini adalah filter yang sangat ketat untuk menghindari pemotongan teks valid.
        if len(line) > 5: # Harus cukup panjang agar dianggap garis pemisah
            # Periksa jika semua karakter adalah non-alphanumeric dan non-spasi
            if all(not char.isalnum() and not char.isspace() for char in line):
                # Dan hanya ada 1 atau 2 karakter unik non-alphanumeric (e.g., hanya '-' atau '*').
                if len(set(char for char in line if not char.isspace())) <= 2:
                    continue

        # Filter baris yang sangat pendek dan tidak mengandung huruf/angka (seringkali sisa-sisa simbol atau numbering yang salah)
        if len(line) <= 3 and not re.search(r'[a-zA-Z0-9]', line):
            continue
        
        # Filter Roman numerals jika tampak sebagai penomoran list/sub-bagian yang berdiri sendiri.
        # Contoh: "I.", "II.", "V."
        if len(line) <= 5 and re.fullmatch(r'^[ivxlc]+\.$', line, re.IGNORECASE):
            continue

        # Semua filter yang lebih agresif (misalnya berdasarkan panjang atau komposisi kata)
        # dihilangkan sepenuhnya di sini. Teks akan dipertahankan sebisa mungkin.

        cleaned_lines.append(line)
    
    # Gabungkan baris yang bersih dengan satu newline. 
    # Pada tahap ini, kita memaksimalkan retensi teks. Rekonstruksi paragraf yang lebih canggih
    # atau identifikasi bagian-bagian dokumen akan dilakukan di `02_case_representation.py`
    # karena di sana kita memiliki konteks lebih baik tentang pola-pola dalam dokumen.
    return "\n".join(cleaned_lines).strip()

# --- Fungsi Penyimpanan File ---

def save_text_file(content: str, file_path: Path) -> bool:
    """
    Menyimpan konten teks ke file yang ditentukan.
    """
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        return True
    except Exception as e:
        logger.error(f"Error saving text file {file_path}: {e}")
        return False

def save_json_file(data: Any, file_path: Path) -> bool:
    """
    Menyimpan data (umumnya daftar kasus atau kamus) ke file JSON.
    """
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"JSON saved: {file_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving JSON file {file_path}: {e}")
        return False

def load_existing_cases_data(file_path: Path) -> Dict[str, Dict[str, Any]]:
    """
    Memuat data kasus yang sudah ada dari file JSON. 
    Mengembalikan kamus yang dipetakan berdasarkan case_id untuk pencarian cepat.
    """
    if file_path.exists():
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    # Mengubah list kasus menjadi dict {case_id: case_obj}
                    return {c.get('case_id', generate_case_id_from_filename(f"temp_{idx}")): c 
                            for idx, c in enumerate(data) if c.get('case_id')}
                logger.warning(f"Existing file {file_path} is not a list. Starting fresh.")
        except Exception as e:
            logger.warning(f"Error loading existing data from {file_path}: {e}. Starting fresh.")
    return {} # Mengembalikan dict kosong jika tidak ada file atau error

# --- Fungsi Utama Pemrosesan ---

def main():
    """
    Fungsi utama untuk mengekstrak teks dari semua file PDF di INPUT_DIR,
    membersihkannya, dan menyimpan ke file teks mentah dan JSON metadata.
    """
    try:
        # Langkah 1: Pastikan semua direktori yang diperlukan ada.
        ensure_directories()
        
        # Langkah 2: Periksa apakah folder input PDF ada dan berisi file.
        if not INPUT_DIR.exists():
            logger.error(f"Input directory not found: {INPUT_DIR}")
            print(f"Silakan buat folder '{INPUT_DIR}' dan masukkan file PDF di dalamnya.")
            return
        
        # Langkah 3: Muat data kasus yang sudah ada untuk menghindari pemrosesan ulang.
        # Menggunakan OUTPUT_CASES_JSON sebagai sumber utama untuk data yang sudah diproses.
        existing_cases_data = load_existing_cases_data(OUTPUT_CASES_JSON)
        logger.info(f"Loaded {len(existing_cases_data)} existing cases from {OUTPUT_CASES_JSON}.")
        
        # Langkah 4: Dapatkan daftar semua file PDF di direktori input.
        pdf_files = list(INPUT_DIR.glob("*.pdf"))
        
        if not pdf_files:
            logger.warning(f"No PDF files found in {INPUT_DIR}")
            print(f"Tidak ada file PDF di folder '{INPUT_DIR}'")
            return
        
        logger.info(f"Found {len(pdf_files)} PDF files to process.")
        print(f"🔍 Ditemukan {len(pdf_files)} file PDF untuk diproses.")
        
        # Variabel untuk melacak status pemrosesan
        processed_count = 0
        failed_count = 0
        skipped_count = 0
        
        # Daftar untuk menyimpan semua metadata kasus yang baru diproses
        all_processed_cases_metadata = list(existing_cases_data.values())

        # Kamus untuk menyimpan teks mentah yang diekstrak untuk referensi
        extracted_raw_texts = {} 
        
        # Langkah 5: Iterasi (ulang) melalui setiap file PDF.
        for pdf_file_path in tqdm(pdf_files, desc="Memproses PDF"):
            try:
                filename = pdf_file_path.name
                file_stem = pdf_file_path.stem # Nama file tanpa ekstensi
                
                # Cek apakah file ini sudah diproses sebelumnya
                # Membandingkan stem nama file, bukan case_id lengkap yang termasuk timestamp.
                # Ini lebih robust untuk pengecekan duplikasi dari file asli.
                is_already_processed = False
                for existing_case_id in existing_cases_data.keys():
                    if file_stem.lower() in existing_case_id.lower():
                        is_already_processed = True
                        break

                if is_already_processed:
                    logger.info(f"File {filename} (stem: {file_stem}) already processed, skipping.")
                    skipped_count += 1
                    continue
                
                logger.info(f"Processing: {filename}")
                
                # Ekstrak teks dari PDF
                full_text = extract_text_from_pdf(pdf_file_path)
                
                # Periksa apakah ekstraksi berhasil dan teks cukup panjang.
                if not full_text or len(full_text.strip()) < 200: # Batas minimal 200 karakter
                    logger.error(f"Failed to extract sufficient text from {filename} or text too short.")
                    failed_count += 1
                    continue
                
                # Hasilkan case_id unik, termasuk timestamp untuk memastikan keunikan setiap kali diproses
                case_id = f"{generate_case_id_from_filename(file_stem)}_{datetime.now().strftime('%Y%m%d%H%M%S')}"

                # Simpan teks mentah ke file .txt di direktori RAW_TXT_DIR
                txt_filename = RAW_TXT_DIR / f"{case_id}.txt"
                if not save_text_file(full_text, txt_filename):
                    failed_count += 1
                    continue # Lanjutkan ke file berikutnya jika gagal menyimpan
                
                # Tambahkan metadata kasus ke daftar
                # Ini akan menjadi dasar untuk 02_case_representation.py
                case_metadata = {
                    "case_id": case_id,
                    "file_name": filename,
                    "file_size": os.path.getsize(pdf_file_path), # Ukuran file PDF asli
                    "raw_text_file": str(txt_filename), # Path ke file teks mentah
                    "processed_at": datetime.now().isoformat(),
                    "extracted_text_preview": full_text[:500] + "..." if len(full_text) > 500 else full_text # Pratinjau teks
                }
                all_processed_cases_metadata.append(case_metadata)
                
                # Opsional: simpan teks mentah ke kamus jika Anda ingin menggabungkannya nanti
                # Ini bisa menjadi sangat besar dan mungkin tidak perlu disimpan jika hanya untuk debugging
                extracted_raw_texts[case_id] = full_text
                
                processed_count += 1
                logger.info(f"Successfully processed {filename} -> {case_id}")
                
                # Simpan progress ke file JSON setiap beberapa file yang berhasil diproses.
                if processed_count % 5 == 0:
                    # Simpan daftar metadata kasus (penting untuk pipeline selanjutnya)
                    save_json_file(all_processed_cases_metadata, OUTPUT_CASES_JSON)
                    # Simpan teks mentah yang diekstrak (opsional)
                    # save_json_file(extracted_raw_texts, OUTPUT_TEXT_MAP_JSON) 
                    logger.info(f"Progress saved: {processed_count} files processed so far.")
                
            except Exception as e:
                logger.error(f"Error processing {filename}: {e}", exc_info=True) # exc_info=True untuk traceback
                failed_count += 1
                continue
        
        # Langkah 6: Simpan semua metadata kasus yang berhasil diproses ke file JSON final.
        if all_processed_cases_metadata:
            save_json_file(all_processed_cases_metadata, OUTPUT_CASES_JSON)
            # Simpan juga teks mentah yang diekstrak jika diperlukan
            # save_json_file(extracted_raw_texts, OUTPUT_TEXT_MAP_JSON)
        else:
            logger.warning("No cases were successfully processed. No output JSON created.")
            
        # Langkah 7: Tampilkan ringkasan pemrosesan.
        print(f"\n=== RINGKASAN PEMROSESAN ===")
        print(f"Total file PDF ditemukan: {len(pdf_files)}")
        print(f"Berhasil diproses: {processed_count}")
        print(f"Gagal diproses: {failed_count}")
        print(f"Dilewati (sudah ada): {skipped_count}")
        print(f"Total kasus dalam database (termasuk yang sudah ada): {len(all_processed_cases_metadata)}")
        
        if processed_count > 0:
            print(f"\nFile hasil tersimpan di:")
            print(f"- Raw text files: {RAW_TXT_DIR}")
            print(f"- Processed cases metadata: {OUTPUT_CASES_JSON}")
            # print(f"- Extracted raw texts (optional): {OUTPUT_TEXT_MAP_JSON}") # Jika Anda memilih untuk menyimpannya
        print("=============================")
            
    except Exception as e:
        logger.critical(f"Critical error in main function: {e}", exc_info=True)
        print(f"Terjadi error kritis pada pemrosesan: {e}")

# --- Titik Masrip Skrip ---
# Memastikan fungsi main() dijalankan hanya ketika skrip dieksekusi langsung.
if __name__ == "__main__":
    main()
