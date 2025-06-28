import os
import re
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime

# Konfigurasi path
RAW_DIR = Path("data/raw")
OUTPUT_FILE = Path("data/processed/cases.json")
LOG_FILE = Path("data/logs/extraction.log")

def setup_logging():
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(LOG_FILE, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

class EnhancedPatternExtractor:
    def __init__(self):
        # Pattern untuk nomor perkara - lebih comprehensive
        self.NOMOR_PERKARA_PATTERNS = [
            re.compile(r"Nomor\s*[:\-]?\s*(\d{1,5}[\/\-]\w{2,10}[\/\-\.\w]*\d{4})", re.IGNORECASE),
            re.compile(r"No\.\s*(\d{1,5}[\/\-]\w{2,10}[\/\-\.\w]*\d{4})", re.IGNORECASE),
            re.compile(r"(\d{1,5}\s*[\/\-]\s*\w{2,10}[\/\-\.\w]*\s*[\/\-]\s*\d{4})", re.IGNORECASE),
            re.compile(r"PUTUSAN\s+Nomor\s+(\d{1,5}[\/\-\.\w]*\d{4})", re.IGNORECASE),
            re.compile(r"(\d{1,5}\s*K\s*[\/\-]\s*Pid[\.\s]*Sus[\/\-\.\w]*\d{4})", re.IGNORECASE),
            re.compile(r"(\d{1,5}\s*[\/\-]\s*PID[\.\s]*SUS[\/\-\.\w]*\d{4})", re.IGNORECASE)
        ]
        
        # Pattern untuk tanggal - lebih fleksibel
        self.DATE_PATTERNS = [
            re.compile(r"(\d{1,2})\s+(Januari|Februari|Maret|April|Mei|Juni|Juli|Agustus|September|Oktober|November|Desember)\s+(\d{4})", re.IGNORECASE),
            re.compile(r"(\d{1,2})[\/\-](\d{1,2})[\/\-](\d{4})"),
            re.compile(r"(\d{4})[\/\-](\d{1,2})[\/\-](\d{1,2})"),
            re.compile(r"tanggal\s+(\d{1,2})\s+(\w+)\s+(\d{4})", re.IGNORECASE),
            re.compile(r"pada\s+hari\s+\w+\s+tanggal\s+(\d{1,2})\s+(\w+)\s+(\d{4})", re.IGNORECASE)
        ]
        
        # Pattern untuk jenis perkara
        self.JENIS_PERKARA_PATTERNS = [
            re.compile(r"Tindak\s+Pidana\s+Korupsi", re.IGNORECASE),
            re.compile(r"Tipikor", re.IGNORECASE),
            re.compile(r"Korupsi", re.IGNORECASE),
            re.compile(r"Narkotika", re.IGNORECASE),
            re.compile(r"Pidana\s+Khusus", re.IGNORECASE),
            re.compile(r"Pidana\s+Umum", re.IGNORECASE),
            re.compile(r"Perdata", re.IGNORECASE),
            re.compile(r"Tata\s+Usaha\s+Negara", re.IGNORECASE),
            re.compile(r"TUN", re.IGNORECASE)
        ]
        
        # Pattern untuk pasal
        self.PASAL_PATTERNS = [
            re.compile(r"Pasal\s+\d+(?:\s+ayat\s*\(\d+\))?(?:\s+(?:jo\.?|juncto|dan)\s+Pasal\s+\d+(?:\s+ayat\s*\(\d+\))?)*", re.IGNORECASE),
            re.compile(r"melanggar\s+Pasal\s+\d+[^\.]*", re.IGNORECASE),
            re.compile(r"berdasarkan\s+Pasal\s+\d+[^\.]*", re.IGNORECASE)
        ]
        
        # Pattern untuk data personal
        self.PERSONAL_PATTERNS = {
            'nama': [
                re.compile(r"Nama\s*(?:Lengkap)?\s*[:\-]\s*([A-Z][^:\n]{2,50})", re.IGNORECASE),
                re.compile(r"Terdakwa\s*[:\-]\s*([A-Z][^:\n]{2,50})", re.IGNORECASE),
                re.compile(r"a\.n\.\s*([A-Z][^:\n]{2,50})", re.IGNORECASE)
            ],
            'umur': [
                re.compile(r"Umur[\/\s]*Tanggal\s*[:\-]\s*(\d{1,3})\s*(?:tahun|thn)", re.IGNORECASE),
                re.compile(r"Umur\s*[:\-]\s*(\d{1,3})\s*(?:tahun|thn)", re.IGNORECASE),
                re.compile(r"(\d{1,3})\s*(?:tahun|thn)", re.IGNORECASE)
            ],
            'jenis_kelamin': [
                re.compile(r"Jenis\s+Kelamin\s*[:\-]\s*(Laki-laki|Perempuan)", re.IGNORECASE),
                re.compile(r"Kelamin\s*[:\-]\s*(Laki-laki|Perempuan)", re.IGNORECASE),
                re.compile(r"(Laki-laki|Perempuan)", re.IGNORECASE)
            ],
            'pekerjaan': [
                re.compile(r"Pekerjaan\s*[:\-]\s*([^:\n]{3,50})", re.IGNORECASE),
                re.compile(r"Jabatan\s*[:\-]\s*([^:\n]{3,50})", re.IGNORECASE)
            ],
            'alamat': [
                re.compile(r"(?:Tempat\s+Tinggal|Alamat)\s*[:\-]\s*([^:\n]{10,200})", re.IGNORECASE),
                re.compile(r"beralamat\s+di\s+([^:\n]{10,200})", re.IGNORECASE),
                re.compile(r"bertempat\s+tinggal\s+di\s+([^:\n]{10,200})", re.IGNORECASE)
            ]
        }

class ImprovedSmartExtractor:
    def __init__(self):
        self.patterns = EnhancedPatternExtractor()
        self.logger = logging.getLogger(__name__)
        
        # Bulan Indonesia ke angka
        self.BULAN_MAP = {
            'januari': '01', 'februari': '02', 'maret': '03', 'april': '04',
            'mei': '05', 'juni': '06', 'juli': '07', 'agustus': '08',
            'september': '09', 'oktober': '10', 'november': '11', 'desember': '12'
        }

    def clean_text(self, text: str) -> str:
        """Bersihkan teks dari karakter tidak perlu"""
        if not text:
            return ""
        
        # Hapus karakter null dan normalize whitespace
        text = text.replace('\x00', ' ').replace('\r\n', '\n')
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n+', '\n', text)
        return text.strip()

    def extract_nomor_perkara(self, text: str) -> str:
        """Ekstrak nomor perkara dengan berbagai pattern"""
        # Cari di 5000 karakter pertama
        search_area = text[:5000]
        
        for pattern in self.patterns.NOMOR_PERKARA_PATTERNS:
            matches = pattern.findall(search_area)
            if matches:
                # Ambil yang paling lengkap (terpanjang)
                nomor = max(matches, key=len)
                # Bersihkan spacing berlebih
                nomor = re.sub(r'\s+', ' ', nomor).strip()
                self.logger.debug(f"Found nomor perkara: {nomor}")
                return nomor
        
        # Fallback: cari pattern nomor di awal dokumen
        lines = search_area.split('\n')[:20]  # 20 baris pertama
        for line in lines:
            if re.search(r'\d{1,5}[\/\-]\w+[\/\-]\d{4}', line):
                match = re.search(r'(\d{1,5}[\/\-][\w\-\.]+[\/\-]\d{4})', line)
                if match:
                    return match.group(1).strip()
        
        return ""

    def extract_tanggal(self, text: str) -> str:
        """Ekstrak tanggal dengan berbagai format"""
        search_area = text[:8000]
        
        for pattern in self.patterns.DATE_PATTERNS:
            matches = pattern.finditer(search_area)
            for match in matches:
                # Cek konteks untuk menghindari tanggal lahir
                start_pos = max(0, match.start() - 100)
                end_pos = min(len(search_area), match.end() + 100)
                context = search_area[start_pos:end_pos].lower()
                
                # Skip jika konteks menunjukkan tanggal lahir
                if any(keyword in context for keyword in ['lahir', 'usia', 'ktp', 'identitas']):
                    continue
                
                # Format tanggal
                if len(match.groups()) == 3:
                    day, month, year = match.groups()
                    
                    # Konversi bulan jika dalam bentuk nama
                    if month.lower() in self.BULAN_MAP:
                        month = self.BULAN_MAP[month.lower()]
                    
                    # Validasi tanggal
                    try:
                        day_num = int(day)
                        month_num = int(month) if month.isdigit() else 0
                        year_num = int(year)
                        
                        if 1 <= day_num <= 31 and 1 <= month_num <= 12 and 1900 <= year_num <= 2030:
                            return match.group(0)
                    except ValueError:
                        continue
        
        return ""

    def extract_jenis_perkara(self, text: str) -> str:
        """Ekstrak jenis perkara"""
        search_area = text[:5000]
        
        for pattern in self.patterns.JENIS_PERKARA_PATTERNS:
            match = pattern.search(search_area)
            if match:
                return match.group(0).strip().title()
        
        # Fallback berdasarkan konteks
        if re.search(r'korupsi|suap|gratifikasi', search_area, re.IGNORECASE):
            return "Tindak Pidana Korupsi"
        elif re.search(r'narkoba|narkotika|psikotropika', search_area, re.IGNORECASE):
            return "Narkotika"
        
        return ""

    def extract_pasal(self, text: str) -> List[str]:
        """Ekstrak pasal-pasal yang dilanggar"""
        pasal_found = set()
        
        for pattern in self.patterns.PASAL_PATTERNS:
            matches = pattern.finditer(text)
            for match in matches:
                pasal_text = match.group(0).strip()
                # Bersihkan teks pasal
                pasal_text = re.sub(r'\s+', ' ', pasal_text)
                pasal_found.add(pasal_text)
        
        return sorted(list(pasal_found))

    def extract_personal_data(self, text: str, field: str) -> str:
        """Ekstrak data personal berdasarkan field"""
        if field not in self.patterns.PERSONAL_PATTERNS:
            return ""
        
        search_area = text[:10000]  # Cari di 10k karakter pertama
        
        for pattern in self.patterns.PERSONAL_PATTERNS[field]:
            matches = pattern.findall(search_area)
            if matches:
                # Ambil match pertama yang valid
                for match in matches:
                    if isinstance(match, tuple):
                        match = match[0] if match else ""
                    
                    match = match.strip()
                    
                    # Validasi berdasarkan field
                    if field == 'umur' and match.isdigit() and 10 <= int(match) <= 100:
                        return match
                    elif field == 'jenis_kelamin' and match.lower() in ['laki-laki', 'perempuan']:
                        return match.title()
                    elif field in ['nama', 'pekerjaan', 'alamat'] and len(match) >= 3:
                        # Bersihkan dari karakter aneh
                        match = re.sub(r'[^\w\s\.\-,]', '', match).strip()
                        if match:
                            return match
        
        return ""

    def extract_status_hukuman(self, text: str) -> str:
        """Ekstrak status hukuman/putusan"""
        # Cari di bagian akhir dokumen
        search_area = text[-5000:] if len(text) > 5000 else text
        
        # Pattern untuk putusan
        patterns = [
            r'(?:menyatakan|memutuskan).*?(?:terbukti|bersalah|tidak terbukti|bebas)[^\.]*\.?',
            r'(?:pidana|hukuman).*?(?:penjara|denda|kurungan)[^\.]*\.?',
            r'(?:terdakwa|pemohon).*?(?:dipidana|dijatuhi|dihukum)[^\.]*\.?'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, search_area, re.IGNORECASE | re.DOTALL)
            if matches:
                # Ambil yang terpendek tapi informatif
                for match in matches:
                    match = match.strip()
                    if 20 <= len(match) <= 200:
                        return match
        
        return ""

    def extract_ringkasan_fakta(self, text: str, max_len: int = 800) -> str:
        """Ekstrak ringkasan fakta dari dokumen"""
        # Bersihkan teks
        text = self.clean_text(text)
        
        # Hilangkan disclaimer dan footer
        cut_patterns = [
            r'Disclaimer.*$',
            r'Halaman\s+\d+.*$',
            r'Mahkamah\s+Agung.*$',
            r'kepaniteraan@mahkamahagung\.go\.id.*$'
        ]
        
        for pattern in cut_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.DOTALL)
        
        # Ambil bagian utama (skip header)
        lines = text.split('\n')
        content_start = 0
        
        # Skip header lines
        for i, line in enumerate(lines[:50]):
            if len(line.strip()) > 50 and not re.match(r'^[A-Z\s]+$', line.strip()):
                content_start = i
                break
        
        # Gabungkan konten utama
        content = '\n'.join(lines[content_start:])
        
        # Potong sesuai max_len
        if len(content) > max_len:
            content = content[:max_len]
            # Potong di akhir kalimat terdekat
            last_period = content.rfind('.')
            if last_period > max_len // 2:
                content = content[:last_period + 1]
        
        return content.strip()

    def extract_metadata(self, text: str) -> Dict[str, str]:
        """Ekstrak semua metadata dari teks"""
        if not text:
            return {}
        
        text = self.clean_text(text)
        
        # Ekstrak semua field
        metadata = {
            "no_perkara": self.extract_nomor_perkara(text),
            "tanggal": self.extract_tanggal(text),
            "jenis_perkara": self.extract_jenis_perkara(text),
            "pasal": "; ".join(self.extract_pasal(text)),
            "nama": self.extract_personal_data(text, 'nama'),
            "umur": self.extract_personal_data(text, 'umur'),
            "jenis_kelamin": self.extract_personal_data(text, 'jenis_kelamin'),
            "pekerjaan": self.extract_personal_data(text, 'pekerjaan'),
            "alamat": self.extract_personal_data(text, 'alamat'),
            "status_hukuman": self.extract_status_hukuman(text),
            "ringkasan_fakta": self.extract_ringkasan_fakta(text)
        }
        
        # Log hasil ekstraksi
        found_fields = [k for k, v in metadata.items() if v]
        self.logger.debug(f"Extracted fields: {found_fields}")
        
        return metadata

def process_all_cases():
    """Proses semua file txt dalam folder raw"""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Pastikan folder ada
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    # Cari file txt
    txt_files = list(RAW_DIR.glob("*.txt"))
    if not txt_files:
        logger.warning(f"No .txt files found in {RAW_DIR}")
        print(f"❌ Tidak ada file .txt ditemukan di {RAW_DIR}")
        return
    
    logger.info(f"Found {len(txt_files)} .txt files to process")
    print(f"🔍 Ditemukan {len(txt_files)} file untuk diproses")
    
    extractor = ImprovedSmartExtractor()
    results = []
    success_count = 0
    error_count = 0
    
    for file_path in txt_files:
        try:
            logger.info(f"Processing {file_path.name}")
            
            # Baca file
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            if not text.strip():
                logger.warning(f"File {file_path.name} is empty")
                continue
            
            # Ekstrak metadata
            metadata = extractor.extract_metadata(text)
            
            # Tambahkan info file
            metadata.update({
                "case_id": file_path.stem,
                "file_name": file_path.name,
                "file_size": len(text),
                "processed_at": datetime.now().isoformat()
            })
            
            results.append(metadata)
            success_count += 1
            
            # Log hasil
            found_count = sum(1 for v in metadata.values() if v and v != "")
            print(f"✅ {file_path.name} - {found_count} field terisi")
            
        except Exception as e:
            error_count += 1
            logger.error(f"Failed to process {file_path.name}: {e}")
            print(f"❌ Error processing {file_path.name}: {e}")
    
    # Simpan hasil
    if results:
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Successfully saved {len(results)} cases to {OUTPUT_FILE}")
        print(f"💾 Berhasil menyimpan {len(results)} kasus ke {OUTPUT_FILE}")
        
        # Statistik
        print(f"\n📊 STATISTIK EKSTRAKSI:")
        print(f"Total file: {len(txt_files)}")
        print(f"Berhasil: {success_count}")
        print(f"Error: {error_count}")
        
        # Hitung field yang berhasil diekstrak
        field_stats = {}
        for result in results:
            for field, value in result.items():
                if field not in ['case_id', 'file_name', 'file_size', 'processed_at']:
                    if field not in field_stats:
                        field_stats[field] = 0
                    if value and value.strip():
                        field_stats[field] += 1
        
        print(f"\n📈 EKSTRAKSI PER FIELD:")
        for field, count in sorted(field_stats.items()):
            percentage = (count / len(results)) * 100
            print(f"{field}: {count}/{len(results)} ({percentage:.1f}%)")
    
    else:
        logger.warning("No cases were successfully processed")
        print("❌ Tidak ada kasus yang berhasil diproses")

if __name__ == "__main__":
    process_all_cases()