# 🧠 Case-Based Reasoning (CBR) for Legal Decision Support

Sistem ini adalah implementasi sederhana dari metode **Case-Based Reasoning (CBR)** untuk menganalisis putusan pengadilan. Sistem memproses dokumen PDF putusan, mengekstrak informasi penting, dan memungkinkan pencarian kasus mirip berdasarkan ringkasan fakta.

---

## 📁 Struktur Direktori

```
CBR_Project/
│
├── data/                       # Folder utama untuk semua data
│   ├── eval/                   # Data untuk evaluasi performa sistem
│   │   ├── queries.json        # Daftar pertanyaan atau kasus uji
│   │   ├── retrieval_metrics.csv     # Metrik evaluasi retrieval
│   │   └── prediction_metrics.csv    # Metrik evaluasi prediksi
│   ├── logs/                   # Log selama proses ekstraksi dan pembersihan
│   │   ├── extraction.log
│   │   └── cleaning.log
│   ├── pdf/                    # PDF putusan asli (input awal)
│   ├── raw/                    # Output teks mentah hasil ekstraksi dari PDF
│   ├── processed/              # Hasil representasi kasus dalam format JSON
│   │   └── cases.json
│   └── results/                # Hasil retrieval dan prediksi solusi
│       ├── retrieved_cases.json
│       └── predictions.csv
│
├── logs/                      # Folder log tambahan (opsional)
│
├── scripts/                   # Skrip Python utama untuk menjalankan tiap tahap
│   ├── 01_pdf_to_text.py              # Ekstraksi teks dari PDF
│   ├── 02_case_representation.py      # Representasi kasus ke format JSON
│   ├── 03_retrieval.py                # Retrieval kasus mirip menggunakan TF-IDF/BERT
│   ├── 04_predict.py                  # Prediksi solusi dari top-k hasil retrieval
│   ├── 05_evaluation.py               # Evaluasi performa model
│   └── make_queries.py                # Membuat query uji untuk evaluasi
│
├── notebook/                 # Versi notebook dari tiap tahap (opsional)
│   ├── 01_pdf_to_text.ipynb
│   ├── 02_case_representation.ipynb
│   ├── 03_retrieval.ipynb
│   ├── 04_predict.ipynb
│   └── 05_evaluation.ipynb
│
├── reports/                  # Laporan akhir
│   └── laporan_CBR.docx
│
├── requirements.txt          # Daftar dependensi Python
├── .gitignore                # File untuk mengecualikan file dari Git
└── README.md                 # Dokumentasi proyek
```

---



## ✅ Alur Proses & Eksekusi

Eksekusinya :


### 1. Ekstrak PDF → Teks Mentah

```bash

python scripts/01_pdf_to_text.py #Run yang pertama 

```

📁 Output ke: `data/raw/*.txt` 


### 2. Ekstrak Metadata → JSON Terstruktur

```bash

python scripts/02_case_representation.py #Run yang ke 2

```
             

📁 Output ke: `data/processed/cases.json`  

## opsional Run 


### 3. Query & Retrieval

```bash

python scripts/make_queries.py         # Menyusun query uji

python scripts/03_retrieval.py      
   # TF-IDF / BERT Retrieval

```

📁 Output: `results/retrieved_cases.json`


### 4. Prediksi Solusi

```bash
python scripts/04_predict.py
```

📁 Output: `results/predictions.csv`

### 5. Evaluasi

python scripts/05_evaluation.py

📁 Output: `eval/retrieval_metrics.csv` & `prediction_metrics.csv`

---

## 💻 Instalasi

### 🔧 Persiapan

1. Pastikan Python 3.8+ sudah terpasang
2. Install semua dependensi dengan:

```bash
pip install -r requirements.txt
```

### 📦 requirements.txt

```txt
pandas
scikit-learn
transformers
pdfminer.six
jupyter
```

---

## 🧪 Contoh `queries.json`

```json
[
  {
    "query_id": "q1",
    "query_text": "Terdakwa menerima suap dalam proyek pembangunan jalan",
    "ground_truth": "putusan_113_k_pid.sus_2020_..."
  }
]
```

---

## 📓 Menjalankan versi Notebook

> Alternatif interaktif untuk skrip Python

### Jalankan Jupyter:

```bash
jupyter notebook
```

### File Notebook:

* 📘 `notebook/01_pdf_to_text.ipynb`
* 📘 `notebook/02_case_representation.ipynb`
* 📘 `notebook/03_retrieval.ipynb`
* 📘 `notebook/04_predict.ipynb`
* 📘 `notebook/05_evaluation.ipynb`

---

## 📄 Laporan

📂 `reports/laporan_CBR.docx`
Berisi ringkasan tahapan, diagram pipeline, metrik evaluasi, serta diskusi kasus-kasus yang gagal (error analysis).

---

## ✨ Credits

* Dibuat oleh: Ellyas Prambudyas
* Proyek ini ditujukan untuk pembelajaran sistem penalaran berbasis kasus (CBR) dengan domain hukum (putusan pengadilan).
