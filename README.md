# UAS Machine Learning - Estimasi Harga Mobil Bekas

**Nama:** Ikbar Rabbani Purwanto  
**NIM:** 1003240029  
**Kasus:** Regresi

---

# Deskripsi Proyek

Proyek ini membangun model Machine Learning untuk **memprediksi harga mobil bekas** berdasarkan karakteristik kendaraan seperti:

- Merk mobil
- Model mobil
- Tahun produksi
- Transmisi
- Jenis bahan bakar
- Mileage
- Engine Size
- Pajak kendaraan
- Fuel Economy (MPG)

Seluruh proses dilakukan secara end-to-end mulai dari pengambilan dataset, Exploratory Data Analysis (EDA), preprocessing, training model, hyperparameter tuning, hingga evaluasi model.

---

# Dataset

Dataset yang digunakan berasal dari Kaggle.

**Nama Dataset**

Cars Dataset (Audi, BMW, Ford, Hyundai, Skoda, VW)

**Sumber**

https://www.kaggle.com/datasets/aishwaryamuthukumar/cars-dataset-audi-bmw-ford-hyundai-skoda-vw

Dataset berisi:

- 72.435 data kendaraan
- 11 atribut
- 6 merek mobil:
  - Audi
  - BMW
  - Ford
  - Hyundai
  - Skoda
  - Volkswagen

Target yang diprediksi adalah:

```
price
```

Semua harga pada dataset menggunakan **Pound Sterling (GBP / £)**.

---

# Struktur Project

```text
UAS-SEMESTER-EMAPT-ML-1003240029/
│
├── data/
│   ├── used_cars_dataset.csv
│   ├── train.csv
│   └── test.csv
│
├── models/
│   ├── model.joblib
│   └── metadata.json
│
├── reports/
│   ├── 01_distribusi_harga.png
│   ├── 02_missing_values.png
│   ├── 03_umur_vs_harga.png
│   ├── 04_harga_per_bahan_bakar.png
│   ├── 05_korelasi_numerik.png
│   ├── 06_actual_vs_predicted.png
│   ├── 07_residual_plot.png
│   ├── cv_results.csv
│   ├── worst_5_errors.csv
│   ├── test_metrics.json
│   ├── eda_summary.json
│   └── eda_interpretation.md
│
├── src/
│   ├── config.py
│   ├── load_data.py
│   ├── data_utils.py
│   ├── modeling.py
│   ├── eda.py
│   ├── train.py
│   └── evaluate.py
│
├── requirements.txt
└── README.md
```

---

# Library

- Python 3.12+
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn
- Joblib
- KaggleHub

---

# Menjalankan Project

## Install dependency

```bash
pip install -r requirements.txt
```

---

## Download Dataset

```bash
python -m src.load_data
```

Dataset akan otomatis diunduh menggunakan KaggleHub dan disimpan ke folder:

```
data/
```

---

## Exploratory Data Analysis

```bash
python -m src.eda
```

Output:

- Statistik dataset
- Missing value
- Duplicate data
- 5 grafik EDA
- Ringkasan hasil EDA

---

## Training Model

```bash
python -m src.train
```

Model yang dibandingkan:

- Linear Regression
- Ridge Regression
- Random Forest

Evaluasi menggunakan:

- 5 Fold Cross Validation
- MAE (Mean Absolute Error)

Random Forest kemudian dilakukan GridSearchCV untuk memperoleh parameter terbaik.

---

## Evaluasi Model

```bash
python -m src.evaluate
```

Output:

- MAE
- RMSE
- R²
- Scatter Plot
- Residual Plot
- Worst Prediction

---

# Hasil Exploratory Data Analysis

Dataset terdiri dari:

| Informasi | Nilai |
|------------|-------|
| Jumlah Data | 72.435 |
| Jumlah Kolom | 11 |
| Missing Value | 0 |
| Duplicate Data | 842 |

Distribusi kendaraan:

- Tahun produksi antara 1996–2020
- Mayoritas kendaraan diproduksi tahun 2016–2019
- Mayoritas menggunakan transmisi Manual
- Fuel type didominasi Petrol
- Model Ford Fiesta merupakan model yang paling banyak muncul

Rata-rata harga kendaraan:

```
£16,580
```

Median harga:

```
£14,495
```

---

# Feature yang Digunakan

## Numerical

- mileage
- engine_size
- tax
- mpg
- vehicle_age

## Categorical

- make
- model
- transmission
- fuel_type

Target:

```
price
```

---

# Feature Engineering

Model menambahkan fitur baru:

```
vehicle_age
```

yang dihitung menggunakan rumus:

```
vehicle_age = reference_year - year
```

Feature ini digunakan karena umur kendaraan sangat berpengaruh terhadap harga jual mobil bekas.

---

# Model Machine Learning

Tiga algoritma dibandingkan:

1. Linear Regression
2. Ridge Regression
3. Random Forest Regressor

Target ditransformasikan menggunakan:

```
log1p(price)
```

untuk mengurangi pengaruh outlier pada harga kendaraan.

---

# Hasil Cross Validation

| Model | MAE |
|--------|------:|
| Linear Regression | 1,561.62 |
| Ridge Regression | 1,584.04 |
| Random Forest | 1,150.31 |

Model terbaik setelah proses tuning adalah:

```
Random Forest (GridSearchCV)
```

---

# Hasil Evaluasi Test Set

| Metric | Nilai |
|---------|-------:|
| MAE | 1,108.65 |
| RMSE | 1,827.19 |
| R² Score | 0.9620 |
| P90 Absolute Error | 2,417.01 |

Nilai R² sebesar **96,20%** menunjukkan bahwa model mampu menjelaskan sebagian besar variasi harga kendaraan pada data uji.

---

# Output yang Dihasilkan

Folder `reports/` akan berisi:

- Distribusi harga
- Missing value
- Vehicle Age vs Price
- Fuel Type vs Price
- Correlation Heatmap
- Actual vs Prediction
- Residual Plot
- Cross Validation Result
- Worst 5 Prediction
- Ringkasan EDA
- Ringkasan Evaluasi

---

# Preprocessing

Pipeline preprocessing terdiri dari:

- Duplicate Removal
- Missing Value Imputation
- One Hot Encoding
- Standard Scaler
- Vehicle Age Engineering

Semua preprocessing dilakukan menggunakan **Scikit-Learn Pipeline** sehingga proses training dan evaluasi menggunakan transformasi yang konsisten.

---

# Reproduksi

Untuk menjalankan ulang seluruh proses:

```bash
python -m src.load_data
python -m src.eda
python -m src.train
python -m src.evaluate
```

---

# Menjalankan REST API

Project ini menyediakan REST API menggunakan **FastAPI** untuk melakukan prediksi harga mobil bekas menggunakan model Machine Learning yang telah dilatih.

## Membuat Virtual Environment API

### Windows

```powershell
python -m venv .venv-api
.\.venv-api\Scripts\Activate.ps1
```

### macOS / Linux

```bash
python -m venv .venv-api
source .venv-api/bin/activate
```

Install dependency API.

```bash
pip install -r requirements-api.txt
```

Jalankan server FastAPI.

```bash
uvicorn app.main:app --reload
```

Apabila berhasil dijalankan, server akan aktif pada alamat berikut.

```
http://127.0.0.1:8000
```

---

# Dokumentasi API

FastAPI secara otomatis menyediakan dokumentasi interaktif.

| Dokumentasi | URL |
|-------------|-----|
| Swagger UI | http://127.0.0.1:8000/docs |
| ReDoc | http://127.0.0.1:8000/redoc |

Melalui halaman tersebut pengguna dapat mencoba endpoint secara langsung tanpa menggunakan Postman.

---

# Daftar Endpoint

## GET /

Menampilkan informasi singkat mengenai layanan API.

### Request

```http
GET /
```

### Response

```json
{
  "project": "Estimasi Harga Kendaraan Bekas",
  "dataset": "Cars Dataset (Audi, BMW, Ford, Hyundai, Skoda, VW)",
  "endpoint": "POST /predict-harga",
  "documentation": "/docs"
}
```

---

## GET /health

Digunakan untuk memastikan model Machine Learning berhasil dimuat.

### Request

```http
GET /health
```

### Response

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_version": "1.0.0"
}
```

Keterangan:

- **status** menunjukkan kondisi API.
- **model_loaded** menunjukkan apakah model berhasil dimuat.
- **model_version** merupakan versi model yang tersimpan pada `metadata.json`.

---

## POST /predict-harga

Endpoint utama untuk melakukan prediksi harga kendaraan bekas.

### Request

```json
{
  "make": "Ford",
  "model": "Fiesta",
  "year": 2018,
  "transmission": "Manual",
  "fuel_type": "Petrol",
  "mileage": 65000,
  "tax": 145,
  "mpg": 55.4,
  "engine_size": 1.5
}
```

### Contoh Response

```json
{
  "estimasi_harga": 13980.25,
  "mata_uang": "GBP",
  "rentang_perkiraan": {
    "minimum": 11563.24,
    "maximum": 16397.26
  },
  "keyakinan": "tinggi",
  "model_version": "1.0.0"
}
```

---

# Validasi Input

API menggunakan **Pydantic** sehingga seluruh request akan divalidasi sebelum diproses oleh model Machine Learning.

Validasi yang diterapkan antara lain:

- Seluruh field wajib diisi.
- Tidak diperbolehkan mengirim field di luar skema.
- Nilai `year` harus berada pada rentang yang ditentukan.
- Nilai `mileage`, `tax`, `mpg`, dan `engine_size` harus berada pada batas yang diperbolehkan.
- `transmission` hanya menerima:
  - Manual
  - Automatic
- `fuel_type` hanya menerima:
  - Petrol
  - Diesel
  - Hybrid
  - Electric

Apabila request tidak sesuai dengan skema tersebut, API akan mengembalikan status **422 Unprocessable Entity**.

---

# Menjalankan Automated Testing

Project menyediakan pengujian otomatis menggunakan **Pytest**.

Install dependency testing.

```bash
python -m pip install -r requirements-dev.txt
```

Jalankan seluruh pengujian.

```bash
python -m pytest tests/ -v
```

Contoh hasil apabila seluruh pengujian berhasil dijalankan.

```text
============================= test session starts =============================

collected 8 items

tests/test_api.py::test_root_returns_service_info PASSED
tests/test_api.py::test_health_returns_200_and_model_loaded PASSED
tests/test_api.py::test_valid_prediction_returns_expected_schema PASSED
tests/test_api.py::test_missing_required_field_returns_422 PASSED
tests/test_api.py::test_unknown_enum_returns_422 PASSED
tests/test_api.py::test_out_of_range_value_returns_422 PASSED
tests/test_api.py::test_older_vehicle_is_predicted_cheaper PASSED
tests/test_api.py::test_higher_mileage_is_not_predicted_more_expensive PASSED

============================== 8 passed ==============================
```

---

# Penjelasan Pengujian

Project memiliki delapan buah unit test untuk memastikan setiap endpoint berjalan sesuai dengan yang diharapkan.

| No | Nama Test | Fungsi |
|----|-----------|--------|
| 1 | `test_root_returns_service_info` | Memastikan endpoint root (`GET /`) dapat diakses dan menampilkan informasi layanan. |
| 2 | `test_health_returns_200_and_model_loaded` | Memastikan model berhasil dimuat dan endpoint health berjalan dengan benar. |
| 3 | `test_valid_prediction_returns_expected_schema` | Memastikan request yang valid menghasilkan prediksi dengan struktur response yang benar. |
| 4 | `test_missing_required_field_returns_422` | Memastikan field wajib yang tidak dikirim menghasilkan HTTP 422. |
| 5 | `test_unknown_enum_returns_422` | Memastikan nilai enum yang tidak valid ditolak oleh API. |
| 6 | `test_out_of_range_value_returns_422` | Memastikan nilai numerik di luar batas menghasilkan HTTP 422. |
| 7 | `test_older_vehicle_is_predicted_cheaper` | Memastikan kendaraan dengan tahun produksi lebih lama diprediksi memiliki harga lebih rendah dibanding kendaraan yang lebih baru dengan spesifikasi yang sama. |
| 8 | `test_higher_mileage_is_not_predicted_more_expensive` | Memastikan kendaraan dengan mileage lebih tinggi tidak diprediksi memiliki harga lebih mahal dibanding kendaraan identik dengan mileage lebih rendah. |

---

# Alur Sistem

```text
Dataset
   │
   ▼
load_data.py
   │
   ▼
Exploratory Data Analysis (EDA)
   │
   ▼
Training Model
   │
   ▼
Random Forest Terbaik
   │
   ▼
model.joblib
   │
   ▼
FastAPI
   │
   ▼
REST API
   │
   ▼
Prediksi Harga Mobil Bekas
```

---

# Teknologi yang Digunakan

| Komponen | Teknologi |
|----------|-----------|
| Bahasa Pemrograman | Python 3.12+ |
| Machine Learning | Scikit-Learn |
| Data Processing | Pandas |
| Komputasi Numerik | NumPy |
| Visualisasi Data | Matplotlib |
| Statistik | Seaborn |
| Model Serialization | Joblib |
| REST API | FastAPI |
| Web Server | Uvicorn |
| Validasi Data | Pydantic |
| Automated Testing | Pytest |
| Dataset Downloader | KaggleHub |

---

# Struktur API

```
GET   /                 → Informasi layanan API
GET   /health           → Status model Machine Learning
POST  /predict-harga    → Prediksi harga kendaraan bekas
```

Seluruh endpoint telah diuji menggunakan **Pytest** dan menghasilkan **8/8 pengujian berhasil (100% passed)** sehingga API siap digunakan sebagai layanan prediksi harga kendaraan bekas.

# Catatan

Model yang dihasilkan disimpan pada:

```
models/model.joblib
```

Informasi training disimpan pada:

```
models/metadata.json
```

Seluruh grafik dan hasil analisis akan tersimpan pada folder:

```
reports/
```

Sehingga seluruh eksperimen dapat direproduksi kembali tanpa melakukan perubahan pada source code.