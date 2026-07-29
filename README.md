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