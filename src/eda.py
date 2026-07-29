"""EDA wajib: audit kualitas data, lima grafik, dan temuan yang dapat ditafsirkan."""

from __future__ import annotations

import json
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-uas-ml")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import REPORTS_DIR, TARGET
from src.data_utils import normalize_text
from src.load_data import load_raw_data

sns.set_theme(style="whitegrid", context="notebook")


def cleaned_for_eda(raw: pd.DataFrame) -> pd.DataFrame:
    frame = raw.copy()

    frame.columns = [str(c).strip().lower() for c in frame.columns]

    # Normalisasi kolom kategorikal
    for column in ["make", "model", "transmission", "fuel_type"]:
        if column in frame.columns:
            frame[column] = frame[column].map(normalize_text)

    # Pastikan kolom numerik
    numeric_columns = [
        "year",
        "price",
        "mileage",
        "tax",
        "mpg",
        "engine_size",
    ]

    for column in numeric_columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    # Feature engineering sederhana
    reference_year = int(frame["year"].max()) + 1
    frame["vehicle_age"] = reference_year - frame["year"]

    return frame


def save_current(filename: str) -> None:
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / filename, dpi=180, bbox_inches="tight")
    plt.close()


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    raw = load_raw_data()
    df = cleaned_for_eda(raw)

    print("=== NILAI HILANG ===")
    print(df.isna().sum().to_string())

    print("\n=== STATISTIK DESKRIPTIF ===")
    print(df.describe(include="all").transpose().to_string())

    print(f"\nJumlah baris duplikat: {raw.duplicated().sum():,}")

    # ==========================================================
    # 1. Distribusi Harga
    # ==========================================================

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.histplot(
        df[TARGET].dropna(),
        bins=35,
        kde=True,
        ax=axes[0],
        color="#2563EB",
    )

    axes[0].set(
        title="Distribusi Harga Kendaraan",
        xlabel="Price",
        ylabel="Jumlah",
    )

    sns.histplot(
        np.log1p(df[TARGET].dropna()),
        bins=35,
        kde=True,
        ax=axes[1],
        color="#0F766E",
    )

    axes[1].set(
        title="Distribusi log(1 + Price)",
        xlabel="log(1 + price)",
        ylabel="Jumlah",
    )

    save_current("01_distribusi_harga.png")

    # ==========================================================
    # 2. Missing Value
    # ==========================================================

    missing = df.isna().sum().sort_values(ascending=False)

    plt.figure(figsize=(10, 5))

    sns.barplot(
        x=missing.values,
        y=missing.index,
        color="#F59E0B",
    )

    plt.title("Jumlah Missing Value")
    plt.xlabel("Jumlah")
    plt.ylabel("Kolom")

    save_current("02_missing_values.png")

    # ==========================================================
    # 3. Vehicle Age vs Price
    # ==========================================================

    sample = df.dropna(subset=["vehicle_age", TARGET]).copy()

    upper = sample[TARGET].quantile(0.99)

    sample = sample[sample[TARGET] <= upper]

    plt.figure(figsize=(10, 5.5))

    sns.scatterplot(
        data=sample,
        x="vehicle_age",
        y=TARGET,
        alpha=0.35,
        color="#2563EB",
    )

    sns.regplot(
        data=sample,
        x="vehicle_age",
        y=TARGET,
        scatter=False,
        order=2,
        color="#DC2626",
    )

    plt.title("Vehicle Age vs Price")
    plt.xlabel("Vehicle Age (Year)")
    plt.ylabel("Price")

    save_current("03_vehicle_age_vs_price.png")

    # ==========================================================
    # 4. Fuel Type vs Price
    # ==========================================================

    common_fuels = df["fuel_type"].value_counts().head(6).index

    fuel = df[
        (df["fuel_type"].isin(common_fuels))
        & (df[TARGET] <= df[TARGET].quantile(0.99))
    ]

    plt.figure(figsize=(10, 5.5))

    sns.boxplot(
        data=fuel,
        x="fuel_type",
        y=TARGET,
        showfliers=False,
        color="#6DCBF4",
    )

    plt.title("Price by Fuel Type")
    plt.xlabel("Fuel Type")
    plt.ylabel("Price")

    save_current("04_price_by_fuel_type.png")

    # ==========================================================
    # 5. Korelasi
    # ==========================================================

    numeric = df[
        [
            "year",
            "mileage",
            "engine_size",
            "tax",
            "mpg",
            "vehicle_age",
            TARGET,
        ]
    ]

    plt.figure(figsize=(8, 6))

    sns.heatmap(
        numeric.corr(method="spearman"),
        annot=True,
        fmt=".2f",
        cmap="vlag",
        center=0,
    )

    plt.title("Spearman Correlation")

    save_current("05_korelasi_numerik.png")

    findings = {
        "rows": int(raw.shape[0]),
        "columns": int(raw.shape[1]),
        "duplicates": int(raw.duplicated().sum()),
        "missing_after_normalization": {
            k: int(v) for k, v in missing.items()
        },
        "price_q1": float(df[TARGET].quantile(0.25)),
        "price_q3": float(df[TARGET].quantile(0.75)),
        "price_p99": float(df[TARGET].quantile(0.99)),
        "reference_year": int(df["year"].max()) + 1,
    }

    (REPORTS_DIR / "eda_summary.json").write_text(
        json.dumps(findings, indent=2),
        encoding="utf-8",
    )

    interpretation = f"""# Ringkasan EDA

## 1. Distribusi Harga
Distribusi harga kendaraan bersifat right-skewed. Nilai persentil ke-99 sekitar {findings['price_p99']:,.0f}. Oleh karena itu transformasi log1p layak dipertimbangkan saat proses training.

## 2. Missing Value
Dataset memiliki sedikit atau bahkan tidak memiliki missing value. Jika terdapat missing value pada tahap preprocessing, penanganannya dapat menggunakan SimpleImputer.

## 3. Hubungan Umur Kendaraan dan Harga
Semakin tua umur kendaraan, secara umum harga semakin rendah. Hubungan ini tidak sepenuhnya linear sehingga model berbasis pohon kemungkinan memiliki performa lebih baik.

## 4. Fuel Type
Harga kendaraan bervariasi menurut jenis bahan bakar. Boxplot menunjukkan adanya perbedaan distribusi harga antar fuel type.

## 5. Korelasi
Heatmap Spearman digunakan untuk melihat hubungan antar fitur numerik seperti year, mileage, engine_size, tax, mpg, vehicle_age, dan price.

## Prakiraan Sebelum Training

- Vehicle age diperkirakan memiliki korelasi negatif terhadap price.
- Random Forest diperkirakan memberikan performa lebih baik dibanding Linear Regression karena mampu menangkap hubungan non-linear.
- Transformasi log terhadap target dapat membantu mengurangi pengaruh kendaraan dengan harga yang sangat tinggi.
"""

    (REPORTS_DIR / "eda_interpretation.md").write_text(
        interpretation,
        encoding="utf-8",
    )

    print(f"\nLima grafik dan ringkasan tersimpan di: {REPORTS_DIR}")


if __name__ == "__main__":
    main()