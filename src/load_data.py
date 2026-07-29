"""Mengunduh dataset publik dan mencetak audit struktur data mentah."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from src.config import DATA_DIR, EXPECTED_COLUMNS, RAW_DATA_PATH

KAGGLE_HANDLE = "aishwaryamuthukumar/cars-dataset-audi-bmw-ford-hyundai-skoda-vw"

COLUMN_ALIASES = {
    "fueltype": "fuel_type",
    "enginesize": "engine_size",
}


def download_dataset(force: bool = False) -> Path:
    """
    Mengunduh dataset Kaggle dan menggabungkan seluruh file CSV
    menjadi satu dataset.
    """

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if RAW_DATA_PATH.exists() and not force:
        return RAW_DATA_PATH

    os.environ.setdefault(
        "KAGGLEHUB_CACHE",
        str(DATA_DIR / ".kagglehub-cache")
    )

    try:
        import kagglehub
    except ImportError as exc:
        raise RuntimeError("Jalankan: pip install -r requirements.txt") from exc

    dataset_dir = Path(kagglehub.dataset_download(KAGGLE_HANDLE))

    csv_files = sorted(dataset_dir.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError("Dataset tidak memiliki file CSV.")

    frames = []

    for csv_file in csv_files:

        df = pd.read_csv(csv_file)

        # normalisasi nama kolom
        df.columns = [
            str(c).strip().replace(" ", "").lower()
            for c in df.columns
        ]

        df = df.rename(columns=COLUMN_ALIASES)

        # tambahkan brand berdasarkan nama file
        df["brand"] = csv_file.stem.capitalize()

        frames.append(df)

    df_all = pd.concat(frames, ignore_index=True)

    df_all.to_csv(RAW_DATA_PATH, index=False)

    return RAW_DATA_PATH


def load_raw_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:

    if not path.exists():
        path = download_dataset()

    df = pd.read_csv(path)

    df.columns = [
        str(c).strip().replace(" ", "").lower()
        for c in df.columns
    ]

    df = df.rename(columns=COLUMN_ALIASES)

    return df


def print_data_audit(df: pd.DataFrame) -> None:

    print(f"Jumlah baris : {df.shape[0]:,}")
    print(f"Jumlah kolom : {df.shape[1]}")

    print("\nNama kolom:")
    print(df.columns.tolist())

    print("\nTipe setiap kolom:")
    print(df.dtypes.to_string())

    print("\nJumlah nilai hilang per kolom:")
    print(df.isna().sum().to_string())

    missing_expected = [
        c for c in EXPECTED_COLUMNS
        if c not in df.columns
    ]

    if missing_expected:
        print(f"\nPERINGATAN: kolom yang tidak ditemukan: {missing_expected}")


def main() -> None:

    path = download_dataset(force=True)

    print(f"Dataset tersimpan di: {path}")

    df = load_raw_data(path)

    print_data_audit(df)


if __name__ == "__main__":
    main()