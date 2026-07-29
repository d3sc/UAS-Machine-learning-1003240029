from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

RAW_DATA_PATH = DATA_DIR / "used_cars_dataset.csv"
TRAIN_DATA_PATH = DATA_DIR / "train.csv"
TEST_DATA_PATH = DATA_DIR / "test.csv"
MODEL_PATH = MODELS_DIR / "model.joblib"
METADATA_PATH = MODELS_DIR / "metadata.json"

TARGET = "price"
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

EXPECTED_COLUMNS = [
    "brand",
    "model",
    "year",
    "price",
    "transmission",
    "mileage",
    "fuel_type",
    "tax",
    "mpg",
    "engine_size",
]

