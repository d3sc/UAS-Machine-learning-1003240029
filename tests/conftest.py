from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def client():
    model_path = ROOT / "models" / "model.joblib"
    metadata_path = ROOT / "models" / "metadata.json"
    if not model_path.exists() or not metadata_path.exists():
        pytest.fail(
            "Artefak model belum ada. Jalankan berurutan: "
            "python -m src.load_data && python -m src.train && python -m src.evaluate"
        )
    with TestClient(app) as test_client:
        yield test_client

