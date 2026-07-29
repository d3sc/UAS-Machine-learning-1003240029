"""FastAPI untuk menyajikan pipeline estimasi harga kendaraan bekas."""

from __future__ import annotations

import json
import logging
import os
from contextlib import asynccontextmanager
from enum import Enum
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = ROOT / "models" / "model.joblib"
DEFAULT_METADATA_PATH = ROOT / "models" / "metadata.json"
LOG_PATH = ROOT / "reports" / "predictions.log"
DEFAULT_BDT_TO_IDR = 135.0

logger = logging.getLogger("vehicle-price-api")
logger.setLevel(logging.INFO)
if not logger.handlers:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)


class Transmission(str, Enum):
    manual = "Manual"
    automatic = "Automatic"


class FuelType(str, Enum):
    petrol = "Petrol"
    diesel = "Diesel"
    hybrid = "Hybrid"
    electric = "Electric"
    cng = "CNG"
    lpg = "LPG"


class VehicleInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    brand: str = Field(min_length=2, max_length=60, examples=["Toyota"])
    car_model: str = Field(min_length=1, max_length=100, examples=["Corolla"])
    model_year: int = Field(ge=1980, le=2026, examples=[2018])
    transmission: Transmission
    body_type: str = Field(min_length=2, max_length=50, examples=["Sedan"])
    fuel_type: FuelType
    engine_capacity: float = Field(gt=300, le=10000, examples=[1500])
    kilometers_run: float = Field(ge=0, le=1_500_000, examples=[65000])


class PriceRange(BaseModel):
    minimum: float
    maximum: float


class PredictionResponse(BaseModel):
    estimasi_harga: float
    mata_uang: str
    kurs_bdt_ke_idr: float
    rentang_perkiraan: PriceRange
    keyakinan: str
    model_version: str


state: dict[str, object] = {"model": None, "metadata": {}, "load_error": None}


def _artifact_paths() -> tuple[Path, Path]:
    model_path = Path(os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH)))
    metadata_path = Path(os.getenv("METADATA_PATH", str(DEFAULT_METADATA_PATH)))
    return model_path, metadata_path


def load_artifacts() -> None:
    model_path, metadata_path = _artifact_paths()
    try:
        state["model"] = joblib.load(model_path)
        state["metadata"] = json.loads(metadata_path.read_text(encoding="utf-8"))
        state["load_error"] = None
        logger.info("model_loaded path=%s", model_path)
    except Exception as exc:
        state["model"] = None
        state["metadata"] = {}
        state["load_error"] = str(exc)
        logger.error("model_load_failed error=%s", exc)


@asynccontextmanager
async def lifespan(_: FastAPI):
    load_artifacts()
    yield
    state["model"] = None


app = FastAPI(
    title="API Estimasi Harga Kendaraan Bekas",
    version="1.0.0",
    description="Kasus B UAS Machine Learning End-to-End - NIM 1003240029",
    lifespan=lifespan,
)


@app.get("/")
def root() -> dict[str, object]:
    return {
        "layanan": "Estimasi harga kendaraan bekas",
        "endpoint_prediksi": "POST /predict-harga",
        "dokumentasi": "/docs",
    }


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "ok" if state["model"] is not None else "degraded",
        "model_loaded": state["model"] is not None,
        "model_version": dict(state["metadata"]).get("model_version") if state["metadata"] else None,
    }


@app.post("/predict-harga", response_model=PredictionResponse)
def predict_price(payload: VehicleInput) -> PredictionResponse:
    model = state["model"]
    metadata = dict(state["metadata"])
    if model is None:
        raise HTTPException(status_code=503, detail="Model belum tersedia. Jalankan training terlebih dahulu.")

    raw = payload.model_dump(mode="json")
    frame = pd.DataFrame([raw])
    prediction_bdt = max(float(np.asarray(model.predict(frame))[0]), 0.0)
    bdt_to_idr = float(os.getenv("BDT_TO_IDR", str(DEFAULT_BDT_TO_IDR)))
    if not np.isfinite(bdt_to_idr) or bdt_to_idr <= 0:
        raise HTTPException(status_code=500, detail="Konfigurasi kurs BDT_TO_IDR tidak valid.")

    metrics = metadata.get("test_metrics", {})
    error_bdt = float(metrics.get("absolute_error_p90", metrics.get("mae", prediction_bdt * 0.30)))
    prediction = prediction_bdt * bdt_to_idr
    error = error_bdt * bdt_to_idr
    lower = max(0.0, prediction - error)
    upper = prediction + error
    relative_width = error / max(prediction, 1.0)
    confidence = "tinggi" if relative_width <= 0.25 else "sedang" if relative_width <= 0.50 else "rendah"

    logger.info(
        "prediction brand=%s model=%s year=%s price_idr=%.2f rate_bdt_to_idr=%.4f confidence=%s",
        payload.brand,
        payload.car_model,
        payload.model_year,
        prediction,
        bdt_to_idr,
        confidence,
    )
    return PredictionResponse(
        estimasi_harga=round(prediction, 2),
        mata_uang="IDR",
        kurs_bdt_ke_idr=bdt_to_idr,
        rentang_perkiraan=PriceRange(minimum=round(lower, 2), maximum=round(upper, 2)),
        keyakinan=confidence,
        model_version=str(metadata.get("model_version", "unknown")),
    )

