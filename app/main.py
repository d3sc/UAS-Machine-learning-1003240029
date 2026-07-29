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

logger = logging.getLogger("vehicle-price-api")
logger.setLevel(logging.INFO)

if not logger.handlers:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    handler = logging.FileHandler(
        LOG_PATH,
        encoding="utf-8",
    )

    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)s %(message)s"
        )
    )

    logger.addHandler(handler)


# =====================================================
# ENUM
# =====================================================

class Transmission(str, Enum):
    manual = "Manual"
    automatic = "Automatic"
    semiauto = "Semi-Auto"


class FuelType(str, Enum):
    petrol = "Petrol"
    diesel = "Diesel"
    hybrid = "Hybrid"
    electric = "Electric"
    other = "Other"


# =====================================================
# REQUEST
# =====================================================

class VehicleInput(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    make: str = Field(
        min_length=2,
        max_length=40,
        examples=["Ford"],
    )

    model: str = Field(
        min_length=1,
        max_length=80,
        examples=["Fiesta"],
    )

    year: int = Field(
        ge=1996,
        le=2026,
        examples=[2018],
    )

    transmission: Transmission

    fuel_type: FuelType

    mileage: float = Field(
        ge=0,
        le=400000,
        examples=[65000],
    )

    tax: float = Field(
        ge=0,
        le=1000,
        examples=[145],
    )

    mpg: float = Field(
        ge=0,
        le=600,
        examples=[55.4],
    )

    engine_size: float = Field(
        ge=0,
        le=7,
        examples=[1.5],
    )


# =====================================================
# RESPONSE
# =====================================================

class PriceRange(BaseModel):
    minimum: float
    maximum: float


class PredictionResponse(BaseModel):
    estimasi_harga: float
    mata_uang: str
    rentang_perkiraan: PriceRange
    keyakinan: str
    model_version: str


# =====================================================
# GLOBAL STATE
# =====================================================

state: dict[str, object] = {
    "model": None,
    "metadata": {},
    "load_error": None,
}


def _artifact_paths() -> tuple[Path, Path]:

    model_path = Path(
        os.getenv(
            "MODEL_PATH",
            str(DEFAULT_MODEL_PATH),
        )
    )

    metadata_path = Path(
        os.getenv(
            "METADATA_PATH",
            str(DEFAULT_METADATA_PATH),
        )
    )

    return model_path, metadata_path


def load_artifacts() -> None:

    model_path, metadata_path = _artifact_paths()

    try:
        state["model"] = joblib.load(model_path)

        state["metadata"] = json.loads(
            metadata_path.read_text(
                encoding="utf-8"
            )
        )

        state["load_error"] = None

        logger.info(
            "Model loaded: %s",
            model_path,
        )

    except Exception as exc:

        state["model"] = None
        state["metadata"] = {}
        state["load_error"] = str(exc)

        logger.exception(
            "Failed loading model"
        )


@asynccontextmanager
async def lifespan(_: FastAPI):

    load_artifacts()

    yield

    state["model"] = None


# =====================================================
# FASTAPI
# =====================================================

app = FastAPI(
    title="API Estimasi Harga Kendaraan Bekas",
    description="UAS Machine Learning End-to-End",
    version="1.0.0",
    lifespan=lifespan,
)


# =====================================================
# ROOT
# =====================================================

@app.get("/")
def root():

    return {
        "project": "Estimasi Harga Kendaraan Bekas",
        "dataset": "Cars Dataset (Audi, BMW, Ford, Hyundai, Skoda, VW)",
        "endpoint": "POST /predict-harga",
        "documentation": "/docs",
    }


# =====================================================
# HEALTH
# =====================================================

@app.get("/health")
def health():

    return {
        "status": "ok" if state["model"] else "degraded",
        "model_loaded": state["model"] is not None,
        "model_version": state["metadata"].get("model_version")
        if state["metadata"]
        else None,
    }


# =====================================================
# PREDICT
# =====================================================

@app.post(
    "/predict-harga",
    response_model=PredictionResponse,
)
def predict_price(payload: VehicleInput):

    model = state["model"]

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model belum tersedia.",
        )

    metadata = dict(state["metadata"])

    frame = pd.DataFrame(
        [payload.model_dump(mode="json")]
    )

    prediction = max(
        float(
            np.asarray(
                model.predict(frame)
            )[0]
        ),
        0.0,
    )

    metrics = metadata.get(
        "test_metrics",
        {},
    )

    error = float(
        metrics.get(
            "absolute_error_p90",
            metrics.get(
                "mae",
                prediction * 0.30,
            ),
        )
    )

    lower = max(
        0.0,
        prediction - error,
    )

    upper = prediction + error

    relative_width = error / max(
        prediction,
        1.0,
    )

    if relative_width <= 0.25:
        confidence = "tinggi"
    elif relative_width <= 0.50:
        confidence = "sedang"
    else:
        confidence = "rendah"

    logger.info(
        (
            "Prediction "
            "make=%s "
            "model=%s "
            "year=%s "
            "price=%.2f "
            "confidence=%s"
        ),
        payload.make,
        payload.model,
        payload.year,
        prediction,
        confidence,
    )

    return PredictionResponse(
        estimasi_harga=round(
            prediction,
            2,
        ),
        mata_uang="GBP",
        rentang_perkiraan=PriceRange(
            minimum=round(
                lower,
                2,
            ),
            maximum=round(
                upper,
                2,
            ),
        ),
        keyakinan=confidence,
        model_version=str(
            metadata.get(
                "model_version",
                "unknown",
            )
        ),
    )