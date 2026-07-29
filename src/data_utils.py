"""Utilitas normalisasi dan transformasi domain di dalam sklearn Pipeline."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


ALIASES = {
    "automatic": "Automatic",
    "manual": "Manual",
    "semi-auto": "Semi-Auto",

    "petrol": "Petrol",
    "gasoline": "Petrol",

    "diesel": "Diesel",
    "hybrid": "Hybrid",
    "electric": "Electric",

    "other": "Other",
}


def normalize_text(value: object) -> object:
    if pd.isna(value):
        return np.nan

    text = re.sub(r"\s+", " ", str(value)).strip()

    if text == "":
        return np.nan

    return ALIASES.get(text.lower(), text.title())


class VehicleDomainTransformer(BaseEstimator, TransformerMixin):
    """
    Feature engineering untuk dataset Used Cars.
    """

    def __init__(self, reference_year: int):
        self.reference_year = reference_year

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        frame = X.copy()

        frame.columns = [
            str(c).strip().lower()
            for c in frame.columns
        ]

        # Pastikan numerik
        numeric_columns = [
            "year",
            "mileage",
            "engine_size",
            "tax",
            "mpg",
        ]

        for col in numeric_columns:
            frame[col] = pd.to_numeric(
                frame[col],
                errors="coerce",
            )

        # Feature engineering
        frame["vehicle_age"] = (
            self.reference_year - frame["year"]
        ).clip(lower=0)

        # Normalisasi kategori
        categorical = [
            "make",
            "model",
            "transmission",
            "fuel_type",
        ]

        for col in categorical:
            frame[col] = frame[col].map(normalize_text)

        return frame[
            [
                "make",
                "model",
                "transmission",
                "fuel_type",
                "mileage",
                "engine_size",
                "tax",
                "mpg",
                "vehicle_age",
            ]
        ]