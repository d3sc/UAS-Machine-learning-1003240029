"""Factory preprocessing dan model agar training serta test memakai kontrak yang sama."""

from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import RANDOM_STATE
from src.data_utils import VehicleDomainTransformer

NUMERIC_FEATURES = [
    "mileage",
    "engine_size",
    "tax",
    "mpg",
    "vehicle_age",
]

CATEGORICAL_FEATURES = [
    "make",
    "model",
    "transmission",
    "fuel_type",
]


def build_preprocessor(reference_year: int) -> Pipeline:
    numeric = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=5,
                ),
            ),
        ]
    )

    columns = ColumnTransformer(
        [
            ("numeric", numeric, NUMERIC_FEATURES),
            ("categorical", categorical, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )

    return Pipeline(
        [
            ("domain", VehicleDomainTransformer(reference_year)),
            ("columns", columns),
        ]
    )


def candidate_models():
    return {
        "linear_regression": LinearRegression(),
        "ridge": Ridge(alpha=10),
        "random_forest": RandomForestRegressor(
            n_estimators=250,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def build_pipeline(model, reference_year):
    regressor = Pipeline(
        [
            ("preprocess", build_preprocessor(reference_year)),
            ("model", model),
        ]
    )

    return TransformedTargetRegressor(
        regressor=regressor,
        func=np.log1p,
        inverse_func=np.expm1,
        check_inverse=False,
    )