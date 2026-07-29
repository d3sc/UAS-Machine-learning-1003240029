"""Split dahulu, bandingkan 3 algoritma dengan 5-fold CV, tuning, lalu simpan pipeline."""

from __future__ import annotations

import json
import platform
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import GridSearchCV, KFold, cross_validate, train_test_split

from src.config import (
    CV_FOLDS,
    METADATA_PATH,
    MODEL_PATH,
    MODELS_DIR,
    RANDOM_STATE,
    REPORTS_DIR,
    TARGET,
    TEST_DATA_PATH,
    TEST_SIZE,
    TRAIN_DATA_PATH,
)

from src.load_data import load_raw_data
from src.modeling import build_pipeline, candidate_models


def prepare_raw_supervised(df):
    frame = df.copy()

    frame.columns = [str(c).strip().lower() for c in frame.columns]

    frame = frame.drop_duplicates()

    frame[TARGET] = pd.to_numeric(frame[TARGET], errors="coerce")

    frame = frame.dropna(subset=[TARGET])

    frame = frame[frame[TARGET] > 0]

    y = frame.pop(TARGET)

    return frame.reset_index(drop=True), y.reset_index(drop=True)


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    X, y = prepare_raw_supervised(load_raw_data())
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # Simpan data mentah hasil split; test belum diprediksi atau dievaluasi di file ini.
    X_train.assign(**{TARGET: y_train.to_numpy()}).to_csv(TRAIN_DATA_PATH, index=False)
    X_test.assign(**{TARGET: y_test.to_numpy()}).to_csv(TEST_DATA_PATH, index=False)

    reference_year = int(X_train["year"].max()) + 1
    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    cv_results: dict[str, dict[str, float]] = {}
    fitted_candidates: dict[str, object] = {}
    for name, estimator in candidate_models().items():
        pipeline = build_pipeline(estimator, reference_year)
        scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            scoring="neg_mean_absolute_error",
            cv=cv,
            n_jobs=-1,
            return_train_score=False,
        )
        fold_mae = -scores["test_score"]
        cv_results[name] = {
            "mae_mean": float(fold_mae.mean()),
            "mae_std": float(fold_mae.std(ddof=1)),
        }
        fitted_candidates[name] = pipeline
        print(f"{name:20s} MAE CV = {fold_mae.mean():,.2f} +/- {fold_mae.std(ddof=1):,.2f}")

    # Tuning eksplisit untuk model non-linear yang menangani tantangan hubungan umur-harga.
    tuned_search = GridSearchCV(
        build_pipeline(candidate_models()["random_forest"], reference_year),
        param_grid={
            "regressor__model__n_estimators": [150, 300],
            "regressor__model__max_depth": [12, None],
            "regressor__model__min_samples_leaf": [1, 3],
            "regressor__model__max_features": ["sqrt"],
        },
        scoring="neg_mean_absolute_error",
        cv=cv,
        n_jobs=-1,
        refit=True,
    )
    tuned_search.fit(X_train, y_train)
    tuned_mae = float(-tuned_search.best_score_)
    cv_results["random_forest_tuned"] = {
        "mae_mean": tuned_mae,
        "mae_std": None,
    }

    base_winner = min(cv_results.keys() - {"random_forest_tuned"}, key=lambda n: cv_results[n]["mae_mean"])
    if tuned_mae <= cv_results[base_winner]["mae_mean"]:
        selected_name = "random_forest_tuned"
        final_pipeline = tuned_search.best_estimator_
        best_params = tuned_search.best_params_
    else:
        selected_name = base_winner
        final_pipeline = fitted_candidates[base_winner].fit(X_train, y_train)
        best_params = {}

    joblib.dump(final_pipeline, MODEL_PATH)
    metadata = {
        "project": "Estimasi Harga Kendaraan Bekas",
        "nim": "1003240029",
        "model_version": "1.0.0",
        "selected_model": selected_name,
        "primary_metric": "MAE",
        "metric_reason": "MAE dipilih karena mudah diinterpretasikan sebagai rata-rata selisih harga kendaraan dan lebih tahan terhadap outlier dibanding RMSE.",
        "cv_results": cv_results,
        "best_params": best_params,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "reference_year": reference_year,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "feature_columns": list(X.columns),
        "dataset_url": "https://www.kaggle.com/datasets/aishwaryamuthukumar/cars-dataset-audi-bmw-ford-hyundai-skoda-vw",
        "dataset_license": "Other",
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "versions": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
        "test_evaluated": False,
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2, allow_nan=True), encoding="utf-8")

    table = pd.DataFrame(cv_results).T.sort_values("mae_mean")
    table.to_csv(REPORTS_DIR / "cv_results.csv")
    print(f"\nModel terpilih: {selected_name}")
    print(f"Pipeline tersimpan: {MODEL_PATH}")
    print("Test set belum dievaluasi. Jalankan: python -m src.evaluate")


if __name__ == "__main__":
    main()
