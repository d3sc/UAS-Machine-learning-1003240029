"""Sentuh test set satu kali untuk evaluasi final dan analisis lima error terburuk."""

from __future__ import annotations

import json
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-uas-ml")

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from src.config import (
    METADATA_PATH,
    MODEL_PATH,
    REPORTS_DIR,
    TARGET,
    TEST_DATA_PATH,
)

sns.set_theme(style="whitegrid")


def main() -> None:

    if not MODEL_PATH.exists() or not TEST_DATA_PATH.exists():
        raise FileNotFoundError(
            "Jalankan `python -m src.train` terlebih dahulu."
        )

    model = joblib.load(MODEL_PATH)

    test = pd.read_csv(TEST_DATA_PATH)

    y_test = pd.to_numeric(
        test.pop(TARGET),
        errors="coerce",
    )

    valid = y_test.notna()

    test = test.loc[valid].reset_index(drop=True)
    y_test = y_test.loc[valid].reset_index(drop=True)

    # Prediksi hanya sekali terhadap test set
    predictions = np.maximum(
        model.predict(test),
        0,
    )

    mae = float(
        mean_absolute_error(
            y_test,
            predictions,
        )
    )

    rmse = float(
        np.sqrt(
            mean_squared_error(
                y_test,
                predictions,
            )
        )
    )

    r2 = float(
        r2_score(
            y_test,
            predictions,
        )
    )

    abs_error = np.abs(
        y_test.to_numpy() - predictions
    )

    p90_error = float(
        np.quantile(
            abs_error,
            0.90,
        )
    )

    metrics = {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "absolute_error_p90": p90_error,
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    (REPORTS_DIR / "test_metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    # ====================================================
    # Actual vs Predicted
    # ====================================================

    plt.figure(figsize=(7, 6))

    sns.scatterplot(
        x=y_test,
        y=predictions,
        alpha=0.55,
    )

    limit = max(
        float(y_test.max()),
        float(predictions.max()),
    )

    plt.plot(
        [0, limit],
        [0, limit],
        "--",
        color="red",
        label="Prediksi Sempurna",
    )

    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title("Actual vs Predicted Price")
    plt.legend()

    plt.tight_layout()

    plt.savefig(
        REPORTS_DIR / "06_actual_vs_predicted.png",
        dpi=180,
    )

    plt.close()

    # ====================================================
    # Residual Plot
    # ====================================================

    residual = y_test.to_numpy() - predictions

    plt.figure(figsize=(8, 5))

    sns.scatterplot(
        x=predictions,
        y=residual,
        alpha=0.55,
    )

    plt.axhline(
        0,
        linestyle="--",
        color="red",
    )

    plt.xlabel("Predicted Price")
    plt.ylabel("Residual")
    plt.title("Residual Plot")

    plt.tight_layout()

    plt.savefig(
        REPORTS_DIR / "07_residual_plot.png",
        dpi=180,
    )

    plt.close()

    # ====================================================
    # Worst Prediction
    # ====================================================

    errors = test.copy()

    errors["actual_price"] = y_test.to_numpy()
    errors["predicted_price"] = predictions
    errors["absolute_error"] = abs_error

    errors.nlargest(
        5,
        "absolute_error",
    ).to_csv(
        REPORTS_DIR / "worst_5_errors.csv",
        index=False,
    )

    metadata = json.loads(
        METADATA_PATH.read_text(
            encoding="utf-8"
        )
    )

    metadata["test_evaluated"] = True
    metadata["test_metrics"] = metrics

    METADATA_PATH.write_text(
        json.dumps(
            metadata,
            indent=2,
            allow_nan=True,
        ),
        encoding="utf-8",
    )

    print("\n===== HASIL EVALUASI =====")
    print(f"MAE  : £{mae:,.2f}")
    print(f"RMSE : £{rmse:,.2f}")
    print(f"R²   : {r2:.4f}")
    print(f"P90 Absolute Error : {p90_error:,.2f}")
    print(f"\nSeluruh hasil disimpan di:\n{REPORTS_DIR}")


if __name__ == "__main__":
    main()