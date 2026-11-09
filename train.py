from __future__ import annotations

import json
import os
import random
from pathlib import Path

os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from model_utils import (
    CATEGORICAL_FEATURES,
    FEATURES,
    NUMERIC_FEATURES,
    TARGET,
    canonicalize_team_names,
)

ROOT = Path(__file__).resolve().parent
DATASET_PATH = ROOT / "dataset" / "ipl_data.csv"
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"
MODEL_PATH = MODELS_DIR / "ipl_score_model.keras"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.joblib"
METRICS_PATH = OUTPUTS_DIR / "evaluation_metrics.json"
SEED = 42


def load_and_validate_data() -> tuple[pd.DataFrame, int]:
    if not DATASET_PATH.is_file():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    frame = pd.read_csv(DATASET_PATH)
    required = {
        "mid",
        "date",
        "venue",
        "bat_team",
        "bowl_team",
        "overs",
        "runs",
        "wickets",
        "runs_last_5",
        "wickets_last_5",
        TARGET,
    }
    missing_columns = required.difference(frame.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing_columns)}")

    original_rows = len(frame)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    numeric_columns = [
        "mid",
        "overs",
        "runs",
        "wickets",
        "runs_last_5",
        "wickets_last_5",
        TARGET,
    ]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["bat_team"] = canonicalize_team_names(frame["bat_team"])
    frame["bowl_team"] = canonicalize_team_names(frame["bowl_team"])

    whole_overs = np.floor(frame["overs"])
    decimal_balls = (frame["overs"] - whole_overs) * 10
    balls_in_over = decimal_balls.round()
    valid_over_notation = (
        np.isclose(decimal_balls, balls_in_over)
        & balls_in_over.between(0, 6)
        & whole_overs.between(0, 20)
        & ~((whole_overs == 20) & (balls_in_over != 0))
    )
    frame["overs_balls"] = whole_overs * 6 + balls_in_over

    valid = (
        frame[list(required)].notna().all(axis=1)
        & valid_over_notation
        & frame["mid"].gt(0)
        & frame["runs"].between(0, 500)
        & frame["wickets"].between(0, 10)
        & frame["runs_last_5"].between(0, 500)
        & frame["wickets_last_5"].between(0, 10)
        & frame[TARGET].between(0, 500)
        & frame[TARGET].ge(frame["runs"])
        & frame["wickets_last_5"].le(frame["wickets"])
        & frame["bat_team"].ne("")
        & frame["bowl_team"].ne("")
        & frame["venue"].astype(str).str.strip().ne("")
        & frame["date"].notna()
    )
    frame = frame.loc[valid].copy()
    removed_rows = original_rows - len(frame)
    if frame.empty:
        raise ValueError("No valid dataset rows remain after validation.")

    inconsistent_totals = frame.groupby("mid")[TARGET].nunique().gt(1)
    if inconsistent_totals.any():
        bad_matches = inconsistent_totals[inconsistent_totals].index.tolist()[:10]
        raise ValueError(
            "Each match must have one final score. Inconsistent match IDs include: "
            f"{bad_matches}"
        )
    inconsistent_dates = frame.groupby("mid")["date"].nunique().gt(1)
    if inconsistent_dates.any():
        bad_matches = inconsistent_dates[inconsistent_dates].index.tolist()[:10]
        raise ValueError(f"Match IDs have inconsistent dates: {bad_matches}")
    return frame, removed_rows


def random_match_split(frame: pd.DataFrame) -> tuple[set, set, set]:
    match_ids = frame["mid"].drop_duplicates().to_numpy()
    if len(match_ids) < 3:
        raise ValueError("At least three matches are required for train/validation/test splits.")
    train_validation_ids, test_ids = train_test_split(
        match_ids, test_size=0.20, random_state=SEED, shuffle=True
    )
    train_ids, validation_ids = train_test_split(
        train_validation_ids, test_size=0.125, random_state=SEED, shuffle=True
    )
    return set(train_ids), set(validation_ids), set(test_ids)


def chronological_match_split(frame: pd.DataFrame) -> tuple[set, set, set]:
    match_dates = (
        frame.groupby("mid", as_index=False)["date"]
        .min()
        .sort_values(["date", "mid"], kind="stable")
    )
    match_ids = match_dates["mid"].tolist()
    if len(match_ids) < 3:
        raise ValueError("At least three matches are required for chronological splits.")
    train_end = max(1, int(len(match_ids) * 0.70))
    validation_end = max(train_end + 1, int(len(match_ids) * 0.80))
    validation_end = min(validation_end, len(match_ids) - 1)
    return (
        set(match_ids[:train_end]),
        set(match_ids[train_end:validation_end]),
        set(match_ids[validation_end:]),
    )


def make_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_model(input_size: int) -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(input_size,)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.15),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(0.10),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(1),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.Huber(delta=10.0),
        metrics=["mae"],
    )
    return model


def fit_and_predict(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    verbose: int = 0,
) -> tuple[tf.keras.Model, ColumnTransformer, np.ndarray, dict]:
    preprocessor = make_preprocessor()
    x_train = preprocessor.fit_transform(train[FEATURES]).astype(np.float32)
    x_validation = preprocessor.transform(validation[FEATURES]).astype(np.float32)
    x_test = preprocessor.transform(test[FEATURES]).astype(np.float32)
    y_train = train[TARGET].to_numpy(dtype=np.float32)
    y_validation = validation[TARGET].to_numpy(dtype=np.float32)

    model = build_model(x_train.shape[1])
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_validation, y_validation),
        epochs=100,
        batch_size=256,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=12, restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5
            ),
        ],
        verbose=verbose,
    )
    predictions = model.predict(x_test, verbose=0).reshape(-1)
    return model, preprocessor, predictions, history.history


def evaluate(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "r2": float(r2_score(actual, predicted)),
    }


def rows_for_ids(frame: pd.DataFrame, match_ids: set) -> pd.DataFrame:
    return frame.loc[frame["mid"].isin(match_ids)].copy()


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    tf.keras.utils.set_random_seed(SEED)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    frame, removed_rows = load_and_validate_data()
    train_ids, validation_ids, test_ids = random_match_split(frame)
    train = rows_for_ids(frame, train_ids)
    validation = rows_for_ids(frame, validation_ids)
    test = rows_for_ids(frame, test_ids)
    model, preprocessor, predictions, history = fit_and_predict(
        train, validation, test, verbose=2
    )
    y_test = test[TARGET].to_numpy(dtype=np.float32)
    baseline_predictions = np.full_like(y_test, float(train[TARGET].mean()))
    residual_quantile = float(np.quantile(np.abs(y_test - predictions), 0.90))
    neural_metrics = evaluate(y_test, predictions)
    baseline_metrics = evaluate(y_test, baseline_predictions)

    chronological_train_ids, chronological_validation_ids, chronological_test_ids = (
        chronological_match_split(frame)
    )
    chronological_train = rows_for_ids(frame, chronological_train_ids)
    chronological_validation = rows_for_ids(frame, chronological_validation_ids)
    chronological_test = rows_for_ids(frame, chronological_test_ids)
    _, _, chronological_predictions, chronological_history = fit_and_predict(
        chronological_train,
        chronological_validation,
        chronological_test,
    )
    chronological_actual = chronological_test[TARGET].to_numpy(dtype=np.float32)
    chronological_baseline = np.full_like(
        chronological_actual, float(chronological_train[TARGET].mean())
    )
    chronological_metrics = {
        "split_method": (
            "Chronological match-level: first 70% train, next 10% validation, "
            "final 20% test."
        ),
        "split_matches": {
            "train": len(chronological_train_ids),
            "validation": len(chronological_validation_ids),
            "test": len(chronological_test_ids),
        },
        "split_date_ranges": {
            "train": [
                chronological_train["date"].min().strftime("%Y-%m-%d"),
                chronological_train["date"].max().strftime("%Y-%m-%d"),
            ],
            "validation": [
                chronological_validation["date"].min().strftime("%Y-%m-%d"),
                chronological_validation["date"].max().strftime("%Y-%m-%d"),
            ],
            "test": [
                chronological_test["date"].min().strftime("%Y-%m-%d"),
                chronological_test["date"].max().strftime("%Y-%m-%d"),
            ],
        },
        "neural_network": evaluate(chronological_actual, chronological_predictions),
        "training_mean_baseline": evaluate(chronological_actual, chronological_baseline),
        "epochs_run": len(chronological_history["loss"]),
    }

    metrics = {
        "dataset_rows": int(len(frame)),
        "rows_removed_as_invalid": int(removed_rows),
        "unique_matches": int(frame["mid"].nunique()),
        "date_range": [
            frame["date"].min().strftime("%Y-%m-%d"),
            frame["date"].max().strftime("%Y-%m-%d"),
        ],
        "features": FEATURES,
        "excluded_columns": [
            "mid",
            "date",
            "batsman",
            "bowler",
            "striker",
            "non-striker",
            TARGET,
        ],
        "split_method": (
            "Random match-level split: 70% train, 10% validation, 20% test "
            "(seed 42)."
        ),
        "split_matches": {
            "train": len(train_ids),
            "validation": len(validation_ids),
            "test": len(test_ids),
        },
        "split_rows": {
            "train": len(train),
            "validation": len(validation),
            "test": len(test),
        },
        "target": TARGET,
        "baseline": {
            "description": "Training-set mean final score for every test delivery.",
            **baseline_metrics,
        },
        "neural_network": neural_metrics,
        "chronological_generalization": chronological_metrics,
        "score_range": {
            "method": "90th percentile of absolute residuals on the random held-out matches",
            "absolute_residual_quantile": residual_quantile,
            "nominal_level": 0.90,
        },
        "training": {
            "seed": SEED,
            "epochs_run": len(history["loss"]),
            "best_validation_loss": float(min(history["val_loss"])),
            "architecture": [
                "Dense(128, relu)",
                "Dropout(0.15)",
                "Dense(64, relu)",
                "Dropout(0.10)",
                "Dense(32, relu)",
                "Dense(1)",
            ],
            "loss": "Huber(delta=10)",
            "optimizer": "Adam(learning_rate=0.001)",
        },
    }

    model.save(MODEL_PATH)
    joblib.dump(
        {
            "preprocessor": preprocessor,
            "features": FEATURES,
            "teams": sorted(
                set(frame["bat_team"].unique()) | set(frame["bowl_team"].unique())
            ),
            "venues": sorted(frame["venue"].unique().tolist()),
            "residual_quantile": residual_quantile,
            "interval_level": 0.90,
        },
        PREPROCESSOR_PATH,
    )
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    comparison = test[["mid", "date", "overs", "runs", "wickets", TARGET]].copy()
    comparison["predicted_total"] = predictions
    comparison["baseline_prediction"] = baseline_predictions
    comparison.to_csv(OUTPUTS_DIR / "actual_vs_predicted.csv", index=False)

    plt.figure(figsize=(7, 7))
    plt.scatter(y_test, predictions, alpha=0.25, s=12)
    limits = [
        min(float(y_test.min()), float(predictions.min())),
        max(float(y_test.max()), float(predictions.max())),
    ]
    plt.plot(limits, limits, "r--", label="Perfect prediction")
    plt.xlabel("Actual final score")
    plt.ylabel("Predicted final score")
    plt.title("Random held-out matches: actual vs predicted")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "actual_vs_predicted.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(history["loss"], label="Training loss")
    plt.plot(history["val_loss"], label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Huber loss")
    plt.title("Training history")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "training_loss.png", dpi=150)
    plt.close()

    print(f"Dataset: {len(frame):,} valid rows across {frame['mid'].nunique()} matches")
    print(f"Invalid rows removed: {removed_rows}")
    print(f"Model: {MODEL_PATH}")
    print(f"Metrics: {METRICS_PATH}")
    print(f"Random-match test metrics: {json.dumps(neural_metrics, indent=2)}")
    print(f"Random-match mean baseline: {json.dumps(baseline_metrics, indent=2)}")
    print(
        "Chronological test metrics: "
        f"{json.dumps(chronological_metrics['neural_network'], indent=2)}"
    )
    print(
        "Chronological mean baseline: "
        f"{json.dumps(chronological_metrics['training_mean_baseline'], indent=2)}"
    )
    print(f"90% residual-based range: +/- {residual_quantile:.2f} runs")
    print(f"Random-split model epochs: {metrics['training']['epochs_run']}")
    print(f"Chronological model epochs: {chronological_metrics['epochs_run']}")


if __name__ == "__main__":
    main()
