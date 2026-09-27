import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from config import DATA_PATH, MODEL_PATH

FEATURE_COLS = ["INSTITUTE", "COURSE", "CANDIDATE CATEGORY", "ALLOTTED QUOTA", "ROUND"]
TARGET_COL = "CLOSING_RANK"


def load_and_prepare_data(data_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads processed_cutoffs.csv, consolidates candidate-level cutoffs,
    and sets up categorical dtypes for LightGBM.
    """
    csv_file = data_path / "processed_cutoffs.csv"
    if not csv_file.exists():
        raise FileNotFoundError(f"Missing {csv_file}. Please run prepare_data.py first.")

    raw_df = pd.read_csv(csv_file)

    # Consolidate so target is the true maximum closing rank for that candidate category
    grouped_df = raw_df.groupby(
        ["YEAR", "ROUND", "INSTITUTE", "COURSE", "ALLOTTED QUOTA", "CANDIDATE CATEGORY"],
        as_index=False,
    ).agg(
        CLOSING_RANK=("CLOSING_RANK", "max"),
        OPENING_RANK=("OPENING_RANK", "min"),
        SEATS_ALLOTTED=("SEATS_ALLOTTED", "sum"),
    )

    # Cast feature columns to pandas category dtype
    for col in FEATURE_COLS:
        grouped_df[col] = grouped_df[col].astype("category")

    return grouped_df, raw_df


def evaluate_temporal(df: pd.DataFrame) -> dict:
    """
    Evaluates out-of-time predictive performance: Train on 2024 -> Test on 2025.
    """
    train_df = df[df["YEAR"] == 2024]
    test_df = df[df["YEAR"] == 2025]

    X_train, y_train = train_df[FEATURE_COLS], train_df[TARGET_COL]
    X_test, y_test = test_df[FEATURE_COLS], test_df[TARGET_COL]

    model = LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
        verbose=-1,
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    r2 = r2_score(y_test, preds)
    n, p = len(y_test), len(FEATURE_COLS)
    adj_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1))
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    mae = float(mean_absolute_error(y_test, preds))

    results = {
        "train_samples_2024": int(len(train_df)),
        "test_samples_2025": int(len(test_df)),
        "temporal_r2": round(float(r2), 4),
        "temporal_adj_r2": round(float(adj_r2), 4),
        "temporal_rmse": round(rmse, 2),
        "temporal_mae": round(mae, 2),
    }
    return results


def evaluate_kfold_cv(df: pd.DataFrame, n_splits: int = 5) -> dict:
    """
    Computes 5-fold cross-validation metrics across the consolidated dataset.
    """
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    r2_scores, adj_r2_scores, rmse_scores, mae_scores = [], [], [], []

    for train_idx, val_idx in kf.split(X):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        m = LGBMRegressor(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42,
            verbose=-1,
        )
        m.fit(X_tr, y_tr)
        preds = m.predict(X_val)

        r2 = r2_score(y_val, preds)
        n, p = len(y_val), len(FEATURE_COLS)
        adj_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1))

        r2_scores.append(r2)
        adj_r2_scores.append(adj_r2)
        rmse_scores.append(np.sqrt(mean_squared_error(y_val, preds)))
        mae_scores.append(mean_absolute_error(y_val, preds))

    results = {
        "cv_folds": n_splits,
        "cv_r2_mean": round(float(np.mean(r2_scores)), 4),
        "cv_r2_std": round(float(np.std(r2_scores)), 4),
        "cv_adj_r2_mean": round(float(np.mean(adj_r2_scores)), 4),
        "cv_rmse_mean": round(float(np.mean(rmse_scores)), 2),
        "cv_mae_mean": round(float(np.mean(mae_scores)), 2),
    }
    return results


def train_production_models(df: pd.DataFrame) -> dict:
    """
    Trains production models on the complete dataset:
    - Standard regressor (expected mean cutoff)
    - Quantile α=0.5 (median realistic cutoff)
    - Quantile α=0.9 (safe cutoff upper bound ~90% confidence)
    - Quantile α=0.1 (ambitious cutoff lower bound ~10% confidence)
    """
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    print("\nTraining production models on full dataset...")

    # 1. Standard regressor
    regressor = LGBMRegressor(
        n_estimators=350,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
        verbose=-1,
    )
    regressor.fit(X, y)

    # 2. Median quantile (alpha=0.5)
    q_median = LGBMRegressor(
        objective="quantile",
        alpha=0.5,
        n_estimators=350,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
        verbose=-1,
    )
    q_median.fit(X, y)

    # 3. Safe quantile (alpha=0.9)
    q_safe = LGBMRegressor(
        objective="quantile",
        alpha=0.9,
        n_estimators=350,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
        verbose=-1,
    )
    q_safe.fit(X, y)

    # 4. Ambitious quantile (alpha=0.1)
    q_ambitious = LGBMRegressor(
        objective="quantile",
        alpha=0.1,
        n_estimators=350,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
        verbose=-1,
    )
    q_ambitious.fit(X, y)

    return {
        "regressor": regressor,
        "q_median": q_median,
        "q_safe": q_safe,
        "q_ambitious": q_ambitious,
    }


def build_seat_catalog(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts all unique seat offerings with historical statistics
    to enable fast query generation during inference.
    """
    catalog = (
        df.groupby(["INSTITUTE", "COURSE", "ALLOTTED QUOTA", "CANDIDATE CATEGORY"], as_index=False)
        .agg(
            HISTORICAL_MIN_RANK=("CLOSING_RANK", "min"),
            HISTORICAL_MAX_RANK=("CLOSING_RANK", "max"),
            HISTORICAL_AVG_RANK=("CLOSING_RANK", "mean"),
            TOTAL_SEATS_ALLOTTED=("SEATS_ALLOTTED", "sum"),
        )
    )
    return catalog


def main():
    print("=" * 60)
    print(" NEET PG SEAT PREDICTOR - LIGHTGBM TRAINING PIPELINE ")
    print("=" * 60)

    # 1. Load Data
    grouped_df, _ = load_and_prepare_data(Path(DATA_PATH))
    print(f"Loaded {len(grouped_df)} consolidated cutoff instances.")

    # 2. Temporal Out-of-Time Validation (2024 -> 2025)
    print("\n1. Running Temporal Validation (Train 2024 -> Test 2025)...")
    temporal_metrics = evaluate_temporal(grouped_df)
    print(f"   Temporal R²:       {temporal_metrics['temporal_r2']}")
    print(f"   Temporal Adj R²:   {temporal_metrics['temporal_adj_r2']}")
    print(f"   Temporal RMSE:     {temporal_metrics['temporal_rmse']}")
    print(f"   Temporal MAE:      {temporal_metrics['temporal_mae']}")

    # 3. 5-Fold Cross Validation
    print("\n2. Running 5-Fold Cross Validation on full dataset...")
    cv_metrics = evaluate_kfold_cv(grouped_df, n_splits=5)
    print(f"   Mean CV R²:        {cv_metrics['cv_r2_mean']} +/- {cv_metrics['cv_r2_std']}")
    print(f"   Mean CV Adj R²:    {cv_metrics['cv_adj_r2_mean']}")
    print(f"   Mean CV RMSE:      {cv_metrics['cv_rmse_mean']}")
    print(f"   Mean CV MAE:       {cv_metrics['cv_mae_mean']}")

    # 4. Train Final Production Models
    models_bundle = train_production_models(grouped_df)

    # 5. Build Seat Catalog for Inference
    seat_catalog = build_seat_catalog(grouped_df)

    # 6. Save Artifacts
    model_dir = Path(MODEL_PATH)
    model_dir.mkdir(parents=True, exist_ok=True)

    bundle_path = model_dir / "seat_predictor_bundle.joblib"
    joblib.dump(
        {
            "models": models_bundle,
            "feature_cols": FEATURE_COLS,
            "categorical_categories": {
                col: list(grouped_df[col].cat.categories) for col in FEATURE_COLS
            },
        },
        bundle_path,
    )
    print(f"\nSaved models bundle to: {bundle_path}")

    catalog_path = model_dir / "seat_catalog.joblib"
    joblib.dump(seat_catalog, catalog_path)
    print(f"Saved seat catalog ({len(seat_catalog)} seats) to: {catalog_path}")

    metrics_path = model_dir / "metrics.json"
    all_metrics = {**temporal_metrics, **cv_metrics}
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"Saved evaluation metrics to: {metrics_path}")

    print("\n Training Complete! Pipeline ready for inference.")


if __name__ == "__main__":
    main()
