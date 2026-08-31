"""
src/models/train_baselines.py
-------------------------------
Baseline models for the fruit-spoilage dataset.

Baselines implemented:
  1. Majority-class (DummyClassifier strategy='most_frequent')
  2. Logistic Regression

Both are evaluated with:
  - Recall (primary metric for food-safety context)
  - Precision
  - F1-score
  - PR-AUC
  - ROC-AUC
  - Accuracy
  - Confusion matrix

Split:  70% train / 15% validation / 15% test (stratified, seed=42)

Results are saved to:
  reports/tables/fruit_baseline_metrics.csv

Run directly:
    python3 src/models/train_baselines.py   (from project root)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR   = PROJECT_ROOT / "reports" / "tables"
RANDOM_SEED   = 42

FRUIT_ENGINEERED = PROCESSED_DIR / "fruit_spoilage_engineered.csv"

# Feature sets
CATEGORICAL_FEATURES = ["fruit"]
NUMERIC_FEATURES     = ["temperature", "humidity", "light", "co2",
                         "temp_humidity_interaction", "co2_light_interaction"]
TARGET               = "target"


def make_fruit_splits(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
           pd.Series, pd.Series, pd.Series]:
    """
    Stratified 70/15/15 split.

    Returns
    -------
    X_train, X_val, X_test, y_train, y_val, y_test
    """
    X = df[CATEGORICAL_FEATURES + NUMERIC_FEATURES]
    y = df[TARGET]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=RANDOM_SEED
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=RANDOM_SEED
    )

    print(f"Train : {len(X_train):,} rows  "
          f"(Good={sum(y_train==0):,}  Bad={sum(y_train==1):,})")
    print(f"Val   : {len(X_val):,} rows  "
          f"(Good={sum(y_val==0):,}  Bad={sum(y_val==1):,})")
    print(f"Test  : {len(X_test):,} rows  "
          f"(Good={sum(y_test==0):,}  Bad={sum(y_test==1):,})")

    return X_train, X_val, X_test, y_train, y_val, y_test


def make_preprocessor() -> ColumnTransformer:
    """
    Build a ColumnTransformer:
      - OneHotEncoder for fruit type
      - StandardScaler for numeric features
    """
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             CATEGORICAL_FEATURES),
            ("num", StandardScaler(), NUMERIC_FEATURES),
        ]
    )


def evaluate(name: str, model, X_test, y_test) -> dict:
    """
    Compute all metrics on the test (or validation) set.

    Returns a dict suitable for a DataFrame row.
    """
    y_pred = model.predict(X_test)

    # Probability for AUC metrics — fall back to binary for DummyClassifier
    try:
        y_prob = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_prob)
        pr_auc  = average_precision_score(y_test, y_prob)
    except AttributeError:
        roc_auc = float("nan")
        pr_auc  = float("nan")

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)

    metrics = {
        "model":     name,
        "recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "f1":        round(f1_score(y_test, y_pred, zero_division=0), 4),
        "pr_auc":    round(pr_auc, 4) if not np.isnan(pr_auc) else "N/A",
        "roc_auc":   round(roc_auc, 4) if not np.isnan(roc_auc) else "N/A",
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }
    return metrics


def train_baselines(save: bool = True) -> pd.DataFrame:
    """
    Train and evaluate both baseline models on the fruit-spoilage dataset.

    Parameters
    ----------
    save : bool
        Save metrics table to CSV.

    Returns
    -------
    pd.DataFrame with one row per model.
    """
    print(f"\n{'='*60}")
    print("FRUIT SPOILAGE — BASELINE MODELS")
    print(f"{'='*60}")
    print(f"Random seed : {RANDOM_SEED}")

    # Load data
    df = pd.read_csv(FRUIT_ENGINEERED)
    print(f"Loaded: {df.shape}")

    # Split
    print("\n── Data Splits ─────────────────────────────────────────────")
    X_train, X_val, X_test, y_train, y_val, y_test = make_fruit_splits(df)

    results = []

    # ── 1. Majority-class baseline ────────────────────────────────────────────
    print("\n── Majority-Class Baseline ─────────────────────────────────")
    dummy = DummyClassifier(strategy="most_frequent", random_state=RANDOM_SEED)
    dummy.fit(X_train[NUMERIC_FEATURES], y_train)  # DummyClassifier ignores X
    r = evaluate("Majority-Class", dummy, X_test[NUMERIC_FEATURES], y_test)
    print(f"  Recall={r['recall']}  Precision={r['precision']}  "
          f"F1={r['f1']}  Accuracy={r['accuracy']}")
    results.append(r)

    # ── 2. Logistic Regression ────────────────────────────────────────────────
    print("\n── Logistic Regression Baseline ────────────────────────────")
    preprocessor = make_preprocessor()
    lr_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_SEED,
        )),
    ])
    lr_pipeline.fit(X_train, y_train)

    # Report validation performance
    r_val = evaluate("Logistic Regression (val)", lr_pipeline, X_val, y_val)
    print(f"  VAL  Recall={r_val['recall']}  Precision={r_val['precision']}  "
          f"F1={r_val['f1']}  ROC-AUC={r_val['roc_auc']}")

    # Report test performance (touch only once)
    r_test = evaluate("Logistic Regression", lr_pipeline, X_test, y_test)
    print(f"  TEST Recall={r_test['recall']}  Precision={r_test['precision']}  "
          f"F1={r_test['f1']}  ROC-AUC={r_test['roc_auc']}  "
          f"PR-AUC={r_test['pr_auc']}")
    results.append(r_test)

    # Confusion matrix
    y_pred_test = lr_pipeline.predict(X_test)
    cm = confusion_matrix(y_test, y_pred_test)
    print(f"\n  Confusion matrix (LR — test):\n{cm}")

    # ── Results table ─────────────────────────────────────────────────────────
    results_df = pd.DataFrame(results)
    print(f"\n── Results Summary ─────────────────────────────────────────")
    print(results_df[["model", "recall", "precision", "f1",
                       "pr_auc", "roc_auc", "accuracy"]].to_string(index=False))

    # ── Save ──────────────────────────────────────────────────────────────────
    if save:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = REPORTS_DIR / "fruit_baseline_metrics.csv"
        results_df.to_csv(out_path, index=False)
        print(f"\n✓ Saved metrics to: {out_path}")

    return results_df


if __name__ == "__main__":
    train_baselines(save=True)
