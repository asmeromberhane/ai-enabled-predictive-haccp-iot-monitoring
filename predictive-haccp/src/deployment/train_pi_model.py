"""
src/deployment/train_pi_model.py
----------------------------------
Train the REDUCED Random Forest model for Raspberry Pi deployment.

The full research RF uses temperature, humidity, light, CO2, fruit type, 
and interaction features. The Pi only has DHT22 sensors (no CO2 or light),
so a reduced model is trained using only Pi-reproducible features.

Reduced feature set:
  - temperature
  - humidity
  - temp_humidity_interaction
  - temp_rolling_mean        (approximated from rolling buffer on Pi)
  - temp_rolling_std
  - humidity_rolling_mean
  - humidity_rolling_std
  - temp_delta
  - humidity_delta
  - door_open_count          (from reed switches)
  - hour_of_day              (from Pi clock)

This model will perform lower than the full RF — that is expected and
must be discussed honestly in the dissertation (reduced vs full model).

Outputs:
  - models/pi_reduced_rf.joblib   (model + scaler dict)
  - reports/tables/pi_model_metrics.csv

Run:
    python3 src/deployment/train_pi_model.py   (from project root)
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, average_precision_score,
    confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR   = PROJECT_ROOT / "reports" / "tables"
MODELS_DIR    = PROJECT_ROOT / "models"
RANDOM_SEED   = 42

# ── Feature set: only what a DHT22 + reed-switch Pi can provide ───────────────
PI_FEATURES = [
    "temperature",
    "humidity",
    "temp_humidity_interaction",
]
# The rolling/temporal features will be approximated in the feature pipeline.
# For training we simulate them from the static dataset using rolling stats.
TARGET = "target"


def make_pi_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Simulate Pi-style features from the static fruit dataset.
    Since the fruit dataset has no timestamps, rolling features are
    approximated using positional rolling on the sorted dataframe.
    These are for illustration only — real Pi features come from sensor_reader.py.
    """
    df = df.copy().sort_values("temperature").reset_index(drop=True)

    W = 10
    df["temp_rolling_mean"]    = df["temperature"].rolling(W, min_periods=1).mean()
    df["temp_rolling_std"]     = df["temperature"].rolling(W, min_periods=1).std().fillna(0)
    df["humidity_rolling_mean"]= df["humidity"].rolling(W, min_periods=1).mean()
    df["humidity_rolling_std"] = df["humidity"].rolling(W, min_periods=1).std().fillna(0)
    df["temp_delta"]           = df["temperature"].diff().fillna(0)
    df["humidity_delta"]       = df["humidity"].diff().fillna(0)
    df["door_open_count"]      = 0   # not in static dataset — set to 0
    df["hour_of_day"]          = 12  # static dataset has no time — set to midday

    return df


PI_ALL_FEATURES = [
    "temperature", "humidity", "temp_humidity_interaction",
    "temp_rolling_mean", "temp_rolling_std",
    "humidity_rolling_mean", "humidity_rolling_std",
    "temp_delta", "humidity_delta",
    "door_open_count", "hour_of_day",
]


def evaluate(name, model, scaler, X, y, set_name) -> dict:
    X_sc   = scaler.transform(X)
    y_pred = model.predict(X_sc)
    y_prob = model.predict_proba(X_sc)[:, 1]
    cm     = confusion_matrix(y, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return {
        "model":     name,
        "set":       set_name,
        "recall":    round(recall_score(y, y_pred), 4),
        "precision": round(precision_score(y, y_pred, zero_division=0), 4),
        "f1":        round(f1_score(y, y_pred), 4),
        "pr_auc":    round(average_precision_score(y, y_prob), 4),
        "roc_auc":   round(roc_auc_score(y, y_prob), 4),
        "accuracy":  round(accuracy_score(y, y_pred), 4),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }


def train_pi_model(save: bool = True):
    print(f"\n{'='*60}")
    print("REDUCED MODEL — RASPBERRY PI DEPLOYMENT")
    print(f"{'='*60}")
    print(f"Features : {PI_ALL_FEATURES}")
    print("⚠  This model uses only Pi-reproducible features.")
    print("   Performance will be lower than the full RF — this is expected.")

    df = pd.read_csv(PROCESSED_DIR / "fruit_spoilage_engineered.csv")
    df = make_pi_features(df)

    X = df[PI_ALL_FEATURES]
    y = df[TARGET]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=RANDOM_SEED
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=RANDOM_SEED
    )
    print(f"Train={len(X_train):,}  Val={len(X_val):,}  Test={len(X_test):,}")

    # Scale (fit on train only)
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)

    # Train RF
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    rf.fit(X_train_sc, y_train)

    # Evaluate
    r_val  = evaluate("Pi-RF", rf, scaler, X_val, y_val, "val")
    r_test = evaluate("Pi-RF (reduced)", rf, scaler, X_test, y_test, "test")

    print(f"\nVAL   Recall={r_val['recall']}  F1={r_val['f1']}  "
          f"ROC-AUC={r_val['roc_auc']}")
    print(f"TEST  Recall={r_test['recall']}  F1={r_test['f1']}  "
          f"ROC-AUC={r_test['roc_auc']}")
    print(f"\n  TP={r_test['tp']}  TN={r_test['tn']}  "
          f"FP={r_test['fp']}  FN={r_test['fn']}")

    if save:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)

        joblib.dump({"model": rf, "scaler": scaler},
                    MODELS_DIR / "pi_reduced_rf.joblib")
        print(f"\n✓ Pi model saved: {MODELS_DIR / 'pi_reduced_rf.joblib'}")

        metrics_df = pd.DataFrame([r_val, r_test])
        metrics_df["note"] = "Reduced feature set for Raspberry Pi deployment"
        metrics_df.to_csv(REPORTS_DIR / "pi_model_metrics.csv", index=False)
        print(f"✓ Metrics saved: {REPORTS_DIR / 'pi_model_metrics.csv'}")

    return rf, scaler, r_test


if __name__ == "__main__":
    train_pi_model(save=True)
