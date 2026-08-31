"""
src/models/train_isolation_forest.py
--------------------------------------
Isolation Forest anomaly detection on the Fruit Spoilage dataset.

Key design decisions:
  - Isolation Forest is UNSUPERVISED — labels are NOT used during training.
  - The model is trained on numeric environmental features only.
  - Strategy: train on 'Good' (normal) observations only, then score all rows.
  - Anomaly scores are compared against known Good/Bad labels post-hoc.
  - This is clearly distinct from the supervised Random Forest.

Outputs:
  - models/fruit_isolation_forest.joblib
  - reports/tables/fruit_isolation_forest_metrics.csv
  - reports/figures/fruit_spoilage/if_anomaly_scores.png
  - reports/figures/fruit_spoilage/if_confusion_matrix.png
  - reports/figures/fruit_spoilage/if_score_distribution.png

Run:
    python3 src/models/train_isolation_forest.py   (from project root)
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR   = PROJECT_ROOT / "reports" / "tables"
FIGURES_DIR   = PROJECT_ROOT / "reports" / "figures" / "fruit_spoilage"
MODELS_DIR    = PROJECT_ROOT / "models"
RANDOM_SEED   = 42

FRUIT_ENGINEERED  = PROCESSED_DIR / "fruit_spoilage_engineered.csv"
NUMERIC_FEATURES  = ["temperature", "humidity", "light", "co2",
                      "temp_humidity_interaction", "co2_light_interaction"]
TARGET            = "target"   # 0=Good, 1=Bad  (NOT used during training)


# ─────────────────────────────────────────────────────────────────────────────
def make_splits(df):
    """Same 70/15/15 stratified split as other models for fair comparison."""
    X = df[NUMERIC_FEATURES]
    y = df[TARGET]
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=RANDOM_SEED
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=RANDOM_SEED
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def evaluate_if(name, y_true, y_pred_if, anomaly_scores, set_name="test") -> dict:
    """
    Evaluate Isolation Forest predictions against known labels.

    IsolationForest.predict() returns +1 (normal) and -1 (anomaly).
    We remap: anomaly (-1) → Bad (1),  normal (+1) → Good (0).
    anomaly_scores are the raw decision function values (lower = more anomalous).
    We negate them so higher = more anomalous (consistent with probability direction).
    """
    y_pred_binary = np.where(y_pred_if == -1, 1, 0)   # -1=anomaly → 1=Bad
    scores_pos    = -anomaly_scores                     # negate: higher = more anomalous

    cm          = confusion_matrix(y_true, y_pred_binary)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2,2) else (0,0,0,0)

    try:
        roc_auc = roc_auc_score(y_true, scores_pos)
        pr_auc  = average_precision_score(y_true, scores_pos)
    except Exception:
        roc_auc = pr_auc = float("nan")

    return {
        "model":     name,
        "set":       set_name,
        "recall":    round(recall_score(y_true, y_pred_binary, zero_division=0), 4),
        "precision": round(precision_score(y_true, y_pred_binary, zero_division=0), 4),
        "f1":        round(f1_score(y_true, y_pred_binary, zero_division=0), 4),
        "pr_auc":    round(pr_auc, 4),
        "roc_auc":   round(roc_auc, 4),
        "accuracy":  round(accuracy_score(y_true, y_pred_binary), 4),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }


def plot_score_distribution(scores_good, scores_bad, save_path):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(-scores_good, bins=50, alpha=0.6, color="#22C55E", label="Good (0)")
    ax.hist(-scores_bad,  bins=50, alpha=0.6, color="#EF4444", label="Bad (1)")
    ax.set_xlabel("Anomaly Score (higher = more anomalous)", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title("Isolation Forest — Anomaly Score Distribution by Class",
                 fontsize=11, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved: {save_path}")


def plot_cm(y_true, y_pred_if, title, save_path):
    y_pred_binary = np.where(y_pred_if == -1, 1, 0)
    cm = confusion_matrix(y_true, y_pred_binary)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=["Good (0)", "Bad (1)"])
    disp.plot(cmap="Oranges", ax=ax, colorbar=False)
    ax.set_title(title, fontsize=11, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
def train_isolation_forest(save: bool = True):
    print(f"\n{'='*60}")
    print("FRUIT SPOILAGE — ISOLATION FOREST")
    print(f"{'='*60}")
    print("⚠  This is UNSUPERVISED — labels are NOT used during training.")
    print("   Labels are used ONLY for post-hoc evaluation.")

    df = pd.read_csv(FRUIT_ENGINEERED)
    print(f"\nLoaded: {df.shape}")

    X_train, X_val, X_test, y_train, y_val, y_test = make_splits(df)
    print(f"Train={len(X_train):,}  Val={len(X_val):,}  Test={len(X_test):,}")

    # ── Scale features (fit on training data only) ────────────────────────────
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_val_sc   = scaler.transform(X_val)
    X_test_sc  = scaler.transform(X_test)

    # ── Train ONLY on 'Good' observations (normal-class strategy) ─────────────
    X_train_good = X_train_sc[y_train.values == 0]
    good_frac    = len(X_train_good) / len(X_train_sc)
    print(f"\nTraining on Good (normal) observations only: "
          f"{len(X_train_good):,} / {len(X_train_sc):,} ({good_frac:.1%})")

    # ── Hyperparameter tuning on validation data ──────────────────────────────
    # We use recall on val to choose contamination.
    # Other params (n_estimators, max_samples, max_features) are tuned similarly.
    param_grid = {
        "contamination":  [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.48],
        "n_estimators":   [100, 200],
        "max_samples":    ["auto", 256],
        "max_features":   [1.0, 0.8],
    }

    print("\n── Hyperparameter search on validation data ─────────────────")
    best_recall   = -1
    best_params   = {}
    best_if_model = None

    for cont in param_grid["contamination"]:
        for n_est in param_grid["n_estimators"]:
            for ms in param_grid["max_samples"]:
                for mf in param_grid["max_features"]:
                    clf = IsolationForest(
                        contamination=cont,
                        n_estimators=n_est,
                        max_samples=ms,
                        max_features=mf,
                        random_state=RANDOM_SEED,
                    )
                    clf.fit(X_train_good)
                    y_pred_val = clf.predict(X_val_sc)
                    y_pred_binary = np.where(y_pred_val == -1, 1, 0)
                    r = recall_score(y_val, y_pred_binary, zero_division=0)
                    if r > best_recall:
                        best_recall  = r
                        best_params  = {
                            "contamination":  cont,
                            "n_estimators":   n_est,
                            "max_samples":    ms,
                            "max_features":   mf,
                        }
                        best_if_model = clf

    print(f"  Best val recall   : {best_recall:.4f}")
    print(f"  Best params       : {best_params}")

    # ── Evaluate on validation set ────────────────────────────────────────────
    y_pred_val   = best_if_model.predict(X_val_sc)
    scores_val   = best_if_model.decision_function(X_val_sc)
    r_val = evaluate_if("IF-tuned", y_val, y_pred_val, scores_val, "val")
    print(f"\n  VAL  Recall={r_val['recall']}  Precision={r_val['precision']}  "
          f"F1={r_val['f1']}  ROC-AUC={r_val['roc_auc']}")

    # ── Final test evaluation (touch once) ────────────────────────────────────
    print("\n── Test Set Evaluation (final — touch once) ─────────────────")
    y_pred_test  = best_if_model.predict(X_test_sc)
    scores_test  = best_if_model.decision_function(X_test_sc)
    r_test = evaluate_if("Isolation Forest (tuned)", y_test,
                          y_pred_test, scores_test, "test")

    print(f"  Recall    : {r_test['recall']}")
    print(f"  Precision : {r_test['precision']}")
    print(f"  F1        : {r_test['f1']}")
    print(f"  PR-AUC    : {r_test['pr_auc']}")
    print(f"  ROC-AUC   : {r_test['roc_auc']}")
    print(f"  Accuracy  : {r_test['accuracy']}")
    print(f"  TP={r_test['tp']}  TN={r_test['tn']}  FP={r_test['fp']}  FN={r_test['fn']}")
    print(f"  False-positive rate: {r_test['fp'] / (r_test['fp'] + r_test['tn']):.4f}")

    # ── Save outputs ──────────────────────────────────────────────────────────
    if save:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)

        # Save model + scaler together as a dict
        joblib.dump({"model": best_if_model, "scaler": scaler},
                    MODELS_DIR / "fruit_isolation_forest.joblib")
        print(f"\n✓ Model saved: {MODELS_DIR / 'fruit_isolation_forest.joblib'}")

        # Metrics
        metrics_df = pd.DataFrame([r_val, r_test])
        metrics_df["best_params"] = str(best_params)
        metrics_df.to_csv(REPORTS_DIR / "fruit_isolation_forest_metrics.csv", index=False)
        print(f"✓ Metrics saved: {REPORTS_DIR / 'fruit_isolation_forest_metrics.csv'}")

        # Score distribution plot
        good_mask = y_test.values == 0
        plot_score_distribution(
            scores_test[good_mask], scores_test[~good_mask],
            FIGURES_DIR / "if_score_distribution.png"
        )

        # Confusion matrix plot
        plot_cm(y_test, y_pred_test,
                "Isolation Forest — Confusion Matrix (Test Set)",
                FIGURES_DIR / "if_confusion_matrix.png")

    print("\n✓ Isolation Forest complete.")
    return best_if_model, r_test


if __name__ == "__main__":
    train_isolation_forest(save=True)
