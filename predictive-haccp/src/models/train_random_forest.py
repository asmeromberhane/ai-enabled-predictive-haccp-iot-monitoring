"""
src/models/train_random_forest.py
-----------------------------------
Random Forest classifier for the Fruit Spoilage dataset.

Pipeline:
  1. Load fruit_spoilage_engineered.csv
  2. Stratified 70/15/15 split (same seed as baselines for fair comparison)
  3. Build sklearn Pipeline (OneHotEncoder + StandardScaler + RandomForest)
  4. Train a default Random Forest (baseline)
  5. GridSearchCV on validation data for hyperparameter tuning
  6. Evaluate final model ONCE on the held-out test set
  7. Save model to models/fruit_random_forest.joblib
  8. Save metrics to reports/tables/fruit_random_forest_metrics.csv
  9. Save feature importances to reports/tables/fruit_rf_feature_importance.csv

Primary metric: Recall (food-safety priority — minimise false negatives)

Run:
    python3 src/models/train_random_forest.py   (from project root)
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
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
from sklearn.model_selection import GridSearchCV, train_test_split, PredefinedSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe for scripts

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR   = PROJECT_ROOT / "reports" / "tables"
FIGURES_DIR   = PROJECT_ROOT / "reports" / "figures" / "fruit_spoilage"
MODELS_DIR    = PROJECT_ROOT / "models"
RANDOM_SEED   = 42

FRUIT_ENGINEERED = PROCESSED_DIR / "fruit_spoilage_engineered.csv"

CATEGORICAL_FEATURES = ["fruit"]
NUMERIC_FEATURES     = ["temperature", "humidity", "light", "co2",
                         "temp_humidity_interaction", "co2_light_interaction"]
ALL_FEATURES         = CATEGORICAL_FEATURES + NUMERIC_FEATURES
TARGET               = "target"


# ─────────────────────────────────────────────────────────────────────────────
def make_splits(df):
    X = df[ALL_FEATURES]
    y = df[TARGET]
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=RANDOM_SEED
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=RANDOM_SEED
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def make_preprocessor():
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
         CATEGORICAL_FEATURES),
        ("num", StandardScaler(), NUMERIC_FEATURES),
    ])


def evaluate(name, model, X, y, set_name="test") -> dict:
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
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
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


def plot_confusion_matrix(model, X_test, y_test, title, save_path):
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_estimator(
        model, X_test, y_test,
        display_labels=["Good (0)", "Bad (1)"],
        cmap="Blues", ax=ax
    )
    ax.set_title(title, fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved: {save_path}")


def plot_feature_importance(pipeline, feature_names, save_path):
    rf    = pipeline.named_steps["classifier"]
    ohe   = pipeline.named_steps["preprocessor"].transformers_[0][1]
    cat_names = list(ohe.get_feature_names_out(CATEGORICAL_FEATURES))
    all_names = cat_names + NUMERIC_FEATURES

    importances = rf.feature_importances_
    idx = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh([all_names[i] for i in reversed(idx)],
            [importances[i] for i in reversed(idx)],
            color="#2563EB")
    ax.set_xlabel("Mean Decrease in Impurity", fontsize=10)
    ax.set_title("Random Forest — Feature Importances (Fruit Spoilage)",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved: {save_path}")
    return pd.DataFrame({"feature": all_names, "importance": importances}).sort_values(
        "importance", ascending=False
    )


# ─────────────────────────────────────────────────────────────────────────────
def train_random_forest(save: bool = True):
    print(f"\n{'='*60}")
    print("FRUIT SPOILAGE — RANDOM FOREST")
    print(f"{'='*60}")

    df = pd.read_csv(FRUIT_ENGINEERED)
    print(f"Loaded: {df.shape}")

    X_train, X_val, X_test, y_train, y_val, y_test = make_splits(df)
    print(f"Train={len(X_train):,}  Val={len(X_val):,}  Test={len(X_test):,}")

    # ── Step 1: Default RF ───────────────────────────────────────────────────
    print("\n── Step 1: Default Random Forest ───────────────────────────")
    default_pipe = Pipeline([
        ("preprocessor", make_preprocessor()),
        ("classifier", RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )),
    ])
    default_pipe.fit(X_train, y_train)
    r_default_val = evaluate("RF-default", default_pipe, X_val, y_val, "val")
    print(f"  VAL  Recall={r_default_val['recall']}  F1={r_default_val['f1']}  "
          f"ROC-AUC={r_default_val['roc_auc']}")

    # ── Step 2: Hyperparameter tuning ────────────────────────────────────────
    print("\n── Step 2: GridSearchCV (val data only) ─────────────────────")
    # Combine train + val and pass PredefinedSplit so GridSearchCV
    # always trains on train and evaluates on val — no data leakage.
    X_tv = pd.concat([X_train, X_val], axis=0).reset_index(drop=True)
    y_tv = pd.concat([y_train, y_val], axis=0).reset_index(drop=True)
    split_idx = [-1] * len(X_train) + [0] * len(X_val)   # -1 = train, 0 = val
    ps = PredefinedSplit(test_fold=split_idx)

    param_grid = {
        "classifier__n_estimators":     [100, 200, 300],
        "classifier__max_depth":        [None, 10, 20],
        "classifier__min_samples_leaf": [1, 2, 4],
        "classifier__max_features":     ["sqrt", "log2"],
        "classifier__class_weight":     ["balanced", "balanced_subsample"],
    }

    gs_pipe = Pipeline([
        ("preprocessor", make_preprocessor()),
        ("classifier", RandomForestClassifier(random_state=RANDOM_SEED, n_jobs=-1)),
    ])

    grid_search = GridSearchCV(
        gs_pipe,
        param_grid,
        cv=ps,
        scoring="recall",         # primary safety metric
        refit=False,              # we'll refit manually with best params
        n_jobs=-1,
        verbose=0,
    )
    grid_search.fit(X_tv, y_tv)

    best_params = grid_search.best_params_
    best_val_recall = grid_search.best_score_
    print(f"  Best val recall : {best_val_recall:.4f}")
    print(f"  Best params     : {best_params}")

    # ── Step 3: Refit on full train set with best params ─────────────────────
    print("\n── Step 3: Final model (train set, best params) ─────────────")
    final_pipe = Pipeline([
        ("preprocessor", make_preprocessor()),
        ("classifier", RandomForestClassifier(
            n_estimators=best_params["classifier__n_estimators"],
            max_depth=best_params["classifier__max_depth"],
            min_samples_leaf=best_params["classifier__min_samples_leaf"],
            max_features=best_params["classifier__max_features"],
            class_weight=best_params["classifier__class_weight"],
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )),
    ])
    final_pipe.fit(X_train, y_train)

    # Validation check
    r_final_val = evaluate("RF-tuned", final_pipe, X_val, y_val, "val")
    print(f"  VAL  Recall={r_final_val['recall']}  F1={r_final_val['f1']}  "
          f"ROC-AUC={r_final_val['roc_auc']}")

    # ── Step 4: FINAL test evaluation (touch once) ────────────────────────────
    print("\n── Step 4: Test Set Evaluation (final — touch once) ─────────")
    r_test = evaluate("Random Forest (tuned)", final_pipe, X_test, y_test, "test")
    cm = confusion_matrix(y_test, final_pipe.predict(X_test))
    print(f"  Recall    : {r_test['recall']}")
    print(f"  Precision : {r_test['precision']}")
    print(f"  F1        : {r_test['f1']}")
    print(f"  PR-AUC    : {r_test['pr_auc']}")
    print(f"  ROC-AUC   : {r_test['roc_auc']}")
    print(f"  Accuracy  : {r_test['accuracy']}")
    print(f"  Confusion matrix:\n{cm}")
    print(f"  TP={r_test['tp']}  TN={r_test['tn']}  FP={r_test['fp']}  FN={r_test['fn']}")

    # ── Save outputs ──────────────────────────────────────────────────────────
    if save:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)

        # Model
        model_path = MODELS_DIR / "fruit_random_forest.joblib"
        joblib.dump(final_pipe, model_path)
        print(f"\n✓ Model saved: {model_path}")

        # Metrics
        metrics_df = pd.DataFrame([r_default_val, r_final_val, r_test])
        metrics_df["best_params"] = [str({}), str({}), str(best_params)]
        metrics_path = REPORTS_DIR / "fruit_random_forest_metrics.csv"
        metrics_df.to_csv(metrics_path, index=False)
        print(f"✓ Metrics saved: {metrics_path}")

        # Feature importances
        fi_df = plot_feature_importance(
            final_pipe, ALL_FEATURES,
            FIGURES_DIR / "rf_feature_importance.png"
        )
        fi_df.to_csv(REPORTS_DIR / "fruit_rf_feature_importance.csv", index=False)
        print(f"✓ Feature importances saved")

        # Confusion matrix plot
        plot_confusion_matrix(
            final_pipe, X_test, y_test,
            "Random Forest — Confusion Matrix (Test Set)",
            FIGURES_DIR / "rf_confusion_matrix.png"
        )

    return final_pipe, r_test


if __name__ == "__main__":
    train_random_forest(save=True)
