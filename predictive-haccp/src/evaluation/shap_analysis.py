"""
src/evaluation/shap_analysis.py
---------------------------------
SHAP explainability for the Random Forest fruit-spoilage model.

Produces:
  1. Global feature importance (mean |SHAP| values) — bar plot
  2. SHAP summary plot (beeswarm)
  3. SHAP dependence plots for top 3 features
  4. Local SHAP waterfall explanation for 3 selected 'Bad' predictions
  5. Interpretation table saved to reports/tables/shap_interpretation.csv

LSTM explainability:
  - Uses permutation importance (feature-level, on aggregated test sequences)
  - Avoids unstable DeepExplainer in favour of reliable permutation method
  - Results saved to reports/tables/uci_lstm_permutation_importance.csv

Run:
    python3 src/evaluation/shap_analysis.py   (from project root)
"""

import sys
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR    = PROJECT_ROOT / "models"
REPORTS_DIR   = PROJECT_ROOT / "reports" / "tables"
SHAP_DIR      = PROJECT_ROOT / "reports" / "figures" / "shap"
RANDOM_SEED   = 42

CATEGORICAL_FEATURES = ["fruit"]
NUMERIC_FEATURES     = ["temperature", "humidity", "light", "co2",
                         "temp_humidity_interaction", "co2_light_interaction"]
ALL_FEATURES         = CATEGORICAL_FEATURES + NUMERIC_FEATURES
TARGET               = "target"


def make_test_split(df):
    X = df[ALL_FEATURES]
    y = df[TARGET]
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=RANDOM_SEED
    )
    _, X_test, _, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=RANDOM_SEED
    )
    return X_train, X_test, y_train, y_test


def get_feature_names(pipeline):
    """Extract all feature names after OHE expansion."""
    ohe = pipeline.named_steps["preprocessor"].transformers_[0][1]
    cat_names = list(ohe.get_feature_names_out(CATEGORICAL_FEATURES))
    return cat_names + NUMERIC_FEATURES


def run_random_forest_shap(pipeline, X_train, X_test, y_test):
    """Run SHAP TreeExplainer on the RF pipeline."""
    print("\n── Random Forest SHAP Analysis ─────────────────────────────")

    # Transform data through the preprocessor
    preprocessor = pipeline.named_steps["preprocessor"]
    rf           = pipeline.named_steps["classifier"]
    feature_names = get_feature_names(pipeline)

    X_train_t = preprocessor.transform(X_train)
    X_test_t  = preprocessor.transform(X_test)

    # Use a background sample for efficiency (200 rows)
    bg_idx = np.random.default_rng(RANDOM_SEED).choice(
        len(X_train_t), size=min(200, len(X_train_t)), replace=False
    )
    background = X_train_t[bg_idx]

    explainer   = shap.TreeExplainer(rf, data=background, feature_perturbation="interventional")
    shap_values = explainer.shap_values(X_test_t)

    # shap_values may be (n, f, 2) or a list — handle both
    if isinstance(shap_values, list):
        sv_bad = shap_values[1]   # class 1 = Bad
    elif shap_values.ndim == 3:
        sv_bad = shap_values[:, :, 1]
    else:
        sv_bad = shap_values

    SHAP_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Global mean |SHAP| bar chart ──────────────────────────────────────
    mean_abs = np.abs(sv_bad).mean(axis=0)
    idx_sort = np.argsort(mean_abs)[::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh([feature_names[i] for i in reversed(idx_sort)],
            [mean_abs[i] for i in reversed(idx_sort)],
            color="#7C3AED")
    ax.set_xlabel("Mean |SHAP value|", fontsize=10)
    ax.set_title("Random Forest SHAP — Global Feature Importance\n(Fruit Spoilage Dataset)",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    fig.savefig(SHAP_DIR / "rf_shap_global_importance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ Global SHAP importance plot saved")

    # ── 2. SHAP summary (beeswarm) ────────────────────────────────────────────
    shap.summary_plot(
        sv_bad, X_test_t,
        feature_names=feature_names,
        show=False, max_display=10,
    )
    plt.title("SHAP Summary Plot — Random Forest (Bad class)", fontweight="bold")
    plt.tight_layout()
    plt.savefig(SHAP_DIR / "rf_shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ SHAP summary (beeswarm) plot saved")

    # ── 3. Dependence plots — top 3 features ─────────────────────────────────
    top3_idx = idx_sort[:3]
    for fi in top3_idx:
        fname = feature_names[fi].replace("/", "_").replace(" ", "_")
        shap.dependence_plot(
            fi, sv_bad, X_test_t,
            feature_names=feature_names,
            show=False, alpha=0.5,
        )
        plt.title(f"SHAP Dependence — {feature_names[fi]}", fontweight="bold")
        plt.tight_layout()
        plt.savefig(SHAP_DIR / f"rf_shap_dependence_{fname}.png",
                    dpi=150, bbox_inches="tight")
        plt.close()
    print(f"  ✓ Dependence plots saved for top 3 features: "
          f"{[feature_names[i] for i in top3_idx]}")

    # ── 4. Local waterfall — 3 high-confidence Bad predictions ───────────────
    bad_indices = np.where(y_test.values == 1)[0]
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    top_bad = bad_indices[np.argsort(y_prob[bad_indices])[-3:]]

    for rank, idx in enumerate(top_bad):
        exp = shap.Explanation(
            values=sv_bad[idx],
            base_values=explainer.expected_value[1]
            if isinstance(explainer.expected_value, (list, np.ndarray))
            else explainer.expected_value,
            data=X_test_t[idx],
            feature_names=feature_names,
        )
        shap.waterfall_plot(exp, show=False, max_display=10)
        plt.title(f"SHAP Local Explanation — Bad sample #{rank+1}", fontweight="bold")
        plt.tight_layout()
        plt.savefig(SHAP_DIR / f"rf_shap_waterfall_bad_{rank+1}.png",
                    dpi=150, bbox_inches="tight")
        plt.close()
    print("  ✓ Local waterfall plots saved for 3 high-risk Bad samples")

    # ── 5. Interpretation table ───────────────────────────────────────────────
    interp_df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs,
        "rank": pd.Series(mean_abs).rank(ascending=False).astype(int).values,
    }).sort_values("rank")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    interp_df.to_csv(REPORTS_DIR / "shap_interpretation.csv", index=False)
    print("  ✓ Interpretation table saved")
    print(interp_df[["rank", "feature", "mean_abs_shap"]].to_string(index=False))

    return interp_df


def run_permutation_importance_lstm():
    """
    Permutation importance for the LSTM model on UCI test2.
    More reliable than SHAP DeepExplainer for this use case.
    """
    from sklearn.inspection import permutation_importance
    from sklearn.base import BaseEstimator, ClassifierMixin

    lstm_model_path = MODELS_DIR / "uci_lstm_best.keras"
    scaler_path     = MODELS_DIR / "uci_lstm_scaler.joblib"

    if not lstm_model_path.exists():
        print("\n── LSTM permutation importance ──────────────────────────────")
        print("  ⚠  LSTM model not found. Run train_lstm.py first.")
        return None

    print("\n── LSTM Permutation Importance (UCI Occupancy Benchmark) ────")

    import tensorflow as tf
    from tensorflow import keras

    model  = keras.models.load_model(str(lstm_model_path))
    scaler = joblib.load(scaler_path)

    PROCESSED_UCI = PROJECT_ROOT / "data" / "processed" / "uci_occupancy"
    test2 = pd.read_csv(PROCESSED_UCI / "uci_test2_engineered.csv",
                        parse_dates=["timestamp"])

    from src.models.train_lstm import (
        LSTM_FEATURE_COLS, TARGET_COL, create_windows, get_available_features
    )
    feature_cols = get_available_features(test2)
    test2_sc     = scaler.transform(test2[feature_cols])
    test2_sc_df  = pd.DataFrame(test2_sc, columns=feature_cols)
    test2_sc_df[TARGET_COL] = test2[TARGET_COL].values

    # Determine seq_len from model input shape
    seq_len = model.input_shape[1]
    X_test, y_test = create_windows(test2_sc_df, feature_cols, seq_len)

    # Wrap Keras model in an sklearn-compatible wrapper for permutation_importance
    class KerasWrapper(BaseEstimator, ClassifierMixin):
        def __init__(self, keras_model):
            self.keras_model = keras_model
        def fit(self, X, y):
            return self
        def predict(self, X):
            return (self.keras_model.predict(X, verbose=0).flatten() >= 0.5).astype(int)
        def score(self, X, y):
            from sklearn.metrics import recall_score
            return recall_score(y, self.predict(X), zero_division=0)

    wrapper = KerasWrapper(model)
    print(f"  Running permutation importance (n_repeats=5) ...")
    result = permutation_importance(
        wrapper, X_test, y_test,
        n_repeats=5, random_state=RANDOM_SEED,
        scoring="recall",
    )

    perm_df = pd.DataFrame({
        "feature":       feature_cols,
        "mean_decrease_recall": result.importances_mean,
        "std":           result.importances_std,
    }).sort_values("mean_decrease_recall", ascending=False)

    perm_df.to_csv(REPORTS_DIR / "uci_lstm_permutation_importance.csv", index=False)
    print("  ✓ LSTM permutation importance saved")
    print(perm_df.head(10).to_string(index=False))

    # Plot
    FIGURES_DIR = PROJECT_ROOT / "reports" / "figures" / "uci_occupancy"
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(perm_df["feature"][::-1], perm_df["mean_decrease_recall"][::-1],
            color="#0891B2")
    ax.set_xlabel("Mean Decrease in Recall (permutation)", fontsize=10)
    ax.set_title("LSTM Permutation Importance — UCI Occupancy Benchmark",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    fig.savefig(SHAP_DIR / "lstm_permutation_importance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ Plot saved")

    return perm_df


def run_shap_analysis(save: bool = True):
    print(f"\n{'='*60}")
    print("EXPLAINABILITY ANALYSIS")
    print(f"{'='*60}")

    # ── Load RF model ─────────────────────────────────────────────────────────
    rf_path = MODELS_DIR / "fruit_random_forest.joblib"
    if not rf_path.exists():
        print(f"ERROR: RF model not found at {rf_path}. Run train_random_forest.py first.")
        return

    pipeline = joblib.load(rf_path)
    df       = pd.read_csv(PROCESSED_DIR / "fruit_spoilage_engineered.csv")
    X_train, X_test, y_train, y_test = make_test_split(df)

    # ── SHAP for RF ───────────────────────────────────────────────────────────
    interp_df = run_random_forest_shap(pipeline, X_train, X_test, y_test)

    # ── Permutation importance for LSTM ───────────────────────────────────────
    run_permutation_importance_lstm()

    print(f"\n✓ Explainability analysis complete.")
    print(f"  Figures saved to: {SHAP_DIR}")
    print(f"  Tables saved to : {REPORTS_DIR}")


if __name__ == "__main__":
    run_shap_analysis()
