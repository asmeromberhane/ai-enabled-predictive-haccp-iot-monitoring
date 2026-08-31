"""
src/models/train_lstm.py
--------------------------
LSTM temporal benchmark on the UCI Occupancy Detection dataset.

IMPORTANT ACADEMIC NOTE:
  This is a TEMPORAL IoT BENCHMARK experiment, NOT food-safety prediction.
  UCI Occupancy data contains office room sensor readings.
  Results must be labelled as 'temporal benchmark' in the dissertation.
  The purpose is to demonstrate LSTM temporal modelling capability using
  IoT sensor data that shares variables (temperature, humidity, light, CO2)
  with the food-safety context.

Pipeline:
  1. Load UCI engineered splits (train / test1 / test2).
  2. Scale features (fit on train only).
  3. Create sliding windows (sequence_length = 10 rows ≈ 10 min).
  4. Build and train LSTM with dropout + early stopping.
  5. Compare sequence lengths: 10, 30.
  6. Evaluate on test2 (chronological final period).
  7. Save model, scaler, and learning curves.

Run:
    python3 src/models/train_lstm.py   (from project root)

Note: TensorFlow is required. Install with:
    pip install tensorflow
"""

import sys
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

PROCESSED_UCI = PROJECT_ROOT / "data" / "processed" / "uci_occupancy"
REPORTS_DIR   = PROJECT_ROOT / "reports" / "tables"
FIGURES_DIR   = PROJECT_ROOT / "reports" / "figures" / "uci_occupancy"
MODELS_DIR    = PROJECT_ROOT / "models"
RANDOM_SEED   = 42

np.random.seed(RANDOM_SEED)

# Features used as LSTM input
LSTM_FEATURE_COLS = [
    "temperature", "humidity", "light", "co2", "humidity_ratio",
    "temperature_delta", "humidity_delta", "co2_delta", "light_delta",
    "temperature_roll_mean", "temperature_roll_std",
    "humidity_roll_mean", "humidity_roll_std",
    "co2_roll_mean", "light_roll_mean",
    "hour_of_day",
]
TARGET_COL = "occupancy"


def load_uci_splits():
    train = pd.read_csv(PROCESSED_UCI / "uci_train_engineered.csv",
                        parse_dates=["timestamp"])
    test1 = pd.read_csv(PROCESSED_UCI / "uci_test1_engineered.csv",
                        parse_dates=["timestamp"])
    test2 = pd.read_csv(PROCESSED_UCI / "uci_test2_engineered.csv",
                        parse_dates=["timestamp"])
    return train, test1, test2


def get_available_features(df):
    return [c for c in LSTM_FEATURE_COLS if c in df.columns]


def create_windows(df, feature_cols, seq_len):
    features = df[feature_cols].values.astype(np.float32)
    targets  = df[TARGET_COL].values.astype(np.int32)
    X, y = [], []
    for i in range(len(df) - seq_len):
        X.append(features[i : i + seq_len])
        y.append(targets[i + seq_len])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


def build_lstm(seq_len, n_features, units=64, dropout=0.3):
    """Build a two-layer LSTM model."""
    import tensorflow as tf
    tf.random.set_seed(RANDOM_SEED)
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization

    model = Sequential([
        LSTM(units, input_shape=(seq_len, n_features),
             return_sequences=True, name="lstm_1"),
        Dropout(dropout, name="dropout_1"),
        LSTM(units // 2, return_sequences=False, name="lstm_2"),
        Dropout(dropout, name="dropout_2"),
        BatchNormalization(name="batch_norm"),
        Dense(32, activation="relu", name="dense_1"),
        Dense(1, activation="sigmoid", name="output"),
    ])
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def evaluate_lstm(model, X, y):
    from sklearn.metrics import (
        recall_score, precision_score, f1_score,
        roc_auc_score, average_precision_score,
        accuracy_score, confusion_matrix,
    )
    y_prob = model.predict(X, verbose=0).flatten()
    y_pred = (y_prob >= 0.5).astype(int)
    cm     = confusion_matrix(y, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2,2) else (0,0,0,0)
    return {
        "recall":    round(recall_score(y, y_pred, zero_division=0), 4),
        "precision": round(precision_score(y, y_pred, zero_division=0), 4),
        "f1":        round(f1_score(y, y_pred, zero_division=0), 4),
        "pr_auc":    round(average_precision_score(y, y_prob), 4),
        "roc_auc":   round(roc_auc_score(y, y_prob), 4),
        "accuracy":  round(accuracy_score(y, y_pred), 4),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }


def plot_learning_curves(history, seq_len, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history.history["loss"], label="Train loss", color="#2563EB")
    axes[0].plot(history.history["val_loss"], label="Val loss",
                 color="#EF4444", linestyle="--")
    axes[0].set_title(f"Loss (seq_len={seq_len})", fontweight="bold")
    axes[0].set_xlabel("Epoch"); axes[0].legend()

    axes[1].plot(history.history["accuracy"], label="Train acc", color="#2563EB")
    axes[1].plot(history.history["val_accuracy"], label="Val acc",
                 color="#EF4444", linestyle="--")
    axes[1].set_title(f"Accuracy (seq_len={seq_len})", fontweight="bold")
    axes[1].set_xlabel("Epoch"); axes[1].legend()

    plt.suptitle("LSTM Learning Curves — UCI Occupancy Benchmark",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved: {save_path}")


def run_lstm_experiment(seq_len, X_train, y_train, X_val, y_val,
                         X_test, y_test, feature_cols):
    """Train and evaluate one LSTM for a given sequence length."""
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    import tensorflow as tf
    tf.random.set_seed(RANDOM_SEED)

    print(f"\n  seq_len={seq_len} | X_train={X_train.shape} | "
          f"X_val={X_val.shape} | X_test={X_test.shape}")

    n_features = X_train.shape[2]
    model = build_lstm(seq_len, n_features)

    # Class weights to handle imbalance
    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    class_weight = {0: 1.0, 1: n_neg / n_pos}
    print(f"  Class weights: {class_weight}")

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5,
                      restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                          patience=3, verbose=0, min_lr=1e-5),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=30,
        batch_size=128,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=0,
    )

    epochs_run = len(history.history["loss"])
    print(f"  Trained for {epochs_run} epochs (early stopping)")

    r = evaluate_lstm(model, X_test, y_test)
    print(f"  TEST → Recall={r['recall']}  Precision={r['precision']}  "
          f"F1={r['f1']}  ROC-AUC={r['roc_auc']}  PR-AUC={r['pr_auc']}")

    return model, history, r


def train_lstm(save: bool = True):
    print(f"\n{'='*60}")
    print("UCI OCCUPANCY — LSTM TEMPORAL BENCHMARK")
    print(f"{'='*60}")
    print("⚠  TEMPORAL IoT BENCHMARK — not food-safety prediction.")
    print("   Variables shared with food-safety context: temp, humidity, CO₂, light.\n")

    # ── Load data ─────────────────────────────────────────────────────────────
    train_df, test1_df, test2_df = load_uci_splits()
    feature_cols = get_available_features(train_df)
    print(f"Feature columns ({len(feature_cols)}): {feature_cols}")
    print(f"Train={len(train_df):,}  Test1={len(test1_df):,}  Test2={len(test2_df):,}")

    # ── Scale features (fit on train only) ────────────────────────────────────
    from sklearn.preprocessing import StandardScaler
    import joblib

    scaler = StandardScaler()
    train_sc = scaler.fit_transform(train_df[feature_cols])
    test1_sc = scaler.transform(test1_df[feature_cols])
    test2_sc = scaler.transform(test2_df[feature_cols])

    train_sc_df = pd.DataFrame(train_sc, columns=feature_cols)
    train_sc_df[TARGET_COL] = train_df[TARGET_COL].values

    test1_sc_df = pd.DataFrame(test1_sc, columns=feature_cols)
    test1_sc_df[TARGET_COL] = test1_df[TARGET_COL].values

    test2_sc_df = pd.DataFrame(test2_sc, columns=feature_cols)
    test2_sc_df[TARGET_COL] = test2_df[TARGET_COL].values

    # ── Results storage ───────────────────────────────────────────────────────
    all_results = []
    best_model  = None
    best_hist   = None
    best_seq    = None
    best_f1     = -1

    # ── Experiment: sequence lengths 10 and 30 ────────────────────────────────
    for seq_len in [10, 30]:
        print(f"\n── Sequence length = {seq_len} ─────────────────────────────")

        X_train, y_train = create_windows(train_sc_df, feature_cols, seq_len)
        X_val,   y_val   = create_windows(test1_sc_df, feature_cols, seq_len)
        X_test,  y_test  = create_windows(test2_sc_df, feature_cols, seq_len)

        model, history, r = run_lstm_experiment(
            seq_len, X_train, y_train, X_val, y_val, X_test, y_test, feature_cols
        )
        r["seq_len"] = seq_len
        r["model"]   = f"LSTM (seq={seq_len})"
        all_results.append(r)

        if r["f1"] > best_f1:
            best_f1    = r["f1"]
            best_model = model
            best_hist  = history
            best_seq   = seq_len

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n── Results Summary ─────────────────────────────────────────")
    results_df = pd.DataFrame(all_results)
    print(results_df[["model", "recall", "precision", "f1",
                       "pr_auc", "roc_auc", "accuracy"]].to_string(index=False))
    print(f"\nBest model: seq_len={best_seq} (F1={best_f1:.4f})")

    # ── Save ──────────────────────────────────────────────────────────────────
    if save:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)

        # Save best model
        best_model.save(str(MODELS_DIR / "uci_lstm_best.keras"))
        print(f"\n✓ LSTM model saved: {MODELS_DIR / 'uci_lstm_best.keras'}")

        # Save scaler
        joblib.dump(scaler, MODELS_DIR / "uci_lstm_scaler.joblib")
        print(f"✓ Scaler saved: {MODELS_DIR / 'uci_lstm_scaler.joblib'}")

        # Save metrics
        results_df.to_csv(REPORTS_DIR / "uci_lstm_metrics.csv", index=False)
        print(f"✓ Metrics saved: {REPORTS_DIR / 'uci_lstm_metrics.csv'}")

        # Learning curves for best model
        plot_learning_curves(
            best_hist, best_seq,
            FIGURES_DIR / f"lstm_learning_curves_seq{best_seq}.png"
        )

    print("\n✓ LSTM training complete.")
    return best_model, results_df


if __name__ == "__main__":
    train_lstm(save=True)
