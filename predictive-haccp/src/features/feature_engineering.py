"""
src/features/feature_engineering.py
-------------------------------------
Feature engineering for both datasets.

FRUIT SPOILAGE
  - Uses base sensor features.
  - Adds temperature × humidity and CO2 × light interaction terms.
  - Defines the FULL research feature set and the REDUCED Pi feature set.
  - One-hot encoding of fruit type is handled inside the sklearn Pipeline
    (see src/models/train_random_forest.py).

UCI OCCUPANCY (temporal)
  - Sorts by timestamp.
  - Creates lag features (1, 5, 10 rows).
  - Creates rolling statistics (mean, std, min, max) — window = 10 rows.
  - Creates rate-of-change (row-to-row delta) features.
  - Extracts hour-of-day and day-of-week from timestamp.
  - Creates LSTM sliding-window arrays.

Run directly:
    python3 -m src.features.feature_engineering   (from project root)
    python3 src/features/feature_engineering.py   (from project root)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_UCI = PROCESSED_DIR / "uci_occupancy"

# ═══════════════════════════════════════════════════════════════════════════
# PART A — FRUIT SPOILAGE FEATURES
# ═══════════════════════════════════════════════════════════════════════════

# Columns that go into the sklearn pipeline (before OHE of fruit)
FRUIT_BASE_FEATURES = ["fruit", "temperature", "humidity", "light", "co2"]

# Numeric-only base features (no fruit type)
FRUIT_NUMERIC_FEATURES = ["temperature", "humidity", "light", "co2"]

# Full research feature set (after engineering — numeric columns only,
# fruit OHE is added by the pipeline at training time)
FRUIT_FULL_FEATURES = FRUIT_NUMERIC_FEATURES + [
    "temp_humidity_interaction",
    "co2_light_interaction",
]

# Reduced feature set for Raspberry Pi deployment
# (only temperature and humidity are available from DHT22)
FRUIT_PI_FEATURES = [
    "temperature",
    "humidity",
    "temp_humidity_interaction",
]

TARGET_COL = "target"


def engineer_fruit_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add interaction features to the cleaned fruit-spoilage dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Output of clean_fruit() — must contain temperature, humidity,
        light, co2, fruit, class_label, target.

    Returns
    -------
    pd.DataFrame with added columns:
        temp_humidity_interaction, co2_light_interaction
    """
    df = df.copy()
    df["temp_humidity_interaction"] = df["temperature"] * df["humidity"]
    df["co2_light_interaction"]     = df["co2"] * df["light"]
    print(f"[fruit] Engineered features added. Final shape: {df.shape}")
    print(f"  Full research features : {FRUIT_FULL_FEATURES}")
    print(f"  Pi reduced features    : {FRUIT_PI_FEATURES}")
    return df


# ═══════════════════════════════════════════════════════════════════════════
# PART B — UCI OCCUPANCY TEMPORAL FEATURES
# ═══════════════════════════════════════════════════════════════════════════

UCI_SENSOR_COLS = ["temperature", "humidity", "light", "co2", "humidity_ratio"]
UCI_TARGET_COL  = "occupancy"
LAG_STEPS       = [1, 5, 10]
ROLLING_WINDOW  = 10   # rows ≈ 10 minutes at 60-second sampling


def engineer_uci_features(df: pd.DataFrame, split_name: str = "") -> pd.DataFrame:
    """
    Build temporal features for one UCI occupancy split.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned UCI split — must be sorted by timestamp.
    split_name : str
        Optional label for logging.

    Returns
    -------
    pd.DataFrame with lag, rolling, rate-of-change and time features.
    NaN rows introduced by lagging / rolling are dropped.
    """
    df = df.copy().sort_values("timestamp").reset_index(drop=True)
    label = f"[uci/{split_name}]" if split_name else "[uci]"

    # ── Lag features ─────────────────────────────────────────────────────────
    for col in ["temperature", "humidity", "co2", "light"]:
        for lag in LAG_STEPS:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)

    # ── Rolling statistics ────────────────────────────────────────────────────
    for col in ["temperature", "humidity", "co2", "light"]:
        df[f"{col}_roll_mean"] = (
            df[col].rolling(window=ROLLING_WINDOW, min_periods=1).mean()
        )
        df[f"{col}_roll_std"] = (
            df[col].rolling(window=ROLLING_WINDOW, min_periods=1).std()
        )
        df[f"{col}_roll_min"] = (
            df[col].rolling(window=ROLLING_WINDOW, min_periods=1).min()
        )
        df[f"{col}_roll_max"] = (
            df[col].rolling(window=ROLLING_WINDOW, min_periods=1).max()
        )

    # ── Rate-of-change features ───────────────────────────────────────────────
    for col in ["temperature", "humidity", "co2", "light"]:
        df[f"{col}_delta"] = df[col].diff()

    # ── Time features ─────────────────────────────────────────────────────────
    df["hour_of_day"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek

    # Drop rows that have NaN from lagging
    before = len(df)
    df = df.dropna().reset_index(drop=True)
    dropped = before - len(df)

    print(f"{label} Shape after feature engineering: {df.shape}  "
          f"(dropped {dropped} NaN rows from lagging)")
    return df


# ═══════════════════════════════════════════════════════════════════════════
# PART C — LSTM SLIDING WINDOWS
# ═══════════════════════════════════════════════════════════════════════════

# Features used as LSTM input (excluding timestamp, occupancy, derived time cols)
LSTM_FEATURE_COLS = [
    "temperature", "humidity", "light", "co2", "humidity_ratio",
    "temperature_delta", "humidity_delta", "co2_delta", "light_delta",
    "temperature_roll_mean", "temperature_roll_std",
    "humidity_roll_mean", "humidity_roll_std",
    "co2_roll_mean", "light_roll_mean",
    "hour_of_day",
]


def create_lstm_windows(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    sequence_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create sliding-window (X, y) arrays for LSTM training.

    For each position i, X[i] contains rows [i : i+sequence_length] of
    feature_cols, and y[i] is the target at row i+sequence_length.

    Parameters
    ----------
    df             : DataFrame sorted chronologically with NaN rows removed.
    feature_cols   : List of feature column names.
    target_col     : Name of the target/label column.
    sequence_length: Number of time steps per window.

    Returns
    -------
    X : np.ndarray  shape (n_samples, sequence_length, n_features)
    y : np.ndarray  shape (n_samples,)
    """
    X_list, y_list = [], []
    features = df[feature_cols].values
    targets  = df[target_col].values

    for i in range(len(df) - sequence_length):
        X_list.append(features[i : i + sequence_length])
        y_list.append(targets[i + sequence_length])

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    print(f"  LSTM windows: X={X.shape}  y={y.shape}  "
          f"(seq_len={sequence_length})")
    return X, y


# ═══════════════════════════════════════════════════════════════════════════
# MAIN — run feature engineering on both datasets and save
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print(f"\n{'='*60}")
    print("FEATURE ENGINEERING")
    print(f"{'='*60}")

    # ── Fruit ────────────────────────────────────────────────────────────────
    print("\n[A] FRUIT SPOILAGE")
    fruit_path = PROCESSED_DIR / "fruit_spoilage_clean.csv"
    df_fruit = pd.read_csv(fruit_path)
    df_fruit_eng = engineer_fruit_features(df_fruit)
    out_fruit = PROCESSED_DIR / "fruit_spoilage_engineered.csv"
    df_fruit_eng.to_csv(out_fruit, index=False)
    print(f"✓ Saved: {out_fruit}")

    # ── UCI — all three splits ────────────────────────────────────────────────
    print("\n[B] UCI OCCUPANCY — TEMPORAL FEATURES")
    uci_files = {
        "train": "uci_train_clean.csv",
        "test1": "uci_test1_clean.csv",
        "test2": "uci_test2_clean.csv",
    }
    uci_eng_splits = {}
    for split_name, fname in uci_files.items():
        df_raw = pd.read_csv(PROCESSED_UCI / fname, parse_dates=["timestamp"])
        df_eng = engineer_uci_features(df_raw, split_name)
        out_path = PROCESSED_UCI / fname.replace("_clean", "_engineered")
        df_eng.to_csv(out_path, index=False)
        print(f"✓ Saved {split_name}: {out_path}")
        uci_eng_splits[split_name] = df_eng

    # ── LSTM window preview (train split, sequence_length=10) ─────────────────
    print("\n[C] LSTM SLIDING WINDOW PREVIEW (train, seq_len=10)")
    df_train = uci_eng_splits["train"]
    available_lstm_cols = [c for c in LSTM_FEATURE_COLS if c in df_train.columns]
    print(f"  LSTM feature columns ({len(available_lstm_cols)}): {available_lstm_cols}")
    X10, y10 = create_lstm_windows(df_train, available_lstm_cols, UCI_TARGET_COL, 10)
    print(f"  X shape: {X10.shape}  y shape: {y10.shape}")

    print("\n✓ Feature engineering complete.")


if __name__ == "__main__":
    main()
