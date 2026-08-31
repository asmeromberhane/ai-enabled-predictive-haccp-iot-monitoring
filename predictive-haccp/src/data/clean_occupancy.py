"""
src/data/clean_occupancy.py
----------------------------
Cleaning pipeline for the UCI Occupancy Detection dataset.

Steps performed:
  1. Load all three split files (train, test1, test2).
  2. Standardise column names.
  3. Parse timestamps and sort chronologically within each split.
  4. Check and report sampling-interval consistency.
  5. Check missing values and exact duplicates.
  6. Verify occupancy class balance per split.
  7. Confirm that the official train/test split is preserved.
  8. Save cleaned versions to data/processed/uci_occupancy/.

Run directly:
    python3 -m src.data.clean_occupancy     (from project root)
    python3 src/data/clean_occupancy.py     (from project root)

NOTE: This dataset is a temporal IoT benchmark, NOT a food-safety dataset.
      Results must be labelled as 'temporal benchmark' in the dissertation.
"""

import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.load_data import load_uci_all  # noqa: E402

PROCESSED_UCI = PROJECT_ROOT / "data" / "processed" / "uci_occupancy"

# Column rename map
COLUMN_RENAME = {
    "date":          "timestamp",
    "Temperature":   "temperature",
    "Humidity":      "humidity",
    "Light":         "light",
    "CO2":           "co2",
    "HumidityRatio": "humidity_ratio",
    "Occupancy":     "occupancy",
}

# Expected sampling interval in seconds
EXPECTED_INTERVAL_S = 60


def _clean_split(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    """
    Apply standard cleaning to one UCI split.

    Parameters
    ----------
    df : pd.DataFrame   Raw split dataframe
    split_name : str    Label for logging ('train', 'test1', 'test2')

    Returns
    -------
    pd.DataFrame        Cleaned split
    """
    print(f"\n  --- {split_name.upper()} ---")
    print(f"  Raw shape: {df.shape}")

    # 1. Rename columns
    df = df.rename(columns=COLUMN_RENAME)

    # 2. Parse timestamps
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    print(f"  Date range : {df['timestamp'].min()} → {df['timestamp'].max()}")

    # 3. Sampling interval
    intervals = df["timestamp"].diff().dropna().dt.total_seconds()
    median_interval = intervals.median()
    irregular = (intervals != EXPECTED_INTERVAL_S).sum()
    print(f"  Interval   : median={median_interval:.0f}s  "
          f"irregular={irregular:,} rows ({irregular/len(df)*100:.1f}%)")

    # 4. Missing values
    missing = df.isnull().sum()
    total_missing = missing.sum()
    print(f"  Missing    : {total_missing} total")
    if total_missing > 0:
        print(missing[missing > 0].to_string())

    # 5. Duplicates
    n_dup = df.duplicated().sum()
    print(f"  Duplicates : {n_dup}")
    if n_dup > 0:
        df = df.drop_duplicates()
        print(f"  → Removed {n_dup} duplicate rows")

    # 6. Occupancy balance
    vc = df["occupancy"].value_counts()
    total = len(df)
    print(f"  Occupancy  : Unoccupied={vc.get(0, 0):,} ({vc.get(0, 0)/total*100:.1f}%)  "
          f"Occupied={vc.get(1, 0):,} ({vc.get(1, 0)/total*100:.1f}%)")

    # 7. Data types
    df["occupancy"] = df["occupancy"].astype(int)
    for col in ["temperature", "humidity", "light", "co2", "humidity_ratio"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    print(f"  Clean shape: {df.shape}")
    return df


def clean_occupancy(save: bool = True) -> dict[str, pd.DataFrame]:
    """
    Run the full cleaning pipeline for all UCI occupancy splits.

    Parameters
    ----------
    save : bool
        If True, write each cleaned split to PROCESSED_UCI.

    Returns
    -------
    dict with keys 'train', 'test1', 'test2'
    """
    raw_splits = load_uci_all()

    print(f"\n{'='*60}")
    print("UCI OCCUPANCY — CLEANING PIPELINE")
    print(f"{'='*60}")
    print(
        "⚠  This is a temporal IoT benchmark dataset only.\n"
        "   It does NOT contain food-safety or HACCP data.\n"
        "   Treat LSTM results as 'temporal benchmark' in the dissertation."
    )

    clean_splits = {}
    for split_name, df_raw in raw_splits.items():
        clean_splits[split_name] = _clean_split(df_raw, split_name)

    # Overall summary
    total_rows = sum(len(v) for v in clean_splits.values())
    print(f"\nTotal clean rows across all splits: {total_rows:,}")
    print(
        "\nOfficial split usage:\n"
        "  train  → datatraining.txt  (training set)\n"
        "  test1  → datatest.txt      (test / val — earlier period)\n"
        "  test2  → datatest2.txt     (test — later period)\n"
        "  Chronological order is preserved. No random cross-validation."
    )

    # Save
    if save:
        PROCESSED_UCI.mkdir(parents=True, exist_ok=True)
        fname_map = {
            "train": "uci_train_clean.csv",
            "test1": "uci_test1_clean.csv",
            "test2": "uci_test2_clean.csv",
        }
        for split_name, df_clean in clean_splits.items():
            out = PROCESSED_UCI / fname_map[split_name]
            df_clean.to_csv(out, index=False)
            print(f"✓ Saved {split_name}: {out}")

    return clean_splits


if __name__ == "__main__":
    clean_occupancy(save=True)
