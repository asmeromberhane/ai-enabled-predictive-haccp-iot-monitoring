"""
src/data/clean_fruit.py
------------------------
Cleaning pipeline for the Fruit Spoilage dataset (Dataset 1.csv).

Steps performed (nothing is written back to raw data):
  1. Standardise column names.
  2. Merge 'BAD' label into 'Bad'.
  3. Remove exact duplicate rows and document count.
  4. Encode target: Good=0, Bad=1.
  5. Inspect class balance and fruit-type distribution.
  6. Check for impossible / suspicious sensor values.
  7. Save cleaned data to data/processed/fruit_spoilage_clean.csv.

Run directly:
    python3 -m src.data.clean_fruit          (from project root)
    python3 src/data/clean_fruit.py          (from project root)
"""

import sys
from pathlib import Path
import pandas as pd

# ── Make sure the project root is on the path when run directly ─────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.load_data import load_fruit_raw  # noqa: E402

# ── Output path ──────────────────────────────────────────────────────────────
PROCESSED_DIR  = PROJECT_ROOT / "data" / "processed"
OUTPUT_PATH    = PROCESSED_DIR / "fruit_spoilage_clean.csv"

# ── Sensor sanity ranges (based on EDA findings) ────────────────────────────
SENSOR_RANGES = {
    "temperature": (10.0, 40.0),   # °C — tighter than physically possible but generous
    "humidity":    (50.0, 100.0),  # %
    "light":       (0.0,  1000.0), # lux
    "co2":         (0.0,  2000.0), # ppm
}

# ── Column name mapping ──────────────────────────────────────────────────────
COLUMN_RENAME = {
    "Fruit":      "fruit",
    "Temp":       "temperature",
    "Humid (%)":  "humidity",
    "Light (Fux)":"light",
    "CO2 (pmm)":  "co2",
    "Class":      "class_label",
}


def clean_fruit(save: bool = True) -> pd.DataFrame:
    """
    Run the full cleaning pipeline for the fruit-spoilage dataset.

    Parameters
    ----------
    save : bool
        If True, write the cleaned DataFrame to OUTPUT_PATH.

    Returns
    -------
    pd.DataFrame
        Cleaned dataset with numeric target column `target` (0/1).
    """
    # ── 1. Load raw data ─────────────────────────────────────────────────────
    df = load_fruit_raw()
    print(f"\n{'='*60}")
    print("FRUIT SPOILAGE — CLEANING PIPELINE")
    print(f"{'='*60}")
    print(f"[1] Raw shape      : {df.shape}")

    # ── 2. Rename columns ────────────────────────────────────────────────────
    df = df.rename(columns=COLUMN_RENAME)
    print(f"[2] Columns renamed: {list(df.columns)}")

    # ── 3. Merge 'BAD' → 'Bad' ───────────────────────────────────────────────
    bad_count_before = (df["class_label"] == "BAD").sum()
    df["class_label"] = df["class_label"].replace("BAD", "Bad")
    print(f"[3] 'BAD' rows normalised to 'Bad': {bad_count_before:,}")
    print(f"    Class distribution now: {df['class_label'].value_counts().to_dict()}")

    # ── 4. Remove exact duplicates ────────────────────────────────────────────
    n_before = len(df)
    df = df.drop_duplicates()
    n_removed = n_before - len(df)
    print(f"[4] Duplicate rows removed: {n_removed:,}  "
          f"({n_before:,} → {len(df):,} rows)")

    # ── 5. Check missing values ──────────────────────────────────────────────
    missing = df.isnull().sum()
    print(f"[5] Missing values per column:")
    print(missing.to_string())

    # ── 6. Sensor sanity check ────────────────────────────────────────────────
    print("[6] Sensor sanity check (rows outside expected range):")
    for col, (lo, hi) in SENSOR_RANGES.items():
        out = df[(df[col] < lo) | (df[col] > hi)]
        print(f"    {col:15s}: {len(out):,} rows outside [{lo}, {hi}]")

    # ── 7. Encode target ──────────────────────────────────────────────────────
    label_map = {"Good": 0, "Bad": 1}
    df["target"] = df["class_label"].map(label_map)
    n_unmapped = df["target"].isnull().sum()
    if n_unmapped > 0:
        print(f"[7] WARNING: {n_unmapped} rows could not be mapped to 0/1. "
              "Check raw class labels.")
    else:
        print(f"[7] Target encoded (Good=0, Bad=1). "
              f"Balance: {df['target'].value_counts().to_dict()}")

    # ── 8. Fruit type distribution ────────────────────────────────────────────
    print(f"[8] Fruit counts:")
    print(df["fruit"].value_counts().to_string())

    # ── 9. Summary ────────────────────────────────────────────────────────────
    print(f"\nCleaned dataset shape: {df.shape}")
    print(df.dtypes.to_string())

    # ── 10. Save ─────────────────────────────────────────────────────────────
    if save:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(OUTPUT_PATH, index=False)
        print(f"\n✓ Saved cleaned dataset to:\n  {OUTPUT_PATH}")

    return df


if __name__ == "__main__":
    clean_fruit(save=True)
