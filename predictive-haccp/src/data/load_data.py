"""
src/data/load_data.py
---------------------
Utility functions to load the raw datasets.
All functions return a DataFrame. Raw files are never modified.
"""

import pandas as pd
from pathlib import Path

# ── Project root (two levels up from this file) ─────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_FRUIT    = PROJECT_ROOT / "data" / "raw" / "fruit_spoilage" / "Dataset 1.csv"
RAW_UCI_DIR  = PROJECT_ROOT / "data" / "raw" / "uci_occupancy"


def load_fruit_raw() -> pd.DataFrame:
    """
    Load the raw fruit-spoilage dataset.

    Returns
    -------
    pd.DataFrame
        10,995 rows × 6 columns (Fruit, Temp, Humid (%), Light (Fux),
        CO2 (pmm), Class) — exactly as stored in the CSV.
    """
    df = pd.read_csv(RAW_FRUIT)
    print(f"[load_fruit_raw] Loaded {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"  Columns : {list(df.columns)}")
    return df


def load_uci_file(filename: str) -> pd.DataFrame:
    """
    Load one UCI occupancy file (datatraining.txt / datatest.txt /
    datatest2.txt).

    Parameters
    ----------
    filename : str
        Basename of the file inside data/raw/uci_occupancy/.

    Returns
    -------
    pd.DataFrame
        Raw file content with the index column dropped.
    """
    path = RAW_UCI_DIR / filename
    df = pd.read_csv(path)
    # The first column is a row-index artefact — drop it.
    if df.columns[0] in ("", "Unnamed: 0"):
        df = df.iloc[:, 1:]
    print(f"[load_uci_file] {filename}: {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


def load_uci_all() -> dict[str, pd.DataFrame]:
    """
    Load all three UCI files and return them as a dict keyed by split name.

    Returns
    -------
    dict with keys: 'train', 'test1', 'test2'
    """
    return {
        "train": load_uci_file("datatraining.txt"),
        "test1": load_uci_file("datatest.txt"),
        "test2": load_uci_file("datatest2.txt"),
    }
