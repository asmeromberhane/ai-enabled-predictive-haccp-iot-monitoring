"""
pi/inference.py
-----------------
Real-time risk inference on the Raspberry Pi.

Loads the reduced Random Forest model (trained on temperature + humidity
features only) and scores each new reading from the CSV log.

Features used (must match training — see src/deployment/train_pi_model.py):
  - temperature
  - humidity
  - temp_humidity_interaction
  - temp_rolling_mean
  - temp_rolling_std
  - humidity_rolling_mean
  - humidity_rolling_std
  - temp_delta
  - humidity_delta
  - door_open_count
  - hour_of_day

Usage:
    python3 pi/inference.py                    # score latest row in CSV
    python3 pi/inference.py --watch            # keep scoring as new rows arrive
    python3 pi/inference.py --simulate         # simulate a reading and score it
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")

# ── Paths ─────────────────────────────────────────────────────────────────────
PI_DIR       = Path(__file__).parent
PROJECT_ROOT = PI_DIR.parent
CONFIG_PATH  = PI_DIR / "config.json"
if not CONFIG_PATH.exists():
    CONFIG_PATH = PI_DIR / "config.example.json"

with open(CONFIG_PATH) as f:
    CONFIG = json.load(f)

MODEL_PATH    = PROJECT_ROOT / CONFIG["model"]["model_path"]
CSV_PATH      = PROJECT_ROOT / CONFIG["logging"]["csv_path"]
FEATURE_COLS  = CONFIG["model"]["feature_cols"]
RISK_THRESHOLD = CONFIG["model"]["risk_threshold"]

RISK_LABELS = {
    0: "✅  LOW RISK   (Good)",
    1: "⚠️  HIGH RISK  (Bad — check refrigerator!)",
}


def load_model():
    if not MODEL_PATH.exists():
        log.error(f"Model not found at {MODEL_PATH}. "
                  "Run src/deployment/train_pi_model.py first.")
        sys.exit(1)
    artifact = joblib.load(MODEL_PATH)
    # May be saved as a dict {"model": ..., "scaler": ...} or plain pipeline
    if isinstance(artifact, dict):
        return artifact["model"], artifact.get("scaler")
    return artifact, None


def score_row(row: pd.Series, model, scaler=None) -> dict:
    """
    Score one reading row. Returns a dict with risk_score and risk_label.
    """
    X = row[FEATURE_COLS].values.reshape(1, -1).astype(float)
    if scaler is not None:
        X = scaler.transform(X)

    risk_prob  = float(model.predict_proba(X)[0, 1])
    risk_label = 1 if risk_prob >= RISK_THRESHOLD else 0

    return {
        "timestamp":   row.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        "fridge_id":   row.get("fridge_id", "unknown"),
        "temperature": row.get("temperature"),
        "humidity":    row.get("humidity"),
        "risk_prob":   round(risk_prob, 4),
        "risk_label":  risk_label,
        "risk_text":   RISK_LABELS[risk_label],
    }


def score_latest_csv(model, scaler):
    """Load CSV and score the last row for each fridge."""
    if not CSV_PATH.exists():
        log.error(f"CSV not found: {CSV_PATH}. Start sensor_reader.py first.")
        return

    df = pd.read_csv(CSV_PATH)
    if df.empty:
        log.warning("CSV is empty — no readings to score yet.")
        return

    for fridge_id, group in df.groupby("fridge_id"):
        latest = group.iloc[-1]
        missing = [c for c in FEATURE_COLS if c not in latest.index]
        if missing:
            log.warning(f"{fridge_id}: missing features {missing}. Skipping.")
            continue
        result = score_row(latest, model, scaler)
        log.info(
            f"[{result['fridge_id']}] "
            f"T={result['temperature']}°C  H={result['humidity']}%  "
            f"Risk={result['risk_prob']:.3f}  {result['risk_text']}"
        )


def simulate_and_score(model, scaler, fridge_id="fridge_a"):
    """Score a synthetic simulated reading."""
    import random
    base = 4.0 if "a" in fridge_id else 5.5
    temp = round(base + random.gauss(0, 0.5), 1)
    hum  = round(min(100, max(50, 85 + random.gauss(0, 3))), 1)
    row_data = {
        "timestamp":            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "fridge_id":            fridge_id,
        "temperature":          temp,
        "humidity":             hum,
        "temp_humidity_interaction": temp * hum,
        "temp_rolling_mean":    temp,
        "temp_rolling_std":     0.1,
        "humidity_rolling_mean":hum,
        "humidity_rolling_std": 0.5,
        "temp_delta":           0.0,
        "humidity_delta":       0.0,
        "door_open_count":      0,
        "hour_of_day":          datetime.now().hour,
    }
    row = pd.Series(row_data)
    result = score_row(row, model, scaler)
    log.info(
        f"[SIMULATED {fridge_id}] "
        f"T={temp}°C  H={hum}%  "
        f"Risk={result['risk_prob']:.3f}  {result['risk_text']}"
    )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--watch",    action="store_true",
                        help="Score every 60s as new rows arrive")
    parser.add_argument("--simulate", action="store_true",
                        help="Score a synthetic reading instead of reading CSV")
    args = parser.parse_args()

    model, scaler = load_model()
    log.info(f"Model loaded from: {MODEL_PATH}")

    if args.simulate:
        for fid in ["fridge_a", "fridge_b"]:
            simulate_and_score(model, scaler, fridge_id=fid)
        return

    if args.watch:
        log.info("Watching for new readings (Ctrl+C to stop)...")
        while True:
            score_latest_csv(model, scaler)
            time.sleep(CONFIG["logging"]["interval_seconds"])
    else:
        score_latest_csv(model, scaler)


if __name__ == "__main__":
    main()
