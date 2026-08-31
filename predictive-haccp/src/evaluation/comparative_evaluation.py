"""
src/evaluation/comparative_evaluation.py
------------------------------------------
Produces the final comparative performance tables for the dissertation.

Tables generated:
  1. Fruit Spoilage — Supervised Classification
     (Majority-class baseline, Logistic Regression, Random Forest)

  2. Fruit Spoilage — Anomaly Detection
     (Isolation Forest vs supervised baselines — discussed separately)

  3. Reduced vs Full RF
     (Full RF, Pi-reduced RF — feature degradation analysis)

  4. UCI Temporal Benchmark (LSTM results from saved CSV)

  5. Summary table for Chapter 6

Outputs:
  - reports/tables/final_comparison_supervised.csv
  - reports/tables/final_comparison_anomaly.csv
  - reports/tables/final_comparison_reduced_vs_full.csv
  - reports/tables/final_comparison_uci_lstm.csv
  - reports/tables/final_summary_all_models.csv

Run:
    python3 src/evaluation/comparative_evaluation.py   (from project root)
"""

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

REPORTS_DIR = PROJECT_ROOT / "reports" / "tables"


def load_csv_safe(path: Path) -> pd.DataFrame | None:
    if path.exists():
        return pd.read_csv(path)
    print(f"  ⚠  Not found: {path.name} — skipping")
    return None


def select_test_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only test-set rows if a 'set' column exists."""
    if df is None:
        return None
    if "set" in df.columns:
        return df[df["set"] == "test"].copy()
    return df.copy()


def format_table(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return df[cols].reset_index(drop=True)


METRIC_COLS = ["model", "recall", "precision", "f1", "pr_auc", "roc_auc", "accuracy"]


def build_tables():
    print(f"\n{'='*60}")
    print("COMPARATIVE EVALUATION — BUILDING FINAL TABLES")
    print(f"{'='*60}")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Supervised classification (fruit) ──────────────────────────────────
    print("\n[1] Supervised Classification — Fruit Spoilage")
    bl  = select_test_rows(load_csv_safe(REPORTS_DIR / "fruit_baseline_metrics.csv"))
    rf  = select_test_rows(load_csv_safe(REPORTS_DIR / "fruit_random_forest_metrics.csv"))

    rows_sup = []
    if bl is not None:
        for _, r in bl.iterrows():
            rows_sup.append({
                "model":     r.get("model", "?"),
                "recall":    r.get("recall"),
                "precision": r.get("precision"),
                "f1":        r.get("f1"),
                "pr_auc":    r.get("pr_auc"),
                "roc_auc":   r.get("roc_auc"),
                "accuracy":  r.get("accuracy"),
            })
    if rf is not None:
        r = rf.iloc[-1]   # last row = tuned RF test result
        rows_sup.append({
            "model":     r.get("model", "Random Forest (tuned)"),
            "recall":    r.get("recall"),
            "precision": r.get("precision"),
            "f1":        r.get("f1"),
            "pr_auc":    r.get("pr_auc"),
            "roc_auc":   r.get("roc_auc"),
            "accuracy":  r.get("accuracy"),
        })

    sup_df = pd.DataFrame(rows_sup)
    sup_df.to_csv(REPORTS_DIR / "final_comparison_supervised.csv", index=False)
    print(sup_df[METRIC_COLS].to_string(index=False))

    # ── 2. Anomaly Detection — Isolation Forest ───────────────────────────────
    print("\n[2] Anomaly Detection — Fruit Spoilage (Isolation Forest)")
    print("    NOTE: Isolation Forest is unsupervised — not directly comparable")
    print("    to supervised models. Discussed separately in dissertation.")
    if_df = select_test_rows(load_csv_safe(REPORTS_DIR / "fruit_isolation_forest_metrics.csv"))
    if if_df is not None:
        anom_df = if_df[METRIC_COLS].copy()
        anom_df.to_csv(REPORTS_DIR / "final_comparison_anomaly.csv", index=False)
        print(anom_df.to_string(index=False))

    # ── 3. Reduced vs Full RF ─────────────────────────────────────────────────
    print("\n[3] Reduced vs Full RF (Raspberry Pi feature degradation)")
    pi_df = select_test_rows(load_csv_safe(REPORTS_DIR / "pi_model_metrics.csv"))
    red_rows = []
    if rf is not None:
        r = rf.iloc[-1]
        red_rows.append({
            "model":         "Full RF (all features)",
            "feature_set":   "All (temp, humidity, light, CO2, fruit type, interactions)",
            "n_features":    9,
            "recall":        r.get("recall"),
            "precision":     r.get("precision"),
            "f1":            r.get("f1"),
            "roc_auc":       r.get("roc_auc"),
            "accuracy":      r.get("accuracy"),
        })
    if pi_df is not None:
        r = pi_df.iloc[-1]
        red_rows.append({
            "model":         "Pi Reduced RF",
            "feature_set":   "Temperature + humidity + interactions + rolling stats",
            "n_features":    11,
            "recall":        r.get("recall"),
            "precision":     r.get("precision"),
            "f1":            r.get("f1"),
            "roc_auc":       r.get("roc_auc"),
            "accuracy":      r.get("accuracy"),
        })
    if red_rows:
        red_df = pd.DataFrame(red_rows)
        red_df.to_csv(REPORTS_DIR / "final_comparison_reduced_vs_full.csv", index=False)
        print(red_df[["model", "n_features", "recall", "f1", "roc_auc"]].to_string(index=False))

    # ── 4. UCI LSTM temporal benchmark ────────────────────────────────────────
    print("\n[4] UCI Temporal Benchmark — LSTM")
    print("    NOTE: UCI results are for temporal IoT modelling only.")
    print("    They are NOT food-safety results and cannot be ranked")
    print("    directly against fruit-spoilage model results.")
    lstm_df = load_csv_safe(REPORTS_DIR / "uci_lstm_metrics.csv")
    if lstm_df is not None:
        lstm_df.to_csv(REPORTS_DIR / "final_comparison_uci_lstm.csv", index=False)
        display_cols = [c for c in METRIC_COLS if c in lstm_df.columns]
        print(lstm_df[display_cols].to_string(index=False))
    else:
        print("  ⚠  LSTM results not found. Run train_lstm.py first.")

    # ── 5. Grand summary table ─────────────────────────────────────────────────
    print("\n[5] Final Summary Table — All Models")
    summary_rows = []

    # Supervised fruit
    if rows_sup:
        for r in rows_sup:
            summary_rows.append({**r, "dataset": "Fruit Spoilage",
                                  "type": "Supervised Classification"})
    # Anomaly detection
    if if_df is not None and not if_df.empty:
        r = if_df.iloc[0]
        summary_rows.append({
            "model":     r.get("model", "Isolation Forest"),
            "recall":    r.get("recall"),
            "precision": r.get("precision"),
            "f1":        r.get("f1"),
            "pr_auc":    r.get("pr_auc"),
            "roc_auc":   r.get("roc_auc"),
            "accuracy":  r.get("accuracy"),
            "dataset":   "Fruit Spoilage",
            "type":      "Unsupervised Anomaly Detection",
        })
    # Pi reduced
    if red_rows and len(red_rows) > 1:
        r = red_rows[1]
        summary_rows.append({
            "model":     r["model"],
            "recall":    r["recall"],
            "precision": r["precision"],
            "f1":        r["f1"],
            "pr_auc":    None,
            "roc_auc":   r["roc_auc"],
            "accuracy":  r["accuracy"],
            "dataset":   "Fruit Spoilage (Pi features)",
            "type":      "Supervised Classification (Reduced)",
        })
    # LSTM
    if lstm_df is not None and not lstm_df.empty:
        best = lstm_df.loc[lstm_df["f1"].idxmax()]
        summary_rows.append({
            "model":     best.get("model", "LSTM"),
            "recall":    best.get("recall"),
            "precision": best.get("precision"),
            "f1":        best.get("f1"),
            "pr_auc":    best.get("pr_auc"),
            "roc_auc":   best.get("roc_auc"),
            "accuracy":  best.get("accuracy"),
            "dataset":   "UCI Occupancy (benchmark)",
            "type":      "Temporal LSTM (benchmark only)",
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(REPORTS_DIR / "final_summary_all_models.csv", index=False)
    cols = ["model", "dataset", "type", "recall", "f1", "roc_auc"]
    available = [c for c in cols if c in summary_df.columns]
    print(summary_df[available].to_string(index=False))

    print(f"\n✓ All comparison tables saved to: {REPORTS_DIR}")


if __name__ == "__main__":
    build_tables()
