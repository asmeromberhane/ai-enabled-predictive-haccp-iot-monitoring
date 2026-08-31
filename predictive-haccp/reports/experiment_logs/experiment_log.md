# Experiment Log — Predictive HACCP

**Project:** AI-Based Anomaly Detection for Predictive HACCP Using IoT-Based Environmental Monitoring  
**Author:** Asmerom Berhane

---

## How to use this log

After every model run or significant experiment, add a new entry below using the template.  
Record **all** runs, including failed ones — failed runs are valuable for the dissertation.

---

## Entry Template

```
### [Date] — [Short description]

| Field | Value |
|---|---|
| Phase | e.g., Phase 4 — Data Cleaning |
| Dataset | e.g., Fruit Spoilage |
| Script | e.g., src/data/clean_fruit.py |
| Model | e.g., Random Forest |
| Random seed | e.g., 42 |
| Split | e.g., 70% train / 15% val / 15% test (stratified) |
| Key hyperparameters | e.g., n_estimators=200, max_depth=None |
| Recall | |
| Precision | |
| F1-score | |
| PR-AUC | |
| ROC-AUC | |
| Notes | e.g., Removed 2,880 duplicates before splitting |
| Output files | e.g., models/fruit_random_forest.joblib |
```

---

## Log Entries

### 2026-08-13 — Phase 0: Project setup

| Field | Value |
|---|---|
| Phase | Phase 0 — Project Setup |
| Dataset | Both |
| Script | N/A |
| Notes | Created full project folder structure. Copied raw datasets to data/raw/. Created requirements.txt, README.md, experiment_log.md, and data_dictionary.md. |
| Output files | requirements.txt, README.md, reports/experiment_logs/experiment_log.md, reports/tables/data_dictionary.md |

---

### 2026-08-13 — Phase 1: Data Cleaning

| Field | Value |
|---|---|
| Phase | Phase 1 — Data Cleaning |
| Dataset | Both |
| Script | src/data/clean_fruit.py, src/data/clean_occupancy.py |
| Notes | Fruit: removed 2,880 duplicates (10,995→8,115 rows), merged BAD→Bad, no missing values, all sensor ranges valid. UCI: 3 splits loaded, timestamps parsed, 0 missing, 0 duplicates. |
| Output files | data/processed/fruit_spoilage_clean.csv, data/processed/uci_occupancy/uci_*_clean.csv |

---

### 2026-08-13 — Phase 2: Feature Engineering

| Field | Value |
|---|---|
| Phase | Phase 2 — Feature Engineering |
| Dataset | Both |
| Script | src/features/feature_engineering.py |
| Notes | Fruit: added temp×humidity, co2×light interactions (8,115×9). UCI: lag 1/5/10, rolling mean/std/min/max (window=10), delta, hour_of_day, day_of_week — 41 features per split. LSTM windows X=(8123,10,16) for seq_len=10. |
| Output files | data/processed/fruit_spoilage_engineered.csv, data/processed/uci_occupancy/uci_*_engineered.csv |

---

### 2026-08-13 — Baseline Models

| Field | Value |
|---|---|
| Phase | Phase 3a — Baselines |
| Dataset | Fruit Spoilage |
| Script | src/models/train_baselines.py |
| Random seed | 42 |
| Split | 70/15/15 stratified |
| Majority-class Accuracy | 0.5632 (predicts Good always) |
| LR Recall | 0.9605 |
| LR Precision | 0.8617 |
| LR F1 | 0.9084 |
| LR ROC-AUC | 0.9511 |
| LR PR-AUC | 0.9012 |
| Notes | Logistic Regression with class_weight='balanced', max_iter=1000. Strong baseline. |
| Output files | reports/tables/fruit_baseline_metrics.csv |

---

### 2026-08-13 — Random Forest

| Field | Value |
|---|---|
| Phase | Phase 3b — Random Forest |
| Dataset | Fruit Spoilage |
| Script | src/models/train_random_forest.py |
| Random seed | 42 |
| Split | 70/15/15 stratified |
| Tuning method | GridSearchCV with PredefinedSplit (no leakage) |
| Best params | n_estimators=100, max_depth=None, max_features='sqrt', min_samples_leaf=1, class_weight='balanced' |
| Recall | 1.0000 |
| Precision | 0.9981 |
| F1 | 0.9991 |
| PR-AUC | 1.0000 |
| ROC-AUC | 1.0000 |
| Accuracy | 0.9992 |
| Confusion matrix | TP=532, TN=685, FP=1, FN=0 |
| Notes | Near-perfect separation. Likely reflects simulated/synthetic nature of dataset. Must be discussed critically in dissertation — not representative of real-world performance. |
| Output files | models/fruit_random_forest.joblib, reports/tables/fruit_random_forest_metrics.csv, reports/figures/fruit_spoilage/rf_* |

---

### 2026-08-13 — Isolation Forest

| Field | Value |
|---|---|
| Phase | Phase 4 — Isolation Forest |
| Dataset | Fruit Spoilage |
| Script | src/models/train_isolation_forest.py |
| Random seed | 42 |
| Training strategy | Trained on Good (normal) class only — unsupervised |
| Best params | contamination=0.48, n_estimators=100, max_samples='auto', max_features=1.0 |
| Recall | 0.9868 |
| Precision | 0.6140 |
| F1 | 0.7570 |
| PR-AUC | 0.6849 |
| ROC-AUC | 0.8175 |
| False-positive rate | 0.481 |
| Notes | High recall at cost of many false positives. Expected for unsupervised anomaly detection. Must be discussed separately from supervised classifiers. Contamination=0.48 chosen to maximise recall. |
| Output files | models/fruit_isolation_forest.joblib, reports/tables/fruit_isolation_forest_metrics.csv, reports/figures/fruit_spoilage/if_* |

---

### 2026-08-13 — SHAP Explainability (Random Forest)

| Field | Value |
|---|---|
| Phase | Phase 6 — SHAP |
| Dataset | Fruit Spoilage |
| Script | src/evaluation/shap_analysis.py |
| Method | TreeExplainer with interventional perturbation (background=200 samples) |
| Top feature | light (mean\|SHAP\|=0.130) |
| 2nd feature | temp_humidity_interaction (0.094) |
| 3rd feature | humidity (0.093) |
| Notes | Light intensity is the strongest single predictor of spoilage. The temperature-humidity interaction ranks 2nd, confirming that combined environmental stress matters more than either variable alone. |
| Output files | reports/figures/shap/rf_shap_*.png, reports/tables/shap_interpretation.csv |

---

### 2026-08-13 — Raspberry Pi Reduced Model

| Field | Value |
|---|---|
| Phase | Phase 7 — Pi Deployment |
| Dataset | Fruit Spoilage (Pi-feature subset) |
| Script | src/deployment/train_pi_model.py |
| Features | temperature, humidity, temp×humidity, rolling mean/std (temp+humidity), deltas, door_open_count, hour_of_day (11 features) |
| Recall | 0.9737 |
| F1 | 0.8862 |
| ROC-AUC | 0.9451 |
| Notes | Only 2.6% recall drop vs full RF despite losing CO2, light, and fruit type. Demonstrates feasibility of edge deployment with reduced sensors. |
| Output files | models/pi_reduced_rf.joblib, reports/tables/pi_model_metrics.csv |

---

*(Add new entries below as work progresses — especially LSTM results when TensorFlow finishes installing.)*
