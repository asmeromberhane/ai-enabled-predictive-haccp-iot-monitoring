# Predictive HACCP Coding Implementation Checklist

## Purpose

This checklist converts the implementation plan into coding tasks that can be started immediately. It reflects the confirmed dataset situation:

- `Dataset 1.csv` is the primary food-spoilage dataset for tabular classification and anomaly detection.
- `occupancy+detection/` is a timestamped IoT benchmark dataset for temporal feature engineering and LSTM.
- Raspberry Pi data is for prototype demonstration and reduced-feature deployment.
- Public datasets do not provide microbiological contamination or swab ground truth, so claims must stay focused on spoilage-related or environmental risk patterns.

---

## 0. Project Setup

- [ ] Create project root folder, for example `predictive-haccp/`.
- [ ] Create standard folders:
  - [ ] `data/raw/fruit_spoilage/`
  - [ ] `data/raw/uci_occupancy/`
  - [ ] `data/raw/raspberry_pi/`
  - [ ] `data/interim/`
  - [ ] `data/processed/`
  - [ ] `notebooks/`
  - [ ] `src/data/`
  - [ ] `src/features/`
  - [ ] `src/models/`
  - [ ] `src/evaluation/`
  - [ ] `src/deployment/`
  - [ ] `reports/figures/`
  - [ ] `reports/tables/`
  - [ ] `reports/experiment_logs/`
  - [ ] `models/`
  - [ ] `tests/`
  - [ ] `pi/`
- [ ] Copy raw datasets without editing them:
  - [ ] `Dataset 1.csv` into `data/raw/fruit_spoilage/`
  - [ ] `datatraining.txt` into `data/raw/uci_occupancy/`
  - [ ] `datatest.txt` into `data/raw/uci_occupancy/`
  - [ ] `datatest2.txt` into `data/raw/uci_occupancy/`
- [ ] Create `requirements.txt`.
- [ ] Create `README.md`.
- [ ] Create `reports/experiment_logs/experiment_log.md`.
- [ ] Create `reports/tables/data_dictionary.md`.
- [ ] Start Git version control.

---

## 1. Dataset Audit Before Modelling

### Fruit Spoilage Dataset: `Dataset 1.csv`

- [ ] Load the dataset in a dataset-audit notebook or script.
- [ ] Confirm shape: expected `10,995 rows x 6 columns`.
- [ ] Confirm columns:
  - [ ] `Fruit`
  - [ ] `Temp`
  - [ ] `Humid (%)`
  - [ ] `Light (Fux)`
  - [ ] `CO2 (pmm)`
  - [ ] `Class`
- [ ] Confirm there is no timestamp column.
- [ ] Record missing values.
- [ ] Record duplicate count: currently observed `2,880` duplicate rows.
- [ ] Investigate whether duplicates should be removed or retained.
- [ ] Standardise class labels:
  - [ ] Convert `BAD` to `Bad`.
  - [ ] Keep final target classes as `Good` and `Bad`.
- [ ] Check fruit category distribution.
- [ ] Check class balance after cleaning.
- [ ] Check numeric ranges for impossible or suspicious values.
- [ ] Save a cleaned version to `data/processed/fruit_spoilage_clean.csv`.
- [ ] Document that detection delay and LSTM are not valid on this dataset because it has no timestamps.

### UCI Occupancy Dataset

- [ ] Load all three files:
  - [ ] `datatraining.txt`
  - [ ] `datatest.txt`
  - [ ] `datatest2.txt`
- [ ] Confirm expected columns:
  - [ ] `date`
  - [ ] `Temperature`
  - [ ] `Humidity`
  - [ ] `Light`
  - [ ] `CO2`
  - [ ] `HumidityRatio`
  - [ ] `Occupancy`
- [ ] Parse `date` as timestamp.
- [ ] Confirm sampling interval is approximately 60 seconds.
- [ ] Confirm chronological ranges for each file.
- [ ] Check missing values.
- [ ] Check duplicate rows.
- [ ] Check occupancy class balance.
- [ ] Decide whether to preserve the official train/test split.
- [ ] Save cleaned versions to `data/processed/uci_occupancy/`.
- [ ] Document that this dataset is a temporal IoT benchmark, not a HACCP or food-safety dataset.

---

## 2. Cleaning and Preprocessing Code

- [ ] Create `src/data/load_data.py`.
- [ ] Create `src/data/clean_fruit.py`.
- [ ] Create `src/data/clean_occupancy.py`.
- [ ] Standardise fruit dataset column names, for example:
  - [ ] `fruit`
  - [ ] `temperature`
  - [ ] `humidity`
  - [ ] `light`
  - [ ] `co2`
  - [ ] `class_label`
- [ ] Standardise occupancy dataset column names, for example:
  - [ ] `timestamp`
  - [ ] `temperature`
  - [ ] `humidity`
  - [ ] `light`
  - [ ] `co2`
  - [ ] `humidity_ratio`
  - [ ] `occupancy`
- [ ] Encode fruit target:
  - [ ] `Good = 0`
  - [ ] `Bad = 1`
- [ ] Encode occupancy target:
  - [ ] `0 = unoccupied`
  - [ ] `1 = occupied`
- [ ] Use train-only fitted preprocessing where models require scaling.
- [ ] Save preprocessing decisions in the data dictionary.

---

## 3. Exploratory Data Analysis

### Fruit Spoilage EDA

- [ ] Plot class distribution.
- [ ] Plot fruit category distribution.
- [ ] Plot class distribution by fruit type.
- [ ] Plot temperature by class.
- [ ] Plot humidity by class.
- [ ] Plot light by class.
- [ ] Plot CO2 by class.
- [ ] Create numeric correlation heatmap.
- [ ] Create boxplots for possible outliers.
- [ ] Check whether one variable almost trivially separates `Good` and `Bad`.
- [ ] Save figures to `reports/figures/fruit_spoilage/`.
- [ ] Write short EDA findings for Chapter 4.

### UCI Occupancy EDA

- [ ] Plot temperature over time.
- [ ] Plot humidity over time.
- [ ] Plot light over time.
- [ ] Plot CO2 over time.
- [ ] Plot occupancy over time.
- [ ] Plot class balance per file.
- [ ] Create correlation heatmap.
- [ ] Inspect changes before occupancy transitions.
- [ ] Save figures to `reports/figures/uci_occupancy/`.
- [ ] Write short EDA findings explaining benchmark role only.

---

## 4. Feature Engineering

### Fruit Spoilage Features

- [ ] Use base features:
  - [ ] `fruit`
  - [ ] `temperature`
  - [ ] `humidity`
  - [ ] `light`
  - [ ] `co2`
- [ ] Add optional engineered features:
  - [ ] temperature-humidity interaction
  - [ ] CO2-light interaction
  - [ ] distance from normal temperature range, if justified
  - [ ] distance from normal humidity range, if justified
- [ ] One-hot encode fruit type inside a pipeline.
- [ ] Define full research feature set.
- [ ] Define reduced Raspberry Pi feature set using only:
  - [ ] temperature
  - [ ] humidity
  - [ ] engineered temperature/humidity features

### UCI Temporal Features

- [ ] Sort by timestamp.
- [ ] Create lag features:
  - [ ] lag 1
  - [ ] lag 5
  - [ ] lag 10
- [ ] Create rolling features:
  - [ ] rolling mean
  - [ ] rolling standard deviation
  - [ ] rolling minimum
  - [ ] rolling maximum
- [ ] Create rate-of-change features:
  - [ ] temperature change
  - [ ] humidity change
  - [ ] CO2 change
  - [ ] light change
- [ ] Create time features:
  - [ ] hour of day
  - [ ] day of week
- [ ] Create LSTM sliding windows:
  - [ ] 10-minute sequence
  - [ ] 30-minute sequence
  - [ ] 60-minute sequence, if computationally practical
- [ ] Save processed feature datasets.

---

## 5. Validation Design

### Fruit Spoilage

- [ ] Use stratified train/validation/test split.
- [ ] Keep raw test set untouched until final evaluation.
- [ ] Fit preprocessing only on training data.
- [ ] Use baseline model before advanced models.
- [ ] Do not claim temporal prediction or detection delay.

### UCI Occupancy

- [ ] Preserve chronological order.
- [ ] Prefer official split:
  - [ ] `datatraining.txt` for training
  - [ ] `datatest.txt` and `datatest2.txt` for testing/validation, with the role documented
- [ ] Avoid random cross-validation for temporal experiments.
- [ ] Fit scalers only on training data.
- [ ] Create sequence windows after splitting or with leakage-safe boundaries.

---

## 6. Baseline Models

- [ ] Implement majority-class baseline for fruit dataset.
- [ ] Implement Logistic Regression baseline for fruit dataset.
- [ ] Implement simple threshold baseline if EDA supports it.
- [ ] Implement majority-class or Logistic Regression baseline for UCI occupancy.
- [ ] Save baseline metrics before training Random Forest, Isolation Forest, or LSTM.

---

## 7. Random Forest Implementation

### Fruit Spoilage Random Forest

- [ ] Create `src/models/train_random_forest.py`.
- [ ] Build sklearn pipeline with preprocessing and model.
- [ ] Train default Random Forest.
- [ ] Tune selected hyperparameters:
  - [ ] `n_estimators`
  - [ ] `max_depth`
  - [ ] `min_samples_split`
  - [ ] `min_samples_leaf`
  - [ ] `max_features`
  - [ ] `class_weight`
- [ ] Evaluate on validation data.
- [ ] Select final model.
- [ ] Evaluate once on test data.
- [ ] Save model to `models/fruit_random_forest.joblib`.
- [ ] Save metrics to `reports/tables/fruit_random_forest_metrics.csv`.

### Metrics

- [ ] Recall
- [ ] Precision
- [ ] F1-score
- [ ] PR-AUC
- [ ] ROC-AUC
- [ ] Accuracy
- [ ] Confusion matrix

---

## 8. Isolation Forest Implementation

- [ ] Create `src/models/train_isolation_forest.py`.
- [ ] Select numeric environmental features.
- [ ] Decide whether to train only on `Good` observations or all observations.
- [ ] Document that labels are not used during Isolation Forest training.
- [ ] Tune selected parameters:
  - [ ] `n_estimators`
  - [ ] `contamination`
  - [ ] `max_samples`
  - [ ] `max_features`
- [ ] Convert anomaly scores to predicted normal/anomaly labels.
- [ ] Compare predictions with `Good`/`Bad` labels.
- [ ] Analyse false positives and false negatives.
- [ ] Save anomaly score plots.
- [ ] Save model to `models/fruit_isolation_forest.joblib`.
- [ ] Save metrics table.

---

## 9. LSTM Temporal Benchmark

- [ ] Create `src/models/train_lstm.py`.
- [ ] Use UCI occupancy dataset only unless a timestamped food-safety dataset becomes available.
- [ ] Scale numeric features using training data only.
- [ ] Build sliding windows.
- [ ] Train first LSTM model.
- [ ] Add dropout and early stopping.
- [ ] Compare sequence lengths.
- [ ] Evaluate on chronological test data.
- [ ] Save learning curves.
- [ ] Save model and scaler.
- [ ] Clearly state that this is temporal IoT benchmark modelling, not food-spoilage prediction.

---

## 10. Explainability

- [ ] Use SHAP TreeExplainer for Random Forest.
- [ ] Produce global feature importance.
- [ ] Produce SHAP summary plot.
- [ ] Produce dependence plots for key variables.
- [ ] Produce local explanations for selected `Bad` predictions.
- [ ] For LSTM, use simpler sensitivity or permutation analysis if SHAP is unstable.
- [ ] Save explainability outputs to `reports/figures/shap/`.
- [ ] Write interpretation table for dissertation.

---

## 11. Comparative Evaluation

- [ ] Compare baseline vs Random Forest on fruit dataset.
- [ ] Compare Random Forest vs Isolation Forest carefully, noting supervised vs unsupervised difference.
- [ ] Evaluate UCI LSTM separately from fruit models.
- [ ] Do not rank fruit and UCI results as one direct competition.
- [ ] Create final metrics tables:
  - [ ] fruit supervised classification
  - [ ] fruit anomaly detection
  - [ ] UCI temporal benchmark
- [ ] Write error analysis.
- [ ] Discuss practical significance, not only metric scores.

---

## 12. Raspberry Pi Prototype Code

- [ ] Create `pi/sensor_reader.py`.
- [ ] Create `pi/feature_pipeline.py`.
- [ ] Create `pi/inference.py`.
- [ ] Create `pi/azure_sender.py`, if Azure is used.
- [ ] Create `pi/config.example.json`.
- [ ] Log every reading to local CSV.
- [ ] Collect:
  - [ ] timestamp
  - [ ] refrigerator ID
  - [ ] temperature
  - [ ] humidity
  - [ ] door state
  - [ ] door-open count
  - [ ] door-open duration
- [ ] Add missing-reading handling.
- [ ] Add rolling feature calculation.
- [ ] Load reduced model for inference.
- [ ] Save risk/anomaly score with each reading.
- [ ] Test prediction latency.
- [ ] Document that Raspberry Pi deployment uses reduced features unless CO2/light sensors are added.

---

## 13. Documentation During Coding

- [ ] Keep `experiment_log.md` updated after every model run.
- [ ] Record random seeds.
- [ ] Record train/validation/test split method.
- [ ] Record hyperparameters.
- [ ] Record package versions.
- [ ] Save generated figures.
- [ ] Save model files with clear names.
- [ ] Save failed experiments and reasons.
- [ ] Save screenshots of notebooks, outputs, Raspberry Pi logs, and dashboard if used.
- [ ] Maintain AI-use declaration notes.

---

## 14. Minimum Coding Completion Criteria

The implementation is technically complete when the project has:

- [ ] Dataset audit for both datasets.
- [ ] Cleaned fruit spoilage dataset.
- [ ] Cleaned UCI occupancy dataset.
- [ ] Baseline model results.
- [ ] Random Forest model on fruit spoilage data.
- [ ] Isolation Forest model on fruit spoilage data.
- [ ] LSTM or temporal benchmark model on UCI occupancy data.
- [ ] Leakage-safe validation.
- [ ] Metrics tables.
- [ ] SHAP explanation for Random Forest.
- [ ] Clear limitation statement about no microbiological ground truth.
- [ ] Raspberry Pi logging prototype or, if hardware is unavailable, documented prototype code and test plan.
- [ ] Dissertation-ready figures and tables.

---

## 15. Recommended First Coding Order

1. Project folder structure.
2. Raw data copy.
3. Dataset audit notebook.
4. Fruit cleaning script.
5. UCI cleaning script.
6. Fruit EDA.
7. UCI EDA.
8. Baseline models.
9. Fruit Random Forest.
10. Fruit Isolation Forest.
11. UCI temporal features.
12. UCI LSTM.
13. SHAP analysis.
14. Evaluation tables.
15. Raspberry Pi prototype scripts.
16. Final documentation and dissertation figures.

