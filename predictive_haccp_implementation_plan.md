# Implementation Plan  
## Explainable AI Framework for Predictive HACCP Using IoT Environmental Monitoring

### Working project title
**AI-Based Anomaly Detection for Predictive HACCP Using IoT-Based Environmental Monitoring**

### Core project idea
The project will develop an explainable AI framework using two complementary public datasets and a Raspberry Pi edge prototype:

1. **Primary food-safety dataset**  
   *Multi-Parameter Dataset for ML-Based Fruit Spoilage Prediction in an IoT-Enabled Cold Storage System*  
   Mendeley Data, DOI: `10.17632/czz68d9fwj.1`

2. **Temporal IoT benchmark dataset**  
   *Occupancy Detection Dataset*  
   UCI Machine Learning Repository, Dataset ID `357`

3. **Deployment prototype**  
   Raspberry Pi 4 with two DHT22 temperature/humidity sensors and two reed-switch door sensors, connected to two refrigerators for live environmental monitoring.

The fruit-spoilage dataset will provide the main food-safety experiment. The UCI dataset will provide ordered time-series data for temporal modelling, lag features, rolling windows and LSTM evaluation. The Raspberry Pi prototype will demonstrate live edge deployment rather than provide the main training dataset.

---

# 1. Research Questions

## RQ1
Can environmental variables such as temperature, humidity, light and CO₂ be used to identify conditions associated with food spoilage?

## RQ2
How do Random Forest, Isolation Forest and LSTM perform when applied to environmental monitoring data?

## RQ3
Can the proposed framework support temporal learning and early-warning analysis using continuous IoT sensor data?

## RQ4
Can the selected model be deployed on a low-cost Raspberry Pi edge device for real-time environmental risk monitoring?

## RQ5
Which environmental variables contribute most strongly to model predictions, according to SHAP analysis?

---

# 2. Overall System Architecture

```text
Primary Food Dataset
(Fruit Spoilage)
        |
        v
Data Cleaning and EDA
        |
        v
Random Forest + Isolation Forest
        |
        v
Food-Safety Risk Evaluation
        |
        v
SHAP Explainability
        |
        +-------------------+
                            |
UCI Temporal Dataset       |
        |                   |
        v                   |
Lag Features + Rolling Windows
        |
        v
LSTM + Time-Based Validation
        |
        v
Temporal Generalisation Evaluation
                            |
                            v
                   Best Deployment Model
                            |
                            v
Raspberry Pi + DHT22 + Reed Switch
                            |
                            v
Live Inference + CSV + Azure IoT Hub
                            |
                            v
Dashboard / Risk Alert
```

---

# 3. Phase 0 — Project Setup and Reproducibility

## Tasks

- Create the project folder structure.
- Create a GitHub repository.
- Create a Python virtual environment.
- Install the required libraries.
- Create a requirements file.
- Create a data dictionary template.
- Create an experiment log.
- Record all AI tools used for the required AI-use declaration.

## Suggested folder structure

```text
predictive-haccp/
├── data/
│   ├── raw/
│   │   ├── fruit_spoilage/
│   │   ├── uci_occupancy/
│   │   └── raspberry_pi/
│   ├── interim/
│   └── processed/
├── notebooks/
│   ├── 01_dataset_audit.ipynb
│   ├── 02_fruit_eda.ipynb
│   ├── 03_uci_eda.ipynb
│   ├── 04_feature_engineering.ipynb
│   ├── 05_random_forest.ipynb
│   ├── 06_isolation_forest.ipynb
│   ├── 07_lstm.ipynb
│   ├── 08_evaluation.ipynb
│   └── 09_shap.ipynb
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── evaluation/
│   └── deployment/
├── pi/
│   ├── sensor_reader.py
│   ├── feature_pipeline.py
│   ├── inference.py
│   ├── azure_sender.py
│   └── config.example.json
├── models/
├── reports/
│   ├── figures/
│   ├── tables/
│   └── experiment_logs/
├── tests/
├── requirements.txt
└── README.md
```

## Suggested packages

```text
pandas
numpy
scikit-learn
tensorflow
matplotlib
shap
joblib
imbalanced-learn
scipy
statsmodels
jupyter
azure-iot-device
gpiozero
adafruit-circuitpython-dht
```

## Deliverables

- Working repository.
- Reproducible Python environment.
- Clear file structure.
- Experiment log template.
- AI-use declaration notes.

---

# 4. Phase 1 — Download and Audit Both Datasets

Do not begin modelling until the raw files have been inspected.

## 4.1 Fruit Spoilage Dataset Audit

Check:

- Number of rows and columns.
- Exact feature names.
- Whether temperature and humidity are independently measured.
- Whether the target label is genuine and how it is defined.
- Whether timestamps exist.
- Number of fruit categories.
- Missing values.
- Duplicate rows.
- Class distribution.
- Measurement ranges.
- Dataset licence.
- Collection methodology.
- Whether the data are measured, simulated or mixed.

## 4.2 UCI Occupancy Dataset Audit

Check:

- Number of files.
- Number of rows.
- Timestamp format.
- Sampling interval.
- Temperature and humidity columns.
- Other variables such as light and CO₂.
- Occupancy label distribution.
- Missing values and duplicates.
- Whether files represent different time periods.
- Whether the official split should be preserved.
- Licence and citation information.

## 4.3 Produce a dataset suitability table

| Criterion | Fruit Spoilage | UCI Occupancy |
|---|---|---|
| Food-safety relevance | Primary | Benchmark only |
| Temperature | Verify | Yes |
| Humidity | Verify | Yes |
| Target label | Verify | Occupancy |
| Timestamp | Verify | Yes |
| Sequential structure | Verify | Yes |
| Random Forest | Expected | Yes |
| Isolation Forest | Expected | Yes |
| LSTM | Only if ordered | Yes |
| Detection delay | Only if timestamped | Yes |
| Raspberry Pi feature match | Partial/strong | Strong |
| Citable and public | Verify DOI/licence | Yes |

## Decision rule

- If the fruit dataset contains timestamps, use it for all three models.
- If it lacks timestamps, use it for Random Forest and Isolation Forest, and use UCI for LSTM and temporal evaluation.
- Do not manufacture temporal order where none exists.
- Do not claim microbiological contamination unless the dataset contains microbiological ground truth.

## Deliverables

- Dataset audit notebook.
- Data dictionary.
- Dataset-selection table for Chapter 3.
- Written explanation of the role of each dataset.

---

# 5. Phase 2 — Literature Review in Parallel

The literature review should continue throughout implementation.

## Main themes

1. HACCP and Environmental Monitoring Programmes.
2. Food spoilage and cold-storage monitoring.
3. Temperature, humidity, CO₂ and light as food-quality indicators.
4. IoT monitoring in food manufacturing and cold storage.
5. Random Forest for environmental risk classification.
6. Isolation Forest for anomaly detection.
7. LSTM for time-series environmental monitoring.
8. Explainable AI and SHAP.
9. Edge AI and Raspberry Pi deployment.
10. Limitations of environmental indicators without microbiological ground truth.

## Minimum evidence table

| Paper | Domain | Dataset | Variables | Model | Validation | Main finding | Limitation | Relevance |
|---|---|---|---|---|---|---|---|---|

## Writing rule

Every literature subsection should end with:

- what previous studies achieved;
- what remains limited;
- how the current project responds to that limitation.

## Deliverables

- Literature matrix.
- Draft Chapter 2.
- Clearly stated research gap linked to the research questions.

---

# 6. Phase 3 — Define the Exact Modelling Tasks

The three models should not be treated as identical.

## 6.1 Random Forest

### Purpose
Supervised classification using the genuine target label.

### Fruit dataset
Predict the spoilage or quality label from environmental variables.

### UCI dataset
Predict occupancy only as a temporal benchmark, not as a food-safety outcome.

## 6.2 Isolation Forest

### Purpose
Unsupervised anomaly detection.

### Training strategy
Train mainly on normal or non-spoiled observations where possible.

### Evaluation
Compare anomaly scores with known labels while clearly explaining that the model does not use labels during training.

## 6.3 LSTM

### Purpose
Temporal sequence classification or prediction.

### Preferred use
Use ordered observations from the UCI dataset if the fruit dataset has no reliable timestamps.

### Input
Sliding sequences of temperature, humidity, light, CO₂ and related variables.

### Output
The target state at the next time point or at the end of the sequence.

## 6.4 Baseline models

Include at least one transparent baseline:

- Majority-class classifier.
- Logistic Regression.
- Simple threshold rule where domain guidance supports it.

## Deliverables

- Final experimental design table.
- Model-input and model-output definitions.
- Baseline definition.

---

# 7. Phase 4 — Data Cleaning and Preprocessing

## Common cleaning tasks

- Standardise column names.
- Convert numeric fields.
- Parse timestamps.
- Remove exact duplicates.
- Document missing-value handling.
- Inspect impossible sensor values.
- Preserve raw data unchanged.
- Save processed data separately.

## Fruit dataset

- Encode fruit type if required.
- Encode the target label.
- Inspect class balance.
- Scale features only where the model requires it.
- Preserve an untouched test set.

## UCI dataset

- Sort by timestamp.
- Check interval consistency.
- Keep data in chronological order.
- Create a time-based split.
- Avoid random cross-validation that leaks future observations into the past.

## Class imbalance

- Report the positive-class rate.
- Prefer class weights for Random Forest and LSTM.
- Use SMOTE only on the training set and only if justified.
- Never apply SMOTE before the time-based split.

## Deliverables

- Cleaning scripts.
- Processed datasets.
- Preprocessing flow diagram.
- Data-quality table.

---

# 8. Phase 5 — Exploratory Data Analysis

## Fruit dataset visualisations

- Target-class distribution.
- Temperature distribution by class.
- Humidity distribution by class.
- CO₂ distribution by class.
- Light distribution by class.
- Fruit category versus target.
- Correlation heatmap.
- Boxplots for outliers.
- Pairwise plots for important variables.

## UCI dataset visualisations

- Temperature and humidity over time.
- Occupancy over time.
- Light and CO₂ over time.
- Daily or hourly patterns.
- Correlation heatmap.
- Class distribution.
- Sensor patterns before state changes.

## Questions to answer

- Which variables differ most across target classes?
- Is the label trivially determined by one variable?
- Are there suspiciously deterministic patterns?
- Is class imbalance realistic?
- Are time trends stable?
- Are there differences between training and test periods?

## Deliverables

- EDA figures for Chapter 4.
- EDA summary table.
- Notes on data limitations.

---

# 9. Phase 6 — Feature Engineering

## Tabular features

- Original temperature.
- Original humidity.
- CO₂.
- Light.
- Fruit type where relevant.
- Temperature × humidity interaction.
- Distance from normal operating range where justified.

## Temporal features for UCI

- Lag 1, lag 5 and lag 10.
- Rolling mean.
- Rolling standard deviation.
- Rate of temperature change.
- Rate of humidity change.
- Hour of day.
- Day of week.
- Time since the last state change.

## Raspberry Pi features

- Current temperature for each refrigerator.
- Current humidity for each refrigerator.
- Rolling mean and rolling standard deviation.
- Temperature delta.
- Humidity delta.
- Door-open count per rolling window.
- Door-open duration.
- Time since the last opening.
- Time of day.

## Critical requirement

The features used on the Raspberry Pi must match the features expected by the deployed model. If the primary model requires CO₂ or light but the Pi does not measure them, choose one of these approaches:

1. Deploy a reduced model trained only on Pi-reproducible features.
2. Add compatible sensors.
3. Use the full model for offline research and a reduced deployment model for the Pi.

The third option is usually the safest and most honest.

## Deliverables

- Feature engineering script.
- Feature dictionary.
- Full research feature set.
- Reduced Raspberry Pi feature set.

---

# 10. Phase 7 — Validation Design

## Fruit dataset

If timestamped:

- chronological train/validation/test split.

If not timestamped:

- stratified train/validation/test split;
- explain that detection delay is not evaluated on this dataset.

## UCI dataset

Use a chronological split:

- earliest observations for training;
- middle period for validation;
- final period for testing.

Optionally use walk-forward validation for a stronger temporal analysis.

## Important safeguards

- Fit scalers only on training data.
- Tune hyperparameters only on validation data.
- Touch the final test set once.
- Keep all preprocessing inside pipelines where possible.
- Set random seeds.
- Log every experiment.

## Deliverables

- Validation diagram.
- Split statistics.
- Leakage-prevention statement.

---

# 11. Phase 8 — Random Forest Implementation

## Steps

1. Build a preprocessing pipeline.
2. Train a default Random Forest baseline.
3. Tune:
   - number of trees;
   - maximum depth;
   - minimum samples per split;
   - minimum samples per leaf;
   - maximum features;
   - class weights.
4. Select parameters using validation data.
5. Evaluate on the held-out test set.
6. Save the final model with Joblib.
7. Apply SHAP TreeExplainer.

## Main metrics

- Recall as the primary safety-oriented metric.
- Precision.
- F1-score.
- PR-AUC.
- ROC-AUC.
- Accuracy for completeness.
- Confusion matrix.

## Deliverables

- Trained Random Forest.
- Hyperparameter table.
- Metrics table.
- SHAP plots.
- Saved model file.

---

# 12. Phase 9 — Isolation Forest Implementation

## Steps

1. Select the environmental features.
2. Fit primarily on normal-class observations where available.
3. Tune:
   - contamination;
   - number of estimators;
   - maximum samples;
   - maximum features.
4. Convert anomaly scores into predicted classes.
5. Compare anomaly predictions with known labels.
6. Analyse false positives and false negatives.
7. Save the fitted model.

## Evaluation

- Recall.
- Precision.
- F1-score.
- PR-AUC where scores are available.
- False-positive rate.
- Detection rate.
- Anomaly-score distributions.

## Important interpretation

Isolation Forest is unsupervised and should be discussed separately from supervised classifiers. It is not automatically inferior if its classification metrics are lower.

## Deliverables

- Trained Isolation Forest.
- Anomaly-score plots.
- Evaluation table.
- Saved model file.

---

# 13. Phase 10 — LSTM Implementation

## Steps

1. Use the timestamped dataset.
2. Sort observations chronologically.
3. Scale numeric features using training data only.
4. Select sequence lengths such as 10, 30 or 60 minutes.
5. Create sliding windows.
6. Build an LSTM model.
7. Use dropout and early stopping.
8. Tune:
   - sequence length;
   - number of LSTM units;
   - dropout;
   - learning rate;
   - batch size.
9. Evaluate on the held-out temporal test set.
10. Save the final network and scaler.

## Evaluation

- Recall.
- Precision.
- F1-score.
- PR-AUC.
- ROC-AUC.
- Confusion matrix.
- Detection delay where event onset is definable.

## Deliverables

- Trained LSTM.
- Learning curves.
- Sequence-length comparison.
- Saved model and scaler.

---

# 14. Phase 11 — Comparative Evaluation

## Fair-comparison rules

- Compare Random Forest and LSTM directly only when they predict the same target on the same samples.
- Discuss Isolation Forest as an unsupervised alternative.
- Do not rank results from different datasets as though they came from one experiment.
- Report each dataset's role clearly.

## Statistical analysis

- Use McNemar's test only for paired classifiers evaluated on the same test observations.
- Report confidence intervals where possible.
- Compare against the defined baseline.
- Discuss practical significance, not only statistical significance.

## Detection delay

Calculate detection delay only when:

- observations have ordered timestamps;
- event onset is known;
- the alert time can be identified.

Do not claim detection delay on an unordered dataset.

## Deliverables

- Comparative performance table.
- Baseline comparison.
- Statistical test results.
- Error analysis.

---

# 15. Phase 12 — Raspberry Pi and IoT Prototype

## 15.1 Hardware

- Raspberry Pi 4.
- Two DHT22 temperature/humidity sensors.
- Two reed-switch door sensors.
- Two refrigerators.
- Breadboard and jumper wires.
- Suitable pull-up resistors if required.
- 32 GB or larger microSD card.
- Raspberry Pi power supply.

## 15.2 Experimental setup

### Refrigerator A — baseline
- Operated normally.
- Door opened only when needed.
- Provides stable-reference behaviour.

### Refrigerator B — controlled disturbance
- More frequent door openings.
- Carefully controlled short door-open periods.
- No unsafe food handling and no deliberate contamination.

## 15.3 Data collection

Record every one to five minutes:

- timestamp;
- refrigerator ID;
- temperature;
- humidity;
- door state;
- door-open count;
- door-open duration.

Save:

- local CSV backup;
- optional Azure IoT Hub message.

## 15.4 Data collection period

Prefer 14 days:

- initial baseline period;
- controlled-disturbance period.

If the schedule becomes too tight, use a shorter justified period rather than abandoning the prototype.

## 15.5 Deployment model

Train a reduced model using only features reproducible on the Pi:

- temperature;
- humidity;
- engineered temporal features;
- door-event features where supported by the deployment experiment.

If no public dataset contains door events, treat door activity as contextual information or a rule-based feature in the prototype rather than pretending it was part of the original training target.

## 15.6 Edge inference

The Raspberry Pi will:

1. read sensor values;
2. update rolling features;
3. load the saved model;
4. generate a risk or anomaly score;
5. log the output;
6. send readings and predictions to Azure;
7. display or trigger an alert.

## 15.7 Prototype testing

Test:

- sensor reliability;
- missing-reading recovery;
- reed-switch accuracy against a manual event log;
- CSV logging;
- model loading;
- prediction latency;
- Azure transmission;
- response to controlled door-opening events.

## Deliverables

- Working prototype.
- Wiring diagram.
- Hardware photographs.
- Sensor-data CSV.
- Live inference screenshots.
- Test-results table.
- Short demonstration video if allowed.

---

# 16. Phase 13 — Explainability

## Random Forest

Use SHAP TreeExplainer to produce:

- global feature importance;
- summary plot;
- dependence plots;
- local explanation for selected high-risk cases.

## LSTM

Possible options:

- SHAP DeepExplainer if stable;
- permutation importance on aggregated temporal features;
- sensitivity analysis.

Do not force a complex LSTM explanation method if it becomes unreliable. A careful limitation is better than a misleading plot.

## Deliverables

- SHAP figures.
- Interpretation table.
- Examples of local explanations.

---

# 17. Phase 14 — Discussion

The discussion should answer:

1. Which environmental variables were most informative?
2. How did supervised, unsupervised and deep-learning approaches differ?
3. Did results generalise across the two datasets?
4. How well did the reduced model transfer to Raspberry Pi data?
5. Did live sensor disturbances change risk or anomaly scores as expected?
6. What would be required for industrial deployment?
7. What are the consequences of not having microbiological swab ground truth?
8. How did the loss of company data affect the scope?
9. Why is the work still relevant to predictive HACCP?

## Required limitation statement

The system does not confirm microbial contamination. It identifies environmental conditions or sensor patterns associated with elevated spoilage or operational risk. Microbiological validation remains future work.

---

# 18. Phase 15 — Dissertation Writing Sequence

Write continuously rather than waiting for implementation to finish.

## Draft order

1. Chapter 1 — Introduction.
2. Chapter 2 — Literature Review.
3. Chapter 3 — Methodology and System Design.
4. Chapter 4 — Data Preparation and Model Development.
5. Chapter 5 — Raspberry Pi Prototype.
6. Chapter 6 — Results and Evaluation.
7. Chapter 7 — Conclusion, Reflection and Future Work.
8. Abstract.
9. AI-use declaration.
10. Management summary and presentation material if required.

## Evidence to capture during implementation

- screenshots;
- code versions;
- error logs;
- hardware photographs;
- diagrams;
- experiment settings;
- parameter tables;
- failed experiments and reasons;
- supervisor feedback and resulting changes.

---

# 19. Two-Week Supervisor Draft Plan

| Day | Technical work | Writing work | Output |
|---|---|---|---|
| 1 | Set up repository; download both datasets; inspect raw files | Create full dissertation structure | Project workspace ready |
| 2 | Complete dataset audit | Draft Chapter 1 | Introduction draft |
| 3 | Begin EDA | HACCP and environmental monitoring literature | Literature section 1 |
| 4 | Continue EDA | IoT, edge AI and food-safety AI literature | Literature section 2 |
| 5 | Finalise EDA and data-quality notes | Critical review and research gap | Chapter 2 draft |
| 6 | Clean datasets and define features | Methodology and dataset-selection sections | Chapter 3 partial |
| 7 | Define validation and baseline | Complete Chapter 3 | Methodology draft |
| 8 | Train Random Forest | Write RF development section | Preliminary RF results |
| 9 | Train Isolation Forest | Write IF development section | Preliminary IF results |
| 10 | Prepare LSTM sequences and train initial model | Write LSTM method | Preliminary LSTM results |
| 11 | Evaluate models and run SHAP | Draft results section | Chapter 6 partial |
| 12 | Set up Raspberry Pi architecture and sensor code | Draft Chapter 5 design section | Prototype chapter draft |
| 13 | Write discussion, limitations and conclusion | Complete missing sections | Full Version 1 |
| 14 | Format, proofread and check references | Final review | Draft sent to supervisor |

---

# 20. Six-Week Completion Plan

| Week | Main focus | Deliverables |
|---|---|---|
| 1 | Dataset audit, literature and hardware setup | Data confirmed, Pi logging started |
| 2 | EDA, preprocessing, methodology and first draft | Dataset and methodology chapters |
| 3 | RF, Isolation Forest and LSTM | Preliminary model results; supervisor draft |
| 4 | Hyperparameter tuning, SHAP and Pi deployment | Final models and working edge prototype |
| 5 | Full evaluation, prototype testing and discussion | Final results and discussion |
| 6 | Supervisor corrections, proofreading and presentation preparation | Submission-ready dissertation |

---

# 21. Minimum Success Criteria

The project will be considered technically complete when it includes:

- a documented audit of both datasets;
- a reproducible preprocessing pipeline;
- Random Forest implementation;
- Isolation Forest implementation;
- LSTM implementation;
- suitable validation without leakage;
- baseline comparison;
- recall, precision, F1 and PR-AUC;
- detection-delay analysis only where valid;
- SHAP explainability;
- a working Raspberry Pi monitoring prototype;
- local CSV logging;
- real-time model inference;
- documented test execution;
- a critical discussion of limitations;
- complete dissertation and AI-use declaration.

---

# 22. High-Mark Enhancements

Complete these only after the core system works:

- walk-forward validation;
- McNemar's test for comparable classifiers;
- confidence intervals;
- Azure dashboard;
- automated alerts;
- model latency and resource-use measurements on the Pi;
- transferability analysis across datasets;
- reduced-versus-full model comparison;
- ablation study;
- public GitHub repository with a reproducible README;
- short user guide and demonstration video.

---

# 23. Immediate Next Actions

1. Download the exact raw files for both datasets.
2. Store them unchanged under `data/raw/`.
3. Run the dataset audit before writing modelling code.
4. Confirm whether the fruit dataset contains timestamps.
5. Confirm the genuine target column and class balance.
6. Start Raspberry Pi sensor logging as soon as the hardware is available.
7. Draft Chapter 1 and the literature matrix while data collection runs.
8. Freeze the experimental design only after the raw-file audit is complete.
