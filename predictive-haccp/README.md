# Predictive HACCP — Explainable AI for IoT Environmental Monitoring

**Project title:** AI-Based Anomaly Detection for Predictive HACCP Using IoT-Based Environmental Monitoring  
**Author:** Asmerom Berhane  
**Degree:** Final Year Project  

---

## Project Overview

This project develops an **Explainable AI (XAI) framework** for **Predictive HACCP** (Hazard Analysis and Critical Control Points) using IoT environmental sensor data. It combines supervised classification, unsupervised anomaly detection, temporal deep learning, and edge deployment on a Raspberry Pi.

### Research Questions

| # | Question |
|---|---|
| RQ1 | Can environmental variables (temperature, humidity, light, CO₂) identify conditions associated with food spoilage? |
| RQ2 | How do Random Forest, Isolation Forest, and LSTM perform on environmental monitoring data? |
| RQ3 | Can the framework support temporal learning and early-warning analysis using continuous IoT data? |
| RQ4 | Can the model be deployed on a Raspberry Pi for real-time risk monitoring? |
| RQ5 | Which environmental variables contribute most, according to SHAP analysis? |

---

## Datasets

| Dataset | Role | Source |
|---|---|---|
| `Dataset 1.csv` — Fruit Spoilage | Primary food-safety dataset; Random Forest + Isolation Forest | Mendeley Data, DOI: `10.17632/czz68d9fwj.1` |
| UCI Occupancy Detection | Temporal IoT benchmark; LSTM + lag features | UCI ML Repository, ID `357` |
| Raspberry Pi (live) | Edge deployment prototype | Collected via DHT22 + reed switches |

---

## Project Structure

```
predictive-haccp/
├── data/
│   ├── raw/                    # Original, untouched datasets
│   │   ├── fruit_spoilage/
│   │   ├── uci_occupancy/
│   │   └── raspberry_pi/
│   ├── interim/                # Partially processed data
│   └── processed/              # Final cleaned datasets
├── notebooks/                  # Jupyter notebooks (EDA, modelling)
├── src/
│   ├── data/                   # Data loading and cleaning scripts
│   ├── features/               # Feature engineering scripts
│   ├── models/                 # Model training scripts
│   ├── evaluation/             # Evaluation and metrics scripts
│   └── deployment/             # Reduced model for deployment
├── pi/                         # Raspberry Pi edge scripts
├── models/                     # Saved model files (.joblib, .keras)
├── reports/
│   ├── figures/                # All generated plots
│   │   ├── fruit_spoilage/
│   │   ├── uci_occupancy/
│   │   └── shap/
│   ├── tables/                 # Metrics tables and data dictionary
│   └── experiment_logs/        # Experiment log
├── tests/                      # Unit tests
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Launch Jupyter

```bash
jupyter notebook
```

---

## Models

| Model | Dataset | Task | Type |
|---|---|---|---|
| Random Forest | Fruit Spoilage | Spoilage classification | Supervised |
| Isolation Forest | Fruit Spoilage | Anomaly detection | Unsupervised |
| LSTM | UCI Occupancy | Temporal state prediction | Supervised (deep learning) |
| Reduced RF | Raspberry Pi | Edge inference | Supervised (reduced features) |

---

## Key Results

> *To be filled in after model training.*

---

## Limitations

- The datasets do not include microbiological ground truth (swab tests or contamination counts).
- The system identifies **environmental conditions associated with spoilage risk**, not confirmed contamination.
- Microbiological validation remains future work.

---

## Reproducibility

- All random seeds are set in each script.
- Preprocessing is fitted only on training data.
- Experiment settings are logged in `reports/experiment_logs/experiment_log.md`.

---

## AI Use Declaration

AI coding assistants were used during development. All AI-generated code was reviewed, tested, and adapted. A full AI-use declaration is included in the dissertation appendix.

---

## Citation

If you use this work, please cite the original datasets:

- **Fruit Spoilage Dataset:** Mendeley Data, DOI: `10.17632/czz68d9fwj.1`
- **UCI Occupancy Detection:** Candanedo, L.M. & Feldheim, V. (2016). Accurate occupancy detection of an office room from light, temperature, humidity and CO2 measurements using statistical learning models. *Energy and Buildings*, 112, 28–39.
