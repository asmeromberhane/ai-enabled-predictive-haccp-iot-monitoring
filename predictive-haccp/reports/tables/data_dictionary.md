# Data Dictionary — Predictive HACCP

**Project:** AI-Based Anomaly Detection for Predictive HACCP Using IoT-Based Environmental Monitoring

---

## 1. Fruit Spoilage Dataset (`Dataset 1.csv`)

**Source:** Mendeley Data, DOI: `10.17632/czz68d9fwj.1`  
**Licence:** Verify before publication  
**Role:** Primary food-safety dataset for supervised classification (Random Forest) and anomaly detection (Isolation Forest).  
**Rows:** 10,995 (raw) → 8,115 after removing 2,880 exact duplicates  
**Timestamp:** Not present — no temporal modelling or detection-delay analysis applicable  

### Raw Columns

| Column name (raw) | Standardised name | Type | Description | Notes |
|---|---|---|---|---|
| `Fruit` | `fruit` | Categorical | Type of fruit monitored | Values: Tomato, Orange, Pineapple, Banana |
| `Temp` | `temperature` | Float | Ambient temperature in °C | Range: 21–27 °C |
| `Humid (%)` | `humidity` | Float | Relative humidity (%) | Range: 71–95 % |
| `Light (Fux)` | `light` | Float | Light intensity in lux | Range: 4.2–268.4 lux |
| `CO2 (pmm)` | `co2` | Float | CO₂ concentration (ppm) | Range: 20–478 ppm |
| `Class` | `class_label` | Categorical → Binary | Spoilage class | Raw: Good, Bad, BAD → Cleaned: Good (0), Bad (1) |

### Target Encoding

| Raw label | Cleaned label | Numeric code |
|---|---|---|
| `Good` | `Good` | `0` |
| `Bad` | `Bad` | `1` |
| `BAD` | `Bad` (merged) | `1` |

### Class Balance (after cleaning)

| Class | Count | Percentage |
|---|---|---|
| Good | 5,667 | 51.5% |
| Bad | 5,328 | 48.5% |

### Data Quality Issues

| Issue | Count | Decision |
|---|---|---|
| Exact duplicate rows | 2,880 | Remove before splitting — document in dissertation |
| Missing values | 0 | None found |
| `BAD` label inconsistency | 711 rows | Merged into `Bad` |

---

## 2. UCI Occupancy Detection Dataset

**Source:** UCI Machine Learning Repository, Dataset ID `357`  
**Citation:** Candanedo, L.M. & Feldheim, V. (2016). Accurate occupancy detection of an office room from light, temperature, humidity and CO2 measurements using statistical learning models. *Energy and Buildings*, 112, 28–39.  
**Licence:** Public domain / research use  
**Role:** Temporal IoT benchmark for LSTM, lag features, rolling windows. **Not a food-safety dataset.**  
**Sampling interval:** Approximately 60 seconds  

### Raw Columns

| Column name (raw) | Standardised name | Type | Description |
|---|---|---|---|
| `date` | `timestamp` | Datetime | Observation timestamp |
| `Temperature` | `temperature` | Float | Room temperature in °C |
| `Humidity` | `humidity` | Float | Relative humidity (%) |
| `Light` | `light` | Float | Light intensity (lux) |
| `CO2` | `co2` | Float | CO₂ concentration (ppm) |
| `HumidityRatio` | `humidity_ratio` | Float | Humidity ratio (kg water/kg air) |
| `Occupancy` | `occupancy` | Binary | 0 = Unoccupied, 1 = Occupied |

### File-Level Summary

| File | Rows | Date range | Unoccupied | Occupied |
|---|---|---|---|---|
| `datatraining.txt` | 8,143 | 2015-02-04 to 2015-02-10 | 6,414 (78.8%) | 1,729 (21.2%) |
| `datatest.txt` | 2,665 | 2015-02-02 to 2015-02-04 | 1,693 (63.5%) | 972 (36.5%) |
| `datatest2.txt` | 9,752 | 2015-02-11 to 2015-02-18 | 7,703 (79.0%) | 2,049 (21.0%) |

### Split Strategy

| Split | Source file(s) | Temporal order preserved |
|---|---|---|
| Training | `datatraining.txt` | Yes |
| Validation / Test | `datatest.txt`, `datatest2.txt` | Yes |

---

## 3. Raspberry Pi Live Data (Prototype)

**Role:** Edge deployment demonstration. Not the primary training data.  
**Collection:** DHT22 temperature/humidity sensors + reed-switch door sensors on two refrigerators.

### Pi Feature Set (Reduced)

| Feature | Description |
|---|---|
| `timestamp` | Observation time |
| `fridge_id` | Refrigerator identifier (A or B) |
| `temperature` | Current temperature (°C) |
| `humidity` | Current humidity (%) |
| `door_state` | 0 = closed, 1 = open |
| `door_open_count` | Number of door opens in rolling window |
| `door_open_duration` | Total open duration in rolling window (seconds) |
| `temp_rolling_mean` | Rolling mean temperature |
| `temp_rolling_std` | Rolling standard deviation of temperature |
| `humidity_rolling_mean` | Rolling mean humidity |
| `humidity_rolling_std` | Rolling standard deviation of humidity |
| `temp_delta` | Temperature change from previous reading |
| `humidity_delta` | Humidity change from previous reading |
| `hour_of_day` | Hour extracted from timestamp |

---

## 4. Engineered Features

### Fruit Spoilage Tabular Features

| Feature | Description |
|---|---|
| `temp_humidity_interaction` | `temperature × humidity` |
| `co2_light_interaction` | `co2 × light` |
| `fruit_*` | One-hot encoded fruit type columns |

### UCI Temporal Features

| Feature | Description |
|---|---|
| `temp_lag_1` / `_5` / `_10` | Temperature lagged by 1, 5, 10 rows |
| `humidity_lag_1` etc. | Humidity lagged |
| `co2_lag_1` etc. | CO₂ lagged |
| `temp_rolling_mean` | Rolling mean (window defined in script) |
| `temp_rolling_std` | Rolling standard deviation |
| `temp_change` | Row-to-row temperature difference |
| `humidity_change` | Row-to-row humidity difference |
| `hour_of_day` | Hour extracted from timestamp |
| `day_of_week` | Day of week (0=Monday) |

---

*Last updated: 2026-08-13 — Phase 0 project setup*
