# Geomagnetic Storm Predictor

Forecasting 3-hour-ahead geomagnetic storms from OMNI solar-wind parameters, solar-flare activity, and CME activity. This project was developed through the AI4ALL Ignite program and compares three deployed machine learning classifiers — XGBoost, LSTM, and TCN — through an interactive Streamlit dashboard.

The modeling dataset (`data/time_binned_dataset.csv`) bins 1995–2024 space-weather observations into 3-hour windows. For each forecast horizon — 3, 6, 12, and 24 hours — the dataset includes a future Ap target and a binary storm label. For example, `ap_target_3h` stores the future Ap value 3 hours ahead, and `storm_3h = 1` if that future Ap index is at least 50, approximately corresponding to Kp 5 geomagnetic storm conditions. The final deployed models focus on the 3-hour-ahead prediction task (`storm_3h`).

## Problem Statement

Geomagnetic storms can disrupt satellites, GPS, radio communication, aviation systems, and power infrastructure. Because strong storms are rare but high-impact events, prediction systems need to identify elevated storm risk while balancing two competing goals: catching true storms and reducing false alarms.

This project asks:

> Can solar-wind, flare, CME, and geomagnetic history data be used to predict whether a geomagnetic storm will occur in the next 3 hours?

## Key Results

- Built a time-binned modeling dataset where each row represents a 3-hour space-weather interval.
- Trained and deployed three 3-hour storm classifiers: XGBoost, LSTM, and TCN.
- Used a chronological split to avoid time leakage: training on 2010–2021 and testing on 2022–2024.
- Compared models on the same held-out test period using ROC-AUC, PR-AUC, precision, recall, and F1.
- Developed a Streamlit dashboard for side-by-side model comparison and forecast exploration.

| Model | Input style | Test PR-AUC | Precision | Recall | F1 | Operating threshold |
|---|---|---:|---:|---:|---:|---:|
| XGBoost | Single 3-hour bin, no current Ap features | 0.575 | 0.375 | 0.684 | 0.485 | 0.90 |
| LSTM | 48-hour sequence with solar-wind + Ap history | 0.610 | 0.729 | 0.448 | 0.555 | 0.795 |
| TCN | 48-hour sequence of solar-wind features, no current Ap | 0.602 | 0.611 | 0.552 | 0.580 | 0.60 |

The TCN achieved the strongest final F1 score, the LSTM achieved the highest precision and PR-AUC, and XGBoost provided a more interpretable tabular baseline with higher recall.

## Methodologies

We framed geomagnetic storm prediction as a rare-event binary classification task. The main target, `storm_3h`, indicates whether a geomagnetic storm occurs in the next 3-hour forecast window.

For each row at time `T`, features are built only from information available at or before `T`. Targets are created by shifting future Ap values forward by forecast horizon: `ap_target_{H}h` stores the future Ap value, and `storm_{H}h` marks whether that future Ap crosses the storm threshold. Rows with missing targets are dropped so all horizons share the same rows and can be compared directly.

Our workflow included:

- aggregating space-weather observations into 3-hour time bins
- engineering features from solar wind, magnetic-field, flare, CME, and recurrence variables
- using chronological train/test splits to prevent time leakage
- testing both tabular and sequence-modeling approaches
- handling severe class imbalance through oversampling, weighted losses, and focal loss
- tuning operating thresholds to balance precision and recall
- deploying final model artifacts and cached predictions in Streamlit

Final deployed models:

- **XGBoost:** tree-based classifier using engineered tabular features from the current 3-hour bin
- **LSTM:** recurrent sequence model using a rolling 48-hour history
- **TCN:** causal dilated convolutional sequence model using a rolling 48-hour solar-wind history

## Streamlit App

The interactive dashboard (`streamlit_app.py`) serves three deployed classifiers:

- XGBoost model artifact: `time-series-modeling/xgboost_storm_3h_deployed.joblib`
- LSTM cached predictions: `time-series-modeling/lstm_storm_3h_predictions.parquet`
- TCN cached predictions: `time-series-modeling/tcn_storm_3h_predictions.parquet`

The dashboard includes:

- an overview page with a forecast read-out for the latest scored bin, headline dataset statistics, and Ap activity across three solar cycles
- model performance comparison on the held-out 2022–2024 test period
- adjustable decision thresholds for each model
- confusion matrices and precision-recall curves
- forecast explorer for selected test-period windows
- storm explorer for individual events, with the solar-wind, flare and CME conditions at their peak

`streamlit_app.py` is a thin entry point; the dashboard itself lives in the `app/` package:

| Module | Responsibility |
|---|---|
| `app/theme.py` | design tokens for light and dark, global CSS, active-theme detection |
| `app/data.py` | dataset loading, model artifacts, cached test-period scoring |
| `app/charts.py` | Altair chart builders (theme-aware, interactive) |
| `app/ui.py` | shared components — hero, stat tiles, badges, model cards |
| `app/nav.py`, `app/pages/` | navigation and one module per page |

The app follows the viewer's light/dark setting; both palettes are defined in `.streamlit/config.toml`, and the theme can be changed from the ⋮ menu under Settings → Appearance.

Run locally:

```bash
pip install -r requirements.txt
python -m streamlit run streamlit_app.py
```

To deploy on [Streamlit Community Cloud](https://share.streamlit.io), point a
new app at this repo with `streamlit_app.py` as the entrypoint.

## Data Sources

This project uses public space-weather datasets, including:

- OMNI-style solar-wind and interplanetary magnetic-field variables
- NASA DONKI space-weather event data, including CME activity
- solar flare event data
- space-weather index data, including Ap-index targets

These sources were combined into `data/time_binned_dataset.csv`, a 3-hour binned modeling dataset used for training and evaluation.

## Technologies Used

- Python
- pandas, NumPy
- scikit-learn
- XGBoost
- PyTorch
- imbalanced-learn
- matplotlib
- Streamlit
- joblib
- parquet
- Git/GitHub

## Repository Layout

- `data/` — raw, cleaned, and combined time-binned datasets
- `exploratory-data-analysis/` — EDA notebooks for individual data sources
- `notebooks/` — dataset assembly and baseline modeling notebooks
- `time-series-modeling/` — final time-aware models, deployed artifacts, metadata files, and cached predictions
- `streamlit_app.py` — Streamlit dashboard for model comparison and forecast exploration
- `requirements.txt` — Python dependencies

## Authors

This project was completed in collaboration with: 

- Nithila Sadheesh
- Kamsi Ozorji
- Nancy Nakyung Kwak
- Chan Li
- Hafsah Khan
- Nana Oppong Ampofo

## AI4ALL Ignite

This project was developed as part of AI4ALL Ignite, applying machine learning, responsible AI thinking, technical communication, and deployment skills to a real-world space-weather prediction problem.