"""Dataset, model artifacts, and test-period scoring.

Serves three deployed classifiers on the team's time-binned dataset (3-hour
bins) to predict geomagnetic storms (Ap-index threshold): a gradient-boosting
model (XGBoost) that scores each bin from its own features, an LSTM that reads a
rolling 48-hour window of solar-wind history, and a TCN (causal, dilated
convolutions) that reads the same kind of window without the current Ap index.
All three use the same time-based split as the modeling notebooks
(train < 2022-01-01, test >= 2022-01-01).

This module is the app's whole data surface — the pages read from here and
nowhere else. Scoring behaviour is unchanged from the original single-file app.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

DATA_PATH = "data/time_binned_dataset.csv"
SPLIT_DATE = "2022-01-01"
HORIZON = 3
MODEL_NAMES = ("XGBoost", "LSTM", "TCN")
TARGET = f"storm_{HORIZON}h"

XGB_MODEL_PATH = Path("time-series-modeling/xgboost_storm_3h_deployed.joblib")
METADATA_PATHS = {
    "XGBoost": Path("time-series-modeling/xgboost_storm_3h_metadata.json"),
    "LSTM": Path("time-series-modeling/lstm_storm_3h_metadata.json"),
    "TCN": Path("time-series-modeling/tcn_storm_3h_metadata.json"),
}
LSTM_PRED_PATH = Path("time-series-modeling/lstm_storm_3h_predictions.parquet")
TCN_PRED_PATH = Path("time-series-modeling/tcn_storm_3h_predictions.parquet")

# One-line architecture summaries for the model cards. The metadata files carry
# the full `model_type` strings, which are too dense to read at a glance.
MODEL_BLURBS = {
    "XGBoost": "Gradient-boosted trees over engineered features from the current "
               "3-hour bin. No current-Ap features, so it reads conditions rather "
               "than the disturbance itself.",
    "LSTM": "Recurrent sequence model over a rolling 48-hour window of solar-wind "
            "and Ap history — it learns how a storm builds, not just how one bin looks.",
    "TCN": "Causal dilated convolutions over the same 48-hour window, without ever "
           "seeing the current Ap index. The hardest setup, and the strongest F1.",
}

FLARE_CLASS_LETTERS = {0: "A", 1: "B", 2: "C", 3: "M", 4: "X"}


@st.cache_data(show_spinner="Loading dataset…")
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, parse_dates=["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    float_cols = df.select_dtypes("float64").columns
    df[float_cols] = df[float_cols].astype("float32")
    return df


@st.cache_data(show_spinner="Loading model metadata…")
def load_metadata(model_choice: str) -> dict:
    with open(METADATA_PATHS[model_choice]) as f:
        return json.load(f)


@st.cache_resource(show_spinner="Loading deployed XGBoost model…")
def load_xgb_model():
    return joblib.load(XGB_MODEL_PATH)


def _score_xgb(df):
    metadata = load_metadata("XGBoost")
    model = load_xgb_model()
    features = metadata["features"]
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required XGBoost features: {missing}")
    test = df[df["datetime"] >= SPLIT_DATE].copy()
    proba = model.predict_proba(test[features])[:, 1]
    return test.reset_index(drop=True), proba


def _score_lstm(df):
    """Serve the LSTM's cached test-period predictions (computed offline by
    time-series-modeling/train_lstm_deploy.py). Kept torch-free so the dashboard
    process stays light and stable."""
    preds = pd.read_parquet(LSTM_PRED_PATH)
    test = df[df["datetime"] >= SPLIT_DATE].merge(preds, on="datetime", how="inner")
    test = test.sort_values("datetime").reset_index(drop=True)
    return test, test["storm_probability"].to_numpy()


def _score_tcn(df):
    """Serve the TCN's cached test-period predictions (exported directly by
    time-series-modeling/tcn.ipynb's "Export for the Streamlit App" section, the
    same way xgboost.ipynb saves its own deploy artifacts). Kept torch-free so the
    dashboard process stays light and stable."""
    preds = pd.read_parquet(TCN_PRED_PATH)
    test = df[df["datetime"] >= SPLIT_DATE].merge(preds, on="datetime", how="inner")
    test = test.sort_values("datetime").reset_index(drop=True)
    return test, test["storm_probability"].to_numpy()


@st.cache_resource(show_spinner="Scoring test period…")
def score_model(model_choice: str, horizon: int = HORIZON):
    if horizon != HORIZON:
        raise ValueError("The deployed models currently support only the 3-hour horizon.")
    df = load_data()
    if model_choice == "LSTM":
        return _score_lstm(df)
    if model_choice == "TCN":
        return _score_tcn(df)
    return _score_xgb(df)


@st.cache_resource(show_spinner="Scoring test period…")
def score_all():
    """Every model's probability on the identical test rows, in one frame.

    All three score the same rows in the same order, so the storm label from any
    of them is the shared ground truth.
    """
    merged = None
    for name in MODEL_NAMES:
        test, proba = score_model(name)
        scored = test.assign(**{f"proba_{name}": proba})
        if merged is None:
            merged = scored
        else:
            merged = merged.merge(
                scored[["datetime", f"proba_{name}"]], on="datetime", how="inner"
            )
    return merged.sort_values("datetime").reset_index(drop=True)


def storm_threshold(df: pd.DataFrame, horizon: int = HORIZON) -> float:
    """Ap threshold implied by the storm labels."""
    return float(df.loc[df[f"storm_{horizon}h"] == 1, f"ap_target_{horizon}h"].min())


def operating_threshold(model_choice: str) -> float:
    """The tuned decision threshold each model was deployed with."""
    return round(float(load_metadata(model_choice)["operating_threshold"]), 2)


def flare_letter(value) -> str:
    return FLARE_CLASS_LETTERS.get(int(value), "—") if pd.notna(value) else "—"
