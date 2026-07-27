"""Geomagnetic Storm Predictor — Streamlit app.

Serves three deployed classifiers on the team's time-binned dataset (3-hour bins)
to predict geomagnetic storms (Ap-index threshold): a gradient-boosting model
(XGBoost) that scores each bin from its own features, an LSTM that reads a rolling
48-hour window of solar-wind history, and a TCN (causal, dilated convolutions) that
reads the same kind of window without the current Ap index. All three use the same
time-based split as the modeling notebooks (train < 2022-01-01, test >= 2022-01-01).
"""
import json
from pathlib import Path
import joblib

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

DATA_PATH = "data/time_binned_dataset.csv"
SPLIT_DATE = "2022-01-01"
HORIZONS = [3]
MODEL_NAMES = ("XGBoost", "LSTM", "TCN")
XGB_MODEL_PATH = Path("time-series-modeling/xgboost_storm_3h_deployed.joblib")
XGB_METADATA_PATH = Path("time-series-modeling/xgboost_storm_3h_metadata.json")
LSTM_METADATA_PATH = Path("time-series-modeling/lstm_storm_3h_metadata.json")
LSTM_PRED_PATH = Path("time-series-modeling/lstm_storm_3h_predictions.parquet")
TCN_METADATA_PATH = Path("time-series-modeling/tcn_storm_3h_metadata.json")
TCN_PRED_PATH = Path("time-series-modeling/tcn_storm_3h_predictions.parquet")

# Palette (validated for CVD safety and contrast)
BLUE = "#2a78d6"
RED = "#e34948"
GREEN = "#1a9e77"  # third categorical color (TCN); kept distinct from BLUE/RED under CVD
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"
SEQ_BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

st.set_page_config(page_title="Geomagnetic Storm Predictor", page_icon="🌌", layout="wide")


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


@st.cache_data(show_spinner="Loading dataset…")
def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    float_cols = df.select_dtypes("float64").columns
    df[float_cols] = df[float_cols].astype("float32")
    return df

METADATA_PATHS = {
    "XGBoost": XGB_METADATA_PATH,
    "LSTM": LSTM_METADATA_PATH,
    "TCN": TCN_METADATA_PATH,
}


@st.cache_data(show_spinner="Loading model metadata…")
def load_metadata(model_choice):
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
def train_model(horizon, model_choice):
    if horizon != 3:
        raise ValueError("The deployed models currently support only the 3-hour horizon.")
    df = load_data()
    if model_choice == "LSTM":
        test, proba = _score_lstm(df)
    elif model_choice == "TCN":
        test, proba = _score_tcn(df)
    else:
        test, proba = _score_xgb(df)
    return test, proba


def storm_threshold(df, horizon):
    """Ap threshold implied by the storm labels."""
    return df.loc[df[f"storm_{horizon}h"] == 1, f"ap_target_{horizon}h"].min()


# ---------------------------------------------------------------- sidebar --
st.sidebar.title("🌌 Storm Predictor")
page = st.sidebar.radio("Page", ["Overview", "Model performance", "Forecast explorer"])
st.sidebar.markdown("---")
st.sidebar.caption(
    "Three deployed models — **XGBoost**, **LSTM**, and **TCN** — all trained on "
    "2010–2021 and tested on 2022–2024."
)
horizon = st.sidebar.select_slider("Forecast horizon (hours)", options=HORIZONS, value=3)
st.sidebar.caption(
    f"Predicting whether a geomagnetic storm occurs {horizon} hours ahead, "
    "from solar-wind, flare, and CME conditions in the current 3-hour bin."
)

df = load_data()
target = f"storm_{horizon}h"

# --------------------------------------------------------------- overview --
if page == "Overview":
    st.title("Geomagnetic Storm Predictor")
    st.write(
        "Forecasting geomagnetic storms from OMNI solar-wind parameters, "
        "solar-flare and CME activity, aggregated into 3-hour bins. "
        f"A storm is a bin where the Ap index exceeds "
        f"**{storm_threshold(df, horizon):.0f}** (horizon: {horizon} h)."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Time bins", f"{len(df):,}")
    c2.metric(
        "Date range",
        f"{df['datetime'].min():%Y}–{df['datetime'].max():%Y}",
    )
    c3.metric("Storm bins", f"{int(df[target].sum()):,}")
    c4.metric("Storm rate", f"{df[target].mean() * 100:.2f}%")

    st.subheader("Ap index over time")
    daily = df.set_index("datetime")["ap_now"].resample("D").mean()
    fig, ax = plt.subplots(figsize=(11, 3))
    ax.plot(daily.index, daily.values, color=BLUE, linewidth=0.8)
    ax.set_ylabel("Ap (daily mean)", color=INK_SECONDARY, fontsize=9)
    style_axes(ax)
    st.pyplot(fig, width='stretch')
    plt.close(fig)

    st.subheader(f"Storm bins per year ({horizon} h horizon)")
    yearly = df.groupby(df["datetime"].dt.year)[target].sum()
    fig, ax = plt.subplots(figsize=(11, 3))
    ax.bar(yearly.index, yearly.values, color=BLUE, width=0.7)
    ax.set_ylabel("Storm bins", color=INK_SECONDARY, fontsize=9)
    style_axes(ax)
    st.pyplot(fig, width='stretch')
    plt.close(fig)
    st.caption(
        "Storm counts track the ~11-year solar cycle: maxima around 2003, "
        "2015, and 2024, quiet minima around 2009 and 2019."
    )

    with st.expander("Browse the dataset"):
        st.dataframe(df.tail(500), width='stretch')

# -------------------------------------------------------------- model page --
elif page == "Model performance":
    st.title("Model performance")
    st.write(
        "All three deployed models scored on the **same** held-out test period "
        "(2022–2024), so the numbers are directly comparable. Use the sliders to "
        "move each model's decision threshold — they start at each model's tuned "
        "optimum (the value that maximised F1 on validation)."
    )

    # Score all models on the identical test period.
    scored = {}
    for name in MODEL_NAMES:
        test_m, proba_m = train_model(horizon, name)
        meta_m = load_metadata(name)
        scored[name] = (test_m, proba_m, meta_m)

    # y_true is shared (same rows, same order for both models).
    y_true = scored["XGBoost"][0][target].values

    # ---- per-model decision-threshold sliders (default = tuned optimum) ----
    thr_cols = st.columns(len(MODEL_NAMES))
    thresholds = {}
    recommended = {}
    for col, name in zip(thr_cols, MODEL_NAMES):
        proba_m = scored[name][1]
        rec_thr = round(float(scored[name][2]["operating_threshold"]), 2)
        recommended[name] = rec_thr
        thresholds[name] = col.slider(
            f"{name} decision threshold",
            min_value=0.05, max_value=0.99, value=rec_thr, step=0.01,
        )
        rec_f1 = f1_score(y_true, (proba_m >= rec_thr).astype(int), zero_division=0)
        col.caption(f"⭐ Recommended: **{rec_thr:.2f}** → best F1 **{rec_f1:.2f}** "
                    f"(maximised on validation)")

    def _preds(name):
        proba_m = scored[name][1]
        thr = thresholds[name]
        return proba_m, (proba_m >= thr).astype(int), thr

    # ---- side-by-side metric comparison ----
    st.subheader("Head-to-head metrics")
    rows = {}
    for name in MODEL_NAMES:
        proba_m, y_pred_m, thr = _preds(name)
        rows[name] = {
            "Threshold": round(thr, 2),
            "Precision": round(precision_score(y_true, y_pred_m, zero_division=0), 2),
            "Recall": round(recall_score(y_true, y_pred_m, zero_division=0), 2),
            "F1": round(f1_score(y_true, y_pred_m, zero_division=0), 2),
            "ROC AUC": round(roc_auc_score(y_true, proba_m), 3),
            "PR AUC": round(average_precision_score(y_true, proba_m), 3),
        }
    st.dataframe(pd.DataFrame(rows).T, width="stretch")
    st.caption("ROC AUC and PR AUC are threshold-independent, so they don't move with the sliders.")

    # ---- confusion matrices side by side ----
    st.subheader("Confusion matrices (at the selected thresholds)")
    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUES)
    cm_cols = st.columns(len(MODEL_NAMES))
    cms = {}
    for col, name in zip(cm_cols, MODEL_NAMES):
        proba_m, y_pred_m, thr = _preds(name)
        cm = confusion_matrix(y_true, y_pred_m)
        cms[name] = cm
        with col:
            fig, ax = plt.subplots(figsize=(4.4, 3.6))
            ax.imshow(cm, cmap=cmap)
            for (i, j), v in np.ndenumerate(cm):
                frac = cm[i, j] / cm.max()
                ax.text(j, i, f"{v:,}", ha="center", va="center", fontsize=11,
                        color="white" if frac > 0.5 else INK)
            ax.set_title(f"{name}  (thr {thr:.2f})", fontsize=10, color=INK)
            ax.set_xticks([0, 1], ["No storm", "Storm"], fontsize=9, color=INK_SECONDARY)
            ax.set_yticks([0, 1], ["No storm", "Storm"], fontsize=9, color=INK_SECONDARY)
            ax.set_xlabel("Predicted", color=INK_SECONDARY, fontsize=9)
            ax.set_ylabel("Actual", color=INK_SECONDARY, fontsize=9)
            fig.set_facecolor(SURFACE)
            st.pyplot(fig, width='stretch')
            plt.close(fig)

    # ---- overlaid precision-recall curves ----
    st.subheader("Precision–recall curves")
    model_colors = {"XGBoost": BLUE, "LSTM": RED, "TCN": GREEN}
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for name in MODEL_NAMES:
        proba_m, y_pred_m, thr = _preds(name)
        prec, rec, _ = precision_recall_curve(y_true, proba_m)
        ap_model = average_precision_score(y_true, proba_m)
        ax.plot(rec, prec, color=model_colors[name], linewidth=2, label=f"{name} (AP {ap_model:.2f})")
        # recommended operating point (star) — always visible so the optimum stays anchored
        rec_pred = (proba_m >= recommended[name]).astype(int)
        ax.scatter([recall_score(y_true, rec_pred)], [precision_score(y_true, rec_pred, zero_division=0)],
                   color=model_colors[name], marker="*", s=190, zorder=4,
                   edgecolor="white", linewidth=1.0)
        # current selection (dot) — moves with the slider
        ax.scatter([recall_score(y_true, y_pred_m)], [precision_score(y_true, y_pred_m, zero_division=0)],
                   color=model_colors[name], s=55, zorder=5, edgecolor="white", linewidth=1.2)
    ax.set_xlabel("Recall", color=INK_SECONDARY, fontsize=9)
    ax.set_ylabel("Precision", color=INK_SECONDARY, fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=8, frameon=False, labelcolor=INK_SECONDARY)
    style_axes(ax)
    st.pyplot(fig, width='stretch')
    plt.close(fig)
    st.caption("⭐ = recommended threshold (best F1) · ● = your selected threshold.")

    # ---- narrative ----
    bits = []
    for name in MODEL_NAMES:
        tn, fp, fn, tp = cms[name].ravel()
        bits.append(f"**{name}** catches {tp} of {tp + fn} storms with {fp} false alarms")
    st.write(
        f"On the test period ({SPLIT_DATE} to {df['datetime'].max():%Y-%m-%d}), "
        + "; ".join(bits) + "."
    )

# --------------------------------------------------------------- explorer --
else:
    st.title("Forecast explorer")
    st.write(
        "Step through the held-out test period (2022 onward) and compare a "
        "model's storm probability with what actually happened."
    )

    explorer_model = st.radio("Model", list(MODEL_NAMES), horizontal=True)
    test, proba = train_model(horizon, explorer_model)
    test = test.assign(storm_probability=proba)

    min_d, max_d = test["datetime"].min().date(), test["datetime"].max().date()
    window = st.slider(
        "Test-period window",
        min_value=min_d, max_value=max_d,
        value=(pd.Timestamp("2024-05-01").date(), pd.Timestamp("2024-06-01").date()),
        format="YYYY-MM-DD",
    )
    mask = (test["datetime"].dt.date >= window[0]) & (test["datetime"].dt.date <= window[1])
    view = test[mask]

    if view.empty:
        st.info("No bins in the selected window.")
    else:
        storms = view[view[target] == 1]
        fig, ax = plt.subplots(figsize=(11, 3.4))
        ax.plot(
            view["datetime"], view["storm_probability"],
            color=BLUE, linewidth=1.4, label="Predicted storm probability",
        )
        if not storms.empty:
            ax.scatter(
                storms["datetime"], np.full(len(storms), 1.02),
                color=RED, marker="v", s=36, label="Actual storm bin", zorder=3,
            )
        ax.set_ylim(0, 1.08)
        ax.set_ylabel("P(storm)", color=INK_SECONDARY, fontsize=9)
        ax.legend(fontsize=8, frameon=False, loc="upper left", labelcolor=INK_SECONDARY)
        style_axes(ax)
        st.pyplot(fig, width='stretch')
        plt.close(fig)

        st.caption(
            f"{len(view):,} bins shown · {len(storms)} actual storm bins in window. "
            "Try May 2024 — the Gannon storm, the strongest in two decades."
        )

        st.subheader("Conditions at the highest-risk bin")
        peak = view.loc[view["storm_probability"].idxmax()]
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("When", f"{peak['datetime']:%Y-%m-%d %H:%M}")
        c2.metric("P(storm)", f"{peak['storm_probability']:.2f}")
        c3.metric("Bz GSM (nT)", f"{peak['bz_gsm_nt_last']:.1f}")
        c4.metric("Flow speed (km/s)", f"{peak['flow_speed_kms_last']:.0f}")
        c5.metric("Actual outcome", "Storm" if peak[target] == 1 else "No storm")
