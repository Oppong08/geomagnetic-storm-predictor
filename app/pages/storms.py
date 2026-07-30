"""Storm explorer — open one real event and see what every model made of it."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from app import charts, ui
from app.data import (
    MODEL_NAMES,
    TARGET,
    flare_letter,
    operating_threshold,
    score_all,
)
from app.theme import tokens


def render() -> None:
    t = tokens()
    scored = score_all()
    colors = dict(zip(MODEL_NAMES, t["series"]))

    ui.page_header(
        "Storm explorer",
        "Pick a real storm from the test period and see the full solar-wind, flare and "
        "CME picture at its peak — alongside what each of the three models predicted at "
        "that exact moment.",
    )

    storms = scored[scored[TARGET] == 1].sort_values("datetime").reset_index(drop=True)
    if storms.empty:
        st.info("No storm bins found in the scored test period.")
        return

    # Contiguous storm bins (no gap larger than one bin) are treated as one event.
    storms["event_id"] = (storms["datetime"].diff() > pd.Timedelta(hours=3)).cumsum()
    events = (
        storms.groupby("event_id")
        .agg(
            start=("datetime", "min"), end=("datetime", "max"),
            peak_ap=("ap_now", "max"), n_bins=("datetime", "size"),
        )
        .reset_index()
        .sort_values("peak_ap", ascending=False)
    )

    event_bins = _picker(storms, events)
    peak = event_bins.loc[event_bins["ap_now"].idxmax()]

    _model_read(peak, colors)
    _conditions(peak)
    _event_shape(event_bins, t)
    _event_table(event_bins)
    ui.footer()


# ------------------------------------------------------------------ picker --
def _picker(storms: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    options = {
        f"{row.start:%d %b %Y %H:%M} → {row.end:%d %b %Y %H:%M}   ·   "
        f"peak Ap {row.peak_ap:.0f} · {row.n_bins} bin{'s' if row.n_bins > 1 else ''}": row.event_id
        for row in events.itertuples()
    }
    with st.container(border=True):
        choice = st.selectbox(
            f"Storm event — {len(events)} in the test period, strongest first",
            list(options),
        )
    return storms[storms["event_id"] == options[choice]].sort_values("datetime")


# -------------------------------------------------------------- model read --
def _model_read(peak: pd.Series, colors: dict) -> None:
    thresholds = {name: operating_threshold(name) for name in MODEL_NAMES}
    fired = {name: peak[f"proba_{name}"] >= thresholds[name] for name in MODEL_NAMES}
    caught = sum(fired.values())

    if caught == len(MODEL_NAMES):
        tone, verdict = "quiet", "All three models called it"
    elif caught == 0:
        tone, verdict = "storm", "Every model missed this one"
    else:
        tone, verdict = "elevated", f"{caught} of {len(MODEL_NAMES)} models called it"

    ui.section(
        f"Peak of the event — {peak['datetime']:%d %b %Y %H:%M} UTC",
        "Each model's probability for this bin, against its own tuned threshold. "
        "The thresholds differ, so the same probability can be a storm call for one "
        "model and a quiet call for another.",
    )
    with st.container(border=True):
        st.markdown(ui.badge(verdict, tone), unsafe_allow_html=True)
        st.write("")
        cols = st.columns(len(MODEL_NAMES), gap="medium")
        for col, name in zip(cols, MODEL_NAMES):
            with col:
                st.markdown(
                    ui.probability_bar(
                        name, colors[name], float(peak[f"proba_{name}"]),
                        thresholds[name], fired[name],
                    ),
                    unsafe_allow_html=True,
                )


# -------------------------------------------------------------- conditions --
def _conditions(peak: pd.Series) -> None:
    ui.section(
        "Conditions at the peak",
        "A southward (negative) Bz with high flow speed and pressure is the classic "
        "signature that precedes a storm; flare class and CME counts add the eruptive "
        "context that drove it.",
    )
    ui.stat_row(
        [
            ("Ap now", f"{peak['ap_now']:.0f}", "observed disturbance"),
            ("Bz GSM", f"{peak['bz_gsm_nt_last']:.1f} nT", "southward = coupling"),
            ("Flow speed", f"{peak['flow_speed_kms_last']:.0f} km/s", "solar-wind bulk speed"),
            ("Proton density", f"{peak['proton_density_cm3_last']:.1f} cm⁻³", "particles per cm³"),
        ]
    )
    st.write("")
    ui.stat_row(
        [
            ("Flow pressure", f"{peak['flow_pressure_npa_last']:.2f} nPa", "push on the magnetosphere"),
            ("Flare class", flare_letter(peak["flare_max_class_now"]), "strongest flare this bin"),
            (
                "CMEs, prior 72h",
                f"{int(peak['cme_count_72h'])}" if pd.notna(peak["cme_count_72h"]) else "—",
                "eruptions in transit",
            ),
            (
                "Fastest CME, 72h",
                f"{peak['cme_max_speed_72h']:.0f} km/s" if pd.notna(peak["cme_max_speed_72h"]) else "—",
                "peak ejection speed",
            ),
        ]
    )


# ------------------------------------------------------------ event shape --
def _event_shape(event_bins: pd.DataFrame, t: dict) -> None:
    if len(event_bins) < 2:
        return
    ui.section(
        "How the event unfolded",
        "Model probabilities bin by bin, with the observed Ap shaded behind them as a "
        "share of this event's peak. Both are put on one 0–1 axis rather than a second "
        "scale, so the shapes stay honestly comparable.",
    )
    data = event_bins.copy()
    data["ap_share"] = data["ap_now"] / data["ap_now"].max()
    with st.container(border=True):
        st.altair_chart(charts.event_conditions(data, t), theme=None)


# ------------------------------------------------------------ event table --
def _event_table(event_bins: pd.DataFrame) -> None:
    display_cols = {
        "datetime": "Time",
        "ap_now": "Ap",
        "bz_gsm_nt_last": "Bz GSM (nT)",
        "flow_speed_kms_last": "Flow speed (km/s)",
        "proton_density_cm3_last": "Density (cm⁻³)",
        "flow_pressure_npa_last": "Pressure (nPa)",
        "cme_count_72h": "CMEs (72h)",
        "cme_max_speed_72h": "Max CME speed (km/s)",
        "proba_XGBoost": "P XGBoost",
        "proba_LSTM": "P LSTM",
        "proba_TCN": "P TCN",
    }
    table = event_bins[list(display_cols)].rename(columns=display_cols)

    with st.expander(f"All {len(event_bins)} bins in this event"):
        st.dataframe(
            table,
            hide_index=True,
            width="stretch",
            column_config={
                "Time": st.column_config.DatetimeColumn(format="DD MMM YYYY HH:mm"),
                "Ap": st.column_config.NumberColumn(format="%.0f"),
                "Bz GSM (nT)": st.column_config.NumberColumn(format="%.1f"),
                "Flow speed (km/s)": st.column_config.NumberColumn(format="%.0f"),
                "Density (cm⁻³)": st.column_config.NumberColumn(format="%.1f"),
                "Pressure (nPa)": st.column_config.NumberColumn(format="%.2f"),
                "CMEs (72h)": st.column_config.NumberColumn(format="%.0f"),
                "Max CME speed (km/s)": st.column_config.NumberColumn(format="%.0f"),
                **{
                    label: st.column_config.ProgressColumn(
                        format="%.2f", min_value=0.0, max_value=1.0
                    )
                    for label in ("P XGBoost", "P LSTM", "P TCN")
                },
            },
        )
