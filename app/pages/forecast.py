"""Forecast explorer — walk the test period and compare prediction to reality."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from app import charts, ui
from app.data import (
    MODEL_NAMES,
    TARGET,
    operating_threshold,
    score_all,
)
from app.theme import tokens

_WINDOW_KEY = "fx_window"

# Named windows worth landing on. Gannon is the headline: the strongest storm in
# two decades, and the clearest test of whether a model saw it coming.
_PRESETS = {
    "Gannon storm · May 2024": ("2024-05-01", "2024-06-01"),
    "Halloween-scale run · Mar 2023": ("2023-03-15", "2023-04-15"),
    "Quiet spell · Jul 2022": ("2022-07-01", "2022-08-01"),
}


def render() -> None:
    t = tokens()
    scored = score_all()
    colors = dict(zip(MODEL_NAMES, t["series"]))

    ui.page_header(
        "Forecast explorer",
        "Step through the held-out test period and compare a model's storm probability "
        "with what actually happened. Red bands mark real storm bins — a model doing its "
        "job lifts its curve just before or during them.",
    )

    model, window = _controls(scored)
    threshold = operating_threshold(model)

    view = scored[
        (scored["datetime"].dt.date >= window[0]) & (scored["datetime"].dt.date <= window[1])
    ].copy()

    if view.empty:
        st.info("No bins fall inside the selected window. Widen the range or pick a preset.")
        return

    view["storm_probability"] = view[f"proba_{model}"]
    view["is_storm"] = view[TARGET] == 1
    view["outcome"] = view["is_storm"].map({True: "Storm", False: "No storm"})

    _timeline(view, t, threshold, colors[model], model)
    _peak(view, model)
    ui.footer()


# ---------------------------------------------------------------- controls --
def _controls(scored: pd.DataFrame):
    min_d = scored["datetime"].min().date()
    max_d = scored["datetime"].max().date()
    default = (pd.Timestamp("2024-05-01").date(), pd.Timestamp("2024-06-01").date())
    st.session_state.setdefault(_WINDOW_KEY, default)

    with st.container(border=True):
        left, right = st.columns([1, 1.6], gap="large", vertical_alignment="center")
        with left:
            model = st.segmented_control(
                "Model", options=list(MODEL_NAMES), default=MODEL_NAMES[2],
                key="fx_model", width="stretch",
            ) or MODEL_NAMES[2]
        with right:
            window = st.slider(
                "Window", min_value=min_d, max_value=max_d,
                format="YYYY-MM-DD", key=_WINDOW_KEY,
            )

        preset_cols = st.columns(len(_PRESETS) + 1, gap="small")
        for col, (label, (start, end)) in zip(preset_cols, _PRESETS.items()):
            if col.button(label, width="stretch", key=f"preset_{label}"):
                st.session_state[_WINDOW_KEY] = (
                    pd.Timestamp(start).date(), pd.Timestamp(end).date(),
                )
                st.rerun()
        if preset_cols[-1].button("Full test period", width="stretch", key="preset_full"):
            st.session_state[_WINDOW_KEY] = (min_d, max_d)
            st.rerun()

    return model, window


# ---------------------------------------------------------------- timeline --
def _timeline(view: pd.DataFrame, t: dict, threshold: float, color: str, model: str) -> None:
    storms = int(view["is_storm"].sum())
    ui.section(
        f"{model} · predicted storm probability",
        f"{len(view):,} three-hour bins in view · {storms} real storm bin"
        f"{'' if storms == 1 else 's'} · dashed line is the model's tuned decision "
        f"threshold ({threshold:.2f}).",
    )
    with st.container(border=True):
        st.altair_chart(
            charts.forecast_timeline(view, t, threshold, color), theme=None
        )


# -------------------------------------------------------------- peak bin --
def _peak(view: pd.DataFrame, model: str) -> None:
    peak = view.loc[view["storm_probability"].idxmax()]
    called = peak["storm_probability"] >= operating_threshold(model)
    actual = bool(peak[TARGET] == 1)

    if actual and called:
        tone, verdict = "storm", "Caught storm"
    elif actual and not called:
        tone, verdict = "watch", "Missed storm"
    elif called:
        tone, verdict = "elevated", "False alarm"
    else:
        tone, verdict = "quiet", "Quiet, correctly"

    ui.section(
        "Highest-risk bin in this window",
        "The conditions the model was reacting to at its most confident moment.",
    )
    with st.container(border=True):
        st.markdown(ui.badge(verdict, tone), unsafe_allow_html=True)
        st.write("")
        ui.stat_row(
            [
                ("When", f"{peak['datetime']:%d %b %Y}", f"{peak['datetime']:%H:%M} UTC"),
                ("P(storm)", f"{peak['storm_probability']:.2f}", f"{model} · threshold {operating_threshold(model):.2f}"),
                ("Bz GSM", f"{peak['bz_gsm_nt_last']:.1f} nT", "southward field couples to Earth's"),
                ("Flow speed", f"{peak['flow_speed_kms_last']:.0f} km/s", "solar-wind bulk speed"),
                ("Ap 3h later", f"{peak['ap_target_3h']:.0f}", "what actually happened"),
            ]
        )
