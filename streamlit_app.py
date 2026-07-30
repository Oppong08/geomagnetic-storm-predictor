"""Geomagnetic Storm Predictor — Streamlit dashboard.

Entry point only: it applies the theme, builds the navigation, and hands off to
the page modules in ``app/pages``. The dashboard serves three deployed
classifiers (XGBoost, LSTM, TCN) over a 3-hour storm-forecast horizon; the data
and scoring layer lives in ``app/data.py``.

Run with:  streamlit run streamlit_app.py
"""
import streamlit as st

st.set_page_config(
    page_title="Geomagnetic Storm Predictor",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded",
)

from app import nav, theme  # noqa: E402 — must follow set_page_config

# Theme has to settle before any page renders, so charts and components are
# built with the palette that matches the surface they land on.
theme.bootstrap()

pages = nav.build()
page = st.navigation(pages, position="sidebar")

with st.sidebar:
    st.markdown(
        '<div class="sp-brand"><span class="sp-mark"></span>Storm Predictor</div>'
        '<div class="sp-brand-sub">3-hour geomagnetic storm forecasting</div>',
        unsafe_allow_html=True,
    )
    theme.appearance_readout()
    st.divider()
    st.caption(
        "Three deployed models — **XGBoost**, **LSTM** and **TCN** — trained on "
        "2010–2021 and scored on a held-out 2022–2024 test period."
    )
    st.caption(
        "The horizon is fixed at **3 hours**: the only one these models were trained "
        "and validated for."
    )

page.run()
