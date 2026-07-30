"""Overview — what the system does, how it is doing, and where to look next."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from app import charts, ui
from app.data import (
    HORIZON,
    MODEL_BLURBS,
    MODEL_NAMES,
    SPLIT_DATE,
    TARGET,
    load_data,
    load_metadata,
    operating_threshold,
    score_all,
    storm_threshold,
)
from app.theme import tokens

_RANGES = {"5y": 5, "10y": 10, "20y": 20, "All": None}


def render() -> None:
    t = tokens()
    df = load_data()
    threshold = storm_threshold(df)
    colors = dict(zip(MODEL_NAMES, t["series"]))

    _hero(df, threshold, colors)
    _latest_bin(colors)
    _headline_stats(df)
    _ap_history(df, t, threshold)
    _cycle_and_physics(df, t)
    _model_lineup(colors)
    _next_steps()
    ui.footer()


# ------------------------------------------------------------------ hero --
def _hero(df: pd.DataFrame, threshold: float, colors: dict) -> None:
    chips = [
        ui.chip("Horizon", f"{HORIZON} hours"),
        ui.chip("Storm =", f"Ap ≥ {threshold:.0f}"),
        ui.chip("Coverage", f"{df['datetime'].min():%Y}–{df['datetime'].max():%Y} · {len(df):,} bins"),
    ] + [ui.chip(name, dot=colors[name]) for name in MODEL_NAMES]

    ui.hero(
        eyebrow="Space weather · rare-event forecasting",
        title="Geomagnetic Storm Predictor",
        lede=(
            "The Sun streams charged particles at Earth continuously, and occasionally "
            "hurls far larger eruptions — <b>coronal mass ejections</b>. When that material "
            "hits Earth's magnetic field it compresses and disturbs it, and GPS, radio, "
            "aviation and power grids feel it. Three machine-learning models read the "
            "current solar-wind, flare and CME conditions and forecast whether the "
            "<b>Ap index</b> will cross the storm threshold three hours from now."
        ),
        chips=chips,
    )


# ------------------------------------------------------- latest scored bin --
def _latest_bin(colors: dict) -> None:
    """The most recent bin any model scored, framed as a forecast readout.

    Deliberately labelled as the last bin *on record* — these are cached
    predictions replayed over a historical test period, not a live feed, and the
    page should not imply otherwise.
    """
    scored = score_all()
    row = scored.iloc[-1]
    thresholds = {name: operating_threshold(name) for name in MODEL_NAMES}
    fired = {name: row[f"proba_{name}"] >= thresholds[name] for name in MODEL_NAMES}
    votes = sum(fired.values())
    actual_storm = bool(row[TARGET] == 1)

    if votes == 0:
        tone, verdict = "quiet", "Quiet"
        reading = "No model called a storm for this bin."
    elif votes == len(MODEL_NAMES):
        tone, verdict = "storm", "Storm watch"
        reading = "All three models crossed their decision thresholds."
    else:
        tone, verdict = "elevated", "Elevated"
        reading = f"{votes} of {len(MODEL_NAMES)} models crossed their decision thresholds."

    ui.section(
        "Latest bin on record",
        "The final 3-hour bin of the held-out test period, read the way the dashboard "
        "would read a live one. Each model is scored against its own tuned threshold.",
    )

    with st.container(border=True):
        head, body = st.columns([1, 1.25], gap="large", vertical_alignment="top")
        with head:
            st.markdown(
                f'<div style="font-size:.72rem;letter-spacing:.09em;text-transform:uppercase;'
                f'color:var(--sp-ink-muted);margin-bottom:.5rem">'
                f"{row['datetime']:%d %b %Y · %H:%M} UTC</div>{ui.badge(verdict, tone)}",
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<p style="margin:.85rem 0 0;color:var(--sp-ink-2);font-size:.9rem;'
                f'line-height:1.55">{reading}</p>',
                unsafe_allow_html=True,
            )
            outcome = "a storm" if actual_storm else "no storm"
            st.markdown(
                f'<p style="margin:.55rem 0 0;color:var(--sp-ink-muted);font-size:.82rem">'
                f"What actually happened three hours later: <b>{outcome}</b> "
                f"(Ap {row['ap_target_3h']:.0f}).</p>",
                unsafe_allow_html=True,
            )
        with body:
            st.markdown(
                "".join(
                    ui.probability_bar(
                        name, colors[name], float(row[f"proba_{name}"]),
                        thresholds[name], fired[name],
                    )
                    for name in MODEL_NAMES
                ),
                unsafe_allow_html=True,
            )

    st.caption(
        "Predictions are replayed from cached test-period scores — this dashboard is not "
        "wired to a live space-weather feed."
    )


# ------------------------------------------------------------ stat tiles --
def _headline_stats(df: pd.DataFrame) -> None:
    yearly = df.groupby(df["datetime"].dt.year).agg(storms=(TARGET, "sum"))
    peak_row = df.loc[df["ap_now"].idxmax()]
    storm_bins = int(df[TARGET].sum())

    storms_by_year = yearly["storms"].tolist()
    peak_by_year = (
        df.groupby(df["datetime"].dt.year)["ap_now"].max().tolist()
    )

    # No sparkline on "Time bins": bins per year is a near-constant ~2,920, so a
    # chart of it would show nothing but leap years and the partial first year.
    ui.stat_row(
        [
            (
                "Time bins", f"{len(df):,}",
                f"{df['datetime'].min():%Y}–{df['datetime'].max():%Y}", None,
                "Every 3-hour interval in the dataset.",
            ),
            (
                "Storm bins", f"{storm_bins:,}", "per year", storms_by_year,
                "Bins where the Ap index 3 hours later crossed the storm threshold.",
            ),
            (
                "Storm rate", f"{df[TARGET].mean() * 100:.2f}%", "of all bins", None,
                "Storms are rare, which is exactly what makes this hard: a model that "
                "never calls one is still right 98% of the time.",
            ),
            (
                "Peak Ap on record", f"{peak_row['ap_now']:.0f}",
                f"{peak_row['datetime']:%b %Y} · yearly peaks", peak_by_year,
                "The most disturbed 3-hour bin in the dataset.",
            ),
        ]
    )


# ------------------------------------------------------------ Ap history --
def _ap_history(df: pd.DataFrame, t: dict, threshold: float) -> None:
    ui.section(
        "Ap index through three solar cycles",
        "Geomagnetic disturbance since 1995 — the shaded band is the worst reading in "
        "each period, the line is the typical level. Everything above the dashed line "
        "is what the models exist to catch. Drag to pan, scroll to zoom.",
    )

    picked = st.segmented_control(
        "Range", options=list(_RANGES), default="All", key="ap_range",
        label_visibility="collapsed",
    ) or "All"

    years = _RANGES[picked]
    window = df
    if years:
        cutoff = df["datetime"].max() - pd.DateOffset(years=years)
        window = df[df["datetime"] >= cutoff]

    with st.container(border=True):
        st.altair_chart(charts.ap_timeline(_resample_ap(window, years), t, threshold), theme=None)


def _resample_ap(window: pd.DataFrame, years: int | None) -> pd.DataFrame:
    """Bucket Ap to roughly one point per pixel-worth of chart.

    Daily resolution over thirty years is unreadable, and monthly over one year
    throws away every storm; the rule scales the bucket to the span.
    """
    if years is not None and years <= 5:
        freq, label = "W", "Week of %d %b %Y"
    elif years is not None and years <= 10:
        freq, label = "MS", "%B %Y"
    else:
        freq, label = "QS", "Q%q %Y"

    out = (
        window.set_index("datetime")
        .resample(freq)
        .agg(ap_max=("ap_now", "max"), ap_mean=("ap_now", "mean"), storms=(TARGET, "sum"))
        .dropna(subset=["ap_max"])
        .reset_index()
    )
    if freq == "QS":
        out["period"] = out["datetime"].dt.to_period("Q").astype(str)
    else:
        out["period"] = out["datetime"].dt.strftime(label)
    return out


# ------------------------------------------------- solar cycle + physics --
def _cycle_and_physics(df: pd.DataFrame, t: dict) -> None:
    left, right = st.columns([1.05, 1], gap="large")

    with left:
        ui.section(
            "Storms follow the solar cycle",
            "Storm counts rise and fall on the Sun's ~11-year rhythm — maxima around "
            "2003, 2015 and 2024, quiet minima around 2009 and 2019.",
        )
        yearly = (
            df.groupby(df["datetime"].dt.year)[TARGET].sum()
            .rename("storms").rename_axis("year").reset_index()
        )
        with st.container(border=True):
            st.altair_chart(charts.storms_per_year(yearly, t), theme=None)

    with right:
        ui.section(
            "How a storm forms",
            "Not to scale — a schematic of the chain the features are trying to capture.",
        )
        with st.container(border=True):
            ui.magnetosphere_diagram()
            ui.steps(
                [
                    ("Eruption", "A flare or CME launches charged material off the Sun."),
                    ("Transit", "It crosses to Earth in hours to days, carrying its own magnetic field."),
                    ("Compression", "A southward field couples to Earth's, compressing it and driving Ap up."),
                ]
            )


# ---------------------------------------------------------- model line-up --
def _model_lineup(colors: dict) -> None:
    ui.section(
        "The three models",
        f"All trained on 2010–2021 and scored on the same held-out {SPLIT_DATE[:4]}–2024 "
        "test period, so these numbers are directly comparable. Each is read straight "
        "from its deployed metadata file.",
    )

    meta = {name: load_metadata(name) for name in MODEL_NAMES}
    best = {
        "F1": max(MODEL_NAMES, key=lambda n: meta[n]["storm_f1"]),
        "precision": max(MODEL_NAMES, key=lambda n: meta[n]["storm_precision"]),
        "recall": max(MODEL_NAMES, key=lambda n: meta[n]["storm_recall"]),
        "PR AUC": max(MODEL_NAMES, key=lambda n: meta[n]["test_pr_auc"]),
    }

    cols = st.columns(len(MODEL_NAMES), gap="medium")
    for col, name in zip(cols, MODEL_NAMES):
        m = meta[name]
        wins = [label for label, winner in best.items() if winner == name]
        accolade = "Best " + " & ".join(wins) if wins else None
        with col:
            st.markdown(
                ui.model_card(
                    name,
                    colors[name],
                    MODEL_BLURBS[name],
                    [
                        ("PR AUC", f"{m['test_pr_auc']:.3f}"),
                        ("ROC AUC", f"{m['test_roc_auc']:.3f}"),
                        ("Precision", f"{m['storm_precision']:.2f}"),
                        ("Recall", f"{m['storm_recall']:.2f}"),
                        ("F1", f"{m['storm_f1']:.2f}"),
                        ("Threshold", f"{m['operating_threshold']:.2f}"),
                    ],
                    accolade,
                ),
                unsafe_allow_html=True,
            )

    st.caption(
        "PR AUC is the fairest single summary here: with storms at under 2% of bins, "
        "ROC AUC flatters every model. Precision is how many storm calls were real; "
        "recall is how many real storms were caught."
    )


# ------------------------------------------------------------ next steps --
def _next_steps() -> None:
    from app import nav

    ui.section("Where to go next")
    links = [
        ("performance", "Move each decision threshold and watch precision trade against recall."),
        ("forecast", "Step through the test period and compare predictions with what happened."),
        ("storms", "Open a real storm event and see the conditions at its peak."),
    ]
    for col, (key, blurb) in zip(st.columns(3, gap="medium"), links):
        with col, st.container(border=True):
            st.page_link(nav.page(key), width="stretch")
            st.caption(blurb)
