"""Altair chart builders.

Every builder takes the active mode's tokens and returns a finished chart, so a
chart can never be rendered with the wrong palette for the surface it lands on.
Charts are drawn on a transparent background and inherit the fill of whatever
card they sit in.

House rules, applied everywhere: one y-axis per chart (never two scales), 2px
lines, >=8px markers, rounded bar ends anchored to the baseline, recessive
grid and axis ink, a legend whenever there are two or more series, and a hover
layer on every chart.
"""
from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd

from app.theme import BODY_FONT, STATUS

_LINE_WIDTH = 2
_POINT_SIZE = 90


def _finish(chart: alt.Chart, t: dict, height: int) -> alt.Chart:
    """Shared chrome: typography, recessive axes, transparent surface."""
    return (
        chart.properties(height=height, width="container")
        .configure_view(stroke=None, fill=None)
        .configure(background="transparent", font=BODY_FONT)
        .configure_axis(
            labelColor=t["ink_muted"],
            titleColor=t["ink_secondary"],
            labelFontSize=11,
            titleFontSize=11,
            titleFontWeight=500,
            titlePadding=12,
            labelPadding=6,
            gridColor=t["grid"],
            gridWidth=1,
            domainColor=t["grid"],
            tickColor=t["grid"],
            tickSize=4,
        )
        .configure_legend(
            labelColor=t["ink_secondary"],
            titleColor=t["ink_muted"],
            labelFontSize=11,
            titleFontSize=10,
            symbolStrokeWidth=3,
            symbolSize=110,
            orient="top",
            direction="horizontal",
            offset=6,
            padding=0,
        )
        .configure_text(font=BODY_FONT)
    )


def _series_scale(t: dict, names: list[str]) -> alt.Scale:
    """Colour follows the model, never its rank — the domain is fixed."""
    return alt.Scale(domain=list(names), range=list(t["series"][: len(names)]))


# --------------------------------------------------------------- Ap timeline --
def ap_timeline(series: pd.DataFrame, t: dict, threshold: float, height: int = 320) -> alt.Chart:
    """Ap over time: the peak envelope shaded behind the typical level.

    Thirty years of daily values is 11,000 points in 900 pixels — a solid block
    of ink that hides the very spikes it is meant to show. Plotting the period
    maximum as an area with the mean as a line keeps both the storms and the
    baseline legible at any zoom, and both are the same measure on one axis.

    `series` needs columns: datetime, ap_max, ap_mean, storms, period.
    """
    ramp = t["sequential"]
    # Steps far apart in the ramp, so the band and the line stay distinguishable
    # for a reader who cannot separate two neighbouring blues.
    envelope, mean_line = (ramp[2], ramp[6]) if t["mode"] == "dark" else (ramp[1], ramp[5])

    base = alt.Chart(series)
    hover = alt.selection_point(
        fields=["datetime"], nearest=True, on="pointermove", empty=False, clear="pointerout"
    )
    x = alt.X("datetime:T", title=None, axis=alt.Axis(format="%Y", tickCount=8))

    # Both layers encode a constant series name so Vega-Lite builds one shared
    # legend — identity is never left to colour alone.
    series_scale = alt.Scale(
        domain=["Peak in period", "Typical level"], range=[envelope, mean_line]
    )
    area = base.transform_calculate(
        series='"Peak in period"'
    ).mark_area(opacity=0.45, interpolate="monotone").encode(
        x=x,
        y=alt.Y("ap_max:Q", title="Ap index", scale=alt.Scale(nice=True)),
        color=alt.Color("series:N", scale=series_scale, title=None),
    )
    line = base.transform_calculate(
        series='"Typical level"'
    ).mark_line(strokeWidth=_LINE_WIDTH, interpolate="monotone").encode(
        x=x,
        y="ap_mean:Q",
        color=alt.Color("series:N", scale=series_scale, title=None),
    )

    rule_thr = alt.Chart(pd.DataFrame({"y": [threshold]})).mark_rule(
        color=STATUS["critical"], strokeDash=[5, 4], strokeWidth=1.5, opacity=0.9
    ).encode(y="y:Q")

    label_thr = alt.Chart(
        pd.DataFrame({"y": [threshold], "label": [f"Storm threshold · Ap {threshold:.0f}"]})
    ).mark_text(
        align="left", baseline="bottom", dx=6, dy=-6, fontSize=10.5, fontWeight=600,
        color=STATUS["critical"],
    ).encode(y="y:Q", text="label:N", x=alt.value(4))

    crosshair = base.mark_rule(color=t["ink_muted"], strokeWidth=1).encode(
        x="datetime:T",
        opacity=alt.condition(hover, alt.value(0.7), alt.value(0)),
        tooltip=[
            alt.Tooltip("period:N", title="Period"),
            alt.Tooltip("ap_max:Q", title="Peak Ap", format=".0f"),
            alt.Tooltip("ap_mean:Q", title="Typical Ap", format=".1f"),
            alt.Tooltip("storms:Q", title="Storm bins", format="d"),
        ],
    ).add_params(hover)

    dot = base.mark_point(
        size=_POINT_SIZE, filled=True, color=mean_line,
        stroke=t["surface"], strokeWidth=2,
    ).encode(
        x="datetime:T", y="ap_mean:Q",
        opacity=alt.condition(hover, alt.value(1), alt.value(0)),
    )

    layered = alt.layer(area, line, rule_thr, label_thr, crosshair, dot).add_params(
        alt.selection_interval(bind="scales", encodings=["x"])
    )
    return _finish(layered, t, height)


# --------------------------------------------------------- storms per year --
def storms_per_year(yearly: pd.DataFrame, t: dict, height: int = 260) -> alt.Chart:
    """Storm bins per calendar year. `yearly` needs columns: year, storms."""
    peaks = {2003, 2015, 2024}
    data = yearly.assign(peak=yearly["year"].isin(peaks))
    bar_color = t["sequential"][4]

    # One colour for every bar: the solar maxima are called out with dated
    # labels, so shading them a second blue would encode the same fact twice —
    # and two steps of one sequential ramp is not a categorical distinction.
    bars = alt.Chart(data).mark_bar(
        cornerRadiusTopLeft=4, cornerRadiusTopRight=4, color=bar_color,
    ).encode(
        x=alt.X(
            "year:O", title=None,
            scale=alt.Scale(paddingInner=0.28),
            axis=alt.Axis(labelAngle=0, values=list(range(1995, 2025, 5))),
        ),
        y=alt.Y("storms:Q", title="Storm bins", scale=alt.Scale(nice=True)),
        tooltip=[
            alt.Tooltip("year:O", title="Year"),
            alt.Tooltip("storms:Q", title="Storm bins", format="d"),
        ],
    )

    labels = alt.Chart(data[data["peak"]]).mark_text(
        dy=-9, fontSize=10.5, fontWeight=600, color=t["ink_secondary"],
    ).encode(x=alt.X("year:O", scale=alt.Scale(paddingInner=0.28)), y="storms:Q", text="year:O")

    return _finish(alt.layer(bars, labels), t, height)


# ------------------------------------------------------- confusion matrix --
def confusion_matrix(cm: np.ndarray, t: dict, height: int = 210) -> alt.Chart:
    """2x2 outcome grid on the sequential ramp.

    The ramp runs light->dark on the light surface and dark->light on the dark
    one, so the cell that needs inverted ink is the opposite end in each mode.
    """
    labels = ["No storm", "Storm"]
    rows = [
        {
            "actual": labels[i],
            "predicted": labels[j],
            "count": int(cm[i, j]),
            "share": float(cm[i, j] / cm.max()),
            "outcome": _outcome_name(i, j),
        }
        for i in range(2)
        for j in range(2)
    ]
    data = pd.DataFrame(rows)

    cells = alt.Chart(data).mark_rect(cornerRadius=4, stroke=t["surface"], strokeWidth=2).encode(
        x=alt.X("predicted:N", title="Predicted", sort=labels, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("actual:N", title="Actual", sort=labels),
        color=alt.Color(
            "share:Q",
            scale=alt.Scale(range=t["sequential"], domain=[0, 1]),
            legend=None,
        ),
        tooltip=[
            alt.Tooltip("outcome:N", title="Outcome"),
            alt.Tooltip("count:Q", title="Bins", format=","),
        ],
    )

    if t["mode"] == "dark":
        # High share = pale cell, so its ink must go dark.
        heavy_ink, light_ink = "#0b0b0b", t["ink"]
    else:
        # High share = deep blue cell, so its ink must go white.
        heavy_ink, light_ink = "#ffffff", t["ink"]

    text = alt.Chart(data).mark_text(fontSize=13, fontWeight=600).encode(
        x=alt.X("predicted:N", sort=labels),
        y=alt.Y("actual:N", sort=labels),
        text=alt.Text("count:Q", format=","),
        color=alt.condition(
            alt.datum.share > 0.55, alt.value(heavy_ink), alt.value(light_ink)
        ),
    )

    return _finish(alt.layer(cells, text), t, height)


def _outcome_name(actual: int, predicted: int) -> str:
    return {
        (0, 0): "Correct quiet call",
        (0, 1): "False alarm",
        (1, 0): "Missed storm",
        (1, 1): "Caught storm",
    }[(actual, predicted)]


# ----------------------------------------------------- precision–recall --
def pr_curves(curves: pd.DataFrame, points: pd.DataFrame, t: dict, height: int = 320) -> alt.Chart:
    """Overlaid PR curves. `curves`: model, recall, precision. `points`: model,
    recall, precision, kind ("tuned" | "selected")."""
    names = list(dict.fromkeys(curves["model"]))
    scale = _series_scale(t, names)

    lines = alt.Chart(curves).mark_line(strokeWidth=_LINE_WIDTH).encode(
        x=alt.X("recall:Q", title="Recall", scale=alt.Scale(domain=[0, 1])),
        y=alt.Y("precision:Q", title="Precision", scale=alt.Scale(domain=[0, 1.02])),
        color=alt.Color("model:N", scale=scale, title=None),
    )

    # Note: none of these layers may set legend=None. Vega-Lite resolves the
    # colour scale across layers, and a single null legend suppresses the shared
    # one — which would leave three series identified by colour alone.
    marker_tooltip = [
        alt.Tooltip("model:N", title="Model"),
        alt.Tooltip("label:N", title="Point"),
        alt.Tooltip("recall:Q", title="Recall", format=".2f"),
        alt.Tooltip("precision:Q", title="Precision", format=".2f"),
    ]

    # A ring rather than a filled shape: at the default slider position the tuned
    # and selected points sit on top of each other, and a ring around the dot
    # still reads as two things.
    tuned = alt.Chart(points[points["kind"] == "tuned"]).mark_point(
        size=300, filled=False, strokeWidth=2,
    ).encode(
        x="recall:Q", y="precision:Q",
        color=alt.Color("model:N", scale=scale, title=None),
        tooltip=marker_tooltip,
    )

    selected = alt.Chart(points[points["kind"] == "selected"]).mark_point(
        size=130, filled=True, stroke=t["surface"], strokeWidth=2,
    ).encode(
        x="recall:Q", y="precision:Q",
        color=alt.Color("model:N", scale=scale, title=None),
        tooltip=marker_tooltip,
    )

    return _finish(alt.layer(lines, tuned, selected), t, height)


# ------------------------------------------------------ forecast timeline --
def forecast_timeline(
    view: pd.DataFrame, t: dict, threshold: float, series_color: str, height: int = 300
) -> alt.Chart:
    """Predicted storm probability through a window, with real storms marked.

    `view` needs columns: datetime, storm_probability, is_storm, ap_now.
    """
    fill = alt.Gradient(
        gradient="linear",
        stops=[
            alt.GradientStop(color=series_color, offset=1),
            alt.GradientStop(color=series_color + "00", offset=0),
        ],
        x1=1, x2=1, y1=1, y2=0,
    )

    base = alt.Chart(view)
    hover = alt.selection_point(
        fields=["datetime"], nearest=True, on="pointermove", empty=False, clear="pointerout"
    )

    storms = view[view["is_storm"]]
    storm_rules = alt.Chart(storms).mark_rule(
        color=STATUS["critical"], strokeWidth=2, opacity=0.34,
    ).encode(x="datetime:T")

    area = base.mark_area(
        line={"color": series_color, "strokeWidth": _LINE_WIDTH},
        color=fill,
        interpolate="monotone",
    ).encode(
        x=alt.X(
            "datetime:T", title=None,
            axis=alt.Axis(format="%d %b", labelOverlap="greedy", tickCount=8),
        ),
        y=alt.Y(
            "storm_probability:Q",
            title="Predicted P(storm)",
            scale=alt.Scale(domain=[0, 1]),
            axis=alt.Axis(format=".1f"),
        ),
    )

    thr_rule = alt.Chart(pd.DataFrame({"y": [threshold]})).mark_rule(
        color=t["ink_muted"], strokeDash=[5, 4], strokeWidth=1.5,
    ).encode(y="y:Q")

    thr_label = alt.Chart(
        pd.DataFrame({"y": [threshold], "label": [f"Decision threshold {threshold:.2f}"]})
    ).mark_text(
        align="left", baseline="bottom", dy=-4, fontSize=10.5, color=t["ink_muted"],
    ).encode(y="y:Q", text="label:N", x=alt.value(6))

    crosshair = base.mark_rule(color=t["ink_muted"], strokeWidth=1).encode(
        x="datetime:T",
        opacity=alt.condition(hover, alt.value(0.7), alt.value(0)),
        tooltip=[
            alt.Tooltip("datetime:T", title="Bin", format="%d %b %Y %H:%M"),
            alt.Tooltip("storm_probability:Q", title="P(storm)", format=".2f"),
            alt.Tooltip("ap_now:Q", title="Ap (now)", format=".0f"),
            alt.Tooltip("outcome:N", title="Actual"),
        ],
    ).add_params(hover)

    dot = base.mark_point(
        size=_POINT_SIZE, filled=True, color=series_color,
        stroke=t["surface"], strokeWidth=2,
    ).encode(
        x="datetime:T", y="storm_probability:Q",
        opacity=alt.condition(hover, alt.value(1), alt.value(0)),
    )

    layered = alt.layer(storm_rules, area, thr_rule, thr_label, crosshair, dot).add_params(
        alt.selection_interval(bind="scales", encodings=["x"])
    )
    return _finish(layered, t, height)


# ------------------------------------------------------ event conditions --
def event_conditions(event: pd.DataFrame, t: dict, height: int = 220) -> alt.Chart:
    """Ap through a storm event, with each model's probability alongside.

    Two measures share one 0-1 axis by plotting Ap as a share of the event peak,
    which keeps this to a single y-scale rather than a dual axis.
    """
    long = event.melt(
        id_vars=["datetime"],
        value_vars=["proba_XGBoost", "proba_LSTM", "proba_TCN"],
        var_name="model",
        value_name="probability",
    )
    long["model"] = long["model"].str.replace("proba_", "", regex=False)
    names = ["XGBoost", "LSTM", "TCN"]

    # The observed-Ap band joins the same colour scale in neutral grey, so it
    # gets a legend entry without pretending to be a fourth model.
    band_label = "Observed Ap (share of peak)"
    scale = alt.Scale(
        domain=[*names, band_label],
        range=[*t["series"][:3], t["ink_muted"]],
    )
    x = alt.X(
        "datetime:T", title=None,
        axis=alt.Axis(format="%d %b %H:%M", labelOverlap="greedy", tickCount=6),
    )

    ap = alt.Chart(event).transform_calculate(
        band=f'"{band_label}"'
    ).mark_area(opacity=0.22).encode(
        x=x,
        y=alt.Y("ap_share:Q", title="0–1 scale", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("band:N", scale=scale, title=None),
        tooltip=[
            alt.Tooltip("datetime:T", title="Bin", format="%d %b %Y %H:%M"),
            alt.Tooltip("ap_now:Q", title="Ap (now)", format=".0f"),
        ],
    )

    lines = alt.Chart(long).mark_line(
        strokeWidth=_LINE_WIDTH, point=alt.OverlayMarkDef(size=45, filled=True),
    ).encode(
        x=x,
        y=alt.Y("probability:Q", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("model:N", scale=scale, title=None),
        tooltip=[
            alt.Tooltip("datetime:T", title="Bin", format="%d %b %Y %H:%M"),
            alt.Tooltip("model:N", title="Model"),
            alt.Tooltip("probability:Q", title="P(storm)", format=".2f"),
        ],
    )

    return _finish(alt.layer(ap, lines), t, height)
