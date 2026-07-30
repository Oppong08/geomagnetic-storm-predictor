"""Reusable presentation components.

These are thin HTML fragments styled by the stylesheet in :mod:`app.theme`.
They exist so the same hero, chip, tile, and badge appear identically on every
page instead of being re-improvised per page.
"""
from __future__ import annotations

from base64 import b64encode
from html import escape

import streamlit as st

from app.theme import STATUS, tokens

# Status tones. Each carries an icon so colour is never the only signal.
_TONES = {
    "quiet": ("✓", STATUS["good"]),
    "elevated": ("▲", STATUS["warning"]),
    "watch": ("⚠", STATUS["serious"]),
    "storm": ("●", STATUS["critical"]),
    "accolade": ("★", None),
    "neutral": ("·", None),
}


def _rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def _html(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


# ------------------------------------------------------------------ hero --
def hero(eyebrow: str, title: str, lede: str, chips: list[str] | None = None) -> None:
    chip_html = f'<div class="sp-chips">{"".join(chips)}</div>' if chips else ""
    _html(
        f'<div class="sp-hero">'
        f'<p class="sp-eyebrow">{escape(eyebrow)}</p>'
        f"<h1>{escape(title)}</h1>"
        f"<p>{lede}</p>"
        f"{chip_html}"
        f"</div>"
    )


def chip(label: str, value: str | None = None, dot: str | None = None) -> str:
    """A single meta chip. `dot` paints a small colour swatch beside the text."""
    dot_html = f'<span class="sp-dot" style="background:{dot}"></span>' if dot else ""
    value_html = f" <b>{escape(value)}</b>" if value else ""
    return f'<span class="sp-chip">{dot_html}{escape(label)}{value_html}</span>'


def page_header(title: str, lede: str) -> None:
    """Compact header for the non-Overview pages."""
    _html(
        f'<div class="sp-hero" style="padding:1.5rem 1.7rem 1.35rem">'
        f'<h1 style="font-size:1.8rem">{escape(title)}</h1>'
        f'<p style="margin-bottom:0">{lede}</p>'
        f"</div>"
    )


# --------------------------------------------------------------- section --
def section(title: str, description: str | None = None) -> None:
    desc = f"<p>{description}</p>" if description else ""
    _html(f'<div class="sp-section"><h2>{escape(title)}</h2>{desc}</div>')


# ----------------------------------------------------------------- badge --
def badge(text: str, tone: str = "neutral") -> str:
    icon, color = _TONES.get(tone, _TONES["neutral"])
    t = tokens()
    color = color or (t["accent"] if tone == "accolade" else t["ink_muted"])
    return (
        f'<span class="sp-badge" style="color:{color};border-color:{_rgba(color, 0.42)};'
        f'background:{_rgba(color, 0.12)}">{icon} {escape(text)}</span>'
    )


# ------------------------------------------------------------- stat tile --
def stat(
    label: str,
    value: str,
    sub: str | None = None,
    spark: list[float] | None = None,
    help_text: str | None = None,
) -> str:
    sub_html = f'<div class="s">{escape(sub)}</div>' if sub else ""
    spark_html = ""
    if spark:
        spark_html = (
            f'<img class="sp-spark" alt="" '
            f'src="data:image/svg+xml;base64,{_sparkline_svg(spark)}" />'
        )
    title = f' title="{escape(help_text)}"' if help_text else ""
    return (
        f'<div class="sp-stat"{title}><div class="k">{escape(label)}</div>'
        f'<div class="v">{escape(value)}</div>{sub_html}{spark_html}</div>'
    )


def stat_row(items) -> None:
    """A wrapping grid of stat tiles.

    A grid rather than ``st.columns`` because fixed columns squeeze each tile
    below its content width on a narrow viewport, and Streamlit answers that by
    truncating the number — the one thing on the tile that must stay readable.
    Items are ``(label, value, sub)`` or ``(label, value, sub, spark, help)``.
    """
    cells = "".join(stat(*item) for item in items)
    _html(f'<div class="sp-kpis">{cells}</div>')


def _sparkline_svg(values: list[float], width: int = 190, height: int = 40) -> str:
    """Bar sparkline, drawn in the accent colour at low emphasis."""
    t = tokens()
    top = max(values) or 1
    n = len(values)
    slot = width / n
    bar_w = max(slot * 0.62, 1.0)
    bars = "".join(
        f'<rect x="{i * slot + (slot - bar_w) / 2:.2f}" '
        f'y="{height - max(v / top * height, 1):.2f}" '
        f'width="{bar_w:.2f}" height="{max(v / top * height, 1):.2f}" rx="1"/>'
        for i, v in enumerate(values)
    )
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'preserveAspectRatio="none"><g fill="{t["accent"]}" opacity="0.62">{bars}</g></svg>'
    )
    return b64encode(svg.encode()).decode()


# ------------------------------------------------------- probability bar --
def probability_bar(
    label: str,
    color: str,
    value: float,
    threshold: float,
    fired: bool | None = None,
) -> str:
    """One model's probability against its own decision threshold.

    The threshold tick is what makes the number readable: 0.62 is a storm call
    for the TCN and a quiet call for the LSTM, because their tuned thresholds
    differ.
    """
    if fired is None:
        fired = value >= threshold
    verdict = "calls storm" if fired else "below threshold"
    return (
        f'<div class="sp-pbar">'
        f'<div class="row"><span class="lbl">'
        f'<span class="sp-dot" style="background:{color}"></span>{escape(label)}</span>'
        f'<span class="val">{value:.2f}</span></div>'
        f'<div class="sp-track">'
        f'<div class="sp-fill" style="width:{max(value, 0.012) * 100:.1f}%;background:{color}"></div>'
        f'<div class="sp-thr" style="left:{threshold * 100:.1f}%"></div>'
        f"</div>"
        f'<div class="foot">threshold {threshold:.2f} · {verdict}</div>'
        f"</div>"
    )


# ------------------------------------------------------------ model card --
def model_card(
    name: str,
    color: str,
    blurb: str,
    metrics: list[tuple[str, str]],
    accolade: str | None = None,
) -> str:
    rows = "".join(f"<dt>{escape(k)}</dt><dd>{escape(v)}</dd>" for k, v in metrics)
    accolade_html = f'<div>{badge(accolade, "accolade")}</div>' if accolade else ""
    return (
        f'<div class="sp-model">'
        f'<div class="name"><span class="sp-dot" style="background:{color}"></span>{escape(name)}</div>'
        f'<div class="arch">{escape(blurb)}</div>'
        f"<dl>{rows}</dl>"
        f"{accolade_html}"
        f"</div>"
    )


# ---------------------------------------------------------------- footer --
def footer() -> None:
    _html(
        '<div class="sp-footer">'
        "<b>Data</b> — NASA OMNI solar-wind parameters, GOES solar-flare events, and "
        "DONKI CME catalogue, aggregated into 3-hour bins (1995–2024). "
        "<b>Split</b> — models trained on 2010–2021, scored on the held-out 2022–2024 "
        "test period. "
        "<b>Scope</b> — the dashboard replays cached predictions over that historical "
        "test period; it is not connected to a live space-weather feed and is not an "
        "operational forecast. "
        "<br>Built for the AI4ALL Ignite program."
        "</div>"
    )


# ------------------------------------------------- magnetosphere diagram --
def magnetosphere_diagram() -> None:
    """Schematic (not to scale) of solar wind and CME material compressing
    Earth's magnetosphere — the compression is what drives the Ap index up, and
    what the models are trying to anticipate.

    Vector rather than a rendered image, so it stays crisp at any size and is
    rebuilt in the active theme's colours. It ships as a data URI because
    Streamlit's markdown sanitiser strips SVG shape elements, keeping only the
    text nodes — inline markup would render as a bare list of labels.
    """
    _html(
        f'<img alt="Schematic: solar wind and CMEs from the Sun compress the sunward '
        f'side of Earth\'s magnetosphere" '
        f'src="data:image/svg+xml;base64,{_magnetosphere_svg()}" '
        f'style="display:block;width:100%;height:auto" />'
    )


def _magnetosphere_svg() -> str:
    t = tokens()
    ink = t["ink_secondary"]
    muted = t["ink_muted"]
    accent = t["accent"]
    critical = STATUS["critical"]

    streamlines = "".join(
        f'<path d="M 96 {y} L 300 {y}" stroke="{muted}" stroke-width="1.6" '
        f'stroke-linecap="round" opacity="0.75" marker-end="url(#sp-arrow)"/>'
        for y in (46, 76, 106, 136, 166)
    )

    svg = f"""
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 212"
     font-family="Inter, system-ui, sans-serif">
  <defs>
    <radialGradient id="sp-sun" cx="35%" cy="35%">
      <stop offset="0%" stop-color="{STATUS['warning']}"/>
      <stop offset="100%" stop-color="{STATUS['serious']}"/>
    </radialGradient>
    <marker id="sp-arrow" viewBox="0 0 10 10" refX="9" refY="5"
            markerWidth="5" markerHeight="5" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="{muted}"/>
    </marker>
  </defs>

  <!-- Sun, bleeding off the left edge -->
  <circle cx="26" cy="106" r="58" fill="url(#sp-sun)" opacity="0.92"/>
  <text x="70" y="196" fill="{ink}" font-size="12">Sun</text>

  <!-- Solar wind / CME transit -->
  {streamlines}
  <text x="150" y="26" fill="{ink}" font-size="12">Solar wind · CME</text>

  <!-- Magnetosphere: compressed on the sunward side, drawn out into a tail -->
  <path d="M 424 26 C 356 62 356 150 424 186" fill="none"
        stroke="{accent}" stroke-width="2.2" stroke-linecap="round"/>
  <path d="M 424 26 C 480 30 540 24 592 18" fill="none"
        stroke="{accent}" stroke-width="2.2" stroke-linecap="round" opacity="0.85"/>
  <path d="M 424 186 C 480 182 540 188 592 194" fill="none"
        stroke="{accent}" stroke-width="2.2" stroke-linecap="round" opacity="0.85"/>
  <text x="470" y="14" fill="{accent}" font-size="12">Magnetosphere</text>

  <!-- Earth -->
  <circle cx="438" cy="106" r="19" fill="{accent}"/>
  <circle cx="438" cy="106" r="19" fill="none" stroke="{t['surface']}" stroke-width="2"/>
  <text x="424" y="150" fill="{ink}" font-size="12">Earth</text>

  <!-- Compression -->
  <path d="M 366 66 C 340 84 340 128 366 146" fill="none" stroke="{critical}"
        stroke-width="3" stroke-linecap="round" opacity="0.9"/>
  <path d="M 318 106 L 352 106" stroke="{critical}" stroke-width="1.6"
        marker-end="url(#sp-arrow)" opacity="0.9"/>
  <text x="196" y="204" fill="{critical}" font-size="11.5" font-weight="600">
    Compression here drives the Ap index up
  </text>
</svg>
"""
    return b64encode(svg.encode()).decode()


def steps(items: list[tuple[str, str]]) -> None:
    """Numbered explainer steps under a diagram."""
    t = tokens()
    cells = "".join(
        f'<div style="flex:1;min-width:150px">'
        f'<div style="display:flex;align-items:center;gap:.45rem;margin-bottom:.28rem">'
        f'<span style="width:19px;height:19px;border-radius:50%;flex:none;'
        f'background:{_rgba(t["accent"], 0.16)};color:{t["accent"]};font-size:.7rem;'
        f'font-weight:700;display:inline-flex;align-items:center;justify-content:center">{i}</span>'
        f'<b style="font-size:.85rem;color:{t["ink"]}">{escape(title)}</b></div>'
        f'<div style="font-size:.8rem;color:{t["ink_secondary"]};line-height:1.5">{escape(body)}</div>'
        f"</div>"
        for i, (title, body) in enumerate(items, start=1)
    )
    _html(f'<div style="display:flex;gap:1.4rem;flex-wrap:wrap;margin-top:.9rem">{cells}</div>')
