"""Design tokens, global CSS, and the light/dark switch.

The palette is not decorative: series colours are slots 1-3 of a categorical
palette validated for colour-vision-deficiency separation and contrast in both
modes, and the status colours are reserved — a status hue never doubles as a
series hue. Every colour a chart or component uses comes from :func:`tokens`,
so the two modes can never drift apart.
"""
from __future__ import annotations

import streamlit as st

MODES = ("light", "dark")

# Reserved status colours — identical in both modes, and never used for a series.
# Anything painted with these also ships an icon and a label, so the colour is
# never the only carrier of meaning.
STATUS = {
    "good": "#0ca30c",
    "warning": "#fab219",
    "serious": "#ec835a",
    "critical": "#d03b3b",
}

_LIGHT = {
    "mode": "light",
    # Categorical slots 1-3 — XGBoost, LSTM, TCN.
    "series": ("#2a78d6", "#eb6834", "#1baf7a"),
    # Sequential ramp, light -> dark, for magnitude encodings.
    "sequential": ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
    "plane": "#f7f8fa",       # page background
    "surface": "#ffffff",     # cards and chart surfaces
    "surface_alt": "#f0f1f4", # recessed wells
    "ink": "#0b0b0b",
    "ink_secondary": "#52514e",
    "ink_muted": "#898781",
    "grid": "#e1e0d9",
    "border": "rgba(11,11,11,0.10)",
    "hairline": "#e1e0d9",
    "accent": "#2a78d6",
    "accent_2": "#1baf7a",
    "aurora": "linear-gradient(100deg, rgba(42,120,214,0.22) 0%, rgba(27,175,122,0.20) 52%, rgba(42,120,214,0.05) 100%)",
    "chip_bg": "rgba(11,11,11,0.045)",
}

_DARK = {
    "mode": "dark",
    "series": ("#3987e5", "#d95926", "#199e70"),
    # On a dark surface the ramp runs dark -> light and stops at the ordinal
    # floor (step 600) so even the lowest step stays visible.
    "sequential": ["#184f95", "#1c5cab", "#256abf", "#2a78d6", "#3987e5", "#5598e7", "#86b6ef"],
    "plane": "#0b0c10",
    "surface": "#12131a",
    "surface_alt": "#191b24",
    "ink": "#ffffff",
    "ink_secondary": "#c3c2b7",
    "ink_muted": "#898781",
    "grid": "#262832",
    "border": "rgba(255,255,255,0.10)",
    "hairline": "#262832",
    "accent": "#3987e5",
    "accent_2": "#199e70",
    "aurora": "linear-gradient(100deg, rgba(57,135,229,0.26) 0%, rgba(25,158,112,0.20) 52%, rgba(57,135,229,0.02) 100%)",
    "chip_bg": "rgba(255,255,255,0.06)",
}

_TOKENS = {"light": _LIGHT, "dark": _DARK}

HEADING_FONT = "'Space Grotesk', Inter, system-ui, -apple-system, 'Segoe UI', sans-serif"
BODY_FONT = "Inter, system-ui, -apple-system, 'Segoe UI', sans-serif"


DEFAULT_MODE = "dark"


def current_mode() -> str:
    """Whichever theme Streamlit is actually rendering, this run.

    The app follows the client rather than trying to lead it. Streamlit decides
    the theme from the viewer's Appearance setting (which defaults to their OS
    preference) and reports the result here; ``config.toml`` supplies a designed
    palette for each. Reading it back is what keeps our components, charts and
    SVG on the same surface as Streamlit's own chrome — a mismatch would put
    dark cards on a light page.
    """
    context_theme = getattr(st.context, "theme", None)
    reported = context_theme.get("type") if context_theme else None
    return reported if reported in MODES else DEFAULT_MODE


def tokens(mode: str | None = None) -> dict:
    """The design tokens for a mode (defaults to the active one)."""
    return _TOKENS[mode or current_mode()]


def bootstrap() -> str:
    """Inject the stylesheet for the active theme. Returns the mode."""
    mode = current_mode()
    _inject_css(tokens(mode))
    return mode


def appearance_readout() -> None:
    """Show the active theme and where to change it. Render in the sidebar.

    Streamlit owns the theme switch: it lives in the ⋮ menu under Settings →
    Appearance, and there is no supported API for a page to set it. Rather than
    ship a control that only half works, this points at the real one.
    """
    mode = current_mode()
    icon, label = ("☀", "Light") if mode == "light" else ("☾", "Dark")
    st.markdown(
        f'<div class="sp-appearance">'
        f'<span class="sp-appearance-now">{icon} {label} theme</span>'
        f'<span class="sp-appearance-hint">Switch in ⋮ → Settings → Appearance</span>'
        f"</div>",
        unsafe_allow_html=True,
    )


def _rgba_css(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def _inject_css(t: dict) -> None:
    """Component styling that Streamlit's theme tokens can't express.

    Kept deliberately small: layout and colour come from the theme config
    wherever possible, and this only adds the pieces the design needs that have
    no config equivalent — the aurora hero band, chips, stat tiles, and the
    typographic scale for section headers.
    """
    st.markdown(
        f"""
        <style>
          :root {{
            --sp-plane: {t['plane']};
            --sp-surface: {t['surface']};
            --sp-surface-alt: {t['surface_alt']};
            --sp-ink: {t['ink']};
            --sp-ink-2: {t['ink_secondary']};
            --sp-ink-muted: {t['ink_muted']};
            --sp-border: {t['border']};
            --sp-hairline: {t['hairline']};
            --sp-accent: {t['accent']};
            --sp-accent-2: {t['accent_2']};
            --sp-chip-bg: {t['chip_bg']};
            --sp-good: {STATUS['good']};
            --sp-warning: {STATUS['warning']};
            --sp-critical: {STATUS['critical']};
            --sp-heading-font: {HEADING_FONT};
            --sp-body-font: {BODY_FONT};
          }}

          /* Tighten the default page padding; the hero supplies its own. */
          .block-container {{ padding-top: 2.4rem; padding-bottom: 4rem; max-width: 1320px; }}

          /* ---------------------------------------------------- hero band -- */
          .sp-hero {{
            position: relative;
            border: 1px solid var(--sp-border);
            border-radius: 1rem;
            background: {t['aurora']}, var(--sp-surface);
            padding: 1.9rem 2rem 1.6rem;
            margin-bottom: 1.5rem;
            overflow: hidden;
          }}
          .sp-hero::after {{
            content: "";
            position: absolute; inset: 0 0 auto 0; height: 2px;
            background: linear-gradient(90deg, var(--sp-accent), var(--sp-accent-2), transparent);
          }}
          .sp-hero h1 {{
            font-family: var(--sp-heading-font);
            font-weight: 600; font-size: 2.35rem; line-height: 1.1;
            letter-spacing: -0.02em; margin: 0 0 0.5rem; color: var(--sp-ink);
          }}
          .sp-hero p {{
            margin: 0 0 1.05rem; max-width: 62ch;
            color: var(--sp-ink-2); font-size: 1.02rem; line-height: 1.55;
          }}
          .sp-eyebrow {{
            font-family: var(--sp-heading-font);
            font-size: 0.72rem; font-weight: 600; letter-spacing: 0.14em;
            text-transform: uppercase; color: var(--sp-accent); margin: 0 0 0.6rem;
          }}

          /* -------------------------------------------------------- chips -- */
          .sp-chips {{ display: flex; flex-wrap: wrap; gap: 0.45rem; }}
          .sp-chip {{
            display: inline-flex; align-items: center; gap: 0.4rem;
            background: var(--sp-chip-bg);
            border: 1px solid var(--sp-border);
            border-radius: 999px;
            padding: 0.3rem 0.7rem;
            font-size: 0.78rem; color: var(--sp-ink-2); white-space: nowrap;
          }}
          .sp-chip b {{ color: var(--sp-ink); font-weight: 600; }}
          .sp-dot {{ width: 8px; height: 8px; border-radius: 50%; flex: none; }}

          /* ------------------------------------------------ section header -- */
          .sp-section {{ margin: 2.1rem 0 0.9rem; }}
          .sp-section h2 {{
            font-family: var(--sp-heading-font);
            font-size: 1.28rem; font-weight: 600; letter-spacing: -0.01em;
            margin: 0; color: var(--sp-ink);
          }}
          .sp-section p {{
            margin: 0.28rem 0 0; color: var(--sp-ink-2);
            font-size: 0.9rem; line-height: 1.5; max-width: 76ch;
          }}

          /* -------------------------------------------------- status badge -- */
          .sp-badge {{
            display: inline-flex; align-items: center; gap: 0.45rem;
            border-radius: 999px; padding: 0.32rem 0.8rem;
            font-size: 0.82rem; font-weight: 600; border: 1px solid;
          }}

          /* ---------------------------------------------------- stat tile -- */
          /* auto-fit so tiles reflow instead of squeezing their numbers into
             an ellipsis on a narrow viewport. */
          .sp-kpis {{
            display: grid; gap: 0.7rem;
            grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
          }}
          .sp-stat {{
            border: 1px solid var(--sp-border); border-radius: 0.8rem;
            background: var(--sp-surface); padding: 0.95rem 1.05rem;
            height: 100%; display: flex; flex-direction: column;
          }}
          .sp-spark {{
            display: block; width: 100%; height: 40px;
            margin-top: auto; padding-top: 0.7rem;
          }}
          .sp-stat .k {{
            font-size: 0.7rem; font-weight: 600; letter-spacing: 0.09em;
            text-transform: uppercase; color: var(--sp-ink-muted); margin-bottom: 0.35rem;
          }}
          .sp-stat .v {{
            font-family: var(--sp-heading-font);
            font-size: 1.55rem; font-weight: 600; line-height: 1.15; color: var(--sp-ink);
          }}
          .sp-stat .s {{ font-size: 0.78rem; color: var(--sp-ink-2); margin-top: 0.28rem; }}

          /* --------------------------------------------------- model card -- */
          .sp-model {{
            border: 1px solid var(--sp-border); border-radius: 0.9rem;
            background: var(--sp-surface); padding: 1.1rem 1.15rem 1rem;
            height: 100%; display: flex; flex-direction: column; gap: 0.7rem;
          }}
          .sp-model .name {{
            font-family: var(--sp-heading-font);
            font-size: 1.05rem; font-weight: 600; color: var(--sp-ink);
            display: flex; align-items: center; gap: 0.5rem;
          }}
          .sp-model .arch {{ font-size: 0.83rem; color: var(--sp-ink-2); line-height: 1.5; }}
          .sp-model dl {{
            display: grid; grid-template-columns: 1fr auto; gap: 0.3rem 1rem;
            margin: 0; font-size: 0.84rem; font-variant-numeric: tabular-nums;
          }}
          .sp-model dt {{ color: var(--sp-ink-2); }}
          .sp-model dd {{ margin: 0; text-align: right; color: var(--sp-ink); font-weight: 600; }}

          /* ------------------------------------------------ probability bar -- */
          .sp-pbar {{ margin-bottom: 0.85rem; }}
          .sp-pbar .row {{
            display: flex; justify-content: space-between; align-items: baseline;
            font-size: 0.85rem; margin-bottom: 0.32rem;
          }}
          .sp-pbar .row .lbl {{
            display: flex; align-items: center; gap: 0.45rem; color: var(--sp-ink);
          }}
          .sp-pbar .row .val {{
            font-variant-numeric: tabular-nums; font-weight: 600; color: var(--sp-ink);
          }}
          .sp-track {{
            position: relative; height: 10px; border-radius: 999px;
            background: var(--sp-surface-alt); border: 1px solid var(--sp-border);
            overflow: visible;
          }}
          .sp-fill {{ height: 100%; border-radius: 999px; }}
          .sp-thr {{
            position: absolute; top: -4px; bottom: -4px; width: 2px;
            background: var(--sp-ink-muted); border-radius: 2px;
          }}
          .sp-pbar .foot {{ font-size: 0.74rem; color: var(--sp-ink-muted); margin-top: 0.3rem; }}

          /* ------------------------------------------------------- footer -- */
          .sp-footer {{
            border-top: 1px solid var(--sp-hairline);
            margin-top: 3rem; padding-top: 1.1rem;
            font-size: 0.8rem; color: var(--sp-ink-muted); line-height: 1.65;
          }}
          .sp-footer b {{ color: var(--sp-ink-2); font-weight: 600; }}

          /* Tabular figures where columns must align. */
          [data-testid="stDataFrame"] {{ font-variant-numeric: tabular-nums; }}

          /* Give bordered containers the card surface. Falls back to a plain
             outlined box if this internal test id ever changes. */
          [data-testid="stVerticalBlockBorderWrapper"]:has(> div > [data-testid="stVerticalBlock"]) {{
            background: var(--sp-surface);
            border-radius: 0.9rem;
          }}
          [data-testid="stMetric"] {{ background: var(--sp-surface); }}

          /* Sidebar identity. */
          [data-testid="stSidebarNav"] {{ margin-top: 0.4rem; }}
          .sp-brand {{
            font-family: var(--sp-heading-font);
            font-size: 1.05rem; font-weight: 600; letter-spacing: -0.01em;
            color: var(--sp-ink); display: flex; align-items: center; gap: 0.5rem;
            margin-bottom: 0.15rem;
          }}
          .sp-brand-sub {{
            font-size: 0.74rem; color: var(--sp-ink-muted); margin-bottom: 0.9rem;
          }}
          .sp-appearance {{
            display: flex; flex-direction: column; gap: 0.15rem;
            border: 1px solid var(--sp-border); border-radius: 0.6rem;
            background: var(--sp-surface-alt);
            padding: 0.55rem 0.7rem; margin-bottom: 0.2rem;
          }}
          .sp-appearance-now {{ font-size: 0.84rem; font-weight: 600; color: var(--sp-ink); }}
          .sp-appearance-hint {{ font-size: 0.72rem; color: var(--sp-ink-muted); }}

          /* Aurora ring instead of an emoji, so the mark renders identically
             everywhere rather than depending on the platform's emoji font. */
          .sp-mark {{
            width: 15px; height: 15px; border-radius: 50%; flex: none;
            background: conic-gradient(from 210deg, var(--sp-accent), var(--sp-accent-2), var(--sp-accent));
            box-shadow: 0 0 0 3px {_rgba_css(t['accent'], 0.16)};
          }}
        </style>
        """,
        unsafe_allow_html=True,
    )
