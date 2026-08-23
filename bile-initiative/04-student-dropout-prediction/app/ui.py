"""
Design system and custom visuals for EduRisk AI - Student Dropout
Early-Warning System.

This module owns presentation only. It never touches the model, the schema or
the prediction path.

Theme architecture
------------------
Streamlit 1.62 compiles its theme into generated class names and exposes no CSS
custom properties, so a stylesheet cannot read the active theme at runtime, and
``st.context.theme.type`` is documented as unreliable during a theme change.
The app therefore does not implement a theme switcher of its own. Instead:

* Streamlit's native theme system owns every base colour, declared per mode in
  ``.streamlit/config.toml`` under ``[theme.light]`` and ``[theme.dark]``.
* Everything in this file is written to be theme-independent, so it composes
  correctly on top of whichever mode Streamlit resolved:
    - text colour is never set; it inherits, and de-emphasis uses ``opacity``
    - surfaces, borders and separators are neutral grey alphas, which read as a
      recessed panel on a light page and a raised one on a dark page
    - SVG text is drawn in ``currentColor``
* The only fixed colours are the four risk/series hues below. Each was checked
  against both the light (#ffffff) and dark (#0f1115) page colours; the
  diverging pair was validated for colour-vision separation on both.

No plotting dependency is used - every visual is inline HTML/SVG.
"""

from __future__ import annotations

from html import escape
from typing import Iterable, Sequence

import streamlit as st

from model_service import FeatureContribution

# ---------------------------------------------------------------------------
# Fixed hues (contrast-checked on light #ffffff and dark #0f1115)
# ---------------------------------------------------------------------------

RISK_LOW = "#0ca30c"    # 3.35:1 light / 5.63:1 dark
RISK_HIGH = "#d03b3b"   # 4.80:1 light / 3.93:1 dark

# Diverging pair for the contribution chart. Validated as a two-slot palette on
# both surfaces: CVD separation dE 20.7, normal-vision dE 29.5, contrast >= 3:1.
LOWERS = "#3b82d9"      # moves the estimate down
RAISES = "#dd5150"      # moves the estimate up


def risk_colour(is_elevated: bool) -> str:
    return RISK_HIGH if is_elevated else RISK_LOW


# ---------------------------------------------------------------------------
# Stylesheet
# ---------------------------------------------------------------------------

STYLES = """
<style>
  /* Neutral tokens. Grey alphas and inherited ink compose correctly over the
     light and the dark page alike, so none of this needs a theme branch. */
  .stApp {
    --sd-line: rgba(128, 128, 128, 0.32);
    --sd-line-soft: rgba(128, 128, 128, 0.19);
    --sd-line-faint: rgba(128, 128, 128, 0.12);
    --sd-fill: rgba(128, 128, 128, 0.05);
    --sd-fill-strong: rgba(128, 128, 128, 0.11);
    --sd-shadow: 0 1px 2px rgba(0, 0, 0, 0.04), 0 4px 16px rgba(0, 0, 0, 0.04);
    --sd-r-sm: 8px;
    --sd-r-md: 12px;
    --sd-r-lg: 16px;
  }

  .stMainBlockContainer { padding-top: 3.4rem; padding-bottom: 5rem; max-width: 1180px; }

  /* Deploy button is noise in a demo; the ⋮ menu stays so Settings, theme and
     "Rerun" remain reachable. */
  [data-testid="stAppDeployButton"] { display: none; }

  /* ------------------------------------------------------------------ type */
  .stApp p.sd-eyebrow {
    font-size: 0.685rem; font-weight: 660; letter-spacing: 0.13em;
    text-transform: uppercase; opacity: 0.5; margin: 0 0 0.55rem 0;
  }
  .stApp p.sd-section-title {
    font-size: 1.06rem; font-weight: 640; margin: 0 0 0.15rem 0;
    letter-spacing: -0.006em;
  }
  .stApp p.sd-section-caption {
    font-size: 0.86rem; opacity: 0.62; margin: 0; line-height: 1.55; max-width: 76ch;
  }
  .stApp p.sd-note {
    font-size: 0.84rem; line-height: 1.65; opacity: 0.66; margin: 0.25rem 0 0 0;
  }
  .sd-rule { height: 1px; background: var(--sd-line-faint); border: 0; margin: 2rem 0 1.6rem; }
  .sd-rule--tight { margin: 1.15rem 0; }

  /* ------------------------------------------------------------------ hero */
  .sd-hero {
    border: 1px solid var(--sd-line-soft); border-radius: var(--sd-r-lg);
    background: var(--sd-fill); padding: 1.6rem 1.75rem 0;
    margin-bottom: 1.5rem; overflow: hidden;
  }
  .sd-hero-brand {
    font-size: clamp(1.7rem, 3.6vw, 2.3rem); font-weight: 680;
    letter-spacing: -0.032em; line-height: 1.08; margin: 0 0 0.28rem 0;
  }
  .sd-hero-subtitle {
    font-size: clamp(0.95rem, 1.7vw, 1.1rem); font-weight: 600;
    letter-spacing: -0.008em; line-height: 1.3; opacity: 0.78;
    margin: 0 0 0.65rem 0;
  }
  .sd-hero-lede {
    font-size: 0.95rem; line-height: 1.6; opacity: 0.62; margin: 0;
    max-width: 62ch;
  }
  .sd-hero-strip {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(132px, 1fr));
    margin: 1.5rem -1.75rem 0; border-top: 1px solid var(--sd-line-soft);
  }
  .sd-hero-stat {
    padding: 0.85rem 1.1rem; border-right: 1px solid var(--sd-line-faint);
    min-width: 0;
  }
  .sd-hero-stat:first-child { padding-left: 1.75rem; }
  .sd-hero-stat:last-child { border-right: 0; }
  .sd-hero-stat-label {
    font-size: 0.655rem; font-weight: 640; letter-spacing: 0.09em;
    text-transform: uppercase; opacity: 0.5; margin-bottom: 0.22rem;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
  }
  .sd-hero-stat-value {
    font-size: 0.95rem; font-weight: 620; line-height: 1.3;
    font-variant-numeric: tabular-nums;
    display: flex; align-items: center; gap: 0.4rem;
  }
  .sd-dot { width: 7px; height: 7px; border-radius: 50%; flex: 0 0 auto; }

  /* ----------------------------------------------------------------- cards */
  div[class*="st-key-sdcard"] {
    border: 1px solid var(--sd-line-soft) !important;
    border-radius: var(--sd-r-md) !important;
    background: var(--sd-fill) !important;
    padding: 1.05rem 1.2rem 1rem !important;
  }
  div[class*="st-key-sdcard"] div[data-testid="stVerticalBlock"] { gap: 0.7rem; }
  /* side-by-side cards share a height so rows stay level */
  div[data-testid="stColumn"]
    > div[data-testid="stVerticalBlock"]
    > div[data-testid="stLayoutWrapper"]:only-child { height: 100%; }
  div[data-testid="stColumn"] div[class*="st-key-sdcard"] { height: 100%; }

  .sd-card-head {
    display: flex; align-items: flex-start; gap: 0.7rem;
    padding-bottom: 0.7rem; margin-bottom: 0.25rem;
    border-bottom: 1px solid var(--sd-line-faint);
  }
  .sd-card-index {
    font-size: 0.7rem; font-weight: 700; letter-spacing: 0.04em;
    opacity: 0.58; padding-top: 0.2rem; flex: 0 0 auto;
    font-variant-numeric: tabular-nums;
  }
  .sd-card-title { font-size: 0.985rem; font-weight: 640; line-height: 1.3; }
  .sd-card-sub { font-size: 0.795rem; opacity: 0.58; line-height: 1.45; margin-top: 0.12rem; }

  /* ------------------------------------------------------------- action bar */
  div[class*="st-key-sdaction"] {
    border: 1px solid var(--sd-line-soft) !important;
    border-radius: var(--sd-r-md) !important;
    background: var(--sd-fill-strong) !important;
    padding: 0.9rem 1.1rem !important;
  }
  div[class*="st-key-sdaction"] button { font-weight: 650; font-size: 1.02rem; }
  div[class*="st-key-sdaction"] button div { padding: 0.15rem 0; }
  .sd-action-hint { font-size: 0.855rem; opacity: 0.68; line-height: 1.5; }
  .sd-action-hint strong { opacity: 1; font-weight: 620; }

  /* ---------------------------------------------------------------- result */
  .sd-result {
    border: 1px solid var(--sd-line-soft);
    border-left: 3px solid var(--sd-accent);
    border-radius: var(--sd-r-lg); background: var(--sd-fill);
    box-shadow: var(--sd-shadow);
    padding: 1.35rem 1.5rem 1.4rem;
  }
  .sd-result-top {
    display: flex; align-items: center; justify-content: space-between;
    gap: 1rem; flex-wrap: wrap; margin-bottom: 1.25rem;
  }
  .sd-pill {
    display: inline-flex; align-items: center; gap: 0.5rem;
    font-size: 0.94rem; font-weight: 650; letter-spacing: -0.004em;
    border: 1px solid var(--sd-accent); border-radius: 999px;
    padding: 0.3rem 0.85rem 0.3rem 0.7rem;
  }
  .sd-pill-dot { width: 9px; height: 9px; border-radius: 50%; background: var(--sd-accent); }
  .sd-result-meta { font-size: 0.8rem; opacity: 0.58; }
  .sd-result-body { display: flex; gap: 2.25rem; align-items: flex-end; flex-wrap: wrap; }
  .sd-figure { flex: 0 0 auto; min-width: 132px; }
  .sd-figure-value {
    font-size: clamp(2.5rem, 6.5vw, 3.5rem); font-weight: 660;
    letter-spacing: -0.04em; line-height: 0.92; color: var(--sd-accent);
    font-variant-numeric: tabular-nums;
  }
  .sd-figure-label {
    font-size: 0.7rem; letter-spacing: 0.1em; text-transform: uppercase;
    opacity: 0.55; margin-top: 0.55rem; font-weight: 640;
  }
  .sd-gauge { flex: 1 1 320px; min-width: 0; padding-bottom: 0.25rem; }
  .sd-gauge-head {
    display: flex; justify-content: space-between; align-items: baseline;
    font-size: 0.72rem; letter-spacing: 0.08em; text-transform: uppercase;
    font-weight: 640; opacity: 0.5; margin-bottom: 0.45rem;
  }
  .sd-gauge-track {
    position: relative; height: 12px; border-radius: 6px;
    background: var(--sd-fill-strong);
    box-shadow: inset 0 0 0 1px var(--sd-line-faint);
  }
  .sd-gauge-fill { height: 100%; border-radius: 6px; min-width: 12px; }
  .sd-gauge-mark {
    position: absolute; top: -5px; bottom: -5px; width: 2px;
    background: currentColor; opacity: 0.8; border-radius: 1px;
    transform: translateX(-1px);
  }
  .sd-gauge-foot { position: relative; height: 1.05rem; margin-top: 0.45rem; }
  .sd-gauge-foot span {
    position: absolute; transform: translateX(-50%); white-space: nowrap;
    font-size: 0.715rem; opacity: 0.58;
  }
  .sd-result-stats {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(136px, 1fr));
    margin: 1.4rem -0.4rem 0; padding-top: 1.1rem;
    border-top: 1px solid var(--sd-line-faint);
  }
  .sd-result-stats > div {
    padding: 0 1rem; border-right: 1px solid var(--sd-line-faint); min-width: 0;
  }
  .sd-result-stats > div:last-child { border-right: 0; }
  .sd-stat-label {
    font-size: 0.655rem; font-weight: 640; letter-spacing: 0.085em;
    text-transform: uppercase; opacity: 0.52; margin-bottom: 0.22rem;
  }
  .sd-stat-value {
    font-size: 1.08rem; font-weight: 630; line-height: 1.25;
    font-variant-numeric: tabular-nums;
  }

  /* --------------------------------------------------------- metric tiles */
  .sd-metrics {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(178px, 1fr));
    gap: 0.7rem;
  }
  .sd-metric {
    border: 1px solid var(--sd-line-soft); border-radius: var(--sd-r-md);
    background: var(--sd-fill); padding: 0.95rem 1.05rem 1rem;
  }
  .sd-metric-label {
    font-size: 0.68rem; font-weight: 640; letter-spacing: 0.09em;
    text-transform: uppercase; opacity: 0.52;
  }
  .sd-metric-value {
    font-size: 1.75rem; font-weight: 650; letter-spacing: -0.03em;
    line-height: 1.1; margin: 0.35rem 0 0.4rem;
    font-variant-numeric: tabular-nums;
  }
  .sd-metric-note { font-size: 0.775rem; opacity: 0.6; line-height: 1.45; }
  .sd-metric--accent .sd-metric-value { color: var(--sd-accent); }
  .sd-metric--accent { border-color: var(--sd-accent); }

  /* -------------------------------------------------------------- callouts */
  .sd-callout {
    border: 1px solid var(--sd-line-soft); border-left: 3px solid var(--sd-accent, var(--sd-line));
    border-radius: var(--sd-r-md); background: var(--sd-fill);
    padding: 0.95rem 1.15rem; font-size: 0.875rem; line-height: 1.65;
  }
  .sd-callout-title {
    font-weight: 640; font-size: 0.8rem; letter-spacing: 0.075em;
    text-transform: uppercase; opacity: 0.62; margin-bottom: 0.4rem;
  }
  .sd-callout-body { opacity: 0.86; }
  .sd-callout-body strong { opacity: 1; font-weight: 640; }

  /* ----------------------------------------------------------- definitions */
  .sd-dl { display: grid; gap: 0.42rem; font-size: 0.82rem; }
  .sd-dl-row { display: flex; justify-content: space-between; gap: 1rem; }
  .sd-dl-key { opacity: 0.55; }
  .sd-dl-val { font-weight: 620; text-align: right; font-variant-numeric: tabular-nums; }

  /* --------------------------------------------------------------- sidebar */
  .sd-brand { display: flex; align-items: center; gap: 0.6rem; margin-bottom: 0.2rem; }
  .sd-brand-mark {
    width: 30px; height: 30px; border-radius: 9px; flex: 0 0 auto;
    border: 1px solid var(--sd-line); display: flex; align-items: center;
    justify-content: center; font-size: 0.72rem; font-weight: 700;
    letter-spacing: -0.02em; opacity: 0.75;
  }
  .sd-brand-name { font-size: 0.9rem; font-weight: 650; line-height: 1.2; }
  .sd-brand-sub {
    font-size: 0.7rem; opacity: 0.55; letter-spacing: 0.05em;
    text-transform: uppercase; margin-top: 0.1rem;
  }
  .sd-readout {
    border: 1px solid var(--sd-line-soft); border-left: 3px solid var(--sd-accent);
    border-radius: var(--sd-r-md); padding: 0.7rem 0.85rem; background: var(--sd-fill);
  }
  .sd-readout-label {
    font-size: 0.645rem; font-weight: 640; letter-spacing: 0.085em;
    text-transform: uppercase; opacity: 0.52;
  }
  .sd-readout-value {
    font-size: 1.35rem; font-weight: 650; margin-top: 0.15rem;
    letter-spacing: -0.02em; color: var(--sd-accent);
    font-variant-numeric: tabular-nums;
  }
  .sd-readout-sub { font-size: 0.76rem; opacity: 0.62; margin-top: 0.15rem; }

  /* ----------------------------------------------------------------- chart */
  .sd-legend {
    display: flex; gap: 1.1rem; flex-wrap: wrap;
    font-size: 0.785rem; opacity: 0.7; margin: 0.1rem 0 0.6rem;
  }
  .sd-legend-item { display: inline-flex; align-items: center; gap: 0.38rem; }
  .sd-swatch { width: 10px; height: 10px; border-radius: 2px; flex: 0 0 auto; }
  .sd-scroll { overflow-x: auto; overflow-y: hidden; max-width: 100%; }
  .sd-scroll svg { font-family: inherit; }

  /* ---------------------------------------------------------------- footer */
  .sd-footer {
    display: flex; justify-content: space-between; align-items: baseline;
    gap: 1rem 2rem; flex-wrap: wrap;
    margin-top: 3rem; padding-top: 1.1rem;
    border-top: 1px solid var(--sd-line-faint);
    font-size: 0.78rem; line-height: 1.6; opacity: 0.58;
  }
  .sd-footer-brand { font-weight: 640; opacity: 0.85; }
  .sd-footer-meta { text-align: right; }
  @media (max-width: 640px) {
    .sd-footer { flex-direction: column; }
    .sd-footer-meta { text-align: left; }
  }

  /* ------------------------------------------------- Streamlit refinements */
  div[data-testid="stTabs"] button p { font-size: 0.94rem; font-weight: 560; }
  div[data-testid="stTabs"] [data-baseweb="tab-list"] { gap: 0.35rem; }
  div[data-testid="stExpander"] details { border-radius: var(--sd-r-md); }
  div[data-testid="stExpander"] summary p { font-weight: 580; font-size: 0.9rem; }
  section[data-testid="stSidebar"] hr { margin: 1.05rem 0; }

  /* ---------------------------------------------------------- small screens */
  @media (max-width: 640px) {
    .stMainBlockContainer { padding-left: 1rem; padding-right: 1rem; padding-top: 3rem; }
    .sd-hero { padding: 1.25rem 1.15rem 0; border-radius: var(--sd-r-md); }
    .sd-hero-strip { margin: 1.2rem -1.15rem 0; grid-template-columns: repeat(2, 1fr); }
    .sd-hero-stat { padding: 0.7rem 0.9rem; border-bottom: 1px solid var(--sd-line-faint); }
    .sd-hero-stat:first-child { padding-left: 1.15rem; }
    .sd-hero-stat:nth-child(2n) { border-right: 0; }
    .sd-result { padding: 1.05rem 1.05rem 1.15rem; }
    .sd-result-body { gap: 1.3rem; }
    .sd-result-stats { margin: 1.1rem 0 0; padding-top: 0.95rem; gap: 0.85rem 0; }
    .sd-result-stats > div { padding: 0; border-right: 0; }
  }
</style>
"""


def inject_styles() -> None:
    st.markdown(STYLES, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Type helpers
# ---------------------------------------------------------------------------


def eyebrow(text: str) -> None:
    st.markdown(f'<p class="sd-eyebrow">{escape(text)}</p>', unsafe_allow_html=True)


def section_intro(title: str, caption: str = "", label: str = "") -> None:
    head = f'<p class="sd-eyebrow">{escape(label)}</p>' if label else ""
    tail = f'<p class="sd-section-caption">{escape(caption)}</p>' if caption else ""
    st.markdown(
        f'{head}<p class="sd-section-title">{escape(title)}</p>{tail}',
        unsafe_allow_html=True,
    )


def note(html: str) -> None:
    st.markdown(f'<p class="sd-note">{html}</p>', unsafe_allow_html=True)


def rule(tight: bool = False) -> None:
    cls = "sd-rule sd-rule--tight" if tight else "sd-rule"
    st.markdown(f'<hr class="{cls}" />', unsafe_allow_html=True)


def card_head(index: int | None, title: str, caption: str) -> None:
    """Card header. Pass an index for the numbered input sections, None elsewhere."""
    badge = f'<span class="sd-card-index">{index:02d}</span>' if index else ""
    st.markdown(
        f'<div class="sd-card-head">{badge}'
        f'<div><div class="sd-card-title">{escape(title)}</div>'
        f'<div class="sd-card-sub">{escape(caption)}</div></div></div>',
        unsafe_allow_html=True,
    )


def callout(title: str, body_html: str, accent: str | None = None) -> None:
    style = f' style="--sd-accent:{accent}"' if accent else ""
    st.markdown(
        f'<div class="sd-callout"{style}>'
        f'<div class="sd-callout-title">{escape(title)}</div>'
        f'<div class="sd-callout-body">{body_html}</div></div>',
        unsafe_allow_html=True,
    )


def definition_list(rows: Iterable[tuple[str, str]]) -> None:
    body = "".join(
        f'<div class="sd-dl-row"><span class="sd-dl-key">{escape(k)}</span>'
        f'<span class="sd-dl-val">{escape(v)}</span></div>'
        for k, v in rows
    )
    st.markdown(f'<div class="sd-dl">{body}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------


def hero(
    brand: str,
    subtitle: str,
    lede: str,
    stats: Sequence[tuple[str, str, str | None]],
) -> None:
    """Product header plus a system-status strip.

    Three levels of identity, in order: the product name, what it is, what it
    does. ``stats`` is (label, value, dot).
    """
    strip = "".join(
        '<div class="sd-hero-stat">'
        f'<div class="sd-hero-stat-label">{escape(label)}</div>'
        '<div class="sd-hero-stat-value">'
        + (f'<span class="sd-dot" style="background:{dot}"></span>' if dot else "")
        + f"{escape(value)}</div></div>"
        for label, value, dot in stats
    )
    st.markdown(
        '<div class="sd-hero">'
        f'<h1 class="sd-hero-brand">{escape(brand)}</h1>'
        f'<p class="sd-hero-subtitle">{escape(subtitle)}</p>'
        f'<p class="sd-hero-lede">{escape(lede)}</p>'
        f'<div class="sd-hero-strip">{strip}</div>'
        "</div>",
        unsafe_allow_html=True,
    )


def footer(brand_line: str, meta_lines: Sequence[str]) -> None:
    """Quiet page footer carrying product identity and attribution."""
    meta = "<br />".join(escape(line) for line in meta_lines)
    st.markdown(
        '<div class="sd-footer">'
        f'<div class="sd-footer-brand">{escape(brand_line)}</div>'
        f'<div class="sd-footer-meta">{meta}</div>'
        "</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Metric tiles
# ---------------------------------------------------------------------------


def metric_tiles(items: Sequence[tuple[str, str, str]], accent_label: str | None = None,
                 accent: str | None = None) -> None:
    """``items`` is (label, value, note). One tile may be accented."""
    tiles = []
    for label, value, tile_note in items:
        highlight = accent_label is not None and label == accent_label
        cls = "sd-metric sd-metric--accent" if highlight else "sd-metric"
        style = f' style="--sd-accent:{accent}"' if highlight and accent else ""
        tiles.append(
            f'<div class="{cls}"{style}>'
            f'<div class="sd-metric-label">{escape(label)}</div>'
            f'<div class="sd-metric-value">{escape(value)}</div>'
            f'<div class="sd-metric-note">{escape(tile_note)}</div></div>'
        )
    st.markdown(f'<div class="sd-metrics">{"".join(tiles)}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Result panel
# ---------------------------------------------------------------------------


def _gauge(probability: float, threshold: float, colour: str) -> str:
    """Probability bar with the decision threshold marked in place.

    Emitted as one line on purpose: a blank line inside a raw HTML block ends the
    block in Markdown, and everything after it would render as literal text.
    """
    fill = max(0.0, min(1.0, probability)) * 100
    mark = max(0.0, min(1.0, threshold)) * 100
    return (
        '<div class="sd-gauge-head"><span>Risk probability</span>'
        f"<span>{probability * 100:.1f}%</span></div>"
        '<div class="sd-gauge-track">'
        f'<div class="sd-gauge-fill" style="width:{fill:.2f}%;background:{colour}"></div>'
        f'<div class="sd-gauge-mark" style="left:{mark:.2f}%"></div></div>'
        f'<div class="sd-gauge-foot"><span style="left:{mark:.2f}%">'
        f"threshold {threshold * 100:.0f}%</span></div>"
    )


def result_panel(
    label: str,
    is_elevated: bool,
    probability: float,
    threshold: float,
    stats: Sequence[tuple[str, str]],
) -> None:
    colour = risk_colour(is_elevated)
    relation = "at or above" if is_elevated else "below"
    meta = f"Estimate is {relation} the saved {threshold:.2f} decision threshold"

    stat_html = "".join(
        f'<div><div class="sd-stat-label">{escape(name)}</div>'
        f'<div class="sd-stat-value">{escape(value)}</div></div>'
        for name, value in stats
    )

    st.markdown(
        f'<div class="sd-result" style="--sd-accent:{colour}">'
        '<div class="sd-result-top">'
        f'<span class="sd-pill"><span class="sd-pill-dot"></span>{escape(label)}</span>'
        f'<span class="sd-result-meta">{escape(meta)}</span></div>'
        '<div class="sd-result-body">'
        f'<div class="sd-figure"><div class="sd-figure-value">{probability * 100:.1f}%</div>'
        '<div class="sd-figure-label">dropout probability</div></div>'
        f'<div class="sd-gauge">{_gauge(probability, threshold, colour)}</div>'
        "</div>"
        f'<div class="sd-result-stats">{stat_html}</div>'
        "</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Sidebar pieces
# ---------------------------------------------------------------------------


def brand(mark: str, name: str, sub: str) -> None:
    st.markdown(
        f'<div class="sd-brand"><div class="sd-brand-mark">{escape(mark)}</div>'
        f'<div><div class="sd-brand-name">{escape(name)}</div>'
        f'<div class="sd-brand-sub">{escape(sub)}</div></div></div>',
        unsafe_allow_html=True,
    )


def readout(label: str, is_elevated: bool, probability: float, stale: bool) -> None:
    colour = risk_colour(is_elevated)
    sub = "inputs changed since this run" if stale else f"{label} · current inputs"
    st.markdown(
        f'<div class="sd-readout" style="--sd-accent:{colour}">'
        '<div class="sd-readout-label">Last assessment</div>'
        f'<div class="sd-readout-value">{probability * 100:.1f}%</div>'
        f'<div class="sd-readout-sub">{escape(sub)}</div></div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Contribution chart
# ---------------------------------------------------------------------------


def _bar_path(x0: float, y: float, height: float, length: float,
              radius: float, positive: bool) -> str:
    """Bar squared off at the zero axis, rounded at the data end."""
    length = max(length, 0.75)
    r = min(radius, length, height / 2)
    if positive:
        x1 = x0 + length
        return (
            f"M {x0:.2f} {y:.2f} H {x1 - r:.2f} A {r:.2f} {r:.2f} 0 0 1 {x1:.2f} {y + r:.2f} "
            f"V {y + height - r:.2f} A {r:.2f} {r:.2f} 0 0 1 {x1 - r:.2f} {y + height:.2f} "
            f"H {x0:.2f} Z"
        )
    x1 = x0 - length
    return (
        f"M {x0:.2f} {y:.2f} H {x1 + r:.2f} A {r:.2f} {r:.2f} 0 0 0 {x1:.2f} {y + r:.2f} "
        f"V {y + height - r:.2f} A {r:.2f} {r:.2f} 0 0 0 {x1 + r:.2f} {y + height:.2f} "
        f"H {x0:.2f} Z"
    )


def contribution_chart(
    contributions: Sequence[FeatureContribution],
    label_lookup,
    value_lookup,
) -> None:
    """Diverging bars: how far each input moves this student's log-odds.

    Text is drawn in ``currentColor`` so it follows the active theme; only the
    two bar hues are fixed, and both clear 3:1 on the light and dark surfaces.
    """
    if not contributions:
        return

    st.markdown(
        '<div class="sd-legend">'
        f'<span class="sd-legend-item"><span class="sd-swatch" style="background:{RAISES}"></span>'
        "Moves the estimate up</span>"
        f'<span class="sd-legend-item"><span class="sd-swatch" style="background:{LOWERS}"></span>'
        "Moves the estimate down</span></div>",
        unsafe_allow_html=True,
    )

    row_h, bar_h = 33.0, 13.0
    label_w, value_w, width = 250.0, 132.0, 920.0
    plot_w = width - label_w - value_w
    x_zero = label_w + plot_w / 2
    half = plot_w / 2 - 8
    largest = max(abs(c.contribution) for c in contributions) or 1.0
    height = row_h * len(contributions) + 16

    parts = [
        f'<line x1="{x_zero:.2f}" y1="4" x2="{x_zero:.2f}" y2="{height - 10:.2f}" '
        'stroke="currentColor" stroke-width="1" opacity="0.26" />'
    ]
    for index, item in enumerate(contributions):
        row_top = 10 + index * row_h
        bar_y = row_top + (row_h - bar_h) / 2 - 5
        baseline = bar_y + bar_h - 2
        positive = item.contribution > 0
        length = half * abs(item.contribution) / largest
        if index:
            parts.append(
                f'<line x1="8" y1="{row_top - 3:.2f}" x2="{width - 8:.0f}" '
                f'y2="{row_top - 3:.2f}" stroke="currentColor" stroke-width="1" opacity="0.08" />'
            )
        parts.append(
            f'<text x="{label_w - 16:.2f}" y="{baseline:.2f}" font-size="13" '
            f'fill="currentColor" text-anchor="end">{escape(label_lookup(item.feature))}</text>'
        )
        parts.append(
            f'<path d="{_bar_path(x_zero, bar_y, bar_h, length, 4.0, positive)}" '
            f'fill="{RAISES if positive else LOWERS}" />'
        )
        parts.append(
            f'<text x="{width - value_w + 16:.2f}" y="{baseline:.2f}" font-size="13" '
            f'fill="currentColor" opacity="0.66">{escape(value_lookup(item.feature))}</text>'
        )

    st.markdown(
        '<div class="sd-scroll">'
        f'<svg viewBox="0 0 {width:.0f} {height:.0f}" width="{width:.0f}" height="{height:.0f}" '
        'role="img" preserveAspectRatio="xMinYMin meet" '
        "aria-label=\"Per-feature contributions to this student's estimate.\" "
        'style="display:block;max-width:100%;min-width:620px;height:auto">'
        f'{"".join(parts)}</svg></div>',
        unsafe_allow_html=True,
    )
