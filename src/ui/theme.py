"""Robinhood-style theming for the Streamlit dashboard.

Two palettes (dark / light) selectable at runtime. `inject_css(theme)` returns a
`<style>` block that repaints Streamlit's chrome into a clean, Robinhood-like
look (near-black canvas, signature green for gains, red for losses, Inter
typography, card surfaces, pill tabs). `style_fig(fig, theme)` restyles Plotly
figures to match (transparent canvas, green/red line with a soft area fill, muted
gridlines).
"""

from __future__ import annotations

from typing import Any

# Robinhood signature colors.
RH_GREEN = "#00C805"          # gains / primary accent
RH_GREEN_DARK = "#00A806"     # gains on a white background (more contrast)
RH_RED = "#FF5000"            # losses (dark)
RH_RED_LIGHT = "#E23F44"      # losses on white

_PALETTES: dict[str, dict[str, str]] = {
    "dark": {
        "bg": "#000000",
        "surface": "#16181D",
        "surface_2": "#1C1F26",
        "border": "#23272E",
        "text": "#FFFFFF",
        "text_muted": "#9BA1A6",
        "green": RH_GREEN,
        "red": RH_RED,
        "accent": RH_GREEN,
        "grid": "rgba(255,255,255,0.06)",
        "chart_fill": "rgba(0,200,5,0.14)",
    },
    "light": {
        "bg": "#FFFFFF",
        "surface": "#FFFFFF",
        "surface_2": "#F5F6F7",
        "border": "#E3E6E8",
        "text": "#0B0E11",
        "text_muted": "#6A7178",
        "green": RH_GREEN_DARK,
        "red": RH_RED_LIGHT,
        "accent": RH_GREEN_DARK,
        "grid": "rgba(0,0,0,0.06)",
        "chart_fill": "rgba(0,168,6,0.12)",
    },
}


def palette(theme: str) -> dict[str, str]:
    return _PALETTES.get(theme, _PALETTES["dark"])


def inject_css(theme: str = "dark") -> str:
    """Return a <style> block that themes the whole Streamlit app."""
    p = palette(theme)
    return f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {{
  --rh-bg: {p['bg']};
  --rh-surface: {p['surface']};
  --rh-surface-2: {p['surface_2']};
  --rh-border: {p['border']};
  --rh-text: {p['text']};
  --rh-muted: {p['text_muted']};
  --rh-green: {p['green']};
  --rh-red: {p['red']};
  --rh-accent: {p['accent']};
}}

/* ---- base canvas + typography ---- */
html, body, .stApp, [data-testid="stAppViewContainer"] {{
  background: var(--rh-bg) !important;
  color: var(--rh-text) !important;
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif !important;
  -webkit-font-smoothing: antialiased;
}}
[data-testid="stMainBlockContainer"], .block-container {{
  padding-top: 2.4rem !important;
  max-width: 1180px;
}}

/* ---- hide Streamlit chrome for an app-like feel ---- */
[data-testid="stToolbar"], [data-testid="stDecoration"],
#MainMenu, footer, [data-testid="stStatusWidget"] {{ display: none !important; }}
[data-testid="stHeader"] {{ background: transparent !important; height: 0 !important; }}

/* ---- headings ---- */
h1, h2, h3, h4 {{
  color: var(--rh-text) !important;
  font-weight: 700 !important;
  letter-spacing: -0.02em;
}}
h1 {{ font-size: 1.9rem !important; }}
h2 {{ font-size: 1.35rem !important; }}
h3 {{ font-size: 1.12rem !important; }}
[data-testid="stCaptionContainer"], .stCaption, small {{ color: var(--rh-muted) !important; }}
p, span, label, li {{ color: var(--rh-text); }}
a {{ color: var(--rh-accent) !important; }}
code {{
  background: var(--rh-surface-2) !important;
  color: var(--rh-text) !important;
  border-radius: 6px; padding: 1px 6px; font-size: 0.82em;
}}
hr, [data-testid="stDivider"] {{ border-color: var(--rh-border) !important; }}

/* ---- tabs → Robinhood underline nav ---- */
[data-testid="stTabs"] [data-baseweb="tab-list"] {{
  gap: 6px; border-bottom: 1px solid var(--rh-border);
}}
[data-testid="stTabs"] button[data-baseweb="tab"] {{
  background: transparent !important;
  color: var(--rh-muted) !important;
  font-weight: 600 !important;
  padding: 10px 14px !important;
}}
[data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"] {{
  color: var(--rh-text) !important;
}}
[data-testid="stTabs"] [data-baseweb="tab-highlight"] {{
  background: var(--rh-accent) !important; height: 3px !important; border-radius: 3px;
}}

/* ---- metrics → clean cards with big numbers ---- */
[data-testid="stMetric"] {{
  background: var(--rh-surface) !important;
  border: 1px solid var(--rh-border);
  border-radius: 14px;
  padding: 16px 18px !important;
}}
[data-testid="stMetricLabel"] p {{
  color: var(--rh-muted) !important; font-weight: 600 !important;
  font-size: 0.8rem !important; text-transform: none;
}}
[data-testid="stMetricValue"] {{
  color: var(--rh-text) !important; font-weight: 700 !important;
  font-size: 1.7rem !important; letter-spacing: -0.02em;
}}
[data-testid="stMetricDelta"] {{ font-weight: 600 !important; }}
[data-testid="stMetricDelta"] svg {{ display: none; }}

/* ---- dataframes ---- */
[data-testid="stDataFrame"] {{
  border: 1px solid var(--rh-border) !important;
  border-radius: 14px !important; overflow: hidden;
}}
[data-testid="stDataFrame"] * {{ font-family: 'Inter', sans-serif !important; }}

/* ---- buttons ---- */
.stButton > button, [data-testid="stBaseButton-secondary"] {{
  background: var(--rh-surface-2) !important;
  color: var(--rh-text) !important;
  border: 1px solid var(--rh-border) !important;
  border-radius: 999px !important;
  font-weight: 600 !important;
  padding: 8px 18px !important;
  transition: all .15s ease;
}}
.stButton > button:hover {{ border-color: var(--rh-accent) !important; color: var(--rh-accent) !important; }}
[data-testid="stBaseButton-primary"] {{
  background: var(--rh-accent) !important;
  color: #001a01 !important; border: none !important;
  border-radius: 999px !important; font-weight: 700 !important;
}}

/* ---- segmented control (theme toggle) ---- */
[data-testid="stSegmentedControl"] button {{
  border-radius: 999px !important; font-weight: 600 !important;
}}

/* ---- inputs / selects ---- */
[data-baseweb="select"] > div, .stTextInput input, .stNumberInput input {{
  background: var(--rh-surface-2) !important;
  border-color: var(--rh-border) !important;
  color: var(--rh-text) !important; border-radius: 10px !important;
}}

/* ---- alerts / notifications → tinted surfaces ---- */
[data-testid="stAlert"], .stAlert, [data-testid="stNotification"] {{
  border-radius: 12px !important; border: 1px solid var(--rh-border) !important;
  background: var(--rh-surface) !important; color: var(--rh-text) !important;
}}

/* ---- expanders → cards ---- */
[data-testid="stExpander"] {{
  border: 1px solid var(--rh-border) !important;
  border-radius: 14px !important; background: var(--rh-surface) !important;
}}
[data-testid="stExpander"] summary {{ color: var(--rh-text) !important; }}

/* ---- plotly container transparency ---- */
[data-testid="stPlotlyChart"] {{ background: transparent !important; }}
</style>
"""


def style_fig(fig: Any, theme: str = "dark", *, positive: bool | None = None,
              fill: bool = True) -> Any:
    """Restyle a Plotly figure to the Robinhood aesthetic.

    - transparent canvas, Inter font, muted gridlines
    - a single accent color (green for up / red for down when `positive` given)
    - a soft area fill under line traces (Robinhood's signature look)
    """
    p = palette(theme)
    accent = p["green"] if positive is None or positive else p["red"]
    # Robinhood-ish qualitative sequence for multi-line charts (keeps traces
    # distinct while staying on-brand).
    colorway = [p["green"], "#4C8DFF", "#B084FF", p["text_muted"], "#FFB020"]
    fig.update_layout(
        colorway=colorway,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter, sans-serif", "color": p["text_muted"], "size": 12},
        legend={"bgcolor": "rgba(0,0,0,0)", "font": {"color": p["text_muted"]}},
        hoverlabel={"bgcolor": p["surface_2"], "font": {"color": p["text"],
                                                          "family": "Inter, sans-serif"}},
        xaxis={"gridcolor": p["grid"], "zeroline": False, "linecolor": p["border"],
               "tickfont": {"color": p["text_muted"]}},
        yaxis={"gridcolor": p["grid"], "zeroline": False, "linecolor": p["border"],
               "tickfont": {"color": p["text_muted"]}},
    )
    # Single-trace line charts get the signature accent color + soft area fill.
    if len(fig.data) == 1:
        fill_color = ("rgba(0,200,5,0.14)" if accent == p["green"]
                      else "rgba(255,80,0,0.12)")
        for tr in fig.data:
            if getattr(tr, "mode", None) and "lines" in str(tr.mode):
                tr.line.color = accent
                tr.line.width = 2.4
                if fill:
                    tr.fill = "tozeroy"
                    tr.fillcolor = fill_color
    return fig
