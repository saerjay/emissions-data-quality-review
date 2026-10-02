"""
Emissions Data Quality Review: a Streamlit app for checking emissions data before it is
used in a sustainability report.

Written for a general professional audience first (dashboard, review queue, plain-language
explanations), with the full technical detail kept in the Methods tab.
Run:  streamlit run app.py
"""
import html
import json
import os
from string import Template

import math

import altair as alt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from design import MAP_RAMP, STATUS, TOKENS
from mapview import available_measures, default_country, map_frame, trend_frame, trend_summary

from review import (build_review_queue, flags_by_year, year_insight, unusualness_label, friendly_name, key_takeaways,
                    readiness_verdict)
from validation import FALLBACK_TRANSFORM, ID_COLS, EmissionsDataValidator, attach_review_flags

# Paths resolve from this file, so the app works whatever directory it is launched from
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA = os.path.join(BASE_DIR, "data", "shap_selected_emissions_data.csv")
REPORT_DIR = os.path.join(BASE_DIR, "reports")
VIZ_DIR = os.path.join(BASE_DIR, "visualizations")

# Colors come from design.py: the chosen theme (teal, periwinkle, sky, rust), WCAG-tested and
# checked under simulated color blindness in tests/test_design.py. Teal = normal / Low,
# periwinkle = Medium, rust = flagged / High, sky = backgrounds and context. Color is never the
# only cue: priorities carry an icon and a word, flagged points use a triangle marker.
SERIES_BLUE = TOKENS['teal']          # normal series (name kept for compatibility)
ANOMALY_COLOR = TOKENS['rust']
TOWARD_ANOMALY = ANOMALY_COLOR
TOWARD_NORMAL = SERIES_BLUE
LOW_COLOR = TOKENS['teal']
MEDIUM_COLOR = TOKENS['periwinkle']
CONTEXT_BAND = TOKENS['sky']           # background context only (typical-range band), never a category
INK_MUTED = TOKENS['ink_muted']
# Outline for fills under 3:1 on white (periwinkle -> darker periwinkle, sky -> teal)
OUTLINE = {TOKENS['periwinkle']: TOKENS['periwinkle_dark'], TOKENS['sky']: TOKENS['teal']}
BADGE_TEXT = {'High': TOKENS['rust'], 'Medium': TOKENS['periwinkle_dark'], 'Low': TOKENS['primary']}


def outlined(field, mapping):
    """Altair stroke rule: outline each category's fill with its accessible edge color."""
    expr = alt.value(TOKENS['card'])
    for category, fill in reversed(list(mapping.items())):
        if fill in OUTLINE:
            expr = alt.condition(f"datum['{field}'] == '{category}'", alt.value(OUTLINE[fill]), expr)
    return expr


# Sequential single-hue map scale: pale sage to deep forest
MAP_SCALE = [[i / (len(MAP_RAMP) - 1), c] for i, c in enumerate(MAP_RAMP)]

# Global styling. No ghost buttons: everything clickable is solid forest with white text (>= 7:1),
# every focusable element shows a visible focus ring, and cards separate sections without heavy lines.
GLOBAL_CSS = Template("""
<style>
.block-container { max-width: 1320px; padding-top: 3.25rem; }
h1, h2, h3, h4 { color: $ink; letter-spacing: -0.01em; }
p, li { line-height: 1.55; }

/* keyboard focus */
*:focus-visible { outline: 3px solid $primary !important; outline-offset: 2px !important; }

/* hero */
.hero { padding: 0.25rem 0 1rem 0; }
.hero .eyebrow { color: $primary; font-weight: 700; font-size: 0.85rem; letter-spacing: 0.08em;
                 text-transform: uppercase; margin: 0 0 0.25rem 0; }
.hero h1 { font-size: 2.6rem; line-height: 1.15; margin: 0 0 0.5rem 0; padding: 0; }
.hero .lede { font-size: 1.1rem; color: $ink; max-width: 62rem; margin: 0 0 0.9rem 0; }
.chips { display: flex; flex-wrap: wrap; gap: 0.5rem; list-style: none; padding: 0; margin: 0; }
.chips li { background: $card; border: 1px solid $border; border-radius: 999px; padding: 0.2rem 0.75rem;
            font-size: 0.9rem; color: $ink_muted; margin: 0; }

/* status card */
.status { display: flex; gap: 0.9rem; align-items: flex-start; border-radius: 14px; padding: 1rem 1.2rem;
          border: 1px solid $border; border-left-width: 6px; margin: 0.25rem 0 1rem 0; }
.status svg { flex: none; margin-top: 0.15rem; }
.status .title { font-size: 1.2rem; font-weight: 700; color: $ink; margin: 0; }
.status .detail { color: $ink; margin: 0.15rem 0 0 0; }

/* metric cards */
[data-testid="stMetric"] { background: $card; border: 1px solid $border; border-radius: 14px;
                           padding: 0.9rem 1.1rem; }
[data-testid="stMetricLabel"] p { color: $ink_muted; font-size: 0.95rem; font-weight: 600; }
[data-testid="stMetricValue"] { color: $ink; font-weight: 700; }

/* section cards (containers with a key starting with "card") */
[class*="st-key-card"] { background: $card; border: 1px solid $border; border-radius: 16px;
                         padding: 1.1rem 1.25rem 1.2rem 1.25rem; }
.section-note { color: $ink_muted; font-size: 0.95rem; margin: -0.25rem 0 0.6rem 0; }

/* part-to-whole legend */
.legend { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 0.6rem;
          list-style: none; padding: 0; margin: 0.4rem 0 0 0; }
.legend li { display: flex; gap: 0.6rem; align-items: flex-start; margin: 0; }
.legend .sw { width: 14px; height: 14px; border-radius: 4px; flex: none; margin-top: 0.3rem; }
.legend .name { font-weight: 700; color: $ink; }
.legend .num { color: $ink; }
.legend .meaning { color: $ink_muted; font-size: 0.9rem; display: block; }

/* where to look first */
.look { list-style: none; padding: 0; margin: 0; }
.look li { border-top: 1px solid $border; padding: 0.65rem 0; margin: 0; }
.look li:first-child { border-top: none; padding-top: 0.2rem; }
.look .head { display: flex; justify-content: space-between; gap: 0.5rem; flex-wrap: wrap; }
.look .country { font-weight: 700; color: $ink; }
.look .badge { font-size: 0.8rem; font-weight: 700; border-radius: 999px; padding: 0.05rem 0.55rem;
               border: 1.5px solid currentColor; }
.look .when { color: $ink_muted; font-size: 0.9rem; }
.look .what { color: $ink; margin: 0.15rem 0 0 0; }

/* coverage meter */
.meter { height: 12px; background: $grid; border-radius: 999px; overflow: hidden; margin: 0.4rem 0 0.3rem 0; }
.meter span { display: block; height: 100%; background: $primary; border-radius: 999px; }

/* message boxes: theme tints (config.toml) plus a colored left edge, like the verdict card */
[data-testid="stAlertContainer"] { border-radius: 12px; border-left: 6px solid transparent; }
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentInfo"]),
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentSuccess"]) { border-left-color: $teal; }
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentWarning"]),
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentError"]) { border-left-color: $rust; }

/* clickable elements: solid, high contrast */
[data-testid="stDownloadButton"] button,
[data-testid="stBaseButton-secondary"] {
    background: $primary !important; color: #ffffff !important; border: 2px solid $primary !important;
    font-weight: 600 !important; border-radius: 10px !important;
}
[data-testid="stDownloadButton"] button:hover,
[data-testid="stBaseButton-secondary"]:hover { background: $primary_hover !important; border-color: $primary_hover !important; }
[data-testid="stDownloadButton"] button p { color: #ffffff !important; }
[data-testid="stExpander"] details { border: 2px solid $primary !important; border-radius: 12px; background: $card; }
[data-testid="stExpander"] summary { background: $primary_tint; color: $ink; font-weight: 600; border-radius: 10px; }
[data-testid="stExpander"] summary:hover { background: #cfe6de; }
[data-testid="stExpander"] summary svg { color: $primary; }

/* tabs */
[data-baseweb="tab-list"] { gap: 0.25rem; border-bottom: 1px solid $border; }
button[data-baseweb="tab"] { padding: 0.55rem 0.9rem; border-radius: 10px 10px 0 0; }
button[data-baseweb="tab"] p { color: $ink; font-weight: 600; font-size: 1rem; }
button[data-baseweb="tab"]:hover { background: $primary_tint; }
button[data-baseweb="tab"][aria-selected="true"] p { color: $primary; font-weight: 700; }
[data-baseweb="tab-highlight"] { background-color: $primary !important; height: 3px !important; }
[data-baseweb="select"] > div { border-color: $ink_muted !important; }
</style>
""").substitute(**TOKENS)

# Inline SVG status icons (decorative: the words carry the meaning)
STATUS_ICONS = {
    'ready': '<path d="M20 6 9 17l-5-5" fill="none" stroke="currentColor" stroke-width="2.5" '
             'stroke-linecap="round" stroke-linejoin="round"/>',
    'review': '<circle cx="11" cy="11" r="7" fill="none" stroke="currentColor" stroke-width="2.5"/>'
              '<path d="m20 20-4-4" stroke="currentColor" stroke-width="2.5" stroke-linecap="round"/>',
    'not_ready': '<circle cx="12" cy="12" r="9" fill="none" stroke="currentColor" stroke-width="2.5"/>'
                 '<path d="M12 7v6M12 16.5v.5" stroke="currentColor" stroke-width="2.5" stroke-linecap="round"/>',
}

PRIORITY_STYLE = {   # priority -> (color, icon, one-line meaning)
    'High': (ANOMALY_COLOR, '▲', "Breaks a basic rule. Fix before using the data."),
    'Medium': (MEDIUM_COLOR, '●', "Unusual for this country in this year. Check against the source."),
    'Low': (LOW_COLOR, '○', "This country is unusual every year. Likely real; confirm once."),
}

TRANSFORM_NAMES = {
    'none': 'No scaling', 'signed_log': 'Log', 'asinh': 'Inverse hyperbolic sine (log-like)',
    'cbrt': 'Cube root', 'yeo_johnson': 'Yeo-Johnson (fitted power)', 'box_cox': 'Box-Cox (fitted power)',
}

UNUSUALNESS_HELP = ("How unusual this record is compared with the study countries. 99% means it looks more "
                   "unusual than 99 out of 100 normal records.")

ERROR_LABELS = {
    'unit_x1000': "Wrong unit (1,000× too big)",
    'unit_div1000': "Wrong unit (1,000× too small)",
    'decimal_shift_x10': "Misplaced decimal (10× too big)",
    'sign_flip': "Negative number",
    'column_swap': "Values in the wrong column",
    'duplicate_record': "Same record entered twice",
    'missing_value': "Missing value",
}


# ---------------------------------------------------------------------------
# Data and settings
# ---------------------------------------------------------------------------
@st.cache_data
def load_csv(source):
    return pd.read_csv(source)


@st.cache_data(show_spinner="Checking every record...")
def run_pipeline(df, contamination, transform, use_ratios, range_z=4.0):
    validator = EmissionsDataValidator(df, contamination=contamination, transform=transform,
                                       use_ratios=use_ratios, range_z=range_z)
    summary = validator.run()
    scored = attach_review_flags(validator.scored, read_report("outlier_review.csv"))
    return summary, validator.schema_failures, scored, validator.shap_values, validator.feature_cols


def recommended_transform(report_dir=REPORT_DIR):
    """Returns (transform, reason) chosen by data_prep.py's statistical tests, or a fallback."""
    path = os.path.join(report_dir, "transform_selection.csv")
    if os.path.exists(path):
        selection = pd.read_csv(path)
        chosen = selection[selection['chosen']]
        if len(chosen):
            return chosen['transform'].iloc[0], chosen['reason'].iloc[0]
    return FALLBACK_TRANSFORM, (f"No transform_selection.csv found; run data_prep.py to select a transform "
                                f"statistically. Using {FALLBACK_TRANSFORM}.")


def recommended_config(report_dir=REPORT_DIR):
    """
    Returns (config, reason). Prefers the configuration chosen by evaluation.py, which is
    judged on what matters (planted errors caught vs clean rows flagged, on validation
    countries). Falls back to the normality-based transform choice from data_prep.py.
    """
    path = os.path.join(report_dir, "model_config.json")
    if os.path.exists(path):
        with open(path, encoding='utf-8') as f:
            saved = json.load(f)
        config = {k: saved[k] for k in ('transform', 'use_ratios', 'contamination', 'range_z')}
        return config, saved.get('reason', '')
    transform, _ = recommended_transform(report_dir)
    return ({'transform': transform, 'use_ratios': True, 'contamination': 0.02, 'range_z': 4.0},
            "No model_config.json found: run evaluation.py to choose settings by detection performance. "
            "Using the transform chosen for normality by data_prep.py.")


def snapshot_date():
    """Retrieval date of the raw data snapshot (from its manifest), formatted for APA."""
    import glob
    from datetime import datetime
    manifests = sorted(glob.glob(os.path.join(BASE_DIR, "data", "raw", "*.manifest.json")))
    if manifests:
        with open(manifests[-1], encoding='utf-8') as f:
            stamp = json.load(f).get('downloaded_at', '')
        try:
            d = datetime.fromisoformat(stamp)
            return f"{d:%B} {d.day}, {d.year}"
        except ValueError:
            pass
    return "the date of download"


def read_report(name):
    path = os.path.join(REPORT_DIR, name)
    return pd.read_csv(path) if os.path.exists(path) else None


def read_evaluation():
    path = os.path.join(REPORT_DIR, "model_config.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def priority_label(p):
    return f"{PRIORITY_STYLE[p][1]} {p}"


def show_chart(chart, **kwargs):
    """Renders an Altair chart with a transparent background so it sits cleanly on its card."""
    st.altair_chart(chart.properties(background='transparent'), **kwargs)


def show_image(name, caption):
    path = os.path.join(VIZ_DIR, name)
    if os.path.exists(path):
        st.image(path, caption=caption)


# ---------------------------------------------------------------------------
# Charts (shared style: horizontal bars, direct value labels, one axis, quiet grid)
# ---------------------------------------------------------------------------
def labelled_bars(data, value, category, color=SERIES_BLUE, value_format=',.0f', axis_title='',
                  domain=None, sort='-x', color_field=None, color_scale=None, tooltip=None):
    if '%' in value_format:   # shares: ticks stop at 100% even when the domain leaves room for labels
        axis = alt.Axis(format=value_format, grid=True, values=[0, 0.25, 0.5, 0.75, 1])
    else:
        axis = alt.Axis(grid=True, tickCount=5)
    x = alt.X(f'{value}:Q', title=axis_title, axis=axis,
              scale=alt.Scale(domain=domain) if domain else alt.Undefined)
    y = alt.Y(f'{category}:N', sort=sort, title=None, axis=alt.Axis(labelLimit=320, labelFontSize=13))
    base = alt.Chart(data).encode(x=x, y=y, tooltip=tooltip or [category, alt.Tooltip(f'{value}:Q', format=value_format)])
    fill = (alt.Color(f'{color_field}:N', scale=color_scale, legend=None) if color_field else alt.value(color))
    bars = base.mark_bar(cornerRadiusEnd=4).encode(color=fill)
    text = base.mark_text(align='left', dx=6, fontSize=13, color=INK_MUTED).encode(
        text=alt.Text(f'{value}:Q', format=value_format))
    return (bars + text).properties(height=alt.Step(34))


# ---------------------------------------------------------------------------
# Tab 1: Dashboard
# ---------------------------------------------------------------------------
def status_card(level, message):
    """Readiness verdict as an accessible status card: icon + bold word + detail, tinted by status."""
    style = STATUS[level]
    word, _, detail = message.partition(':')
    icon = (f'<svg width="26" height="26" viewBox="0 0 24 24" aria-hidden="true" style="color:{style["edge"]}">'
            f'{STATUS_ICONS[level]}</svg>')
    st.markdown(
        f'<div class="status" role="status" style="background:{style["tint"]};border-left-color:{style["edge"]}">'
        f'{icon}<div><p class="title">{html.escape(word)}</p>'
        f'<p class="detail">{html.escape(detail.strip())}</p></div></div>', unsafe_allow_html=True)


PART_GROUPS = [   # part-to-whole order: ready first, then rising urgency
    ('Ready', TOKENS['sky'], "Passed every check."),
    ('Low', LOW_COLOR, "Unusual every year for that country. Likely real."),
    ('Medium', MEDIUM_COLOR, "Unusual for that country and year. Check the source."),
    ('High', ANOMALY_COLOR, "Breaks a basic rule. Fix before use."),
]


def render_part_to_whole(rows, counts, ready):
    """One bar for all records: a neutral outlined track (ready) with colored segments for the rest."""
    data = pd.DataFrame({'group': [g for g, _, _ in PART_GROUPS],
                         'records': [ready] + [int(counts.get(g, 0)) for g, _, _ in PART_GROUPS[1:]]})
    data['share'] = data['records'] / rows
    data['x2'] = data['records'].cumsum()
    data['x'] = data['x2'] - data['records']
    scale = alt.Scale(domain=[0, rows])
    track = alt.Chart(pd.DataFrame({'x': [0], 'x2': [rows]})).mark_bar(
        height=34, color=TOKENS['sky'], stroke=TOKENS['teal'], strokeWidth=1, cornerRadius=6).encode(
        x=alt.X('x:Q', axis=None, scale=scale), x2='x2:Q')
    attention = data[(data['group'] != 'Ready') & (data['records'] > 0)]
    segments = alt.Chart(attention).mark_bar(height=34, strokeWidth=1.5).encode(
        stroke=outlined('group', {g: c for g, c, _ in PART_GROUPS[1:]}),
        x=alt.X('x:Q', axis=None, scale=scale), x2='x2:Q',
        color=alt.Color('group:N', scale=alt.Scale(domain=[g for g, _, _ in PART_GROUPS[1:]],
                                                   range=[c for _, c, _ in PART_GROUPS[1:]]), legend=None),
        tooltip=[alt.Tooltip('group:N', title='Group'), alt.Tooltip('records:Q', title='Records', format=','),
                 alt.Tooltip('share:Q', title='Share', format='.1%')])
    show_chart((track + segments).properties(height=44), width='stretch')
    items = ''.join(
        f'<li><span class="sw" style="background:{color};'
        f'{"border:1.5px solid " + OUTLINE[color] if color in OUTLINE else ""}"></span><span>'
        f'<span class="name">{name}</span> <span class="num">{n:,} ({n / rows:.1%})</span>'
        f'<span class="meaning">{meaning}</span></span></li>'
        for (name, color, meaning), n in zip(PART_GROUPS, data['records']))
    st.markdown(f'<ul class="legend" aria-label="Records by group">{items}</ul>', unsafe_allow_html=True)


def render_look_first(queue):
    urgent = queue[queue['priority'].isin(['High', 'Medium'])]
    if urgent.empty:
        st.markdown('<p class="section-note">Nothing urgent. Low-priority records are listed in '
                    '<b>Review Records</b>.</p>', unsafe_allow_html=True)
        return
    years = urgent.groupby('country')['year'].agg(['size', 'min', 'max'])
    items = []
    for _, r in urgent.drop_duplicates('country').head(4).iterrows():
        n, first, last = years.loc[r['country']]
        when = (f"{r['year']}" if n == 1 else f"{first}, {n} records" if first == last
                else f"{n} years, {first}–{last}")
        color = BADGE_TEXT[r['priority']]
        label = r.get('unusualness_label', '')
        items.append(
            f'<li><div class="head"><span class="country">{html.escape(str(r["country"]))}</span>'
            f'<span class="badge" style="color:{color}">{html.escape(priority_label(r["priority"]))}</span></div>'
            f'<div class="when">{html.escape(when)}'
            f'{" · Unusualness " + html.escape(str(label)) if label else ""}</div>'
            f'<p class="what">{html.escape(r["what_looks_wrong"])}</p></li>')
    st.markdown(f'<ul class="look">{"".join(items)}</ul>', unsafe_allow_html=True)


# Colors for the year chart (as specified): Medium = rust, Low = sky (outlined in teal, since sky
# alone is too pale on white). Rust vs sky is the strongest pair in the theme for every type of
# color blindness (delta E > 30). High appears only when rules are broken; it uses periwinkle here.
YEAR_COLORS = {'High': TOKENS['periwinkle'], 'Medium': TOKENS['rust'], 'Low': TOKENS['sky']}


def render_flags_by_year(queue, scored):
    table = flags_by_year(queue, scored)
    if table.empty:
        st.markdown('<p class="section-note">No records flagged.</p>', unsafe_allow_html=True)
        return
    table['label'] = table['priority'].map(priority_label)
    order = [p for p in ['High', 'Medium', 'Low'] if p in set(table['priority'])]
    stack = {p: i for i, p in enumerate(order)}          # High at the base, then Medium, then Low
    table['stack'] = table['priority'].map(stack)
    chart = alt.Chart(table).mark_bar(strokeWidth=1).encode(
        stroke=outlined('priority', {p: YEAR_COLORS[p] for p in order}),
        x=alt.X('year:O', title=None, axis=alt.Axis(labelAngle=0, values=list(range(2000, 2026, 5)))),
        y=alt.Y('share:Q', stack='zero', title="Share of each year's records",
                axis=alt.Axis(format='%', tickCount=5)),
        color=alt.Color('priority:N', scale=alt.Scale(domain=order, range=[YEAR_COLORS[p] for p in order]),
                        legend=alt.Legend(title=None, orient='top', labelExpr="datum.label + ' priority'")),
        order=alt.Order('stack:Q'),
        tooltip=[alt.Tooltip('year:O', title='Year'), alt.Tooltip('label:N', title='Priority'),
                 alt.Tooltip('records:Q', title='Records'), alt.Tooltip('total:Q', title='Records checked'),
                 alt.Tooltip('share:Q', title='Share of the year', format='.1%')],
    ).properties(height=330)
    show_chart(chart, width='stretch')
    st.markdown(f'<p class="section-note"><b>Insight:</b> {html.escape(year_insight(queue, scored))}</p>',
                unsafe_allow_html=True)


def render_dashboard(summary, queue, evaluation, coverage, scored):
    level, message = readiness_verdict(queue)
    status_card(level, message)

    counts = queue['priority'].value_counts() if len(queue) else pd.Series(dtype=int)
    rows = summary['rows']
    ready = rows - (queue['row'].nunique() if len(queue) else 0)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Records checked", f"{rows:,}", help="One record = one country in one year.")
    k2.metric("Ready to use", f"{ready / rows:.0%}", help=f"{ready:,} records passed every check.")
    k3.metric("Need a check", f"{int(counts.get('Medium', 0)):,}",
              help="Unusual for that country and year. A person should compare them with the source. "
                   f"Another {int(counts.get('Low', 0)):,} records are from countries that are unusual every "
                   "year (low priority).")
    k4.metric("Must fix", f"{int(counts.get('High', 0)):,}",
              help="Records that break a basic rule, such as a missing, duplicate or impossible value.")

    with st.container(key="card_records"):
        st.markdown("#### Where the records stand")
        st.markdown(f'<p class="section-note">All {rows:,} records, from ready to most urgent.</p>',
                    unsafe_allow_html=True)
        render_part_to_whole(rows, counts, ready)

    left, right = st.columns([1.15, 1], gap="medium")
    with left, st.container(key="card_takeaways"):
        st.markdown("#### Key takeaways")
        for item in key_takeaways(summary, queue, evaluation=evaluation, coverage=coverage):
            st.markdown(f"- {item}")
    with right, st.container(key="card_look"):
        st.markdown("#### Where to look first")
        st.markdown('<p class="section-note">One line per country, most urgent first. '
                    'The full list is in <b>Review Records</b>.</p>', unsafe_allow_html=True)
        render_look_first(queue)

    left, right = st.columns([1, 1.15], gap="medium")
    with left, st.container(key="card_years"):
        st.markdown("#### Flagged records by year")
        st.markdown('<p class="section-note">Share of each year\'s records flagged for review, by priority.</p>',
                    unsafe_allow_html=True)
        render_flags_by_year(queue, scored)
    with right, st.container(key="card_reliability"):
        st.markdown("#### How reliable is the checker?")
        per_type = read_report("evaluation_test.csv")
        if evaluation and per_type is not None:
            per_type = per_type.assign(mistake=per_type['error_type'].map(ERROR_LABELS))
            st.markdown(f'<p class="section-note">{evaluation["n_per_type"]} copies of common data quality '
                        'mistakes were artificially injected into the data on countries the checker had never '
                        'seen. To determine reliability, the checker was tested on this data and the proportion '
                        'of mistakes it found is below.</p>', unsafe_allow_html=True)
            show_chart(labelled_bars(per_type, 'recall', 'mistake', value_format='.0%', domain=[0, 1.14],
                                          axis_title='Share of hidden mistakes found'), width='stretch')
            st.caption(f"False alarms: {evaluation['test']['false_positive_rate']:.1%} of good records were flagged "
                       "by mistake. See **Test the Checker** for details.")
        else:
            st.info("Run `python evaluation.py` to measure how reliable the checker is.")

    if coverage:
        with st.container(key="card_coverage"):
            total = coverage['countries_checked'] + coverage['countries_unchecked']
            share = coverage['countries_checked'] / total if total else 0
            st.markdown("#### Coverage")
            st.markdown(
                f'<p class="section-note"><b>{coverage["countries_checked"]:,} of {total:,} countries checked</b> '
                f'({share:.0%}). {coverage["countries_unchecked"]} could not be checked; see <b>Data Coverage</b>.'
                f'</p><div class="meter" role="img" aria-label="{coverage["countries_checked"]} of {total} '
                f'countries checked"><span style="width:{share * 100:.1f}%"></span></div>',
                unsafe_allow_html=True)
            st.caption("Countries were excluded when a key measure was never reported for them. In practice, "
                       "every excluded country was missing at least half of its values, and every included country "
                       "was complete. The quality review engine needs a full history for each country to learn "
                       "what normal looks like, and filling in a whole missing series would mean inventing data.")


# ---------------------------------------------------------------------------
# Tab: Emissions Map
# ---------------------------------------------------------------------------
def _fmt_value(v):
    if pd.isna(v):
        return "no data"
    if abs(v) >= 1000:
        return f"{v:,.0f}"
    return f"{v:.3g}"


def map_figure(frame, measure, selected_iso):
    lo, hi = frame['color_value'].min(), frame['color_value'].max()
    ticks = list(range(math.floor(lo), math.ceil(hi) + 1)) or [lo]
    labels = frame['unusualness'].map(unusualness_label)
    fig = go.Figure(go.Choropleth(
        locations=frame['iso_code'], z=frame['color_value'], text=frame['country'],
        customdata=np.stack([frame['value'].map(_fmt_value), labels], axis=-1),
        colorscale=MAP_SCALE, zmin=lo, zmax=hi, marker_line_color=TOKENS['card'], marker_line_width=0.5,
        colorbar=dict(title=dict(text=friendly_name(measure), side='right'), thickness=12, len=0.75,
                      tickvals=ticks, ticktext=[_fmt_value(10 ** t) for t in ticks]),
        hovertemplate="<b>%{text}</b><br>" + friendly_name(measure) +
                      ": %{customdata[0]}<br>Unusualness: %{customdata[1]}<extra></extra>",
        name='Countries', showlegend=False,
    ))
    flagged = frame[frame['flagged']]
    if len(flagged):
        fig.add_trace(go.Scattergeo(
            locations=flagged['iso_code'], mode='markers', name='Flagged record',
            text=flagged['country'], customdata=flagged['unusualness'].map(unusualness_label),
            marker=dict(symbol='triangle-up', size=14, color=ANOMALY_COLOR, line=dict(color='#ffffff', width=2)),
            hovertemplate="<b>%{text}</b><br>Flagged · unusualness %{customdata}<extra></extra>"))
    if selected_iso in set(frame['iso_code']):
        fig.add_trace(go.Choropleth(
            locations=[selected_iso], z=[0], colorscale=[[0, 'rgba(0,0,0,0)'], [1, 'rgba(0,0,0,0)']],
            showscale=False, marker_line_color=TOKENS['ink'], marker_line_width=2.5, hoverinfo='skip',
            name='Selected country', showlegend=False))
    fig.update_geos(showframe=False, showcoastlines=False, showcountries=False, showland=True,
                    landcolor='#d5dde0', bgcolor='rgba(0,0,0,0)', projection_type='natural earth',
                    lataxis_range=[-57, 84])
    fig.update_layout(height=390, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor='rgba(0,0,0,0)',
                      legend=dict(orientation='h', x=0, y=0.02, bgcolor='rgba(255,255,255,0.9)',
                                  font=dict(size=13, color=TOKENS['ink'])),
                      font=dict(family='sans-serif', color=TOKENS['ink']))
    return fig


def trend_chart(trend, measure, log_scale):
    y_scale = alt.Scale(type='log') if log_scale else alt.Scale(zero=False)
    name = friendly_name(measure)
    positive = trend if not log_scale else trend[trend['value'] > 0]
    base = alt.Chart(positive).encode(x=alt.X('year:O', title=None, axis=alt.Axis(labelAngle=0,
                                                                                  values=list(range(2000, 2026, 5)))))
    band = base.mark_area(color=CONTEXT_BAND, opacity=0.55).encode(
        y=alt.Y('typical_low:Q', scale=y_scale, title=name), y2='typical_high:Q',
        tooltip=[alt.Tooltip('year:O'), alt.Tooltip('typical_low:Q', title='Typical low', format=',.3~g'),
                 alt.Tooltip('typical_high:Q', title='Typical high', format=',.3~g')])
    line = base.mark_line(color=SERIES_BLUE, strokeWidth=2.5).encode(y=alt.Y('value:Q', scale=y_scale))
    pts = positive.assign(status=np.where(positive['flagged'], 'Flagged', 'Not flagged'),
                          unusual=positive['unusualness'].map(unusualness_label),
                          change=positive['pct_change'])
    points = alt.Chart(pts).mark_point(filled=True, size=90).encode(
        x='year:O', y=alt.Y('value:Q', scale=y_scale),
        color=alt.Color('status:N', scale=alt.Scale(domain=['Not flagged', 'Flagged'],
                                                    range=[SERIES_BLUE, ANOMALY_COLOR]),
                        legend=alt.Legend(title=None, orient='top')),
        shape=alt.Shape('status:N', scale=alt.Scale(domain=['Not flagged', 'Flagged'],
                                                    range=['circle', 'triangle-up']), legend=None),
        tooltip=[alt.Tooltip('year:O', title='Year'), alt.Tooltip('value:Q', title=name, format=',.4~g'),
                 alt.Tooltip('change:Q', title='Change from last year', format='+.1%'),
                 alt.Tooltip('unusual:N', title='Unusualness'), 'status'])
    return (band + line + points).properties(height=300)


def render_map(scored, queue):
    measures = available_measures(scored)
    if not measures:
        st.info("This file has no emission measures to map.")
        return
    countries = sorted(scored['country'].unique())
    iso_to_country = dict(zip(scored['iso_code'], scored['country']))

    # A click on the map picks a country (applied before the country menu is drawn)
    event = st.session_state.get('emissions_map')
    points = (event or {}).get('selection', {}).get('points', []) if isinstance(event, dict) else \
        (getattr(getattr(event, 'selection', None), 'points', None) or [])
    clicked = iso_to_country.get(points[0].get('location')) if points else None
    if clicked and clicked != st.session_state.get('_map_last_click'):
        st.session_state['map_country'] = clicked
        st.session_state['_map_last_click'] = clicked
    if st.session_state.get('map_country') not in countries:
        st.session_state['map_country'] = default_country(queue, countries)

    st.markdown("Choose an emission type, a country, and the year. Rust triangles mark records the quality "
                "review engine flagged. The graph on the right demonstrates the trend for that emission over time.")
    c1, c2, c3 = st.columns([1.1, 1.1, 1.6])
    country = c2.selectbox("Country", countries, key='map_country')
    # Open on the emission type that caused this country's flags, so the chart explains the flag
    drivers = scored.loc[(scored['country'] == country) & scored['is_anomaly'], 'top_driver']         if 'top_driver' in scored else pd.Series(dtype=object)
    drivers = drivers[drivers.isin(measures)]
    start = measures.index(drivers.mode().iloc[0]) if len(drivers) else 0
    measure = c1.selectbox("Emission type", measures, index=start,
                           format_func=lambda m: friendly_name(m)[:1].upper() + friendly_name(m)[1:])
    years = sorted(scored['year'].unique())
    flagged_years = scored.loc[(scored['country'] == country) & scored['is_anomaly'], 'year']
    year = c3.select_slider("Map year", options=years,
                            value=int(flagged_years.max()) if len(flagged_years) else int(max(years)))

    frame = map_frame(scored, measure, year)
    selected_iso = scored.loc[scored['country'] == country, 'iso_code'].iloc[0]
    trend = trend_frame(scored, country, measure)

    left, right = st.columns([1.35, 1], gap="large")
    with left:
        st.markdown(f"**{friendly_name(measure)[:1].upper() + friendly_name(measure)[1:]} by country, {year}**")
        st.plotly_chart(map_figure(frame, measure, selected_iso), key='emissions_map', on_select='rerun',
                        selection_mode='points', width='stretch',
                        config={'displayModeBar': False, 'scrollZoom': False})
        st.caption("Darker teal = higher value. Colors use a log scale because countries differ by thousands "
                   "of times. Grey = no data.")
    with right:
        st.markdown(f"**{country}: {friendly_name(measure)} over time**")
        log_scale = st.checkbox("Log scale", value=True, key='trend_log',
                                help="Keeps the light blue typical range and this country readable on one chart.")
        show_chart(trend_chart(trend, measure, log_scale), width='stretch')
        st.caption("Light blue band: the middle half of all countries that year. Teal line: this country.")
        st.caption("Flags apply to the whole record; the reason is shown below the charts.")

    st.markdown(f"**What this shows:** {trend_summary(trend, measure)}")
    record = trend[trend['year'] == year]
    if len(record):
        r = record.iloc[0]
        k1, k2, k3 = st.columns(3)
        k1.metric(f"{country}, {year}", EmissionsDataValidator._fmt(r['value'], measure),
                  help=friendly_name(measure))
        k2.metric("Change from last year", "n/a" if pd.isna(r['pct_change']) else f"{r['pct_change']:+.1%}")
        k3.metric("Unusualness", unusualness_label(r['unusualness']), help=UNUSUALNESS_HELP)
        match = queue[(queue['country'] == country) & (queue['year'] == year)] if len(queue) else queue
        if len(match):
            m = match.iloc[0]
            st.info(f"**Flagged ({m['priority']} priority):** {m['what_looks_wrong']} {m['next_step']}")


# ---------------------------------------------------------------------------
# Tab 2: Review Records (the work queue)
# ---------------------------------------------------------------------------
def render_review(queue, scored, contributions):
    st.markdown("Every record that needs attention, most urgent first. Filter the list, download it, "
                "or choose a record to see why it was flagged.")
    c1, c2, c3 = st.columns([1, 1.4, 1.4])
    priorities = c1.multiselect("Priority", list(PRIORITY_STYLE), default=['High', 'Medium'],
                                format_func=priority_label)
    countries = c2.multiselect("Country", sorted(queue['country'].unique()) if len(queue) else [],
                               placeholder="All countries")
    issues = c3.multiselect("Issue", sorted(queue['issue'].unique()) if len(queue) else [],
                            placeholder="All issues")
    view = queue[queue['priority'].isin(priorities)] if len(queue) else queue
    if countries:
        view = view[view['country'].isin(countries)]
    if issues:
        view = view[view['issue'].isin(issues)]

    st.caption(f"Showing {len(view):,} of {len(queue):,} records.")
    table = view.assign(priority=view['priority'].map(priority_label))[
        ['priority', 'country', 'year', 'issue', 'what_looks_wrong', 'next_step', 'unusualness_label',
         'pattern', 'check']]
    st.dataframe(table, hide_index=True, width='stretch', column_config={
        'priority': st.column_config.TextColumn('Priority', width='small'),
        'country': 'Country', 'year': st.column_config.NumberColumn('Year', format='%d'),
        'issue': 'Issue',
        'what_looks_wrong': st.column_config.TextColumn('What looks wrong', width='large'),
        'next_step': st.column_config.TextColumn('Next step', width='large'),
        'unusualness_label': st.column_config.TextColumn('Unusualness', help=UNUSUALNESS_HELP),
        'pattern': 'How often', 'check': 'Found by',
    })
    st.download_button("Download this review list (CSV)", table.to_csv(index=False), "review_list.csv", "text/csv",
                       type="primary")

    st.divider()
    st.markdown("#### Look closer at one record")
    if view.empty:
        st.info("No records match these filters.")
        return
    labels = {r['row']: f"{priority_label(r['priority'])} · {r['country']} {r['year']} · {r['issue']}"
              for _, r in view.iterrows()}
    choice = st.selectbox("Choose a record to investigate", list(labels), format_func=labels.get)
    rec = view[view['row'] == choice].iloc[0]
    c1, c2, c3 = st.columns([1.2, 1.2, 0.8])
    c1.markdown(f"**What looks wrong**  \n{rec['what_looks_wrong']}")
    c2.markdown(f"**What to do next**  \n{rec['next_step']}")
    c3.markdown(f"**Unusualness**  \n{rec['unusualness_label']}", help=UNUSUALNESS_HELP)
    if choice not in scored.index:
        st.caption("This record broke a basic rule, so it was set aside before the statistical checks.")
        return
    row = scored.loc[choice]
    if row.get('eda_review'):
        st.info(f"This year was also a sudden jump or drop in the country's own history ({row['eda_review']}). "
                "Records that are unusual in two ways are the strongest leads.")

    driver = row.get('top_driver') if isinstance(row.get('top_driver'), str) else None
    left, right = st.columns(2, gap="large")
    if driver:
        with left:
            st.markdown(f"**{row['country']}: {friendly_name(driver)} over time**")
            history = scored[scored['country'] == row['country']].sort_values('year').assign(
                status=lambda d: np.where(d['is_anomaly'], 'Flagged', 'Not flagged'))
            line = alt.Chart(history).mark_line(color=SERIES_BLUE, strokeWidth=2).encode(
                x=alt.X('year:O', title=None, axis=alt.Axis(labelAngle=0, values=list(range(2000, 2026, 5)))),
                y=alt.Y(f'{driver}:Q', title=friendly_name(driver)))
            points = alt.Chart(history).mark_point(filled=True, size=80).encode(
                x='year:O', y=f'{driver}:Q',
                color=alt.Color('status:N', scale=alt.Scale(domain=['Not flagged', 'Flagged'],
                                                            range=[SERIES_BLUE, ANOMALY_COLOR]),
                                legend=alt.Legend(title=None, orient='top')),
                shape=alt.Shape('status:N', scale=alt.Scale(domain=['Not flagged', 'Flagged'],
                                                            range=['circle', 'triangle-up']), legend=None),
                tooltip=['year', alt.Tooltip(f'{driver}:Q', format=',.4~g'), 'status'])
            show_chart((line + points).properties(height=280), width='stretch')
            st.caption("How to read this: rust triangles are flagged years. If only one or two years jump "
                       "away from the rest of the line, those values may have been entered incorrectly. If most "
                       "years are flagged, the country is consistently different from other countries, which is "
                       "usually real rather than an error.")
    if contributions is not None and choice in contributions.index:
        with right:
            st.markdown("**Factors that influenced a manual review**")
            contrib = contributions.loc[choice].rename('push').reset_index().rename(columns={'index': 'feature'})
            contrib['name'] = contrib['feature'].map(friendly_name)
            contrib['effect'] = np.where(contrib['push'] > 0, 'Made it look unusual', 'Made it look normal')
            chart = alt.Chart(contrib).mark_bar(cornerRadiusEnd=4).encode(
                x=alt.X('push:Q', title='Effect on the flag (SHAP value)'),
                y=alt.Y('name:N', sort='-x', title=None, axis=alt.Axis(labelLimit=260)),
                color=alt.Color('effect:N', scale=alt.Scale(domain=['Made it look unusual', 'Made it look normal'],
                                                            range=[TOWARD_ANOMALY, TOWARD_NORMAL]),
                                legend=alt.Legend(title=None, orient='top')),
                tooltip=['name', 'effect', alt.Tooltip('push:Q', format='.3f')],
            ).properties(height=alt.Step(30))
            show_chart(chart, width='stretch')
            st.caption("Rust bars pushed this record toward review; teal bars pushed it toward normal. Longer bars had more influence.")


# ---------------------------------------------------------------------------
# Tab 3: Test the Checker (measured performance + improvement story)
# ---------------------------------------------------------------------------
def default_settings_text(evaluation):
    range_text = 'off' if evaluation['range_z'] is None else f"strictness {evaluation['range_z']:g}"
    return (f"{TRANSFORM_NAMES.get(evaluation['transform'], evaluation['transform']).lower()} scaling, per-person "
            f"and per-dollar ratios {'on' if evaluation['use_ratios'] else 'off'}, "
            f"{evaluation['contamination']:.0%} flag share, range check {range_text}")


def render_test_the_checker(scored, evaluation):
    st.markdown("#### Out-of-sample evaluation on held-out countries")
    if not evaluation:
        st.info("Run `python evaluation.py` to measure performance.")
        return
    # Countries in the data the model actually used (after cleaning), not the pre-cleaning split
    n = scored.groupby('split')['country'].nunique().to_dict() if 'split' in scored else {}
    grid = read_report("evaluation_grid.csv")
    count = lambda k: f", {n[k]} countries" if k in n else ""
    st.markdown(
        "Before any modeling, countries were divided into three data sets. The split was **grouped** by country, "
        "so all years of a country fall in the same set and no country appears in two sets. It was also "
        "**stratified** by size (quartiles of each country's median CO₂), so each set has the same mix of "
        "small and large emitters. Countries without enough data were then excluded (see Data Coverage).\n\n"
        f"- **Training set** (60%{count('train')}): used to fit the model and learn the normal range of values.\n"
        f"- **Validation set** (20%{count('validation')}): used to compare "
        f"{len(grid) if grid is not None else 'the candidate'} model settings and select the best one.\n"
        f"- **Test set** (20%{count('test')}): held out and scored once, after the settings were fixed.\n\n"
        "Known errors were injected into the validation and test sets, and each set was scored on how many "
        "errors it caught and how many correct records it wrongly flagged. Because the test countries played no "
        "part in training or in choosing settings, the test results estimate performance on new data.")
    st.info(f"These reliability scores were measured for the settings this app uses "
            f"({default_settings_text(evaluation)}).")
    test, val = evaluation['test'], evaluation['validation']
    c1, c2, c3 = st.columns(3)
    c1.metric("Hidden mistakes found", f"{test['recall_all']:.0%}", help="All seven types of mistake, test set.")
    c2.metric("Mistakes rules alone would miss, found", f"{test['recall_statistical']:.0%}",
              help="Wrong units, misplaced decimals and swapped columns: every value looks possible on its own.")
    over = test['false_positive_rate'] - evaluation['fpr_budget']
    c3.metric("Good records flagged by mistake", f"{test['false_positive_rate']:.1%}",
              f"{over * 100:+.1f} points vs the {evaluation['fpr_budget']:.0%} target", delta_color="inverse")
    if over > 0:
        st.warning(f"The false positive rate was {val['false_positive_rate']:.1%} on the validation set and "
                   f"{test['false_positive_rate']:.1%} on the test set, above the {evaluation['fpr_budget']:.0%} "
                   "target. Two test-set countries account for this: their values are unusual in every year, so "
                   "each of their years is flagged. The review list now ranks this pattern as low priority. The "
                   "model settings were not changed after the test results were seen, because tuning on the test "
                   "set would bias the estimate.")

    frames = [read_report(f"evaluation_{s}.csv") for s in ('validation', 'test')]
    if all(f is not None for f in frames):
        per_type = pd.concat(frames)
        per_type['group'] = per_type['split'].map({'validation': 'Validation set', 'test': 'Test set'})
        per_type['mistake'] = per_type['error_type'].map(ERROR_LABELS)
        order = list(frames[1].sort_values('recall', ascending=False)['error_type'].map(ERROR_LABELS))
        chart = alt.Chart(per_type).mark_bar(cornerRadiusEnd=4).encode(
            x=alt.X('recall:Q', title='Share of hidden mistakes found', axis=alt.Axis(format='%'),
                    scale=alt.Scale(domain=[0, 1])),
            y=alt.Y('mistake:N', title=None, sort=order, axis=alt.Axis(labelLimit=320, labelFontSize=13)),
            yOffset=alt.YOffset('group:N', sort=['Validation set', 'Test set']),
            stroke=outlined('group', {'Validation set': MEDIUM_COLOR, 'Test set': SERIES_BLUE}),
            color=alt.Color('group:N', scale=alt.Scale(domain=['Validation set', 'Test set'],
                                                       range=[MEDIUM_COLOR, SERIES_BLUE]),
                            legend=alt.Legend(title=None, orient='top')),
            tooltip=['mistake', 'group', alt.Tooltip('recall:Q', format='.0%')],
        ).properties(height=alt.Step(18))
        show_chart(chart, width='stretch')
        st.caption("Similar results on the validation and test sets indicate the checker generalizes to new "
                   "countries rather than fitting only the training data.")

    st.divider()
    render_improvement_story(evaluation)


def render_improvement_story(evaluation):
    """Quality improvement: measure, find the gap, change one thing, measure again."""
    grid = read_report("evaluation_grid.csv")
    if grid is None:
        return
    budget = evaluation['fpr_budget']
    fair = grid[grid['false_positive_rate'] <= budget]
    baseline = fair[fair['range_z'].isna()].sort_values('recall_statistical', ascending=False).iloc[0]
    before, after = baseline['recall_statistical'], evaluation['validation']['recall_statistical']
    before_all, after_all = baseline['recall_all'], evaluation['validation']['recall_all']
    st.markdown("#### Quality improvement: measure, fix, measure again")
    st.markdown(
        "1. **Measure.** The first version found only "
        f"**{before:.0%}** of the mistakes that rules alone miss.\n"
        "2. **Find the cause.** Hidden wrong-unit mistakes were often missed. The pattern check only knows the "
        "range of values it studied, so a number 1,000 times too big looked like the largest real country.\n"
        "3. **Change one thing.** A range check was added that flags values far outside anything seen before.\n"
        f"4. **Measure again.** On the same test, the checker now finds **{after:.0%}** of the mistakes that "
        f"rules alone miss, and **{after_all:.0%}** of all hidden mistakes (up from {before_all:.0%}), while "
        f"keeping false alarms under the {budget:.0%} target on the validation set.")
    groups = ['Mistakes rules alone miss', 'All hidden mistakes']
    versions = ['Before: pattern check only', 'After: + range check']
    data = pd.DataFrame({'measure': [groups[0], groups[0], groups[1], groups[1]],
                         'version': versions * 2, 'found': [before, after, before_all, after_all]})
    base = alt.Chart(data).encode(
        x=alt.X('found:Q', title='Share of hidden mistakes found (validation set)',
                axis=alt.Axis(format='%', values=[0, 0.25, 0.5, 0.75, 1]),
                scale=alt.Scale(domain=[0, 1.12])),
        y=alt.Y('measure:N', title=None, sort=groups, axis=alt.Axis(labelLimit=320, labelFontSize=13)),
        yOffset=alt.YOffset('version:N', sort=versions),
        tooltip=['measure', 'version', alt.Tooltip('found:Q', format='.0%')])
    bars = base.mark_bar(cornerRadiusEnd=4, strokeWidth=1).encode(
        stroke=outlined('version', dict(zip(versions, [MEDIUM_COLOR, SERIES_BLUE]))),
        color=alt.Color('version:N', scale=alt.Scale(domain=versions, range=[MEDIUM_COLOR, SERIES_BLUE]),
                        legend=alt.Legend(title=None, orient='top')))
    text = base.mark_text(align='left', dx=6, fontSize=13, color=INK_MUTED).encode(
        text=alt.Text('found:Q', format='.0%'))
    show_chart((bars + text).properties(height=alt.Step(26)), width='stretch')


# ---------------------------------------------------------------------------
# Tab 4: Data Coverage
# ---------------------------------------------------------------------------
def render_coverage():
    st.markdown("A quality review can only occur if enough data exists. This page lists what data is missing. "
                "Countries were excluded if they met the missing data threshold: a key measure (CO₂ emissions, "
                "methane emissions, greenhouse gas per person or population) missing in every year, that is, "
                "100% missing for that measure.")
    summary = read_report("completeness.csv")
    if summary is None:
        st.info("Run `python data_prep.py` to create the coverage report.")
        return
    summary = summary.assign(measure=summary['feature'].map(friendly_name),
                             share=summary['pct_values_present'] / 100)
    st.markdown("#### How complete is each measure?")
    show_chart(labelled_bars(summary, 'share', 'measure', value_format='.0%', domain=[0, 1.12],
                                  axis_title='Share of country-years with a value'), width='stretch')

    by_country = read_report("completeness_by_country.csv")
    if by_country is not None:
        missing = by_country[by_country['status'] != 'complete'].copy()
        missing['missing'] = missing['features_missing_entirely'].fillna('').map(
            lambda s: ', '.join(friendly_name(f) for f in s.split(', ') if f))
        st.markdown(f"#### Countries that could not be checked ({len(missing)})")
        st.caption("Most are small islands and territories. Together they make up a very small share of "
                   "world emissions, however it is important to note which countries were excluded and manually "
                   "review them to identify any systematic exclusions or patterns that may limit the use of this "
                   "engine.")
        st.dataframe(missing[['country', 'missing', 'pct_values_present']], hide_index=True, width='stretch',
                     column_config={'country': 'Country', 'missing': 'Missing entirely',
                                    'pct_values_present': st.column_config.ProgressColumn(
                                        'Values present', format='%.0f%%', min_value=0, max_value=100)})


# ---------------------------------------------------------------------------
# Tab 5: Methods (technical detail for data scientists)
# ---------------------------------------------------------------------------
def setting_comparison(grid, budget):
    """Best share of rule-invisible mistakes found for each option, within the false-alarm limit."""
    fair = grid[grid['false_positive_rate'] <= budget].copy()
    fair['range_z'] = fair['range_z'].map(lambda z: 'Off' if pd.isna(z) else f"Strictness {z:g}")
    rows = []
    for setting, col, label in [
        ('Scaling method', 'transform', lambda v: TRANSFORM_NAMES.get(v, v)),
        ('Per-person and per-dollar ratios', 'use_ratios', lambda v: 'On' if v else 'Off'),
        ('Flag share', 'contamination', lambda v: f"{v:.0%}"),
        ('Range check', 'range_z', lambda v: v),
    ]:
        values = grid[col].map(lambda z: 'Off' if pd.isna(z) else f"Strictness {z:g}") if col == 'range_z' \
            else grid[col]
        options = sorted(values.unique(), key=str)
        best = fair.groupby(col)['recall_statistical'].max()
        chosen = fair.sort_values('recall_statistical', ascending=False).iloc[0][col] if len(fair) else None
        for option in options:
            score = best.get(option, np.nan)
            note = ('Default (chosen)' if option == chosen else
                    f'Never met the {budget:.0%} false-alarm limit' if pd.isna(score) else '')
            rows.append({'Setting': setting, 'Option': label(option), 'Best share found': score, 'Note': note})
    return pd.DataFrame(rows)


def render_study_design(evaluation):
    grid = read_report("evaluation_grid.csv")
    st.markdown("#### Study design")
    if not evaluation or grid is None:
        st.info("Run `python evaluation.py` to produce the study results.")
        return
    budget = evaluation['fpr_budget']
    st.markdown(
        "Instead of choosing settings by habit or by how the data looked, every combination of settings was "
        "tested and scored on the actual goal: catching data mistakes without flagging too many good records. "
        "This matters because the setting that looks best on paper is not always the one that works best. For "
        "example, the scaling method that made the data look most evenly spread was not the one that caught the "
        "most mistakes.\n\n"
        f"**How it was tested.** {evaluation['n_per_type']} copies of each of 7 common mistakes were injected into "
        f"the validation countries, repeated with {len(evaluation['seeds'])} different random draws. Each "
        f"combination was scored on (1) the share of mistakes found that basic rules cannot catch and (2) the share "
        f"of good records flagged by mistake. The winner found the most mistakes while flagging no more than "
        f"{budget:.0%} of good records. It was then scored once on the test countries.")

    st.markdown("#### What was compared")
    st.markdown(f"Four settings were varied, giving {len(grid)} combinations. The table shows the best result each "
                f"option achieved while staying within the {budget:.0%} false-alarm limit.")
    table = setting_comparison(grid, budget)
    st.dataframe(table.assign(**{'Best share found': table['Best share found'] * 100}), hide_index=True,
                 width='stretch', column_config={
                     'Setting': st.column_config.TextColumn(width='medium'),
                     'Option': st.column_config.TextColumn(width='medium'),
                     'Best share found': st.column_config.ProgressColumn(
                         'Best share of mistakes found (rules alone miss)', format='%.0f%%', min_value=0,
                         max_value=100),
                     'Note': st.column_config.TextColumn(width='medium')})
    by = lambda setting: table[table['Setting'] == setting].set_index('Option')['Best share found']
    scaling, ratios, ranges = by('Scaling method'), by('Per-person and per-dollar ratios'), by('Range check')
    scaled = scaling.drop('No scaling', errors='ignore').dropna()
    flag = by('Flag share')
    over = [o for o, v in flag.items() if pd.isna(v)]
    st.markdown(
        "**What the comparison shows.**\n\n"
        f"- **Scaling** mattered a lot, but the choice of method mattered little: no scaling found "
        f"{scaling.get('No scaling', np.nan):.0%}, while the scaling methods found "
        f"{scaled.min():.0%} to {scaled.max():.0%}.\n"
        f"- **Per-person and per-dollar ratios** helped: {ratios.get('On', np.nan):.0%} with them, "
        f"{ratios.get('Off', np.nan):.0%} without.\n"
        f"- **The range check** made the biggest difference: {ranges.max():.0%} at its best setting, "
        f"{ranges.get('Off', np.nan):.0%} when turned off.\n"
        + (f"- **Flag shares of {', '.join(over)}** found more mistakes but flagged more than {budget:.0%} of good "
           "records, so none of them met the false-alarm limit." if over else ""))

    st.markdown("#### Model strictness")
    range_default = 'off' if evaluation['range_z'] is None else f"{evaluation['range_z']:g}"
    st.markdown(
        "Two settings control how strict the checker is. Stricter settings catch more "
        "mistakes but also flag more good records, which means more manual review.\n\n"
        f"- **How many records to flag** (default {evaluation['contamination']:.0%}): the share of normal records "
        "the pattern check treats as suspicious. Raising it flags more records.\n"
        f"- **Range check strictness** (default {range_default}): how far a value must be from normal before it "
        "is flagged. A lower number is stricter.\n\n"
        "The app uses the tested defaults, so the reliability scores on Test the Checker describe exactly "
        "what you see.")


def render_methods(df, scored, features, evaluation):
    st.markdown("#### How the three checks work")
    c1, c2, c3 = st.columns(3)
    c1.markdown("**1. Rule check**  \nHard rules every record must follow: no missing values, no duplicates, "
                "no negative emissions, valid country codes and years. *Tool: pandera.*")
    c2.markdown("**2. Range check**  \nFlags a value far outside anything seen in the study countries. "
                "Good at catching wrong units. *Tool: robust z-score (median and MAD).*")
    c3.markdown("**3. Pattern check**  \nFlags records whose values do not fit together, even when each one "
                "is possible alone. *Tool: Isolation Forest, explained with SHAP.*")
    st.caption("Everything below is the technical detail behind the pages above.")
    render_study_design(evaluation)

    with st.expander("Model settings and all configurations compared"):
        if evaluation:
            range_label = 'off' if evaluation['range_z'] is None else f"z > {evaluation['range_z']:g}"
            st.markdown(f"Transform `{evaluation['transform']}` · ratio features "
                        f"{'on' if evaluation['use_ratios'] else 'off'} · contamination "
                        f"{evaluation['contamination']:.0%} · range check {range_label}.  \n{evaluation['reason']}")
        grid = read_report("evaluation_grid.csv")
        if grid is not None:
            st.dataframe(grid.sort_values('recall_statistical', ascending=False).round(3),
                         hide_index=True, width='stretch')

    with st.expander("Where flagged records sit (scatter plot)"):
        c1, c2, c3 = st.columns([2, 2, 1])
        x = c1.selectbox("X axis", features, index=0, format_func=friendly_name)
        y = c2.selectbox("Y axis", features, index=min(1, len(features) - 1), format_func=friendly_name)
        scale = alt.Scale(type='symlog') if c3.checkbox("Log axes", value=True) else alt.Scale()
        plot_df = scored.assign(status=np.where(scored['is_anomaly'], 'Flagged', 'Not flagged'))
        status = alt.Scale(domain=['Not flagged', 'Flagged'], range=[SERIES_BLUE, ANOMALY_COLOR])
        show_chart(alt.Chart(plot_df).mark_point(filled=True).encode(
            x=alt.X(f'{x}:Q', scale=scale, title=friendly_name(x)),
            y=alt.Y(f'{y}:Q', scale=scale, title=friendly_name(y)),
            color=alt.Color('status:N', scale=status, legend=alt.Legend(title=None, orient='top')),
            shape=alt.Shape('status:N', scale=alt.Scale(domain=['Not flagged', 'Flagged'],
                                                        range=['circle', 'triangle-up']), legend=None),
            size=alt.condition(alt.datum.status == 'Flagged', alt.value(90), alt.value(28)),
            opacity=alt.condition(alt.datum.status == 'Flagged', alt.value(1.0), alt.value(0.4)),
            order=alt.Order('is_anomaly:O'),
            tooltip=['country', 'year', alt.Tooltip(f'{x}:Q', format=',.3~g'), alt.Tooltip(f'{y}:Q', format=',.3~g')],
        ).properties(height=420), width='stretch')

    with st.expander("Data preparation audit trail (every row removed, in order)"):
        log = read_report("cleaning_log.csv")
        if log is not None:
            st.dataframe(log, hide_index=True, width='stretch')
        split = read_report("split_summary.csv")
        if split is not None:
            st.markdown("**Train / validation / test split** (grouped by country, stratified by CO₂ size quartile)")
            st.dataframe(split.pivot_table(index='stratum', columns='split', values='countries', fill_value=0),
                         width='stretch')

    with st.expander("Missing data investigation"):
        diag = read_report("missingness_diagnosis.csv")
        if diag is not None:
            st.caption("Gaps in whole years (reporting lag) or whole countries (never covered) are structural. "
                       "Filling them in would invent data, so they are reported instead.")
            st.dataframe(diag, hide_index=True, width='stretch')
        show_image("missingness_by_year.png", "Missing values by year")

    with st.expander("Outlier investigation"):
        profile = read_report("outlier_profile.csv")
        if profile is not None:
            st.caption("Most extreme values across countries are just skew (a few very large countries), so they "
                       "are kept. Sudden jumps within one country are flagged for review, never deleted.")
            st.dataframe(profile, hide_index=True, width='stretch')
        show_image("outlier_scale_comparison.png", "Outliers on the raw vs log scale")
        show_image("outlier_spikes.png", "Strongest year-over-year events")

    with st.expander("Transformation and feature selection"):
        selection = read_report("transform_selection.csv")
        if selection is not None:
            st.markdown("**Transformation** (Shapiro-Wilk W; 1.0 = perfectly normal)")
            st.dataframe(selection.round(4), hide_index=True, width='stretch')
        lam = read_report("transform_lambda_tests.csv")
        if lam is not None:
            st.markdown("**Likelihood-ratio test, H0: λ = 0 (a log is the best power transform)**")
            st.dataframe(lam, hide_index=True, width='stretch')
        shap_rank = read_report("shap_feature_importance.csv")
        if shap_rank is not None:
            st.markdown("**SHAP feature ranking (training countries)**")
            st.dataframe(shap_rank.round(4), hide_index=True, width='stretch')
        stability = read_report("feature_stability.csv")
        if stability is not None:
            st.markdown("**Selection stability** (share of 30 country-level bootstrap resamples)")
            st.dataframe(stability, hide_index=True, width='stretch')

    with st.expander("Data preview"):
        st.dataframe(df.head(50), hide_index=True, width='stretch')


# ---------------------------------------------------------------------------
# Tab 6: About
# ---------------------------------------------------------------------------
def render_about():
    st.markdown("#### The business problem")
    st.markdown(
        "Companies publish emissions numbers in climate reports and use them to set targets. If a number is wrong "
        "because of a unit mix-up, a typo or a duplicate, the report is wrong too. Just dropping the missing or "
        "unusual values can result in significant data loss. Instead, teams can examine trends and identify how "
        "unusual or anomalous a specific point is. This allows the team to retain more data and produce "
        "higher-quality, more accurate reports. This app filters the data, analyzes it, and produces a report to "
        "streamline which values require manual review or augmentation from another source.\n\n"
        "This quality review engine checks every record automatically and gives the team a short, ranked list to "
        "review. It uses public country data from Our World in Data. The same approach works for supplier or "
        "facility data.")
    st.markdown("#### Goals and Approach")
    st.dataframe(pd.DataFrame([
        ("Quality assurance", "Clear rules every record must pass, with a plain reason for each failure"),
        ("Quality assurance", "Automated tests written before the code (test-driven development, 150+ tests)"),
        ("Quality assurance", "Full audit trail: every removed row and every decision is logged"),
        ("Quality improvement", "Measured performance with hidden mistakes, found a gap, fixed it, measured again"),
        ("Quality improvement", "Out-of-sample testing: grouped, stratified training, validation and test sets; test set scored once"),
        ("Data investigation", "Missing data and outliers studied before deciding how to treat them"),
        ("Communication", "Results explained in plain language, with a prioritized action list"),
    ], columns=['Goal', 'Approach']), hide_index=True, width='stretch')
    st.markdown("#### How it works, step by step")
    st.markdown(
        "1. **Collect**: download the data and save a copy with a fingerprint (checksum), so results can be repeated.\n"
        "2. **Clean**: study the gaps first, then remove only what cannot be checked, and log every step.\n"
        "3. **Check**: three checks (rules, range, pattern) look at every record.\n"
        "4. **Test**: hide known mistakes and measure how many are found on new countries.\n"
        "5. **Report**: turn the results into a readiness verdict and a ranked review list.")
    st.caption("Built with Python, pandas, pandera, scikit-learn, SHAP and Streamlit.")
    st.markdown("#### Data source")
    st.caption(
        "Ritchie, H., Rosado, P., & Roser, M. (2023). *CO₂ and greenhouse gas emissions*. Our World in Data. "
        "[https://ourworldindata.org/co2-and-greenhouse-gas-emissions]"
        "(https://ourworldindata.org/co2-and-greenhouse-gas-emissions)\n\n"
        "Rosado, P., Ritchie, H., Roser, M., Mathieu, E., & Macdonald, B. (n.d.). *Data on CO₂ and greenhouse gas "
        f"emissions by Our World in Data* [Data set]. GitHub. Retrieved {snapshot_date()}, from "
        "[https://github.com/owid/co2-data](https://github.com/owid/co2-data)")


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
def render_hero(df):
    years = f"{int(df['year'].min())}–{int(df['year'].max())}" if 'year' in df and len(df) else ''
    chips = ["Source: Our World in Data",
             f"{df['country'].nunique():,} countries" if 'country' in df else '', years,
             f"Data retrieved {snapshot_date()}"]
    chip_html = ''.join(f'<li>{html.escape(c)}</li>' for c in chips if c)
    st.markdown(
        '<header class="hero"><p class="eyebrow">Sustainability data quality</p>'
        '<h1>Emissions Data Quality Review</h1>'
        '<p class="lede">Before emissions data go into a sustainability report, they need to be checked for quality '
        'and accuracy. This app uses public country data from Our World in Data to screen for anomalies, quality, '
        'and data cleanliness. A report is created to highlight which records need manual review.</p>'
        f'<ul class="chips" aria-label="About this data">{chip_html}</ul></header>', unsafe_allow_html=True)


def coverage_numbers():
    by_country = read_report("completeness_by_country.csv")
    if by_country is None:
        return None
    return {'countries_checked': int(by_country['in_final_dataset'].sum()),
            'countries_unchecked': int((~by_country['in_final_dataset']).sum())}


def main():
    st.set_page_config(page_title="Emissions Data Quality Review", page_icon=":material/public:", layout="wide")
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

    # Settings are the tested defaults chosen by evaluation.py (see Methods > Study design)
    config, _ = recommended_config()
    contamination, transform = config['contamination'], config['transform']
    use_ratios, range_z = config['use_ratios'], config['range_z']

    if not os.path.exists(DEFAULT_DATA):
        st.error("No data found. Run `python data_prep.py` first.")
        st.stop()
    df = load_csv(DEFAULT_DATA)
    render_hero(df)

    missing_ids = [c for c in ID_COLS if c not in df.columns]
    if missing_ids:
        st.error(f"The file is missing required columns: {', '.join(missing_ids)}")
        st.stop()

    summary, failures, scored, contributions, features = run_pipeline(df, contamination, transform, use_ratios, range_z)
    queue = build_review_queue(scored, failures)
    evaluation = read_evaluation()
    coverage = coverage_numbers()

    tabs = st.tabs(["Dashboard", "Emissions Map", "Review Records", "Test the Checker", "Data Coverage",
                    "Methods", "About This Project"])
    with tabs[0]:
        render_dashboard(summary, queue, evaluation, coverage, scored)
    with tabs[1]:
        render_map(scored, queue)
    with tabs[2]:
        render_review(queue, scored, contributions)
    with tabs[3]:
        render_test_the_checker(scored, evaluation)
    with tabs[4]:
        render_coverage()
    with tabs[5]:
        render_methods(df, scored, features, evaluation)
    with tabs[6]:
        render_about()


if __name__ == "__main__":
    main()
