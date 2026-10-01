"""
Accessibility checks for the design tokens.
WCAG 2.2: normal text >= 4.5:1, large text and graphical marks >= 3:1.
Color-vision deficiency: Machado et al. (2009) simulation, OKLab delta E x100.
"""
import os
import tomllib
from itertools import combinations

import pytest

import design
from design import (DATA_COLORS, OUTLINES, SECONDARY_ENCODED, THEME, TOKENS, contrast, delta_e)


def test_contrast_formula_matches_wcag_reference_values():
    assert contrast('#000000', '#ffffff') == pytest.approx(21.0, abs=0.01)
    assert contrast('#ffffff', '#ffffff') == pytest.approx(1.0)
    assert contrast('#767676', '#ffffff') == pytest.approx(4.54, abs=0.02)   # WCAG's classic AA gray


# --- the user's theme is used exactly ----------------------------------------------------
def test_theme_colors_are_the_four_chosen_colors():
    assert THEME == {'teal': '#478978', 'periwinkle': '#8B95C9', 'sky': '#ACD7EC', 'rust': '#9D5E49'}
    assert set(DATA_COLORS.values()) <= set(THEME.values())


# --- text and interactive elements ------------------------------------------------------
@pytest.mark.parametrize('text', ['ink', 'ink_muted', 'primary', 'periwinkle_dark'])
@pytest.mark.parametrize('surface', ['page', 'card', 'sidebar'])
def test_text_meets_aa_on_every_surface(text, surface):
    assert contrast(TOKENS[text], TOKENS[surface]) >= 4.5


def test_button_text_is_readable():
    assert contrast('#ffffff', TOKENS['primary']) >= 4.5
    assert contrast('#ffffff', TOKENS['primary_hover']) >= 4.5


@pytest.mark.parametrize('status', ['ready', 'review', 'not_ready'])
def test_status_cards_are_readable(status):
    tint, edge = design.STATUS[status]['tint'], design.STATUS[status]['edge']
    assert contrast(TOKENS['ink'], tint) >= 4.5
    assert contrast(edge, tint) >= 3.0


# --- data marks --------------------------------------------------------------------------
@pytest.mark.parametrize('color', list(DATA_COLORS.values()) + [THEME['sky']])
def test_every_data_fill_is_visible_on_cards(color):
    # Fills under 3:1 (periwinkle, sky) must be drawn with an outline that reaches 3:1
    edge = color if contrast(color, TOKENS['card']) >= 3.0 else OUTLINES[color]
    assert contrast(edge, TOKENS['card']) >= 3.0


@pytest.mark.parametrize('a, b', list(combinations(DATA_COLORS, 2)))
def test_every_pair_of_data_colors_is_distinguishable(a, b):
    x, y = DATA_COLORS[a], DATA_COLORS[b]
    assert delta_e(x, y) >= 15
    assert delta_e(x, y, 'tritan') >= 6
    for kind in ('protan', 'deutan'):
        # >= 8 is the target; 6-8 is allowed only for pairs that always carry a second cue
        # (icons, shapes or words), listed in SECONDARY_ENCODED
        assert delta_e(x, y, kind) >= 6
        if delta_e(x, y, kind) < 8:
            assert frozenset((a, b)) in SECONDARY_ENCODED, (a, b, kind)


def test_map_ramp_gets_darker_step_by_step():
    lum = [design.luminance(c) for c in design.MAP_RAMP]
    assert lum == sorted(lum, reverse=True)
    assert design.MAP_RAMP[1] == THEME['sky'] and THEME['teal'] in design.MAP_RAMP


@pytest.mark.parametrize('a, b, kind, expected', [
    ('#2b7bb0', '#cf6436', 'protan', 18.0),
    ('#a98a3e', '#cf6436', 'deutan', 1.9),
    ('#8a8577', '#2b7bb0', None, 14.3),
])
def test_delta_e_matches_the_palette_validator(a, b, kind, expected):
    assert delta_e(a, b, kind) == pytest.approx(expected, abs=0.1)


# --- the app and pipeline use only these tokens --------------------------------------------
def test_streamlit_theme_uses_the_same_tokens():
    path = os.path.join(os.path.dirname(design.__file__), '.streamlit', 'config.toml')
    with open(path, 'rb') as f:
        theme = tomllib.load(f)['theme']
    assert theme['primaryColor'] == TOKENS['primary']
    assert theme['backgroundColor'] == TOKENS['page']
    assert theme['textColor'] == TOKENS['ink']


def test_app_draws_data_only_in_theme_colors():
    import app
    allowed = set(DATA_COLORS.values())
    assert {c for c, _, _ in app.PRIORITY_STYLE.values()} <= allowed
    assert {app.SERIES_BLUE, app.ANOMALY_COLOR, app.LOW_COLOR, app.MEDIUM_COLOR} <= allowed
    root = os.path.dirname(app.__file__)
    for name in ['app.py', 'data_prep.py', '.streamlit/config.toml']:
        source = open(os.path.join(root, name), encoding='utf-8').read().lower()
        for retired in ['#2b7bb0', '#cf6436', '#1a9e74', '#2f5d46', '#f6f3ec', '#8a8577', 'normal_gray']:
            assert retired not in source, (name, retired)


# --- Message boxes (st.info / success / warning / error) use the theme (new: written first) --------
@pytest.mark.parametrize('kind', ['info', 'success', 'warning', 'error'])
def test_message_boxes_are_readable_and_match_the_theme_file(kind):
    style = design.ALERTS[kind]
    assert contrast(style['text'], style['background']) >= 4.5
    assert contrast(style['edge'], style['background']) >= 3.0
    path = os.path.join(os.path.dirname(design.__file__), '.streamlit', 'config.toml')
    with open(path, 'rb') as f:
        theme = tomllib.load(f)['theme']
    streamlit_name = {'info': 'blue', 'success': 'green', 'warning': 'yellow', 'error': 'red'}[kind]
    assert theme[f'{streamlit_name}BackgroundColor'] == style['background']
    assert theme[f'{streamlit_name}TextColor'] == style['text']
