"""
Design tokens: the app theme is built from four chosen colors (teal, periwinkle, sky, rust),
with darker shades of the same hues where accessibility needs them. Everything is tested in
tests/test_design.py against WCAG 2.2 contrast and simulated color blindness.

Roles
  - Teal = normal values and Low priority; periwinkle = Medium; rust = flagged and High.
  - Sky = backgrounds, tints, the neutral "Ready" track and context bands; it is too pale to
    carry data on its own, so sky fills are always outlined in teal.
  - Periwinkle fills are just under 3:1 on white, so they are outlined in a darker periwinkle.
  - Buttons, links and focus rings use a darker teal: white text on the original teal is 4.1:1,
    below the 4.5:1 needed for normal-size text.
  - Teal vs rust is the weakest pair for red-green color blindness (delta E 6.6), so wherever they
    appear together a second cue is always shown: triangle vs circle markers, priority icons, words.
"""

THEME = {'teal': '#478978', 'periwinkle': '#8B95C9', 'sky': '#ACD7EC', 'rust': '#9D5E49'}

TOKENS = {
    # surfaces (light tints of sky)
    'page': '#f4f8f9',
    'card': '#ffffff',
    'sidebar': '#e6f2f7',
    'border': '#cfe1e8',
    'grid': '#e2edf1',
    'axis': '#6f868e',
    # text
    'ink': '#1f2d33',
    'ink_muted': '#4f5d63',
    # interactive (darker teal)
    'primary': '#2f6356',
    'primary_hover': '#244e43',
    'primary_tint': '#e3f0ec',
    # data (the chosen theme, plus darker shades for outlines and text)
    **THEME,
    'periwinkle_dark': '#5a64a0',
}

# The only colors that encode data categories
DATA_COLORS = {'low': THEME['teal'], 'medium': THEME['periwinkle'], 'high': THEME['rust']}

# Pairs below the delta E 8 colorblind target that always carry a second cue (icons, shapes, words)
SECONDARY_ENCODED = {frozenset(('low', 'high'))}

# Outlines for fills that are under 3:1 against the card
OUTLINES = {THEME['periwinkle']: TOKENS['periwinkle_dark'], THEME['sky']: THEME['teal']}

STATUS = {
    'ready': {'tint': '#e5f2f8', 'edge': THEME['teal'], 'word': 'Ready'},
    'review': {'tint': '#eceef8', 'edge': TOKENS['periwinkle_dark'], 'word': 'Ready after review'},
    'not_ready': {'tint': '#f5ebe7', 'edge': THEME['rust'], 'word': 'Not ready'},
}

# Message boxes (st.info / success / warning / error): tints of the theme with readable text.
# Wired into Streamlit through .streamlit/config.toml; the edge is drawn by the app's CSS.
ALERTS = {
    'info': {'background': '#e5f2f8', 'text': '#1f4f63', 'edge': THEME['teal']},
    'success': {'background': '#e3f0ec', 'text': TOKENS['primary'], 'edge': THEME['teal']},
    'warning': {'background': '#f5ebe7', 'text': '#7a4433', 'edge': THEME['rust']},
    'error': {'background': '#f5ebe7', 'text': '#7a4433', 'edge': THEME['rust']},
}

# Sequential single-hue-family map scale: near white, sky, through teal, to deep teal
MAP_RAMP = ['#eef7fb', THEME['sky'], '#86bcc2', '#64a399', THEME['teal'], '#356b5e', '#234a41']


def _channel(c):
    c = c / 255
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def luminance(hex_color):
    """WCAG relative luminance of a #rrggbb color."""
    h = hex_color.lstrip('#')
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return 0.2126 * _channel(r) + 0.7152 * _channel(g) + 0.0722 * _channel(b)


def contrast(a, b):
    """WCAG contrast ratio between two colors (1 to 21)."""
    la, lb = sorted([luminance(a), luminance(b)], reverse=True)
    return (la + 0.05) / (lb + 0.05)


# --- Color-vision deficiency simulation ------------------------------------------------
# Machado, Oliveira & Fernandes (2009) matrices at full severity, applied in linear RGB.
CVD_MATRICES = {
    'protan': ((0.152286, 1.052583, -0.204868), (0.114503, 0.786281, 0.099216), (-0.003882, -0.048116, 1.051998)),
    'deutan': ((0.367322, 0.860646, -0.227968), (0.280085, 0.672501, 0.047413), (-0.011820, 0.042940, 0.968881)),
    'tritan': ((1.255528, -0.076749, -0.178779), (-0.078411, 0.930809, 0.147602), (0.004733, 0.691367, 0.303900)),
}


def _linear_rgb(hex_color):
    h = hex_color.lstrip('#')
    return [_channel(int(h[i:i + 2], 16)) for i in (0, 2, 4)]


def _oklab(rgb):
    r, g, b = (min(max(c, 0.0), 1.0) for c in rgb)
    l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
    m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
    s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b
    l, m, s = (v ** (1 / 3) for v in (l, m, s))
    return (0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
            1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
            0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s)


def simulate(hex_color, kind):
    """Linear RGB of a color as seen with a color-vision deficiency ('protan', 'deutan', 'tritan')."""
    rgb = _linear_rgb(hex_color)
    return [sum(row[i] * rgb[i] for i in range(3)) for row in CVD_MATRICES[kind]]


def delta_e(a, b, kind=None):
    """Perceptual difference (OKLab delta E x100), optionally under simulated color blindness."""
    ra, rb = (simulate(c, kind) if kind else _linear_rgb(c) for c in (a, b))
    la, lb = _oklab(ra), _oklab(rb)
    return 100 * sum((x - y) ** 2 for x, y in zip(la, lb)) ** 0.5


def chroma(hex_color):
    """OKLCH chroma: how colorful a color is (below about 0.1 it reads as grey)."""
    _, a, b = _oklab(_linear_rgb(hex_color))
    return (a * a + b * b) ** 0.5
