"""
Shared fixtures. Everything is synthetic and built in memory so the suite runs
offline, in seconds, and every expected result can be worked out by hand.
"""
import matplotlib
matplotlib.use("Agg")  # no GUI windows during tests

import numpy as np
import pandas as pd
import pytest

FEATURES = ['population', 'gdp', 'co2', 'primary_energy_consumption',
            'methane', 'nitrous_oxide', 'ghg_per_capita']


@pytest.fixture
def raw_owid():
    """
    A miniature OWID file with known problems baked in:
      - years 1998-2024 (pre-2000 rows must be scoped out)
      - an aggregate region with no ISO code ("World")
      - gdp unpublished for 2023-2024 (reporting lag)
      - "Tiny Isle" never reports gdp (uncovered country)
      - "Gapland" has interior gaps in methane (interpolable)
      - "Newstate" has leading gaps in energy (not interpolable -> rows dropped)
    """
    rng = np.random.default_rng(0)
    countries = {
        'Alpha': 'ALP', 'Bravo': 'BRV', 'Charlie': 'CHR', 'Delta': 'DLT',
        'Echo': 'ECH', 'Gapland': 'GAP', 'Newstate': 'NEW', 'Tiny Isle': 'TNY', 'World': None,
    }
    rows = []
    for i, (country, iso) in enumerate(countries.items()):
        scale = 10 ** (i % 4)
        for year in range(1998, 2025):
            t = year - 1998
            rows.append({
                'country': country, 'year': year, 'iso_code': iso,
                'population': 1e6 * scale * (1 + 0.01 * t),
                'gdp': 1e10 * scale * (1 + 0.02 * t) * rng.uniform(0.98, 1.02),
                'co2': 10.0 * scale * (1 + 0.01 * t),
                'primary_energy_consumption': 50.0 * scale,
                'methane': 2.0 * scale + t * 0.1,
                'nitrous_oxide': 1.0 * scale,
                'ghg_per_capita': 5.0 + (i % 3),
            })
    df = pd.DataFrame(rows)
    df['extra_column'] = 'ignored'

    df.loc[df['year'] >= 2023, 'gdp'] = np.nan
    df.loc[df['country'] == 'Tiny Isle', 'gdp'] = np.nan
    gap = (df['country'] == 'Gapland') & df['year'].isin([2004, 2005, 2010, 2011, 2012, 2015, 2016, 2019, 2020])
    df.loc[gap, 'methane'] = np.nan
    df.loc[(df['country'] == 'Newstate') & (df['year'] <= 2002), 'primary_energy_consumption'] = np.nan
    return df


@pytest.fixture
def panel():
    """Clean country-year panel: 30 countries x 2000-2022 with smooth trends."""
    rng = np.random.default_rng(42)
    rows = []
    for c in range(30):
        base_pop = 10 ** rng.uniform(5, 9)
        gdp_pc = 10 ** rng.uniform(3, 4.8)
        co2_pc = 10 ** rng.uniform(-0.5, 1.2)
        for year in range(2000, 2023):
            growth = 1 + 0.01 * (year - 2000)
            pop = base_pop * growth
            rows.append({
                'country': f'Country{c:02d}', 'year': year, 'iso_code': f'C{chr(65 + c // 26)}{chr(65 + c % 26)}',
                'ghg_per_capita': co2_pc * 1.3 * rng.uniform(0.97, 1.03),
                'gdp': pop * gdp_pc * rng.uniform(0.97, 1.03),
                'population': pop,
                'co2': pop * co2_pc / 1e6 * rng.uniform(0.97, 1.03),
            })
    return pd.DataFrame(rows)
