"""
Map and trend helpers for the Emissions Map tab (written before mapview.py existed).
"""
import numpy as np
import pandas as pd
import pytest

from mapview import MEASURES, available_measures, default_country, map_frame, trend_frame, trend_summary


@pytest.fixture
def scored():
    rows = []
    for c, (iso, base) in enumerate({'Aland': ('ALA', 10.0), 'Bora': ('BOR', 100.0), 'Cyra': ('CYR', 1000.0),
                                     'Dune': ('DUN', 50.0)}.values()):
        for y in range(2000, 2005):
            rows.append({'country': c, 'iso_code': iso, 'year': y, 'co2': base * (1 + 0.1 * (y - 2000)),
                         'methane': base / 10, 'population': 1e6, 'is_anomaly': False,
                         'unusualness': 50.0})
    df = pd.DataFrame(rows)
    df['country'] = df['country'].map({0: 'Aland', 1: 'Bora', 2: 'Cyra', 3: 'Dune'})
    df.loc[(df['country'] == 'Bora') & (df['year'] == 2003), ['co2', 'is_anomaly', 'unusualness']] = \
        [5000.0, True, 99.8]
    return df


def test_only_emission_measures_that_exist_are_offered(scored):
    options = available_measures(scored)
    assert options == ['co2', 'methane']
    assert 'population' not in MEASURES


def test_map_frame_has_one_row_per_country_for_the_year(scored):
    frame = map_frame(scored, 'co2', 2003)
    assert len(frame) == 4
    assert {'iso_code', 'country', 'value', 'color_value', 'flagged', 'unusualness'}.issubset(frame.columns)
    bora = frame.set_index('country').loc['Bora']
    assert bora['flagged'] and bora['value'] == 5000
    assert bora['color_value'] == pytest.approx(np.log10(5000))


def test_trend_frame_includes_typical_range_and_yearly_change(scored):
    trend = trend_frame(scored, 'Bora', 'co2')
    assert list(trend['year']) == [2000, 2001, 2002, 2003, 2004]
    assert {'value', 'flagged', 'unusualness', 'typical_low', 'typical_high', 'pct_change'}.issubset(trend.columns)
    y2000 = trend.set_index('year').loc[2000]
    others = scored[scored['year'] == 2000]['co2']
    assert y2000['typical_low'] == pytest.approx(others.quantile(0.25))
    assert y2000['typical_high'] == pytest.approx(others.quantile(0.75))
    assert np.isnan(trend['pct_change'].iloc[0])
    assert trend.set_index('year').loc[2003, 'pct_change'] == pytest.approx(5000 / 120 - 1)


def test_trend_summary_is_a_plain_sentence(scored):
    text = trend_summary(trend_frame(scored, 'Bora', 'co2'), 'co2')
    assert 'CO₂ emissions' in text
    assert '2000' in text and '2004' in text
    assert 'flagged in 1 year' in text.lower()
    assert '2003' in text


def test_default_country_is_the_top_record_to_review():
    queue = pd.DataFrame({'country': ['Low land', 'Mid land'], 'priority': ['Low', 'Medium']})
    assert default_country(queue, ['Aland', 'Mid land', 'Low land']) == 'Mid land'
    assert default_country(queue.iloc[0:0], ['Aland', 'Bora']) == 'Aland'
