"""
Data helpers for the Emissions Map tab: what the map colors, what the trend graph draws,
and the plain-language summary under it. Kept free of Streamlit and Plotly so they can be tested.
"""
import numpy as np
import pandas as pd

from review import PRIORITY_ORDER, friendly_name

# Emission measures a visitor can map, in menu order (population and GDP are inputs, not emissions)
MEASURES = ['co2', 'methane', 'nitrous_oxide', 'ghg_per_capita', 'co2_per_capita',
            'methane_per_capita', 'nitrous_oxide_per_capita', 'co2_per_gdp']


def available_measures(scored):
    return [m for m in MEASURES if m in scored.columns]


def map_frame(scored, measure, year):
    """One row per country for `year`: its value, a log-scaled color value, flag and unusualness."""
    frame = scored.loc[scored['year'] == year,
                       ['country', 'iso_code', measure, 'is_anomaly', 'unusualness']] \
        .rename(columns={measure: 'value', 'is_anomaly': 'flagged'}).drop_duplicates('country')
    # Emissions span 5+ orders of magnitude; on a linear color scale almost every country would be
    # the same pale shade. Log10 spreads them out; tiny or negative values are clipped for color only.
    positive = frame['value'].where(frame['value'] > 0)
    floor = positive.min() if positive.notna().any() else 1.0
    frame['color_value'] = np.log10(positive.fillna(floor).clip(lower=floor))
    return frame.reset_index(drop=True)


def trend_frame(scored, country, measure):
    """The country's yearly values, with the middle half of all countries as a typical range."""
    band = scored.groupby('year')[measure].quantile([0.25, 0.75]).unstack()
    band.columns = ['typical_low', 'typical_high']
    trend = scored.loc[scored['country'] == country, ['year', measure, 'is_anomaly', 'unusualness']] \
        .rename(columns={measure: 'value', 'is_anomaly': 'flagged'}).sort_values('year')
    trend = trend.merge(band, left_on='year', right_index=True, how='left')
    trend['pct_change'] = trend['value'].pct_change()
    return trend.reset_index(drop=True)


def trend_summary(trend, measure):
    """One or two short sentences describing the trend and any flagged years."""
    if trend.empty:
        return "No data for this country."
    name = friendly_name(measure)
    name = name[:1].upper() + name[1:]
    first, last = trend.iloc[0], trend.iloc[-1]
    change = last['value'] / first['value'] - 1 if first['value'] else np.nan
    if pd.isna(change):
        text = f"{name}: {int(first['year'])} to {int(last['year'])}."
    else:
        direction = 'rose' if change > 0.02 else 'fell' if change < -0.02 else 'stayed about the same'
        amount = f" {abs(change):.0%}" if direction != 'stayed about the same' else ''
        text = f"{name} {direction}{amount} from {int(first['year'])} to {int(last['year'])}."
    flagged = trend.loc[trend['flagged'], 'year'].astype(int).tolist()
    if flagged:
        years = ', '.join(map(str, flagged[:6])) + (' and more' if len(flagged) > 6 else '')
        text += f" Flagged in {len(flagged)} year{'s' if len(flagged) != 1 else ''}: {years}."
    else:
        text += " No years were flagged."
    return text


def default_country(queue, countries):
    """The country of the most urgent record to review, else the first country."""
    if len(queue):
        top = queue.sort_values('priority', key=lambda s: s.map(PRIORITY_ORDER), kind='stable')
        for c in top['country']:
            if c in countries:
                return c
    return countries[0]
