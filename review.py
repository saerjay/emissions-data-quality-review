"""
Business-facing review logic: turns technical check results into a prioritized work queue,
plain-language explanations, a readiness verdict and key takeaways.

Kept separate from the Streamlit page so every sentence the app shows can be unit tested.
Wording targets a general professional audience (about a high-school reading level).

Priority rules
  High   - the record breaks a basic rule (missing, duplicate, impossible or unrealistic
           value). It must be fixed before the data is used.
  Medium - a one-off statistical flag: the record is unusual for that country in that year.
           These are the most likely data-entry or unit errors.
  Low    - the country is flagged in most years. A country that is always unusual (a very
           large economy, a tiny island state) is usually a real difference, not an error.
"""
import math

import numpy as np
import pandas as pd

from validation import EmissionsDataValidator

PRIORITY_ORDER = {'High': 0, 'Medium': 1, 'Low': 2}
CONSISTENT_SHARE = 0.75      # flagged in at least 75% of a country's years = consistently unusual
CONSISTENT_MIN_YEARS = 3

FRIENDLY = {
    'co2': 'CO₂ emissions',
    'methane': 'methane emissions',
    'nitrous_oxide': 'nitrous oxide emissions',
    'population': 'population',
    'gdp': 'GDP',
    'primary_energy_consumption': 'energy use',
    'ghg_per_capita': 'greenhouse gas per person',
    'co2_per_capita': 'CO₂ per person',
    'methane_per_capita': 'methane per person',
    'nitrous_oxide_per_capita': 'nitrous oxide per person',
    'energy_per_capita': 'energy use per person',
    'co2_per_gdp': 'CO₂ per dollar of GDP',
    'gdp_per_capita': 'GDP per person',
    'iso_code': 'country code',
    'year': 'year',
    'country, year': 'country and year',
    'co2 / population': 'CO₂ per person',
}

RULE_ISSUES = {
    'completeness': ('Missing value', "Fill in the value from the source, or mark it as not reported."),
    'uniqueness': ('Duplicate record', "Keep one copy and remove the other."),
    'validity': ('Impossible value', "Correct the value using the original source."),
    'plausibility': ('Unrealistic value', "Check the units. The numbers do not fit together."),
}

CHECK_NAMES = {
    'rule': 'Rule check',
    'range': 'Range check',
    'model': 'Pattern check',
    'model + range': 'Range check + Pattern check',
}


def unusualness_label(value):
    """
    Plain label for unusualness: the share of reference records that look less unusual.
    Rule breaks are not scored (NaN) and are labelled separately in the review queue.
    """
    if value is None or pd.isna(value):
        return "Not scored"
    if value >= 99:
        return f"Very high ({value:.1f}%)"
    if value >= 95:
        return f"High ({value:.1f}%)"
    if value >= 80:
        return f"Moderate ({value:.0f}%)"
    return f"Low ({value:.0f}%)"


def friendly_name(column):
    """Plain-English name for a column (falls back to replacing underscores)."""
    return FRIENDLY.get(column, str(column).replace('_', ' '))


def _cap(text):
    """Capitalises the first letter only ('CO₂ emissions' stays as is)."""
    return text[:1].upper() + text[1:]


def _rule_rows(failures):
    rows = []
    if failures is None or failures.empty:
        return rows
    for row_id, group in failures.dropna(subset=['row']).groupby('row'):
        first = group.iloc[0]
        issues, sentences, steps = [], [], []
        for _, f in group.drop_duplicates(['rule', 'column']).iterrows():
            issue, step = RULE_ISSUES.get(f['dimension'], ('Rule broken', "Correct the value using the source."))
            name = friendly_name(f['column'])
            if f['dimension'] == 'completeness':
                sentence = f"{_cap(name)} {'are' if name.endswith('emissions') else 'is'} missing."
            elif f['dimension'] == 'uniqueness':
                sentence = "This country and year appear more than once."
            else:
                verb = 'are' if name.endswith('emissions') else 'is'
                sentence = f"{_cap(name)} {verb} {f['failing_value']}. {f['why']}"
            if issue not in issues:
                issues.append(issue)
                steps.append(step)
            if sentence not in sentences:
                sentences.append(sentence)
        rows.append({
            'row': int(row_id), 'country': first['country'], 'year': first['year'],
            'priority': 'High', 'issue': ', '.join(issues), 'what_looks_wrong': ' '.join(sentences),
            'next_step': ' '.join(steps), 'check': CHECK_NAMES['rule'], 'pattern': 'Breaks a basic rule',
            'unusualness': np.nan,
        })
    return rows


def _driver(row):
    driver = row.get('top_driver')
    if isinstance(driver, str) and driver:
        return driver
    ranges = row.get('range_features')
    if isinstance(ranges, str) and ranges:
        return ranges.split(', ')[0]
    return None


def _statistical_sentence(row, driver):
    if driver is None or f'{driver}_pctile' not in row:
        return "This record's values do not fit the usual pattern."
    pct = row[f'{driver}_pctile']
    higher = pct >= 50
    value = EmissionsDataValidator._fmt(row[driver], driver)
    name = _cap(friendly_name(driver))
    if row.get('out_of_range'):
        verb = 'are' if name.endswith('emissions') else 'is'
        return f"{name} ({value}) {verb} far {'higher' if higher else 'lower'} than in other countries."
    share = math.floor(pct if higher else 100 - pct)     # never round 99.5 up to a misleading 100
    verb = 'are' if name.endswith('emissions') else 'is'
    return f"{name} ({value}) {verb} {'higher' if higher else 'lower'} than {share}% of records."


def _statistical_rows(scored):
    rows = []
    if scored is None or scored.empty:
        return rows
    years = scored.groupby('country').size()
    flagged_years = scored[scored['is_anomaly']].groupby('country').size()
    for idx, row in scored[scored['is_anomaly']].iterrows():
        k, n = int(flagged_years[row['country']]), int(years[row['country']])
        consistent = k >= CONSISTENT_MIN_YEARS and k / n >= CONSISTENT_SHARE
        layer = row.get('detected_by') or 'model'
        far = 'range' in layer
        if consistent:
            step = "Likely a real difference for this country. Confirm once, then mark it as expected."
        elif far:
            step = "Check the unit and the source value. A value this far off is often a unit mistake."
        else:
            step = "Confirm the value with the source. It may be real, but it is unusual."
        rows.append({
            'row': int(idx), 'country': row['country'], 'year': row['year'],
            'priority': 'Low' if consistent else 'Medium',
            'issue': 'Far outside normal range' if far else 'Unusual combination of values',
            'what_looks_wrong': _statistical_sentence(row, _driver(row)),
            'next_step': step,
            'check': CHECK_NAMES.get(layer, 'Pattern check'),
            'pattern': f"Flagged in {k} of {n} years" + (" (every year)" if k == n else ""),
            'split': row.get('split'),
            'unusualness': row.get('unusualness', np.nan),
        })
    return rows


def build_review_queue(scored, failures):
    """One row per record that needs attention, most urgent first."""
    queue = pd.DataFrame(_rule_rows(failures) + _statistical_rows(scored))
    if queue.empty:
        return pd.DataFrame(columns=['row', 'country', 'year', 'priority', 'issue', 'what_looks_wrong',
                                     'next_step', 'check', 'pattern', 'unusualness', 'unusualness_label'])
    queue['unusualness_label'] = np.where(queue['priority'].eq('High') & queue['unusualness'].isna(),
                                          'Rule broken', queue['unusualness'].map(unusualness_label))
    # A record that breaks a rule is listed once, as High
    queue = queue.sort_values('priority', key=lambda s: s.map(PRIORITY_ORDER), kind='stable')
    queue = queue.drop_duplicates('row', keep='first')
    return queue.sort_values(['priority', 'country', 'year'],
                             key=lambda s: s.map(PRIORITY_ORDER) if s.name == 'priority' else s) \
        .reset_index(drop=True)


def readiness_verdict(queue):
    """(level, message) where level is 'not_ready', 'review' or 'ready'."""
    counts = queue['priority'].value_counts() if len(queue) else pd.Series(dtype=int)
    high, medium = int(counts.get('High', 0)), int(counts.get('Medium', 0))
    if high:
        return 'not_ready', (f"Not ready: {high:,} record{'s' if high != 1 else ''} break basic rules "
                             "and must be fixed first.")
    if medium:
        return 'review', (f"Ready after review: {medium:,} record{'s' if medium != 1 else ''} "
                          "look unusual and need a quick check.")
    return 'ready', "Ready to use: no records need urgent action."


def key_takeaways(summary, queue, evaluation=None, coverage=None):
    """Three to five short sentences that sum up the review, in report wording."""
    rows = summary['rows']
    counts = queue['priority'].value_counts() if len(queue) else pd.Series(dtype=int)
    ready = rows - queue['row'].nunique() if len(queue) else rows
    items = [f"{ready:,} of {rows:,} records ({ready / rows:.0%}) passed quality checks and are ready to report."]

    high = int(counts.get('High', 0))
    items.append(f"{high:,} records break basic rules, like missing or impossible values. Fix these first."
                 if high else "No records break basic rules, such as missing, duplicate or impossible values.")

    medium = int(counts.get('Medium', 0))
    items.append(f"{medium:,} records look unusual for that country and year. "
                 "This indicates a need for a manual review.")

    if evaluation:
        test = evaluation['test']
        items.append(f"Limitation: When testing on new countries, the quality review found "
                     f"{test['recall_all']:.0%} of hidden mistakes. "
                     f"It flagged {test['false_positive_rate']:.1%} of good records by mistake.")
    if coverage and coverage.get('countries_unchecked'):
        items.append(f"Scope: {coverage['countries_unchecked']} countries could not be checked because key data "
                     "is missing. These are mostly small islands, microstates and overseas territories.")
    return items[:5]


def flags_by_year(queue, scored):
    """Records needing review per year and priority, as a share of that year's checked records."""
    totals = scored.groupby('year').size().rename('total')
    if queue.empty:
        return pd.DataFrame(columns=['year', 'priority', 'records', 'total', 'share'])
    counts = queue.groupby(['year', 'priority']).size().rename('records').reset_index()
    table = counts.merge(totals, left_on='year', right_index=True, how='left')
    table['share'] = table['records'] / table['total']
    return table


def _join_names(names, limit=3):
    names = list(dict.fromkeys(names))
    if len(names) > limit:                       # "A, B, C and 2 more"
        return ', '.join(names[:limit]) + f" and {len(names) - limit} more"
    if len(names) > 1:                           # "A, B and C"
        return ', '.join(names[:-1]) + ' and ' + names[-1]
    return names[0] if names else ''


def year_insight(queue, scored, ratio=1.5, min_extra=3):
    """
    One plain sentence about the year-by-year pattern. A year is called out only if its flagged
    share is at least `ratio` times the typical (median) year AND it has at least `min_extra` more
    flagged records than typical, so a small wobble is never described as a spike.
    """
    totals = scored.groupby('year').size()
    flagged = (queue.groupby('year')['row'].nunique() if len(queue) else pd.Series(dtype=float)) \
        .reindex(totals.index, fill_value=0)
    share = flagged / totals
    typical_share, typical_count = share.median(), flagged.median()
    spikes = share[(share >= ratio * typical_share) & (flagged - typical_count >= min_extra)]
    if spikes.empty:
        return (f"No year stands out: the share of records flagged stays between {share.min():.1%} and "
                f"{share.max():.1%} each year.")
    year = spikes.idxmax()
    # Explain the extra flags: records that are not part of the every-year (Low) pattern
    extra = queue[(queue['year'] == year) & (queue['priority'] != 'Low')]
    rows = scored.loc[scored.index.intersection(extra['row'])]
    drivers = rows['top_driver'].dropna() if 'top_driver' in rows else pd.Series(dtype=object)
    driver = friendly_name(drivers.mode().iloc[0]) if len(drivers) else 'several measures'
    text = (f"{int(year)} stands out: {share[year]:.1%} of records were flagged, versus {typical_share:.1%} in a "
            f"typical year. Most of the extra flags involve {driver}, in {_join_names(extra['country'])}.")
    others = [str(int(y)) for y in spikes.index if y != year]
    if others:
        text += f" {_join_names(others)} also stand{'s' if len(others) == 1 else ''} out."
    return text
