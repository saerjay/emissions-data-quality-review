"""
Business-facing review logic (written before review.py existed): turns technical check
results into a prioritized work queue, plain-language explanations and a readiness verdict.
"""
import numpy as np
import pandas as pd
import pytest

from review import (PRIORITY_ORDER, build_review_queue, friendly_name, key_takeaways,
                    readiness_verdict)


def _scored(rows):
    base = {'split': 'test', 'anomaly_score': -0.01, 'is_anomaly': True, 'out_of_range': False,
            'model_anomaly': True, 'detected_by': 'model', 'top_driver': 'co2', 'range_features': '',
            'co2': 100.0, 'co2_pctile': 99.5}
    return pd.DataFrame([{**base, **r} for r in rows])


def _failures(rows):
    cols = ['row', 'country', 'year', 'column', 'rule', 'dimension', 'failing_value', 'why']
    return pd.DataFrame(rows, columns=cols)


# --- names ------------------------------------------------------------------------
def test_friendly_names_are_plain_english():
    assert friendly_name('co2') == 'CO₂ emissions'
    assert friendly_name('methane_per_capita') == 'methane per person'
    assert friendly_name('unknown_col') == 'unknown col'


# --- queue ------------------------------------------------------------------------
@pytest.fixture
def queue():
    # Country A: flagged only in 2010 (one-off). Country B: flagged every year (consistent).
    scored = _scored(
        [{'country': 'A', 'year': y, 'is_anomaly': y == 2010} for y in range(2005, 2015)]
        + [{'country': 'B', 'year': y, 'detected_by': 'range', 'out_of_range': True, 'model_anomaly': False,
            'top_driver': 'methane', 'methane': 0.001, 'methane_pctile': 0.2} for y in range(2005, 2015)]
    )
    failures = _failures([[99, 'C', 2012, 'co2', 'not_nullable', 'completeness', 'nan', 'Required value is missing.']])
    return build_review_queue(scored, failures)


def test_rule_failures_are_high_priority(queue):
    row = queue[queue['country'] == 'C'].iloc[0]
    assert row['priority'] == 'High'
    assert row['issue'] == 'Missing value'
    assert 'fill in' in row['next_step'].lower()
    assert 'CO₂ emissions are missing' in row['what_looks_wrong']


def test_one_off_flags_are_medium_priority(queue):
    row = queue[queue['country'] == 'A'].iloc[0]
    assert row['priority'] == 'Medium'
    assert row['year'] == 2010
    assert 'CO₂ emissions' in row['what_looks_wrong']
    assert 'higher than' in row['what_looks_wrong']


def test_countries_flagged_every_year_are_low_priority(queue):
    rows = queue[queue['country'] == 'B']
    assert len(rows) == 10
    assert (rows['priority'] == 'Low').all()
    assert 'every year' in rows.iloc[0]['pattern'].lower() or '10 of 10' in rows.iloc[0]['pattern']
    assert 'lower than' in rows.iloc[0]['what_looks_wrong']


def test_queue_is_sorted_by_priority(queue):
    ranks = queue['priority'].map(PRIORITY_ORDER)
    assert ranks.is_monotonic_increasing


def test_every_record_has_plain_language_fields(queue):
    for col in ['issue', 'what_looks_wrong', 'next_step', 'check']:
        assert queue[col].str.len().gt(0).all(), col


def test_duplicate_is_one_queue_row_per_record():
    failures = _failures([
        [1, 'D', 2001, 'country, year', 'multiple_fields_uniqueness', 'uniqueness', 'D 2001', 'dup'],
        [7, 'D', 2001, 'country, year', 'multiple_fields_uniqueness', 'uniqueness', 'D 2001', 'dup'],
    ])
    queue = build_review_queue(_scored([{'country': 'E', 'year': 2001, 'is_anomaly': False}]), failures)
    assert len(queue) == 2
    assert set(queue['issue']) == {'Duplicate record'}


# --- verdict + takeaways ----------------------------------------------------------
def test_verdict_levels():
    q = pd.DataFrame({'priority': ['High', 'Medium']})
    assert readiness_verdict(q)[0] == 'not_ready'
    assert readiness_verdict(q[q['priority'] == 'Medium'])[0] == 'review'
    assert readiness_verdict(q.iloc[0:0])[0] == 'ready'


def test_takeaways_are_short_plain_sentences(queue):
    summary = {'rows': 30, 'schema_rows': 1}
    evaluation = {'test': {'recall_statistical': 0.64, 'false_positive_rate': 0.04, 'recall_all': 0.8}}
    items = key_takeaways(summary, queue, evaluation=evaluation, coverage={'countries_unchecked': 19})
    assert 3 <= len(items) <= 5
    assert all(len(t.split()) <= 40 for t in items)
    assert any('80%' in t for t in items)


# --- Confidence labels (new: written first) --------------------------------------
from review import unusualness_label


@pytest.mark.parametrize('value, label', [
    (99.6, 'Very high'), (99.0, 'Very high'), (97.0, 'High'), (90.0, 'Moderate'), (40.0, 'Low'),
])
def test_unusualness_labels(value, label):
    assert unusualness_label(value).startswith(label)


def test_queue_carries_unusualness():
    scored = _scored([{'country': 'A', 'year': 2010, 'unusualness': 99.7}]
                     + [{'country': 'A', 'year': y, 'is_anomaly': False, 'unusualness': 30.0}
                        for y in range(2000, 2010)])
    failures = _failures([[99, 'C', 2012, 'co2', 'not_nullable', 'completeness', 'nan', 'missing']])
    queue = build_review_queue(scored, failures).set_index('country')
    assert queue.loc['A', 'unusualness'] == pytest.approx(99.7)
    assert np.isnan(queue.loc['C', 'unusualness'])         # rule breaks are not scored on unusualness
    assert queue.loc['C', 'unusualness_label'] == 'Rule broken'


def test_takeaways_use_the_report_wording(queue):
    summary = {'rows': 30, 'schema_rows': 0}
    evaluation = {'test': {'recall_statistical': 0.64, 'false_positive_rate': 0.04, 'recall_all': 0.8}}
    items = key_takeaways(summary, queue, evaluation=evaluation, coverage={'countries_unchecked': 19})
    assert 'passed quality checks and are ready to report' in items[0]
    assert any('indicates a need for a manual review' in t for t in items)
    assert any(t.startswith('Limitation:') and 'found 80% of hidden mistakes' in t for t in items)
    assert items[-1].startswith('Scope: 19 countries could not be checked')


def test_a_statistical_record_at_100_percent_is_not_called_a_rule_break():
    assert unusualness_label(100.0) == 'Very high (100.0%)'


# --- Flagged share by year + automated insight (new: written first) -------------------
from review import flags_by_year, year_insight


def _year_data(spike=True):
    """10 countries x 2000-2005. Country C9 is flagged every year (Low); 2003 gets extra Medium flags."""
    rows = []
    for c in range(10):
        for y in range(2000, 2006):
            rows.append({'country': f'C{c}', 'year': y, 'is_anomaly': c == 9, 'top_driver': 'co2',
                         'detected_by': 'model', 'co2': 1.0, 'co2_pctile': 99.0, 'out_of_range': False})
    scored = pd.DataFrame(rows)
    if spike:
        hit = scored['year'].eq(2003) & scored['country'].isin(['C1', 'C2', 'C3', 'C4'])
        scored.loc[hit, ['is_anomaly', 'top_driver']] = [True, 'methane_per_capita']
        scored['methane_per_capita'] = 1.0
        scored['methane_per_capita_pctile'] = 99.0
    return scored


def test_flags_by_year_reports_share_of_each_years_records():
    scored = _year_data()
    table = flags_by_year(build_review_queue(scored, None), scored)
    y2003 = table[table['year'] == 2003].set_index('priority')
    assert y2003.loc['Medium', 'records'] == 4
    assert y2003.loc['Medium', 'share'] == pytest.approx(0.4)
    assert y2003.loc['Low', 'share'] == pytest.approx(0.1)
    assert set(table['total']) == {10}


def test_insight_names_the_spike_year_its_drivers_and_countries():
    scored = _year_data()
    text = year_insight(build_review_queue(scored, None), scored)
    assert '2003' in text
    assert '50.0%' in text and '10.0%' in text          # spike share vs typical share
    assert 'methane per person' in text
    assert 'C1' in text


def test_insight_says_so_when_no_year_stands_out():
    scored = _year_data(spike=False)
    text = year_insight(build_review_queue(scored, None), scored)
    assert text.startswith('No year stands out')
