"""
Evaluation harness (written before evaluation.py existed).

Measures the whole QA pipeline against its real goal: planted errors of known types must be
caught (recall) without flagging too many clean rows (false-positive rate).
"""
import numpy as np
import pandas as pd
import pytest

from evaluation import (ERROR_TYPES, STATISTICAL_TYPES, choose_config, evaluate_config,
                        inject_errors, run_grid)

FEATURES = ['ghg_per_capita', 'gdp', 'population', 'co2']


@pytest.fixture(scope='module')
def split_panel():
    """
    120 synthetic countries x 23 years, split 72 / 24 / 24 by country. A model needs enough
    training countries to know what a typical *unseen* country looks like: with only ~18,
    most validation countries look novel and are flagged (measured: 60% false positives).
    """
    rng = np.random.default_rng(42)
    rows = []
    for c in range(120):
        pop, gdp_pc, co2_pc = 10 ** rng.uniform(5, 9), 10 ** rng.uniform(3, 4.8), 10 ** rng.uniform(-0.5, 1.2)
        for year in range(2000, 2023):
            p = pop * (1 + 0.01 * (year - 2000))
            rows.append({'country': f'C{c:03d}', 'year': year,
                         'iso_code': f'{chr(65 + c // 676)}{chr(65 + c // 26 % 26)}{chr(65 + c % 26)}',
                         'ghg_per_capita': co2_pc * 1.3 * rng.uniform(0.97, 1.03),
                         'gdp': p * gdp_pc * rng.uniform(0.97, 1.03), 'population': p,
                         'co2': p * co2_pc / 1e6 * rng.uniform(0.97, 1.03)})
    df = pd.DataFrame(rows)
    order = sorted(df['country'].unique())
    split = {c: 'train' if i < 72 else 'validation' if i < 96 else 'test' for i, c in enumerate(order)}
    return df.assign(split=df['country'].map(split))


# --- inject_errors ---------------------------------------------------------------
def test_injects_the_requested_number_of_each_type_into_the_eval_split(split_panel):
    bad, labels = inject_errors(split_panel, FEATURES, n_per_type=4, split='validation', seed=0)
    assert labels['error_type'].value_counts().to_dict() == {t: 4 for t in ERROR_TYPES}
    assert (bad.loc[labels['row'], 'split'] == 'validation').all()
    assert labels['row'].is_unique


def test_each_error_type_corrupts_the_value_as_described(split_panel):
    bad, labels = inject_errors(split_panel, FEATURES, n_per_type=3, split='validation', seed=1)
    for _, lab in labels.iterrows():
        row, col, t = lab['row'], lab['column'], lab['error_type']
        if t == 'unit_x1000':
            assert bad.at[row, col] == pytest.approx(split_panel.at[row, col] * 1000)
        elif t == 'unit_div1000':
            assert bad.at[row, col] == pytest.approx(split_panel.at[row, col] / 1000)
        elif t == 'decimal_shift_x10':
            assert bad.at[row, col] == pytest.approx(split_panel.at[row, col] * 10)
        elif t == 'sign_flip':
            assert bad.at[row, col] == pytest.approx(-abs(split_panel.at[row, col]))
        elif t == 'missing_value':
            assert np.isnan(bad.at[row, col])
        elif t == 'column_swap':
            a, b = col.split('<->')
            assert bad.at[row, a] == pytest.approx(split_panel.at[row, b])
            assert bad.at[row, b] == pytest.approx(split_panel.at[row, a])
        elif t == 'duplicate_record':
            assert row >= len(split_panel)                       # appended copy
            original = bad[(bad['country'] == bad.at[row, 'country']) & (bad['year'] == bad.at[row, 'year'])]
            assert len(original) == 2


def test_injection_is_reproducible_and_leaves_input_untouched(split_panel):
    before = split_panel.copy()
    a = inject_errors(split_panel, FEATURES, n_per_type=2, split='validation', seed=5)
    b = inject_errors(split_panel, FEATURES, n_per_type=2, split='validation', seed=5)
    pd.testing.assert_frame_equal(split_panel, before)
    pd.testing.assert_frame_equal(a[1], b[1])


# --- evaluate_config ---------------------------------------------------------------
@pytest.fixture
def evaluated(split_panel):
    # Synthetic countries legitimately differ ~60x, so this test states its range threshold
    # (z = 3) explicitly; on real data the harness chooses the threshold on validation
    config = {'transform': 'asinh', 'use_ratios': True, 'contamination': 0.02, 'range_z': 3.0}
    return evaluate_config(split_panel, config, n_per_type=5, split='validation', seed=0)


def test_schema_error_types_are_always_caught(evaluated):
    per_type, _ = evaluated
    recall = per_type.set_index('error_type')['recall']
    assert recall['missing_value'] == 1.0
    assert recall['duplicate_record'] == 1.0


def test_large_unit_errors_are_mostly_caught(evaluated):
    per_type, _ = evaluated
    assert per_type.set_index('error_type').loc['unit_x1000', 'recall'] >= 0.8


def test_per_type_table_attributes_each_catch_to_a_layer(evaluated):
    per_type, _ = evaluated
    assert {'error_type', 'injected', 'caught', 'recall', 'by_schema', 'by_model'}.issubset(per_type.columns)
    assert (per_type['caught'] <= per_type['injected']).all()
    assert (per_type['by_schema'] + per_type['by_model'] >= per_type['caught']).all()


def test_summary_reports_false_positives_on_clean_rows_only(evaluated):
    _, summary = evaluated
    assert 0 <= summary['false_positive_rate'] <= 0.15
    assert 0 <= summary['recall_statistical'] <= 1
    assert summary['clean_rows'] > 0


# --- grid + choice -------------------------------------------------------------------
def test_grid_returns_one_averaged_row_per_config(split_panel):
    grid = [{'transform': t, 'use_ratios': r, 'contamination': 0.02, 'range_z': 4.0}
            for t in ['none', 'asinh'] for r in [True, False]]
    results = run_grid(split_panel, grid, n_per_type=3, split='validation', seeds=[0, 1])
    assert len(results) == 4
    assert {'transform', 'use_ratios', 'contamination', 'range_z', 'recall_statistical',
            'false_positive_rate'}.issubset(results.columns)


def test_choice_maximises_recall_within_the_review_budget():
    results = pd.DataFrame([
        {'transform': 'a', 'use_ratios': True, 'contamination': 0.02, 'range_z': 4.0, 'recall_statistical': 0.70, 'false_positive_rate': 0.02},
        {'transform': 'b', 'use_ratios': True, 'contamination': 0.05, 'range_z': 4.0, 'recall_statistical': 0.90, 'false_positive_rate': 0.06},
        {'transform': 'c', 'use_ratios': False, 'contamination': 0.02, 'range_z': None, 'recall_statistical': 0.80, 'false_positive_rate': 0.025},
    ])
    config, reason = choose_config(results, fpr_budget=0.03)
    assert config == {'transform': 'c', 'use_ratios': False, 'contamination': 0.02, 'range_z': None}
    assert '0.03' in reason or '3' in reason


def test_choice_falls_back_to_lowest_false_positive_rate_when_nothing_fits_budget():
    results = pd.DataFrame([
        {'transform': 'a', 'use_ratios': True, 'contamination': 0.05, 'range_z': 4.0, 'recall_statistical': 0.9, 'false_positive_rate': 0.08},
        {'transform': 'b', 'use_ratios': True, 'contamination': 0.02, 'range_z': 3.0, 'recall_statistical': 0.6, 'false_positive_rate': 0.05},
    ])
    config, _ = choose_config(results, fpr_budget=0.01)
    assert config['transform'] == 'b'


def test_statistical_types_exclude_what_the_schema_catches_by_rule():
    assert 'missing_value' not in STATISTICAL_TYPES
    assert 'duplicate_record' not in STATISTICAL_TYPES
    assert set(STATISTICAL_TYPES) <= set(ERROR_TYPES)
