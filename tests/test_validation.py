"""
Validator behaviour added for the train / validation / test workflow and for reporting
missing data as a quality finding (written before the implementation).
"""
import numpy as np
import pandas as pd
import pytest

from transforms import fit_transform_params
from validation import EmissionsDataValidator, completeness_summary


@pytest.fixture
def split_panel(panel):
    """The shared panel with a country-level split column (18 train, 6 validation, 6 test)."""
    countries = sorted(panel['country'].unique())
    split = {c: 'train' if i < 18 else 'validation' if i < 24 else 'test' for i, c in enumerate(countries)}
    return panel.assign(split=panel['country'].map(split))


# --- fit on train, score everything -------------------------------------------
def test_model_is_fit_on_training_rows_only(split_panel):
    v = EmissionsDataValidator(split_panel, explain=False)
    v.run()
    assert v.n_fit == (split_panel['split'] == 'train').sum()
    assert set(v.scored['split']) == {'train', 'validation', 'test'}      # but everything is scored


def test_without_split_column_the_model_fits_on_all_rows(panel):
    v = EmissionsDataValidator(panel, explain=False)
    v.run()
    assert v.n_fit == len(panel)


def test_fitted_transform_learns_lambda_from_training_rows(split_panel):
    v = EmissionsDataValidator(split_panel, transform='yeo_johnson', use_ratios=False, explain=False)
    v.run()
    train = split_panel.loc[split_panel['split'] == 'train', v.feature_cols]
    assert v.transform_params == pytest.approx(fit_transform_params(train, 'yeo_johnson'))


def test_threshold_is_set_on_training_rows(split_panel):
    v = EmissionsDataValidator(split_panel, contamination=0.05, explain=False)
    v.run()
    train_rate = v.scored.loc[v.scored['split'] == 'train', 'is_anomaly'].mean()
    assert train_rate == pytest.approx(0.05, abs=0.015)


def test_fast_mode_skips_shap(split_panel):
    v = EmissionsDataValidator(split_panel, explain=False)
    v.run()
    assert v.shap_values.empty
    assert 'is_anomaly' in v.scored


# --- data quality dimensions ----------------------------------------------------
@pytest.mark.parametrize('column, bad_value, dimension', [
    ('co2', np.nan, 'completeness'),
    ('co2', -1.0, 'validity'),
    ('iso_code', 'xx1', 'validity'),
    ('population', 1.0, 'plausibility'),     # implied CO2 per person becomes absurd
])
def test_each_failure_is_labelled_with_a_quality_dimension(panel, column, bad_value, dimension):
    panel = panel.copy()
    if isinstance(bad_value, str):
        panel[column] = panel[column].astype(object)
    panel.loc[4, column] = bad_value
    failures = EmissionsDataValidator(panel).validate_schema()
    assert dimension in set(failures['dimension'])


def test_duplicates_are_a_uniqueness_finding(panel):
    dup = pd.concat([panel, panel.iloc[[0]]], ignore_index=True)
    failures = EmissionsDataValidator(dup).validate_schema()
    assert set(failures['dimension']) == {'uniqueness'}


# --- completeness summary ---------------------------------------------------------
def test_completeness_summary_classifies_countries_per_feature():
    df = pd.DataFrame({
        'country': ['A'] * 3 + ['B'] * 3 + ['C'] * 3,
        'year': [2000, 2001, 2002] * 3,
        'co2': [1, 2, 3, 1, np.nan, 3, np.nan, np.nan, np.nan],
        'methane': [1.0] * 9,
    })
    out = completeness_summary(df, ['co2', 'methane']).set_index('feature')
    assert out.loc['co2', 'pct_values_present'] == pytest.approx(100 * 5 / 9, abs=0.1)
    assert out.loc['co2', ['countries_complete', 'countries_partial', 'countries_missing']].tolist() == [1, 1, 1]
    assert out.loc['methane', 'countries_complete'] == 3


# --- Out-of-range layer (new: written first) ------------------------------------
# An Isolation Forest cannot extrapolate: its splits lie inside the training range, so a
# value 1000x beyond the largest training value scores like the largest training value.
# A robust z-score against the training distribution catches what lies far outside it.
def test_value_far_outside_training_range_is_flagged(split_panel):
    df = split_panel.copy()
    row = df.index[(df['split'] == 'validation')][10]
    df.loc[row, 'gdp'] *= 1000
    v = EmissionsDataValidator(df, explain=False)
    v.run()
    assert v.scored.loc[row, 'out_of_range']
    assert v.scored.loc[row, 'is_anomaly']
    assert 'gdp' in v.scored.loc[row, 'range_features']


def test_range_check_uses_training_rows_only(split_panel):
    # Making the TEST countries wildly extreme must not move the reference: training rows keep
    # exactly the same range scores. (A reference fitted on all rows would shift.)
    base = EmissionsDataValidator(split_panel, explain=False)
    base.run()
    df = split_panel.copy()
    test_rows = df.index[df['split'] == 'test']
    df.loc[test_rows, 'gdp'] *= 1000
    shifted = EmissionsDataValidator(df, explain=False)
    shifted.run()
    train_rows = df.index[df['split'] == 'train']
    pd.testing.assert_frame_equal(base.range_scores.loc[train_rows], shifted.range_scores.loc[train_rows])
    # Most shifted test rows land outside (synthetic countries already span ~4 orders of magnitude)
    assert shifted.scored.loc[test_rows, 'out_of_range'].mean() > 0.5


def test_typical_rows_are_not_out_of_range(split_panel):
    v = EmissionsDataValidator(split_panel, explain=False)
    v.run()
    assert v.scored['out_of_range'].mean() < 0.02


def test_range_threshold_can_be_disabled(split_panel):
    df = split_panel.copy()
    row = df.index[(df['split'] == 'validation')][10]
    df.loc[row, 'gdp'] *= 1000
    v = EmissionsDataValidator(df, explain=False, range_z=None)
    v.run()
    assert not v.scored['out_of_range'].any()


# --- Unusualness (new: written first) --------------------------------------
# Unusualness = share of reference (training) records that look LESS unusual than this one,
# taken from whichever layer (pattern or range) finds the record more unusual.
@pytest.fixture
def scored_validator(split_panel):
    df = split_panel.copy()
    df.loc[df.index[(df['split'] == 'validation')][10], 'gdp'] *= 1000
    v = EmissionsDataValidator(df, contamination=0.02, explain=False)
    v.run()
    return v


def test_unusualness_is_a_percentage_for_every_scored_row(scored_validator):
    c = scored_validator.scored['unusualness']
    assert c.notna().all()
    assert c.between(0, 100).all()


def test_flagged_rows_are_highly_unusual(scored_validator):
    flagged = scored_validator.scored[scored_validator.scored['is_anomaly']]
    assert (flagged['unusualness'] >= 95).all()


def test_training_unusualness_is_spread_evenly(scored_validator):
    train = scored_validator.scored[scored_validator.scored['split'] == 'train']['unusualness']
    assert 40 <= train.median() <= 60


def test_unusualness_rises_as_the_pattern_score_falls(scored_validator):
    s = scored_validator.scored[~scored_validator.scored['out_of_range']].sort_values('anomaly_score')
    # anomaly_score falls = more unusual, so unusualness must never fall as the score rises
    assert s['unusualness'].is_monotonic_decreasing
