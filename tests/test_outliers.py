"""
Outlier detection used in exploratory analysis (written before outliers.py existed).

Handling policy under test:
  - cross-sectional extremes are mostly skew, so they are log-transformed, not removed
  - within-country spikes are flagged for human review, never silently deleted
"""
import numpy as np
import pandas as pd
import pytest

from outliers import (signed_log, iqr_outliers, robust_zscore, outlier_profile,
                      temporal_spikes)


# --- signed_log --------------------------------------------------------------
def test_signed_log_matches_log1p_for_positives():
    x = pd.Series([0.0, 1.0, 99.0])
    assert np.allclose(signed_log(x), np.log1p(x))


def test_signed_log_preserves_sign_for_carbon_sinks():
    # Net GHG can be negative (land-use sinks); log must not produce NaN
    out = signed_log(pd.Series([-9.0, 9.0]))
    assert out.iloc[0] == pytest.approx(-out.iloc[1])
    assert not out.isna().any()


# --- IQR (Tukey fences) ------------------------------------------------------
def test_iqr_flags_only_values_beyond_fences():
    s = pd.Series([10, 11, 12, 13, 14, 15, 100])
    flags = iqr_outliers(s)
    assert flags.tolist() == [False] * 6 + [True]


def test_iqr_flags_low_outliers_too():
    s = pd.Series([-100, 10, 11, 12, 13, 14, 15])
    assert iqr_outliers(s).iloc[0]


def test_iqr_multiplier_widens_fences():
    s = pd.Series([10, 11, 12, 13, 14, 15, 22])
    assert iqr_outliers(s, k=1.5).iloc[-1]
    assert not iqr_outliers(s, k=3.0).iloc[-1]


# --- robust z-score (median / MAD) --------------------------------------------
def test_robust_z_is_zero_at_median_and_ignores_the_outlier_itself():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 1000.0])
    z = robust_zscore(s)
    # median = 3.5; the 1000 does not inflate the spread like a standard deviation would
    assert z.iloc[-1] > 100
    assert abs(z.iloc[2]) < 1


def test_robust_z_handles_zero_mad_without_dividing_by_zero():
    s = pd.Series([5.0, 5.0, 5.0, 5.0, 9.0])
    z = robust_zscore(s)
    assert np.isfinite(z).all()
    assert z.iloc[-1] > z.iloc[0]


def test_robust_z_all_identical_values_are_not_outliers():
    z = robust_zscore(pd.Series([3.0] * 5))
    assert (z == 0).all()


# --- outlier_profile ----------------------------------------------------------
def test_profile_shows_log_scale_removes_skew_driven_outliers(panel):
    profile = outlier_profile(panel, ['gdp', 'population', 'co2']).set_index('feature')
    assert {'skew_raw', 'skew_log', 'pct_iqr_raw', 'pct_iqr_log', 'pct_robust_z_log'}.issubset(profile.columns)
    # Heavy right tail on the raw scale, near-symmetric after the log transform
    assert (profile['skew_raw'] > 1).all()
    assert (profile['skew_log'].abs() < profile['skew_raw']).all()
    assert (profile['pct_iqr_log'] <= profile['pct_iqr_raw']).all()


# --- temporal_spikes ----------------------------------------------------------
def _series(values, country='Libya'):
    return pd.DataFrame({'country': country, 'year': range(2000, 2000 + len(values)), 'co2': values})


def test_spike_and_revert_is_detected_and_labelled():
    values = [50.0, 51, 52, 53, 54, 10, 55, 56, 57, 58]   # one-year collapse, then recovery
    panel = pd.concat([_series(values), _series(np.linspace(100, 120, 10), 'Steady')])
    spikes = temporal_spikes(panel, 'co2')
    assert len(spikes) >= 1
    hit = spikes[spikes['year'] == 2005].iloc[0]
    assert hit['country'] == 'Libya'
    assert hit['pattern'] == 'spike_and_revert'
    assert hit['pct_change'] == pytest.approx(10 / 54 - 1, rel=1e-6)
    assert 'Steady' not in spikes['country'].values


def test_rebound_after_a_spike_is_linked_to_the_same_event():
    values = [50.0, 51, 52, 53, 54, 10, 55, 56, 57, 58]
    panel = pd.concat([_series(values), _series(np.linspace(100, 120, 10), 'Steady')])
    spikes = temporal_spikes(panel, 'co2').set_index('year')
    # 2006 is the recovery, not an independent level shift
    assert spikes.loc[2006, 'pattern'] == 'rebound'


def test_persistent_jump_is_a_level_shift_not_a_spike():
    values = [10.0, 10.2, 10.4, 10.6, 10.8, 30, 30.3, 30.6, 30.9, 31.2]  # rebasing / methodology change
    panel = pd.concat([_series(values, 'Shifter'), _series(np.linspace(100, 120, 10), 'Steady')])
    spikes = temporal_spikes(panel, 'co2')
    hit = spikes[(spikes['country'] == 'Shifter') & (spikes['year'] == 2005)].iloc[0]
    assert hit['pattern'] == 'level_shift'


def test_changes_are_never_computed_across_countries():
    # Country A ends at 1000, country B starts at 1: that is not a year-over-year jump
    panel = pd.concat([_series(np.linspace(990, 1000, 10), 'A'), _series(np.linspace(1, 1.1, 10), 'B')])
    assert temporal_spikes(panel, 'co2').empty


def test_smooth_panel_has_no_spikes(panel):
    for feature in ['gdp', 'population', 'co2']:
        assert temporal_spikes(panel, feature).empty


def test_spike_output_is_reviewable(panel):
    panel = panel.copy()
    idx = panel.index[(panel['country'] == 'Country03') & (panel['year'] == 2010)][0]
    panel.loc[idx, 'co2'] *= 20
    spikes = temporal_spikes(panel, 'co2')
    expected_cols = {'country', 'year', 'feature', 'value', 'previous_value', 'pct_change', 'robust_z', 'pattern'}
    assert expected_cols.issubset(spikes.columns)
    assert spikes['feature'].eq('co2').all()
    assert ((spikes['country'] == 'Country03') & (spikes['year'] == 2010)).any()
